use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use uuid::Uuid;

use agent_engine::{AccountRotatedEngine, ComputeEngine, EngineInput, EnginesConfig};
use agent_runtime::ExecutionService;

use super::types::{
    format_prompt_from_messages, resolve_gateway_target, ChatCompletionRequest, GatewayTarget,
};
use crate::AppState;

/// Endpoint chuẩn OpenAI POST /v1/chat/completions
pub async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let target = match resolve_gateway_target(&payload.model) {
        Ok(res) => res,
        Err(err_msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": err_msg,
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
    };
    let (system_ctx, prompt) = format_prompt_from_messages(&payload.messages);

    let execution = match ExecutionService::from_workspace(&state.workspace) {
        Ok(service) => service,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };
    let engine: Arc<dyn ComputeEngine> = match target {
        GatewayTarget::AccountRotated(engine_kind, model_override) => {
            Arc::new(AccountRotatedEngine::new(
                state.vault.clone(),
                engine_kind,
                None,
                model_override,
            ))
        }
        GatewayTarget::ConfiguredApi {
            engine_name,
            model_override,
        } => {
            let cfg = EnginesConfig::load_hierarchical(Some(std::path::Path::new(&state.workspace)));
            match agent_engine::configured_engine(&cfg, &engine_name, model_override.as_deref()) {
                Ok(e) => e,
                Err(err) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({ "error": { "message": err.to_string() } })),
                    )
                        .into_response();
                }
            }
        }
    };

    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created_ts = chrono::Utc::now().timestamp();
    let model_name = payload.model.clone();

    if payload.stream {
        use futures::sink::SinkExt;
        let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Event, Infallible>>(128);

        let initial_chunk = serde_json::json!({
            "id": &completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": &model_name,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "" },
                "finish_reason": null
            }]
        });
        let _ = tx.try_send(Ok(Event::default().data(initial_chunk.to_string())));

        let tx_mutex = Arc::new(std::sync::Mutex::new(tx.clone()));
        let overflow = Arc::new(AtomicBool::new(false));
        let callback_overflow = overflow.clone();
        let cid = completion_id.clone();
        let mname = model_name.clone();

        let stream_cb = Arc::new(move |chunk: &str| {
            let chunk_data = serde_json::json!({
                "id": &cid,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": &mname,
                "choices": [{
                    "index": 0,
                    "delta": { "content": chunk },
                    "finish_reason": null
                }]
            });
            if let Ok(mut tx_guard) = tx_mutex.lock() {
                if tx_guard
                    .try_send(Ok(Event::default().data(chunk_data.to_string())))
                    .is_err()
                {
                    callback_overflow.store(true, Ordering::SeqCst);
                }
            } else {
                callback_overflow.store(true, Ordering::SeqCst);
            }
        });

        let mut input = EngineInput::new(prompt, &state.workspace);
        if !system_ctx.is_empty() {
            input.system_context = system_ctx;
        }
        input.temperature = payload.temperature;
        input.max_tokens = payload.max_tokens;
        input.tools = payload.tools.clone();
        input.tool_choice = payload.tool_choice.clone();
        input.response_format = payload.response_format.clone();
        input.messages = Some(
            payload
                .messages
                .iter()
                .map(|m| agent_engine::ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                })
                .collect(),
        );
        input.on_stream_chunk = Some(stream_cb);

        tokio::spawn(async move {
            let outcome = {
                let run = execution.execute(engine, &input);
                tokio::pin!(run);
                loop {
                    tokio::select! {
                        result = &mut run => break Some(result),
                        _ = tokio::time::sleep(std::time::Duration::from_millis(25)) => {
                            if tx.is_closed() { break None; }
                            if overflow.load(Ordering::SeqCst) {
                                break Some(Err(agent_engine::EngineError::ProcessFailed("Stream buffer exhausted; output is incomplete".into())));
                            }
                        }
                    }
                }
            };
            let Some(outcome) = outcome else {
                return;
            };
            let outcome = if overflow.load(Ordering::SeqCst) {
                Err(agent_engine::EngineError::ProcessFailed(
                    "Stream buffer exhausted; output is incomplete".into(),
                ))
            } else {
                outcome
            };
            match outcome {
                Ok(output) if output.exit_code == 0 => {
                    let stop_chunk = serde_json::json!({
                        "id": &completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": &model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    });
                    let _ = tx
                        .send(Ok(Event::default().data(stop_chunk.to_string())))
                        .await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                }
                result => {
                    let message = match result {
                        Ok(output) => format!("Engine exited with code {}", output.exit_code),
                        Err(err) => err.to_string(),
                    };
                    let err_chunk = serde_json::json!({
                        "id": &completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": &model_name,
                        "error": { "message": message }
                    });
                    let _ = tx
                        .send(Ok(Event::default().data(err_chunk.to_string())))
                        .await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                }
            }
        });

        Sse::new(rx).into_response()
    } else {
        let mut input = EngineInput::new(prompt, &state.workspace);
        if !system_ctx.is_empty() {
            input.system_context = system_ctx;
        }
        input.temperature = payload.temperature;
        input.max_tokens = payload.max_tokens;
        input.tools = payload.tools;
        input.tool_choice = payload.tool_choice;
        input.response_format = payload.response_format;
        input.messages = Some(
            payload
                .messages
                .iter()
                .map(|m| agent_engine::ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                })
                .collect(),
        );
        match execution.execute(engine, &input).await {
            Ok(output) if output.exit_code == 0 => {
                let mut msg = serde_json::json!({
                    "role": "assistant",
                    "content": output.stdout_content
                });
                if !output.tool_calls.is_empty() {
                    let calls: Vec<serde_json::Value> = output
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments.to_string()
                                }
                            })
                        })
                        .collect();
                    msg["tool_calls"] = serde_json::Value::Array(calls);
                }
                let finish_reason = output
                    .finish_reason
                    .unwrap_or_else(|| "stop".to_string());
                let resp = serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": msg,
                        "finish_reason": finish_reason
                    }],
                    "usage": output.usage
                });
                Json(resp).into_response()
            }
            result => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": match result { Ok(output) => format!("Engine exited with code {}", output.exit_code), Err(err) => err.to_string() },
                        "type": "engine_error"
                    }
                })),
            )
                .into_response(),
        }
    }
}

/// Endpoint chuẩn OpenAI GET /v1/models
pub async fn list_models_handler() -> Json<serde_json::Value> {
    let now = chrono::Utc::now().timestamp();
    Json(serde_json::json!({
        "object": "list",
        "data": [
            { "id": "codex", "object": "model", "created": now, "owned_by": "agentcore" },
            { "id": "codex/gpt-4o", "object": "model", "created": now, "owned_by": "agentcore" },
            { "id": "claude", "object": "model", "created": now, "owned_by": "agentcore" },
            { "id": "claude/claude-sonnet-5", "object": "model", "created": now, "owned_by": "agentcore" },
            { "id": "openai/gpt-4o-mini", "object": "model", "created": now, "owned_by": "tuzygent" },
            { "id": "ollama/qwen2.5:latest", "object": "model", "created": now, "owned_by": "tuzygent" },
        ]
    }))
}
