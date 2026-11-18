use std::convert::Infallible;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use agent_core::prelude::*;
use agent_engine::AccountRotatedEngine;
use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::sink::SinkExt;
use serde::{Deserialize, Serialize};

use crate::auth::TenantContext;
use crate::gateway::{resolve_gateway_target, GatewayTarget};
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct BrainRunReq {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_turns: Option<usize>,
    pub response_format: Option<ResponseFormat>,
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BrainRunResp {
    pub tenant_id: String,
    pub session_id: Option<String>,
    pub model: String,
    pub output: String,
    pub finish_reason: Option<String>,
    pub usage: Option<TokenUsage>,
    pub tool_calls: Vec<ToolCall>,
}

fn record_tenant_usage(workspace: &str, tenant_id: &str, model: &str, usage: &TokenUsage) {
    let storage_dir = Path::new(workspace).join("storage");
    if std::fs::create_dir_all(&storage_dir).is_err() {
        return;
    }
    let path = storage_dir.join("brain_usage.jsonl");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let line = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "tenant_id": tenant_id,
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        });
        let _ = writeln!(file, "{}", line);
    }
}

/// POST /api/v1/brain/run — Unified AI Brain endpoint (multi-tenant, session-aware, SSE-capable).
pub async fn brain_run_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantContext>>,
    Json(payload): Json<BrainRunReq>,
) -> Response {
    let tenant = tenant_ext.map(|Extension(t)| t).unwrap_or_default();
    let cfg = EnginesConfig::load_hierarchical(Some(Path::new(&state.workspace)));
    let model_spec = payload
        .model
        .clone()
        .unwrap_or_else(|| cfg.defaults.engine.clone());

    if let Some(ref policy) = tenant.policy {
        if !policy.allowed_engines.is_empty()
            && !policy
                .allowed_engines
                .iter()
                .any(|allowed| model_spec.to_lowercase().starts_with(&allowed.to_lowercase()))
        {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": format!("Model/engine '{}' is not permitted for tenant '{}'", model_spec, tenant.tenant_id)
                })),
            )
                .into_response();
        }
        if let (Some(limit), Some(req_max)) = (policy.max_tokens_per_request, payload.max_tokens) {
            if req_max > limit {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "error": format!("max_tokens ({}) exceeds tenant limit ({})", req_max, limit)
                    })),
                )
                    .into_response();
            }
        }
    }

    let engine: Arc<dyn ComputeEngine> = match resolve_gateway_target(&model_spec) {
        Ok(GatewayTarget::AccountRotated(kind, model_override)) => {
            Arc::new(AccountRotatedEngine::new(
                state.vault.clone(),
                kind,
                None,
                model_override,
            ))
        }
        Ok(GatewayTarget::ConfiguredApi {
            engine_name,
            model_override,
        }) => match agent_engine::configured_engine(&cfg, &engine_name, model_override.as_deref()) {
            Ok(e) => e,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": err.to_string() })),
                )
                    .into_response();
            }
        },
        Err(_) => match agent_engine::configured_engine(&cfg, &model_spec, None) {
            Ok(e) => e,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": err.to_string() })),
                )
                    .into_response();
            }
        },
    };

    let mut messages = Vec::new();
    let session_mem = payload.session_id.as_deref().and_then(|sid| {
        agent_memory::SessionMemory::open(Path::new(&state.workspace), &tenant.tenant_id, sid).ok()
    });
    if let Some(ref mem) = session_mem {
        if let Ok(history) = mem.list_within_budget(16_000) {
            for turn in history {
                messages.push(ChatMessage {
                    role: turn.role,
                    content: turn.content,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                });
            }
        }
        if let Some(last_user) = payload.messages.iter().rev().find(|m| m.role == "user") {
            let _ = mem.append("user", &last_user.content);
        }
    }
    messages.extend(payload.messages.clone());

    let brain = match BrainClient::builder()
        .with_engine(engine)
        .with_workspace(&state.workspace)
        .with_project_id(&tenant.tenant_id)
        .with_bus(state.bus.clone())
        .with_max_turns(payload.max_turns.unwrap_or(10))
        .build()
    {
        Ok(b) => b,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": err.to_string() })),
            )
                .into_response();
        }
    };

    let opts = BrainChatOptions {
        temperature: payload.temperature,
        max_tokens: payload.max_tokens,
        response_format: payload.response_format,
        max_turns: payload.max_turns,
        actor: Some(tenant.tenant_id.clone()),
    };

    if payload.stream {
        let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Event, Infallible>>(128);
        let tx_sync = Arc::new(std::sync::Mutex::new(tx.clone()));
        let stream_cb = Arc::new(move |chunk: &str| {
            let ev = serde_json::json!({ "delta": chunk });
            if let Ok(mut guard) = tx_sync.lock() {
                let _ = guard.try_send(Ok(Event::default().data(ev.to_string())));
            }
        });
        let workspace = state.workspace.clone();
        let tenant_id = tenant.tenant_id.clone();
        let sid = payload.session_id.clone();

        tokio::spawn(async move {
            match brain.chat_stream(messages, opts, stream_cb).await {
                Ok(out) => {
                    if let Some(ref u) = out.usage {
                        record_tenant_usage(&workspace, &tenant_id, &model_spec, u);
                    }
                    if let Some(ref mem) = session_mem {
                        let _ = mem.append("assistant", &out.stdout_content);
                    }
                    let done = serde_json::json!({
                        "done": true,
                        "tenant_id": tenant_id,
                        "session_id": sid,
                        "usage": out.usage,
                    });
                    let _ = tx.send(Ok(Event::default().data(done.to_string()))).await;
                }
                Err(err) => {
                    let err_ev = serde_json::json!({ "error": err.to_string() });
                    let _ = tx.send(Ok(Event::default().data(err_ev.to_string()))).await;
                }
            }
        });

        return Sse::new(rx).into_response();
    }

    match brain.chat(messages, opts).await {
        Ok(out) => {
            if let Some(ref u) = out.usage {
                record_tenant_usage(&state.workspace, &tenant.tenant_id, &model_spec, u);
            }
            if let Some(ref mem) = session_mem {
                let _ = mem.append("assistant", &out.stdout_content);
            }
            Json(BrainRunResp {
                tenant_id: tenant.tenant_id,
                session_id: payload.session_id,
                model: model_spec,
                output: out.stdout_content,
                finish_reason: out.finish_reason,
                usage: out.usage,
                tool_calls: out.tool_calls,
            })
            .into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

/// GET /api/v1/openapi.json — Machine-readable OpenAPI 3.1.0 schema for client SDK generation.
pub async fn openapi_spec_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Tuzygent AI Brain API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Unified AI Brain Service (OpenAI-compatible Gateway, Brain Tool-Loop, Autonomous OODA Tasks, WebSocket Events)"
        },
        "paths": {
            "/health": { "get": { "summary": "Health check" } },
            "/v1/models": { "get": { "summary": "List available models & engines" } },
            "/v1/chat/completions": { "post": { "summary": "OpenAI-compatible chat completions (streaming, tools, response_format)" } },
            "/api/v1/brain/run": { "post": { "summary": "Multi-tenant, session-aware AI Brain execution (JSON & SSE)" } },
            "/api/v1/projects/logs": { "get": { "summary": "Query unified project logs (brain_usage, scratchpad, session, tasks)" } },
            "/api/v1/projects/{project_id}/logs": { "get": { "summary": "Query logs for a specific project/tenant" } },
            "/api/v1/tasks": { "get": { "summary": "List background OODA tasks" }, "post": { "summary": "Submit async OODA task (supports Idempotency-Key)" } },
            "/api/v1/tasks/{id}": { "get": { "summary": "Get OODA task status & result" } },
            "/api/v1/ws": { "get": { "summary": "WebSocket real-time InterAgentBus stream" } }
        }
    }))
}
