use super::OpenAiCompatibleEngine;
use crate::traits::{EngineError, EngineInput, EngineKind, EngineOutput, TokenUsage, ToolCall};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

pub(super) async fn execute_streaming(
    mut res: reqwest::Response,
    input: &EngineInput,
    start: Instant,
) -> Result<EngineOutput, EngineError> {
    let mut buffer = String::new();
    let mut full_content = String::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<TokenUsage> = None;
    let mut partial_calls: BTreeMap<usize, PartialToolCall> = BTreeMap::new();

    while let Some(chunk) = res
        .chunk()
        .await
        .map_err(|e| EngineError::ApiRequestFailed(format!("Stream read error: {}", e)))?
    {
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim().to_string();
            buffer.drain(..=newline_pos);

            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }
            let data = line.trim_start_matches("data:").trim();
            if data == "[DONE]" {
                continue;
            }
            let Ok(val) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            if let Some(u) = OpenAiCompatibleEngine::parse_usage(val.get("usage")) {
                usage = Some(u);
            }

            if let Some(choice) = val.get("choices").and_then(|c| c.get(0)) {
                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    finish_reason = Some(fr.to_string());
                }
                if let Some(delta) = choice.get("delta") {
                    if let Some(content_delta) = delta.get("content").and_then(|v| v.as_str()) {
                        full_content.push_str(content_delta);
                        if let Some(ref cb) = input.on_stream_chunk {
                            cb(content_delta);
                        }
                    }
                    if let Some(tc_arr) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tc_arr {
                            let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let entry = partial_calls.entry(idx).or_default();
                            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                entry.id.push_str(id);
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                    entry.name.push_str(name);
                                }
                                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                    entry.arguments.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let tool_calls = partial_calls
        .into_values()
        .map(|p| {
            let args = serde_json::from_str::<Value>(&p.arguments)
                .unwrap_or_else(|_| Value::String(p.arguments));
            ToolCall { id: p.id, name: p.name, arguments: args }
        })
        .collect();

    Ok(EngineOutput {
        stdout_content: full_content.trim().to_string(),
        exit_code: 0,
        execution_duration_ms: start.elapsed().as_millis(),
        engine_used: EngineKind::CustomAgentApi,
        session_id: None,
        tool_calls,
        finish_reason,
        usage,
    })
}
