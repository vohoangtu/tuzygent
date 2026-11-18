mod streaming;

#[cfg(test)]
mod tests;

use crate::traits::{
    ChatMessage, ComputeEngine, EngineError, EngineInput, EngineKind, EngineOutput, ResponseFormat,
    TokenUsage, ToolCall,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::Instant;

pub struct OpenAiCompatibleEngine {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    client: reqwest::Client,
}

impl OpenAiCompatibleEngine {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    fn build_messages(input: &EngineInput) -> Vec<Value> {
        if let Some(ref msgs) = input.messages {
            if !msgs.is_empty() {
                return msgs.iter().map(Self::serialize_message).collect();
            }
        }
        let mut out = Vec::new();
        if !input.system_context.trim().is_empty() {
            out.push(json!({ "role": "system", "content": input.system_context }));
        }
        out.push(json!({ "role": "user", "content": input.prompt_or_command }));
        out
    }

    fn serialize_message(msg: &ChatMessage) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("role".to_string(), json!(msg.role));
        if msg.content.is_empty() && !msg.tool_calls.is_empty() {
            obj.insert("content".to_string(), Value::Null);
        } else {
            obj.insert("content".to_string(), json!(msg.content));
        }
        if let Some(ref id) = msg.tool_call_id {
            obj.insert("tool_call_id".to_string(), json!(id));
        }
        if !msg.tool_calls.is_empty() {
            let calls: Vec<Value> = msg
                .tool_calls
                .iter()
                .map(|tc| {
                    let args_str = match &tc.arguments {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
                    };
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": { "name": tc.name, "arguments": args_str }
                    })
                })
                .collect();
            obj.insert("tool_calls".to_string(), Value::Array(calls));
        }
        Value::Object(obj)
    }

    fn build_request_body(&self, input: &EngineInput, stream: bool) -> Value {
        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), json!(self.model));
        body.insert("messages".to_string(), Value::Array(Self::build_messages(input)));

        if let Some(temp) = input.temperature {
            body.insert("temperature".to_string(), json!(temp));
        }
        if let Some(max_tok) = input.max_tokens {
            body.insert("max_tokens".to_string(), json!(max_tok));
        }
        if !input.tools.is_empty() {
            let tools: Vec<Value> = input
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body.insert("tools".to_string(), Value::Array(tools));
        }
        if let Some(ref choice) = input.tool_choice {
            body.insert("tool_choice".to_string(), choice.clone());
        }
        if let Some(ref fmt) = input.response_format {
            let fmt_json = match fmt {
                ResponseFormat::Text => json!({ "type": "text" }),
                ResponseFormat::JsonObject => json!({ "type": "json_object" }),
                ResponseFormat::JsonSchema { name, schema, strict } => json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": strict.unwrap_or(true)
                    }
                }),
            };
            body.insert("response_format".to_string(), fmt_json);
        }
        if stream {
            body.insert("stream".to_string(), json!(true));
        }
        Value::Object(body)
    }

    pub(crate) fn parse_tool_calls(raw_calls: Option<&Vec<Value>>) -> Vec<ToolCall> {
        let Some(arr) = raw_calls else {
            return Vec::new();
        };
        arr.iter()
            .filter_map(|item| {
                let id = item.get("id")?.as_str()?.to_string();
                let func = item.get("function")?;
                let name = func.get("name")?.as_str()?.to_string();
                let args_val = match func.get("arguments") {
                    Some(Value::String(s)) => {
                        serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => json!({}),
                };
                Some(ToolCall { id, name, arguments: args_val })
            })
            .collect()
    }

    pub(crate) fn parse_usage(usage_val: Option<&Value>) -> Option<TokenUsage> {
        let u = usage_val?;
        if u.is_null() {
            return None;
        }
        let prompt_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let completion_tokens =
            u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let total_tokens = u
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(prompt_tokens + completion_tokens);
        Some(TokenUsage { prompt_tokens, completion_tokens, total_tokens })
    }
}

#[async_trait]
impl ComputeEngine for OpenAiCompatibleEngine {
    async fn execute(&self, input: &EngineInput) -> Result<EngineOutput, EngineError> {
        let start = Instant::now();
        let url = format!("{}/chat/completions", self.base_url);
        let stream = input.on_stream_chunk.is_some();
        let req_body = self.build_request_body(input, stream);

        let mut req = self
            .client
            .post(&url)
            .timeout(crate::process::timeout(input))
            .json(&req_body);
        if !self.api_key.trim().is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let res = req
            .send()
            .await
            .map_err(|e| EngineError::ApiRequestFailed(format!("HTTP POST error to LLM: {}", e)))?;

        let status = res.status().as_u16() as i32;
        if status != 200 {
            let err_text = res.text().await.unwrap_or_default();
            return Err(EngineError::ApiRequestFailed(format!(
                "LLM API returned status {}: {}",
                status, err_text
            )));
        }

        if stream {
            return streaming::execute_streaming(res, input, start).await;
        }

        let resp_json: Value = res.json().await.map_err(|e| {
            EngineError::ApiRequestFailed(format!("Failed parsing LLM JSON: {}", e))
        })?;

        let choice = &resp_json["choices"][0];
        let message = &choice["message"];
        let content = message["content"].as_str().unwrap_or("").trim().to_string();
        let tool_calls = Self::parse_tool_calls(message["tool_calls"].as_array());
        let finish_reason = choice["finish_reason"].as_str().map(str::to_string);
        let usage = Self::parse_usage(resp_json.get("usage"));

        Ok(EngineOutput {
            stdout_content: content,
            exit_code: 0,
            execution_duration_ms: start.elapsed().as_millis(),
            engine_used: EngineKind::CustomAgentApi,
            session_id: None,
            tool_calls,
            finish_reason,
            usage,
        })
    }

    fn engine_kind(&self) -> EngineKind {
        EngineKind::CustomAgentApi
    }
}
