use crate::bus::InterAgentBus;
use crate::mcp_gateway::{McpToolCallParams, McpToolDefinition, McpToolGateway};
use agent_domain::{AgentId, InterAgentEvent};
use agent_engine::{
    ChatMessage, ComputeEngine, EngineError, EngineInput, ResponseFormat, StreamCallback,
    TokenUsage, ToolDefinition,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Configuration for an in-process `ToolCallingLoop` (App Brain mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolLoopConfig {
    pub max_turns: usize,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub response_format: Option<ResponseFormat>,
    pub actor: Option<String>,
    pub timeout_seconds: u64,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            temperature: None,
            max_tokens: None,
            response_format: None,
            actor: None,
            timeout_seconds: 120,
        }
    }
}

/// Result returned by `ToolCallingLoop::run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolLoopResult {
    pub final_answer: String,
    pub messages: Vec<ChatMessage>,
    pub turns_used: usize,
    pub tool_calls_executed: usize,
    pub total_usage: TokenUsage,
    pub finish_reason: Option<String>,
}

/// In-process LLM Tool-Calling (ReAct) loop backed by `McpToolGateway`.
///
/// Unlike `AutonomousOodaLoop`, `ToolCallingLoop` does not spawn OS shells or
/// sandboxes; it executes registered application/MCP tools directly via
/// `McpToolGateway` with guardrail policy enforcement and audit logging.
pub struct ToolCallingLoop {
    engine: Arc<dyn ComputeEngine>,
    gateway: Arc<Mutex<McpToolGateway>>,
    bus: Option<InterAgentBus>,
    config: ToolLoopConfig,
}

impl ToolCallingLoop {
    pub fn new(
        engine: Arc<dyn ComputeEngine>,
        gateway: Arc<Mutex<McpToolGateway>>,
    ) -> Self {
        Self {
            engine,
            gateway,
            bus: None,
            config: ToolLoopConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ToolLoopConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_bus(mut self, bus: InterAgentBus) -> Self {
        self.bus = Some(bus);
        self
    }

    fn to_tool_definition(def: &McpToolDefinition) -> ToolDefinition {
        let mut props = serde_json::Map::new();
        for (name, ptype) in &def.input_schema.properties {
            props.insert(name.clone(), json!({ "type": ptype.as_str() }));
        }
        ToolDefinition {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: json!({
                "type": "object",
                "properties": Value::Object(props),
                "required": def.input_schema.required,
                "additionalProperties": !def.input_schema.disallow_additional_properties,
            }),
        }
    }

    fn publish_event(&self, event_name: &str, payload: Value) {
        if let Some(ref bus) = self.bus {
            let _ = bus.publish(InterAgentEvent::new(
                AgentId::from_str("brain-tool-loop"),
                None,
                event_name,
                payload,
            ));
        }
    }

    /// Execute the multi-turn tool-calling loop until the model returns a final
    /// text/structured response or `max_turns` is exhausted.
    pub async fn run(
        &self,
        mut messages: Vec<ChatMessage>,
        on_stream: Option<StreamCallback>,
    ) -> Result<ToolLoopResult, EngineError> {
        let mut total_usage = TokenUsage::default();
        let mut tool_calls_executed = 0usize;
        let max_turns = self.config.max_turns.max(1);

        for turn in 1..=max_turns {
            let tools = {
                let gw = self.gateway.lock().await;
                gw.list_enabled_tools()
                    .into_iter()
                    .map(Self::to_tool_definition)
                    .collect::<Vec<_>>()
            };

            let mut input = EngineInput::default();
            input.timeout_seconds = self.config.timeout_seconds;
            input.temperature = self.config.temperature;
            input.max_tokens = self.config.max_tokens;
            input.response_format = self.config.response_format.clone();
            input.tools = tools;
            input.on_stream_chunk = on_stream.clone();
            input.messages = Some(messages.clone());
            if let Some(last_user) = messages.iter().rev().find(|m| m.role == "user") {
                input.prompt_or_command = last_user.content.clone();
            }

            let output = self.engine.execute(&input).await?;
            if let Some(ref u) = output.usage {
                total_usage.prompt_tokens += u.prompt_tokens;
                total_usage.completion_tokens += u.completion_tokens;
                total_usage.total_tokens += u.total_tokens;
            }

            self.publish_event(
                "brain.turn_completed",
                json!({
                    "turn": turn,
                    "tool_calls": output.tool_calls.len(),
                    "finish_reason": output.finish_reason,
                }),
            );

            if output.tool_calls.is_empty() {
                messages.push(ChatMessage::assistant(&output.stdout_content));
                return Ok(ToolLoopResult {
                    final_answer: output.stdout_content,
                    messages,
                    turns_used: turn,
                    tool_calls_executed,
                    total_usage,
                    finish_reason: output.finish_reason,
                });
            }

            messages.push(ChatMessage::assistant_tool_calls(
                &output.stdout_content,
                output.tool_calls.clone(),
            ));

            for tc in &output.tool_calls {
                let call_res = {
                    let mut gw = self.gateway.lock().await;
                    gw.invoke_tool_call(McpToolCallParams {
                        server_id: None,
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                        actor: self.config.actor.clone(),
                    })
                };

                let (result_text, is_error) = match call_res {
                    Ok(res) => {
                        let text = res
                            .content
                            .into_iter()
                            .map(|c| c.text)
                            .collect::<Vec<_>>()
                            .join("\n");
                        (text, res.is_error)
                    }
                    Err(err) => (format!("Tool error: {}", err), true),
                };

                tool_calls_executed += 1;
                self.publish_event(
                    "brain.tool_called",
                    json!({
                        "turn": turn,
                        "tool": tc.name,
                        "is_error": is_error,
                    }),
                );

                messages.push(ChatMessage::tool_result(&tc.id, result_text));
            }
        }

        Err(EngineError::ProcessFailed(format!(
            "ToolCallingLoop exceeded max_turns ({})",
            max_turns
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_gateway::{FnMcpToolHandler, McpPropertyType, McpToolCallResult, McpToolSchema};
    use agent_domain::GuardrailPolicy;
    use agent_engine::{EngineKind, EngineOutput, ToolCall};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SequenceToolEngine {
        step: AtomicUsize,
    }

    #[async_trait]
    impl ComputeEngine for SequenceToolEngine {
        async fn execute(&self, input: &EngineInput) -> Result<EngineOutput, EngineError> {
            let idx = self.step.fetch_add(1, Ordering::SeqCst);
            if idx == 0 {
                Ok(EngineOutput {
                    stdout_content: String::new(),
                    engine_used: EngineKind::Mock,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "add_numbers".to_string(),
                        arguments: json!({"a": 2, "b": 3}),
                    }],
                    finish_reason: Some("tool_calls".to_string()),
                    usage: Some(TokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                    }),
                    ..Default::default()
                })
            } else {
                let msgs = input.messages.as_ref().unwrap();
                let tool_msg = msgs.iter().rev().find(|m| m.role == "tool").unwrap();
                Ok(EngineOutput {
                    stdout_content: format!("Result is {}", tool_msg.content),
                    engine_used: EngineKind::Mock,
                    finish_reason: Some("stop".to_string()),
                    usage: Some(TokenUsage {
                        prompt_tokens: 20,
                        completion_tokens: 4,
                        total_tokens: 24,
                    }),
                    ..Default::default()
                })
            }
        }

        fn engine_kind(&self) -> EngineKind {
            EngineKind::Mock
        }
    }

    #[tokio::test]
    async fn tool_loop_executes_tool_and_returns_final_answer() {
        let mut policy = GuardrailPolicy::default_enterprise();
        policy.allow_shell_execution = false;
        let mut gateway = McpToolGateway::new(policy);

        let schema = McpToolSchema::new()
            .with_required_property("a", McpPropertyType::Integer)
            .with_required_property("b", McpPropertyType::Integer);
        gateway.register_tool_with_handler(
            McpToolDefinition::new("app", "add_numbers", "Add two integers", schema),
            Arc::new(FnMcpToolHandler(|params| {
                let a = params.arguments["a"].as_i64().unwrap_or(0);
                let b = params.arguments["b"].as_i64().unwrap_or(0);
                Ok(McpToolCallResult::success_text((a + b).to_string()))
            })),
        );

        let engine = Arc::new(SequenceToolEngine {
            step: AtomicUsize::new(0),
        });
        let loop_runner = ToolCallingLoop::new(engine, Arc::new(Mutex::new(gateway)));

        let res = loop_runner
            .run(vec![ChatMessage::user("What is 2 + 3?")], None)
            .await
            .unwrap();

        assert_eq!(res.final_answer, "Result is 5");
        assert_eq!(res.turns_used, 2);
        assert_eq!(res.tool_calls_executed, 1);
        assert_eq!(res.total_usage.total_tokens, 39);
    }
}
