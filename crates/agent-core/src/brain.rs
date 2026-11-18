use agent_domain::{GuardrailPolicy, Priority, Task, TaskResult};
use agent_engine::{
    ChatMessage, ComputeEngine, EngineInput, EngineOutput, ResponseFormat, StreamCallback,
};
use agent_memory::{DurableMemoryStore, ScratchpadManager};
use agent_runtime::{
    AutonomousOodaLoop, InterAgentBus, McpToolDefinition, McpToolGateway, McpToolHandler,
    SandboxConfig, ToolCallingLoop, ToolLoopConfig, ToolLoopResult,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Options for `BrainClient::chat`, `BrainClient::chat_stream`, and `BrainClient::run_tools`.
#[derive(Debug, Clone, Default)]
pub struct BrainChatOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub response_format: Option<ResponseFormat>,
    pub max_turns: Option<usize>,
    pub actor: Option<String>,
}

/// Unified high-level SDK client for embedding `tuzygent` as an AI Brain in Rust apps
/// or powering `agent-server` endpoints.
pub struct BrainClient {
    engine: Arc<dyn ComputeEngine>,
    workspace: PathBuf,
    project_id: String,
    guardrail: GuardrailPolicy,
    gateway: Arc<Mutex<McpToolGateway>>,
    bus: InterAgentBus,
    sandbox_config: Option<SandboxConfig>,
    max_turns: usize,
    timeout_seconds: u64,
}

pub struct BrainClientBuilder {
    engine: Option<Arc<dyn ComputeEngine>>,
    workspace: PathBuf,
    project_id: String,
    guardrail: GuardrailPolicy,
    gateway: Option<McpToolGateway>,
    pending_tools: Vec<(McpToolDefinition, Arc<dyn McpToolHandler>)>,
    bus: Option<InterAgentBus>,
    sandbox_config: Option<SandboxConfig>,
    max_turns: usize,
    timeout_seconds: u64,
}

impl Default for BrainClientBuilder {
    fn default() -> Self {
        let mut guardrail = GuardrailPolicy::default_enterprise();
        guardrail.allow_shell_execution = false;
        Self {
            engine: None,
            workspace: PathBuf::from("."),
            project_id: "default".to_string(),
            guardrail,
            gateway: None,
            pending_tools: Vec::new(),
            bus: None,
            sandbox_config: None,
            max_turns: 10,
            timeout_seconds: 120,
        }
    }
}

impl BrainClientBuilder {
    pub fn with_engine(mut self, engine: Arc<dyn ComputeEngine>) -> Self {
        self.engine = Some(engine);
        self
    }

    pub fn with_workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = workspace.into();
        self
    }

    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        let p = project_id.into();
        if !p.trim().is_empty() {
            self.project_id = p;
        }
        self
    }

    pub fn with_guardrail(mut self, guardrail: GuardrailPolicy) -> Self {
        self.guardrail = guardrail;
        self
    }

    pub fn with_tool_gateway(mut self, gateway: McpToolGateway) -> Self {
        self.gateway = Some(gateway);
        self
    }

    pub fn register_tool(
        mut self,
        definition: McpToolDefinition,
        handler: Arc<dyn McpToolHandler>,
    ) -> Self {
        self.pending_tools.push((definition, handler));
        self
    }

    pub fn with_bus(mut self, bus: InterAgentBus) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn with_sandbox_config(mut self, config: SandboxConfig) -> Self {
        self.sandbox_config = Some(config);
        self
    }

    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns.max(1);
        self
    }

    pub fn with_timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds.max(1);
        self
    }

    pub fn build(self) -> anyhow::Result<BrainClient> {
        let engine = self
            .engine
            .ok_or_else(|| anyhow::anyhow!("BrainClient requires a ComputeEngine"))?;
        let mut gateway = self
            .gateway
            .unwrap_or_else(|| McpToolGateway::new(self.guardrail.clone()));
        for (def, handler) in self.pending_tools {
            gateway.register_tool_with_handler(def, handler);
        }
        Ok(BrainClient {
            engine,
            workspace: self.workspace,
            project_id: self.project_id,
            guardrail: self.guardrail,
            gateway: Arc::new(Mutex::new(gateway)),
            bus: self.bus.unwrap_or_else(|| InterAgentBus::new(128)),
            sandbox_config: self.sandbox_config,
            max_turns: self.max_turns,
            timeout_seconds: self.timeout_seconds,
        })
    }
}

impl BrainClient {
    pub fn builder() -> BrainClientBuilder {
        BrainClientBuilder::default()
    }

    pub fn engine(&self) -> &Arc<dyn ComputeEngine> {
        &self.engine
    }

    pub fn gateway(&self) -> &Arc<Mutex<McpToolGateway>> {
        &self.gateway
    }

    pub fn bus(&self) -> &InterAgentBus {
        &self.bus
    }

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Single-turn or multi-turn chat without tool loop.
    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        options: BrainChatOptions,
    ) -> anyhow::Result<EngineOutput> {
        self.chat_internal(messages, options, None).await
    }

    /// Streaming chat with token/delta callback.
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        options: BrainChatOptions,
        on_chunk: StreamCallback,
    ) -> anyhow::Result<EngineOutput> {
        self.chat_internal(messages, options, Some(on_chunk)).await
    }

    async fn chat_internal(
        &self,
        messages: Vec<ChatMessage>,
        options: BrainChatOptions,
        on_chunk: Option<StreamCallback>,
    ) -> anyhow::Result<EngineOutput> {
        let mut input = EngineInput::default();
        input.working_dir = self.workspace.to_string_lossy().to_string();
        input.timeout_seconds = self.timeout_seconds;
        input.temperature = options.temperature;
        input.max_tokens = options.max_tokens;
        input.response_format = options.response_format;
        input.on_stream_chunk = on_chunk;
        if let Some(last_user) = messages.iter().rev().find(|m| m.role == "user") {
            input.prompt_or_command = last_user.content.clone();
        }
        input.messages = Some(messages);
        let out = self.engine.execute(&input).await?;
        Ok(out)
    }

    /// In-process ReAct / Tool-Calling loop backed by `McpToolGateway`.
    pub async fn run_tools(
        &self,
        messages: Vec<ChatMessage>,
        options: BrainChatOptions,
        on_stream: Option<StreamCallback>,
    ) -> anyhow::Result<ToolLoopResult> {
        let cfg = ToolLoopConfig {
            max_turns: options.max_turns.unwrap_or(self.max_turns),
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            response_format: options.response_format,
            actor: options.actor,
            timeout_seconds: self.timeout_seconds,
        };
        let tool_loop = ToolCallingLoop::new(self.engine.clone(), self.gateway.clone())
            .with_config(cfg)
            .with_bus(self.bus.clone());
        let res = tool_loop.run(messages, on_stream).await?;
        Ok(res)
    }

    /// Autonomous OODA goal execution with workspace memory and optional sandbox.
    pub async fn run_goal(
        &self,
        title: &str,
        goal: &str,
        priority: Priority,
    ) -> anyhow::Result<TaskResult> {
        let ws_str = self.workspace.to_string_lossy().to_string();
        let mut task = Task::new(title, goal, priority);
        let scratchpad = ScratchpadManager::for_task(&ws_str, &task.id.0);
        let durable = DurableMemoryStore::new(&ws_str);

        let mut ooda = AutonomousOodaLoop::new(
            self.engine.clone(),
            self.guardrail.clone(),
            scratchpad,
            durable,
            &ws_str,
        )
        .with_project_id(&self.project_id)
        .with_timeout_seconds(self.timeout_seconds)
        .with_bus(self.bus.clone());

        if let Some(ref sb) = self.sandbox_config {
            ooda = ooda.with_sandbox_config(sb.clone());
        }

        let res = ooda
            .run_goal_detailed(&mut task)
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_engine::{EngineError, EngineKind, MockEngine, TokenUsage, ToolCall};
    use agent_runtime::{FnMcpToolHandler, McpPropertyType, McpToolCallResult, McpToolSchema};
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct BrainMockEngine(AtomicUsize);

    #[async_trait]
    impl ComputeEngine for BrainMockEngine {
        async fn execute(&self, _input: &EngineInput) -> Result<EngineOutput, EngineError> {
            let n = self.0.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Ok(EngineOutput {
                    engine_used: EngineKind::Mock,
                    tool_calls: vec![ToolCall {
                        id: "c1".into(),
                        name: "lookup".into(),
                        arguments: json!({"key": "status"}),
                    }],
                    ..Default::default()
                })
            } else {
                Ok(EngineOutput {
                    stdout_content: "OK: healthy".into(),
                    engine_used: EngineKind::Mock,
                    usage: Some(TokenUsage {
                        prompt_tokens: 5,
                        completion_tokens: 3,
                        total_tokens: 8,
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
    async fn brain_client_supports_chat_and_run_tools() {
        let chat_brain = BrainClient::builder()
            .with_engine(Arc::new(MockEngine::new("hi")))
            .with_project_id("app-alpha")
            .build()
            .unwrap();
        let chat_out = chat_brain
            .chat(vec![ChatMessage::user("ping")], BrainChatOptions::default())
            .await
            .unwrap();
        assert_eq!(chat_out.stdout_content, "hi: ping");
        assert_eq!(chat_brain.project_id(), "app-alpha");

        let schema = McpToolSchema::new().with_required_property("key", McpPropertyType::String);
        let tool_brain = BrainClient::builder()
            .with_engine(Arc::new(BrainMockEngine(AtomicUsize::new(0))))
            .register_tool(
                McpToolDefinition::new("app", "lookup", "Lookup key", schema),
                Arc::new(FnMcpToolHandler(|p| {
                    Ok(McpToolCallResult::success_text(format!(
                        "val={}",
                        p.arguments["key"]
                    )))
                })),
            )
            .build()
            .unwrap();
        let res = tool_brain
            .run_tools(
                vec![ChatMessage::user("check status")],
                BrainChatOptions::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(res.final_answer, "OK: healthy");
        assert_eq!(res.tool_calls_executed, 1);
    }
}
