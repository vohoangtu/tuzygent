use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Process execution error: {0}")]
    ProcessFailed(String),
    #[error("CLI engine not found or unreachable: {0}")]
    CliNotFound(String),
    #[error("API request failed: {0}")]
    ApiRequestFailed(String),
    #[error("Timeout exceeded during computation")]
    Timeout,
    #[error("Red line policy rejected execution: {0}")]
    PolicyRejected(String),
}

impl EngineError {
    pub fn is_auth_or_quota(&self) -> bool {
        let msg = self.to_string().to_lowercase();
        msg.contains("auth")
            || msg.contains("quota")
            || msg.contains("401")
            || msg.contains("403")
            || msg.contains("rate limit")
            || msg.contains("rate_limit")
            || msg.contains("unauthorized")
            || msg.contains("exhausted")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EngineKind {
    ClaudeCode,
    Antigravity,
    Codex,
    CustomAgentApi,
    Mock,
    Generic(String),
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineKind::ClaudeCode => write!(f, "claude"),
            EngineKind::Antigravity => write!(f, "antigravity"),
            EngineKind::Codex => write!(f, "codex"),
            EngineKind::CustomAgentApi => write!(f, "custom_api"),
            EngineKind::Mock => write!(f, "mock"),
            EngineKind::Generic(name) => write!(f, "generic:{}", name),
        }
    }
}

impl EngineKind {
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "claude" | "claude_code" | "claudecode" => EngineKind::ClaudeCode,
            "antigravity" | "agy" => EngineKind::Antigravity,
            "codex" => EngineKind::Codex,
            "custom" | "custom_api" => EngineKind::CustomAgentApi,
            "mock" => EngineKind::Mock,
            other => EngineKind::Generic(other.to_string()),
        }
    }
}

pub type EngineEventCallback = std::sync::Arc<dyn Fn(&serde_json::Value) + Send + Sync>;

pub type StreamCallback = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

/// A single message in a multi-turn LLM conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            tool_call_id: None,
            tool_calls,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
        }
    }
}

/// Tool/function schema exposed to a `ComputeEngine` for native tool calling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A single tool invocation requested by an LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Structured output format requested from an LLM API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default)]
        strict: Option<bool>,
    },
}

/// Token accounting reported by an LLM engine.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EngineInput {
    #[serde(skip)]
    pub on_event: Option<EngineEventCallback>,
    #[serde(skip)]
    pub command_sandbox: Option<crate::process::CommandSandbox>,
    pub prompt_or_command: String,
    pub system_context: String,
    pub working_dir: String,
    pub timeout_seconds: u64,
    #[serde(skip)]
    pub on_stream_chunk: Option<StreamCallback>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_output_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ChatMessage>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

impl std::fmt::Debug for EngineInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineInput")
            .field("prompt_or_command", &self.prompt_or_command)
            .field("system_context", &self.system_context)
            .field("working_dir", &self.working_dir)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("has_stream_callback", &self.on_stream_chunk.is_some())
            .field("messages_count", &self.messages.as_ref().map(|m| m.len()))
            .field("tools_count", &self.tools.len())
            .field("response_format", &self.response_format)
            .finish()
    }
}

impl EngineInput {
    pub fn new(prompt: impl Into<String>, working_dir: impl Into<String>) -> Self {
        Self {
            prompt_or_command: prompt.into(),
            system_context: String::new(),
            working_dir: working_dir.into(),
            timeout_seconds: 300,
            on_stream_chunk: None,
            command_sandbox: None,
            on_event: None,
            temperature: None,
            max_tokens: None,
            max_output_bytes: None,
            messages: None,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
        }
    }

    pub fn with_stream_callback(mut self, cb: StreamCallback) -> Self {
        self.on_stream_chunk = Some(cb);
        self
    }

    pub fn with_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = Some(messages);
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: serde_json::Value) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl Default for EngineInput {
    fn default() -> Self {
        Self {
            prompt_or_command: String::new(),
            system_context: String::new(),
            working_dir: ".".to_string(),
            timeout_seconds: 300,
            on_stream_chunk: None,
            command_sandbox: None,
            on_event: None,
            temperature: None,
            max_tokens: None,
            max_output_bytes: None,
            messages: None,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub stdout_content: String,
    pub exit_code: i32,
    pub execution_duration_ms: u128,
    pub engine_used: EngineKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl Default for EngineOutput {
    fn default() -> Self {
        Self {
            session_id: None,
            stdout_content: String::new(),
            exit_code: 0,
            execution_duration_ms: 0,
            engine_used: EngineKind::Mock,
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
        }
    }
}

#[async_trait]
pub trait ComputeEngine: Send + Sync {
    async fn execute(&self, input: &EngineInput) -> Result<EngineOutput, EngineError>;
    fn engine_kind(&self) -> EngineKind;
}
