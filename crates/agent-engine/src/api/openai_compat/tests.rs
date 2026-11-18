use super::*;
use crate::traits::ToolDefinition;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn spawn_mock_server(
    response_body: String,
    content_type: &str,
) -> (String, tokio::sync::oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let ct = content_type.to_string();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let n = socket.read(&mut buf).await.unwrap();
        let raw_req = String::from_utf8_lossy(&buf[..n]).to_string();
        let _ = tx.send(raw_req);

        let http_resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            ct,
            response_body.len(),
            response_body
        );
        let _ = socket.write_all(http_resp.as_bytes()).await;
    });

    (format!("http://{}", addr), rx)
}

#[tokio::test]
async fn openai_compat_sends_params_tools_and_parses_usage_and_tool_calls() {
    let resp_json = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Hanoi\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 12,
            "total_tokens": 37
        }
    })
    .to_string();

    let (base_url, rx) = spawn_mock_server(resp_json, "application/json").await;
    let engine = OpenAiCompatibleEngine::new(&base_url, "test-key", "gpt-4o-mini");

    let input = EngineInput::new("Check weather", ".")
        .with_temperature(0.2)
        .with_max_tokens(1024)
        .with_response_format(ResponseFormat::JsonObject)
        .with_tools(vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get city weather".to_string(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }]);

    let out = engine.execute(&input).await.unwrap();
    let raw_req = rx.await.unwrap();

    assert!(raw_req.contains("\"temperature\":0.2"));
    assert!(raw_req.contains("\"max_tokens\":1024"));
    assert!(raw_req.contains("\"get_weather\""));
    assert!(raw_req.contains("\"json_object\""));

    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].id, "call_123");
    assert_eq!(out.tool_calls[0].name, "get_weather");
    assert_eq!(out.tool_calls[0].arguments["city"], "Hanoi");
    assert_eq!(out.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(
        out.usage,
        Some(TokenUsage {
            prompt_tokens: 25,
            completion_tokens: 12,
            total_tokens: 37,
        })
    );
}

#[tokio::test]
async fn openai_compat_streams_sse_chunks_and_tool_calls() {
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc_1\",\"function\":{\"name\":\"ping\",\"arguments\":\"{\\\"x\\\":\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
        "data: [DONE]\n\n"
    )
    .to_string();

    let (base_url, rx) = spawn_mock_server(sse, "text/event-stream").await;
    let engine = OpenAiCompatibleEngine::new(&base_url, "", "qwen2.5");

    let chunks = Arc::new(Mutex::new(Vec::<String>::new()));
    let chunks_cb = chunks.clone();
    let input = EngineInput::new("Say hi", ".").with_stream_callback(Arc::new(move |c| {
        chunks_cb.lock().unwrap().push(c.to_string());
    }));

    let out = engine.execute(&input).await.unwrap();
    let raw_req = rx.await.unwrap();
    assert!(raw_req.contains("\"stream\":true"));

    assert_eq!(out.stdout_content, "Hello world!");
    assert_eq!(*chunks.lock().unwrap(), vec!["Hello ", "world!"]);
    assert_eq!(out.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].arguments, json!({"x": 1}));
    assert_eq!(out.usage.unwrap().total_tokens, 15);
}
