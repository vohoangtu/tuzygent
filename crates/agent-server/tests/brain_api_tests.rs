use std::sync::Arc;

use agent_core::prelude::*;
use agent_engine::AccountVault;
use agent_runtime::{SqliteTaskStore, TaskService};
use agent_server::gateway::{resolve_gateway_target, GatewayTarget};
use agent_server::{build_app, AppState, AuthState, TenantKeyPolicy};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

fn setup_test_app() -> (axum::Router, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("tuzy_srv_brain_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let ws = dir.to_string_lossy().to_string();

    let bus = InterAgentBus::new(32);
    let supervisor = Arc::new(AgentSupervisor::new(bus.clone()));
    let vault = Arc::new(AccountVault::new());
    let store = Arc::new(SqliteTaskStore::new(&ws).unwrap());
    let task_service = Arc::new(TaskService::new(store, &ws, 2, None, None));

    let state = AppState::new(supervisor, bus, ws, vault, task_service);
    let auth = AuthState::new(Arc::new("master-secret".to_string()), vec![]).with_tenant_key(
        "tenant-a-key",
        TenantKeyPolicy {
            tenant_id: "tenant-a".to_string(),
            allowed_engines: vec!["mock".to_string()],
            max_tokens_per_request: Some(512),
        },
    );
    (build_app(state, auth), dir)
}

#[test]
fn resolve_gateway_target_routes_api_and_account_engines() {
    assert_eq!(
        resolve_gateway_target("openai/gpt-4o-mini").unwrap(),
        GatewayTarget::ConfiguredApi {
            engine_name: "openai".to_string(),
            model_override: Some("gpt-4o-mini".to_string()),
        }
    );
    assert_eq!(
        resolve_gateway_target("ollama/qwen2.5:latest").unwrap(),
        GatewayTarget::ConfiguredApi {
            engine_name: "ollama".to_string(),
            model_override: Some("qwen2.5:latest".to_string()),
        }
    );
    assert!(matches!(
        resolve_gateway_target("codex/gpt-4o").unwrap(),
        GatewayTarget::AccountRotated(_, Some(_))
    ));
}

#[tokio::test]
async fn brain_run_enforces_tenant_policy_and_serves_openapi() {
    let (app, dir) = setup_test_app();

    // 1. OpenAPI spec
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["openapi"], "3.1.0");
    assert!(json["paths"]["/api/v1/brain/run"].is_object());

    // 2. Tenant A allowed engine ("mock")
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/brain/run")
                .header("authorization", "Bearer tenant-a-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "messages": [{"role": "user", "content": "hello brain"}],
                        "session_id": "s-1",
                        "max_tokens": 256
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["tenant_id"], "tenant-a");
    assert_eq!(json["session_id"], "s-1");
    assert!(json["output"].as_str().unwrap().contains("hello brain"));

    // 3. Tenant A disallowed engine ("openai/gpt-4o") -> 403
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/brain/run")
                .header("authorization", "Bearer tenant-a-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "openai/gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // 4. Tenant A max_tokens exceeded -> 403
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/brain/run")
                .header("authorization", "Bearer tenant-a-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "mock",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2048
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let _ = std::fs::remove_dir_all(&dir);
}
