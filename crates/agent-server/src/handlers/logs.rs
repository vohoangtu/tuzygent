use std::path::Path;

use agent_core::prelude::{read_project_logs, ProjectLogKind, ProjectLogQuery};
use axum::{
    extract::{Extension, Path as AxumPath, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;

use crate::auth::TenantContext;
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct ProjectLogsQueryParams {
    pub project_id: Option<String>,
    pub kind: Option<String>,
    pub limit: Option<usize>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
}

/// GET /api/v1/projects/logs
pub async fn list_project_logs_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantContext>>,
    Query(params): Query<ProjectLogsQueryParams>,
) -> impl IntoResponse {
    query_logs(state, tenant_ext, None, params).await
}

/// GET /api/v1/projects/{project_id}/logs
pub async fn get_project_logs_by_id_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantContext>>,
    AxumPath(project_id): AxumPath<String>,
    Query(params): Query<ProjectLogsQueryParams>,
) -> impl IntoResponse {
    query_logs(state, tenant_ext, Some(project_id), params).await
}

async fn query_logs(
    state: AppState,
    tenant_ext: Option<Extension<TenantContext>>,
    path_project: Option<String>,
    params: ProjectLogsQueryParams,
) -> impl IntoResponse {
    let tenant = tenant_ext.map(|Extension(t)| t).unwrap_or_default();
    let requested_project = path_project.or(params.project_id);

    let effective_project = if tenant.policy.is_some() {
        if let Some(ref req_proj) = requested_project {
            if req_proj != &tenant.tenant_id {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "error": format!("Tenant '{}' cannot read logs for project '{}'", tenant.tenant_id, req_proj)
                    })),
                )
                    .into_response();
            }
        }
        Some(tenant.tenant_id)
    } else {
        requested_project
    };

    let query = ProjectLogQuery {
        project_id: effective_project.clone(),
        kind: params
            .kind
            .as_deref()
            .map(ProjectLogKind::parse)
            .unwrap_or(ProjectLogKind::All),
        limit: params.limit.unwrap_or(50),
        session_id: params.session_id,
        task_id: params.task_id,
    };

    match read_project_logs(Path::new(&state.workspace), &query).await {
        Ok(logs) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "project_id": effective_project,
                "count": logs.len(),
                "logs": logs,
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}
