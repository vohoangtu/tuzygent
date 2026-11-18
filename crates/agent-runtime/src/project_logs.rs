use crate::task_store::{SqliteTaskStore, TaskStore};
use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectLogKind {
    All,
    Brain,
    Scratchpad,
    Session,
    Tasks,
}

impl ProjectLogKind {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "brain" | "usage" | "api" => Self::Brain,
            "scratchpad" | "execution" | "exec" => Self::Scratchpad,
            "session" | "sessions" | "chat" => Self::Session,
            "task" | "tasks" => Self::Tasks,
            _ => Self::All,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectLogEntry {
    pub timestamp: String,
    pub source: String,
    pub level: String,
    pub project_id: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ProjectLogQuery {
    pub project_id: Option<String>,
    pub kind: ProjectLogKind,
    pub limit: usize,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
}

impl Default for ProjectLogQuery {
    fn default() -> Self {
        Self {
            project_id: None,
            kind: ProjectLogKind::All,
            limit: 50,
            session_id: None,
            task_id: None,
        }
    }
}

/// Collect unified project logs across brain usage, scratchpads, sessions, and tasks.
pub async fn read_project_logs(
    workspace: &Path,
    query: &ProjectLogQuery,
) -> Result<Vec<ProjectLogEntry>> {
    let mut entries = Vec::new();
    let limit = query.limit.clamp(1, 500);

    if matches!(query.kind, ProjectLogKind::All | ProjectLogKind::Brain) {
        collect_brain_usage_logs(workspace, query, &mut entries);
    }
    if matches!(query.kind, ProjectLogKind::All | ProjectLogKind::Scratchpad) {
        collect_scratchpad_logs(workspace, query, &mut entries);
    }
    if matches!(query.kind, ProjectLogKind::All | ProjectLogKind::Session) {
        collect_session_logs(workspace, query, limit, &mut entries);
    }
    if matches!(query.kind, ProjectLogKind::All | ProjectLogKind::Tasks) {
        collect_task_logs(workspace, query, limit, &mut entries).await;
    }

    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    entries.truncate(limit);
    Ok(entries)
}

fn collect_brain_usage_logs(
    workspace: &Path,
    query: &ProjectLogQuery,
    out: &mut Vec<ProjectLogEntry>,
) {
    let path = workspace.join("storage").join("brain_usage.jsonl");
    let Ok(raw) = std::fs::read_to_string(path) else {
        return;
    };
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(val) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let tenant = val
            .get("tenant_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        if let Some(ref filter_proj) = query.project_id {
            if &tenant != filter_proj {
                continue;
            }
        }
        let model = val.get("model").and_then(|v| v.as_str()).unwrap_or("-");
        let total = val.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let ts = val
            .get("timestamp")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        out.push(ProjectLogEntry {
            timestamp: ts,
            source: "brain_usage".to_string(),
            level: "INFO".to_string(),
            project_id: tenant,
            summary: format!("model={} total_tokens={}", model, total),
            details: Some(val),
        });
    }
}

fn collect_scratchpad_logs(
    workspace: &Path,
    query: &ProjectLogQuery,
    out: &mut Vec<ProjectLogEntry>,
) {
    let mem_dir = workspace.join("memory");
    let Ok(read_dir) = std::fs::read_dir(&mem_dir) else {
        return;
    };
    let proj_label = query
        .project_id
        .clone()
        .unwrap_or_else(|| "workspace".to_string());

    for entry in read_dir.flatten() {
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !fname.starts_with("scratchpad") || !fname.ends_with(".md") {
            continue;
        }
        if let Some(ref tid) = query.task_id {
            if !fname.contains(tid) && fname != "scratchpad.md" {
                continue;
            }
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        let start = lines.len().saturating_sub(25);
        for line in &lines[start..] {
            out.push(ProjectLogEntry {
                timestamp: String::new(),
                source: format!("scratchpad:{}", fname),
                level: if line.contains("ERROR") || line.contains("Failed") {
                    "ERROR".to_string()
                } else {
                    "INFO".to_string()
                },
                project_id: proj_label.clone(),
                summary: line.chars().take(240).collect(),
                details: None,
            });
        }
    }
}

fn collect_session_logs(
    workspace: &Path,
    query: &ProjectLogQuery,
    limit: usize,
    out: &mut Vec<ProjectLogEntry>,
) {
    let db_path = workspace.join("memory").join("sessions.db");
    if !db_path.exists() {
        return;
    }
    let Ok(conn) = Connection::open(db_path) else {
        return;
    };
    let sql = "SELECT project, session_id, role, content, created_at FROM session_messages ORDER BY id DESC LIMIT ?1";
    let Ok(mut stmt) = conn.prepare(sql) else {
        return;
    };
    let rows = stmt.query_map([limit as i64 * 2], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, String>(4)?,
        ))
    });
    let Ok(rows) = rows else {
        return;
    };
    for row in rows.flatten() {
        let (project, session_id, role, content, created_at) = row;
        if let Some(ref filter_proj) = query.project_id {
            if &project != filter_proj {
                continue;
            }
        }
        if let Some(ref filter_sid) = query.session_id {
            if &session_id != filter_sid {
                continue;
            }
        }
        let preview: String = content.chars().take(160).collect();
        out.push(ProjectLogEntry {
            timestamp: created_at,
            source: format!("session:{}", session_id),
            level: "INFO".to_string(),
            project_id: project,
            summary: format!("[{}] {}", role, preview),
            details: None,
        });
    }
}

async fn collect_task_logs(
    workspace: &Path,
    query: &ProjectLogQuery,
    limit: usize,
    out: &mut Vec<ProjectLogEntry>,
) {
    let db_path = workspace.join("storage").join("tasks.db");
    if !db_path.exists() {
        return;
    }
    let Ok(store) = SqliteTaskStore::new(workspace) else {
        return;
    };
    let Ok(tasks) = store.list_tasks(limit, 0).await else {
        return;
    };
    let proj_label = query
        .project_id
        .clone()
        .unwrap_or_else(|| "default".to_string());
    for t in tasks {
        if let Some(ref tid) = query.task_id {
            if &t.id.0 != tid {
                continue;
            }
        }
        let status_str = format!("{:?}", t.status);
        let level = if status_str.starts_with("Failed") {
            "ERROR"
        } else {
            "INFO"
        };
        out.push(ProjectLogEntry {
            timestamp: t.created_at.to_rfc3339(),
            source: format!("task:{}", t.id.0),
            level: level.to_string(),
            project_id: proj_label.clone(),
            summary: format!("{} -> {}", t.title, status_str),
            details: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_memory::SessionMemory;

    #[tokio::test]
    async fn read_project_logs_filters_by_project_and_kind() {
        let dir = std::env::temp_dir().join(format!("tuzy_proj_logs_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("storage")).unwrap();
        std::fs::create_dir_all(dir.join("memory")).unwrap();

        std::fs::write(
            dir.join("storage/brain_usage.jsonl"),
            concat!(
                "{\"timestamp\":\"2026-11-18T10:00:00Z\",\"tenant_id\":\"proj-a\",\"model\":\"gpt-4o\",\"total_tokens\":100}\n",
                "{\"timestamp\":\"2026-11-18T10:01:00Z\",\"tenant_id\":\"proj-b\",\"model\":\"ollama\",\"total_tokens\":50}\n"
            ),
        )
        .unwrap();

        let sess = SessionMemory::open(&dir, "proj-a", "s1").unwrap();
        sess.append("user", "hello from proj-a").unwrap();

        let logs = read_project_logs(
            &dir,
            &ProjectLogQuery {
                project_id: Some("proj-a".to_string()),
                kind: ProjectLogKind::All,
                limit: 20,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(logs.len(), 2);
        assert!(logs.iter().all(|l| l.project_id == "proj-a"));

        let brain_only = read_project_logs(
            &dir,
            &ProjectLogQuery {
                project_id: Some("proj-a".to_string()),
                kind: ProjectLogKind::Brain,
                limit: 20,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(brain_only.len(), 1);
        assert_eq!(brain_only[0].source, "brain_usage");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
