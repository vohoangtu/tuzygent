use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single turn stored in `SessionMemory`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTurn {
    pub id: i64,
    pub project: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

/// SQLite-backed conversation session history isolated by `(project, session_id)`.
pub struct SessionMemory {
    db_path: PathBuf,
    project: String,
    session_id: String,
}

impl SessionMemory {
    pub fn open(
        workspace: &Path,
        project: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self> {
        let dir = workspace.join("memory");
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("sessions.db");
        let conn = Connection::open(&db_path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS session_messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session_lookup
                ON session_messages(project, session_id, id);
            "#,
        )?;
        Ok(Self {
            db_path,
            project: project.into(),
            session_id: session_id.into(),
        })
    }

    fn connect(&self) -> Result<Connection> {
        let conn = Connection::open(&self.db_path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    pub fn append(&self, role: &str, content: &str) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "INSERT INTO session_messages(project, session_id, role, content, created_at) VALUES(?1, ?2, ?3, ?4, ?5)",
            params![
                self.project,
                self.session_id,
                role,
                content,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<SessionTurn>> {
        let db = self.connect()?;
        let mut stmt = db.prepare(
            "SELECT id, project, session_id, role, content, created_at FROM session_messages WHERE project=?1 AND session_id=?2 ORDER BY id ASC",
        )?;
        let rows = stmt
            .query_map(params![self.project, self.session_id], |r| {
                Ok(SessionTurn {
                    id: r.get(0)?,
                    project: r.get(1)?,
                    session_id: r.get(2)?,
                    role: r.get(3)?,
                    content: r.get(4)?,
                    created_at: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Return recent turns fitting within `max_chars` (always preserving the newest turns).
    pub fn list_within_budget(&self, max_chars: usize) -> Result<Vec<SessionTurn>> {
        let all = self.list()?;
        let mut selected = Vec::new();
        let mut used = 0usize;
        for turn in all.into_iter().rev() {
            let cost = turn.content.chars().count() + turn.role.chars().count() + 8;
            if used + cost > max_chars && !selected.is_empty() {
                break;
            }
            used += cost;
            selected.push(turn);
        }
        selected.reverse();
        Ok(selected)
    }

    pub fn clear(&self) -> Result<()> {
        let db = self.connect()?;
        db.execute(
            "DELETE FROM session_messages WHERE project=?1 AND session_id=?2",
            params![self.project, self.session_id],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_memory_isolates_projects_and_respects_budget() {
        let dir = std::env::temp_dir().join(format!("tuzy_sess_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let s1 = SessionMemory::open(&dir, "tenant-a", "s1").unwrap();
        let s2 = SessionMemory::open(&dir, "tenant-b", "s1").unwrap();

        s1.append("user", "first message").unwrap();
        s1.append("assistant", "second message").unwrap();
        s1.append("user", "latest message").unwrap();
        s2.append("user", "other tenant").unwrap();

        assert_eq!(s1.list().unwrap().len(), 3);
        assert_eq!(s2.list().unwrap().len(), 1);

        let budgeted = s1.list_within_budget(30).unwrap();
        assert_eq!(budgeted.len(), 1);
        assert_eq!(budgeted[0].content, "latest message");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
