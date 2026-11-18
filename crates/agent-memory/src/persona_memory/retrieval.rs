use anyhow::Result;
use rusqlite::params;

use super::service::MemoryService;
use super::types::{MemoryRecord, RetrievedContext};

impl MemoryService {
    pub(crate) fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
        Ok(MemoryRecord {
            id: r.get(0)?,
            persona_id: r.get(1)?,
            kind: r.get(2)?,
            scope: r.get(3)?,
            status: r.get(4)?,
            content: r.get(5)?,
            conditions: r.get(6)?,
            source: r.get(7)?,
            verified: r.get(8)?,
            profile_version: r.get(9)?,
            created_at: r.get(10)?,
        })
    }

    pub fn list(&self) -> Result<Vec<MemoryRecord>> {
        let db = self.connect()?;
        let mut stmt = db.prepare(
            "SELECT id,persona,kind,scope,status,content,conditions,source,verified,version,created FROM memories WHERE project=?1 AND (persona=?2 OR (scope='project' AND status='active')) ORDER BY created DESC,id",
        )?;
        let rows = stmt
            .query_map(params![self.project, self.persona], Self::row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn show(&self, id: &str) -> Result<MemoryRecord> {
        self.list()?
            .into_iter()
            .find(|r| r.id == id)
            .ok_or_else(|| anyhow::anyhow!("memory unavailable"))
    }

    pub fn retrieve(&self, task_id: &str, query: &str) -> Result<RetrievedContext> {
        let tokens = crate::extract_query_tokens(query);
        let mut result = RetrievedContext {
            memory_ids: vec![],
            text: String::new(),
        };
        if tokens.is_empty() {
            return Ok(result);
        }
        let expression = tokens
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");
        let db = self.connect()?;
        let mut stmt = db.prepare(
            "SELECT m.id FROM memory_search JOIN memories m ON m.id=memory_search.id WHERE memory_search MATCH ?1 AND m.project=?2 AND (m.persona=?3 OR m.scope='project') AND m.status='active' AND m.kind IN ('procedure','knowledge') ORDER BY bm25(memory_search,0,1,2), m.verified DESC,m.created DESC,m.id LIMIT 8",
        )?;
        let ids = stmt
            .query_map(params![expression, self.project, self.persona], |r| {
                r.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for id in ids {
            let r = self.show(&id)?;
            let entry = format!(
                "\n[Memory {} | source {}]\n{}\nApplies when: {}\n",
                r.id, r.source, r.content, r.conditions
            );
            let remaining = 12000usize.saturating_sub(result.text.chars().count());
            if remaining == 0 {
                break;
            }
            result.text.extend(entry.chars().take(remaining));
            result.memory_ids.push(id);
        }
        db.execute(
            "INSERT INTO retrievals VALUES(?1,?2,?3,?4,?5)",
            params![
                self.project,
                self.persona,
                task_id,
                serde_json::to_string(&result.memory_ids)?,
                chrono::Utc::now().to_rfc3339()
            ],
        )?;
        Ok(result)
    }

    pub fn upsert_embedding(&self, memory_id: &str, model: &str, vector: &[f32]) -> Result<()> {
        let db = self.connect()?;
        let blob = crate::embeddings::encode_f32_le(vector);
        db.execute(
            "INSERT INTO memory_embeddings(memory_id, model, dims, vector) VALUES(?1, ?2, ?3, ?4)
             ON CONFLICT(memory_id) DO UPDATE SET model=excluded.model, dims=excluded.dims, vector=excluded.vector",
            params![memory_id, model, vector.len() as i64, blob],
        )?;
        Ok(())
    }

    /// Hybrid retrieval combining FTS5 BM25 keyword search and dense vector cosine similarity
    /// via Reciprocal Rank Fusion (RRF).
    pub async fn retrieve_hybrid(
        &self,
        task_id: &str,
        query: &str,
        embedder: &dyn crate::embeddings::EmbeddingProvider,
    ) -> Result<RetrievedContext> {
        let bm25_ctx = self.retrieve(task_id, query)?;
        let query_vecs = embedder.embed(&[query.to_string()]).await?;
        let Some(query_vec) = query_vecs.into_iter().next() else {
            return Ok(bm25_ctx);
        };

        let db = self.connect()?;
        let mut stmt = db.prepare(
            "SELECT e.memory_id, e.vector FROM memory_embeddings e
             JOIN memories m ON m.id = e.memory_id
             WHERE m.project=?1 AND (m.persona=?2 OR m.scope='project')
               AND m.status='active' AND m.kind IN ('procedure','knowledge')",
        )?;
        let mut vec_scored: Vec<(String, f32)> = stmt
            .query_map(params![self.project, self.persona], |r| {
                let id: String = r.get(0)?;
                let blob: Vec<u8> = r.get(1)?;
                Ok((id, blob))
            })?
            .filter_map(|row| row.ok())
            .map(|(id, blob)| {
                let vec = crate::embeddings::decode_f32_le(&blob);
                let sim = crate::embeddings::cosine_similarity(&query_vec, &vec);
                (id, sim)
            })
            .collect();
        vec_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut rrf: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for (rank, id) in bm25_ctx.memory_ids.iter().enumerate() {
            *rrf.entry(id.clone()).or_insert(0.0) += 1.0 / (60.0 + rank as f32 + 1.0);
        }
        for (rank, (id, sim)) in vec_scored.into_iter().take(8).enumerate() {
            if sim > 0.0 {
                *rrf.entry(id).or_insert(0.0) += 1.0 / (60.0 + rank as f32 + 1.0);
            }
        }

        let mut ranked: Vec<(String, f32)> = rrf.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = RetrievedContext {
            memory_ids: Vec::new(),
            text: String::new(),
        };
        for (id, _) in ranked.into_iter().take(8) {
            let Ok(r) = self.show(&id) else {
                continue;
            };
            let entry = format!(
                "\n[Memory {} | source {}]\n{}\nApplies when: {}\n",
                r.id, r.source, r.content, r.conditions
            );
            let remaining = 12000usize.saturating_sub(result.text.chars().count());
            if remaining == 0 {
                break;
            }
            result.text.extend(entry.chars().take(remaining));
            result.memory_ids.push(id);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::embeddings::EmbeddingProvider;
    use crate::persona_memory::service::MemoryService;
    use anyhow::Result;
    use async_trait::async_trait;
    use rusqlite::params;

    struct MockEmbedder;

    #[async_trait]
    impl EmbeddingProvider for MockEmbedder {
        fn model_name(&self) -> &str {
            "mock-embed"
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    if t.contains("automobile") || t.contains("car") {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn hybrid_retrieval_finds_semantic_match_missed_by_bm25() {
        let dir = std::env::temp_dir().join(format!("tuzy_hyb_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let svc = MemoryService::open(&dir, "p1", "a1").unwrap();
        {
            let db = svc.connect().unwrap();
            db.execute(
                "INSERT INTO memories(id,project,persona,kind,scope,status,content,conditions,source,verified,version,created)
                 VALUES('mem-car','p1','a1','knowledge','project','active','Always check car tire pressure','vehicle maintenance','test',1,1,'2026-01-01')",
                params![],
            )
            .unwrap();
        }
        svc.upsert_embedding("mem-car", "mock-embed", &[1.0, 0.0]).unwrap();

        let ctx = svc
            .retrieve_hybrid("t1", "automobile inspection", &MockEmbedder)
            .await
            .unwrap();
        assert!(ctx.memory_ids.contains(&"mem-car".to_string()));
        assert!(ctx.text.contains("tire pressure"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
