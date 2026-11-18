use agent_runtime::{read_project_logs, ProjectLogKind, ProjectLogQuery};
use anyhow::Result;

use crate::cli::{LogsArgs, OutputFormat};

pub async fn handle(args: LogsArgs) -> Result<()> {
    let query = ProjectLogQuery {
        project_id: args.project.clone(),
        kind: ProjectLogKind::parse(&args.kind),
        limit: args.limit,
        session_id: args.session.clone(),
        task_id: args.task.clone(),
    };
    let logs = read_project_logs(args.workspace.as_path(), &query).await?;

    match args.format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&logs)?);
        }
        OutputFormat::Table => {
            if logs.is_empty() {
                println!("(No logs found for workspace: {})", args.workspace.display());
                return Ok(());
            }
            println!(
                "{:<26} | {:<18} | {:<5} | {:<12} | {}",
                "TIMESTAMP", "SOURCE", "LEVEL", "PROJECT", "SUMMARY"
            );
            println!("{}", "-".repeat(100));
            for entry in logs {
                let ts = if entry.timestamp.is_empty() {
                    "-"
                } else {
                    entry.timestamp.as_str()
                };
                println!(
                    "{:<26} | {:<18} | {:<5} | {:<12} | {}",
                    ts.chars().take(26).collect::<String>(),
                    entry.source.chars().take(18).collect::<String>(),
                    entry.level,
                    entry.project_id.chars().take(12).collect::<String>(),
                    entry.summary
                );
            }
        }
    }
    Ok(())
}
