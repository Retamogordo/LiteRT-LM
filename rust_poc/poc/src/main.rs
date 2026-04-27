mod chat_loop;
mod date_time_tool;

use anyhow::{anyhow, Context, Result};
use litert_lm::{Api, ConversationConfig, Engine, EngineSettings};
use std::env;

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let so_path = arg_value(&args, "--so").or_else(|| env::var("LITERT_LM_SO").ok());
    let so_path = so_path.ok_or_else(|| anyhow!("missing --so <path> (or set LITERT_LM_SO)"))?;

    let model_path = arg_value(&args, "--model").or_else(|| env::var("LITERT_LM_MODEL").ok());
    let model_path = model_path.ok_or_else(|| anyhow!("missing --model <path> (or set LITERT_LM_MODEL)"))?;

    let prompt = "Please introduce yourself.".to_string();
    let backend = arg_value(&args, "--backend").unwrap_or_else(|| "cpu".to_string());
    let max_tokens: Option<i32> = arg_value(&args, "--max-tokens").and_then(|s| s.parse().ok());

    let api = unsafe { Api::load(&so_path) }.context("loading native LiteRT-LM .so")?;
    
    let min_log_level: i32 = arg_value(&args, "--min-log-level").and_then(|s| s.parse().ok()).unwrap_or(3);
    api.set_min_log_level(min_log_level);

    let mut settings = EngineSettings::new(&api, &model_path, &backend)
        .with_context(|| format!("creating EngineSettings backend={backend} model={model_path}"))?;
    // Avoid forcing a max token count unless explicitly requested; some models
    // only support specific max-token values.
    if let Some(max_tokens) = max_tokens {
        settings.set_max_num_tokens(max_tokens);
    }

    let engine = Engine::new(&api, &settings).context("creating Engine")?;

    let tools = serde_json::json!([date_time_tool::DateTimeTool.declaration()]);
    let conversation_config = ConversationConfig::create(
        &engine,
        /*session_config=*/ None,
        /*system_message=*/ None,
        /*tools=*/ Some(&tools),
        /*messages=*/ None,
        /*enable_constrained_decoding=*/ false,
    )
    .context("creating ConversationConfig")?;

    let conversation = engine
        .create_conversation_with_config(Some(&conversation_config))
        .context("creating Conversation (with tools)")?;

    chat_loop::chat_loop(conversation, &prompt)?;


    Ok(())
}

