use anyhow::{anyhow, Context, Result};
use clap::Parser;
use localai::chat_api::{ChatClient, ChatConfig, Message};
use localai::gui::run_gui;
use localai::guide_knowledge::{
    build_grounded_system_prompt, build_grounded_user_prompt, GuideKnowledgeBase,
};
use std::io::{self, Write};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Base URL de l'API Ollama
    #[arg(long, default_value = "http://localhost:11434")]
    host: String,

    /// Modele Ollama
    #[arg(long, default_value = "qwen2.5-coder:14b")]
    model: String,

    /// Message system
    #[arg(
        long,
        default_value = "Tu es JARVIS V1, un assistant IA specialise en informatique"
    )]
    system: String,

    /// Temperature (mode non-streaming OpenAI-compatible)
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    /// Max tokens (mode non-streaming OpenAI-compatible)
    #[arg(long, default_value_t = 1024)]
    max_tokens: u32,

    /// Timeout HTTP en secondes
    #[arg(long, default_value_t = 600)]
    timeout_seconds: u64,

    /// Active le mode streaming en CLI (utilise /api/chat)
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Force le mode CLI (sinon la GUI est lancee)
    #[arg(long, default_value_t = false)]
    cli: bool,

    /// Prompt utilisateur (optionnel: si fourni, execute une requete CLI)
    prompt: Option<String>,

    /// Chemin du guide officiel utilise en mode RAG
    #[arg(long, default_value = "guide-production-rochias.txt")]
    guide: String,

    /// Desactive l'usage du guide (mode libre)
    #[arg(long, default_value_t = false)]
    no_guide: bool,

    /// Nombre maximum d'extraits recupere pour la reponse
    #[arg(long, default_value_t = 6)]
    source_limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cli_num_ctx = if args.model == "gpt-oss:20b" {
        Some(2048)
    } else {
        None
    };
    let config = ChatConfig {
        host: args.host,
        model: args.model,
        system: args.system,
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        timeout_seconds: args.timeout_seconds,
        reasoning_effort: None,
        num_ctx: cli_num_ctx,
        ..ChatConfig::default()
    };

    if args.cli || args.prompt.is_some() {
        let prompt = args
            .prompt
            .ok_or_else(|| anyhow!("Mode CLI: il faut fournir un prompt"))?;
        run_cli(
            config,
            args.stream,
            &prompt,
            &args.guide,
            args.no_guide,
            args.source_limit,
        )?;
    } else {
        run_gui(config)?;
    }

    Ok(())
}

fn run_cli(
    config: ChatConfig,
    stream: bool,
    prompt: &str,
    guide_path: &str,
    no_guide: bool,
    source_limit: usize,
) -> Result<()> {
    let gpt_oss_mode = config.model == "gpt-oss:20b";
    let base_system = config.system.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Impossible de creer le runtime async")?;
    let client = ChatClient::new(config)?;
    let (messages, source_count) =
        prepare_messages(prompt, &base_system, guide_path, no_guide, source_limit)?;

    if stream {
        runtime.block_on(client.stream_with_messages(messages, |chunk| {
            print!("{chunk}");
            io::stdout().flush().ok();
        }))?;
        println!();
    } else if gpt_oss_mode {
        let response = runtime.block_on(client.oneshot_ollama_with_messages(messages))?;
        if !response.content.is_empty() {
            println!("{}", response.content);
        } else if !response.thinking.is_empty() {
            println!("{}", response.thinking);
        } else {
            println!("<reponse vide>");
        }
    } else {
        let text = runtime.block_on(client.oneshot_with_messages(messages))?;
        println!("{text}");
    }

    if source_count > 0 {
        eprintln!("Sources utilisees: {source_count}");
    }

    Ok(())
}

fn prepare_messages(
    prompt: &str,
    base_system: &str,
    guide_path: &str,
    no_guide: bool,
    source_limit: usize,
) -> Result<(Vec<Message>, usize)> {
    if no_guide {
        return Ok((build_plain_messages(prompt, base_system), 0));
    }

    let guide_file = Path::new(guide_path);
    if !guide_file.exists() {
        return Ok((build_plain_messages(prompt, base_system), 0));
    }

    let kb = GuideKnowledgeBase::load(guide_file)?;
    let hits = kb.search(prompt, source_limit.max(1));
    let source_count = hits.len();
    let grounded_user = build_grounded_user_prompt(prompt, &hits);
    let grounded_system = build_grounded_system_prompt(base_system);

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: grounded_system,
        },
        Message {
            role: "user".to_string(),
            content: grounded_user,
        },
    ];

    Ok((messages, source_count))
}

fn build_plain_messages(prompt: &str, base_system: &str) -> Vec<Message> {
    vec![
        Message {
            role: "system".to_string(),
            content: base_system.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        },
    ]
}
