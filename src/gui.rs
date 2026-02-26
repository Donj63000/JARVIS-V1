use crate::chat_api::{AssistantDelta, ChatClient, ChatConfig, Message};
use crate::guide_knowledge::{
    build_grounded_system_prompt, build_grounded_user_prompt, GuideKnowledgeBase,
};
use anyhow::{anyhow, Context, Result};
use eframe::egui::{self, Align, Button, Color32, Frame, Layout, RichText, ScrollArea, Stroke};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const MODEL_OPTIONS: [&str; 2] = ["qwen2.5-coder:14b", "gpt-oss:20b"];
const GPT_OSS_MODEL: &str = "gpt-oss:20b";
const WIN95_DESKTOP: Color32 = Color32::from_rgb(0, 128, 128);
const WIN95_FACE: Color32 = Color32::from_rgb(192, 192, 192);
const WIN95_LIGHT: Color32 = Color32::from_rgb(223, 223, 223);
const WIN95_HIGHLIGHT: Color32 = Color32::from_rgb(255, 255, 255);
const WIN95_SHADOW: Color32 = Color32::from_rgb(128, 128, 128);
const WIN95_SHADOW_DARK: Color32 = Color32::from_rgb(64, 64, 64);
const WIN95_TITLE_BLUE: Color32 = Color32::from_rgb(0, 0, 128);
const WIN95_TEXT: Color32 = Color32::from_rgb(0, 0, 0);
const WIN95_WINDOW_BG: Color32 = Color32::from_rgb(236, 233, 216);
const WIN95_INPUT_BG: Color32 = Color32::from_rgb(255, 255, 255);
const USER_BUBBLE: Color32 = Color32::from_rgb(235, 244, 255);
const ASSISTANT_BUBBLE: Color32 = Color32::from_rgb(255, 255, 228);
const LOGO_SIZE_MIN: f32 = 86.0;
const LOGO_SIZE_MAX: f32 = 128.0;
const LOGO_COLUMN_MIN_WIDTH: f32 = 104.0;
const LOGO_COLUMN_MAX_WIDTH: f32 = 156.0;
const HEADER_ROW_HEIGHT_QWEN: f32 = 78.0;
const HEADER_ROW_HEIGHT_GPT: f32 = 126.0;

#[derive(Clone, Copy)]
enum ChatRole {
    User,
    Assistant,
}

struct ChatLine {
    role: ChatRole,
    text: String,
    thinking: String,
    include_in_context: bool,
}

enum WorkerEvent {
    Chunk(String),
    Thinking(String),
    Done,
    Error(String),
}

#[derive(Clone, Copy)]
struct HeaderLayout {
    controls_col_width: f32,
    logo_col_width: f32,
    logo_size: f32,
    row_height: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReasoningLevel {
    Low,
    Medium,
    High,
}

impl ReasoningLevel {
    fn as_api_value(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    fn as_ui_label(self) -> &'static str {
        match self {
            Self::Low => "Bas",
            Self::Medium => "Moyen",
            Self::High => "Haut",
        }
    }
}

pub fn run_gui(config: ChatConfig) -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 680.0])
            .with_min_inner_size([620.0, 460.0])
            .with_title("JARVIS V1"),
        ..Default::default()
    };

    eframe::run_native(
        "JARVIS V1",
        native_options,
        Box::new(move |cc| {
            configure_theme(&cc.egui_ctx);
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(Messenger95App::new(config.clone())))
        }),
    )
    .map_err(|err| anyhow!("Impossible de lancer l'interface graphique: {err}"))
}

fn configure_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::light();
    style.visuals.override_text_color = Some(WIN95_TEXT);
    style.visuals.panel_fill = WIN95_DESKTOP;
    style.visuals.window_fill = WIN95_FACE;
    style.visuals.extreme_bg_color = WIN95_INPUT_BG;
    style.visuals.faint_bg_color = WIN95_WINDOW_BG;
    style.visuals.widgets.noninteractive.bg_fill = WIN95_FACE;
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, WIN95_SHADOW_DARK);
    style.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, WIN95_TEXT);
    style.visuals.widgets.inactive.bg_fill = WIN95_FACE;
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, WIN95_SHADOW_DARK);
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, WIN95_TEXT);
    style.visuals.widgets.hovered.bg_fill = WIN95_LIGHT;
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, WIN95_SHADOW_DARK);
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, WIN95_TEXT);
    style.visuals.widgets.active.bg_fill = WIN95_SHADOW;
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, WIN95_SHADOW_DARK);
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, WIN95_TEXT);
    style.visuals.selection.bg_fill = Color32::from_rgb(10, 36, 106);
    style.visuals.selection.stroke = Stroke::new(1.0, WIN95_HIGHLIGHT);
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.window_margin = egui::Margin::same(8);
    style
        .text_styles
        .insert(egui::TextStyle::Heading, egui::FontId::proportional(20.0));
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Monospace, egui::FontId::monospace(13.0));
    style
        .text_styles
        .insert(egui::TextStyle::Small, egui::FontId::proportional(12.0));
    ctx.set_style(style);
}

fn paint_bevel(
    ui: &egui::Ui,
    rect: egui::Rect,
    top_left_outer: Color32,
    bottom_right_outer: Color32,
    top_left_inner: Color32,
    bottom_right_inner: Color32,
) {
    if rect.width() < 3.0 || rect.height() < 3.0 {
        return;
    }

    let painter = ui.painter();
    let min = rect.min;
    let max = rect.max;

    painter.line_segment(
        [egui::pos2(min.x, min.y), egui::pos2(max.x - 1.0, min.y)],
        Stroke::new(1.0, top_left_outer),
    );
    painter.line_segment(
        [egui::pos2(min.x, min.y), egui::pos2(min.x, max.y - 1.0)],
        Stroke::new(1.0, top_left_outer),
    );
    painter.line_segment(
        [
            egui::pos2(min.x, max.y - 1.0),
            egui::pos2(max.x - 1.0, max.y - 1.0),
        ],
        Stroke::new(1.0, bottom_right_outer),
    );
    painter.line_segment(
        [
            egui::pos2(max.x - 1.0, min.y),
            egui::pos2(max.x - 1.0, max.y - 1.0),
        ],
        Stroke::new(1.0, bottom_right_outer),
    );

    if rect.width() < 5.0 || rect.height() < 5.0 {
        return;
    }

    painter.line_segment(
        [
            egui::pos2(min.x + 1.0, min.y + 1.0),
            egui::pos2(max.x - 2.0, min.y + 1.0),
        ],
        Stroke::new(1.0, top_left_inner),
    );
    painter.line_segment(
        [
            egui::pos2(min.x + 1.0, min.y + 1.0),
            egui::pos2(min.x + 1.0, max.y - 2.0),
        ],
        Stroke::new(1.0, top_left_inner),
    );
    painter.line_segment(
        [
            egui::pos2(min.x + 1.0, max.y - 2.0),
            egui::pos2(max.x - 2.0, max.y - 2.0),
        ],
        Stroke::new(1.0, bottom_right_inner),
    );
    painter.line_segment(
        [
            egui::pos2(max.x - 2.0, min.y + 1.0),
            egui::pos2(max.x - 2.0, max.y - 2.0),
        ],
        Stroke::new(1.0, bottom_right_inner),
    );
}

fn raised_panel<R>(
    ui: &mut egui::Ui,
    fill: Color32,
    margin: i8,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let panel = Frame::default()
        .fill(fill)
        .inner_margin(egui::Margin::same(margin))
        .show(ui, add_contents);
    paint_bevel(
        ui,
        panel.response.rect,
        WIN95_HIGHLIGHT,
        WIN95_SHADOW_DARK,
        WIN95_LIGHT,
        WIN95_SHADOW,
    );
    panel
}

fn sunken_panel<R>(
    ui: &mut egui::Ui,
    fill: Color32,
    margin: i8,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let panel = Frame::default()
        .fill(fill)
        .inner_margin(egui::Margin::same(margin))
        .show(ui, add_contents);
    paint_bevel(
        ui,
        panel.response.rect,
        WIN95_SHADOW_DARK,
        WIN95_HIGHLIGHT,
        WIN95_SHADOW,
        WIN95_LIGHT,
    );
    panel
}

fn compute_header_layout(available_width: f32, gpt_oss_mode: bool) -> HeaderLayout {
    let logo_size = (available_width * 0.10).clamp(LOGO_SIZE_MIN, LOGO_SIZE_MAX);
    let logo_col_width = (logo_size + 16.0).clamp(LOGO_COLUMN_MIN_WIDTH, LOGO_COLUMN_MAX_WIDTH);
    let controls_col_width = (available_width - logo_col_width - 6.0).max(240.0);
    let row_height = if gpt_oss_mode {
        HEADER_ROW_HEIGHT_GPT
    } else {
        HEADER_ROW_HEIGHT_QWEN
    };

    HeaderLayout {
        controls_col_width,
        logo_col_width,
        logo_size,
        row_height,
    }
}

fn render_logo(ui: &mut egui::Ui, size: f32) -> egui::Response {
    ui.add(
        egui::Image::new(egui::include_image!("../images/logo1.png"))
            .fit_to_exact_size(egui::vec2(size, size))
            .texture_options(egui::TextureOptions::LINEAR),
    )
}

fn is_connection_refused_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("os error 10061")
        || lower.contains("connection refused")
        || lower.contains("connect error")
        || lower.contains("aucune connexion")
}

fn is_model_memory_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("requires more system memory")
        || lower.contains("not enough memory")
        || lower.contains("insufficient memory")
        || lower.contains("out of memory")
}

fn format_worker_error(err: &str, host: &str) -> String {
    if is_connection_refused_error(err) {
        return format!(
            "Connexion impossible a {host}.\nLe serveur Ollama ne repond pas (connexion refusee).\nLance-le puis reessaie:\npowershell -ExecutionPolicy Bypass -File .\\scripts\\start-local-model.ps1 -Model qwen2.5-coder:14b"
        );
    }

    if is_model_memory_error(err) {
        return "Memoire insuffisante pour charger ce modele.\nActions recommandees:\n- Basculer sur un modele plus leger.\n- Fermer les applications lourdes (RAM libre).\n- Reduire num_ctx/max_tokens.\n\nDetail technique:\n".to_string()
            + err;
    }

    format!("Erreur: {err}")
}

fn format_anyhow_chain(err: &anyhow::Error) -> String {
    let mut out = err.to_string();
    for (index, cause) in err.chain().skip(1).enumerate() {
        out.push_str(&format!("\n  cause {}: {}", index + 1, cause));
    }
    out
}

pub struct Messenger95App {
    host: String,
    model: String,
    reasoning_level: ReasoningLevel,
    system: String,
    system_grounded: String,
    temperature: f32,
    max_tokens: u32,
    num_ctx: u32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    use_streaming: bool,
    timeout_seconds: u64,
    use_factory_data: bool,
    input: String,
    status: String,
    source_count_last_query: usize,
    pending: bool,
    messages: Vec<ChatLine>,
    pending_assistant_index: Option<usize>,
    worker_rx: Option<Receiver<WorkerEvent>>,
    guide_kb: Option<Arc<GuideKnowledgeBase>>,
    guide_info: String,
}

impl Messenger95App {
    fn new(config: ChatConfig) -> Self {
        let model = if MODEL_OPTIONS.contains(&config.model.as_str()) {
            config.model
        } else {
            "qwen2.5-coder:14b".to_string()
        };
        let system_grounded = build_grounded_system_prompt(&config.system);

        let (guide_kb, guide_info) = match GuideKnowledgeBase::try_load_default() {
            Some(kb) => {
                let path = kb.source_path().display().to_string();
                let info = format!("Guide charge: {path} ({} chunks)", kb.chunk_count());
                (Some(Arc::new(kb)), info)
            }
            None => (
                None,
                "Guide non charge (guide-production-rochias.txt absent)".to_string(),
            ),
        };

        let mut welcome = "Bienvenue dans JARVIS V1. Ecris un message puis clique sur Envoyer."
            .to_string();
        if guide_kb.is_some() {
            welcome.push_str(
                "\nDonnees USINE disponibles: active le toggle \"Donnees USINE\" pour repondre depuis la documentation Rochias.",
            );
        } else {
            welcome.push_str(
                "\nMode guide inactif: ajoute guide-production-rochias.txt a la racine du projet.",
            );
        }

        Self {
            host: config.host,
            model,
            reasoning_level: ReasoningLevel::Medium,
            system: config.system,
            system_grounded,
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            num_ctx: config.num_ctx.unwrap_or(0),
            repeat_penalty: config.repeat_penalty.unwrap_or(1.20),
            repeat_last_n: config.repeat_last_n.unwrap_or(256),
            use_streaming: true,
            timeout_seconds: config.timeout_seconds,
            use_factory_data: false,
            input: String::new(),
            status: "Pret".to_string(),
            source_count_last_query: 0,
            pending: false,
            messages: vec![ChatLine {
                role: ChatRole::Assistant,
                text: welcome,
                thinking: String::new(),
                include_in_context: false,
            }],
            pending_assistant_index: None,
            worker_rx: None,
            guide_kb,
            guide_info,
        }
    }

    fn is_gpt_oss_selected(&self) -> bool {
        self.model == GPT_OSS_MODEL
    }

    fn gpt_oss_min_predict(level: ReasoningLevel) -> u32 {
        match level {
            ReasoningLevel::Low => 2048,
            ReasoningLevel::Medium => 4096,
            ReasoningLevel::High => 8192,
        }
    }

    fn gpt_oss_min_ctx(level: ReasoningLevel) -> u32 {
        match level {
            ReasoningLevel::Low => 8192,
            ReasoningLevel::Medium => 16384,
            ReasoningLevel::High => 32768,
        }
    }

    fn compute_budget(&self) -> (u32, Option<u32>) {
        if !self.is_gpt_oss_selected() {
            return (self.max_tokens, None);
        }

        let min_predict = Self::gpt_oss_min_predict(self.reasoning_level);
        let num_predict = self.max_tokens.max(min_predict).clamp(512, 32768);

        let min_ctx = Self::gpt_oss_min_ctx(self.reasoning_level);
        let mut num_ctx = if self.num_ctx == 0 {
            min_ctx
        } else {
            self.num_ctx
        };
        num_ctx = num_ctx.max(min_ctx);
        num_ctx = num_ctx.max(num_predict.saturating_add(1024));

        (num_predict, Some(num_ctx))
    }

    fn current_thinking_text(&self) -> &str {
        if let Some(index) = self.pending_assistant_index {
            if let Some(line) = self.messages.get(index) {
                if !line.thinking.trim().is_empty() {
                    return line.thinking.as_str();
                }
            }
        }

        self.messages
            .iter()
            .rev()
            .find(|line| {
                matches!(line.role, ChatRole::Assistant) && !line.thinking.trim().is_empty()
            })
            .map(|line| line.thinking.as_str())
            .unwrap_or("Aucune chaine de pensee disponible pour le moment.")
    }

    fn send_prompt(&mut self) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.pending {
            return;
        }
        let gpt_oss_mode = self.is_gpt_oss_selected();

        let factory_mode = self.use_factory_data && self.guide_kb.is_some();
        let (prompt_for_model, source_count) = if factory_mode {
            if let Some(kb) = &self.guide_kb {
                let hits = kb.search(&prompt, 6);
                (build_grounded_user_prompt(&prompt, &hits), hits.len())
            } else {
                (prompt.clone(), 0)
            }
        } else {
            (prompt.clone(), 0)
        };

        let mut system_prompt = if factory_mode {
            self.system_grounded.clone()
        } else {
            self.system.clone()
        };
        system_prompt.push_str(
            "\n\nCONTRAINTES DE LANGUE:\n- Reponds exclusivement en francais.\n- Si une trace de raisonnement (thinking) est produite, elle doit aussi etre en francais.\n- Ne reponds jamais avec une reponse vide.",
        );

        let messages_for_request = self.build_context_messages(&prompt_for_model, &system_prompt);

        self.input.clear();
        self.pending = true;
        self.source_count_last_query = source_count;
        self.status = if factory_mode {
            format!("Generation... ({source_count} source(s))")
        } else if gpt_oss_mode {
            format!(
                "Generation... (raisonnement: {})",
                self.reasoning_level.as_api_value()
            )
        } else {
            "Generation en cours...".to_string()
        };

        self.messages.push(ChatLine {
            role: ChatRole::User,
            text: prompt.clone(),
            thinking: String::new(),
            include_in_context: true,
        });
        self.messages.push(ChatLine {
            role: ChatRole::Assistant,
            text: String::new(),
            thinking: String::new(),
            include_in_context: true,
        });
        self.pending_assistant_index = Some(self.messages.len() - 1);
        let (num_predict, num_ctx_opt) = self.compute_budget();

        let config = ChatConfig {
            host: self.host.clone(),
            model: self.model.clone(),
            system: system_prompt,
            temperature: self.temperature,
            max_tokens: num_predict,
            timeout_seconds: if gpt_oss_mode && self.reasoning_level == ReasoningLevel::High {
                self.timeout_seconds.max(1800)
            } else {
                self.timeout_seconds
            },
            reasoning_effort: if gpt_oss_mode {
                Some(self.reasoning_level.as_api_value().to_string())
            } else {
                None
            },
            num_ctx: num_ctx_opt,
            repeat_penalty: if gpt_oss_mode {
                Some(self.repeat_penalty)
            } else {
                None
            },
            repeat_last_n: if gpt_oss_mode {
                Some(self.repeat_last_n)
            } else {
                None
            },
            top_k: None,
            top_p: None,
            min_p: None,
            seed: None,
            stop: Vec::new(),
            keep_alive: Some("5m".to_string()),
            ..ChatConfig::default()
        };

        let use_streaming = self.use_streaming;
        let (tx, rx) = mpsc::channel::<WorkerEvent>();
        self.worker_rx = Some(rx);

        thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            let result = match runtime {
                Ok(rt) => rt.block_on(async {
                    let history = messages_for_request;
                    let client = ChatClient::new(config.clone())?;
                    let mut streamed_any_content = false;
                    let mut stream_error: Option<anyhow::Error> = None;

                    if use_streaming {
                        let stream_result = client
                            .stream_with_messages_detailed(
                                history.clone(),
                                |delta: AssistantDelta| {
                                    if !delta.thinking.is_empty() {
                                        let _ = tx.send(WorkerEvent::Thinking(delta.thinking));
                                    }
                                    if !delta.content.is_empty() {
                                        streamed_any_content = true;
                                        let _ = tx.send(WorkerEvent::Chunk(delta.content));
                                    }
                                },
                            )
                            .await;

                        if let Err(err) = stream_result {
                            stream_error = Some(err);
                        }
                    }

                    if stream_error.is_some() || !streamed_any_content {
                        let mut safe = config.clone();
                        if safe.reasoning_effort.as_deref() == Some("high") {
                            safe.reasoning_effort = Some("medium".to_string());
                        }
                        safe.repeat_penalty = Some(safe.repeat_penalty.unwrap_or(1.1).max(1.25));
                        safe.repeat_last_n = Some(safe.repeat_last_n.unwrap_or(64).max(256));
                        safe.max_tokens = safe.max_tokens.max(4096);
                        safe.num_ctx = Some(safe.num_ctx.unwrap_or(16384).max(16384));

                        let safe_client = ChatClient::new(safe)?;
                        let response = safe_client
                            .oneshot_ollama_with_messages(history.clone())
                            .await
                            .context("Fallback oneshot echoue")?;

                        if !response.thinking.is_empty() {
                            let _ = tx.send(WorkerEvent::Thinking(response.thinking));
                        }
                        if !response.content.is_empty() {
                            streamed_any_content = true;
                            let _ = tx.send(WorkerEvent::Chunk(response.content));
                        }
                    }

                    if gpt_oss_mode && !streamed_any_content {
                        let mut final_cfg = config.clone();
                        final_cfg.reasoning_effort = Some("low".to_string());
                        final_cfg.max_tokens = 2048.max(final_cfg.max_tokens.min(4096));
                        final_cfg.num_ctx = Some(final_cfg.num_ctx.unwrap_or(16384).max(16384));
                        final_cfg.repeat_penalty =
                            Some(final_cfg.repeat_penalty.unwrap_or(1.1).max(1.25));
                        final_cfg.repeat_last_n =
                            Some(final_cfg.repeat_last_n.unwrap_or(64).max(256));

                        let mut final_messages = history.clone();
                        final_messages.push(Message {
                            role: "user".to_string(),
                            content: "Donne maintenant UNIQUEMENT la reponse finale (pas la chaine de pensee). Reponds en francais, long, structure (titres + listes), avec exemples concrets.".to_string(),
                        });

                        let final_client = ChatClient::new(final_cfg)?;
                        let final_resp = final_client
                            .oneshot_ollama_with_messages(final_messages)
                            .await
                            .context("Finalizer pass echoue")?;

                        if !final_resp.content.trim().is_empty() {
                            let _ = tx.send(WorkerEvent::Chunk(final_resp.content));
                            streamed_any_content = true;
                        }
                    }

                    if !streamed_any_content {
                        let _ = tx.send(WorkerEvent::Chunk(
                            "Le modele n'a pas produit de reponse finale. Essaie think=medium et/ou augmente repeat_penalty."
                                .to_string(),
                        ));
                    }

                    Ok::<(), anyhow::Error>(())
                }),
                Err(err) => Err(anyhow!("Impossible de creer le runtime async: {err}")),
            };

            match result {
                Ok(()) => {
                    let _ = tx.send(WorkerEvent::Done);
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(format_anyhow_chain(&err)));
                }
            }
        });
    }

    fn drain_worker_events(&mut self) {
        let mut disconnected = false;

        loop {
            let event = match self.worker_rx.as_ref() {
                Some(rx) => rx.try_recv(),
                None => break,
            };

            match event {
                Ok(WorkerEvent::Chunk(chunk)) => self.append_assistant_chunk(&chunk),
                Ok(WorkerEvent::Thinking(thinking)) => self.append_assistant_thinking(&thinking),
                Ok(WorkerEvent::Done) => {
                    self.pending = false;
                    self.status = "Pret".to_string();
                    self.pending_assistant_index = None;
                    self.worker_rx = None;
                    break;
                }
                Ok(WorkerEvent::Error(err)) => {
                    self.pending = false;
                    self.status = if is_connection_refused_error(&err) {
                        "Serveur hors ligne".to_string()
                    } else if is_model_memory_error(&err) {
                        "Memoire insuffisante".to_string()
                    } else {
                        "Erreur".to_string()
                    };

                    if let Some(index) = self.pending_assistant_index {
                        if let Some(line) = self.messages.get_mut(index) {
                            if line.text.trim().is_empty() {
                                line.text = format_worker_error(&err, &self.host);
                            } else {
                                line.text.push_str("\n\n");
                                line.text.push_str(&format_worker_error(&err, &self.host));
                            }
                            line.include_in_context = false;
                        } else {
                            self.messages.push(ChatLine {
                                role: ChatRole::Assistant,
                                text: format_worker_error(&err, &self.host),
                                thinking: String::new(),
                                include_in_context: false,
                            });
                        }
                    } else {
                        self.messages.push(ChatLine {
                            role: ChatRole::Assistant,
                            text: format_worker_error(&err, &self.host),
                            thinking: String::new(),
                            include_in_context: false,
                        });
                    }

                    self.pending_assistant_index = None;
                    self.worker_rx = None;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if disconnected {
            self.pending = false;
            self.status = "Pret".to_string();
            self.pending_assistant_index = None;
            self.worker_rx = None;
        }
    }

    fn append_assistant_chunk(&mut self, chunk: &str) {
        if let Some(index) = self.pending_assistant_index {
            if let Some(line) = self.messages.get_mut(index) {
                line.text.push_str(chunk);
            }
        }
    }

    fn append_assistant_thinking(&mut self, thinking: &str) {
        if let Some(index) = self.pending_assistant_index {
            if let Some(line) = self.messages.get_mut(index) {
                line.thinking.push_str(thinking);
            }
        }
    }

    fn reset_chat(&mut self) {
        self.messages.clear();
        self.messages.push(ChatLine {
            role: ChatRole::Assistant,
            text: "Nouvelle conversation demarree.".to_string(),
            thinking: String::new(),
            include_in_context: false,
        });
        self.status = "Pret".to_string();
        self.source_count_last_query = 0;
    }

    fn build_context_messages(&self, new_user_prompt: &str, system_prompt: &str) -> Vec<Message> {
        const MAX_CONTEXT_LINES: usize = 24;

        let mut context_messages = vec![Message {
            role: "system".to_string(),
            content: system_prompt.to_string(),
        }];

        let contextual_lines: Vec<&ChatLine> = self
            .messages
            .iter()
            .filter(|line| line.include_in_context && !line.text.trim().is_empty())
            .collect();

        let start_index = contextual_lines.len().saturating_sub(MAX_CONTEXT_LINES);
        for line in contextual_lines.into_iter().skip(start_index) {
            let role = match line.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            context_messages.push(Message {
                role: role.to_string(),
                content: line.text.clone(),
            });
        }

        context_messages.push(Message {
            role: "user".to_string(),
            content: new_user_prompt.to_string(),
        });

        context_messages
    }

    fn render_message(ui: &mut egui::Ui, line: &ChatLine) -> egui::Rect {
        let (label, fill, is_user, title_fill) = match line.role {
            ChatRole::User => ("Vous", USER_BUBBLE, true, Color32::from_rgb(0, 72, 124)),
            ChatRole::Assistant => ("JARVIS V1", ASSISTANT_BUBBLE, false, WIN95_TITLE_BLUE),
        };
        let max_bubble_width = (ui.available_width() * 0.82).clamp(220.0, 760.0);
        let row_layout = if is_user {
            Layout::right_to_left(Align::TOP)
        } else {
            Layout::left_to_right(Align::TOP)
        };

        let mut bubble_rect = egui::Rect::NOTHING;
        ui.horizontal(|ui| {
            ui.set_width(ui.available_width());
            ui.with_layout(row_layout, |ui| {
                bubble_rect = ui
                    .scope(|ui| {
                        ui.set_max_width(max_bubble_width);
                        raised_panel(ui, fill, 8, |ui| {
                            sunken_panel(ui, title_fill, 4, |ui| {
                                ui.label(RichText::new(label).strong().color(WIN95_HIGHLIGHT));
                            });
                            ui.add_space(6.0);
                            sunken_panel(ui, WIN95_INPUT_BG, 6, |ui| {
                                if line.text.is_empty() {
                                    ui.label(RichText::new("...").italics());
                                } else {
                                    ui.add(egui::Label::new(line.text.as_str()).wrap());
                                }
                            });
                        })
                        .response
                        .rect
                    })
                    .inner;
            });
        });

        bubble_rect
    }
}

impl eframe::App for Messenger95App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_worker_events();

        if ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl) {
            self.send_prompt();
        }

        if self.guide_kb.is_none() {
            self.use_factory_data = false;
        }
        let factory_mode_active = self.use_factory_data && self.guide_kb.is_some();

        egui::TopBottomPanel::top("window_chrome")
            .exact_height(56.0)
            .frame(
                Frame::default()
                    .fill(WIN95_FACE)
                    .inner_margin(egui::Margin::same(6)),
            )
            .show(ctx, |ui| {
                raised_panel(ui, WIN95_FACE, 4, |ui| {
                    let title_bar = Frame::default()
                        .fill(WIN95_TITLE_BLUE)
                        .inner_margin(egui::Margin::symmetric(8, 4))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("[*]").color(WIN95_HIGHLIGHT).strong());
                                ui.label(
                                    RichText::new("JARVIS V1")
                                        .strong()
                                        .color(WIN95_HIGHLIGHT),
                                );
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    ui.label(
                                        RichText::new(format!("Statut: {}", self.status))
                                            .small()
                                            .color(WIN95_HIGHLIGHT),
                                    );
                                });
                            });
                        });
                    paint_bevel(
                        ui,
                        title_bar.response.rect,
                        Color32::from_rgb(0, 0, 168),
                        Color32::from_rgb(0, 0, 72),
                        Color32::from_rgb(0, 0, 192),
                        Color32::from_rgb(0, 0, 56),
                    );
                });
            });

        egui::TopBottomPanel::bottom("input_bar")
            .resizable(false)
            .frame(
                Frame::default()
                    .fill(WIN95_FACE)
                    .inner_margin(egui::Margin::same(8)),
            )
            .show(ctx, |ui| {
                raised_panel(ui, WIN95_FACE, 8, |ui| {
                    sunken_panel(ui, WIN95_FACE, 6, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Message:").strong());

                            let send_width = 100.0;
                            let input_width = (ui.available_width() - send_width - 24.0).max(120.0);
                            let response = sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                ui.add_enabled_ui(!self.pending, |ui| {
                                    ui.add_sized(
                                        [input_width, 24.0],
                                        egui::TextEdit::singleline(&mut self.input)
                                            .hint_text("Ecris ton message ici..."),
                                    )
                                })
                                .inner
                            })
                            .inner;

                            let enter_pressed = response.lost_focus()
                                && ui.input(|i| i.key_pressed(egui::Key::Enter));

                            if ui
                                .add_enabled(
                                    !self.pending,
                                    Button::new("Envoyer")
                                        .min_size(egui::vec2(send_width, 24.0))
                                        .fill(WIN95_FACE)
                                        .stroke(Stroke::new(1.0, WIN95_SHADOW_DARK)),
                                )
                                .clicked()
                                || enter_pressed
                            {
                                self.send_prompt();
                            }
                        });

                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    !self.pending,
                                    Button::new("Nouveau chat")
                                        .min_size(egui::vec2(120.0, 24.0))
                                        .fill(WIN95_FACE)
                                        .stroke(Stroke::new(1.0, WIN95_SHADOW_DARK)),
                                )
                                .clicked()
                            {
                                self.reset_chat();
                            }
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(RichText::new("Ctrl+Entree pour envoyer").small());
                                if self.pending {
                                    ui.label(RichText::new("Generation...").strong());
                                }
                            });
                        });
                    });

                    ui.add_space(6.0);
                    sunken_panel(ui, WIN95_FACE, 4, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "Donnees USINE: {}",
                                    if factory_mode_active { "OUI" } else { "NON" }
                                ))
                                .small()
                                .strong(),
                            );
                            ui.separator();
                            ui.label(RichText::new(self.guide_info.as_str()).small());
                            if self.source_count_last_query > 0 {
                                ui.separator();
                                ui.label(
                                    RichText::new(format!(
                                        "{} source(s) pour la derniere reponse",
                                        self.source_count_last_query
                                    ))
                                    .small(),
                                );
                            }
                        });
                    });
                });
            });

        egui::CentralPanel::default()
            .frame(
                Frame::default()
                    .fill(WIN95_DESKTOP)
                    .inner_margin(egui::Margin::same(10)),
            )
            .show(ctx, |ui| {
                raised_panel(ui, WIN95_FACE, 8, |ui| {
                    sunken_panel(ui, WIN95_FACE, 6, |ui| {
                        ui.horizontal(|ui| {
                            let gpt_oss_selected = self.is_gpt_oss_selected();
                            let header_layout =
                                compute_header_layout(ui.available_width(), gpt_oss_selected);

                            ui.allocate_ui_with_layout(
                                egui::vec2(header_layout.controls_col_width, header_layout.row_height),
                                Layout::top_down(Align::LEFT),
                                |ui| {
                                    let host_width = if gpt_oss_selected {
                                        108.0
                                    } else {
                                        146.0
                                    };

                                    ui.horizontal_wrapped(|ui| {
                                        ui.label("Host:");
                                        sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                            ui.add_sized(
                                                [host_width, 20.0],
                                                egui::TextEdit::singleline(&mut self.host),
                                            );
                                        });
                                        ui.label("Modele:");
                                        sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                            egui::ComboBox::from_id_salt("model_selector")
                                                .width(118.0)
                                                .selected_text(self.model.as_str())
                                                .show_ui(ui, |ui| {
                                                    for option in MODEL_OPTIONS {
                                                        ui.selectable_value(
                                                            &mut self.model,
                                                            option.to_string(),
                                                            option,
                                                        );
                                                    }
                                                });
                                        });

                                        if gpt_oss_selected {
                                            ui.label("Raisonn.:");
                                            sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                                egui::ComboBox::from_id_salt(
                                                    "reasoning_level_selector",
                                                )
                                                .width(84.0)
                                                .selected_text(self.reasoning_level.as_ui_label())
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(
                                                        &mut self.reasoning_level,
                                                        ReasoningLevel::Low,
                                                        "Bas",
                                                    );
                                                    ui.selectable_value(
                                                        &mut self.reasoning_level,
                                                        ReasoningLevel::Medium,
                                                        "Moyen",
                                                    );
                                                    ui.selectable_value(
                                                        &mut self.reasoning_level,
                                                        ReasoningLevel::High,
                                                        "Haut",
                                                    );
                                                });
                                            });
                                        } else {
                                            ui.label("Max tokens:");
                                            sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                                ui.add_sized(
                                                    [88.0, 20.0],
                                                    egui::DragValue::new(&mut self.max_tokens)
                                                        .range(256..=16384)
                                                        .speed(64),
                                                );
                                            });
                                        }
                                    });

                                    if gpt_oss_selected {
                                        ui.add_space(2.0);
                                        ui.horizontal_wrapped(|ui| {
                                            ui.label("Max:");
                                            sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                                ui.add_sized(
                                                    [84.0, 20.0],
                                                    egui::DragValue::new(&mut self.max_tokens)
                                                        .range(512..=32768)
                                                        .speed(128),
                                                );
                                            });

                                            ui.label("Ctx:");
                                            sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                                ui.add_sized(
                                                    [72.0, 20.0],
                                                    egui::DragValue::new(&mut self.num_ctx)
                                                        .range(0..=131072)
                                                        .speed(512),
                                                );
                                            });
                                            ui.label(RichText::new("0=auto").small());

                                            ui.label("Penalty:");
                                            sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                                ui.add_sized(
                                                    [66.0, 20.0],
                                                    egui::DragValue::new(&mut self.repeat_penalty)
                                                        .range(1.05..=1.60)
                                                        .speed(0.01),
                                                );
                                            });

                                            ui.label("Last_n:");
                                            sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                                ui.add_sized(
                                                    [66.0, 20.0],
                                                    egui::DragValue::new(&mut self.repeat_last_n)
                                                        .range(0..=4096)
                                                        .speed(64),
                                                );
                                            });

                                            ui.checkbox(&mut self.use_streaming, "Streaming");
                                        });
                                    }
                                },
                            );

                            ui.allocate_ui_with_layout(
                                egui::vec2(header_layout.logo_col_width, header_layout.row_height),
                                Layout::right_to_left(Align::Center),
                                |ui| {
                                    render_logo(ui, header_layout.logo_size);
                                },
                            );
                        });
                        if self.is_gpt_oss_selected() {
                            ui.label(
                                RichText::new(
                                    "Conseil: think=high peut boucler. Augmente repeat_penalty (>=1.2) ou passe en moyen.",
                                )
                                .small(),
                            );
                        }
                        ui.horizontal(|ui| {
                            let has_factory_docs = self.guide_kb.is_some();
                            let changed = ui
                                .add_enabled(
                                    has_factory_docs,
                                    egui::Checkbox::new(
                                        &mut self.use_factory_data,
                                        "Donnees USINE",
                                    ),
                                )
                                .changed();
                            if changed {
                                self.source_count_last_query = 0;
                                self.status = if self.use_factory_data {
                                    "Mode usine".to_string()
                                } else {
                                    "Mode libre".to_string()
                                };
                            }

                            ui.label(
                                RichText::new(if factory_mode_active { "OUI" } else { "NON" })
                                    .strong(),
                            );
                            if !has_factory_docs {
                                ui.label(
                                    RichText::new(
                                        "guide-production-rochias.txt absent: mode usine indisponible.",
                                    )
                                    .small(),
                                );
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("System:").strong());
                            let changed = sunken_panel(ui, WIN95_INPUT_BG, 3, |ui| {
                                ui.add_sized(
                                    [ui.available_width(), 24.0],
                                    egui::TextEdit::singleline(&mut self.system),
                                )
                                .changed()
                            })
                            .inner;
                            if changed {
                                self.system_grounded = build_grounded_system_prompt(&self.system);
                            }
                        });
                        if factory_mode_active {
                            ui.label(
                                RichText::new(
                                    "Mode usine actif: reponse basee sur extraits + citations obligatoires.",
                                )
                                .color(WIN95_TEXT),
                            );
                        } else {
                            ui.label(
                                RichText::new(
                                    "Mode libre actif: conversation directe avec le modele, sans recherche documentaire.",
                                )
                                .color(WIN95_TEXT),
                            );
                        }
                    });
                });

                ui.add_space(8.0);

                raised_panel(ui, WIN95_FACE, 6, |ui| {
                    sunken_panel(ui, WIN95_WINDOW_BG, 4, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Conversation").strong());
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(
                                    RichText::new(format!("{} message(s)", self.messages.len()))
                                        .small(),
                                );
                            });
                        });
                    });
                    ui.add_space(6.0);
                    let show_thinking_panel = self.is_gpt_oss_selected();
                    let content_height = ui.available_height().max(220.0);
                    if show_thinking_panel {
                        let thinking_width = (ui.available_width() * 0.34).clamp(260.0, 420.0);
                        let chat_width = (ui.available_width() - thinking_width - 8.0).max(260.0);
                        ui.horizontal(|ui| {
                            ui.allocate_ui_with_layout(
                                egui::vec2(chat_width, content_height),
                                Layout::top_down(Align::LEFT),
                                |ui| {
                                    sunken_panel(ui, WIN95_INPUT_BG, 8, |ui| {
                                        ui.set_min_height((content_height - 16.0).max(160.0));
                                        ScrollArea::vertical()
                                            .id_salt("conversation_scroll_with_thinking")
                                            .stick_to_bottom(true)
                                            .auto_shrink([false, false])
                                            .show(ui, |ui| {
                                                for line in &self.messages {
                                                    Self::render_message(ui, line);
                                                    ui.add_space(6.0);
                                                }
                                            });
                                    });
                                },
                            );

                            ui.allocate_ui_with_layout(
                                egui::vec2(thinking_width, content_height),
                                Layout::top_down(Align::LEFT),
                                |ui| {
                                    sunken_panel(ui, WIN95_WINDOW_BG, 4, |ui| {
                                        ui.label(RichText::new("Chaine de pensee").strong());
                                        ui.label(
                                            RichText::new("Trace de raisonnement gpt-oss")
                                                .small()
                                                .italics(),
                                        );
                                    });
                                    ui.add_space(4.0);
                                    sunken_panel(ui, WIN95_INPUT_BG, 8, |ui| {
                                        ui.set_min_height((ui.available_height() - 16.0).max(160.0));
                                        ScrollArea::vertical()
                                            .id_salt("thinking_scroll_panel")
                                            .auto_shrink([false, false])
                                            .show(ui, |ui| {
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(self.current_thinking_text())
                                                            .monospace()
                                                            .size(13.0),
                                                    )
                                                    .wrap(),
                                                );
                                            });
                                    });
                                },
                            );
                        });
                    } else {
                        ui.allocate_ui_with_layout(
                            egui::vec2(ui.available_width(), content_height),
                            Layout::top_down(Align::LEFT),
                            |ui| {
                                sunken_panel(ui, WIN95_INPUT_BG, 8, |ui| {
                                    ui.set_min_height((content_height - 16.0).max(160.0));
                                    ScrollArea::vertical()
                                        .id_salt("conversation_scroll_single")
                                        .stick_to_bottom(true)
                                        .auto_shrink([false, false])
                                        .show(ui, |ui| {
                                            for line in &self.messages {
                                                Self::render_message(ui, line);
                                                ui.add_space(6.0);
                                            }
                                        });
                                });
                            },
                        );
                    }
                });
            });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const LOGO_HIGH_RES_MIN_SCALE: f32 = 3.0;

    fn test_app_with_messages(messages: Vec<ChatLine>) -> Messenger95App {
        Messenger95App {
            host: "http://localhost:11434".to_string(),
            model: "qwen2.5-coder:14b".to_string(),
            reasoning_level: ReasoningLevel::Medium,
            system: "system".to_string(),
            system_grounded: "system-grounded".to_string(),
            temperature: 0.2,
            max_tokens: 1024,
            num_ctx: 0,
            repeat_penalty: 1.20,
            repeat_last_n: 256,
            use_streaming: true,
            timeout_seconds: 600,
            use_factory_data: false,
            input: String::new(),
            status: "Pret".to_string(),
            source_count_last_query: 0,
            pending: false,
            messages,
            pending_assistant_index: None,
            worker_rx: None,
            guide_kb: None,
            guide_info: String::new(),
        }
    }

    fn render_rect_for_line(line: ChatLine, available_width: f32) -> egui::Rect {
        let ctx = egui::Context::default();
        let mut rendered_rect = None;

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(available_width, 400.0),
                    Layout::top_down(Align::LEFT),
                    |ui| {
                        rendered_rect = Some(Messenger95App::render_message(ui, &line));
                    },
                );
            });
        });

        rendered_rect.expect("message should be rendered")
    }

    fn parse_png_dimensions(bytes: &[u8]) -> (u32, u32) {
        const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
        assert!(
            bytes.len() >= 24,
            "png should contain signature + IHDR dimensions"
        );
        assert_eq!(
            &bytes[0..8],
            PNG_SIGNATURE.as_slice(),
            "unexpected png signature"
        );
        assert_eq!(&bytes[12..16], b"IHDR", "first png chunk must be IHDR");

        let width = u32::from_be_bytes(
            bytes[16..20]
                .try_into()
                .expect("png width should be encoded on 4 bytes"),
        );
        let height = u32::from_be_bytes(
            bytes[20..24]
                .try_into()
                .expect("png height should be encoded on 4 bytes"),
        );
        (width, height)
    }

    #[test]
    fn model_options_include_qwen_coder_14b() {
        assert_eq!(MODEL_OPTIONS.len(), 2);
        assert!(MODEL_OPTIONS.contains(&"qwen2.5-coder:14b"));
        assert!(MODEL_OPTIONS.contains(&GPT_OSS_MODEL));
    }

    #[test]
    fn app_new_keeps_generation_limits_from_config() {
        let config = ChatConfig {
            model: "qwen2.5-coder:14b".to_string(),
            temperature: 0.45,
            max_tokens: 1536,
            num_ctx: Some(4096),
            repeat_penalty: Some(1.28),
            repeat_last_n: Some(384),
            timeout_seconds: 321,
            ..ChatConfig::default()
        };

        let app = Messenger95App::new(config);

        assert_eq!(app.temperature, 0.45);
        assert_eq!(app.max_tokens, 1536);
        assert_eq!(app.num_ctx, 4096);
        assert_eq!(app.repeat_penalty, 1.28);
        assert_eq!(app.repeat_last_n, 384);
        assert_eq!(app.timeout_seconds, 321);
    }

    #[test]
    fn compute_budget_for_qwen_keeps_max_tokens_and_auto_ctx() {
        let app = Messenger95App::new(ChatConfig {
            model: "qwen2.5-coder:14b".to_string(),
            max_tokens: 1337,
            num_ctx: Some(9999),
            ..ChatConfig::default()
        });

        let (predict, ctx) = app.compute_budget();
        assert_eq!(predict, 1337);
        assert_eq!(ctx, None);
    }

    #[test]
    fn compute_budget_for_gpt_oss_enforces_minimums_from_reasoning_level() {
        let mut app = Messenger95App::new(ChatConfig {
            model: GPT_OSS_MODEL.to_string(),
            max_tokens: 1024,
            num_ctx: None,
            ..ChatConfig::default()
        });
        app.reasoning_level = ReasoningLevel::High;

        let (predict, ctx) = app.compute_budget();
        assert_eq!(predict, 8192);
        assert_eq!(ctx, Some(32768));
    }

    #[test]
    fn compute_budget_for_gpt_oss_respects_ctx_override_with_headroom() {
        let mut app = Messenger95App::new(ChatConfig {
            model: GPT_OSS_MODEL.to_string(),
            max_tokens: 5000,
            num_ctx: Some(5200),
            ..ChatConfig::default()
        });
        app.reasoning_level = ReasoningLevel::Medium;

        let (predict, ctx) = app.compute_budget();
        assert_eq!(predict, 5000);
        assert_eq!(ctx, Some(16384));
    }

    #[test]
    fn build_context_messages_skips_blank_and_non_context_lines() {
        let app = test_app_with_messages(vec![
            ChatLine {
                role: ChatRole::Assistant,
                text: "intro".to_string(),
                thinking: String::new(),
                include_in_context: false,
            },
            ChatLine {
                role: ChatRole::User,
                text: "   ".to_string(),
                thinking: String::new(),
                include_in_context: true,
            },
            ChatLine {
                role: ChatRole::User,
                text: "Question 1".to_string(),
                thinking: String::new(),
                include_in_context: true,
            },
            ChatLine {
                role: ChatRole::Assistant,
                text: "Reponse 1".to_string(),
                thinking: String::new(),
                include_in_context: true,
            },
        ]);

        let context = app.build_context_messages("Question 2", "SYSTEM");
        assert_eq!(context.len(), 4);
        assert_eq!(context[0].role, "system");
        assert_eq!(context[0].content, "SYSTEM");
        assert_eq!(context[1].role, "user");
        assert_eq!(context[1].content, "Question 1");
        assert_eq!(context[2].role, "assistant");
        assert_eq!(context[2].content, "Reponse 1");
        assert_eq!(context[3].role, "user");
        assert_eq!(context[3].content, "Question 2");
    }

    #[test]
    fn build_context_messages_keeps_only_recent_lines() {
        let mut lines = Vec::new();
        for i in 0..30 {
            lines.push(ChatLine {
                role: if i % 2 == 0 {
                    ChatRole::User
                } else {
                    ChatRole::Assistant
                },
                text: format!("line-{i}"),
                thinking: String::new(),
                include_in_context: true,
            });
        }
        let app = test_app_with_messages(lines);

        let context = app.build_context_messages("final-question", "SYSTEM");

        assert_eq!(context.len(), 26);
        assert_eq!(context[0].role, "system");
        assert_eq!(context[1].content, "line-6");
        assert_eq!(context[24].content, "line-29");
        assert_eq!(context[25].role, "user");
        assert_eq!(context[25].content, "final-question");
    }

    #[test]
    fn current_thinking_text_prefers_latest_assistant_trace() {
        let app = test_app_with_messages(vec![
            ChatLine {
                role: ChatRole::User,
                text: "Q1".to_string(),
                thinking: String::new(),
                include_in_context: true,
            },
            ChatLine {
                role: ChatRole::Assistant,
                text: "R1".to_string(),
                thinking: "Trace 1".to_string(),
                include_in_context: true,
            },
            ChatLine {
                role: ChatRole::Assistant,
                text: "R2".to_string(),
                thinking: "Trace 2".to_string(),
                include_in_context: true,
            },
        ]);

        assert_eq!(app.current_thinking_text(), "Trace 2");
    }

    #[test]
    fn header_layout_stays_compact_and_within_available_width() {
        let layout_gpt = compute_header_layout(620.0, true);
        let layout_qwen = compute_header_layout(620.0, false);

        assert!(
            layout_gpt.controls_col_width + layout_gpt.logo_col_width + 6.0 <= 621.0,
            "header width overflow in gpt mode"
        );
        assert!(
            layout_qwen.controls_col_width + layout_qwen.logo_col_width + 6.0 <= 621.0,
            "header width overflow in qwen mode"
        );
        assert!(
            layout_gpt.row_height > layout_qwen.row_height,
            "gpt mode should reserve more height for extra controls"
        );
        assert!(
            layout_gpt.row_height <= 140.0,
            "gpt controls should stay compact"
        );
    }

    #[test]
    fn logo_png_is_high_resolution_for_display_size() {
        let (width, height) = parse_png_dimensions(include_bytes!("../images/logo1.png"));
        let min_required = (LOGO_SIZE_MAX * LOGO_HIGH_RES_MIN_SCALE) as u32;
        assert_eq!(
            width, height,
            "logo should stay square for predictable rendering"
        );
        assert!(
            width >= min_required && height >= min_required,
            "logo source {}x{} should be at least {}x{}",
            width,
            height,
            min_required,
            min_required
        );
    }

    #[test]
    fn logo_renders_large_and_right_aligned() {
        let ctx = egui::Context::default();
        let layout = compute_header_layout(920.0, true);
        let mut logo_rect = None;
        let mut column_rect = None;

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let response = ui.allocate_ui_with_layout(
                    egui::vec2(layout.logo_col_width, layout.row_height),
                    Layout::right_to_left(Align::Center),
                    |ui| render_logo(ui, layout.logo_size).rect,
                );
                column_rect = Some(response.response.rect);
                logo_rect = Some(response.inner);
            });
        });

        let logo_rect = logo_rect.expect("logo should be rendered");
        let column_rect = column_rect.expect("logo column should be rendered");

        assert!(
            (logo_rect.width() - layout.logo_size).abs() <= 1.0,
            "logo width {} should be close to {}",
            logo_rect.width(),
            layout.logo_size
        );
        assert!(
            (logo_rect.height() - layout.logo_size).abs() <= 1.0,
            "logo height {} should be close to {}",
            logo_rect.height(),
            layout.logo_size
        );
        assert!(
            logo_rect.right() >= column_rect.right() - 2.0,
            "logo should be right-aligned (logo right={}, column right={})",
            logo_rect.right(),
            column_rect.right()
        );
    }

    #[test]
    fn render_message_long_text_stays_within_expected_width() {
        let line = ChatLine {
            role: ChatRole::Assistant,
            text: "mot tres long ".repeat(120),
            thinking: String::new(),
            include_in_context: true,
        };
        let available_width = 420.0;
        let expected_max_width = (available_width * 0.82f32).clamp(220.0, 760.0);

        let rect = render_rect_for_line(line, available_width);

        assert!(
            rect.width() <= expected_max_width + 1.0,
            "bubble width {} exceeded max {}",
            rect.width(),
            expected_max_width
        );
    }

    #[test]
    fn render_message_supports_long_user_text_without_overflowing_row() {
        let line = ChatLine {
            role: ChatRole::User,
            text: "ceci est un message utilisateur assez long ".repeat(80),
            thinking: String::new(),
            include_in_context: true,
        };
        let available_width = 500.0;
        let row_rect =
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(available_width, 400.0));

        let rect = render_rect_for_line(line, available_width);

        assert!(
            rect.width() <= row_rect.width() + 1.0,
            "bubble width {} exceeded row width {}",
            rect.width(),
            row_rect.width()
        );
    }
}
