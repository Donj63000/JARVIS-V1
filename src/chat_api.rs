use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ChatConfig {
    pub host: String,
    pub model: String,
    pub system: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
    pub reasoning_effort: Option<String>,
    pub num_ctx: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub seed: Option<i32>,
    pub stop: Vec<String>,
    pub keep_alive: Option<String>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            model: "qwen2.5-coder:14b".to_string(),
            system: "Tu es JARVIS V1, un assistant IA specialise en informatique".to_string(),
            temperature: 0.2,
            max_tokens: 1024,
            timeout_seconds: 600,
            reasoning_effort: None,
            num_ctx: None,
            repeat_penalty: None,
            repeat_last_n: None,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: None,
            stop: Vec::new(),
            keep_alive: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AssistantDelta {
    pub content: String,
    pub thinking: String,
}

#[derive(Clone, Debug, Default)]
pub struct AssistantResponse {
    pub content: String,
    pub thinking: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct ChatClient {
    client: reqwest::Client,
    config: ChatConfig,
}

impl ChatClient {
    pub fn new(config: ChatConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .context("Impossible de construire le client HTTP")?;

        Ok(Self { client, config })
    }

    pub async fn oneshot(&self, prompt: &str) -> Result<String> {
        self.oneshot_with_messages(self.build_messages(prompt))
            .await
    }

    pub async fn oneshot_with_messages(&self, messages: Vec<Message>) -> Result<String> {
        let req = OpenAIChatCompletionsRequest {
            model: self.config.model.clone(),
            messages,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            stream: false,
        };

        let resp = self
            .client
            .post(self.url("/v1/chat/completions"))
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, "Bearer ollama")
            .json(&req)
            .send()
            .await
            .context("Echec de requete HTTP")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Erreur HTTP {status}: {body}"));
        }

        let out: OpenAIChatCompletionsResponse = resp
            .json()
            .await
            .context("Reponse JSON illisible (format inattendu)")?;

        Ok(extract_openai_text(&out))
    }

    pub async fn oneshot_ollama_with_messages(
        &self,
        messages: Vec<Message>,
    ) -> Result<AssistantResponse> {
        let req = OllamaChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: false,
            think: self.config.reasoning_effort.clone(),
            keep_alive: self.config.keep_alive.clone(),
            options: OllamaChatOptions {
                num_predict: self.config.max_tokens,
                temperature: self.config.temperature,
                num_ctx: self.config.num_ctx,
                repeat_penalty: self.config.repeat_penalty,
                repeat_last_n: self.config.repeat_last_n,
                top_k: self.config.top_k,
                top_p: self.config.top_p,
                min_p: self.config.min_p,
                seed: self.config.seed,
                stop: self.config.stop.clone(),
            },
        };

        let resp = self
            .client
            .post(self.url("/api/chat"))
            .header(CONTENT_TYPE, "application/json")
            .json(&req)
            .send()
            .await
            .context("Echec de requete HTTP")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Erreur HTTP {status}: {body}"));
        }

        let parsed: OllamaChatChunk = resp
            .json()
            .await
            .context("Reponse JSON illisible (format inattendu)")?;

        let message = parsed.message.unwrap_or_default();
        Ok(AssistantResponse {
            content: message.content,
            thinking: message.thinking,
        })
    }

    pub async fn stream<F>(&self, prompt: &str, mut on_chunk: F) -> Result<()>
    where
        F: FnMut(&str) + Send,
    {
        self.stream_with_messages(self.build_messages(prompt), move |chunk| on_chunk(chunk))
            .await
    }

    pub async fn stream_with_messages<F>(
        &self,
        messages: Vec<Message>,
        mut on_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(&str) + Send,
    {
        self.stream_with_messages_detailed(messages, move |delta| {
            if !delta.content.is_empty() {
                on_chunk(&delta.content);
            }
        })
        .await
    }

    pub async fn stream_with_messages_detailed<F>(
        &self,
        messages: Vec<Message>,
        mut on_delta: F,
    ) -> Result<()>
    where
        F: FnMut(AssistantDelta) + Send,
    {
        let req = OllamaChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: true,
            think: self.config.reasoning_effort.clone(),
            keep_alive: self.config.keep_alive.clone(),
            options: OllamaChatOptions {
                num_predict: self.config.max_tokens,
                temperature: self.config.temperature,
                num_ctx: self.config.num_ctx,
                repeat_penalty: self.config.repeat_penalty,
                repeat_last_n: self.config.repeat_last_n,
                top_k: self.config.top_k,
                top_p: self.config.top_p,
                min_p: self.config.min_p,
                seed: self.config.seed,
                stop: self.config.stop.clone(),
            },
        };

        let resp = self
            .client
            .post(self.url("/api/chat"))
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/x-ndjson")
            .json(&req)
            .send()
            .await
            .context("Echec de requete HTTP")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Erreur HTTP {status}: {body}"));
        }

        let mut http_stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();

        while let Some(item) = http_stream.next().await {
            let chunk = item.context("Erreur de lecture du flux HTTP")?;
            buffer.extend_from_slice(&chunk);

            let lines = drain_ndjson_lines(&mut buffer);
            for line in lines {
                let parsed = parse_ollama_chunk(&line)?;

                if let Some(msg) = parsed.message {
                    if !msg.content.is_empty() || !msg.thinking.is_empty() {
                        on_delta(AssistantDelta {
                            content: msg.content,
                            thinking: msg.thinking,
                        });
                    }
                }

                if parsed.done {
                    return Ok(());
                }
            }
        }

        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            let line = line.trim();

            if !line.is_empty() {
                let parsed = parse_ollama_chunk(line)?;
                if let Some(msg) = parsed.message {
                    if !msg.content.is_empty() || !msg.thinking.is_empty() {
                        on_delta(AssistantDelta {
                            content: msg.content,
                            thinking: msg.thinking,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    fn build_messages(&self, prompt: &str) -> Vec<Message> {
        vec![
            Message {
                role: "system".to_string(),
                content: self.config.system.clone(),
            },
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ]
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.config.host.trim_end_matches('/'), path)
    }
}

#[derive(Serialize, Debug)]
struct OpenAIChatCompletionsRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct OpenAIChatCompletionsResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize, Debug)]
struct AssistantMessage {
    content: Option<String>,
}

#[derive(Serialize, Debug)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    options: OllamaChatOptions,
}

#[derive(Serialize, Debug)]
struct OllamaChatOptions {
    num_predict: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct OllamaChatChunk {
    pub message: Option<OllamaMessage>,
    #[serde(default)]
    pub done: bool,
}

#[derive(Deserialize, Debug, Default)]
pub struct OllamaMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub thinking: String,
}

fn extract_openai_text(response: &OpenAIChatCompletionsResponse) -> String {
    response
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .unwrap_or_else(|| "<reponse vide>".to_string())
}

pub fn drain_ndjson_lines(buffer: &mut Vec<u8>) -> Vec<String> {
    let mut lines = Vec::new();

    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
        let bytes: Vec<u8> = buffer.drain(..=pos).collect();
        let line = String::from_utf8_lossy(&bytes);
        let line = line.trim();
        if !line.is_empty() {
            lines.push(line.to_string());
        }
    }

    lines
}

pub fn parse_ollama_chunk(line: &str) -> Result<OllamaChatChunk> {
    serde_json::from_str(line)
        .with_context(|| format!("Impossible de parser la ligne NDJSON: {line}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_ndjson_lines_keeps_partial_tail() {
        let mut buffer = b"{\"done\":false}\n{\"done\":true}".to_vec();

        let lines = drain_ndjson_lines(&mut buffer);

        assert_eq!(lines, vec![r#"{"done":false}"#]);
        assert_eq!(buffer, b"{\"done\":true}".to_vec());
    }

    #[test]
    fn parse_ollama_chunk_parses_done() {
        let chunk = parse_ollama_chunk(r#"{"message":{"content":"Hi"},"done":false}"#).unwrap();

        assert!(!chunk.done);
        assert_eq!(chunk.message.unwrap().content, "Hi");
    }

    #[test]
    fn parse_ollama_chunk_parses_thinking() {
        let chunk =
            parse_ollama_chunk(r#"{"message":{"content":"OK","thinking":"trace"},"done":false}"#)
                .unwrap();

        let msg = chunk.message.unwrap();
        assert_eq!(msg.content, "OK");
        assert_eq!(msg.thinking, "trace");
    }

    #[test]
    fn extract_openai_text_handles_empty() {
        let response = OpenAIChatCompletionsResponse { choices: vec![] };
        assert_eq!(extract_openai_text(&response), "<reponse vide>");
    }
}
