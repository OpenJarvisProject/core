//! Minimal assistant loop: route each request, then dispatch it (OJP-9).
//!
//!   answer_directly -> the router model answers itself (already loaded)
//!   coding/reasoning -> the "main" backend: a second in-process GGUF, or
//!                       any OpenAI-compatible server (Ollama, llama-server,
//!                       another OJP device), or the router model if neither
//!                       is configured
//!   device_control  -> the router model emits a grammar-constrained tool
//!                       call, executed by tools.rs
use std::io::Write;

use anyhow::{Context, Result};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::high_level::completion::SamplingOptions;
use llama_crab::{ChatMessage, CompletionOptions, Llama, Role, StreamControl};
use serde_json::{Value, json};

use crate::router::{Route, Router};
use crate::tools;

const MAX_TOKENS: usize = 512;
const SYSTEM_PROMPT: &str = "You are a helpful, concise personal assistant running locally on the user's device.";

pub enum Backend {
    Local(Llama),
    Remote { base_url: String, model: String },
}

impl Backend {
    fn name(&self) -> String {
        match self {
            Backend::Local(_) => "main (in-process)".into(),
            Backend::Remote { base_url, model } => format!("{model} @ {base_url}"),
        }
    }
}

fn options() -> CompletionOptions {
    CompletionOptions { sampling: SamplingOptions::chat(), ..CompletionOptions::new(MAX_TOKENS) }
}

/// Stream a reply from an in-process model to stdout.
fn generate_local(llama: &mut Llama, messages: &[ChatMessage]) -> Result<String> {
    let reply = llama.create_chat_completion_stream_with(messages, BuiltinTemplate::ChatMl, &[], options(), |chunk| {
        print!("{}", chunk.text);
        let _ = std::io::stdout().flush();
        StreamControl::Continue
    })?;
    println!();
    Ok(reply.content)
}

/// Non-streaming OpenAI-compatible chat completion.
fn generate_remote(base_url: &str, model: &str, messages: &[ChatMessage]) -> Result<String> {
    let body = json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect::<Vec<_>>(),
    });
    let reply: Value = ureq::post(format!("{}/v1/chat/completions", base_url.trim_end_matches('/')))
        .send_json(&body)?
        .body_mut()
        .read_json()?;
    let text = reply["choices"][0]["message"]["content"]
        .as_str()
        .context("no choices[0].message.content in reply")?
        .to_string();
    println!("{text}");
    Ok(text)
}

pub fn run(mut router: Router, mut main: Option<Backend>) -> Result<()> {
    let main_name = main.as_ref().map_or("router model (no main configured)".into(), Backend::name);
    eprintln!("ready. coding/reasoning -> {main_name}. Empty line or Ctrl-D quits.");

    // One shared history, so follow-ups keep context whichever model answers.
    let mut history = vec![ChatMessage::new(Role::System, SYSTEM_PROMPT)];
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        let request = line.trim();

        let decision = router.classify(request)?;
        eprintln!("[route: {} in {} ms]", decision.route.label(), decision.elapsed.as_millis());

        history.push(ChatMessage::new(Role::User, request));
        let reply = match (decision.route, main.as_mut()) {
            (Route::DeviceControl, _) => {
                let text = tools::handle(router.llama_mut(), request).unwrap_or_else(|e| format!("Couldn't do that: {e:#}"));
                println!("{text}");
                text
            }
            (Route::Coding | Route::Reasoning, Some(Backend::Local(llama))) => generate_local(llama, &history)?,
            (Route::Coding | Route::Reasoning, Some(Backend::Remote { base_url, model })) => {
                generate_remote(base_url, model, &history)?
            }
            _ => generate_local(router.llama_mut(), &history)?,
        };
        history.push(ChatMessage::new(Role::Assistant, reply));
    }
}
