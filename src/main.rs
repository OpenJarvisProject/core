//! Stage 1 validation smoke test (see docs repo's architecture/runtime.md):
//! CPU-only, stock unmodified model, no GPU backend involved yet.
use llama_crab::{Llama, LlamaParams};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string());

    println!("Loading model: {model_path}");
    let mut llama = Llama::load(
        LlamaParams::new(&model_path)
            .with_n_ctx(2048)
            .with_n_gpu_layers(0), // force CPU-only for this stage
    )?;
    println!("Model loaded. Running completion...");

    let response = llama.create_completion("The capital of France is", 32)?;
    println!("--- completion ---\n{}\n------------------", response.text);

    Ok(())
}
