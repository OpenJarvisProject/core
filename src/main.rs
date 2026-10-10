//! Stage 1 validation smoke test (see docs repo's architecture/runtime.md):
//! CPU-only, stock unmodified model, no GPU backend involved yet.
use llama_crab::{Llama, LlamaParams};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string());

    // Stage 1 default is 0 (CPU-only, no GPU variable). Stage 2 sets
    // OJP_GPU_LAYERS=99 to offload everything to the GPU backend,
    // same binary, one variable changed, per the validation plan.
    let gpu_layers: i32 = std::env::var("OJP_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    println!("Loading model: {model_path} (n_gpu_layers={gpu_layers})");
    let mut llama = Llama::load(
        LlamaParams::new(&model_path)
            .with_n_ctx(2048)
            .with_n_gpu_layers(gpu_layers),
    )?;
    println!("Model loaded. Running completion...");

    let response = llama.create_completion("The capital of France is", 32)?;
    println!("--- completion ---\n{}\n------------------", response.text);

    Ok(())
}
