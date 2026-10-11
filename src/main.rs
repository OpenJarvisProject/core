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

    // For investigating the fixed compute-buffer VRAM cost (doesn't
    // shrink with pruning - see model-tooling's README). n_ctx default
    // unchanged from the original hardcoded value; n_ubatch, when set,
    // overrides llama.cpp's own default (observed as 512 from
    // sched_reserve's "worst-case: n_tokens = 512" log line).
    let n_ctx: u32 = std::env::var("OJP_N_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let n_ubatch: Option<u32> = std::env::var("OJP_N_UBATCH").ok().and_then(|v| v.parse().ok());

    println!(
        "Loading model: {model_path} (n_gpu_layers={gpu_layers}, n_ctx={n_ctx}, n_ubatch={n_ubatch:?})"
    );
    let mut params = LlamaParams::new(&model_path)
        .with_n_ctx(n_ctx)
        .with_n_gpu_layers(gpu_layers);
    if let Some(n) = n_ubatch {
        params = params.with_n_ubatch(n);
    }
    let mut llama = Llama::load(params)?;
    println!("Model loaded. Running completion...");

    let response = llama.create_completion("The capital of France is", 32)?;
    println!("--- completion ---\n{}\n------------------", response.text);

    Ok(())
}
