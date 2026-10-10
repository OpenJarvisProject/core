# ojp-core

The reference OpenJarvisProject runtime. Rust application layer, with
`llama.cpp` (via the [`llama-crab`](https://crates.io/crates/llama-crab)
binding) as the inference engine — see the
[`docs`](https://github.com/OpenJarvisProject/docs) repo's
`architecture/runtime.md` for the full language-split reasoning, and
`architecture/runtime.md`'s validation methodology for why this starts
CPU-only.

## Current state

Stage 1 of the validation plan only: a CPU-only smoke test proving the
Rust↔`llama.cpp` pipeline (model loading, tokenization, generation
loop) works at all, with zero GPU variables involved. Not yet an agent,
not yet a daemon, not yet anything from the OJPs — just the foundation
everything else sits on.

## Running the smoke test

```bash
mkdir -p models
curl -sL -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
cargo run --release
```

Model files are gitignored — this is just a stock, unmodified,
pre-quantized test model, not anything produced by this project's own
compression pipeline.

## Next steps (not yet done)

- Stage 2: GPU backend, Vulkan first (see the ROCm/RX580 caveat in the
  `docs` repo's `model-compression.md`)
- Stage 3: swap in a project-produced pruned/quantized model on the
  same harness
- aarch64 Linux + Android cross-compilation targets
