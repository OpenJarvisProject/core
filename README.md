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
cargo build --release
./target/release/ojp-core                      # Stage 1: CPU only
OJP_GPU_LAYERS=99 ./target/release/ojp-core     # Stage 2: Vulkan GPU
```

Model files are gitignored — this is just a stock, unmodified,
pre-quantized test model, not anything produced by this project's own
compression pipeline.

## Build notes (real gotchas hit getting this running, not hypothetical)

- **Broken default Vulkan driver**: if `vulkaninfo` fails or picks a
  non-functional driver, check `/usr/share/vulkan/icd.d/` for multiple
  installed ICDs (e.g. a proprietary AMDVLK alongside Mesa's RADV) and
  force the working one, e.g.
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`.
- **Missing `SPIRV-Headers`**: `llama.cpp`'s Vulkan backend
  (`ggml-vulkan`) requires the `SPIRV-Headers` CMake package
  (`spirv/unified1/spirv.hpp`), which some distro Vulkan packages don't
  ship. If you don't have (or don't want) root to install it
  system-wide, build it to a local prefix instead — it's header-only,
  no privileged install actually required:
  ```bash
  git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git
  cmake -S SPIRV-Headers -B spirv-headers-build -DCMAKE_INSTALL_PREFIX=/path/to/.deps
  cmake --build spirv-headers-build --target install
  ```
  Then build with both `CMAKE_PREFIX_PATH=/path/to/.deps` (so CMake's
  `find_package(SPIRV-Headers)` succeeds) **and**
  `CPLUS_INCLUDE_PATH=/path/to/.deps/include` (because upstream
  `ggml-vulkan`'s `CMakeLists.txt` finds the package but never actually
  wires its include path into the `ggml-vulkan` target — it assumes a
  full Vulkan SDK install bundles these headers already, which isn't
  true on a plain Manjaro/Arch `vulkan-headers` install. The env var
  fixes it at the compiler level regardless of that CMake gap).
- Combine all of the above into one build command, e.g.:
  ```bash
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  CMAKE_PREFIX_PATH=/path/to/.deps \
  CPLUS_INCLUDE_PATH=/path/to/.deps/include \
  cargo build --release
  ```

## Next steps (not yet done)

- Stage 3: swap in a project-produced pruned/quantized model on the
  same harness
- aarch64 Linux + Android cross-compilation targets
- Distributed/multi-device layer-split inference for the large-brain/
  overnight tier (see `llama.cpp`'s RPC backend) — not relevant to the
  interactive-speed tiers, only the big/slow one
