# mini_ai C++20

CPU-only educational autoregressive byte language model, implemented with the C++20 standard library. The Python implementation has been retired on this rewrite branch.

## Build and test
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Train and generate
`mini_ai_train [text-file] [checkpoint]` trains 100 batches and saves a MAI3 checkpoint. Re-running resumes model weights, Adam first/second moments, optimizer step, and model step. `mini_ai_test [checkpoint] [prompt]` runs the complete causal attention + ReLU feed-forward forward pass and samples bytes.

The model has learned byte/position embeddings, one causal single-head self-attention block, residual MLP, softmax cross-entropy, manually derived gradients for Q/K/V (including softmax and causal masking), and Adam.

Checkpoint files are binary and configuration-checked. This remains intentionally small: one head/block, CPU only, fixed context, and deterministic sampling seed.
