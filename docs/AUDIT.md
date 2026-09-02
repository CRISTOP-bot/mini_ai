# Correctness audit (cpp20-rewrite)

## Repairs

- `src/mini_ai.cpp`: training Q/K/V now start at zero (the previous implementation accidentally added the residual input to each projection). The causal attention backward pass explicitly computes `dV=A^T dZ`, `dA=dZ V^T`, `dS=A*(dA-sum(A*dA))`, and `dQ=dS K/sqrt(d)`, `dK=dS^T Q/sqrt(d)`. Gradients from every target and every batch item accumulate before one Adam update.
- `src/mini_ai.cpp`: loss and gradients are both normalized by `batch*seq`; Adam validates parameter/gradient shapes and checkpoint writes detect I/O failures. MAI3 load rejects truncation after the header/config.
- `include/mini_ai/tensor.hpp`: multidimensional `index` validates each coordinate.
- `include/mini_ai/tokenizer.hpp`: decode rejects IDs outside the byte vocabulary instead of silently masking them.
- `include/mini_ai/dataset.hpp`: zero-sized samples are rejected. Dataset windows remain bounded to `0..ids.size()-seq-1`, with `y[t]=ids[p+t+1]`.
- `CMakeLists.txt` and CI: optional ASan/UBSan build (`-DMINI_AI_SANITIZERS=ON`) and a dedicated workflow job.
- `tests/unit.cpp`: deterministic initialization/logits, context truncation, checkpoint step persistence, tokenizer bounds, and one training smoke test.

## Verification status

The source was fully read before edits. The execution environment used for this audit did not provide `git`, a C++ compiler, CMake, or a sanitizer runtime, so local compilation, CTest, finite-difference gradient checks, sanitizer execution, and remote CI are **NOT TESTED** here. The GitHub workflow is intended to provide those checks after publication.

Known limitation: MAI3 is a native binary format (raw `size_t`/`Config`), so it is not portable across incompatible ABI/endianness platforms. Checkpoint load is configuration and shape checked.
