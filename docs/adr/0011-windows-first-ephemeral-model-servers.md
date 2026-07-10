# ADR 0011: Windows-first ephemeral model servers

The local execution policy is native Windows first. WSL2 is permitted only for a Backend whose native candidate has failed the same conformance and English/Finnish acceptance fixtures; it is not a global deployment target. The current exception is Parakeet/NeMo alignment.

Local Structured Text uses a manifest-selected GGUF through one temporary stock `llama-server.exe`, not an embedded Python binding. The existing runner manager owns the exclusive GPU lease and batches adjacent text tasks. A stdlib control worker starts the server on `127.0.0.1` with a generated API key and one slot, polls `/health`, sends schema-constrained `/v1/chat/completions` requests, and terminates the child before the next model family. Shutdown requires process exit and, when observable, disappearance of the server GPU PID. Baseline/load/peak/post-exit aggregate VRAM is recorded with a tolerance because Windows WDDM and unrelated applications make exact global equality unreliable.

One typed `local-llm.toml` represents one evaluation variant. It freezes the exact target GGUF, optional compatible drafter, full repository and llama.cpp commits, independent SHA-256 hashes, reviewed license, context/batch settings, and speculative mode. MTP is disabled by default and evaluated as a separate variant. Context is allocated at server startup; increasing it requires a bounded relaunch and must not silently jump to 256K on a 24 GB GPU.

Candidate model names do not enter workflow contracts. English/Finnish schema validity, story quality, cold/warm throughput, load time, peak VRAM, and cleanup determine which manifest becomes the curated default. This keeps model selection empirical while preserving a simple fixed workflow and one-GPU lifecycle.
