# tpu-mini-sglang

A small (~4500 LOC), learning-oriented LLM inference library for TPUs, written
in JAX and modeled on [SGLang](https://github.com/sgl-project/sglang). The
codebase is intentionally compact — the goal is to be readable end-to-end while
still delivering serving performance comparable to production engines on TPU.
It is inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang).

On a single-host `v5litepod-4` running Llama-3.1-8B-Instruct at TP=4, total
throughput is **within 0–8 %** of `vllm-tpu` 0.13.3 on the realistic ShareGPT
open-loop workload across arrival rates from 1 to 16 req/s. See
[REPORT.md](REPORT.md) for the full 19-config matrix and analysis.

## Highlights

- **JAX/Flax**, sharded with `jax.sharding.Mesh` for tensor parallelism (TP).
  Uses JIT precompilation with recompilation monitoring.
- **SGLang-style serving stack**: tokenizer manager + HTTP server in the main
  process, scheduler in a subprocess, detokenizer in a third subprocess,
  connected via ZeroMQ IPC.
- **Paged KV cache** with a **radix-tree prefix cache** for prompt reuse.
- **Overlap scheduling**: the scheduler issues batch `N+1` while batch `N-1`
  is still being read back from device, hiding host/device synchronization cost.
- **Ragged paged attention kernel** vendored under
  `tpu_mini_sglang/kernels/ragged_paged_attention` for efficient
  mixed-length attention.
- **OpenAI-compatible HTTP API** (`/v1/chat/completions`, `/v1/models`) plus
  a native `/generate` endpoint with SSE streaming.
- **Llama** architecture (`LlamaForCausalLM`) is the only model family wired
  up today; the registry in `tpu_mini_sglang/models/registry.py` is the place
  to add more.

## Repository layout

```
tpu_mini_sglang/
  entrypoints/        # HTTP server, OpenAI protocol, engine launcher
  managers/           # tokenizer_manager, scheduler, detokenizer_manager,
                      # schedule_batch, schedule_policy, io structs
  model_executor/     # ModelRunner and ForwardBatch construction
  models/             # Model implementations + registry + loader
  layers/             # Attention, sampler, rotary, swiglu, attention backends
  kernels/            # Vendored Pallas / XLA kernels (ragged paged attention)
  mem_cache/          # KV memory pool, paged allocator, radix cache, tree node
  sampling/           # Sampling params and per-batch sampler state
  launch_server.py    # `python -m tpu_mini_sglang.launch_server`
  server_args.py      # CLI flags / dataclass for server configuration
  sharding.py         # Mesh construction and partition specs

benchmark/            # run_bench.sh harness + offline benchmark
charts/               # Plotting script + generated charts used by REPORT.md
REPORT.md             # Performance comparison vs. vllm-tpu (2026-05-14)
```

## Requirements

- Python **3.12+**
- A TPU host with [JAX TPU support](https://docs.jax.dev/en/latest/installation.html).
  CPU extras are provided for local development.
- `uv` (recommended) — install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AMLeng/tpu-mini-sglang.git
cd tpu-mini-sglang

# 2. Create a Python virtualenv
uv venv

# 3. Install dependencies — pick the extra that matches your hardware
uv sync --extra tpu   # TPU host (recommended runtime target)
# uv sync --extra cpu # Local CPU iteration
```

If you plan to serve gated models (e.g. Llama-3.1), export `HF_TOKEN` before
launching the server, or run `huggingface-cli login` once. Weights download to
`~/.cache/huggingface` on first use.

Dev tooling (`ruff`, `mypy`, `codespell`, `pre-commit`, `matplotlib`) is in the
default `dev` group and installs automatically with `uv sync`. Activate the
pre-commit hooks once with:

```bash
uv run pre-commit install
```

## Running the server

```bash
uv run python -m tpu_mini_sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp 4 \
  --port 30000
```

Key flags (see `tpu_mini_sglang/server_args.py` for the full list):

| Flag | Meaning |
| --- | --- |
| `--model-path` | Local directory or HuggingFace repo ID. |
| `--tp` / `--dp` | Tensor / data parallelism degree. `tp * dp` must equal the device count. |
| `--page-size` | KV-cache page size (default 128). |
| `--max-context-len` | Cap on context length; defaults to the model's max position embeddings. |
| `--max-num-batched-tokens` | Per-step prefill token budget. |
| `--max-num-batched-requests` | Maximum concurrent in-flight requests. |
| `--enable-overlap` | Run scheduler in overlap mode (default `True`). |
| `--skip-server-warmup` | Skip the post-launch 1-token warm-up request. |
| `--skip-scheduler-warmup` | Skip JIT precompilation of batch shapes. |
| `--use-jax-profiler-server` / `--profiler-port` | Start `jax.profiler` on the given port. |

Once the server reports `Application startup complete`, hit it like any
OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Or use the native generate endpoint:

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 16}
  }'
```

## Architecture in one paragraph

A request flows: HTTP handler → `TokenizerManager` (tokenizes, registers a
future) → ZMQ → `Scheduler` subprocess (batches via `PrefillAdder`, runs
`ModelRunner.model_fn` on the JAX mesh, samples) → ZMQ →
`DetokenizerManager` subprocess (incrementally detokenizes) → ZMQ back to
`TokenizerManager` (resolves the future, streams to the HTTP client). The
scheduler keeps a waiting queue, a single running decode batch, and at most one
in-flight chunked prefill. The model runner precompiles a discrete set of
batch shapes at startup so steady-state forward passes hit cached JIT entries.

Scheduling is **prefill-first**: as long as new prompts are admissible, they
preempt decode for a step. This minimizes time-to-first-token at the cost of
inter-token-latency spikes when a large prefill lands. The trade-off is
documented quantitatively in [REPORT.md](REPORT.md).

## Benchmarking

The benchmark harness lives in `benchmark/`:

- `benchmark/run_bench.sh` — alternates `tpu-mini-sglang` and `vllm-tpu` across
  a 19-row workload matrix using `vllm bench serve` as the load generator.
  Read the header of the file for one-time setup steps.
- `benchmark/2026-05-14-tpu-bench.md` — raw per-config output from the last
  reported run.
- `charts/plot_charts.py` — regenerates the figures embedded in `REPORT.md`.

## Development

```bash
uv run pre-commit install        # ruff, codespell, etc.
uv run ruff check .
uv run ruff format --check .
uv run mypy tpu_mini_sglang
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
