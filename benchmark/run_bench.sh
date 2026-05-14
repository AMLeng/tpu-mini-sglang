#!/usr/bin/env bash
# run_bench.sh — Benchmark tpu-mini-sglang vs. tpu-inference (vLLM TPU)
# across a fixed workload matrix. For each config, alternates engines
# (mini-sglang first, then vllm), recording the `Serving Benchmark Result`
# block from `vllm bench serve` to a markdown doc.
#
# ============================================================================
# SETUP (one-time, on a fresh TPU VM)
# ============================================================================
#
# 1. Hardware: TPU v5e `v5litepod-4` (4 chips, 16 GB HBM each). TP=4 is
#    hardcoded; on a different TPU shape, retune --tp /
#    --tensor_parallel_size in this script and (likely) the long-context
#    `max_context = 40960` cap, which is sized for ~16 GB/chip.
#
# 2. System packages: bash 4.4+, coreutils, pgrep, setsid, ss, curl (all
#    standard on Linux). Install uv if not present:
#       curl -LsSf https://astral.sh/uv/install.sh | sh
#
# 3. HuggingFace access: accept the license for
#    `meta-llama/Llama-3.1-8B-Instruct` on huggingface.co, then either
#       huggingface-cli login
#    or `export HF_TOKEN=...` before launching. Weights (~16 GB) auto-
#    download to ~/.cache/huggingface on first server start.
#
# 4. ShareGPT dataset (only needed for `sharegpt-r*` rows; ~670 MB):
#       wget -O ~/ShareGPT_V3_unfiltered_cleaned_split.json \
#         https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
#
# 5. tpu-mini-sglang checkout and its venv:
#       git clone https://github.com/AMLeng/tpu-mini-sglang.git ~/tpu-mini-sglang
#       cd ~/tpu-mini-sglang && uv sync --extra tpu
#
# 6. vLLM venv with `vllm-tpu==0.13.3` (uses Python 3.12):
#       cd ~ && uv venv --python 3.12
#       cd ~ && uv pip install vllm-tpu==0.13.3
#    The script calls ~/.venv/bin/vllm directly, so no activation needed.
#
# 7. Place this script at ~/benchmark_results/run_bench.sh and chmod +x.
#    Logs and the output doc go under ~/benchmark_results/.
#
# Sanity checks before the first run:
#       ~/.venv/bin/python -c 'import vllm; print(vllm.__version__)'
#         # → 0.13.3
#       cd ~/tpu-mini-sglang && uv run python -c 'import jax; print(jax.devices())'
#         # → list of 4 TPU devices
#       ls -lh ~/ShareGPT_V3_unfiltered_cleaned_split.json
#         # → ~670 MB
#       huggingface-cli whoami
#         # → your account
#
# Make sure NO other process is using the TPU before starting. The script's
# preflight checks for stray `tpu_mini_sglang.launch_server` /
# `vllm serve` and port 30000, and refuses to start if it finds any.
#
# ============================================================================
# USAGE
# ============================================================================
#
#   ./run_bench.sh [--dry-run] [--only ID1,ID2,...] [--fresh]
#                  [--out PATH]  [--logs-dir PATH]
#                  [--model NAME]
#
# Examples:
#   ./run_bench.sh --only 01-decode-c1     # validate one row (~13 min)
#   ./run_bench.sh                          # full sweep (~4–8 h)
#   ./run_bench.sh --dry-run                # print commands only
#   ./run_bench.sh --fresh                  # overwrite today's doc
#
# Output doc & re-run behavior:
#   Default writes to ~/benchmark_results/YYYY-MM-DD-tpu-bench.md with
#   per-server logs under ~/benchmark_results/logs/.
#
#   If the doc already exists, the script *appends* new config sections
#   to it, and any prior section for the same id is replaced in place —
#   so re-running `--only 03-decode-c64` cleanly swaps that one section
#   without disturbing the rest. Use `--fresh` to wipe the doc and start
#   over (e.g. a stale run from the same day).
#
#   The doc is written incrementally as each config completes, so partial
#   results survive Ctrl-C / SIGHUP / SIGKILL (a per-server watchdog
#   cleans up servers if the script dies).

set -uo pipefail

# ---------- defaults ----------
MODEL_DEFAULT="meta-llama/Llama-3.1-8B-Instruct"
TPU_MINI_DIR="$HOME/tpu-mini-sglang"
VLLM_BIN="$HOME/.venv/bin/vllm"
SHAREGPT_PATH="$HOME/ShareGPT_V3_unfiltered_cleaned_split.json"
PORT=30000
TP=4
# see vllm/vllm/engine/arg_utils.py for where this constant comes from
VLLM_V5E_MAX_BATCH_TOKENS=512

OUT_DIR="$HOME/benchmark_results"
LOG_DIR_DEFAULT="$OUT_DIR/logs"
DEFAULT_OUT="$OUT_DIR/$(date +%Y-%m-%d)-tpu-bench.md"

INITIAL_WAIT_SEC=60
POLL_INTERVAL_SEC=10
READY_TIMEOUT_SEC=$((25 * 60))
WARMUP_PROBE_TIMEOUT_SEC=300
KILL_GRACE_SEC=30
PORT_FREE_TIMEOUT_SEC=60

# ---------- CLI ----------
DRY_RUN=0
ONLY=""
OUT_PATH="$DEFAULT_OUT"
LOG_DIR="$LOG_DIR_DEFAULT"
MODEL="$MODEL_DEFAULT"
FRESH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --out) OUT_PATH="$2"; shift 2 ;;
    --logs-dir) LOG_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    -h|--help)
      # Print the leading comment block (setup + usage), stripping the
      # "# " prefix. Stops at the first non-comment, non-blank line.
      awk '
        NR == 1 { next }                       # skip shebang
        /^[^#[:space:]]/ { exit }              # stop at first code line
        /^[[:space:]]*$/ { print ""; next }    # preserve blank lines
        { sub(/^# ?/, ""); print }             # strip "# " prefix
      ' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
mkdir -p "$(dirname "$OUT_PATH")" "$LOG_DIR"

# ---------- matrix ----------
# Format: id|workload|in_len|out_len|page_size|conc_or_rate|num_prompts|mode
# page_sizes chosen to match vllm's derived page sizes
CONFIGS=(
  "01-decode-c1|decode|8|1024|128|1|32|closed-loop"
  "02-decode-c16|decode|8|1024|128|16|64|closed-loop"
  "03-decode-c64|decode|8|1024|128|64|256|closed-loop"
  "04-decode-c256|decode|8|1024|128|256|1024|closed-loop"
  "05-prefill-c1|prefill|4096|8|256|1|1000|closed-loop"
  "06-prefill-c16|prefill|4096|8|256|16|1000|closed-loop"
  "07-prefill-c64|prefill|4096|8|256|64|1024|closed-loop"
  "08-balanced-c1|balanced|512|512|128|1|64|closed-loop"
  "09-balanced-c16|balanced|512|512|128|16|64|closed-loop"
  "10-balanced-c64|balanced|512|512|128|64|256|closed-loop"
  "11-balanced-c256|balanced|512|512|128|256|1024|closed-loop"
  "12-longctx-c1|longctx|32768|512|128|1|16|closed-loop"
  "13-longctx-c8|longctx|32768|512|128|8|16|closed-loop"
  "14-longctx-c32|longctx|32768|512|128|32|64|closed-loop"
  "15-sharegpt-r1|sharegpt|0|0|256|1|200|open-loop"
  "16-sharegpt-r2|sharegpt|0|0|256|2|200|open-loop"
  "17-sharegpt-r4|sharegpt|0|0|256|4|200|open-loop"
  "18-sharegpt-r8|sharegpt|0|0|256|8|200|open-loop"
  "19-sharegpt-r16|sharegpt|0|0|256|16|200|open-loop"
)

# ---------- helpers ----------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

size_max_context() {
  local workload="$1" in_len="$2" out_len="$3"
  case "$workload" in
    longctx)  echo 40960 ;;
    sharegpt) echo 4096 ;;
    *)
      # +512 padding covers chat-template wrapping in /v1/chat/completions.
      local needed=$((in_len + out_len + 512))
      if [ "$needed" -lt 2048 ]; then echo 2048; else echo "$needed"; fi
      ;;
  esac
}

size_max_seqs() {
  local concurrency="$1"
  if [ "$concurrency" -gt 256 ]; then echo "$concurrency"; else echo 256; fi
}

# Render an absolute path as ~/... when it lives under $HOME, so the doc is
# portable across users. Falls back to the original path otherwise.
# (Can't use ${p/#$HOME/~} because bash tilde-expands the replacement first.)
display_path() {
  local p="$1"
  if [[ "$p" = "$HOME"/* || "$p" = "$HOME" ]]; then
    echo "~${p#$HOME}"
  else
    echo "$p"
  fi
}

build_mini_sglang_cmd() {
  local model="$1" max_ctx="$2" max_btokens="$3" max_breqs="$4" page_size="$5"
  local mini_dir; mini_dir=$(display_path "$TPU_MINI_DIR")
  cat <<EOF
cd $mini_dir && uv run python -m tpu_mini_sglang.launch_server \\
  --model-path $model \\
  --port $PORT \\
  --tp $TP \\
  --max-context-len $max_ctx \\
  --max-num-batched-tokens $max_btokens \\
  --page-size $page_size \\
  --max-num-batched-requests $max_breqs
EOF
}

build_vllm_cmd() {
  local model="$1" max_ctx="$2" max_seqs="$3"
  local vllm_bin; vllm_bin=$(display_path "$VLLM_BIN")
  cat <<EOF
$vllm_bin serve $model \\
  --port $PORT \\
  --tensor_parallel_size=$TP \\
  --max-model-len=$max_ctx \\
  --max-num-seqs=$max_seqs
EOF
}

build_bench_cmd() {
  local workload="$1" in_len="$2" out_len="$3" cor="$4" np="$5" model="$6"
  local vllm_bin sharegpt
  vllm_bin=$(display_path "$VLLM_BIN")
  sharegpt=$(display_path "$SHAREGPT_PATH")
  case "$workload" in
    sharegpt)
      cat <<EOF
$vllm_bin bench serve \\
  --backend openai-chat \\
  --base-url http://localhost:$PORT \\
  --endpoint /v1/chat/completions \\
  --model $model \\
  --dataset-name sharegpt \\
  --dataset-path $sharegpt \\
  --num-prompts $np \\
  --request-rate $cor \\
  --seed 0 \\
  --temperature 0
EOF
      ;;
    *)
      cat <<EOF
$vllm_bin bench serve \\
  --backend openai-chat \\
  --base-url http://localhost:$PORT \\
  --endpoint /v1/chat/completions \\
  --model $model \\
  --dataset-name random \\
  --random-input-len $in_len \\
  --random-output-len $out_len \\
  --random-range-ratio 0.0 \\
  --num-prompts $np \\
  --max-concurrency $cor \\
  --ignore-eos \\
  --seed 0 \\
  --temperature 0
EOF
      ;;
  esac
}

# ---------- server lifecycle ----------
# Fail fast if a previous run left a server (or anything else holding the TPU
# or port $PORT) behind. JAX's TPU acquisition error is fatal but happens
# AFTER 3 min of `wait_for_ready` sleep, which wastes a lot of wall time.
preflight_check() {
  local issues=()
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then
    issues+=("port $PORT is already in use")
  fi
  local rogue
  rogue=$(pgrep -af 'tpu_mini_sglang\.launch_server' 2>/dev/null | grep -vw $$ || true)
  if [ -n "$rogue" ]; then
    issues+=("found running tpu_mini_sglang.launch_server processes:"$'\n'"$rogue")
  fi
  rogue=$(pgrep -af '(^|/)vllm serve' 2>/dev/null | grep -vw $$ || true)
  if [ -n "$rogue" ]; then
    issues+=("found running 'vllm serve' processes:"$'\n'"$rogue")
  fi
  if [ "${#issues[@]}" -gt 0 ]; then
    echo "PREFLIGHT FAILED — refusing to start to avoid TPU-lock errors:" >&2
    local i
    for i in "${issues[@]}"; do printf '  - %s\n' "$i" >&2; done
    cat >&2 <<HINT

To clean up:
  pkill -f 'tpu_mini_sglang\\.launch_server' || true
  pkill -f '(^|/)vllm serve'                 || true
  sleep 5 && ss -tlnp | grep ':$PORT ' || echo "port $PORT free"

Then re-run.
HINT
    return 1
  fi
  return 0
}

CURRENT_PID=""
cleanup() {
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    log "cleanup: killing pgid $CURRENT_PID"
    kill -TERM -- "-$CURRENT_PID" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$CURRENT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

start_server() {
  local cmd="$1" log_path="$2"
  local script_pid=$$
  # Two layers of liveness protection:
  # 1. setsid puts the server in its own session so we can kill the whole
  #    process group with `kill -- -$PID`.
  # 2. An embedded watchdog polls the parent script's PID and self-terminates
  #    the session if the parent vanishes (e.g., SIGKILL, terminal hang-up
  #    before the EXIT/HUP trap runs). Without this, the setsid'd server
  #    becomes orphaned and survives.
  PARENT_SCRIPT_PID="$script_pid" SERVER_CMD="$cmd" \
    setsid bash -c '
      (
        sleep 10
        while kill -0 "$PARENT_SCRIPT_PID" 2>/dev/null; do sleep 2; done
        echo "[watchdog] parent script $PARENT_SCRIPT_PID gone; killing pgid $$" >&2
        kill -TERM -- -$$ 2>/dev/null
        sleep 5
        kill -KILL -- -$$ 2>/dev/null
      ) &
      exec bash -c "$SERVER_CMD"
    ' >"$log_path" 2>&1 &
  echo $!
}

wait_for_ready() {
  local pid="$1" log_path="$2" engine="$3" model="$4"
  log "engine=$engine: sleeping ${INITIAL_WAIT_SEC}s for initial JIT compile"
  sleep "$INITIAL_WAIT_SEC"
  # Track elapsed by wall-clock, not by sleep accumulation — the warmup curl
  # probe can take up to WARMUP_PROBE_TIMEOUT_SEC and we want that against the
  # timeout budget too.
  local t_start
  t_start=$(date +%s)
  local saw_startup=0
  while :; do
    local now elapsed
    now=$(date +%s)
    elapsed=$(( INITIAL_WAIT_SEC + (now - t_start) ))
    if [ "$elapsed" -ge "$READY_TIMEOUT_SEC" ]; then
      log "ERROR engine=$engine timed out after $((READY_TIMEOUT_SEC/60))min"
      return 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR engine=$engine died before becoming ready"
      return 1
    fi
    if [ "$saw_startup" -eq 0 ] && grep -q "Application startup complete" "$log_path" 2>/dev/null; then
      saw_startup=1
      log "engine=$engine: saw 'Application startup complete'; sending warmup probe"
      sleep 5
    fi
    if [ "$saw_startup" -eq 1 ]; then
      if curl -sf --max-time "$WARMUP_PROBE_TIMEOUT_SEC" -o /dev/null \
        -H "Content-Type: application/json" \
        "http://localhost:$PORT/v1/chat/completions" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}"; then
        log "engine=$engine ready"
        return 0
      fi
      log "engine=$engine: probe returned non-OK, will retry"
    fi
    sleep "$POLL_INTERVAL_SEC"
  done
}

stop_server() {
  local pid="$1"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then return 0; fi
  kill -TERM -- "-$pid" 2>/dev/null || true
  local i=0
  while [ "$i" -lt "$KILL_GRACE_SEC" ] && kill -0 "$pid" 2>/dev/null; do
    sleep 1; i=$((i + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    log "force-killing pgid $pid"
    kill -KILL -- "-$pid" 2>/dev/null || true
    sleep 2
  fi
  for _ in $(seq 1 "$PORT_FREE_TIMEOUT_SEC"); do
    if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then return 0; fi
    sleep 1
  done
  log "WARN port $PORT still in use after kill"
  return 0
}

# ---------- doc writers ----------
write_preamble() {
  local out="$1"
  local vllm_ver py_bin
  py_bin="$(dirname "$VLLM_BIN")/python"
  vllm_ver=$("$py_bin" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "unknown")
  local mini_dir vllm_venv vllm_bin sharegpt log_dir script_disp
  mini_dir=$(display_path "$TPU_MINI_DIR")
  vllm_venv=$(display_path "$(dirname "$(dirname "$VLLM_BIN")")")
  vllm_bin=$(display_path "$VLLM_BIN")
  sharegpt=$(display_path "$SHAREGPT_PATH")
  log_dir=$(display_path "$LOG_DIR")
  script_disp=$(display_path "$SCRIPT_PATH")
  cat > "$out" <<EOF
# tpu-mini-sglang vs. tpu-inference (vLLM TPU) Benchmark

- **Generated:** $(date -Iseconds)
- **Host:** $(hostname)
- **Model:** \`$MODEL\`
- **TP:** $TP, **Port:** $PORT
- **vllm:** \`$vllm_ver\` (installed as \`vllm-tpu==0.13.3\`)

## Replication

This document was produced by \`$script_disp\`. To reproduce:

### 1. Environment

- \`tpu-mini-sglang\` checkout at \`$mini_dir\` with its uv-managed venv at \`$mini_dir/.venv\` (set up with \`uv sync\` inside that directory).
- vLLM venv at \`$vllm_venv\`, set up with Python 3.12 and \`vllm-tpu==0.13.3\`:
  \`\`\`bash
  cd ~ && uv venv --python 3.12 && uv pip install vllm-tpu==0.13.3
  \`\`\`
  After that, \`$vllm_bin\` is the entry point used here for both \`vllm serve\` and \`vllm bench serve\`.
- ShareGPT V3 unfiltered dataset at \`$sharegpt\`.
- Model: \`$MODEL\` (TP=$TP, port=$PORT).

### 2. Server sizing rules (auto-derived per config)

- \`max_context = max(2048, in_len + out_len + 512)\` for synthetic closed-loop workloads (the +512 covers chat-template wrapping).
- Long-context (32768/512) row uses \`max_context = 40960\` for headroom.
- ShareGPT rows use \`max_context = 4096\` (longer ShareGPT prompts get filtered by \`vllm bench serve\`).
- \`tpu-mini-sglang\`: \`--max-num-batched-tokens\` matches vLLM's v5e default for vllm serve
- Both servers: \`--max-num-seqs\` / \`--max-num-batched-requests\` = \`max(256, concurrency)\`.

### 3. Per-config protocol (strict alternation)

For every row in the matrix the script:

1. Starts \`tpu-mini-sglang.launch_server\` in its own process group (\`setsid\`), logging to \`$log_dir/<id>-tpu-mini-sglang.log\`.
2. Sleeps ${INITIAL_WAIT_SEC}s for JIT compilation, then polls the logfile every ${POLL_INTERVAL_SEC}s for \`Application startup complete\`. Aborts the row after $((READY_TIMEOUT_SEC/60)) minutes if not seen.
3. Sends a 1-token warmup probe to \`/v1/chat/completions\` so any remaining compilation finishes before the bench.
4. Runs the \`vllm bench serve\` command and captures its full stdout into the section below.
5. SIGTERMs the entire server process group; if it doesn't exit in ${KILL_GRACE_SEC}s, SIGKILLs it. Then waits for port \`$PORT\` to be free.
6. Repeats steps 1-5 for \`vllm serve\` (\`tpu-inference\`).

### 4. Bench commands

- **Synthetic closed-loop:** \`vllm bench serve --backend openai-chat ... --dataset-name random --random-input-len <I> --random-output-len <O> --random-range-ratio 0.0 --num-prompts <N> --max-concurrency <C> --ignore-eos --seed 0 --temperature 0\`.
- **ShareGPT open-loop:** \`vllm bench serve --backend openai-chat ... --dataset-name sharegpt --dataset-path $sharegpt --num-prompts 200 --request-rate <R> --seed 0 --temperature 0\` (no \`--max-concurrency\`, no \`--ignore-eos\`).

### 5. num-prompts per config

- Decode-heavy (8/1024): \`max(32, 4×C)\`
- Prefill-heavy (4096/8): \`max(1000, 16×C)\`
- Balanced (512/512): \`max(64, 4×C)\`
- Long-context (32768/512): \`max(16, 2×C)\`
- ShareGPT open-loop: 200 prompts per rate

### 6. Re-running

- One row: \`$script_disp --only 05-prefill-c1\` (comma-separate IDs for several).
- Dry run (commands only): \`$script_disp --dry-run\`.

## Matrix Summary

| ID | Workload | in/out | C or rate | Mode | num-prompts |
|---|---|---|---|---|---|
EOF
  for row in "${CONFIGS[@]}"; do
    IFS='|' read -r id workload in_len out_len cor np mode <<<"$row"
    local io cc
    if [ "$workload" = "sharegpt" ]; then io="(dataset)"; else io="$in_len/$out_len"; fi
    if [ "$mode" = "open-loop" ]; then cc="${cor} req/s"; else cc="C=${cor}"; fi
    printf '| %s | %s | %s | %s | %s | %s |\n' "$id" "$workload" "$io" "$cc" "$mode" "$np" >> "$out"
  done
}

# Drop any existing "## <id> — ..." section from the doc, along with the
# `---` + blank-line separator that immediately precedes it. Used when re-
# running a config and we want the new section to replace the old one in
# place rather than duplicate.
remove_section() {
  local out="$1" target="$2"
  [ -f "$out" ] || return 0
  awk -v target="$target" '
    function flush(   k) {
      for (k = 1; k <= buf_n; k++) print buf[k]
      buf_n = 0
    }
    /^## / {
      if ($2 == target) { buf_n = 0; in_target = 1; next }
      flush(); in_target = 0; print; next
    }
    {
      if (in_target) next
      if ($0 == "---" || $0 == "") { buf[++buf_n] = $0; next }
      flush(); print
    }
    END { flush() }
  ' "$out" > "$out.tmp" && mv "$out.tmp" "$out"
}

append_config_header() {
  local out="$1" id="$2" workload="$3" in_len="$4" out_len="$5" cor="$6" np="$7" mode="$8" bench_cmd="$9"
  local io cc
  if [ "$workload" = "sharegpt" ]; then io="(dataset)"; else io="$in_len/$out_len"; fi
  if [ "$mode" = "open-loop" ]; then cc="rate=${cor} req/s"; else cc="C=${cor}"; fi
  {
    echo ""
    echo "---"
    echo ""
    echo "## $id — $workload — $io — $cc — $mode — N=$np"
    echo ""
    echo "**Bench command:**"
    echo ""
    echo '```bash'
    echo "$bench_cmd"
    echo '```'
  } >> "$out"
}

# `vllm bench serve` prints hundreds of lines (TPU init, argparse Namespace
# dump, tqdm progress bar) before the actual results. We only persist the
# `Serving Benchmark Result` block. If that block isn't present (server
# failed to start, bench errored, etc.), fall back to the last 50 lines so
# the doc still surfaces the failure mode without ballooning.
extract_results() {
  local output="$1"
  local extracted
  extracted=$(printf '%s\n' "$output" | awk '
    /Serving Benchmark Result/ {flag=1; print; next}
    flag && /^=+$/ {print; exit}
    flag {print}
  ')
  if [ -n "$extracted" ]; then
    printf '%s' "$extracted"
  else
    printf '%s' "$output" | tail -50
  fi
}

append_engine_section() {
  local out="$1" engine="$2" server_cmd="$3" log_path="$4" bench_output="$5" status="$6" wall_sec="$7"
  local rel_log results
  rel_log=$(realpath --relative-to="$(dirname "$out")" "$log_path" 2>/dev/null || echo "$log_path")
  results=$(extract_results "$bench_output")
  {
    echo ""
    echo "### $engine"
    echo ""
    echo "**Server command:**"
    echo ""
    echo '```bash'
    echo "$server_cmd"
    echo '```'
    echo ""
    echo "**Server log:** [\`$rel_log\`]($rel_log)  "
    echo "**Status:** $status  "
    echo "**Wall time (incl. startup):** ${wall_sec}s"
    echo ""
    echo "**Results:**"
    echo ""
    echo '```'
    echo "$results"
    echo '```'
  } >> "$out"
}

# ---------- orchestration ----------
run_one_engine() {
  local out="$1" id="$2" engine="$3" server_cmd="$4" bench_cmd="$5"
  local log_path="$LOG_DIR/${id}-${engine}.log"
  log "=== ${id} / ${engine} ==="
  log "starting server, logging to $log_path"
  local t0 t1
  t0=$(date +%s)
  CURRENT_PID=$(start_server "$server_cmd" "$log_path")
  log "server pgid=$CURRENT_PID"

  local status bench_output
  if wait_for_ready "$CURRENT_PID" "$log_path" "$engine" "$MODEL"; then
    log "running bench"
    # Command substitution captures stdout/stderr; $? captures the exit code.
    # -e is not set at script top, so we don't need to toggle it here.
    bench_output=$(bash -c "$bench_cmd" 2>&1)
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      status="OK"
    else
      status="BENCH-FAILED (exit $rc)"
    fi
  else
    status="SERVER-START-FAILED"
    bench_output="(server did not become ready; see $log_path)"
  fi

  log "stopping server pgid=$CURRENT_PID"
  stop_server "$CURRENT_PID"
  CURRENT_PID=""
  t1=$(date +%s)

  append_engine_section "$out" "$engine" "$server_cmd" "$log_path" "$bench_output" "$status" "$((t1 - t0))"
  log "${id} / ${engine}: status=$status, wall=$((t1-t0))s"
}

run_one_config() {
  local out="$1" row="$2"
  IFS='|' read -r id workload in_len out_len page_size cor np mode <<<"$row"
  local max_ctx max_seqs
  max_ctx=$(size_max_context "$workload" "$in_len" "$out_len")
  max_seqs=$(size_max_seqs "$cor")

  local mini_cmd vllm_cmd bench_cmd
  # Match vllm's V5E vllm serve default (OPENAI_API_SERVER usage context = 512)
  # so mini-sglang and vllm both run with the same prefill budget
  mini_cmd=$(build_mini_sglang_cmd "$MODEL" "$max_ctx" "$VLLM_V5E_MAX_BATCH_TOKENS" "$max_seqs" "$page_size")
  vllm_cmd=$(build_vllm_cmd "$MODEL" "$max_ctx" "$max_seqs")
  bench_cmd=$(build_bench_cmd "$workload" "$in_len" "$out_len" "$cor" "$np" "$MODEL")

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== $id (max_ctx=$max_ctx, max_seqs=$max_seqs) ==="
    echo "--- bench cmd ---"; echo "$bench_cmd"
    echo "--- tpu-mini-sglang cmd ---"; echo "$mini_cmd"
    echo "--- tpu-inference cmd ---"; echo "$vllm_cmd"
    return 0
  fi

  # If a previous section for this id already exists in the doc (e.g. from
  # an earlier run that we're now re-doing), drop it so the new section
  # cleanly replaces it instead of producing a duplicate.
  remove_section "$out" "$id"
  append_config_header "$out" "$id" "$workload" "$in_len" "$out_len" "$cor" "$np" "$mode" "$bench_cmd"
  run_one_engine "$out" "$id" "tpu-mini-sglang" "$mini_cmd" "$bench_cmd"
  run_one_engine "$out" "$id" "tpu-inference"   "$vllm_cmd" "$bench_cmd"
}

# ---------- main ----------
selected=()
if [ -n "$ONLY" ]; then
  IFS=',' read -ra wanted <<<"$ONLY"
  for row in "${CONFIGS[@]}"; do
    id="${row%%|*}"
    for w in "${wanted[@]}"; do
      if [ "$id" = "$w" ]; then selected+=("$row"); fi
    done
  done
  if [ "${#selected[@]}" -eq 0 ]; then
    echo "No configs matched --only=$ONLY" >&2
    exit 2
  fi
else
  selected=("${CONFIGS[@]}")
fi

if [ "$DRY_RUN" -ne 1 ]; then
  preflight_check || exit 3
  # Write the preamble only if starting fresh, or if the doc doesn't exist
  # yet. Otherwise append new config sections to the existing doc (with
  # remove_section replacing any prior section for the same id in place).
  if [ "$FRESH" -eq 1 ] || [ ! -f "$OUT_PATH" ]; then
    write_preamble "$OUT_PATH"
    log "writing fresh doc to $OUT_PATH"
  else
    log "appending to existing doc $OUT_PATH (preamble preserved)"
  fi
  log "server logs go under $LOG_DIR"
fi

for row in "${selected[@]}"; do
  run_one_config "$OUT_PATH" "$row"
done

log "done"
