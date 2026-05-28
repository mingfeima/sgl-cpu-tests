#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"

export HF_HOME="${HF_HOME:-/home/mingfeima/.cache/huggingface}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/home/mingfeima/models}"
export SGLANG_USE_CPU_ENGINE=1
export TCMALLOC_RELEASE_RATE=0

# Runtime toggles:
export ENABLE_PROFILE=1
export ENABLE_TORCH_COMPILE=0
export MODEL_PATH=Qwen/Qwen3.5-4B
export SGLANG_CPU_OMP_THREADS_BIND="0-39"

ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

if [[ -f "/etc/ssl/certs/ca-certificates.crt" ]]; then
  export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
  export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
fi

export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
export KMP_FORKJOIN_BARRIER_PATTERN="dist,dist"
export KMP_PLAIN_BARRIER_PATTERN="dist,dist"
export KMP_REDUCTION_BARRIER_PATTERN="dist,dist"
# Keep KMP_AFFINITY unset by default to avoid conflicts with libnuma binding.
# If needed, set KMP_AFFINITY in your shell before launching this script.

mkdir -p "${HF_HOME}"

python_bin="/home/mingfeima/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "Missing Python at ${python_bin}"
  exit 1
fi

preloads=()
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libtcmalloc.so.4" ]]; then
  preloads+=("${CONDA_PREFIX}/lib/libtcmalloc.so.4")
elif [[ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" ]]; then
  preloads+=("/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4")
elif [[ -f "/lib/x86_64-linux-gnu/libtcmalloc.so.4" ]]; then
  preloads+=("/lib/x86_64-linux-gnu/libtcmalloc.so.4")
elif [[ -f "/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]]; then
  preloads+=("/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4")
fi
if [[ -f "/home/mingfeima/.venv/lib/libiomp5.so" ]]; then
  preloads+=("/home/mingfeima/.venv/lib/libiomp5.so")
fi

if [[ ${#preloads[@]} -gt 0 ]]; then
  preload_joined="$(IFS=:; echo "${preloads[*]}")"
  export LD_PRELOAD="${preload_joined}"
else
  unset LD_PRELOAD || true
fi

cmd=(
  "${python_bin}" -m sglang.bench_one_batch
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --device cpu
  --disable-radix-cache
  --disable-cuda-graph
  --tp 1
  --mem-fraction-static 0.8
  --max-total-tokens 63356
  --max-prefill-tokens 4096
  --dtype bfloat16
  --batch-size 1
  --input-len 1000
  --output-len 3
  --attention-backend intel_amx
  --profile-filename-prefix qwen3.5-4b-cpu-amx
)

if [[ "${ENABLE_TORCH_COMPILE}" == "1" ]]; then
  cmd+=(--enable-torch-compile)
fi

if [[ "${ENABLE_PROFILE}" == "1" ]]; then
  cmd+=(--profile --profile-record-shapes)
fi

model_slug="${MODEL_PATH//\//--}"
model_slug="${model_slug// /_}"
log_file="/home/mingfeima/models/bench_one_batch_${model_slug}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"
"${cmd[@]}" 2>&1 | tee "${log_file}"
