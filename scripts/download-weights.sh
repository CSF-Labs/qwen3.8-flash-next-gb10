#!/usr/bin/env bash
# Download the official NVIDIA checkpoint into a caller-selected local folder.
# Usage: scripts/download-weights.sh ~/models/Qwen3.8-Flash-Next-NVFP4
set -euo pipefail

MODEL_ID="nvidia/Qwen3.8-Flash-Next-NVFP4"
DOWNLOAD_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 TARGET_FOLDER" >&2
  echo "Example: $0 ~/models/Qwen3.8-Flash-Next-NVFP4" >&2
  exit 2
fi

target="$1"
mkdir -p "$target"
target="$(cd "$target" && pwd -P)"

token_args=()
if [ -n "${HF_TOKEN:-}" ]; then
  token_args=(-e HF_TOKEN)
fi

echo ">> Downloading $MODEL_ID to $target"
echo ">> The download is resumable; rerun this command after an interruption."
docker run --rm \
  --name qwen38-weights-download \
  --entrypoint hf \
  -e HF_HUB_DISABLE_XET=0 \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  "${token_args[@]}" \
  -v "$target:/model" \
  "$DOWNLOAD_IMAGE" \
  download "$MODEL_ID" --local-dir /model --max-workers "${MAX_WORKERS:-8}"

echo ">> Download complete: $target"
