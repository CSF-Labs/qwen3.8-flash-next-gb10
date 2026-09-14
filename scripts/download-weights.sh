#!/usr/bin/env bash
# Download the official NVIDIA checkpoint into a caller-selected local folder.
# Usage: scripts/download-weights.sh ~/models/Qwen3.8-Flash-Next-NVFP4
set -euo pipefail

MODEL_ID="nvidia/Qwen3.8-Flash-Next-NVFP4"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 TARGET_FOLDER" >&2
  echo "Example: $0 ~/models/Qwen3.8-Flash-Next-NVFP4" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: the Hugging Face 'hf' CLI is not installed or is not in PATH." >&2
  echo "Install it with: python3 -m pip install --user -U huggingface_hub" >&2
  exit 127
fi

target="$1"
mkdir -p "$target"
target="$(cd "$target" && pwd -P)"

echo ">> Downloading $MODEL_ID to $target"
echo ">> The download is resumable; rerun this command after an interruption."
HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1 \
  hf download "$MODEL_ID" \
    --local-dir "$target" \
    --max-workers "${MAX_WORKERS:-8}"

echo ">> Download complete: $target"
