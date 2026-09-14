#!/usr/bin/env bash
set -euo pipefail

case "${ENABLE_REDUCED_DRAFT_VOCAB:-0}" in
  0|false|FALSE|no|NO|"")
    unset VLLM_MTP_DRAFT_VOCAB
    ;;
  1|true|TRUE|yes|YES)
    export VLLM_MTP_DRAFT_VOCAB=/opt/qwen/draft_vocab_65536.npy
    ;;
  *)
    echo "Error: ENABLE_REDUCED_DRAFT_VOCAB must be 0 or 1." >&2
    exit 2
    ;;
esac

exec vllm serve "$@"
