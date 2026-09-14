# syntax=docker/dockerfile:1.7
# v0.29.0 is the newest stable official vLLM image at the time this file was written.
# Override only after checking that the pinned compatibility patches still apply.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.title="Qwen3.8-Flash-Next on GB10"
LABEL org.opencontainers.image.description="Official vLLM image with audited, checksum-pinned GB10 compatibility patches"
LABEL org.opencontainers.image.source="https://github.com/blazux/qwen3.8-Flash-DGX"

USER root

ARG VLLM_SITE=/usr/local/lib/python3.12/dist-packages
ARG QWEN_IMPL=${VLLM_SITE}/vllm/models/qwen4_exp/nvidia
ARG AUDITED_COMMIT=d542745cd41045bcd71b9d7dbe659bca014202bf
ARG PATCH_ROOT=https://raw.githubusercontent.com/blazux/qwen3.8-Flash-DGX/${AUDITED_COMMIT}/src

# These four files were reviewed at AUDITED_COMMIT. BuildKit verifies their exact
# contents before any Python patcher is run; a changed upstream file fails the build.
ADD --checksum=sha256:0bfefaa132da029067b4a0181991a867dd00cfee823464a1132e52b4a8c55aef \
    ${PATCH_ROOT}/vllm_ple_mmap.py ${VLLM_SITE}/vllm_ple_mmap.py
ADD --checksum=sha256:bcf59adbe3e317bc6300f6eecd01d2dcdb6388b7341449c74bdcee8c949b1f11 \
    ${PATCH_ROOT}/patch_mamba_block_size.py /tmp/patch_mamba_block_size.py
ADD --checksum=sha256:802f6564e514fe5a228738873570f9ce94c230f40d3a7c932cc5893e2147a010 \
    ${PATCH_ROOT}/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
ADD --checksum=sha256:5630416895c17c49b7b1cba294f2cc12b1bd788122c2d19f93afa058c1ea1a7f \
    ${PATCH_ROOT}/patch_block_fp8_mtp.py /tmp/patch_block_fp8_mtp.py

RUN set -eu; \
    ple="${QWEN_IMPL}/ple_layer.py"; \
    test -f "$ple"; \
    cp "$ple" "$ple.orig"; \
    printf '\nfrom vllm_ple_mmap import apply as _gb10_ple_apply\n_gb10_ple_apply(Qwen4ExpNGramEmbedding)\n' >> "$ple"; \
    python3 /tmp/patch_mamba_block_size.py "${VLLM_SITE}"; \
    python3 /tmp/patch_qsa_exact_topk.py "${QWEN_IMPL}/ops/qsa.py"; \
    python3 /tmp/patch_block_fp8_mtp.py "${VLLM_SITE}"; \
    python3 -m py_compile "$ple" "${VLLM_SITE}/vllm_ple_mmap.py"; \
    rm -f /tmp/patch_*.py

# GB10 reports 99 KiB shared memory, just below the upstream 100 KiB gate. Pinning
# two warps also avoids the documented Blackwell tl.dot race in this vLLM release.
RUN set -eu; \
    utils="${VLLM_SITE}/vllm/third_party/flash_linear_attention/ops/utils.py"; \
    cdh="${VLLM_SITE}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py"; \
    grep -q 'DEFAULT = 102400' "$utils"; \
    sed -i 's/DEFAULT = 102400/DEFAULT = 101376  # GB10: 99 KiB/' "$utils"; \
    grep -q 'for num_warps in \[2, 4\]' "$cdh"; \
    sed -i 's/for num_warps in \[2, 4\]/for num_warps in [2]  # GB10 tl.dot workaround/' "$cdh"

EXPOSE 8000

