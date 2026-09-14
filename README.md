# Qwen3.8-Flash-Next on one GB10 with vLLM

This project serves NVIDIA's official `nvidia/Qwen3.8-Flash-Next-NVFP4`
checkpoint through vLLM's OpenAI-compatible API. It targets a DGX Spark or a
compatible 128 GB GB10/aarch64 system. MTP speculative decoding is enabled by
default with two proposed tokens per decoding step.

The image starts from the official stable `vllm/vllm-openai:v0.29.0` image. A
floating `latest` or `nightly` tag is deliberately not used: the model-specific
patches modify vLLM internals and must be tied to a known-compatible release.
`VLLM_IMAGE` remains a build argument so a newer version can be tested explicitly.

## Host prerequisites (outside Docker)

1. Install a recent NVIDIA driver supported by the DGX Spark software stack.
2. Install Docker Engine, the Compose plugin, and NVIDIA Container Toolkit.
3. Configure Docker's NVIDIA runtime, then restart Docker. NVIDIA documents this
   as `sudo nvidia-ctk runtime configure --runtime=docker` followed by
   `sudo systemctl restart docker`.
4. Confirm GPU passthrough before building:

   ```bash
   docker run --rm --gpus all ubuntu:24.04 nvidia-smi
   ```

5. Create `~/models` on fast local NVMe storage with at least 150 GB free. Place
   each model in its own subfolder. The official checkpoint's approximately 50
   GiB PLE/n-gram table is read via `mmap` while serving, so network filesystems
   are a poor fit.
6. Install the Hugging Face CLI with Xet support on the host. Xet is important
   for the checkpoint's very large files:

   ```bash
   python3 -m pip install --user -U huggingface_hub
   hf --help
   ```

   If your Python user scripts directory is not already in `PATH`, add it before
   running the downloader. Alternatively, install the CLI in a virtual environment.
7. Review and accept both the NVIDIA Open Model License and the linked Qwen
   Community License terms on the model page. If Hugging Face requires
   authentication, create a read-only token and run `hf auth login`, or export
   `HF_TOKEN` before invoking the downloader.

Do not expose this unauthenticated API directly to the Internet. The default bind
address is loopback only; put an authenticated reverse proxy in front of it for
remote access.

## Start

```bash
cp .env.example .env
# Edit .env. Set HF_TOKEN only if the Hub requires it.
mkdir -p ~/models
scripts/download-weights.sh ~/models/Qwen3.8-Flash-Next-NVFP4
docker compose build --pull
docker compose up -d
docker compose logs -f qwen
```

The download is roughly 130+ GB and is resumable. The downloader runs the host's
`hf` CLI directly and requires exactly one destination-folder parameter; it does
not start or pull a Docker image. To use another subfolder, download there and set
`MODEL_FOLDER` in `.env` to that subfolder's name. `MODELS_DIR` defaults to
`~/models`; Compose mounts it read-only at `/models` in the serving container.

Model initialization can take 10–20 minutes. Compose's health check allows 15
minutes before counting failures.

Test the endpoint after the container becomes healthy:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0,"max_tokens":64}'
```

Stop without deleting the weights in `~/models`:

```bash
docker compose down
```

## Why patches are still present

The full NVIDIA NVFP4 checkpoint is too large to coexist with a useful KV cache
in GB10's 128 GB unified-memory pool. The reference project avoids loading the
large PLE lookup table into unified memory and instead maps it from NVMe. It also
works around a GB10 Flash Linear Attention shared-memory boundary, a Blackwell
Triton race, vLLM's hybrid-state prefix-cache block sizing, and NVIDIA-checkpoint
MTP FP8 metadata handling.

`VLLM_QSA_EXACT_TOPK=1` selects the slower PyTorch exact-top-k path because the
stock persistent top-k kernel has documented nondeterministic results on GB10.
Set `EXACT_TOPK=0` only after validating output correctness on your exact driver,
firmware, and vLLM combination.

### Optional reduced MTP draft vocabulary

The image includes an audited 65,536-token draft vocabulary and a runtime patch
that applies it only to MTP proposals. MTP=2 is enabled by default, while the
reduced vocabulary is disabled by default. Enable it in `.env` before creating
the container:

```dotenv
ENABLE_REDUCED_DRAFT_VOCAB=1
```

Then recreate the container:

```bash
docker compose up -d --build --force-recreate
```

The draft head then scores 65,536 likely tokens instead of the full 248,320-token
vocabulary, reducing its weight read from about 1.27 GiB to about 320 MiB per
proposed token. The full target model still verifies proposals and retains its
complete vocabulary; an excluded draft token can still be emitted by the target
after rejection.

For multilingual or unusual-token-heavy workloads, compare MTP acceptance and
throughput with the reduced-vocabulary option both on and off. To disable MTP
entirely, remove the `--speculative-config` pair from `compose.yaml`.

This intentionally omits the reference repo's hybrid weight conversion, request
watcher, shell control frontend, and separately compiled CUDA top-k extension.
The reduced MTP vocabulary is included as an opt-in, checksum-pinned artifact.

## Security and provenance review

The retained files were reviewed from reference commit
`d542745cd41045bcd71b9d7dbe659bca014202bf` (2026-09-13). The review checked the
Dockerfiles, shell entrypoints, Python patchers, network/process execution, token
handling, filesystem deletion, privilege requests, and remote build inputs. No
credential exfiltration, persistence, miner, reverse shell, Docker-socket access,
or unexplained privilege escalation was found.

The retained Python files perform tensor loading/mapping and deterministic source
rewrites only. Docker BuildKit fetches each from the pinned commit with an exact
SHA-256 checksum, so upstream changes cannot silently enter a build. The runtime
drops all Linux capabilities, enables `no-new-privileges`, exposes only loopback
by default, and does not mount the Docker socket. `ipc: host` is retained because
it is vLLM's documented approach for shared-memory performance; it weakens IPC
isolation and should be considered when running untrusted co-tenants.

This is a source review, not a formal security proof. The base image, Python
packages, CUDA stack, model checkpoint, and Hugging Face transport remain external
supply-chain dependencies. For reproducible production deployment, pin the vLLM
image by multi-arch digest and mirror the checkpoint at a reviewed revision.

## Updating vLLM

Do not merely change `VLLM_IMAGE` to `nightly`. First rebuild in a disposable
environment: the patchers intentionally fail when expected source text changes.
Then run coherence, long-context, repeated-prefix, deterministic greedy, and load
tests. Remove a patch when the corresponding upstream fix is included rather than
forcing an old rewrite onto new internals.

## Troubleshooting

- An immediate OOM usually means other GPU/unified-memory users are active. Stop
  them or lower `GPU_MEMORY_UTILIZATION`.
- Very slow or unstable prefill usually means the Hugging Face cache is not on
  local NVMe or the host is swapping.
- If the Hub rate-limits or denies the model, set a read-only `HF_TOKEN` in `.env`.
- If the API is coherent but too slow, keep exact top-k on until correctness has
  been established; performance tuning should come after deterministic tests.
