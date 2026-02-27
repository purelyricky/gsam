"""
modal_vllm_server.py
====================
Self-hosted vLLM inference server on Modal.com for GSAM.

This script deploys an OpenAI-compatible vLLM server on Modal so that GSAM
can run against any open-weight model (e.g. DeepSeek, Llama, Qwen) without
depending on a third-party API.

## Quick Start

1. Install Modal and authenticate:
       pip install modal
       modal token new

2. (Optional) Set your Hugging Face token for gated models:
       modal secret create huggingface HF_TOKEN=hf_...

3. Deploy the server:
       modal deploy modal_vllm_server.py

4. After deployment Modal prints the web URL.  Add it to your .env:
       MODAL_BASE_URL=https://<workspace>--gsam-vllm-server-serve.modal.run/v1
       MODAL_API_KEY=dummy    # vLLM does not enforce auth by default

5. Run GSAM with the modal provider:
       python eval/finance/run_gsam.py \
           --api_provider modal \
           --generator_model <MODEL_NAME> \
           --task_name finer \
           --mode online \
           --save_path ./results

## Configuration

Edit the constants below to change the model, GPU type, and other settings.

MODEL_NAME     – HuggingFace repo id (or local path inside the container).
                 Must match what you pass as --generator_model to run_gsam.py
                 (vLLM also registers "llm" as a short alias – see SERVED_MODEL_NAMES).
GPU_TYPE       – Modal GPU string, e.g. "H100", "A100", "A10G", "T4", "L4".
N_GPU          – Tensor-parallel replicas (set >1 only for very large models).
FAST_BOOT      – True  → enforce eager mode (faster cold-start, slightly slower inference).
                 False → enable CUDA graph capture (slower cold-start, faster inference).
HF_CACHE_VOL   – Name of the Modal Volume used to cache model weights across deploys.
VLLM_CACHE_VOL – Name of the Modal Volume used to cache compiled vLLM artefacts.

## How it works

vLLM's `serve` subcommand starts an OpenAI-compatible HTTP server on port 8000.
Modal's @web_server decorator tunnels that port to a public HTTPS URL.
GSAM's `modal` provider points its openai.OpenAI client at that URL, so every
timed_llm_call() works exactly as it does with OpenAI/SambaNova/Together.
"""

import subprocess
import modal

# ---------------------------------------------------------------------------
# Configuration – edit these to match your model and hardware budget
# ---------------------------------------------------------------------------

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
"""
HuggingFace model ID to serve.  Recommended open-weight models for GSAM:
  - "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"   (fast, strong reasoning, ~15 GB)
  - "Qwen/Qwen2.5-7B-Instruct"                   (fast, good instruction follow)
  - "Qwen/Qwen3-8B-FP8"                          (FP8-quantised, saves GPU memory)
  - "meta-llama/Llama-3.1-8B-Instruct"           (requires HF token for gated access)
  - "deepseek-ai/DeepSeek-V3"                    (large, needs multi-GPU, see N_GPU)
"""

GPU_TYPE = "A10G"
"""
Modal GPU type.  Cost/performance options:
  - "T4"   – cheapest, ~16 GB VRAM, fine for 7B FP16 or small quantised models.
  - "L4"   – mid-range, ~24 GB VRAM.
  - "A10G" – good balance of cost and speed, ~24 GB VRAM.
  - "A100" – fast, 40 or 80 GB VRAM.  Use for 13B+ models.
  - "H100" – fastest, most expensive; needed for 70B+ models.
"""

N_GPU = 1
"""Number of GPUs (tensor-parallel shards).  Increase for large models."""

FAST_BOOT = True
"""
True  → --enforce-eager disables CUDA graph capture → faster cold-start.
False → CUDA graphs enabled → higher steady-state throughput.
Set False for long-running deployments with many concurrent requests.
"""

HF_CACHE_VOL = "huggingface-cache"
VLLM_CACHE_VOL = "vllm-cache"

VLLM_PORT = 8000
MINUTES = 60

# Additional vLLM flags passed to the serve command.
# See: https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
EXTRA_VLLM_ARGS: list[str] = [
    "--max-model-len", "8192",
    "--dtype", "auto",
]

# ---------------------------------------------------------------------------
# Modal image – CUDA base + vLLM + HuggingFace Hub
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.9.1",
        "huggingface-hub[hf_xet]==0.31.4",
        "hf_transfer==0.1.9",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)

# ---------------------------------------------------------------------------
# Persistent volumes for weight + compilation caching
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOL, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name(VLLM_CACHE_VOL, create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------

app = modal.App("gsam-vllm-server")

# ---------------------------------------------------------------------------
# vLLM web server function
# ---------------------------------------------------------------------------


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    # Keep the container alive for 20 min after the last request so that
    # consecutive GSAM runs don't pay the cold-start cost every time.
    scaledown_window=20 * MINUTES,
    # Allow cold-start + model load time before Modal marks the function
    # as unhealthy.
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Accept many concurrent requests in a single container replica.
    # vLLM handles its own request batching internally.
    max_inputs=64,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """
    Start a vLLM OpenAI-compatible server.

    Modal exposes port 8000 via an HTTPS URL.  The endpoint is compatible with
    the OpenAI Python SDK so GSAM can use it without any code changes – just
    point MODAL_BASE_URL at the /v1 path of this URL.
    """
    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        # Register both the full model name and a short "llm" alias so that
        # --generator_model can be either value in run_gsam.py.
        "--served-model-name", MODEL_NAME, "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level", "info",
        "--tensor-parallel-size", str(N_GPU),
    ]
    if FAST_BOOT:
        cmd.append("--enforce-eager")
    cmd.extend(EXTRA_VLLM_ARGS)

    print(f"[gsam-vllm-server] Launching: {' '.join(cmd)}")
    subprocess.Popen(cmd)


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def test():
    """
    Smoke-test the deployed server.

    Run after deploying:
        modal run modal_vllm_server.py

    This sends a single chat-completion request and prints the response so
    you can confirm the server is healthy before running GSAM experiments.
    """
    import urllib.request
    import json
    import time

    url = serve.get_web_url()  # type: ignore[attr-defined]
    base = url.rstrip("/")

    # ------------------------------------------------------------------
    # 1. Health check
    # ------------------------------------------------------------------
    health_url = f"{base}/health"
    print(f"[test] Health check → {health_url}")
    deadline = time.time() + 5 * MINUTES
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=10) as r:
                if r.status == 200:
                    print("[test] Server healthy ✓")
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise RuntimeError("Server did not become healthy in time")

    # ------------------------------------------------------------------
    # 2. Simple chat completion
    # ------------------------------------------------------------------
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly: GSAM vLLM server OK"},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        result = json.loads(r.read())

    content = result["choices"][0]["message"]["content"]
    print(f"[test] Response: {content}")
    print(f"\n✓ Server is ready.  Set these env vars before running GSAM:")
    print(f"    MODAL_BASE_URL={base}/v1")
    print(f"    MODAL_API_KEY=dummy")
    print(f"    MODAL_MODEL_NAME={MODEL_NAME}")
