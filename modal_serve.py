"""
modal_serve.py — Self-hosted DeepSeek on Modal.com via vLLM

Deploys a DeepSeek model as an OpenAI-compatible inference endpoint.
The endpoint works as a drop-in replacement for the SambaNova/Together APIs
already used in this project.

────────────────────────────────────────────────────────
QUICK START
────────────────────────────────────────────────────────

1. Install Modal and authenticate:
        pip install modal
        modal setup          # opens browser, one-time login

2. (If you need a HuggingFace token for model access)
        modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN_HERE

3. Deploy:
        modal deploy modal_serve.py

   The CLI will print something like:
        App deployed. URL: https://YOUR_WORKSPACE--gsam-deepseek-serve.modal.run

4. Set the endpoint in your .env file:
        MODAL_API_URL=https://YOUR_WORKSPACE--gsam-deepseek-serve.modal.run/v1

5. Run experiments with --api_provider modal
   (see MODEL_NAME below for which model name to pass as --generator_model)

────────────────────────────────────────────────────────
MODEL CHOICE — edit MODEL_NAME below
────────────────────────────────────────────────────────

DeepSeek-R1-Distill-Qwen-32B   → 2 × H100  (~$8/hr)    good quality, practical
DeepSeek-R1-Distill-Llama-70B  → 4 × H100  (~$16/hr)   closer to V3 quality
deepseek-ai/DeepSeek-V3-0324   → 8 × H100  (~$32/hr)   matches ACE paper exactly

Default is the 32B distill — cheapest option that still gives good results.
Change N_GPU and the tensor-parallel-size accordingly (must match).

The served model name (what you pass as --generator_model in run commands)
is whatever MODEL_NAME is set to here.
"""

import modal

# ---------------------------------------------------------------------------
# Model selection — edit these two lines to change the model
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
N_GPU = 2          # must match tensor-parallel-size in serve() below

# ---------------------------------------------------------------------------
# Container image — vLLM + HuggingFace Hub on CUDA 12.8
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.8.5",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ---------------------------------------------------------------------------
# Persistent volume caches — weights are downloaded once and reused
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MINUTES = 60
VLLM_PORT = 8000

app = modal.App("gsam-deepseek")


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    # Keep warm for 30 min between requests to avoid cold-start delays mid-experiment.
    # Lower this if you want to save cost when not running experiments.
    scaledown_window=30 * MINUTES,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Add the HuggingFace secret if your model is gated.
    # DeepSeek models on HuggingFace are public so this is optional,
    # but it speeds up downloads via authenticated requests.
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        # Keep eager mode on so the server starts faster.
        # Remove this flag (or set to --no-enforce-eager) for better throughput
        # once you have confirmed the deployment works.
        "--enforce-eager",
        "--tensor-parallel-size", str(N_GPU),
        # Limit context length to reduce memory pressure.
        # The financial tasks in this project use short contexts.
        "--max-model-len", "8192",
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# ---------------------------------------------------------------------------
# Optional: local test entrypoint
# Run with:  modal run modal_serve.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def test():
    import urllib.request
    import json

    url = serve.get_web_url()
    print(f"\nEndpoint URL: {url}")
    print(f"Set in .env:  MODAL_API_URL={url}/v1\n")

    # Health check
    health_url = f"{url}/health"
    print(f"Checking health at {health_url} ...")
    try:
        with urllib.request.urlopen(health_url, timeout=300) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed!\n")
    except Exception as e:
        print(f"Health check failed: {e}\n")
        return

    # Quick inference test
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 64,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    print("Test response:", result["choices"][0]["message"]["content"])
    print("\nDeployment successful. Use these settings in experiments:")
    print(f"  --api_provider modal")
    print(f"  --generator_model {MODEL_NAME}")
    print(f"  --reflector_model {MODEL_NAME}")
    print(f"  --curator_model   {MODEL_NAME}")
