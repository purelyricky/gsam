"""
modal_serve.py — OpenAI-compatible DeepSeek endpoint via vLLM on Modal
=======================================================================

QUICK START
-----------
1.  pip install modal && modal setup          (one-time login)
2.  modal deploy modal_serve.py              (persistent deploy — do NOT use modal run)
3.  Copy the printed URL and set in .env:
        MODAL_API_URL=https://bata24941--gsam-deepseek-serve-deepseekserver-serve.modal.run/v1
4.  Run experiments:  bash run_gsam_experiments_mini.sh

OPTIONAL: pre-download model weights into the volume (saves cold-start time):
    modal run modal_serve.py::download_model

MODEL OPTIONS (edit MODEL_NAME + N_GPUS below)
-----------------------------------------------
  Model                                        N_GPUS   GPU        $/hr    Notes
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B       1      A100-40    ~$3    Fastest/cheapest
  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B      1      A100-80    ~$4    Good quality
  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B      2      H100       ~$8    DEFAULT — matches GSAM paper
  deepseek-ai/DeepSeek-R1-Distill-Llama-70B     4      H100       ~$16   Higher quality
  deepseek-ai/DeepSeek-V3-0324                  8      H100       ~$32   Matches ACE paper exactly

HUGGING FACE TOKENS
-------------------
R1-Distill-* models are public — no HF token required.
DeepSeek-V3-0324 and DeepSeek-R1 (671B) require accepting terms on HF
and setting a token: modal secret create huggingface-secret HF_TOKEN=hf_xxx
"""

import modal
import subprocess
import time

# ---------------------------------------------------------------------------
# CONFIGURE YOUR MODEL HERE
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# For final paper runs: DeepSeek-R1-Distill-Qwen-32B, N_GPUS=2, GPU_TYPE="H100"
N_GPUS   = 4
GPU_TYPE = "H100"   # H100 | A100-40GB | A100-80GB
# ---------------------------------------------------------------------------

VLLM_PORT            = 8000
SCALEDOWN_WINDOW     = 1200   # 20 min — avoids cold starts mid-experiment
MAX_MODEL_LEN        = 32768
GPU_MEMORY_UTIL      = 0.92

APP_NAME = "gsam-deepseek-serve"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system vllm>=0.6.0 huggingface_hub>=0.24.0 hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("gsam-model-cache", create_if_missing=True)
CACHE_DIR   = "/root/.cache/huggingface/hub"

SECRETS = []
if "V3" in MODEL_NAME or MODEL_NAME.split("/")[-1] == "R1":
    SECRETS = [modal.Secret.from_name("huggingface-secret")]


# ---------------------------------------------------------------------------
# Pre-download model weights into the Volume (run once before deploying)
# Usage: modal run modal_serve.py::download_model
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    volumes={CACHE_DIR: model_cache},
    secrets=SECRETS,
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_NAME} ...")
    snapshot_download(MODEL_NAME, ignore_patterns=["*.pt", "*.bin"])
    model_cache.commit()
    print("Download complete.")


# ---------------------------------------------------------------------------
# vLLM server
# ---------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={CACHE_DIR: model_cache},
    secrets=SECRETS,
    timeout=3600,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=64)
class DeepSeekServer:

    @modal.enter()
    def start_server(self):
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  MODEL_NAME,
            "--port",                   str(VLLM_PORT),
            "--host",                   "0.0.0.0",
            "--tensor-parallel-size",   str(N_GPUS),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTIL),
            "--max-model-len",          str(MAX_MODEL_LEN),
            "--served-model-name",      MODEL_NAME,
            "--enable-prefix-caching",
            "--no-enable-log-requests",
        ]
        if "R1" in MODEL_NAME or "r1" in MODEL_NAME.lower():
            cmd += ["--reasoning-parser", "deepseek_r1"]

        self._proc = subprocess.Popen(cmd)

        import urllib.request, urllib.error
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health")
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(3)
        else:
            raise RuntimeError("vLLM server did not start within 10 minutes")

        print(f"vLLM ready — model={MODEL_NAME}  gpus={N_GPUS}x{GPU_TYPE}")

    @modal.exit()
    def stop_server(self):
        self._proc.terminate()
        self._proc.wait(timeout=30)

    @modal.web_server(port=VLLM_PORT, startup_timeout=600)
    def serve(self):
        pass


