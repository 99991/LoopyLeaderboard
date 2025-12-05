# For benchmarking Qwen3-VL-4B with llama server, download models from
# https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/tree/main
# and run the following command
"""
llama-server \
    --jinja \
    --n-gpu-layers 999 \
    --flash-attn auto \
    --ctx-size 1024 \
    --mmproj mmproj-BF16.gguf \
    --model Qwen3-VL-4B-Instruct-Q5_K_M.gguf \
    --port 8000
"""

import benchmark, requests, json, base64, time, warnings
from pathlib import Path

# set url to OpenAI-compatible endpoint here
api_url = "http://127.0.0.1:8000/v1/chat/completions"

options = {
    "temperature": 0,
    "seed": 0,
    "max_tokens": 256,
}

def ocr(path: Path, prompt: str, max_retries: int=20) -> str:
    ext = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}[path.suffix.lower()]

    image_bytes = path.read_bytes()
    base64_data = base64.b64encode(image_bytes).decode("ascii")

    data = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/{ext};base64," + base64_data}
                },
                {"type": "text", "text": prompt}
            ]}
        ],
        **options,
    }

    sleep_time = 0.1
    for _ in range(max_retries):
        with requests.post(api_url, json=data) as r:
            # On error, wait a bit and retry
            if r.status_code != 200:
                warnings.warn(f"Request failed for {path}, prompt {repr(prompt)}, retry in {sleep_time} sec")
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 10) # exponential backoff
                continue

            return r.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"Request failed {max_retries} times in a row")

benchmark.run(ocr)
