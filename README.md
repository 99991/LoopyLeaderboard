<img width="867" height="351" alt="loopy_banner" src="https://github.com/user-attachments/assets/d13b48bd-900c-42b1-93cf-971606873324" />

<hr>

While todays Vision Language Models (VLMs) exhibit impressive Optical Character Recognition (OCR) capabilities,
they often fail catastrophically, repeating the same text over and over again.
[[1]](https://github.com/QwenLM/Qwen3-VL/issues/1611)
[[2]](https://github.com/QwenLM/Qwen3-VL/issues/241)
[[3]](https://github.com/deepseek-ai/DeepSeek-OCR/issues/151)
[[4]](https://github.com/Blaizzy/mlx-vlm/issues/549)
[[5]](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/89)
[[6]](https://huggingface.co/PaddlePaddle/PaddleOCR-VL/discussions/73)

This repository is an attempt at quantifying how often VLMs fall into this repetition trap while performing OCR.

### Evaluation Methodology

The VLM is tasked to generate up to 256 tokens for each of 125 test images with 4 different prompts, resulting in 500 OCR outputs.
Each output is checked for repeating strings.
The output is considered repeating if it contains at least 40 consecutive characters that repeat at least twice.
For example, the text `"The image shows the the the the the the the the the the the"` is considered repeating,
because the substring `" the"` of length 4 is repeated 11 times for a total of 44 characters, which is more than 40.

Finally, the outputs with repeats are counted and a loopy score is calculated, which is the number of repeated outputs divided by the total number of outputs.
For example, if 200 of 500 outputs are repeating, the loopy score is `200 / 500 = 40 %`.
A loopy score of 0 % is considered optimal, while a loopy score of 100 % is complete garbage.

### Running the Benchmark

The file `test_qwen3.py` will download the test images, perform OCR with [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and compute the loopy score.

```bash
# install dependencies
pip install requests torch transformers accelerate pillow
# run benchmark
python3 test_qwen3.py
```

For testing OpenAI-compatible APIs, adjust the URL in `test_api.py` accordingly. For [GGUFs](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/blob/main/Qwen3-VL-2B-Instruct-BF16.gguf), the following [llama-server](https://github.com/ggml-org/llama.cpp/releases/tag/b7278) command was used:

```bash
llama-server \
    --jinja \
    --n-gpu-layers 999 \
    --flash-attn auto \
    --ctx-size 1024 \
    --mmproj mmproj-BF16.gguf \
    --model Qwen3-VL-2B-Instruct-BF16.gguf \
    --port 8000
```

All experiments use the corresponding [BF16 mmproj](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/blob/main/mmproj-BF16.gguf).

### Results

| Model | # Loopy Outputs | Loopy score (smaller is better) |
|-|-|-|
| [Qwen3-VL-2B, BF16](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/blob/main/Qwen3-VL-2B-Instruct-BF16.gguf) (greedy decoding) | 281 | 56.2 % |
| [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (greedy decoding) | 256 | 51.2 % |
| [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (greedy decoding) | 244 | 48.8 % |
| [Qwen3-VL-4B, Q5_K_M](https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/blob/main/Qwen3-VL-4B-Instruct-Q5_K_M.gguf) (greedy decoding) | 223 | 44.6 % |
| [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (non-greedy decoding) | 222 | 44.4 % |
| [Qwen3-VL-4B, BF16](https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/blob/main/Qwen3-VL-4B-Instruct-BF16.gguf) (greedy decoding) | 220 | 44.0 % |
| [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (non-greedy decoding) | 217 | 43.4 % |
| [Qwen3-VL-8B, Q5_K_M](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/blob/main/Qwen3-VL-8B-Instruct-Q5_K_M.gguf) (greedy decoding) | 142 | 28.4 % |
| [Qwen3-VL-30B-A3B, Q4_K_S](https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF/blob/main/Qwen3-VL-30B-A3B-Instruct-Q4_K_S.gguf) (greedy decoding) | 116 | 23.2 % |

You can submit your own results. Just [send a pull request](https://github.com/99991/LoopyLeaderboard/pulls).

### Considerations

* A low loopy score does not necessarily reflect whether the output of a VLM is any good.
For example, a model that always produces the empty string achieves a perfect loopy score of 0 %.
For general OCR capabilites, consider an OCR benchmark, for example [OCRBench v2](https://huggingface.co/spaces/ling99/OCRBench-v2-leaderboard).
* Repeat penalties are not a satisfying solution to the repetition problem,
because they destroy the capability of VLMs to correctly OCR images with naturally repeating text, such as scans of tables.
