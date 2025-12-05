import benchmark, base64, torch, collections
from PIL import Image
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# set model here
model_dir = "Qwen/Qwen3-VL-2B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, dtype=torch.bfloat16, device_map="cuda")

processor = AutoProcessor.from_pretrained(model_dir)

def ocr(path: Path, prompt: str) -> str:
    torch.manual_seed(0)

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    base64_url = f"data:image/jpeg;base64,{data}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_url,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts[0]

benchmark.run(ocr)
