from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test Qwen2.5-VL on one rendered PDF page image.")
    p.add_argument("--image", required=True, help="Path to a page image (PNG/JPG)")
    p.add_argument("--prompt", default="Extract key fields you can read. If unreadable, say [NIECZYTELNE].")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "mps" else "auto",
    )
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)


if __name__ == "__main__":
    main()

