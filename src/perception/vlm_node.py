import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Optional


class VLMPerceptionNode:
    """
    Qwen2-VL based perception node.
    Takes an image + query, returns a grounded visual description.
    This is the first node in the LangGraph pipeline.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", max_image_size: int = 512):
        print(f"[VLMPerceptionNode] Loading {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_image_size = max_image_size
        print("[VLMPerceptionNode] Ready")

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Resize image to avoid OOM — Qwen2-VL is sensitive to resolution."""
        return image.resize((self.max_image_size, self.max_image_size))

    def perceive(self, image: Image.Image, query: str, max_new_tokens: int = 200) -> str:
        """
        Run VLM inference on an image with a text query.
        Returns the model's textual description/answer.
        """
        image = self._prepare_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
