import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict

class CLIPEmbedder:
    """
    CLIP-based image and text embedder.
    Produces 768-dim normalized vectors for ChromaDB storage.
    This is the embedding backbone of the RAG retrieval node.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"[CLIPEmbedder] Loading {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("[CLIPEmbedder] Ready")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single image → 768-dim normalized vector."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            embedding = outputs.pooler_output
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query → 768-dim normalized vector."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze().numpy()

    def similarity(self, image: Image.Image, texts: List[str]) -> Dict[str, float]:
        """Score an image against multiple text queries."""
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        return dict(zip(texts, probs.tolist()))
