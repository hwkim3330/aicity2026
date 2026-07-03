"""Shared CLIP loading + embedding helpers for the Track 4 retrieval baseline."""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(model_name_or_path=MODEL_NAME):
    model = CLIPModel.from_pretrained(model_name_or_path).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    return model, processor


@torch.no_grad()
def embed_images(model, processor, image_paths, batch_size=64):
    embeds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def embed_texts(model, processor, texts, batch_size=64):
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, dim=0)
