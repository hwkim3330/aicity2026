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


def _as_tensor(feats):
    """The installed transformers version's CLIPModel.get_*_features()
    unconditionally returns a BaseModelOutputWithPooling (the projected
    embedding lives in .pooler_output), not a bare tensor like older
    versions and the method's own docstring example still show. Handle
    both so this doesn't silently break again on the next transformers
    bump."""
    return feats.pooler_output if hasattr(feats, "pooler_output") else feats


@torch.no_grad()
def embed_images(model, processor, image_paths, batch_size=64):
    embeds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        feats = _as_tensor(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def embed_texts(model, processor, texts, batch_size=64):
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        feats = _as_tensor(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, dim=0)
