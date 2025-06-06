#!/usr/bin/env python3
import random
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from transformers import ViTFeatureExtractor, ViTModel

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("../data/train")    # must contain “REAL” & “FAKE”
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME        = "dima806/ai_vs_real_image_detection"
N_SAMPLES         = 1000
OUTPUT_DIR        = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── LOAD ViT FEATURE EXTRACTOR & MODEL ─────────────────────────────────────────
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
vit_model         = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# ─── PREPARE DATASET ─────────────────────────────────────────────────────────────
dataset   = ImageFolder(DATA_DIR, transform=None)
real_id   = dataset.class_to_idx["REAL"]
fake_id   = dataset.class_to_idx["FAKE"]

real_idxs = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == real_id]
fake_idxs = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == fake_id]

real_sel  = real_idxs[:N_SAMPLES]
fake_sel  = fake_idxs[:N_SAMPLES]
subset    = real_sel + fake_sel

# ─── EXTRACT ViT [CLS] EMBEDDINGS ────────────────────────────────────────────────
latents = []
labels  = []

with torch.no_grad():
    for idx in subset:
        path, lbl = dataset.samples[idx]
        img = Image.open(path).convert("RGB")

        # Use feature extractor to resize, center-crop, and normalize the image
        inputs = feature_extractor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)  # shape [1, 3, 224, 224]

        outputs = vit_model(pixel_values=pixel_values)
        # outputs.last_hidden_state has shape [1, seq_len, hidden_dim]
        cls_emb = outputs.last_hidden_state[:, 0, :]      # [CLS] token embedding, shape [1, hidden_dim]
        vec     = cls_emb.cpu().numpy().reshape(-1)       # flatten to 1D array
        latents.append(vec)
        labels.append(lbl)

latents = np.stack(latents, axis=0)  # shape [N, hidden_dim]

# ─── t-SNE & PLOT ────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, init="pca", random_state=42)
proj = tsne.fit_transform(latents)  # shape [N, 2]

plt.figure(figsize=(8, 8))
colors = ["red" if l == real_id else "blue" for l in labels]
plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.7, s=25)
plt.legend(
    handles=[
        plt.Line2D([0], [0], marker="o", color="w", label="REAL", markerfacecolor="red", markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="FAKE", markerfacecolor="blue", markersize=6),
    ]
)
plt.title("t-SNE of ViT [CLS] Embeddings: Real (red) vs. Fake (blue)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()

# ─── SAVE TO FILE ───────────────────────────────────────────────────────────────
save_path = OUTPUT_DIR / "tsne_vit_plot.png"
plt.savefig(save_path)
print(f"Saved t-SNE plot to {save_path}")
