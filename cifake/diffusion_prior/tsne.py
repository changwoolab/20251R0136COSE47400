#!/usr/bin/env python3
import random
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("../data/train")    # must contain “REAL” & “FAKE”
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME        = "CompVis/stable-diffusion-v1-4"
N_SAMPLES         = 1000
OUTPUT_DIR        = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
FINETUNED_VAE_CKPT = "vae_recon_finetuned.pt"

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def pad_to_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h   = img.size
    pad_w  = (8 - w % 8) % 8
    pad_h  = (8 - h % 8) % 8
    return transforms.Pad((0, 0, pad_w, pad_h))(img)

to_tensor = transforms.ToTensor()  # [0,255]→[0,1]

# ─── SAMPLE DATA ─────────────────────────────────────────────────────────────────
dataset   = ImageFolder(DATA_DIR, transform=None)
real_id   = dataset.class_to_idx["REAL"]
fake_id   = dataset.class_to_idx["FAKE"]

real_idxs = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == real_id]
fake_idxs = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == fake_id]

real_sel  = real_idxs[:N_SAMPLES]
fake_sel  = fake_idxs[:N_SAMPLES]
subset    = real_sel + fake_sel

# ─── LOAD (AND OPTIONALLY FINETUNED) VAE ENCODER ────────────────────────────────
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae") \
       .to(DEVICE).eval()

print(f"Loading finetuned VAE from {FINETUNED_VAE_CKPT}")
state = torch.load(FINETUNED_VAE_CKPT, map_location=DEVICE)
vae.load_state_dict(state, strict=True)

# ─── EXTRACT LATENTS ─────────────────────────────────────────────────────────────
latents = []
labels  = []

with torch.no_grad():
    for idx in subset:
        path, lbl = dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        img = pad_to_multiple_of_8(img)
        x   = to_tensor(img).unsqueeze(0).to(DEVICE) * 2 - 1  # [-1,1]

        enc    = vae.encode(x)                  # fp32 encode
        latent = enc.latent_dist.mean           # [1,4,H/8,W/8]
        vec    = latent.cpu().view(-1).numpy()  # flatten
        latents.append(vec)
        labels.append(lbl)

latents = np.stack(latents, axis=0)          # [N, latent_dim]

# ─── t-SNE & PLOT ────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, init="pca", random_state=42)
proj = tsne.fit_transform(latents)           # [N,2]

plt.figure(figsize=(8,8))
colors = ["red" if l==real_id else "blue" for l in labels]
plt.scatter(proj[:,0], proj[:,1], c=colors, alpha=0.7, s=25)
plt.legend(handles=[
    plt.Line2D([0],[0], marker='o', color='w', label='REAL',
               markerfacecolor='red', markersize=2),
    plt.Line2D([0],[0], marker='o', color='w', label='FAKE',
               markerfacecolor='blue', markersize=2),
])
plt.title("t-SNE of SD VAE Latents: Real (red) vs. Fake (blue)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()

# ─── SAVE TO FILE ────────────────────────────────────────────────────────────────
save_path = OUTPUT_DIR / "tsne_plot.png"
plt.savefig(save_path)
print(f"Saved t-SNE plot to {save_path}")
