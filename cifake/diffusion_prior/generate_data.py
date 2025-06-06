#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
INPUT_DIR  = Path("../data/train/REAL")
OUTPUT_DIR = Path("outputs")
STEPS      = [1, 2, 3, 4]                # generate 1 through 4 diffusion steps
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def pad_to_multiple_of_8(img: Image.Image) -> Image.Image:
    """Pad right/bottom so width & height are divisible by 8."""
    w, h = img.size
    pad_w = (8 - w % 8) % 8
    pad_h = (8 - h % 8) % 8
    return transforms.Pad((0, 0, pad_w, pad_h))(img)

to_tensor = transforms.ToTensor()  # [0,255]→[0,1]

def to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a [-1,1] torch tensor (1×3×H×W) back to a PIL image."""
    img = (x.clamp(-1,1) + 1) * 0.5
    arr = img.cpu().permute(0,2,3,1).numpy()[0]
    arr = (arr * 255).round().astype("uint8")
    return Image.fromarray(arr)

# ─── PREPARE OUTPUT DIRECTORIES ─────────────────────────────────────────────────

(OUTPUT_DIR / "original").mkdir(parents=True, exist_ok=True)
for step in STEPS:
    (OUTPUT_DIR / f"step_{step}").mkdir(exist_ok=True)

# ─── LOAD VAE & SCHEDULER ────────────────────────────────────────────────────────

vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")\
       .to(DEVICE).half()
scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
scheduler.set_timesteps(max(STEPS), device=DEVICE)

# ─── GENERATE & SAVE ─────────────────────────────────────────────────────────────

for img_path in tqdm(list(INPUT_DIR.iterdir()), desc="Processing images"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    # 1) load, pad, and save original
    img = Image.open(img_path).convert("RGB")
    img = pad_to_multiple_of_8(img)
    img.save(OUTPUT_DIR / "original" / img_path.name)

    # 2) convert to tensor [-1,1]
    x = to_tensor(img).unsqueeze(0).to(DEVICE) * 2 - 1  # shape: [1,3,H,W]

    # 3) encode to latent once
    with torch.no_grad(), torch.cuda.amp.autocast():
        latents = vae.encode(x.half()).latent_dist.sample()
        latents = latents * scheduler.init_noise_sigma

    # 4) for each diffusion step, add noise & decode
    for step in STEPS:
        t     = torch.tensor([step], dtype=torch.long, device=DEVICE)
        noise = torch.randn_like(latents)
        with torch.no_grad(), torch.cuda.amp.autocast():
            noisy_lat = scheduler.add_noise(latents, noise, timesteps=t)
            decoded   = vae.decode(noisy_lat / vae.config.scaling_factor).sample

        # 5) convert back to PIL and save
        pil = to_pil(decoded)
        pil.save(OUTPUT_DIR / f"step_{step}" / img_path.name)

        # cleanup GPU
        del t, noise, noisy_lat, decoded
        torch.cuda.empty_cache()

    # cleanup per-image GPU tensors
    del x, latents
    torch.cuda.empty_cache()

print("Done! Originals in 'outputs/original', noisy images in 'outputs/step_1'–'outputs/step_4'.")
