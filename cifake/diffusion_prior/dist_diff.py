#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import random

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# Define where your REAL and FAKE images for analysis are located
# This directory should contain subfolders named "REAL" and "FAKE"
ANALYSIS_DATA_DIR = Path("../data/train") # Or any other directory like "../data/analysis_set"
# Ensure ANALYSIS_DATA_DIR has "REAL" and "FAKE" subfolders.

BATCH_SIZE    = 1     # Adjust based on GPU memory for inversion
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME    = "CompVis/stable-diffusion-v1-4"
IMG_SIZE      = 224
NUM_INVERSION_STEPS = 30 # Number of DDIM steps for z0 -> zT inversion
N_SAMPLES_PER_CLASS_FOR_ANALYSIS = 100 # Number of samples from each class to analyze

# ─── IMAGE TRANSFORMS ────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2 - 1),   # →[-1,1]
])

# ─── DATALOADERS FOR ANALYSIS ───────────────────────────────────────────────────
analysis_ds = ImageFolder(ANALYSIS_DATA_DIR, transform=transform)
print(f"Dataset classes found: {analysis_ds.classes}")
print(f"Class to index mapping: {analysis_ds.class_to_idx}")

# Get indices for REAL and FAKE images
try:
    real_class_idx = analysis_ds.class_to_idx["REAL"]
    fake_class_idx = analysis_ds.class_to_idx["FAKE"]
except KeyError as e:
    print(f"Error: Class {e} not found in {ANALYSIS_DATA_DIR}. Ensure subfolders are named 'REAL' and 'FAKE'.")
    exit()

real_indices = [i for i, label in enumerate(analysis_ds.targets) if label == real_class_idx]
fake_indices = [i for i, label in enumerate(analysis_ds.targets) if label == fake_class_idx]

# Shuffle and select N_SAMPLES_PER_CLASS_FOR_ANALYSIS
random.shuffle(real_indices)
random.shuffle(fake_indices)

real_indices_subset = real_indices[:min(len(real_indices), N_SAMPLES_PER_CLASS_FOR_ANALYSIS)]
fake_indices_subset = fake_indices[:min(len(fake_indices), N_SAMPLES_PER_CLASS_FOR_ANALYSIS)]

if not real_indices_subset:
    print("Warning: No REAL images selected for analysis. Check dataset and N_SAMPLES_PER_CLASS_FOR_ANALYSIS.")
if not fake_indices_subset:
    print("Warning: No FAKE images selected for analysis. Check dataset and N_SAMPLES_PER_CLASS_FOR_ANALYSIS.")

real_subset_ds = Subset(analysis_ds, real_indices_subset)
fake_subset_ds = Subset(analysis_ds, fake_indices_subset)

real_loader = DataLoader(real_subset_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
fake_loader = DataLoader(fake_subset_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Analyzing {len(real_subset_ds)} REAL images and {len(fake_subset_ds)} FAKE images.")

# ─── LOAD PRE-TRAINED MODELS (All Frozen) ───────────────────────────────────────
# VAE (pre-trained, frozen)
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(torch.float32)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
vae_scale_factor = vae.config.scaling_factor
print("VAE loaded and frozen.")

# UNet (pre-trained, frozen, for DDIM inversion)
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(DEVICE).to(torch.float32)
unet.eval()
for param in unet.parameters():
    param.requires_grad = False
print("UNet loaded and frozen.")

# Scheduler
scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

# Tokenizer and Text Encoder for unconditional embeddings (frozen)
try:
    pipe_for_embeds = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    tokenizer = pipe_for_embeds.tokenizer
    text_encoder = pipe_for_embeds.text_encoder.to(DEVICE).eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    del pipe_for_embeds
    print("Tokenizer and Text Encoder loaded and frozen.")
except Exception as e:
    print(f"Could not load Tokenizer/Text Encoder: {e}. Exiting.")
    exit()

# ─── PREPARE UNCONDITIONAL EMBEDDINGS ───────────────────────────────────────────
@torch.no_grad()
def get_uncond_embeddings(tokenizer_model, text_encoder_model, device_type, model_dtype):
    prompt = "" # Unconditional
    text_inputs = tokenizer_model(
        prompt,
        padding="max_length",
        max_length=tokenizer_model.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    embeddings = text_encoder_model(text_inputs.input_ids.to(device_type))[0]
    return embeddings.to(dtype=model_dtype)

uncond_embeddings = get_uncond_embeddings(tokenizer, text_encoder, DEVICE, unet.dtype)

# ─── DDIM ENCODING FUNCTION (z0 to zT) ─────────────────────────────────────────
@torch.no_grad() # Ensure no gradients are computed anywhere in this analysis function
def ddim_encode_z0_to_zT(z0_scaled, unet_model, ddim_scheduler, num_steps, unconditional_embeddings, device_type, model_dtype):
    if z0_scaled.shape[0] != unconditional_embeddings.shape[0] and unconditional_embeddings.shape[0] == 1:
        batch_uncond_embeddings = unconditional_embeddings.expand(z0_scaled.shape[0], -1, -1)
    else:
        batch_uncond_embeddings = unconditional_embeddings

    xt = z0_scaled.clone().to(dtype=model_dtype)
    ddim_scheduler.set_timesteps(num_steps, device=device_type)
    timesteps_to_iterate = reversed(ddim_scheduler.timesteps)

    for i, t_unet_eval in enumerate(timesteps_to_iterate):
        alpha_prod_t = ddim_scheduler.alphas_cumprod[t_unet_eval]
        if t_unet_eval == ddim_scheduler.timesteps[0]:
            alpha_prod_t_next = ddim_scheduler.final_alpha_cumprod if \
                                hasattr(ddim_scheduler, "final_alpha_cumprod") and \
                                ddim_scheduler.final_alpha_cumprod is not None \
                                else torch.tensor(0.0, device=device_type, dtype=xt.dtype)
        else:
            current_original_idx = (ddim_scheduler.timesteps == t_unet_eval).nonzero(as_tuple=True)[0].item()
            t_next_original_val = ddim_scheduler.timesteps[current_original_idx - 1]
            alpha_prod_t_next = ddim_scheduler.alphas_cumprod[t_next_original_val]

        eps = unet_model(xt, t_unet_eval.unsqueeze(0) if t_unet_eval.ndim == 0 else t_unet_eval,
                         encoder_hidden_states=batch_uncond_embeddings).sample

        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_prod_t = (1.0 - alpha_prod_t).sqrt()
        pred_x0 = (xt - sqrt_one_minus_alpha_prod_t * eps) / (sqrt_alpha_prod_t + 1e-8)
        sqrt_alpha_prod_t_next = alpha_prod_t_next.sqrt()
        sqrt_one_minus_alpha_prod_t_next = (1.0 - alpha_prod_t_next).sqrt()
        xt = sqrt_alpha_prod_t_next * pred_x0 + sqrt_one_minus_alpha_prod_t_next * eps
    return xt

# ─── ANALYSIS FUNCTION ───────────────────────────────────────────────────────────
def analyze_inverted_latents(data_loader, class_name, vae_model, unet_model, ddim_scheduler,
                             num_inv_steps, uncond_embeds, device_type, vae_sf, model_dtype_unet):
    all_mean_diffs = []
    all_std_diffs = []

    progress_bar = tqdm(data_loader, desc=f"Analyzing {class_name} images")
    for batch in progress_bar:
        imgs, _ = batch # Labels are not used for this analysis metric
        imgs = imgs.to(device_type, dtype=vae_model.dtype)

        # 1. VAE Encode to get z0 (unscaled)
        z0_unscaled_dist = vae_model.encode(imgs)
        z0_unscaled = z0_unscaled_dist.latent_dist.mean

        # 2. Scale z0 for DDIM processing
        z0_scaled = z0_unscaled * vae_sf

        # 3. DDIM Encode (z0_scaled -> zT)
        inverted_zt_batch = ddim_encode_z0_to_zT(
            z0_scaled, unet_model, ddim_scheduler, num_inv_steps,
            uncond_embeds, device_type, model_dtype_unet
        )

        # 4. Calculate difference from N(0,1) for each zT in the batch
        for single_zt in inverted_zt_batch: # Iterate over batch dimension
            flattened_zt = single_zt.flatten()
            
            mean_val = torch.mean(flattened_zt)
            std_val = torch.std(flattened_zt)
            
            mean_diff = torch.abs(mean_val - 0.0)
            std_diff = torch.abs(std_val - 1.0)
            
            all_mean_diffs.append(mean_diff.item())
            all_std_diffs.append(std_diff.item())

    if not all_mean_diffs: # Handle empty dataloader case
        return 0.0, 0.0
        
    avg_mean_diff = np.mean(all_mean_diffs)
    avg_std_diff = np.mean(all_std_diffs)
    
    return avg_mean_diff, avg_std_diff

# ─── PERFORM ANALYSIS ────────────────────────────────────────────────────────────
print(f"\nStarting analysis with {NUM_INVERSION_STEPS} DDIM inversion steps...")
print(f"Device: {DEVICE}")

# Analyze REAL images
if len(real_loader.dataset) > 0:
    avg_mean_diff_real, avg_std_diff_real = analyze_inverted_latents(
        real_loader, "REAL", vae, unet, scheduler, NUM_INVERSION_STEPS,
        uncond_embeddings, DEVICE, vae_scale_factor, unet.dtype
    )
    print(f"\n--- Results for REAL Images ({len(real_loader.dataset)} samples) ---")
    print(f"Average Absolute Difference of zT Mean from 0: {avg_mean_diff_real:.6f}")
    print(f"Average Absolute Difference of zT Std from 1:  {avg_std_diff_real:.6f}")
else:
    print("\nNo REAL images to analyze.")

# Analyze FAKE images
if len(fake_loader.dataset) > 0:
    avg_mean_diff_fake, avg_std_diff_fake = analyze_inverted_latents(
        fake_loader, "FAKE", vae, unet, scheduler, NUM_INVERSION_STEPS,
        uncond_embeddings, DEVICE, vae_scale_factor, unet.dtype
    )
    print(f"\n--- Results for FAKE Images ({len(fake_loader.dataset)} samples) ---")
    print(f"Average Absolute Difference of zT Mean from 0: {avg_mean_diff_fake:.6f}")
    print(f"Average Absolute Difference of zT Std from 1:  {avg_std_diff_fake:.6f}")
else:
    print("\nNo FAKE images to analyze.")

print("\nAnalysis complete.")