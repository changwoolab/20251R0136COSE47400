#!/usr/bin/env python3
import random
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL, StableDiffusionPipeline, DDIMScheduler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("../data/train")    # must contain “REAL” & “FAKE”
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME        = "CompVis/stable-diffusion-v1-4"
N_SAMPLES         = 100 # Reduce N_SAMPLES for faster testing, increase later
OUTPUT_DIR        = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
FINETUNED_VAE_CKPT = "vae_recon_finetuned.pt" # Set to None if not using a finetuned VAE
NUM_INVERSION_STEPS = 50 # Number of DDIM steps for inversion (z0 -> zT)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def pad_to_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h   = img.size
    pad_w  = (8 - w % 8) % 8
    pad_h  = (8 - h % 8) % 8
    # Pillow's Pad is (left, top, right, bottom)
    return transforms.Pad((0, 0, pad_w, pad_h))(img)

to_tensor = transforms.ToTensor()  # [0,255]→[0,1]

# ─── SAMPLE DATA ─────────────────────────────────────────────────────────────────
dataset   = ImageFolder(DATA_DIR, transform=None)
try:
    real_id   = dataset.class_to_idx["REAL"]
    fake_id   = dataset.class_to_idx["FAKE"]

    real_idxs = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == real_id]
    fake_idxs = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == fake_id]

    # Ensure N_SAMPLES does not exceed available images for each class
    n_real_available = len(real_idxs)
    n_fake_available = len(fake_idxs)
    
    current_n_samples = min(N_SAMPLES, n_real_available, n_fake_available)
    if current_n_samples < N_SAMPLES:
        print(f"Warning: Requested N_SAMPLES={N_SAMPLES}, but only {current_n_samples} available per class. Using {current_n_samples}.")

    if current_n_samples == 0:
        raise ValueError("No samples found for one or both classes. Check DATA_DIR and class names ('REAL', 'FAKE').")

    # Shuffle indices before selecting to get varied samples if N_SAMPLES is small
    random.shuffle(real_idxs)
    random.shuffle(fake_idxs)

    real_sel  = real_idxs[:current_n_samples]
    fake_sel  = fake_idxs[:current_n_samples]
    subset    = real_sel + fake_sel
    print(f"Selected {len(real_sel)} REAL and {len(fake_sel)} FAKE samples.")

except KeyError as e:
    raise ValueError(
        f"Class '{e.args[0]}' not found in {DATA_DIR}. "
        "Ensure subdirectories are named 'REAL' and 'FAKE'."
    ) from e
except ValueError as e:
    print(f"Error during data sampling: {e}")
    exit()


# ─── LOAD VAE (Potentially Fine-tuned) ──────────────────────────────────────────
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
if FINETUNED_VAE_CKPT and Path(FINETUNED_VAE_CKPT).exists():
    print(f"Loading finetuned VAE from {FINETUNED_VAE_CKPT}")
    try:
        state = torch.load(FINETUNED_VAE_CKPT, map_location=DEVICE)
        # Adjust for potential key mismatches if state_dict was saved directly from a model
        if 'state_dict' in state:
            state = state['state_dict']
        
        # Filter out unexpected keys if any (e.g. from lightning checkpoints)
        vae_state_dict = vae.state_dict()
        filtered_state = {k: v for k, v in state.items() if k in vae_state_dict and v.shape == vae_state_dict[k].shape}
        missing_keys, unexpected_keys = vae.load_state_dict(filtered_state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in VAE state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in VAE state_dict: {unexpected_keys}")
        print("Fine-tuned VAE loaded.")
    except Exception as e:
        print(f"Could not load fine-tuned VAE: {e}. Using pre-trained VAE.")
elif FINETUNED_VAE_CKPT:
    print(f"Fine-tuned VAE checkpoint {FINETUNED_VAE_CKPT} not found. Using pre-trained VAE.")
else:
    print("No fine-tuned VAE checkpoint specified. Using pre-trained VAE.")

vae = vae.to(DEVICE).eval()

# ─── LOAD STABLE DIFFUSION PIPELINE (for UNet and Scheduler) ────────────────────
# Use the loaded VAE in the pipeline
# Using float32 for stability, can change to torch.float16 for speed if GPU supports
dtype = torch.float32
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    vae=vae,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(DEVICE)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
unet = pipeline.unet.eval() # Ensure UNet is in eval mode
scheduler = pipeline.scheduler

# ─── GET UNCONDITIONAL EMBEDDINGS (once) ────────────────────────────────────────
prompt = "" # Unconditional
text_inputs = pipeline.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipeline.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(DEVICE))[0]
uncond_embeddings = text_embeddings.to(dtype=unet.dtype)


# ─── EXTRACT DDIM-INVERTED LATENTS (z_T) ───────────────────────────────────────
latents_zt = []
labels  = []
vae_scale_factor = vae.config.scaling_factor

print(f"Extracting DDIM-inverted latents for {len(subset)} samples...")
with torch.no_grad():
    for idx in tqdm(subset, desc="Processing images"):
        path, lbl = dataset.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open or convert image {path}. Skipping. Error: {e}")
            continue
        
        img = pad_to_multiple_of_8(img)
        x_pil = img

        # 1. Image to VAE latent (z0)
        x_tensor = to_tensor(x_pil).unsqueeze(0).to(DEVICE) * 2 - 1 # Normalize to [-1,1]
        z0_dist = vae.encode(x_tensor.to(dtype=vae.dtype))
        z0 = z0_dist.latent_dist.mean * vae_scale_factor # Use mean, then scale
        z0 = z0.to(dtype=unet.dtype)

        # 2. DDIM Encoding (z0 -> zT)
        # This implements the DDIM encoding process:
        # z_{t+1} = sqrt(a_bar_{t+1})*pred_x0_t + sqrt(1-a_bar_{t+1})*eps_t
        # where pred_x0_t = (z_t - sqrt(1-a_bar_t)*eps_t) / sqrt(a_bar_t)
        # and eps_t is the noise predicted by UNet for z_t at timestep t.
        # Timesteps for this noising process should conceptually go from low to high.
        
        scheduler.set_timesteps(NUM_INVERSION_STEPS, device=DEVICE)
        # scheduler.timesteps are typically [T-1, ..., 0] e.g. [981, 961, ..., 1] for 50 steps.
        # For encoding (noising z0 to zT), we iterate through these timesteps conceptually
        # from the "earliest" time (corresponding to z0) to the "latest" time (zT).
        # The UNet is queried at `t_unet_eval`.
        
        xt = z0 # Start with the clean VAE latent

        # Iterate from t=0 (cleanest) to t=T-1 (noisiest)
        # `reversed(scheduler.timesteps)` gives [1, 21, ..., 981] (low to high values)
        for i, t_unet_eval in enumerate(tqdm(reversed(scheduler.timesteps), desc="DDIM Encoding", leave=False, total=NUM_INVERSION_STEPS)):
            alpha_prod_t = scheduler.alphas_cumprod[t_unet_eval]
            
            # Determine alpha_prod_t_next (for the next noisier state)
            # If t_unet_eval is the last in reversed sequence (e.g., 981, which is scheduler.timesteps[0]),
            # then the next state is pure noise (alpha_prod_T ~ 0).
            if t_unet_eval == scheduler.timesteps[0]: # This t_unet_eval is the highest timestep value
                alpha_prod_t_next = scheduler.final_alpha_cumprod if hasattr(scheduler, "final_alpha_cumprod") and scheduler.final_alpha_cumprod is not None else torch.tensor(0.0, device=DEVICE, dtype=xt.dtype)
            else:
                # Find the original index of t_unet_eval in scheduler.timesteps
                # scheduler.timesteps = [t_N, t_{N-1}, ..., t_1]
                # reversed_timesteps = [t_1, ..., t_{N-1}, t_N]
                # If t_unet_eval is t_k from reversed_timesteps, t_next_unet_eval is t_{k+1}
                # This corresponds to stepping "earlier" in the original scheduler.timesteps array.
                current_original_idx = (scheduler.timesteps == t_unet_eval).nonzero().item()
                t_next_original_val = scheduler.timesteps[current_original_idx - 1] # Previous element in original list is next in reversed
                alpha_prod_t_next = scheduler.alphas_cumprod[t_next_original_val]

            # Predict noise
            # Ensure t_unet_eval is correctly shaped for unet (0-dim or 1-dim tensor)
            t_unet_input = t_unet_eval.unsqueeze(0) if t_unet_eval.ndim == 0 else t_unet_eval
            
            eps = unet(xt, t_unet_input, encoder_hidden_states=uncond_embeddings).sample
            
            # Calculate pred_x0
            sqrt_alpha_prod_t = alpha_prod_t.sqrt()
            sqrt_one_minus_alpha_prod_t = (1.0 - alpha_prod_t).sqrt()
            pred_x0 = (xt - sqrt_one_minus_alpha_prod_t * eps) / (sqrt_alpha_prod_t + 1e-8) # Add epsilon for stability
            
            # Calculate xt_next (the new xt, which is z_{t+1})
            sqrt_alpha_prod_t_next = alpha_prod_t_next.sqrt()
            sqrt_one_minus_alpha_prod_t_next = (1.0 - alpha_prod_t_next).sqrt()
            
            xt = sqrt_alpha_prod_t_next * pred_x0 + sqrt_one_minus_alpha_prod_t_next * eps
            
        inverted_zT = xt # This is the z_T after NUM_INVERSION_STEPS of noising
        
        vec = inverted_zT.cpu().view(-1).numpy()
        latents_zt.append(vec)
        labels.append(lbl)

if not latents_zt:
    print("No latents were extracted. Exiting.")
    exit()

latents_zt_np = np.stack(latents_zt, axis=0)

# ─── t-SNE & PLOT ────────────────────────────────────────────────────────────────
print("Performing t-SNE...")
tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=min(30, len(latents_zt_np)-1)) # Adjust perplexity
proj = tsne.fit_transform(latents_zt_np)

plt.figure(figsize=(10,10)) # Increased figure size
colors = ["red" if l==real_id else "blue" for l in labels]
scatter = plt.scatter(proj[:,0], proj[:,1], c=colors, alpha=0.6, s=20) # Adjusted alpha and size

# Create legend handles manually
real_handle = plt.Line2D([0],[0], marker='o', color='w', label='REAL',
                         markerfacecolor='red', markersize=10)
fake_handle = plt.Line2D([0],[0], marker='o', color='w', label='FAKE',
                         markerfacecolor='blue', markersize=10)

plt.legend(handles=[real_handle, fake_handle], title="Image Type")
plt.title(f"t-SNE of DDIM-Inverted Latents ($z_T$, {NUM_INVERSION_STEPS} steps)\nReal (red) vs. Fake (blue)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)

# ─── SAVE TO FILE ────────────────────────────────────────────────────────────────
save_path = OUTPUT_DIR / f"tsne_ddim_inverted_latents_zT_{current_n_samples}samples_{NUM_INVERSION_STEPS}steps.png"
plt.savefig(save_path)
print(f"Saved t-SNE plot to {save_path}")
plt.show()
