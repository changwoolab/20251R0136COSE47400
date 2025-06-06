#!/usr/bin/env python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import random

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("../data") # Root directory containing train/val/test splits
TRAIN_DIR     = DATA_DIR / "train" # Must contain "REAL" & "FAKE" subfolders
VAL_DIR       = DATA_DIR / "test"  # Must contain "REAL" & "FAKE" subfolders
BATCH_SIZE    = 128     # Adjust based on GPU memory for inversion + classifier
LR            = 1e-4  # Learning rate for the classifier
EPOCHS        = 10    # Number of epochs to train the classifier
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME    = "CompVis/stable-diffusion-v1-4" # Base SD model for VAE & UNet
IMG_SIZE      = 224
NUM_CLASSES   = 2     # REAL vs FAKE
NUM_INVERSION_STEPS = 15 # Number of DDIM steps for z0 -> zT inversion.
CHECKPOINT_DIR = Path("./checkpoints_zt_classifier")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ─── IMAGE TRANSFORMS ────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2 - 1),   # →[-1,1]
])

# ─── DATALOADERS ─────────────────────────────────────────────────────────────────
try:
    train_ds = ImageFolder(TRAIN_DIR, transform=transform)
    val_ds   = ImageFolder(VAL_DIR,   transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"Training on {len(train_ds)} images, validating on {len(val_ds)} images.")
    print(f"Train classes: {train_ds.class_to_idx}")
    print(f"Val classes: {val_ds.class_to_idx}")
    if train_ds.class_to_idx != val_ds.class_to_idx or not all(c in train_ds.class_to_idx for c in ["REAL", "FAKE"]):
         raise ValueError("Class mismatch or REAL/FAKE classes not found. Ensure consistent 'REAL' and 'FAKE' subfolders.")

except FileNotFoundError as e:
    print(f"Error: Data directory not found: {e}. Please check TRAIN_DIR and VAL_DIR paths.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()


# ─── LOAD PRE-TRAINED MODELS (All Frozen for feature extraction) ───────────────
# VAE
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(torch.float32)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
vae_scale_factor = vae.config.scaling_factor
print("VAE loaded and frozen.")

# UNet
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(DEVICE).to(torch.float32)
unet.eval()
for param in unet.parameters():
    param.requires_grad = False
print("UNet loaded and frozen.")

# Scheduler
scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

# Tokenizer and Text Encoder for unconditional embeddings
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
    prompt = ""
    text_inputs = tokenizer_model(
        prompt, padding="max_length", max_length=tokenizer_model.model_max_length,
        truncation=True, return_tensors="pt",
    )
    embeddings = text_encoder_model(text_inputs.input_ids.to(device_type))[0]
    return embeddings.to(dtype=model_dtype)

uncond_embeddings = get_uncond_embeddings(tokenizer, text_encoder, DEVICE, unet.dtype)

# ─── DDIM ENCODING FUNCTION (z0 to zT) ─────────────────────────────────────────
@torch.no_grad() # No gradients needed for this feature extraction process
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

# ─── DETERMINE LATENT DIM FOR CLASSIFIER INPUT ──────────────────────────────────
with torch.no_grad():
    dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=vae.dtype)
    dummy_z0_unscaled_dist = vae.encode(dummy_img)
    dummy_z0_unscaled = dummy_z0_unscaled_dist.latent_dist.mean
    # zT will have the same dimensions as z0
    zt_latent_dim = dummy_z0_unscaled.numel() # Flattened dimension C*H*W

# ─── CLASSIFIER MODEL (Trainable) ───────────────────────────────────────────────
class ZTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

classifier = ZTClassifier(input_dim=zt_latent_dim, num_classes=NUM_CLASSES).to(DEVICE).to(torch.float32)
print(f"Classifier initialized with input dimension {zt_latent_dim}.")

# ─── OPTIMIZER, SCHEDULER, LOSS ──────────────────────────────────────────────────
opt = optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-4)
# Optional: Learning rate scheduler
# lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
criterion  = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────────
print(f"Starting classifier training for {EPOCHS} epochs on {DEVICE}.")
print(f"DDIM Inversion Steps for feature extraction: {NUM_INVERSION_STEPS}")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    classifier.train()
    train_loss_epoch = 0
    train_correct_epoch = 0
    train_total_epoch = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for imgs, labels_cpu in train_bar:
        imgs = imgs.to(DEVICE, dtype=vae.dtype)
        labels = labels_cpu.to(DEVICE)

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda', dtype=torch.bfloat16 if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16):
            # --- Feature Extraction (Frozen Path) ---
            with torch.no_grad(): # Ensure no gradients for feature extraction part
                z0_unscaled_dist = vae.encode(imgs)
                z0_unscaled = z0_unscaled_dist.latent_dist.mean
                z0_scaled = z0_unscaled * vae_scale_factor
                inverted_zt = ddim_encode_z0_to_zT(
                    z0_scaled, unet, scheduler, NUM_INVERSION_STEPS,
                    uncond_embeddings, DEVICE, unet.dtype
                ) # inverted_zt is [B, C_latent, H_latent, W_latent]

            # --- Classification (Trainable Path) ---
            # Ensure inverted_zt is on the right dtype for the classifier if autocast is used for it too
            logits = classifier(inverted_zt.to(classifier.network[1].weight.dtype))
            loss = criterion(logits, labels)

        if DEVICE.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        train_loss_epoch += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        train_correct_epoch += (preds == labels).sum().item()
        train_total_epoch += labels.size(0)

        train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds == labels).float().mean().item():.3f}")

    avg_train_loss = train_loss_epoch / train_total_epoch
    avg_train_acc = train_correct_epoch / train_total_epoch
    print(f"Epoch {epoch} Train Summary: Avg Loss: {avg_train_loss:.4f}, Avg Acc: {avg_train_acc:.4f}")

    # if lr_scheduler:
    #     lr_scheduler.step()

    # ─── VALIDATION ───────────────────────────────────────────────────────────
    classifier.eval()
    val_loss_epoch = 0
    val_correct_epoch = 0
    val_total_epoch = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
    with torch.no_grad(): # No gradients needed for validation
        for imgs, labels_cpu in val_bar:
            imgs = imgs.to(DEVICE, dtype=vae.dtype)
            labels = labels_cpu.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda', dtype=torch.bfloat16 if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16):
                # Feature Extraction (Frozen)
                z0_unscaled_dist = vae.encode(imgs)
                z0_unscaled = z0_unscaled_dist.latent_dist.mean
                z0_scaled = z0_unscaled * vae_scale_factor
                inverted_zt = ddim_encode_z0_to_zT(
                    z0_scaled, unet, scheduler, NUM_INVERSION_STEPS,
                    uncond_embeddings, DEVICE, unet.dtype
                )
                # Classification
                logits = classifier(inverted_zt.to(classifier.network[1].weight.dtype))
                loss = criterion(logits, labels)

            val_loss_epoch += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            val_correct_epoch += (preds == labels).sum().item()
            val_total_epoch += labels.size(0)
            val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds == labels).float().mean().item():.3f}")

    avg_val_loss = val_loss_epoch / val_total_epoch
    avg_val_acc = val_correct_epoch / val_total_epoch
    print(f"Epoch {epoch} Val Summary: Avg Loss: {avg_val_loss:.4f}, Avg Acc: {avg_val_acc:.4f}")

    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        save_path_classifier = CHECKPOINT_DIR / f"best_zt_classifier_e{epoch}_acc{avg_val_acc:.4f}.pt"
        torch.save(classifier.state_dict(), save_path_classifier)
        print(f"New best validation accuracy: {best_val_acc:.4f}. Saved classifier to {save_path_classifier}")

# ─── FINAL SAVE (LAST EPOCH) ───────────────────────────────────────────────────
final_save_path_classifier = CHECKPOINT_DIR / "final_zt_classifier.pt"
torch.save(classifier.state_dict(), final_save_path_classifier)
print(f"Saved final classifier state_dict to {final_save_path_classifier}")
print("Training complete.")
