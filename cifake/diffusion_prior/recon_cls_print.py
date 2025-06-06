#!/usr/bin/env python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("../data")
TRAIN_DIR     = DATA_DIR / "train"
VAL_DIR       = DATA_DIR / "test"
BATCH_SIZE    = 8     # adjust as needed
LR            = 1e-4
EPOCHS        = 2 #TODO
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME    = "CompVis/stable-diffusion-v1-4"
IMG_SIZE      = 224
NUM_CLASSES   = 2      # REAL vs FAKE
RECON_WEIGHT  = 1    # weight for reconstruction loss

# ─── HELPER FUNCTION TO PRINT MODEL PARAMETERS ───────────────────────────────────
def print_model_parameters(model, model_name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print(f"\n--- {model_name} Parameters ---")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("--------------------------")

# ─── IMAGE TRANSFORMS ────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2 - 1),   # →[-1,1]
])

# ─── DATALOADERS ─────────────────────────────────────────────────────────────────
try:
    train_ds = ImageFolder(TRAIN_DIR, transform=transform)
    val_ds   = ImageFolder(VAL_DIR,   transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"Training data: {len(train_ds)} samples. Validation data: {len(val_ds)} samples.")
except FileNotFoundError:
    print(f"Error: Data directory not found. Please check TRAIN_DIR ({TRAIN_DIR}) and VAL_DIR ({VAL_DIR}) paths.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# ─── LOAD SD VAE ────────────────────────────────────────────────────────────────
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")\
       .to(DEVICE)
# ensure float32
vae = vae.to(torch.float32)
# VAE is set to vae.train() later, before the training loop starts,
# but parameter trainability is defined now.

# ─── FREEZE DECODER ──────────────────────────────────────────────────────────────
# keep encoder trainable, freeze decoder and post_quant_conv
for name, param in vae.named_parameters():
    if name.startswith("decoder.") or name.startswith("post_quant_conv."):
        param.requires_grad = False
    else: # Encoder parameters
        param.requires_grad = True # Explicitly ensure encoder parts are trainable

# Print VAE parameters after freezing configuration
print_model_parameters(vae, "VAE (AutoencoderKL)")

# ─── DETERMINE LATENT DIM ─────────────────────────────────────────────────────────
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=vae.dtype)
    # Use vae.eval() context for deterministic output shape if needed, though not strictly necessary for .mean
    vae_eval_context = vae.eval() if vae.training else torch.enable_grad() # temp set to eval if training
    with vae_eval_context:
        latent_dist_output = vae.encode(dummy)
        latent = latent_dist_output.latent_dist.mean
    # Restore original VAE training mode if it was changed
    # No, vae.train() is called before epoch loop. This is fine for dim determination.
latent_dim = latent.numel()
print(f"Determined VAE latent dimension (flattened): {latent_dim}")

# ─── CLASSIFIER ───────────────────────────────────────────────────────────────────
classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(latent_dim, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5), # Added dropout for regularization
    nn.Linear(512, NUM_CLASSES),
).to(DEVICE).to(torch.float32) # Ensure classifier is also float32

# Print Classifier parameters
print_model_parameters(classifier, "Classifier")

# ─── OPTIMIZER, SCALER, LOSSES ────────────────────────────────────────────────────
# include only trainable params (encoder + classifier)
params_to_train = list(filter(lambda p: p.requires_grad, vae.parameters())) \
                  + list(classifier.parameters())
if not params_to_train:
    print("Error: No trainable parameters found for the optimizer. Check model freezing logic.")
    exit()
else:
    num_vae_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    num_classifier_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total parameters for optimizer: {sum(p.numel() for p in params_to_train):,}")
    print(f"  Trainable VAE (encoder) params: {num_vae_trainable:,}")
    print(f"  Trainable Classifier params: {num_classifier_trainable:,}")


opt          = optim.AdamW(params_to_train, lr=LR, weight_decay=1e-4)
lr_scheduler = ExponentialLR(opt, gamma=0.9)
scaler       = torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda') # Enable only for CUDA
cls_criterion  = nn.CrossEntropyLoss()
recon_criterion = nn.MSELoss()

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────────
print(f"\nStarting training for {EPOCHS} epochs on device: {DEVICE}")
for epoch in range(1, EPOCHS+1):
    vae.train() # Set VAE to train mode (affects dropout, batchnorm if any in encoder)
    classifier.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")

    for imgs, labels in train_bar:
        imgs = imgs.to(DEVICE, dtype=vae.dtype) # Ensure dtype consistency
        labels = labels.to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type=='cuda', dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16): # use float16 on cuda, bfloat16 on cpu if available
            # encode
            enc_output = vae.encode(imgs)
            lat        = enc_output.latent_dist.mean # [B,C,H/8,W/8]

            # classification loss
            # Classifier expects float32 or autocast type. lat is already vae.dtype (float32).
            logits     = classifier(lat) # Flatten is the first layer in classifier
            cls_loss   = cls_criterion(logits, labels)

            # reconstruction loss
            # VAE.decode expects input latents to be (latents / scaling_factor)
            # Diffusers VAE.decode method handles the division by scaling_factor internally.
            # So, we should pass the 'lat' (unscaled mean from latent_dist) directly.
            dec_output = vae.decode(lat).sample # Pass unscaled lat
            recon_loss = recon_criterion(dec_output, imgs)

            loss = cls_loss + RECON_WEIGHT * recon_loss

        if DEVICE.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else: # CPU training
            loss.backward()
            opt.step()

        train_bar.set_postfix(cls=f"{cls_loss.item():.4f}",
                              recon=f"{recon_loss.item():.4f}",
                              loss=f"{loss.item():.4f}")

    # ─── VALIDATION ───────────────────────────────────────────────────────────────
    vae.eval()
    classifier.eval()
    correct, total_val_samples = 0, 0
    val_loss_epoch = 0
    with torch.no_grad(): # No gradients needed for validation
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]")
        for imgs, labels in val_bar:
            imgs = imgs.to(DEVICE, dtype=vae.dtype)
            labels = labels.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type=='cuda', dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
                enc_output = vae.encode(imgs)
                lat        = enc_output.latent_dist.mean
                
                logits     = classifier(lat)
                # Optionally calculate validation loss components too
                # val_cls_loss = cls_criterion(logits, labels)
                # dec_output = vae.decode(lat).sample
                # val_recon_loss = recon_criterion(dec_output, imgs)
                # val_loss = val_cls_loss + RECON_WEIGHT * val_recon_loss
                # val_loss_epoch += val_loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_val_samples += labels.size(0)
            val_bar.set_postfix(acc=f"{(preds == labels).float().mean().item():.3f}")

    val_accuracy = correct / total_val_samples if total_val_samples > 0 else 0
    # avg_val_loss = val_loss_epoch / total_val_samples if total_val_samples > 0 else 0
    # print(f" → Epoch {epoch} Val Acc: {val_accuracy:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
    print(f" → Epoch {epoch} Val Acc: {val_accuracy:.4f}")
    lr_scheduler.step()

# ─── SAVE ────────────────────────────────────────────────────────────────────────
Path("checkpoints").mkdir(exist_ok=True) # Ensure checkpoint directory exists
save_path_vae = Path("checkpoints") / "vae_encoder_finetuned_z0.pt"
save_path_classifier = Path("checkpoints") / "classifier_on_z0.pt"

torch.save(vae.state_dict(), save_path_vae)
torch.save(classifier.state_dict(), save_path_classifier)
print(f"\nSaved VAE state_dict to {save_path_vae}")
print(f"Saved classifier state_dict to {save_path_classifier}")
print("Training complete.")