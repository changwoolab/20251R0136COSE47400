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
BATCH_SIZE    = 32     # adjust as needed
LR            = 1e-4
EPOCHS        = 5 #TODO
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME    = "CompVis/stable-diffusion-v1-4"
IMG_SIZE      = 224
NUM_CLASSES   = 2      # REAL vs FAKE
RECON_WEIGHT  = 0    # weight for reconstruction loss

# ─── IMAGE TRANSFORMS ────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),                   
    T.Lambda(lambda x: x * 2 - 1),   # →[-1,1]
])

# ─── DATALOADERS ─────────────────────────────────────────────────────────────────
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

# ─── LOAD SD VAE ────────────────────────────────────────────────────────────────
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")\
       .to(DEVICE)
# ensure float32
vae = vae.to(torch.float32)
vae.train()  # enable training mode for encoder

# ─── FREEZE DECODER ──────────────────────────────────────────────────────────────
# keep encoder trainable, freeze decoder and post_quant_conv
for name, param in vae.named_parameters():
    if name.startswith("decoder.") or name.startswith("post_quant_conv."):
        param.requires_grad = False

# ─── DETERMINE LATENT DIM ─────────────────────────────────────────────────────────
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    latent = vae.encode(dummy).latent_dist.mean
latent_dim = latent.numel()

# ─── CLASSIFIER ───────────────────────────────────────────────────────────────────
classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(latent_dim, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, NUM_CLASSES),
).to(DEVICE)

# ─── OPTIMIZER, SCALER, LOSSES ────────────────────────────────────────────────────
# include only trainable params (encoder + classifier)
params = list(filter(lambda p: p.requires_grad, vae.parameters())) \
       + list(classifier.parameters())
opt          = optim.AdamW(params, lr=LR, weight_decay=1e-4)
lr_scheduler = ExponentialLR(opt, gamma=0.9)
scaler       = torch.cuda.amp.GradScaler()
cls_criterion  = nn.CrossEntropyLoss()
recon_criterion = nn.MSELoss()

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    vae.train()
    classifier.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")

    for imgs, labels in train_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            # encode
            enc    = vae.encode(imgs)
            lat    = enc.latent_dist.mean      # [B,C,H/8,W/8]
            flat   = lat.view(lat.size(0), -1)

            # classification loss
            logits   = classifier(flat)
            cls_loss = cls_criterion(logits, labels)

            # reconstruction loss
            if RECON_WEIGHT > 0:
                dec       = vae.decode(lat / vae.config.scaling_factor).sample
                recon_loss = recon_criterion(dec, imgs)
            else:
                recon_loss = torch.tensor(0.0, device=imgs.device)

            loss = cls_loss + RECON_WEIGHT * recon_loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        train_bar.set_postfix(cls=f"{cls_loss.item():.4f}",
                              recon=f"{recon_loss.item():.4f}")

    # ─── VALIDATION ───────────────────────────────────────────────────────────────
    vae.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            lat = vae.encode(imgs).latent_dist.mean
            flat = lat.view(lat.size(0), -1)
            preds = classifier(flat).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f" → Epoch {epoch} Val Acc: {correct/total:.4f}")
    lr_scheduler.step()

# ─── SAVE ────────────────────────────────────────────────────────────────────────
torch.save(vae.state_dict(),        "vae_no_recon_finetuned.pt")
torch.save(classifier.state_dict(), "latent_no_recon_classifier.pt")
print("Saved VAE encoder and classifier state_dicts.")
