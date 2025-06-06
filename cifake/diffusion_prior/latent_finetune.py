#!/usr/bin/env python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from pathlib import Path
from tqdm.auto import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("../data")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "test"
BATCH_SIZE  = 16   # <--- reduce batch size for memory headroom
LR          = 1e-4
EPOCHS      = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "CompVis/stable-diffusion-v1-4"
IMG_SIZE    = 224
NUM_CLASSES = 2  # REAL vs FAKE

# ─── IMAGE TRANSFORMS ────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),                   
    T.Lambda(lambda x: x * 2 - 1),  # SD-VAE expects [-1,1], in float32
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

# ─── LOAD SD VAE ENCODER ──────────────────────────────────────────────────────────
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")\
       .to(DEVICE).eval()  # keep in eval mode but allow grads
# make sure it's float32
vae = vae.to(torch.float32)

# ─── DETERMINE LATENT DIM ─────────────────────────────────────────────────────────
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=torch.float32)
    latent = vae.encode(dummy).latent_dist.mean
latent_dim = latent.numel()

# ─── BUILD CLASSIFIER ────────────────────────────────────────────────────────────
classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(latent_dim, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, NUM_CLASSES),
).to(DEVICE)

# ─── OPTIMIZER, SCALER, LOSS ─────────────────────────────────────────────────────
# train both vae and classifier
opt    = optim.AdamW(
    list(vae.parameters()) + list(classifier.parameters()),
    lr=LR, weight_decay=1e-4
)
scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    vae.train()
    classifier.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")

    for imgs, labels in train_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            # VAE encode (float32 weights but runs in FP16 ops region)
            enc = vae.encode(imgs)
            lat = enc.latent_dist.mean      # [B,C,H/8,W/8]
            lat = lat.view(lat.size(0), -1) # flatten
            logits = classifier(lat)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        train_bar.set_postfix(loss=loss.item())

    # ─── VALIDATION ───────────────────────────────────────────────────────────────
    vae.eval()
    classifier.eval()
    correct, total = 0, 0

    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            lat = vae.encode(imgs).latent_dist.mean
            lat = lat.view(lat.size(0), -1)
            preds = classifier(lat).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    print(f" → Epoch {epoch} Val Acc: {correct/total:.4f}")

# ─── SAVE ────────────────────────────────────────────────────────────────────────
torch.save(vae.state_dict(),        "vae_finetuned.pt")
torch.save(classifier.state_dict(), "latent_classifier.pt")
print("Saved VAE and classifier state_dicts.")
