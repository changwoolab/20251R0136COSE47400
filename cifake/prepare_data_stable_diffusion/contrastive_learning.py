#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel
from torch.optim.lr_scheduler import StepLR

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("outputs")
STEPS          = [1, 2, 3, 4]
BATCH_SIZE     = 32
EPOCHS         = 1
LR             = 3e-4                    
DEVICE         = torch.device("cuda")
IMG_SIZE       = 224
BASELINE_MODEL = "dima806/ai_vs_real_image_detection"
TEMPERATURE    = 0.1                     # lower T ⇒ sharper contrasts

# ─── AUGMENT + TRANSFORM ────────────────────────────────────────────────────────
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# We'll apply SD‐noise *after* these augmentations to keep things mixed.

# ─── DATASET ────────────────────────────────────────────────────────────────────
class NoisyPairDataset(Dataset):
    def __init__(self, base_dir: Path, steps, aug):
        self.orig_dir  = base_dir / "original"
        self.step_dirs = [base_dir / f"step_{s}" for s in steps]
        self.aug       = aug
        self.samples   = []
        for img_path in sorted(self.orig_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg","jpeg","png"}:
                continue
            for sd in self.step_dirs:
                noisy_path = sd / img_path.name
                if noisy_path.exists():
                    self.samples.append((img_path, noisy_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        real_p, noisy_p = self.samples[idx]
        real = Image.open(real_p).convert("RGB")
        noisy = Image.open(noisy_p).convert("RGB")
        # apply *the same* random augmentation to both images 
        seed = torch.randint(0, 2**32, ()).item()
        torch.manual_seed(seed)
        real_t = self.aug(real)
        torch.manual_seed(seed)
        noisy_t = self.aug(noisy)
        return real_t, noisy_t

# ─── MODEL & LOSS ───────────────────────────────────────────────────────────────
class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        vit = AutoModel.from_pretrained(BASELINE_MODEL)
        self.encoder  = vit          # outputs .pooler_output
        hidden        = vit.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 128),
        )

    def forward(self, x):
        h = self.encoder(pixel_values=x).pooler_output
        z = self.projector(h)
        # **normalize** to unit sphere
        return nn.functional.normalize(z, dim=1)

def nt_xent_loss(z1, z2, T):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                        # 2B×D
    sim = torch.matmul(z, z.T) / T                        # cosine via inner‐prod
    # mask out self‐similarities
    mask = (~torch.eye(2*B, device=sim.device).bool()).float()
    exp_sim = torch.exp(sim) * mask
    # positives at offset ±B
    pos = torch.cat([torch.diag(sim,  B), torch.diag(sim, -B)], dim=0)
    loss = -torch.log(torch.exp(pos) / exp_sim.sum(dim=1))
    return loss.mean()

# ─── TRAINING ───────────────────────────────────────────────────────────────────
def train():
    # 1) data
    ds     = NoisyPairDataset(OUTPUT_DIR, STEPS, augmentation)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # 2) model, optim, sched
    model = ContrastiveModel().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = StepLR(opt, step_size=1, gamma=0.9)

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for real, noisy in pbar:
            real  = real.to(DEVICE, non_blocking=True)
            noisy = noisy.to(DEVICE, non_blocking=True)

            z1 = model(real)
            z2 = model(noisy)
            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            opt.zero_grad()
            loss.backward()
            # **gradient clipping** to stabilize early training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])

        sched.step()

    # 3) save
    torch.save(model.encoder.state_dict(),   "encoder_final.pt")
    torch.save(model.projector.state_dict(), "projector_final.pt")

if __name__ == "__main__":
    train()
