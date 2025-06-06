#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("outputs")
STEPS          = [1, 2, 3, 4]
BATCH_SIZE     = 32
EPOCHS         = 5
LR             = 3e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE       = 224
TEMPERATURE    = 0.1

# ─── AUGMENT + TRANSFORM ────────────────────────────────────────────────────────
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

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
        real  = Image.open(real_p).convert("RGB")
        noisy = Image.open(noisy_p).convert("RGB")
        # apply same random augmentation to both
        seed = torch.randint(0, 2**32, ()).item()
        torch.manual_seed(seed)
        real_t  = self.aug(real)
        torch.manual_seed(seed)
        noisy_t = self.aug(noisy)
        return real_t, noisy_t

# ─── MODEL & LOSS ───────────────────────────────────────────────────────────────
class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        # load pretrained ResNet18
        resnet = models.resnet18(pretrained=False)
        # remove its final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        hidden = resnet.fc.in_features  # 512
        # projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 128),
        )

    def forward(self, x):
        # x: [B,3,224,224]
        h = self.encoder(x)               # [B,512,1,1]
        h = h.view(h.size(0), -1)         # [B,512]
        z = self.projector(h)             # [B,128]
        return nn.functional.normalize(z, dim=1)

def nt_xent_loss(z1, z2, T):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)     # [2B,128]
    sim = torch.matmul(z, z.T) / T     # [2B,2B]
    mask = (~torch.eye(2*B, device=z.device).bool()).float()
    exp_sim = torch.exp(sim) * mask
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
    loss = -torch.log(torch.exp(pos) / exp_sim.sum(dim=1))
    return loss.mean()

# ─── TRAINING ───────────────────────────────────────────────────────────────────
def train():
    # 1) data
    ds     = NoisyPairDataset(OUTPUT_DIR, STEPS, augmentation)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # 2) model, optimizer, scheduler
    model = ContrastiveModel().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = StepLR(opt, step_size=1, gamma=0.9)

    # 3) training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for real, noisy in pbar:
            real, noisy = real.to(DEVICE, non_blocking=True), noisy.to(DEVICE, non_blocking=True)
            z1 = model(real)
            z2 = model(noisy)
            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])
        sched.step()

        # 4) save
        torch.save(model.encoder.state_dict(),   f"model_outputs/encoder_resnet18_epoch{epoch}.pt")
        torch.save(model.projector.state_dict(), f"model_outputs/projector_resnet18_epoch{epoch}.pt")

if __name__ == "__main__":
    train()
