#!/usr/bin/env python3
import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import torchvision.models as models

from utils import CustomImageFolder  # your subclass that loads label2id

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("../data")
TRAIN_DIR    = DATA_DIR / "train"
VAL_DIR      = DATA_DIR / "test"
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
EPOCH_NUM    = 1
ENCODER_CKPT = f"model_outputs/encoder_resnet18_epoch{EPOCH_NUM}.pt"
PROJECTOR_CKPT = f"model_outputs/projector_resnet18_epoch{EPOCH_NUM}.pt"

# ─── TRANSFORM ──────────────────────────────────────────────────────────────────
eval_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ─── DATASETS & LOADERS ──────────────────────────────────────────────────────────
train_ds = CustomImageFolder(TRAIN_DIR, transform=eval_transform)
val_ds   = CustomImageFolder(VAL_DIR,   transform=eval_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ─── CONNECTIVE MODEL DEFINITION ────────────────────────────────────────────────
class ContrastiveResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        hidden = resnet.fc.in_features  # 512
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 128),
        )

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)  # [B,512]
        z = self.projector(h)                   # [B,128]
        return nn.functional.normalize(z, dim=1)

# ─── LOAD PRETRAINED WEIGHTS ────────────────────────────────────────────────────
model = ContrastiveResNet18().to(DEVICE)
model.encoder.load_state_dict(torch.load(ENCODER_CKPT,   map_location=DEVICE))
model.projector.load_state_dict(torch.load(PROJECTOR_CKPT, map_location=DEVICE))
model.eval()

# ─── EXTRACT TRAIN EMBEDDINGS & PROTOTYPES ──────────────────────────────────────
all_embeddings = []
all_labels     = []
with torch.no_grad():
    for imgs, labels in tqdm(train_loader, desc="Embedding train set"):
        imgs = imgs.to(DEVICE)
        z    = model(imgs)                      # [B,128]
        all_embeddings.append(z.cpu())
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings, dim=0)  # [N_train,128]
all_labels     = torch.cat(all_labels,     dim=0)  # [N_train]

# compute one prototype per class
prototypes = []
for cls in sorted(train_ds.class_to_idx.values()):
    cls_mask = (all_labels == cls)
    proto    = all_embeddings[cls_mask].mean(dim=0)
    proto    = proto / proto.norm()  # normalize prototype
    prototypes.append(proto)
prototypes = torch.stack(prototypes, dim=0)  # [num_classes,128]

# ─── EVALUATE ON VAL SET ─────────────────────────────────────────────────────────
cosine = nn.CosineSimilarity(dim=1)
correct, total = 0, 0

with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Evaluating on val set"):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        z      = model(imgs)                     # [B,128]
        # compute cosine similarity to each prototype
        sims   = z @ prototypes.t()              # [B, num_classes]
        preds  = sims.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

print(f"Prototype‐based k=1 accuracy on val set: {correct/total:.4f}")
