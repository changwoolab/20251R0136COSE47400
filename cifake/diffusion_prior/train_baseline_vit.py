#!/usr/bin/env python3
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# torchvision.transforms might still be useful for augmentations if you add them later
# import torchvision.transforms as T # No longer primary for ViT preprocessing
from pathlib import Path
from tqdm.auto import tqdm

from transformers import AutoImageProcessor, AutoModelForImageClassification

from utils import CustomImageFolder  # must implement find_classes() → cfg.label2id

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("../data")
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "test"
BATCH_SIZE  = 64
LR          = 1e-4 # You might need to tune this for ViT
EPOCHS      = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "google/vit-base-patch16-224-in21k"

# ─── IMAGE PROCESSOR & DATA ───────────────────────────────────────────────────────
# Load the image processor associated with the ViT model
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Define the transformation using the ViT image processor
# The processor handles resizing, normalization, and conversion to PyTorch tensors
# It expects PIL Images as input
def vit_transform(pil_image):
    processed_inputs = image_processor(images=pil_image, return_tensors="pt")
    # Squeeze to remove the batch dimension added by the processor for a single image
    return processed_inputs['pixel_values'].squeeze(0)

train_ds = CustomImageFolder(TRAIN_DIR, transform=vit_transform)
val_ds   = CustomImageFolder(VAL_DIR,   transform=vit_transform)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

# ─── MODEL SETUP ─────────────────────────────────────────────────────────────────
# Ensure class_to_idx is available and get the number of classes
if not hasattr(train_ds, 'class_to_idx') or not train_ds.class_to_idx:
    raise RuntimeError("The CustomImageFolder dataset must have a 'class_to_idx' attribute.")
num_classes = len(train_ds.class_to_idx)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True  # This allows to load a pretrained model with a new classifier head
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# AdamW is often a good choice for Transformers
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# Learning rate scheduler can be kept or adjusted
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# ─── TRAIN & EVAL LOOP ────────────────────────────────────────────────────────────
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        # For Hugging Face models, pass inputs as keyword arguments (e.g., pixel_values)
        # The output object typically contains 'logits'
        outputs = model(pixel_values=imgs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_ds)
    epoch_acc  = running_corrects / len(train_ds)
    print(f"  Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=imgs).logits
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_loss += loss.item() * imgs.size(0)
            val_corrects += (preds == labels).sum().item()

    val_epoch_loss = val_loss / len(val_ds)
    val_epoch_acc  = val_corrects / len(val_ds)
    print(f"  Val   Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

    # Save best
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        # Update the saved model filename
        torch.save(model.state_dict(), "vit_base_patch16_best_classifier.pt")
        print("  Saved new best model")

print(f"Training complete. Best Val Acc: {best_acc:.4f}")
