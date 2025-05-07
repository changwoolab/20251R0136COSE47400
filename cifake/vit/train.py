import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from transformers import (
    TrainingArguments,
    Trainer,
)
import torch
from cifake.vit.data_loader import get_dataset
from cifake.vit.vit_processor import get_vit_processor_and_transform, get_vit_model
from utils import compute_metrics

def collate_fn(examples):
    # Stack the pixel values from individual examples into a single tensor.
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    
    # Convert the label strings in examples to corresponding numeric IDs using label2id dictionary.
    labels = torch.tensor([example['label'] for example in examples])
    
    return {"pixel_values": pixel_values, "labels": labels}

def train():
    train_data, test_data = get_dataset()

    model_str = "dima806/ai_vs_real_image_detection"

    processor, train_transforms, val_transforms = get_vit_processor_and_transform(model_str)

    train_data.set_transform(train_transforms)
    test_data.set_transform(val_transforms)

    model = get_vit_model(model_str)

    model_name = "ai_vs_real_image_detection"
    num_train_epochs = 2
    args = TrainingArguments(
        output_dir=model_name,
        logging_dir='./logs',
        eval_strategy="epoch",
        learning_rate=1e-6,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        weight_decay=0.02,
        warmup_steps=50,
        remove_unused_columns=False,
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.evaluate()
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    train()
