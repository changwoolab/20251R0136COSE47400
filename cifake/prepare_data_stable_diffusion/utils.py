from torchvision.datasets import ImageFolder
from transformers import (
    AutoConfig,
)

BASELINE_MODEL  = "dima806/ai_vs_real_image_detection"
cfg = AutoConfig.from_pretrained(BASELINE_MODEL)

class CustomImageFolder(ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Instead of scanning subdirectories alphabetically, 
        load classes & indices from the model's own label2id.
        """
        label2id = cfg.label2id
        classes = list(label2id.keys())
        return classes, label2id
