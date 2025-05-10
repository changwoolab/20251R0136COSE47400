from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)
from utils import labels_list, id2label, label2id

def get_vit_processor_and_transform(model="dima806/ai_vs_real_image_detection"):
    processor = ViTImageProcessor.from_pretrained(model)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    # Define a normalization transformation for the input images
    normalize = Normalize(mean=image_mean, std=image_std)

    # Define a set of transformations for training data
    _train_transforms = Compose(
        [
            Resize((size, size)),             # Resize images to the ViT model's input size
            RandomRotation(90),               # Apply random rotation
            RandomAdjustSharpness(2),         # Adjust sharpness randomly
            ToTensor(),                       # Convert images to tensors
            normalize                         # Normalize images using mean and std
        ]
    )

    # Define a set of transformations for validation data
    _val_transforms = Compose(
        [
            Resize((size, size)),             # Resize images to the ViT model's input size
            ToTensor(),                       # Convert images to tensors
            normalize                         # Normalize images using mean and std
        ]
    )

    # Define a function to apply training transformations to a batch of examples
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    # Define a function to apply validation transformations to a batch of examples
    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples
    
    return processor, train_transforms, val_transforms

def get_vit_model(model="dima806/ai_vs_real_image_detection", num_labels=2):
    # Load a pre-trained ViT model for image classification
    model = ViTForImageClassification.from_pretrained(model, num_labels=len(labels_list))
    model.config.id2label = id2label
    model.config.label2id = label2id
    return model

