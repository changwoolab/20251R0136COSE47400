from PIL import ImageFile
from datasets import Dataset, Image, ClassLabel
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
from utils import ClassLabels

# Enable the option to load truncated images.
# This setting allows the PIL library to attempt loading images even if they are corrupted or incomplete.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Mapping labels to IDs
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

def get_dataset(data_dir: str = 'cifake/data/'):
    # Initialize empty lists to store file names and labels
    file_names = []
    labels = []

    # Iterate through all image files in the specified directory
    for file in sorted((Path(data_dir).glob('*/*/*.*'))):
        label = str(file).split('/')[-2]  # Extract the label from the file path
        labels.append(label)  # Add the label to the list
        file_names.append(str(file))  # Add the file path to the list

    # Create a pandas dataframe from the collected file names and labels
    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})

    # random oversampling of minority class
    # 'y' contains the target variable (label) we want to predict
    y = df[['label']]

    # Drop the 'label' column from the DataFrame 'df' to separate features from the target variable
    df = df.drop(['label'], axis=1)

    # Create a RandomOverSampler object with a specified random seed (random_state=83)
    ros = RandomOverSampler(random_state=83)

    # Use the RandomOverSampler to resample the dataset by oversampling the minority class
    # 'df' contains the feature data, and 'y_resampled' will contain the resampled target variable
    df, y_resampled = ros.fit_resample(df, y)

    # Add the resampled target variable 'y_resampled' as a new 'label' column in the DataFrame 'df'
    df['label'] = y_resampled

    dataset = Dataset.from_pandas(df).cast_column("image", Image())

    dataset = dataset.map(map_label2id, batched=True)

    # Casting label column to ClassLabel Object
    dataset = dataset.cast_column('label', ClassLabels)
    dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")

    train_data = dataset['train']
    test_data = dataset['test']

    return train_data, test_data
