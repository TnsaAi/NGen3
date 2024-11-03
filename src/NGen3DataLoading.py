import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class NGen3Dataset(Dataset):
    """
    NGen-3 Dataset.

    This class implements the dataset class for NGen-3.
    """

    def __init__(self, data_file, tokenizer, max_len):
        """
        Initializes the dataset.

        Args:
            data_file (str): Data file path.
            tokenizer (Tokenizer): Tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(data_file)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Args:
            idx (int): Index.

        Returns:
            dict: Item.
        """
        text = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_data(data_file, tokenizer, batch_size, max_len):
    """
    Loads the data.

    Args:
        data_file (str): Data file path.
        tokenizer (Tokenizer): Tokenizer.
        batch_size (int): Batch size.
        max_len (int): Maximum sequence length.

    Returns:
        DataLoader: Data loader.
    """
    dataset = NGen3Dataset(data_file, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Example usage
tokenizer = ...  # Initialize your tokenizer
data_file = "train.txt"
batch_size = 32
max_len = 512
data_loader = load_data(data_file, tokenizer, batch_size, max_len)