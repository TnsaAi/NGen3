# coding=utf-8
# Copyright (c) 2024, TNSAAI Inc. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the NGen2 Community License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://tnsaai.github.io/-/community/licenses/ngen2/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Copyright (c) 2024, Meta's PyTorch Development Team. All rights reserved.

#Before Commiting changes to this Model make sure you read NGen2 Community License.

import torch
import torch.nn as nn
import math

from src.TNSA_Standard_Libv2.scripts.NGen3TransformerDecoder import TransformerDecoder
from src.TNSA_Standard_Libv2.scripts.NGen3TransformerEncoder import TransformerEncoder

class NGen3Model(nn.Module):
    """
    NGen-3 Model.

    This class implements the NGen-3 model with encoder and decoder.

    Attributes:
        encoder (nn.Module): Encoder.
        decoder (nn.Module): Decoder.
        embedding_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, encoder, decoder, embedding_dim, num_heads, hidden_dim, dropout):
        """
        Initializes the NGen-3 model.

        Args:
            encoder (nn.Module): Encoder.
            decoder (nn.Module): Decoder.
            embedding_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension.
            dropout (float): Dropout probability.
        """
        super(NGen3Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        encoder_output = self.encoder(input_ids, attention_mask)
        decoder_output = self.decoder(encoder_output, attention_mask)
        return decoder_output


class NGen3Config:
    """
    NGen-3 Configuration.

    This class implements the NGen-3 configuration.

    Attributes:
        hidden_size (int): Hidden size.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of layers.
        dropout (float): Dropout probability.
        max_position_embeddings (int): Maximum position embeddings.
    """

    def __init__(self, hidden_size, num_heads, num_layers, dropout, max_position_embeddings):
        """
        Initializes the NGen-3 configuration.

        Args:
            hidden_size (int): Hidden size.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers.
            dropout (float): Dropout probability.
            max_position_embeddings (int): Maximum position embeddings.
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings


def build_ngen3_model(config):
    """
    Builds the NGen-3 model.

    Args:
        config (NGen3Config): NGen-3 configuration.

    Returns:
        NGen3Model: NGen-3 model.
    """
    encoder = TransformerEncoder(config.num_layers, config.d_model, config.nhead, config.dim_feedforward, config.dropout)
    decoder = TransformerDecoder(config.num_layers, config.d_model, config.nhead, config.dim_feedforward, config.dropout)
    model = NGen3Model(encoder, decoder, config.d_model, config.nhead, config.d_model, config.dropout)
    return model





import torch
import torch.nn as nn
from torch.nn import MultiHeadAttention, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.linear1(src2)))
        src = src + self.dropout2(src2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output
    


    import torch
import torch.nn as nn
from torch.nn import MultiHeadAttention, LayerNorm


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt2, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.linear1(tgt2)))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output
    






    import torch
import torch.nn as nn
import math
from typing import Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    This class implements the multi-head attention mechanism with additional features.

    Attributes:
        num_heads (int): Number of attention heads.
        embedding_dim (int): Embedding dimension.
        dropout (float): Dropout probability.
        use_query_linear (bool): Whether to use linear transformation for query.
        use_key_linear (bool): Whether to use linear transformation for key.
        use_value_linear (bool): Whether to use linear transformation for value.
    """

    def __init__(self, num_heads: int, embedding_dim: int, dropout: float = 0.1,
                 use_query_linear: bool = True, use_key_linear: bool = True, use_value_linear: bool = True):
        """
        Initializes the multi-head attention.

        Args:
            num_heads (int): Number of attention heads.
            embedding_dim (int): Embedding dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            use_query_linear (bool, optional): Whether to use linear transformation for query. Defaults to True.
            use_key_linear (bool, optional): Whether to use linear transformation for key. Defaults to True.
            use_value_linear (bool, optional): Whether to use linear transformation for value. Defaults to True.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.use_query_linear = use_query_linear
        self.use_key_linear = use_key_linear
        self.use_value_linear = use_value_linear

        if use_query_linear:
            self.query_linear = nn.Linear(embedding_dim, embedding_dim)
        if use_key_linear:
            self.key_linear = nn.Linear(embedding_dim, embedding_dim)
        if use_value_linear:
            self.value_linear = nn.Linear(embedding_dim, embedding_dim)

        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
        self.attention_layer = SelfAttention(embedding_dim, num_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.use_query_linear:
            query = self.query_linear(query)
        if self.use_key_linear:
            key = self.key_linear(key)
        if self.use_value_linear:
            value = self.value_linear(value)

        attention_output = self.attention_layer(query, key, value)
        output = self.out_linear(attention_output)
        return output


class SelfAttention(nn.Module):
    """
    Self-Attention.

    This class implements the self-attention mechanism.

    Attributes:
        embedding_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        """
        Initializes the self-attention.

        Args:
            embedding_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
        """
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.scaling_factor = math.sqrt(embedding_dim // num_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / self.scaling_factor
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = nn.functional.dropout(attention_weights, p=0.1)
        output = torch.matmul(attention_weights, value)
        return output
    


    # Import necessary libraries
import torch
import torch.nn as nn

# Define a LayerNorm class that inherits from PyTorch's nn.Module
class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        hidden_size (int): Size of the hidden dimension.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, hidden_size, eps=1e-6):
        # Initialize the parent class
        super(LayerNorm, self).__init__()
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # Set epsilon value for numerical stability
        self.eps = eps

    # Define the forward pass
    def forward(self, x):
        """
        Normalize input tensor across the last dimension.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized tensor.
        """
        # Calculate mean across the last dimension
        mean = x.mean(-1, keepdim=True)
        # Calculate standard deviation across the last dimension
        std = x.std(-1, keepdim=True)
        # Normalize input tensor
        return self.weight * (x - mean) / (std + self.eps) + self.bias
    
    # Import necessary libraries
import torch
import torch.nn as nn

# Define an EmbeddingLayer class that inherits from PyTorch's nn.Module
class EmbeddingLayer(nn.Module):
    """
    Word embedding layer.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of the embeddings.
        padding_idx (int, optional): Index for padding tokens. Defaults to 0.
    """
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        # Initialize the parent class
        super(EmbeddingLayer, self).__init__()
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    # Define the forward pass
    def forward(self, input_ids):
        """
        Map input IDs to dense vectors.

        Args:
            input_ids (Tensor): Input IDs.

        Returns:
            Tensor: Embedded input IDs.
        """
        # Return embedded input IDs
        return self.embedding(input_ids)
    

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


import re
import json
from collections import Counter

class Tokenize:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.byte_encoder = self.build_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def build_byte_encoder(self):
        """Creates byte to unicode mapping for byte-level BPE."""
        byte_encoder = {}
        for i in range(33, 127):
            byte_encoder[i] = chr(i)
        for i in range(161, 172):
            byte_encoder[i] = chr(i)
        for i in range(174, 256):
            byte_encoder[i] = chr(i)
        for i in range(256):
            if i not in byte_encoder:
                byte_encoder[i] = chr(256 + i)
        return byte_encoder

    def bytes_to_unicode(self, text):
        """Encodes text into byte-level."""
        return ''.join(self.byte_encoder[byte] if byte in self.byte_encoder else chr(byte) for byte in text)

    def pre_tokenize(self, text):
        """Splits text into words and handles spaces."""
        text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces
        return [self.bytes_to_unicode(text.encode('utf-8'))]  # Encode text into byte-level

    def get_vocab(self):
        return self.vocab

    def train_tokenizer(self, corpus):
        """Train the tokenizer on the corpus."""
        # Pre-tokenize the corpus into byte-level tokens
        tokenized_corpus = [self.pre_tokenize(text) for text in corpus]
        
        # Count all symbol pairs in the corpus
        vocab = Counter()
        for tokens in tokenized_corpus:
            for token in tokens:
                token = ' '.join(token)
                vocab[token] += 1

        # Build initial vocabulary (single characters or bytes)
        vocab_items = sorted(vocab.items(), key=lambda x: -x[1])
        self.vocab = {token: i for i, (token, _) in enumerate(vocab_items)}

        # Learn BPE merges
        self.learn_bpe(vocab)

    def learn_bpe(self, vocab):
        """Learn the BPE merges."""
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges.append(best)

    def get_stats(self, vocab):
        """Get symbol pair frequencies."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """Merge the most frequent pair into a single symbol."""
        new_vocab = {}
        bigram = ' '.join(pair)
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        for word in vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def tokenize(self, text):
        """Tokenizes a given text using byte-level BPE."""
        tokens = []
        text = self.pre_tokenize(text)[0]
        token = ' '.join(text)
        for merge in self.merges:
            token = token.replace(' '.join(merge), ''.join(merge))
        tokens.append(self.vocab[token])
        return tokens

    def save_vocab(self, filename):
        """Save the vocabulary and merges."""
        with open(filename, 'w') as f:
            json.dump({
                "vocab": {str(k): v for k, v in self.vocab.items()},
                "merges": self.merges
            }, f)

# Example usage
if __name__ == "__main__":
    # Sample corpus
    corpus = [
        "Byte Pair Encoding is efficient.",
        "Tokenization helps with text generation."
    ]
    
    # Instantiate tokenizer
    tokenizer = Tokenize(vocab_size=50)
    
    # Train tokenizer
    tokenizer.train_tokenizer(corpus)
    
    # Tokenize a sample text
    sample_text = "Encoding is useful."
    tokenized_output = tokenizer.tokenize(sample_text)
    
    # Print results
    print(f"Tokenized text: {tokenized_output}")
    
    # Save vocabulary
    tokenizer.save_vocab("tokenizer_vocab.json")



import torch
from torch.utils.data import DataLoader



class NGen3DataLoader(DataLoader):
    """
    Custom DataLoader for NGen3Dataset.

    Args:
        dataset (NGen3Dataset): Dataset instance.
        batch_size (int): Batch size.
        shuffle (bool, optional): Shuffle data. Defaults to True.
        num_workers (int, optional): Number of worker threads. Defaults to 4.
        pin_memory (bool, optional): Pin memory. Defaults to True.
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        super(NGen3DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function.

        Args:
            batch (list): List of data samples.

        Returns:
            dict: Collated data.
        """
        input_ids = torch.stack([sample['input_ids'] for sample in batch])
        attention_mask = torch.stack([sample['attention_mask'] for sample in batch])
        labels = torch.stack([sample['labels'] for sample in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    



import torch
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.TNSA_Standard_Libv2.scripts.NGen3Model import build_ngen3_model, config

class NGen3TrainingLoop:
    """
    NGen-3 Training Loop.

    This class implements the training loop for NGen-3.
    """

    def __init__(self, model, device, train_dataset, batch_size, epochs, learning_rate, weight_decay):
        """
        Initializes the training loop.

        Args:
            model (nn.Module): NGen-3 model.
            device (torch.device): Device.
            train_dataset (Dataset): Training dataset.
            batch_size (int): Batch size.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
        """
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train(self):
        """
        Trains the model.
        """
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * self.epochs)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ...  # Load your dataset
model = build_ngen3_model(config)
training_loop = NGen3TrainingLoop(model, device, train_dataset, batch_size=32, epochs=5, learning_rate=1e-5, weight_decay=0.01)
training_loop.train()





import torch
from transformers.optimization import AdamW


class NGen3Optimizer:
    """
    NGen-3 Optimizer.

    This class implements the AdamW optimizer for NGen-3.
    """

    def __init__(self, model, learning_rate, weight_decay):
        """
        Initializes the AdamW optimizer.

        Args:
            model (nn.Module): NGen-3 model.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def get_optimizer(self):
        """
        Returns the AdamW optimizer.

        Returns:
            AdamW: AdamW optimizer.
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)



import torch
from transformers.optimization import get_linear_schedule_with_warmup


class LinearScheduleWithWarmup:
    """
    Linear schedule with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        last_epoch (int, optional): Last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch
        )

    def step(self):
        """
        Update scheduler step.
        """
        self.scheduler.step()

    def state_dict(self):
        """
        Get scheduler state dictionary.

        Returns:
            dict: Scheduler state.
        """
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load scheduler state dictionary.

        Args:
            state_dict (dict): Scheduler state.
        """
        self.scheduler.load_state_dict(state_dict)





        import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from rouge import Rouge


class NGen3EvaluationMetrics:
    """
    NGen-3 Evaluation Metrics.

    This class implements the evaluation metrics for NGen-3.
    """

    def __init__(self, model, device, data_loader):
        """
        Initializes the evaluation metrics.

        Args:
            model (nn.Module): NGen-3 model.
            device (torch.device): Device.
            data_loader (DataLoader): Data loader.
        """
        self.model = model
        self.device = device
        self.data_loader = data_loader

    def evaluate(self):
        """
        Evaluates the model.

        Returns:
            dict: Evaluation metrics.
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_labels = []
        total_preds = []

        with torch.no_grad():
            for batch in self.data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.scores, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_labels.extend(labels.cpu().numpy())
                total_preds.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(total_labels, total_preds)
        f1 = f1_score(total_labels, total_preds, average="macro")
        rouge = Rouge()
        rouge_score = rouge.get_scores(total_preds, total_labels, avg=True)

        return {
            "loss": total_loss / len(self.data_loader),
            "accuracy": accuracy,
            "f1": f1,
            "rouge": rouge_score,
            "classification_report": classification_report(total_labels, total_preds),
            "confusion_matrix": confusion_matrix(total_labels, total_preds),
        }


# Example usage
model = build_ngen3_model(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = load_data("test.txt", tokenizer, batch_size=32, max_len=512)
evaluation_metrics = NGen3EvaluationMetrics(model, device, data_loader)
metrics = evaluation_metrics.evaluate()
print(metrics)





import rouge


class ROUGE:
    """
    ROUGE evaluation metric.

    Args:
        rouge_type (str): ROUGE type (e.g., "rouge-1", "rouge-2", "rouge-l").
    """
    def __init__(self, rouge_type):
        self.rouge_type = rouge_type

    def evaluate(self, predictions, references):
        """
        Evaluate ROUGE score.

        Args:
            predictions (list): Predicted texts.
            references (list): Reference texts.

        Returns:
            dict: ROUGE scores.
        """
        scores = rouge.RougeMetric().compute(predictions, references)
        return scores[self.rouge_type]
    



    import torch
import torch.nn.functional as F


class Perplexity:
    """
    Perplexity evaluation metric.
    """
    def evaluate(self, logits, labels):
        """
        Evaluate perplexity.

        Args:
            logits (Tensor): Model output logits.
            labels (Tensor): Ground truth labels.

        Returns:
            float: Perplexity score.
        """
        loss = F.cross_entropy(logits, labels)
        perplexity = torch.exp(loss)
        return perplexity.item()
    


    import torch


class Accuracy:
    """
    Accuracy evaluation metric.
    """
    def evaluate(self, predictions, labels):
        """
        Evaluate accuracy.

        Args:
            predictions (Tensor): Predicted labels.
            labels (Tensor): Ground truth labels.

        Returns:
            float: Accuracy score.
        """
        correct = (predictions == labels).sum()
        accuracy = correct / len(labels)
        return accuracy.item()
    



    import json


class Config:
    """
    Configuration class.

    Args:
        config_file (str): Configuration file path.
    """
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get(self, key):
        """
        Get configuration value.

        Args:
            key (str): Configuration key.

        Returns:
            value: Configuration value.
        """
        return self.config.get(key)
    




import logging


class Logger:
    """
    Logger class.

    Args:
        log_file (str): Log file path.
    """
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO)

    def info(self, message):
        """
        Log info message.

        Args:
            message (str): Log message.
        """
        logging.info(message)

    def error(self, message):
        """
        Log error message.

        Args:
            message (str): Log message.
        """
        logging.error(message)
        
# coding=utf-8
# Copyright (c) 2024, TNSAAI Inc. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#Copyright (c) 2024, Google's Tensorflow Development Team. All rights reserved.
"""This Code is Copyrighted by TNSA AI"""
