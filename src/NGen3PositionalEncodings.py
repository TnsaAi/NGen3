import torch
import torch.nn as nn
import math
from typing import Tuple

class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding.

    This class implements learned positional encoding, which is an alternative
    to the traditional sinusoidal positional encoding.

    Attributes:
        max_len (int): Maximum sequence length.
        embedding_dim (int): Embedding dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, max_len: int, embedding_dim: int, dropout: float = 0.1):
        """
        Initializes the learned positional encoding.

        Args:
            max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(LearnedPositionalEncoding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with positional encoding.
        """
        return x + self.positional_encoding[:, :x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    This class implements the traditional sinusoidal positional encoding.

    Attributes:
        max_len (int): Maximum sequence length.
        embedding_dim (int): Embedding dimension.
    """

    def __init__(self, max_len: int, embedding_dim: int):
        """
        Initializes the sinusoidal positional encoding.

        Args:
            max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.positional_encoding = self.get_positional_encoding()

    def get_positional_encoding(self) -> torch.Tensor:
        """
        Computes the sinusoidal positional encoding.

        Returns:
            torch.Tensor: Sinusoidal positional encoding tensor.
        """
        positional_encoding = torch.zeros((self.max_len, self.embedding_dim))
        for pos in range(self.max_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embedding_dim)))
                if i + 1 < self.embedding_dim:
                    positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embedding_dim)))
        return positional_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with positional encoding.
        """
        return x + self.positional_encoding[:x.size(1), :]


class PositionalEncoding(nn.Module):
    """
    Positional Encoding.

    This class implements a wrapper around different positional encoding schemes.

    Attributes:
        max_len (int): Maximum sequence length.
        embedding_dim (int): Embedding dimension.
        encoding_type (str): Type of positional encoding.
    """

    def __init__(self, max_len: int, embedding_dim: int, encoding_type: str = "sinusoidal"):
        """
        Initializes the positional encoding.

        Args:
            max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            encoding_type (str, optional): Type of positional encoding. Defaults to "sinusoidal".
        """
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.encoding_type = encoding_type
        if encoding_type == "learned":
            self.positional_encoding = LearnedPositionalEncoding(max_len, embedding_dim)
        elif encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(max_len, embedding_dim)
        else:
            raise ValueError("Invalid encoding type")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with positional encoding.
        """
        return self.positional_encoding(x)