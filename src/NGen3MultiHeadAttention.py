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