import torch
import torch.nn as nn
import math

class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network.

    This class implements a feed forward network with two linear layers and a ReLU activation function.

    Attributes:
        embedding_dim (int): Embedding dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initializes the feed forward network.

        Args:
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(FeedForwardNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GELU(nn.Module):
    """
    GELU Activation Function.

    This class implements the GELU activation function.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes the GELU activation function.
        """
        super(GELU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForwardNetworkBlock(nn.Module):
    """
    Feed Forward Network Block.

    This class implements a feed forward network block with a feed forward network and layer normalization.

    Attributes:
        embedding_dim (int): Embedding dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initializes the feed forward network block.

        Args:
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(FeedForwardNetworkBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feed_forward_network = FeedForwardNetwork(embedding_dim, hidden_dim, dropout)
        self.layer_normalization = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        x = self.feed_forward_network(x)
        x = self.layer_normalization(x + residual)
        return x