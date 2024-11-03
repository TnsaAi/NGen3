import torch
import torch.nn as nn
import math


from NGen3MultiHeadAttention import MultiHeadAttention
from NGen3FeedForwardNetwork import FeedForwardNetworkBlock


class DecoderLayer(nn.Module):
    """
    Decoder Layer.

    This class implements a decoder layer with self-attention, encoder-attention, and feed forward network.

    Attributes:
        embedding_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, embedding_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initializes the decoder layer.

        Args:
            embedding_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(DecoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward_network = FeedForwardNetworkBlock(embedding_dim, hidden_dim, dropout)

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            decoder_input (torch.Tensor): Decoder input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = decoder_input
        decoder_input = self.self_attention(decoder_input, decoder_input)
        decoder_input = decoder_input + residual
        residual = decoder_input
        decoder_input = self.encoder_attention(decoder_input, encoder_output)
        decoder_input = decoder_input + residual
        residual = decoder_input
        decoder_input = self.feed_forward_network(decoder_input)
        decoder_input = decoder_input + residual
        return decoder_input


class Decoder(nn.Module):
    """
    Decoder.

    This class implements a decoder with multiple decoder layers.

    Attributes:
        num_layers (int): Number of decoder layers.
        embedding_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, num_layers: int, embedding_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initializes the decoder.

        Args:
            num_layers (int): Number of decoder layers.
            embedding_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            decoder_input (torch.Tensor): Decoder input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, encoder_output)
        return decoder_input