import torch
import torch.nn as nn
import math

from NGen3TransformerDecoder import TransformerDecoder
from NGen3TransformerEncoder import TransformerEncoder

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