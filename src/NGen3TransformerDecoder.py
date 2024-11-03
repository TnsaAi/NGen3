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