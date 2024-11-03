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