import torch
import numpy as np
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask=None):
        attn_vector = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        if attn_mask is not None:
            attn_vector.masked_fill_(attn_mask, -1e9)

        attn_softmax_vector = self.softmax(attn_vector)
        context_vector = torch.matmul(attn_softmax_vector, V)

        return context_vector, attn_softmax_vector