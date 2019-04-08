import torch.nn as nn

from nlpblock.layer import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.SelfAttention = ScaledDotProductAttention(d_model, d_k, d_v, n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask=None):
        residual, batch = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context_vector : [batch, n_heads, len_q x d_v]
        # attn_softmax_vector : [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        context_vector, attn_softmax_vector = self.SelfAttention(q_s, k_s, v_s, attn_mask)

        # context_vector: [batch, len_q(=len_k), n_heads * d_v]
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch, -1, self.n_heads * self.d_v)

        output = self.linear(context_vector)
        output = self.layernorm(output + residual)

        return output