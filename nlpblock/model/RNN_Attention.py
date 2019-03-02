import torch
import torch.nn as nn
import nlpblock as nb

class RNN_Attention(nn.Module):
    def __init__(self, emb_dim,
                 n_class, n_hidden, bidirectional=False, linearTransform=True):
        super(RNN_Attention, self).__init__()

        self.n_hidden = n_hidden
        self.num_directions = 2 if bidirectional is True else 1

        self.rnn = nb.RNN(emb_dim, n_hidden, bidirectional=bidirectional)
        self.attention = nb.AttentionOne(n_class, n_hidden,
                                         bidirectional=bidirectional, linearTransform=linearTransform)

    def hidden_init(self, input):
        batch = input.size(0)
        return torch.rand([self.num_directions, batch, self.n_hidden])

    def forward(self, input):
        outputs, _ = self.rnn(input, self.hidden_init(input))
        return outputs, self.attention(outputs)

"""
Example to run
model = RNN_Attention(emb_dim=50,
                         n_class=2, n_hidden=128, bidirectional=False, linearTransform=True)
output, attention = model(
    torch.rand([3, 5, 50])  # [batch, seq_len, emb_dim]
)
print(output.shape, attention.shape)
"""
