import torch
import torch.nn as nn
import nlpblock as nb

class BiRNN_Attention(nn.Module):
    def __init__(self, n_vocab, emb_dim,
                 n_class, n_hidden, bidirectional=False, linearTransform=True):
        super(BiRNN_Attention, self).__init__()

        self.n_hidden = n_hidden
        self.num_directions = 2 if bidirectional is True else 1

        self.embedding = nn.Embedding(n_vocab, emb_dim)
        self.rnn = nb.RNN(emb_dim, n_hidden, bidirectional=bidirectional)
        self.attention = nb.AttentionOne(n_class, n_hidden,
                                         bidirectional=bidirectional, linearTransform=linearTransform)

    def hidden_init(self, input):
        batch = input.size(0)
        return torch.rand([self.num_directions, batch, self.n_hidden])

    def forward(self, X):
        input = self.embedding(X)
        outputs, _ = self.rnn(input, self.hidden_init(X))
        output = self.attention(outputs)
        return output

"""
Example to run
model = BiRNN_Attention(n_vocab=200, emb_dim=50,
                         n_class=2, n_hidden=128, bidirectional=True, linearTransform=True)
output = model(
    torch.zeros(4, 10).to(torch.long) # [batch, seq_len]
)
print(output.shape)
"""