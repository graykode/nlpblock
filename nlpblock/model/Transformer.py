import torch
import torch.nn as nn
import nlpblock as nb

class Transofmer(nn.Module):
    def __init__(self, n_enc_vocab, n_dec_vocab):
        super(Transofmer, self).__init__()
        self.n_enc_vocab = n_enc_vocab
        self.n_dec_vocab = n_dec_vocab

    def forward(self, *input):
        return 1

model = Transofmer(n_enc_vocab=20, n_dec_vocab=30)
output,attention = model(
    torch.rand([3, 5, 20]), # [batch, enc_seq_len, n_enc_vocab]
    torch.rand([3, 7, 30])  # [batch, den_seq_len, n_dec_vocab]
)