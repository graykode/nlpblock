'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''
import torch
import torch.nn as nn
import nlpblock as nb

class Transofmer(nn.Module):
    def __init__(self, n_enc_vocab, n_enc_len, n_dec_vocab, n_dec_len, d_model):
        super(Transofmer, self).__init__()
        self.selfAttnEncoder = nb.SelfAttnEncoder(n_enc_vocab=n_enc_vocab,
        n_enc_len=n_enc_len, d_model=d_model)

    def forward(self, *input):
        return 1

model = Transofmer(n_enc_vocab=20, n_enc_len=5, n_dec_vocab=30, n_dec_len=7, d_model=512)
output,attention = model(
    torch.rand([3, 5, 20]), # [batch, n_enc_len, n_enc_vocab]
    torch.rand([3, 7, 30])  # [batch, n_dec_len, n_dec_vocab]
)