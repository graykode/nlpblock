'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''
import torch.nn as nn
import nlpblock as nb

class SelfAttnEncoder(nn.Module):
    def __init__(self, n_enc_vocab, n_enc_len, d_model, n_layers=12, mode='plain'):
        """
        :param mode: `plain` or `bert`
            plain : it has original Transformer Model
            bert : it has BERT Model
        """
        super(SelfAttnEncoder, self).__init__()
        if mode is 'plain':
            self.PosEncoding = nb.PosEncoding(n_position=n_enc_len, d_model=d_model)
            self.enc_emb = nn.Embedding(n_enc_vocab, d_model)
            self.pos_emb = nn.Embedding.from_pretrained(self.PosEncoding.get_sinusoid_encoding_table(), freeze=True)
            self.layers = nn.ModuleList([nb.SelfAttnEncoderLayer(n_enc_vocab=n_enc_vocab) for _ in range(n_layers)])

        elif mode is 'bert':
            pass

    def forward(self, *input):
        return 1
