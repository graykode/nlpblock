'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''
import torch
import torch.nn as nn
import nlpblock as nb

class SelfAttnEncoderLayer(nn.Module):
    def __init__(self, n_enc_vocab):
        super(SelfAttnEncoderLayer, self).__init__()

    def forward(self, *input):
        return 1
