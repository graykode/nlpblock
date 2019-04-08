'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import torch
import numpy as np
import torch.nn as nn

class Masking(nn.Module):
    """

    """
    def __init__(self):
        super(Masking, self).__init__()

    def get_attn_pad_mask(self, seq_q, seq_k, pad_index=0, mask_index=1):
        """
        :param seq_q: []
        :param seq_k:
        :param pad_index: padding index to be changed 
        :return: `seq_k.data` equals `pad_index`, make masking
        """
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(pad_index).unsqueeze(mask_index)  # batch_size x 1 x len_k(=len_q), mask_index is masking
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

    """
    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte()
        return subsequent_mask
    """