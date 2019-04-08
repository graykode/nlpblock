'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''
import torch
import numpy as np
import torch.nn as nn

class PosEncoding(nn.Module):
    def __init__(self, n_position, d_model):
        super(PosEncoding, self).__init__()
        self.n_position = n_position
        self.d_model = d_model

    def get_sinusoid_encoding_table(self):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / self.d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(self.d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(self.n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)