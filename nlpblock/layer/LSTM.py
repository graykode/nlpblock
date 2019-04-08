'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import torch.nn as nn

class LSTM(nn.Module):
    """
        All parameter same torch.nn default setting
        except, nonlinearity(='relu') batch_first(=True), bias(=False)
        See more detail in here, https://pytorch.org/docs/stable/nn.html
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=False,
                 batch_first=True,  dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers ,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, input, h_0):
        return self.lstm(input, h_0)