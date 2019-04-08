'''
    code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import torch.nn as nn

class RNN(nn.Module):
    """
        All parameter same torch.nn default setting
        except, nonlinearity(='relu') batch_first(=True), bias(=False)
        See more detail in here, https://pytorch.org/docs/stable/nn.html
    """
    def __init__(self, input_size, hidden_size, nonlinearity='relu', bias=False,
                 batch_first=True, num_layers=1, dropout=0, bidirectional=False):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          batch_first=batch_first,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, input, h_0):
        return self.rnn(input, h_0)