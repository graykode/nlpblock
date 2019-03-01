import torch.nn as nn

class LSTM(nn.Module):
    """
        All parameter same torch.nn default setting
        except, nonlinearity(='relu') batch_first(=True), bias(=False)
        **number of layer is one** if you want to use multi layer LSTM, Please use nn.LSTM in Pytorch
            The reason why I use only single layer Cell is that you have to separate Attention according to the Layer.
        See more detail in here, https://pytorch.org/docs/stable/nn.html
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=False,
                 batch_first=True,  dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1 ,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, input, h_0):
        return self.lstm(input, h_0)