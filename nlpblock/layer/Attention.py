import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Attention Mechanism.
    This class is super class object about Attention Mechanism.
    Input is F(such as encoder all outputs) and query(such as one time step of decoder)
    I recommend you read first Attention mechanism paper, https://arxiv.org/abs/1409.0473
    """

    def __init__(self, n_hidden, n_layers, bidirectional=False, linearTransform=True):
        """
        :param n_hidden: number of hidden size in Cell units
        :param bidirectional, if True, number of direction is two, false is one, default : True
        :param linearTransform: linear transformation,
                If you use this parameter, [n_hidden*num_direction, n_hidden*num_direction] matrix will product in query.
                False, score(F,query) = F^T * query
                True, score(F, query) = F^T * (W * query)
        """
        super(Attention, self).__init__()
        self.num_directions = 2 if bidirectional is True else 1
        self.n_hidden = n_hidden * self.num_directions
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.linearTransform = linearTransform
        if linearTransform is True:
            self.linear = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def get_attn_vector(self, F, query):
        """
        :param F : a matrix of all time steps of output, [batch, seq_len, num_directions * hidden_size]
        :param query: Matrix that you want to get a attention with F, [batch, num_directions * hidden_size]
        :return: create softmax-nize matrix, [batch, 1, seq_len]
        """
        if self.linearTransform is True:
            query = self.linear(query)

        query = query.unsqueeze(-1)
        attn_vector = torch.bmm(F, query) # α_i→=[α_{i1},α_{i2},...,α_{i{T_x}}] in paper
        return self.softmax(attn_vector).transpose(1, 2)
