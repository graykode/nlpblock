import torch
import torch.nn as nn
from nlpblock.layer.Attention import Attention

class AttentionOne(Attention):
    """
    Classes for only `ONE` time sequence Model such as RNN, LSTM, etc
    When forward in this class, We make query from only RNN, LSTM outputs parameter(=All time steps of output in time sequence Model)
    """
    def __init__(self, n_class, n_hidden, bidirectional=False, linearTransform=True):
        super(AttentionOne, self).__init__(n_hidden, bidirectional, linearTransform)
        self.n_class = n_class
        self.num_directions = 2 if bidirectional is True else 1
        self.classifier = nn.Linear(self.n_hidden, n_class, bias=False)

    def forward(self, outputs, first=0, last=-1):
        """
        In this function, query in Attention will be made from outputs matrix
        :param outputs: a matrix of all time steps of output in RNN ,LSTM, etc(=context),
                Shape : [batch, enc_seq_len, n_hidden * num_directions]
        :param first index in a matrix of all time steps of output in RNN ,LSTM, etc
        :param last index in a matrix of all time steps of output in RNN ,LSTM, etc
        :return: final output, Shape [batch, n_class] for classification
        """

        # reshape outputs matrix to [batch, seq_len, num_directions, n_hidden]
        batch, seq_len,  n_hidden = outputs.size(0), outputs.size(1), outputs.size(-1) // self.num_directions
        outputs = outputs.view(batch, seq_len, self.num_directions, n_hidden)

        # make query in outputs matrix, Shape : [batch, num_directions, n_hidden]
        query = torch.empty([batch, self.num_directions, n_hidden])
        if self.bidirectional is True:
            for i in range(batch):
                query[i][1] = outputs[i][first][-1] # get vector outputs of first time step, backward direction
                query[i][0] = outputs[i][last][0]   # get vector outputs of last time step, forward direction
        else:
            for i in range(batch):
                query[i][0] = outputs[i][last][0]  # get vector outputs of last time step, forward direction

        # reshape to original matrix shape
        outputs = outputs.view(batch, seq_len, -1) # output to [batch, seq_len, num_directions  n_hidden]
        query = query.view(batch, -1)              # query  to [batch, num_directions * n_hidden]

        attn_softmax_vector = self.get_attn_vector(outputs, query)
        context_vector = attn_softmax_vector.bmm(outputs).squeeze(1)

        return self.classifier(context_vector)