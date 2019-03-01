import torch
import torch.nn as nn
from nlpblock.layer.Attention import Attention

class AttentionTwo(Attention):
    """
    Classes for relationship-attention of `TWO` models such as Encoder between Decoder
    Each Model should have same Type.
    In general, I call first model as Encdoer, second model as Decoder
    """

    def __init__(self, n_dec_vocab, n_hidden, bidirectional=False, linearTransform=True):
        super(AttentionTwo, self).__init__(n_hidden, bidirectional, linearTransform)
        self.n_dec_vocab = n_dec_vocab
        self.num_directions = 2 if bidirectional is True else 1
        self.classifier = nn.Linear(self.n_hidden * 2, n_dec_vocab, bias=False)

    def forward(self, enc_output, dec_output):
        """
        :param enc_output: a matrix of all time steps of output in encoder(=context), Shape : [batch, enc_seq_len, n_hidden * num_directions]
        :param dec_output: a matrix of all time steps of output in decoder(=query)  , Shape : [batch, dec_seq_len, n_hidden * num_directions]
        :return: concatenated dec_output and attention weight between encoder and decoder
                first return : mixed output between decoder output and context vector
                second return : softmax-nize attention vector
                , Shape : [batch, dec_seq_len, n_dec_vocab], [batch, dec_seq_len. enc_seq_len]
        """

        # Make Attention between Encoder and Decoder
        batch, dec_seq_len = dec_output.size(0), dec_output.size(1)
        enc_seq_len = enc_output.size(1)

        # Model output(mixed dec_output bewteen attn_softmax_vector), Attention weight
        output = torch.empty([dec_seq_len, batch, self.n_dec_vocab])
        attention = torch.empty([dec_seq_len, batch, enc_seq_len])

        for i in range(dec_seq_len):
            # Divide the decoder into time steps to calculate the total output and the content of the encoder.
            F, query = enc_output, dec_output[:, i, :]

            # make attention weight with softmax fucntion by time step
            attn_softmax_vector = self.get_attn_vector(F, query)
            attention[i] = attn_softmax_vector.squeeze(1)

            context_vector = attn_softmax_vector.bmm(enc_output)

            query = query.unsqueeze(1)
            output[i] = self.classifier(torch.cat((query, context_vector), dim=2)).squeeze(1)

        # final output shape : [batch, dec_seq_len, n_dec_vocab], [batch, dec_seq_len. enc_seq_len]
        return output.transpose(0, 1), attention.transpose(0, 1)