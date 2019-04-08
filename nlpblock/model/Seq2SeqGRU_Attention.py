import torch
import torch.nn as nn
import nlpblock as nb

class Seq2SeqGRU_Attention(nn.Module):
    def __init__(self, n_enc_vocab, n_dec_vocab, n_hidden,
                 n_layers=1, bidirectional=False, linearTransform=True):
        super(Seq2SeqGRU_Attention, self).__init__()

        self.n_hidden = n_hidden
        self.num_directions = 2 if bidirectional is True else 1

        self.encoder = nb.GRU(input_size=n_enc_vocab, hidden_size=n_hidden, bidirectional=bidirectional)
        self.decoder = nb.GRU(input_size=n_dec_vocab, hidden_size=n_hidden, bidirectional=bidirectional)

        self.attention = nb.AttentionTwo(n_dec_vocab, n_hidden,
                                         n_layers=n_layers, bidirectional=bidirectional, linearTransform=linearTransform)

    def hidden_init(self, enc_input):
        batch = enc_input.size(0)
        return torch.rand([self.num_directions , batch, self.n_hidden])

    def forward(self, enc_input, dec_input):
        init_hidden = self.hidden_init(enc_input)

        # Make Original Seq2Seq Model
        enc_output, final_enc_hidden = self.encoder(enc_input, init_hidden)
        dec_output, _ = self.decoder(dec_input, final_enc_hidden)

        # Calculate Attention Weight
        output, attention = self.attention(enc_output, dec_output)
        return output, attention

"""
Example to run
model = Seq2SeqGRU_Attention(n_enc_vocab=20, n_dec_vocab=30,
                                    n_layers=1, n_hidden=128, bidirectional=False, linearTransform=True)
output,attention = model(
    torch.rand([3, 5, 20]), # [batch, enc_seq_len, n_enc_vocab]
    torch.rand([3, 7, 30])  # [batch, den_seq_len, n_dec_vocab]
)
print(output.shape, attention.shape)
"""