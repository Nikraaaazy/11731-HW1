from typing import List
import torch.nn as nn
import torch.tensor as Tensor
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from custom_lstm import script_lnlstm

class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, num_layers = 2):
        super(NMT, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.num_layers = num_layers
        self.source_embedding = nn.Embedding(len(vocab.src), embed_size, padding_idx=0)
        self.target_embedding = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=0)
        self.encoder = script_lnlstm(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.decoder = script_lnlstm(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.affine = nn.Linear(hidden_size*2, hidden_size)
        self.linear = nn.Linear(hidden_size, len(vocab.tgt))
        self.register_buffer("h_0", torch.zeros(num_layers * 2, 1, hidden_size))
        self.register_buffer("c_0", torch.zeros(num_layers * 2, 1, hidden_size))

    def forward(self, src_sents: Tensor, tgt_sents: Tensor) -> Tensor:
        """
        :param src_sents: (T * B) Padded source sequence, masking will be handled by pack padded sequence
        :param tgt_sents: (T * B) Padded target sequence, masking will be handled by masking
        :return: logits (T * B * target_vocab_size)
        """
        # source_length = (src_sents != 0).sum(dim=0)
        src_sents = self.source_embedding(src_sents)
        # src_sents = pack_padded_sequence(src_sents, source_length)
        _, (h, c) = self.encoder(src_sents, (self.h_0.expand(1, src_sents.size()[1], 1), self.c_0.expand(1, src_sents.size()[1], 1)))
        _, B, V = h.size()
        h = h.reshape(self.num_layers, 2, B, V).permute(0, 2, 3, 1).reshape(self.num_layers, B, -1)
        c = c.reshape(self.num_layers, 2, B, V).permute(0, 2, 3, 1).reshape(self.num_layers, B, -1)
        h = self.affine(h)
        c = self.affine(c)
        tgt_sents = self.target_embedding(tgt_sents)
        output, _ = self.decoder(tgt_sents, (h, c))
        output = self.linear(output)
        return output

    def beam_search(self, src_sent, beam_size: int=5, max_decoding_time_step: int=70):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        return hypotheses

