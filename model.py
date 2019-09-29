from typing import List
import torch.nn as nn
import torch.tensor as Tensor
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import math
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    A highly oversimplified version of Multihead attention
    """
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.scale = math.sqrt(self.head_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.q_bias = nn.Parameter(torch.tensor(hidden_size))
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.k_bias = nn.Parameter(torch.tensor(hidden_size))
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.v_bias = nn.Parameter(torch.tensor(hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        :param q: Target sequence, (T_t, B, hidden_size) T_t = 1 at prediction time
        :param k: Source sequence, (T_s, B ,hidden_size)
        :param v: Source sequence, (T_s, B ,hidden_size)
        :return: (T_t, B, hidden_size)
        """
        T_t, B, _ = q.size()
        T_s, _, _ = k.size()
        q = F.relu6(self.q_proj(q) + self.q_bias).view(T_t, B, self.num_heads, self.head_size).view(T_t, B * self.num_heads, -1).permute(1, 0, 2)
        k = F.relu6(self.k_proj(k) + self.k_bias).view(T_s, B, self.num_heads, self.head_size).view(T_s, B * self.num_heads, -1).permute(1, 2, 0)
        v = F.relu6(self.v_proj(v) + self.v_bias).view(T_s, B, self.num_heads, self.head_size).view(T_s, B * self.num_heads, -1).permute(1, 0, 2)
        # Scaled dot product
        product = torch.bmm(q, k) / self.scale
        score = F.softmax(product, dim=-1)
        output = torch.bmm(score, v)
        output = output.permute(1, 0, 2).reshape(T_t, B, self.num_heads, self.head_size).view(T_t, B, -1)
        return output

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
        self.encoder = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.GRU(input_size=embed_size, hidden_size=2*hidden_size, num_layers=num_layers)
        self.multihead = MultiheadAttention(num_heads=4, hidden_size=hidden_size)
        self.linear = nn.Sequential(
                        nn.Linear(4*hidden_size, hidden_size),
                        nn.ReLU6(),
                        nn.Linear(hidden_size, len(vocab.tgt))
        )

    def forward(self, src_sents: Tensor, tgt_sents: Tensor) -> Tensor:
        """
        :param src_sents: (T * B) Padded source sequence, masking will be handled by pack padded sequence
        :param tgt_sents: (T * B) Padded target sequence, masking will be handled by masking
        :return: logits (T * B * target_vocab_size)
        """
        source_length = (src_sents != 0).sum(dim=0)
        src_sents = self.source_embedding(src_sents)
        src_sents = pack_padded_sequence(src_sents, source_length)
        _, h = self.encoder(src_sents)
        _, B, V = h.size()
        h = h.reshape(self.num_layers, 2, B, V).permute(0, 2, 1, 3).reshape(self.num_layers, B, -1)
        tgt_sents = self.target_embedding(tgt_sents)
        output, h_dec = self.decoder(tgt_sents, h)
        attention = self.multihead(h_dec, h, h)
        output = self.linear(torch.cat((output, attention), dim=-1))
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

