from typing import List
import torch.nn as nn
import torch.tensor as Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import math
import torch.nn.functional as F
from nmt import Hypothesis

class MultiheadAttention(nn.Module):
    """
    A highly oversimplified version of Multihead attention
    """
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.scale = math.sqrt(self.head_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
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
        q = F.relu6(self.q_proj(q)).view(T_t, B, self.num_heads, self.head_size).view(T_t, B * self.num_heads, -1).permute(1, 0, 2)
        k = F.relu6(self.k_proj(k)).view(T_s, B, self.num_heads, self.head_size).view(T_s, B * self.num_heads, -1).permute(1, 2, 0)
        v = F.relu6(self.v_proj(v)).view(T_s, B, self.num_heads, self.head_size).view(T_s, B * self.num_heads, -1).permute(1, 0, 2)
        # Scaled dot product
        product = torch.bmm(q, k) / self.scale
        score = self.dropout(F.softmax(product, dim=-1))
        output = torch.bmm(score, v)
        output = output.permute(1, 0, 2).reshape(T_t, B, self.num_heads, self.head_size).reshape(T_t, B, -1)
        return output

class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, num_layers = 2):
        super(NMT, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.register_buffer("start", torch.ones(1,1).long())
        self.vocab = vocab
        self.num_layers = num_layers
        self.source_embedding = nn.Embedding(len(vocab.src), embed_size, padding_idx=0)
        self.target_embedding = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=0)
        self.encoder = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.GRU(input_size=embed_size, hidden_size=2*hidden_size, num_layers=num_layers)
        self.multihead = MultiheadAttention(num_heads=4, hidden_size=hidden_size * 2)
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

        source_output, h = self.encode(src_sents)
        output, _ = self.decode(source_output, tgt_sents, h)

        return output

    def encode(self, src_sents: Tensor):

        source_length = (src_sents != 0).sum(dim=0)
        src_sents = self.source_embedding(src_sents)
        src_sents = pack_padded_sequence(src_sents, source_length)
        source_output, h = self.encoder(src_sents)
        _, B, V = h.size()
        h = h.reshape(self.num_layers, 2, B, V).permute(0, 2, 1, 3).reshape(self.num_layers, B, -1)
        return pad_packed_sequence(source_output)[0], h

    def decode(self, source_output, tgt_sents, h):

        tgt_sents = self.target_embedding(tgt_sents)
        output, h = self.decoder(tgt_sents, h)
        attention = self.multihead(output, source_output, source_output)
        output = self.linear(torch.cat((output, attention), dim=-1))
        return output, h


    def beam_search(self, src_sent, beam_size: int=5, max_decoding_time_step: int=70):
        source_output, h = self.encode(src_sent)
        beam = [([self.start], h, 0.0)]
        hypotheses = []
        for _ in range(max_decoding_time_step):
            new_beam = []
            for sentence, hidden, ll in beam:
                probs, curr_hidden = self.decode(source_output, sentence[-1], hidden)
                log_p = F.log_softmax(probs, dim=-1)
                values, candidates = torch.topk(log_p, beam_size, dim=-1)
                for l, token in zip(values.unbind(-1), candidates.unbind(-1)):
                    new_beam.append((sentence + [token], curr_hidden, ll + l.flatten().item()))
            new_beam.sort(key=lambda x: x[2] / len(x[0]), reverse=True)[:beam_size - len(hypotheses)]
            beam = []
            for sentence, hidden, ll in new_beam:
                if sentence[-1].flatten().item() == 2:
                    hypotheses.append(Hypothesis(value=[x.flatten().item() for x in sentence], score=ll))
                else:
                    beam.append((sentence, hidden, ll))
            if len(hypotheses) >= beam_size:
                break

        return hypotheses[:beam_size]


