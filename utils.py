import math
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence


def read_corpus(file_path, source):
    data = []
    for line in open(file_path, encoding='utf-8'):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    return data

def create_tensor(str_data, vocab_entry):
    return [torch.tensor([vocab_entry[x] for x in sentence]) for sentence in str_data]

def batch_iter(tensor_data: torch.tensor, batch_size, shuffle=True):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    if shuffle:
        random_indices = torch.randperm(len(tensor_data))

    for i in range(0, len(tensor_data), batch_size):
        examples = [tensor_data[idx] for idx in random_indices[i: i + batch_size]]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        source = pad_sequence([e[0] for e in examples])
        target = pad_sequence([e[1] for e in examples])
        target_mask = (target != 0) & (target != 2)
        yield source, target, target_mask
