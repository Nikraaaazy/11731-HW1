# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, create_tensor
from vocab import Vocab, VocabEntry
from model import *
import torch
from tqdm import tqdm



Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references], [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    # List[List[str]]
    vocab = pickle.load(open(args['--vocab'], 'rb'))

    train_data_src = create_tensor(read_corpus(args['--train-src'], source='src'), vocab.src, args["--cuda"])
    train_data_tgt = create_tensor(read_corpus(args['--train-tgt'], source='tgt'), vocab.tgt, args["--cuda"])

    dev_data_src = create_tensor(read_corpus(args['--dev-src'], source='src'), vocab.src, args["--cuda"])
    dev_data_tgt = create_tensor(read_corpus(args['--dev-tgt'], source='tgt'), vocab.tgt, args["--cuda"])

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    all_data = {"Training": train_data, "Validation": dev_data}

    batch_size = int(args['--batch-size'])
    model_save_path = args['--save-to']

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -0.1, 0.1)
    if args["--cuda"]:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    best_ppl = float("inf")
    best_model = None

    for epoch in range(10):
        for phase, data in all_data.items():
            print(f"{phase} Phase")
            total_loss = 0
            total_iter = 0
            temp_loss = 0
            temp_iter = 0
            for source, target, target_mask in tqdm(batch_iter(data, batch_size=batch_size, shuffle=True), total=len(data) // batch_size):
                # if args["--cuda"]:
                #     source, target, target_mask = source.cuda(), target.cuda(), target_mask.cuda()
                optimizer.zero_grad()
                if phase == "Training":
                    model.train()
                    output = model(source, target)
                    target_label = target.roll(-1, dims=0)
                    loss = criterion(output[target_mask], target_label[target_mask])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        output = model(source, target)
                        target_label = target.roll(-1, dims=0)
                        loss = criterion(output[target_mask], target_label[target_mask])
                total_loss += loss
                total_iter += 1
                temp_loss += loss
                temp_iter += 1

                if temp_iter % 10 == 0:
                    print(f"Iter: {total_iter} PPL: {torch.exp(temp_loss / temp_iter).item()}")
                    temp_loss = 0
                    temp_iter = 0
            total_ppl = torch.exp(total_loss / total_iter).item()
            print(f"Total PPL: {total_ppl}")
            if phase == "Validation" and total_ppl < best_ppl:
                best_ppl = total_ppl
                if args["--cuda"] and torch.cuda.device_count() > 1:
                    best_model = model.module.state_dict()
                else:
                    best_model = model.state_dict()
    torch.save(best_model, model_save_path)



def beam_search(model, test_data_src, beam_size: int, max_decoding_time_step: int):

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        src_sent = src_sent.unsqueeze(1).cuda()
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
        for i in len(example_hyps):
            example_hyps[i].value = [model.vocab.tgt.id2word(x) for x in example_hyps[i].value if x != 1 and x != 2]
        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    vocab = pickle.load(open(args['--vocab'], 'rb'))
    test_data_src = create_tensor(read_corpus(args['TEST_SOURCE_FILE'], source='src'), vocab.src, args["--cuda"])

    if args['TEST_TARGET_FILE']:
        test_data_tgt = create_tensor(read_corpus(args['TEST_TARGET_FILE'], source='tgt'), vocab.src, args["--cuda"])

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.load_state_dict(torch.load(args['MODEL_PATH']))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)


    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    torch.manual_seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')

if __name__ == '__main__':
    main()
