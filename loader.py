import torch
from torchtext.legacy import data, datasets

import os
import pytorch_lightning as pl


def imbda_loader(batch_size=64):
    TEXT = data.Field(sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True)
    trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(trainset, min_freq=5)
    LABEL.build_vocab(trainset)

    vocab_size = len(TEXT.vocab)

    trainset, valset = trainset.split(split_ratio=0.8)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=batch_size,
        shuffle=True, repeat=False)

    return train_iter, val_iter, test_iter, vocab_size
