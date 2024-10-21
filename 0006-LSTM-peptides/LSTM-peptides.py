import torch
import numpy as np
import torch.nn as nn
import random
import os

def _onehotencode(seq, vocab=None):

    if not vocab:
        vocab = ['R','H','K','D','E','S','T','N','Q','C','U','G','P','A','I','L','M','F','W','Y','V']

    to_one_hot = dict()
    for i, a in enumerate(vocab):
        v = torch.zeros(len(vocab))
        v[i] = 1
        to_one_hot[a] = v

    result = []
    for l in seq:
        result.append(to_one_hot[l])
    result = np.array(result)
    return torch.Tensor(result), to_one_hot, vocab

class LSTMpeptides(nn.Module):

    def __init__(self, input_d, hidden_d,len_vocab=21, batch=64, layers=1, neurons=64, lr=1e-3):
        super(LSTMpeptides, self).__init__()
        self.input_d = input_d
        self.hidden_d = hidden_d
        self.len_vocab = len_vocab
        self.batch = batch
        self.layers = layers
        self.neurons = neurons
        self.lr = lr
        _,_,self.vocab = _onehotencode('A')

        self.lstm = nn.LSTM(input_d, hidden_d)

    def forward(self, sequence):
        input, _, _ = _onehotencode(sequence)
        lstm_out, _ = self.lstm(input.view(len(sequence), 1, -1))

