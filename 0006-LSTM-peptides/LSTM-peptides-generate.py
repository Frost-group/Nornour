import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import unidecode
from torch.utils.tensorboard import SummaryWriter
def _onehotencode(seq, vocab):

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

def padding_len(filepath):
    # Read the file to find the longest line
    with open(filepath, 'r') as file:
        lines = file.readlines()  # Read all lines
        len_longest = max(len(line.strip()) for line in lines)

    return len_longest

def padding(filepath, max_len):
     # Pad lines to make them all the same length
    with open(filepath, 'w') as file:
        lines = file.readlines()
        for line in lines:
            padded_line = line.strip().ljust(max_len, '_')  # Pad each line with '_'
            file.write(padded_line + '\n')  # Write the padded line back to the file