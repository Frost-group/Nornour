import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import unidecode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

filepath = input('Filename: ')
file = unidecode.unidecode(open(filepath)).read()

vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
len_vocab = len(vocab)


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


class LSTMpeptides(nn.Module):

    def __init__(self, input_d, hidden_d ,len_vocab=22, layers=1):
        super(LSTMpeptides, self).__init__()
        self.input_d = input_d
        self.hidden_d = hidden_d
        self.len_vocab = len_vocab
        self.layers = layers

        self.embed = nn.Embedding(input_d, hidden_d)

        self.lstm = nn.LSTM(input_size= self.input_d, hidden_size=self.hidden_d, num_layers=self.layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, len_vocab)

    def forward(self, sequence, hidden, cell):
        out = self.embed(sequence)
        out, (hidden, cell) = self.lstm(out.unsqeeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch):
        hidden = torch.zeros(self.layers, batch, self.hidden_d).to(device)
        cell = torch.zeros(self.layers, batch, self.hidden_d).to(device)
        return hidden, cell

#hyperparameters

batch_size = 64
input_len = 'nbr lines in file'
seq_len = padding(filepath)
hidden_size = 128
n_epochs = 20
num_layers = 2
neurons= 64
lr = 0.001

model = LSTMpeptides(input_d=input_len, hidden_d=hidden_size, len_vocab=22, batch=batch_size, layers=num_layers, neurons=neurons, lr=lr)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")


class Generator():
    def __init__(self, filepath, hidden_d, layers,  num_epochs, batch, chunck_len, lr=1e-3):
        self.filepath = filepath
        self.chunk_len = chunck_len
        self.num_epoch = num_epochs
        self.batch = batch
        self.print_every = 5
        self.hidden_d = hidden_d
        self.layers = layers
        self.lr = lr

    def AA_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = vocab.index(string[c])

        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx


