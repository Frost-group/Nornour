import torch
import torch.nn as nn
import torch.optim as optim

# Hyper-parameters
# input_size = 784 # 28x28
output_size = 22
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = #to determine
hidden_size = 128
layers = 2

def one_hot_encoder():
    pass

def one_hot_decoder():
    pass

def padding():
    pass

class LSTMpeptides(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, layers=2):
        super(LSTMpeptides, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim )

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, sequence, batch):
        h0 = torch.zeros(self.layers, sequence.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.layers, sequence.size(0), self.hidden_size).requires_grad_()

        embed = self.embedding(sequence)
        out, (hn, cn) = self.lstm(embed, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

model = LSTMpeptides(input_size, hidden_size, output_size, batch_size, layers)

class Dataset:
    def __init__(self, file):
        with open(file, 'r') as doc:
            self.peptides = doc.readlines()
            self.long_pep =  max(len(self.peptides.strip()) for pep in self.peptides)

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W',
                      'Y', 'V', '_']
        self.len_vocab = len(self.vocab)

    def padding(self, peptides):
        for i, pep in enumerate(peptides):
            pad_length = self.long_pep - len(pep)

            if pad_length > 0:
                peptides[i] = pep + '_' * pad_length

        return peptides


    def to_tensor(self, peptides):
        input_data = [ [] for pep in peptides]

        for pep in peptides:
