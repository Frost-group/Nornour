import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import unidecode
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

filepath = '../0003c-APD-Database/antimicrobial_peptides_database.txt'
with open(filepath) as f:
    peptides = [line.strip() for line in f.readlines()]

vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
len_vocab = len(vocab)


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

batch_size = 1
seq_len = 12
hidden_size = 128
n_epochs = 5000
num_layers = 2
neurons= 64
lr = 0.001
chunk_len = 250

#model = LSTMpeptides(input_d=input_len, hidden_d=hidden_size, len_vocab=22, batch=batch_size, layers=num_layers, neurons=neurons, lr=lr)



class Generator():
    def __init__(self, hidden_d, layers,  num_epochs, batch, lr=1e-3):
        self.num_epoch = num_epochs
        self.batch = batch
        self.print_every = 50
        self.hidden_d = hidden_d
        self.layers = layers
        self.lr = lr

    def aa_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = vocab.index(string[c])

        return tensor

    def get_random_batch(self):
        selected_peptides = random.choices(peptides, k=self.batch)

        # Prepare lists to store input and target tensors of varying lengths
        text_input = []
        text_target = []

        for peptide in selected_peptides:
            peptide_len = len(peptide)

            # Input is the peptide up to the second-to-last character
            peptide_input = peptide[:-1]
            # Target is the peptide from the second character onward (prediction target)
            peptide_target = peptide[1:]

            # Convert to tensors
            peptide_input_tensor = self.aa_tensor(peptide_input)
            peptide_target_tensor = self.aa_tensor(peptide_target)

            # Append to the batch
            text_input.append(peptide_input_tensor)
            text_target.append(peptide_target_tensor)

        # Return list of tensors (each peptide may have different length)
        return text_input, text_target

    def generate(self, initial_str = 'A', predict_len=100, temp=1):
        hidden, cell = self.lstm.init_hidden(batch=self.batch)
        initial_input = self.aa_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.lstm(initial_input[p].view(1).to(device), hidden, cell)

        last_aa = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.lstm(initial_input[p].view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temp).exp()
            top_aa = torch.multinomial(output_dist, 1)[0]
            predicted_aa = vocab[top_aa]
            predicted += predicted_aa
            last_aa = self.aa_tensor(predicted_aa)

        return predicted


    def train(self):
        self.lstm = LSTMpeptides(len_vocab, self.hidden_d, len_vocab, self.layers).to(device)

        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f'runs/names0')

        print('===> starting training')

        for epoch in range(1, self.num_epoch + 1 ):
            inp, target = self.get_random_batch()
            hidden, cell = self.lstm.init_hidden(batch=self.batch)

            self.lstm.zero_grad()
            loss = 0

            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.lstm(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f'Loss: {loss}')

            writer.add_scalar('Training loss', loss, global_step=epoch)


genpeptides = Generator(hidden_size, num_layers,  n_epochs, batch_size, lr=1e-3)
genpeptides.train()


