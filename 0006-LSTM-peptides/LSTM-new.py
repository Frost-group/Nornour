import torch
import torch.nn as nn



class LSTM_args():
    def __init__(self):
        self.output_size = 22
        self.num_epochs = 100
        self.batch_size = 100
        self.learning_rate = 0.001
        self.hidden_size = 128
        self.layers = 2


class LSTMpeptides(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, layers=2):
        super(LSTMpeptides, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_size = 128
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim )

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.layers, sequence_length, self.lstm_size),
                torch.zeros(self.layers, sequence_length, self.lstm_size))

model = LSTMpeptides(LSTM_args.input_size, LSTM_args.hidden_size, LSTM_args.output_size, LSTM_args.batch_size, LSTM_args.layers)

class Dataset:
    def __init__(self):

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W',
                      'Y', 'V', '_']
        self.len_vocab = len(self.vocab)

    def open_file(self, filepath):
        with open(filepath, 'r') as doc:
            peptides = doc.readlines()
            long_pep = max(len(pep.strip()) for pep in self.peptides)

        return peptides, long_pep

    def padding(self, peptides):
        for i, pep in enumerate(peptides):
            pad_length = self.long_pep - len(pep)

            if pad_length > 0:
                peptides[i] = pep.strip() + '_' * pad_length

        return peptides
    def to_tensor(self):
        self.to_index = {a: i for i, a in enumerate(self.vocab)}

        result = []
        for i, pep in enumerate(self.peptides):
            result.append([self.to_index[aa] for aa in pep.strip()])

        return torch.tensor(result), self.to_index

    def to_amino(self, tensor):
        sequences = []
        for seq in tensor:
            amino_seq = ''.join([self.vocab[idx] for idx in seq])
            sequences.append(amino_seq)
        return sequences


def train(dataset, model, args):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    peptides, sequence_length = dataset.open_file(dataset)
    peptides = peptides.padding(peptides)
    inp_peptides, to_index = peptides.to_tensor()

    for epoch in range(args.num_epoch):
        state_h, state_c = model.init_state(sequence_length)

        for x , y , batch  in inp_peptides:
            optimizer.zero_grad()

            y_pred, (h0, c0) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})





