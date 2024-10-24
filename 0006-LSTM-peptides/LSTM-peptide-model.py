import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class LSTMArgs:
    def __init__(self):
        self.output_size = 22
        self.num_epochs = 100
        self.batch_size = 64
        self.learning_rate = 0.001
        self.hidden_size = 128
        self.layers = 2


def open_file(filepath):
    try:
        with open(filepath, 'r') as doc:
            peptides = doc.readlines()

        peptides = [pep.strip() for pep in peptides if pep.strip()]

        long_pep = max(len(pep) for pep in peptides)

        return peptides, long_pep

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, None


def padding(peptides, long_pep):
    for i, pep in enumerate(peptides):
        pad_length = long_pep - len(pep)

        if pad_length > 0:
            peptides[i] = pep.strip() + '_' * pad_length

    return peptides


def split_data(peptides):
    len_pep = len(peptides)
    train_tresh = math.floor(len_pep * 0.85)

    train_data = peptides[:train_tresh]
    test_data = peptides[train_tresh:]

    print(f"Total peptides: {len_pep}")
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    return train_data, test_data


def to_tensor(peptides, vocab):
    to_index = {a: i for i, a in enumerate(vocab)}
    result = []

    for pep in peptides:

        indices = [to_index.get(aa, to_index['_']) for aa in pep.strip()]

        indices = [min(i, len(vocab) - 1) for i in indices]
        result.append(indices)

    return torch.tensor(result), to_index


def to_amino(tensor, vocab):
    sequences = []
    for seq in tensor:
        amino_seq = ''.join([vocab[idx] for idx in seq])
        sequences.append(amino_seq)
    return sequences


def create_batches(tensor, batch_size):
    data_size = len(tensor)

    indices = list(range(data_size))

    random.shuffle(indices)

    batch = []

    for start in range(0, data_size, batch_size):
        end = min(start + batch_size, data_size)
        batch_indices = indices[start: end]
        batch.append(tensor[batch_indices, :])

    return batch


class LSTMPeptides(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, layers=2):
        super(LSTMPeptides, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_size = 128
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C',
                      'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim, padding_idx=21)

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





def train(peptides, model, seq_len,  args):
    model.train()

    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W',
             'Y', 'V', '_']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    inp_peptides, _ = to_tensor(peptides, vocab)
    print(f"Max index: {inp_peptides.max().item()}, Vocab size: {len(vocab)}")

    batch_size = args.batch_size

    print('\n-------Starting Training------- \n_________________________________')

    for epoch in range(args.num_epochs):
        state_h, state_c = model.init_state(seq_len)

        for batch in create_batches(inp_peptides, batch_size):

            print(batch.size())

            optimizer.zero_grad()

            x = batch[:, :-1].to(device)
            print(x, x.size())
            y = batch[:, 1:].to(device)
            print(y, y.size())

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'loss': loss.item()})

        print('End of Training \n -----------------------------')

def test(test_data, model, sequence_length, batch_size):

    model.eval()
    test_loss, correct = 0, 0
    num_batches = len(test_data) // batch_size
    criterion = nn.CrossEntropyLoss()

    print('Starting Testing \n -----------------------------')

    with torch.no_grad():
        state_h, state_c = model.init_state(sequence_length)

        for i in range(0, len(test_data), batch_size):
            X = test_data[i:i + batch_size]
            X = X.to(device)

            y = X[:, 1:].to(device)  # Target is the next sequence
            X = X[:, :-1].to(device)  # Input is the current sequence

            y_pred, (state_h, state_c) = model(X, (state_h, state_c))

            test_loss += criterion(y_pred.transpose(1, 2), y).item()

            correct += (y_pred.argmax(2) == y).type(torch.float).sum().item()

            state_h = state_h.detach()
            state_c = state_c.detach()


    test_loss /= num_batches
    accuracy = correct / (len(test_data) * (sequence_length - 1))  # Adjust based on sequence length

    print(f'Test Error: \n Accuracy: {100 * accuracy:.2f}%, Avg loss: {test_loss:.4f} \n')

    print('End of testing \n -----------------------------')
    return test_loss, accuracy


vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
         'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
len_vocab = len(vocab)

args = LSTMArgs()

dataset = '/Users/igorgonteri/Documents/GitHub/Nornour/0003c-APD-Database/antimicrobial_peptides_database.txt'
peptides, long_pep = open_file(dataset)
pep_padded = padding(peptides, long_pep)
train_data, test_data = split_data(pep_padded)

model = LSTMPeptides(long_pep, args.hidden_size, args.output_size, args.batch_size, args.layers)

train(train_data, model, long_pep, args)
test(test_data, model, long_pep, args.batch_size)



