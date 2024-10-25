import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class LSTMArgs:
    def __init__(self):
        self.output_size = 22
        self.num_epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.01
        self.hidden_size = 256
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
        batch_indices = indices[start:end]
        if len(batch_indices) == batch_size:
            batch.append(tensor[batch_indices, :])

    return batch

def plot_loss(loss_values):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss', color='blue')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def temperature_sampling(logits, temperature):
    # Apply temperature and mask padding index (21 in this case)
    padding_index = 21
    logits[:, padding_index] = float('-inf')  # Mask padding index

    # Scale logits by temperature
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)

    # Sample from the adjusted probability distribution
    next_index = torch.multinomial(probabilities, num_samples=1).item()
    return next_index

class LSTMPeptides(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, layers=2):
        super(LSTMPeptides, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_size = 128
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
                      'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim, padding_idx=21)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.layers, batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_state):

        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        output = self.dropout_layer(output)
        logits = self.fc(output)
        return logits, state

    def init_state(self, batch_size):

        return (torch.zeros(self.layers, batch_size, self.hidden_size),
                torch.zeros(self.layers, batch_size, self.hidden_size))


def train(peptides, model, seq_len, args):
    model.train()
    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    inp_peptides, _ = to_tensor(peptides, vocab)
    print(f"Max index: {inp_peptides.max().item()}, Vocab size: {len(vocab)}")

    batch_size = args.batch_size
    print('\n-------Starting Training------- \n_________________________________')

    epoch_losses = []

    for epoch in range(args.num_epochs):
        epoch_loss = 0

        for batch in create_batches(inp_peptides, batch_size):
            current_batch_size = batch.size(0)
            state_h, state_c = model.init_state(current_batch_size)

            optimizer.zero_grad()

            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / (len(peptides)//batch_size)
        epoch_losses.append(avg_epoch_loss)
        print({'epoch': epoch + 1, 'loss': avg_epoch_loss})

    print('-------End of Training--------\n-----------------------------\n')

    return epoch_losses


def test(test_data, model, sequence_length, batch_size):
    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

    model.eval()
    test_loss, correct = 0, 0
    criterion = nn.CrossEntropyLoss()

    # Convert test_data to tensor format
    inp_test_data, _ = to_tensor(test_data, vocab)
    inp_test_data = inp_test_data.to(device)  # Move to the appropriate device

    print('------Starting Testing------\n-----------------------------')

    with torch.no_grad():
        state_h, state_c = model.init_state(batch_size)

        # Iterate over test data in batches
        for batch in create_batches(inp_test_data, batch_size):
            current_batch_size = batch.size(0)
            state_h, state_c = model.init_state(current_batch_size)

            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            # Calculate the loss
            test_loss += criterion(y_pred.transpose(1, 2), y).item()

            # Calculate accuracy
            correct += (y_pred.argmax(2) == y).type(torch.float).sum().item()

            # Detach the hidden states to prevent backpropagation
            state_h = state_h.detach()
            state_c = state_c.detach()

    # Average the loss and compute accuracy
    num_batches = len(inp_test_data) // batch_size
    test_loss /= num_batches
    accuracy = correct / (len(test_data) * (sequence_length - 1))  # Adjust based on sequence length

    print(f'Test Error: \n Accuracy: {100 * accuracy:.2f}%, Avg loss: {test_loss:.4f} \n')
    print('------End of testing--------\n-----------------------------')

    return test_loss, accuracy


def gen_peptides(model, seed, number_aa, vocab, device, temperature=1.0):
    model.eval()

    to_index = {a: i for i, a in enumerate(vocab)}
    index_to_amino = {i: a for i, a in enumerate(vocab)}

    seed_indices = [to_index[aa] for aa in seed if aa in to_index and to_index[aa] != 21]
    input_tensor = torch.LongTensor(seed_indices).unsqueeze(0).to(device)

    state_h, state_c = model.init_state(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    gen_seq = seed

    for _ in range(number_aa):
        with torch.no_grad():
            y_pred, (state_h, state_c) = model(input_tensor, (state_h, state_c))

        next_index = temperature_sampling(y_pred[:, -1, :], temperature)

        if next_index == 21:
            break

        next_amino = index_to_amino[next_index]
        gen_seq += next_amino

        input_tensor = torch.cat((input_tensor, torch.tensor([[next_index]]).to(device)), dim=1)

    return gen_seq



vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
         'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
len_vocab = len(vocab)

args = LSTMArgs()

dataset = '/Users/igorgonteri/Documents/GitHub/Nornour/0003c-APD-Database/antimicrobial_peptides_database.txt'
peptides, long_pep = open_file(dataset)
pep_padded = padding(peptides, long_pep)
train_data, test_data = split_data(pep_padded)

model = LSTMPeptides(long_pep, args.hidden_size, args.output_size, args.batch_size, args.layers)

loss_values = train(train_data, model, long_pep, args)

test(test_data, model, long_pep, args.batch_size)

plot_loss(loss_values)

gen_sequences = set()
temperature = 0.9
while len(gen_sequences) < 1000:
    number_aa = random.randint(2, 20)
    seed = random.choice(peptides[:5])
    random_index = random.randint(0, 20)
    gen_pep = gen_peptides(model, seed,  number_aa, vocab, device, temperature)
    print(gen_pep)
    gen_sequences.add(gen_pep)
    print('Len sequences: ', len(gen_sequences))


gen_sequences = list(gen_sequences)
for pep in gen_sequences:
    print(pep)