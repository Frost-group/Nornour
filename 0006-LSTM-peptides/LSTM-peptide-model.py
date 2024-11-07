#use RW lexicon are actual text to see if it works
#wandb.com to add in the code
#use LLM for peptide pred
#find the georgian work on internet
#add argument parser for the arguments of the ML program

import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
import wandb
import argparse
from tqdm import tqdm

wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class LSTMArgs:
    def __init__(self):
        self.output_size = 22
        self.num_epochs = 50
        self.batch_size = 16
        self.learning_rate = 0.001
        self.hidden_size = 128
        self.layers = 3
        self.dropout = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test LSTM model for peptide sequences')

    # Add arguments for each parameter in LSTMArgs
    parser.add_argument('--output_size', type=int, default=22, help='Size of the output layer (default: 22)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size (default: 128)')
    parser.add_argument('--layers', type=int, default=3, help='Number of LSTM layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (default: 0.5)')

    # Parse the arguments
    args = parser.parse_args()

    # Create an LSTMArgs object and populate it with the parsed arguments
    lstm_args = LSTMArgs()
    lstm_args.output_size = args.output_size
    lstm_args.num_epochs = args.num_epochs
    lstm_args.batch_size = args.batch_size
    lstm_args.learning_rate = args.learning_rate
    lstm_args.hidden_size = args.hidden_size
    lstm_args.layers = args.layers
    lstm_args.dropout = args.dropout

    return lstm_args


args = parse_args()

config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'layers': args.layers,
        'output_size': args.output_size,
        'hidden_size': args.hidden_size,
        'dropout': args.dropout
    }

run = wandb.init(
        project='LSTM-peptides',
        notes=input('notes:  '),
        config=config
    )

print(f"Using configuration: {config}")

sweep_configuration = {
    'method': 'random',
    'name': 'LSTM-peptides sweep',
    'metric': {'goal': 'maximize', 'name': 'avg_epoch_accuracy'},
    'parameters': {
        'epochs': {'max': 200, 'min': 10},
        'batch_size': {'values': [8, 16, 32]},
        'learning_rate': {'max': 0.02, 'min': 0.001},
        'layers': {'values': [1, 2, 3, 4]},
        'hidden_size': {'values': [32, 64, 128, 256]},
        'dropout': {'values': [0.2, 0.5, 0.7]}}
}


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


class LSTMPeptides(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, layers=2, dropout=0.5):
        super(LSTMPeptides, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_size = 128
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers
        self.dropout = dropout

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
                      'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim, padding_idx=21)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.layers, batch_first=True)
        self.dropout_layer = nn.Dropout(p=self.dropout)
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

    def training_step(self, x, y, criterion, is_training=True):
        # Initialize state for the current batch
        batch_size = x.size(0)
        state_h, state_c = self.init_state(batch_size)

        # Enable gradient tracking only if training
        with torch.set_grad_enabled(is_training):
            # Forward pass
            y_pred, (state_h, state_c) = self(x, (state_h, state_c))

            # Calculate loss
            loss = criterion(y_pred.transpose(1, 2), y)

            # Calculate accuracy
            with torch.no_grad():
                predicted = y_pred.argmax(dim=2)
                correct = (predicted == y).float().sum()
                accuracy = correct / y.numel()

        return loss, accuracy


def train(peptides, model):
    model.train()
    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-5)

    inp_peptides, _ = to_tensor(peptides, vocab)
    print(f"Max index: {inp_peptides.max().item()}, Vocab size: {len(vocab)}")

    batch_size = wandb.config.batch_size
    print('\n-------Starting Training------- \n_________________________________')

    for epoch in range(wandb.config.num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(create_batches(inp_peptides, batch_size))

        # Adding tqdm progress bar for batches
        with tqdm(total=num_batches, desc=f"Epoch [{epoch + 1}/{wandb.config.num_epochs}]", leave=True) as pbar:
            for batch in create_batches(inp_peptides, batch_size):
                current_batch_size = batch.size(0)
                state_h, state_c = model.init_state(current_batch_size)

                optimizer.zero_grad()

                x = batch[:, :-1].to(device)
                y = batch[:, 1:].to(device)

                loss, accuracy = model.training_step(x, y, criterion)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_accuracy += accuracy

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy)

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches

        try:
            wandb.log({"accuracy": avg_epoch_accuracy, "loss": avg_epoch_loss})
        except BrokenPipeError:
            print("WandB connection lost, skipping logging for this iteration.")

        print(f"Epoch [{epoch + 1}/{wandb.config.num_epochs}], Loss: {avg_epoch_loss}, Accuracy: {avg_epoch_accuracy}")

    print('-------End of Training--------\n-----------------------------\n')



def test(test_data, model, batch_size):
    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

    model.eval()
    test_loss, correct = 0, 0
    criterion = nn.CrossEntropyLoss()

    # Convert test_data to tensor format
    inp_test_data, _ = to_tensor(test_data, vocab)
    inp_test_data = inp_test_data.to(device)  # Move to the appropriate device

    print('------Starting Testing------\n-----------------------------')

    total_samples = 0

    with torch.no_grad():
        # Iterate over test data in batches
        for batch in create_batches(inp_test_data, batch_size):
            current_batch_size = batch.size(0)

            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            # Use the modified training_step in test mode
            loss, accuracy = model.training_step(x, y, criterion, is_training=False)

            test_loss += loss.item()
            correct += accuracy * y.numel()  # Scale by the number of elements in the batch
            total_samples += y.numel()  # Total number of target elements for accuracy

    # Average the loss and compute accuracy
    num_batches = len(create_batches(inp_test_data, batch_size))  # Total number of batches
    test_loss /= num_batches
    test_accuracy = correct / total_samples  # Accuracy based on total target elements

    print(f'Test Error: \n Accuracy: {100 * accuracy:.2f}%, Avg loss: {test_loss:.4f} \n')
    print('------End of testing--------\n-----------------------------')

    return test_loss, test_accuracy


if __name__ == "__main__":
    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
    len_vocab = len(vocab)

    dataset = '/Users/igorgonteri/Documents/GitHub/Nornour/0003b-RW-Lexicon/RW_lexicon.dat'
    peptides, long_pep = open_file(dataset)

    pep_padded = padding(peptides, long_pep)

    train_data, test_data = split_data(pep_padded)

    model = LSTMPeptides(long_pep, wandb.config.hidden_size, wandb.config.output_size, wandb.config.batch_size,
                         wandb.config.layers)

    train(train_data, model)

    test(test_data, model, wandb.config.batch_size)