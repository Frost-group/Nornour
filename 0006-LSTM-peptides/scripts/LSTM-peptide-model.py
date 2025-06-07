import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
import wandb
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import subprocess

wandb.login()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test LSTM model for peptide sequences')

    parser.add_argument('--dataset_path', type=str, default='../../0003d-DBAASP-Database/DBAASP', help='Path to the dataset to use for training')
    parser.add_argument('--output_size', type=int, default=21, help='Size of the output layer (default: 22)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=0.00063, help='Learning rate (default: 0.00063)')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size (default: 256)')
    parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (default: 0.7)')
    parser.add_argument('--max_length', type=int, default=15, help='Maximum peptide length (default: 15)')
    parser.add_argument('--vocab', type=str, default= "RHKDESTNQCGPAILMFWYV_", help='List of amino acids and padding index')
    parser.add_argument('--save_model', type=bool, default=False, help='Save the trained model (default: False)')

    args = parser.parse_args()
    return args


args = parse_args()

config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'layers': args.layers,
        'output_size': args.output_size,
        'hidden_size': args.hidden_size,
        'dropout': args.dropout
    }

run = wandb.init(
        project='LSTM-peptides_DBAASP',
        config=config
    )

print(f"Using configuration: {config}")


def open_file(filepath):
    """
     Reads a file containing peptide sequences, processes the sequences,
     and determines the length of the longest peptide.

     Args:
         - filepath (str): Path to the file containing peptide sequences.

     Returns:
         - peptides (list[str]): List of cleaned peptide sequences.
         - long_pep (int): Length of the longest peptide in the file.
     """
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
    """
     Pads peptide sequences with underscores ('_') to match the length of the longest peptide.

     Args:
         - peptides (list[str]): List of peptide sequences.
         - long_pep (int): Length of the longest peptide.

     Returns:
         - peptides (list[str]): List of padded peptide sequences.
     """
    for i, pep in enumerate(peptides):
        pad_length = long_pep - len(pep)

        if pad_length > 0:
            peptides[i] = pep.strip() + '_' * pad_length

        if pad_length < 0:
            peptides[i] = peptides[i][:long_pep]

    return peptides


def split_data(peptides):
    """
    Splits the peptide dataset into training and testing sets (85% training, 15% testing).

    Args:
        - peptides (list[str]): List of peptide sequences.

    Returns:
        - train_data (list[str]): Training set of peptide sequences.
        - test_data (list[str]): Testing set of peptide sequences.
    """
    len_pep = len(peptides)
    train_tresh = math.floor(len_pep * 0.85)

    train_data = peptides[:train_tresh]
    test_data = peptides[train_tresh:]

    print(f"Total peptides: {len_pep}")
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    return train_data, test_data


def to_tensor(peptides, vocab):
    """
        Converts peptide sequences into tensor format using a given vocabulary.

        Args:
            - peptides (list[str]): List of peptide sequences.
            - vocab (list[str]): Vocabulary of amino acids and padding.

        Returns:
            - tensor (torch.Tensor): Tensor of encoded peptide sequences.
            - to_index (dict): Dictionary mapping amino acids to indices.
    """
    to_index = {a: i for i, a in enumerate(vocab)}
    result = []

    for pep in peptides:

        indices = [to_index.get(aa, to_index['_']) for aa in pep.strip()]

        indices = [min(i, len(vocab) - 1) for i in indices]
        result.append(indices)

    return torch.tensor(result), to_index


def to_amino(tensor, vocab):
    """
        Converts a tensor of indices back into amino acid sequences.

        Args:
            - tensor (torch.Tensor): Tensor of encoded peptide sequences.
            - vocab (list[str]): Vocabulary of amino acids and padding.

        Returns:
            - sequences (list[str]): List of decoded peptide sequences.
    """
    sequences = []
    for seq in tensor:
        amino_seq = ''.join([vocab[idx] for idx in seq])
        sequences.append(amino_seq)
    return sequences


def create_batches(tensor, batch_size):
    """
       Creates batches of data from a tensor, ensuring each batch has the specified size.

       Args:
           - tensor (torch.Tensor): Tensor of data to be batched.
           - batch_size (int): Number of samples per batch.

       Returns:
           - batch (list[torch.Tensor]): List of data batches.
    """
    data_size = len(tensor)
    indices = list(range(data_size))
    random.shuffle(indices)

    batch = []

    #create all the batches
    for start in range(0, data_size, batch_size):
        end = min(start + batch_size, data_size)
        batch_indices = indices[start:end]
        if len(batch_indices) == batch_size:
            batch.append(tensor[batch_indices, :])

    return batch


class LSTMPeptides(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model designed for peptide sequence modeling.

    Args:
        - input_size (int): Size of the input feature vector for each timestep.

    Attributes:
        - input_size (int): Size of the input feature vector for each timestep.
        - hidden_size (int): Number of hidden units in the LSTM layers.
        - lstm_size (int): Internal dimension size of the LSTM (default set to 128).
        - output_size (int): Size of the output layer.
        - batch_size (int): Batch size used during training or inference.
        - layers (int): Number of LSTM layers.
        - dropout (float): Dropout rate applied after the LSTM layer.
        - vocab (list[str]): Amino acid vocabulary including padding ('_').
        - len_vocab (int): Length of the vocabulary.
        - embedding_dim (int): Dimension of the embedding layer.
        - embedding (nn.Embedding): Embedding layer for amino acid sequences.
        - lstm (nn.LSTM): LSTM layer for sequential processing.
        - dropout_layer (nn.Dropout): Dropout layer to regularize the model.
        - fc (nn.Linear): Fully connected layer to produce output logits.

    Methods:
        - forward(x, prev_state):
            Computes the forward pass through the embedding, LSTM, and output layers.

            Args:
                - x (Tensor): Input tensor of peptide sequences, encoded as indices.
                - prev_state (tuple): Initial hidden and cell states for the LSTM.

            Returns:
                - logits (Tensor): Logits output from the fully connected layer.
                - state (tuple): Updated hidden and cell states.

        - init_state(batch_size):
            Initializes the hidden and cell states for the LSTM.

            Args:
                - batch_size (int): Batch size for the current input.

            Returns:
                - tuple: Initialized hidden and cell states (zeros).

        - training_step(x, y, criterion, is_training=True):
            Performs a single training or evaluation step, including forward pass, loss calculation,
            and accuracy computation.

            Args:
                - x (Tensor): Input tensor of encoded peptide sequences.
                - y (Tensor): Target tensor for the ground truth labels.
                - criterion (callable): Loss function (e.g., CrossEntropyLoss).
                - is_training (bool): Whether the model is in training mode.

            Returns:
                - loss (Tensor): Calculated loss for the batch.
                - accuracy (float): Accuracy metric for the predictions.
    """
    def __init__(self, input_size):
        super(LSTMPeptides, self).__init__()

        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.lstm_size = 128
        self.output_size = args.output_size
        self.batch_size = args.batch_size
        self.layers = args.layers
        self.dropout = args.dropout

        self.vocab = args.vocab

        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim, padding_idx=20)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.layers, batch_first=True)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

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

        batch_size = x.size(0)
        state_h, state_c = self.init_state(batch_size)

        with torch.set_grad_enabled(is_training):

            y_pred, (state_h, state_c) = self(x, (state_h, state_c))

            loss = criterion(y_pred.transpose(1, 2), y)

            #Accuracy calculation
            with torch.no_grad():
                predicted = y_pred.argmax(dim=2)
                correct = (predicted == y).float().sum()
                accuracy = correct / y.numel()

        return loss, accuracy


def train(peptides, model):
    """    Trains the LSTM-based model on a set of peptide sequences.

    Args:
        - peptides (list[str]): List of peptide sequences to train on.
        - model (nn.Module): LSTM model for peptide sequence learning.

    Workflow:
        1. Prepares data by encoding peptides into tensors using the vocabulary.
        2. Defines the loss function (CrossEntropyLoss) and optimizer (Adam).
        3. Divides data into batches of a specified size.
        4. Iterates over epochs and batches, updating model weights using backpropagation.
        5. Tracks and logs loss and accuracy for each epoch.
    """
    model.train()
    vocab = args.vocab
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-5)

    inp_peptides, _ = to_tensor(peptides, vocab)
    print(f"Max index: {inp_peptides.max().item()}, Vocab size: {len(vocab)}")

    batch_size = wandb.config.batch_size
    print('\n-------Starting Training------- \n_________________________________')

    for epoch in range(wandb.config.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(create_batches(inp_peptides, batch_size))

        with tqdm(total=num_batches, desc=f"Epoch [{epoch + 1}/{wandb.config.epochs}]", leave=True) as pbar:
            for batch in create_batches(inp_peptides, batch_size):
                current_batch_size = batch.size(0)
                state_h, state_c = model.init_state(current_batch_size)

                optimizer.zero_grad()

                # Forward pass
                x = batch[:, :-1].to(device)
                y = batch[:, 1:].to(device)

                loss, accuracy = model.training_step(x, y, criterion)

                state_h = state_h.detach()
                state_c = state_c.detach()

                #Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()

                wandb.log({
                    "epoch": epoch + 1,
                    "loss": loss,
                    "accuracy": accuracy
                })

                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy)

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches

        print(f"Epoch [{epoch + 1}/{wandb.config.epochs}], Loss: {avg_epoch_loss}, Accuracy: {avg_epoch_accuracy}")

    print('-------End of Training--------\n-----------------------------\n')

    return round(avg_epoch_loss, 6), round(avg_epoch_accuracy, 4)

def test(test_data, model, batch_size):
    """
    Evaluates the performance of the trained model on a test dataset.

    Args:
        - test_data (list[str]): List of peptide sequences for testing.
        - model (nn.Module): The trained LSTM model for evaluation.
        - batch_size (int): The number of samples per batch for testing.

    Workflow:
        1. Prepares the test data by encoding peptide sequences into tensors.
        2. Defines the loss function (CrossEntropyLoss) for evaluating the model.
        3. Iterates over the test data in batches and computes the model's loss and accuracy.
        4. Computes the average loss and accuracy across all test batches.
        5. Outputs the final test accuracy and average loss.
    """
    vocab = args.vocab

    model.eval()
    test_loss, correct = 0, 0
    criterion = nn.CrossEntropyLoss()

    inp_test_data, _ = to_tensor(test_data, vocab)
    inp_test_data = inp_test_data.to(device)

    print('------Starting Testing------\n-----------------------------')

    total_samples = 0

    with torch.no_grad():

        for batch in create_batches(inp_test_data, batch_size):
            current_batch_size = batch.size(0)

            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            #forward pass and calculate metrics
            loss, accuracy = model.training_step(x, y, criterion, is_training=False)

            test_loss += loss.item()
            correct += accuracy.item() * y.numel()
            total_samples += y.numel()

    num_batches = len(create_batches(inp_test_data, batch_size))
    test_loss /= num_batches
    test_accuracy = correct / total_samples

    print(f'Test logs: \n Accuracy: {100 * test_accuracy:.2f}%, Avg loss: {test_loss:.4f} \n')
    print('------End of testing--------\n-----------------------------')

    return round(test_loss, 6), round(test_accuracy, 4)


if __name__ == "__main__":
    vocab = args.vocab
    len_vocab = len(vocab)

    dataset = args.dataset_path
    peptides, long_pep = open_file(dataset)

    print(f'longest peptide: {long_pep} AA')

    if not args.max_length:
        pep_padded = padding(peptides, long_pep)
        print(f'Size of peptides for model: {long_pep}')
    else:
        pep_padded = padding(peptides, args.max_length)
        print(f'Size of peptides for model: {args.max_length}')

    train_data, test_data = split_data(pep_padded)

    model = LSTMPeptides(long_pep)

    avg_epoch_loss, avg_epoch_accuracy = train(train_data, model)

    test_loss, test_accuracy = test(test_data, model, wandb.config.batch_size)

    if args.save_model:
        model_path = "../models/lstm_peptides_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    today = datetime.today().strftime("%d-%m-%Y")
    base_dir = "../data"
    gen_base_dir = os.path.join(base_dir, f"gen_{today}")
    os.makedirs(gen_base_dir, exist_ok=True)
    count = 1
    while os.path.exists(os.path.join(gen_base_dir, f"generated_peptides_{count}")):
        count += 1
    gen_dir = os.path.join(gen_base_dir, f"generated_peptides_{count}")
    os.makedirs(gen_dir)

    print(f"Directory created: {gen_dir}")

    sum_path = os.path.join(gen_dir, "generation_summary.txt")
    with open(sum_path, 'w') as file:
        file.write(f'Generation number {count}, created on {today}, workflow summary.\n\n')
        file.write('LSTM training summary: \n')

        file.write('Dataset information: \n')
        file.write('\t- Longest peptide: ' + str(long_pep) + ' AA\n')
        file.write(f'\t- Total peptides: {len(peptides)}\n')
        file.write(f'\t- Training set size: {len(train_data)}\n')
        file.write(f'\t- Test set size: {len(test_data)}\n')

        file.write('Hyperparameters: \n')
        for arg, value in vars(args).items():
            file.write(f'\t- {arg}: {value}\n')

        file.write('Accuracies and losses: \n')
        file.write(f'\t- training loss after last epoch: {avg_epoch_loss}\n')
        file.write(f'\t- training accuracy after last epoch: {avg_epoch_accuracy}\n')
        file.write(f'\t- test loss: {test_loss}\n')
        file.write(f'\t- test_accuracy: {test_accuracy}\n')

    input_gen = input('Do you want to run peptide generation ? [y/n]: ')
    if input_gen.lower() == 'y':
        print(f'Generation command: \n')
        print(f"""python LSTM-peptides-generation.py --dataset_path {args.dataset_path} --model_path ../lstm_peptides_model.pt --output_path d --output_size {args.output_size} --epochs {args.epochs} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --hidden_size {args.hidden_size} --layers {args.layers} --dropout {args.dropout} --temperature 1.0 --num_sequences 40000 --min_length 1 --max_length 15 --seed r \n""")
        config_input = input('Keep configuration ? [y/n]: ')
        if config_input.lower() == 'n':
            temperature_input = float(input('Temperature (float): '))
            num_sequences_input = int(input('Number of sequences (int): '))
            max_length_input = int(input('Max length (int): '))
            min_length_input = int(input('Min length (int): '))
            seed_input = input('Seed (str), r for random: ')
        else:
            temperature_input = 1.0
            num_sequences_input = 40000
            max_length_input = 15
            min_length_input = 2
            seed_input = 'r'

        subprocess.run([
            "python",
            "LSTM-peptides-generation.py",
            "--dataset_path", args.dataset_path,
            "--model_path", "../models/lstm_peptides_model.pt",
            "--output_path", "d",
            "--output_size", str(args.output_size),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--hidden_size", str(args.hidden_size),
            "--layers", str(args.layers),
            "--dropout", str(args.dropout),
            "--temperature", str(temperature_input),
            "--num_sequences", str(num_sequences_input),
            "--min_length", str(min_length_input),
            "--max_length", str(max_length_input),
            "--seed", str(seed_input)])
    else:
        pass

"""
Optimal hyperparam RW_lexicon:

--output_size 21
--epochs 20
--batch_size 8
--learning_rate 0.00063
--hidden_size 256
--layers 2
--dropout 0.7

"""