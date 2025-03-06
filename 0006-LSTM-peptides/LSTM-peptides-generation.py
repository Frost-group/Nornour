import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LSTMArgsGen:
    def __init__(self):
        self.dataset_path = '../0003b-RW-Lexicon/RW_lexicon.dat'
        self.output_path = '../0006b-LSTM-data/generation_peptides.fasta'
        self.output_size = 22
        self.epochs = 50
        self.batch_size = 8
        self.learning_rate = 0.00063
        self.hidden_size = 256
        self.layers = 2
        self.dropout = 0.7
        self.save_model = False
        self.temperature = 1.0
        self.num_sequences = 100
        self.min_length = 2
        self.max_length = 15
        self.seed = 'r'

def parse_args():
    parser = argparse.ArgumentParser(description='Train and generate peptides using LSTM model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset used for training')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file for generated sequences (.fasta format)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_size', type=int, default=22, help='Size of the output layer (default: 22)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00063, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for generation')
    parser.add_argument('--num_sequences', type=int, default=100, help='Number of unique sequences to generate')
    parser.add_argument('--min_length', type=int, default=2, help='Minimum peptide length')
    parser.add_argument('--max_length', type=int, default=15, help='Maximum peptide length')
    parser.add_argument('--seed', type=str, default='r', help="Seed sequence for generation, put 'r' for automated seed generation")
    args = parser.parse_args()

    return args

args = parse_args()

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

        self.vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
                      'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

        self.len_vocab = len(self.vocab)

        self.embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=self.len_vocab, embedding_dim=self.embedding_dim, padding_idx=21)

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

            #calculate metrics
            with torch.no_grad():
                predicted = y_pred.argmax(dim=2)
                correct = (predicted == y).float().sum()
                accuracy = correct / y.numel()

        return loss, accuracy


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


def weighted_seed():
    pos1_aa = {'R': 12.158931082981717, 'H': 1.1251758087201125, 'K': 15.618846694796062, 'D': 1.0970464135021099,
     'E': 1.047819971870605, 'S': 2.8129395218002813, 'T': 1.0970464135021099, 'N': 0.8720112517580872,
     'Q': 0.8157524613220816, 'C': 2.229254571026723, 'G': 16.59634317862166, 'P': 1.9338959212376934,
     'A': 5.780590717299578, 'I': 5.836849507735583, 'L': 7.165963431786217, 'M': 1.047819971870605,
     'F': 11.969057665260197, 'W': 4.725738396624473, 'Y': 1.3713080168776373, 'V': 4.6976090014064695}
    pos2_aa = {'R': 11.357243319268635, 'H': 0.9423347398030942, 'K': 13.361462728551334, 'D': 2.0604781997187063,
     'E': 1.188466947960619, 'S': 2.1448663853727146, 'T': 1.4064697609001406, 'N': 2.517580872011252,
     'Q': 1.5119549929676512, 'C': 1.6033755274261603, 'G': 4.585091420534459, 'P': 2.1870604781997187,
     'A': 4.634317862165963, 'I': 8.319268635724333, 'L': 20.028129395218002, 'M': 0.9282700421940928,
     'F': 6.279887482419127, 'W': 9.345991561181433, 'Y': 1.2165963431786215, 'V': 4.09985935302391}
    pos3_aa = {'R': 11.308016877637131, 'H': 2.3558368495077358, 'K': 15.942334739803094, 'D': 0.9774964838255977,
     'E': 1.1111111111111112, 'S': 2.7777777777777777, 'T': 1.3361462728551337, 'N': 1.6947960618846694,
     'Q': 1.9549929676511955, 'C': 2.791842475386779, 'G': 6.8354430379746836, 'P': 7.229254571026724,
     'A': 4.486638537271449, 'I': 5.576652601969058, 'L': 12.637130801687762, 'M': 1.0337552742616034,
     'F': 5.253164556962026, 'W': 8.263009845288325, 'Y': 1.2447257383966244, 'V': 3.931082981715893}
    aa_1 = list(pos1_aa.values())
    aa_2 = list(pos2_aa.values())
    aa_3 = list(pos3_aa.values())
    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

    pos1 = random.choices(vocab, weights=aa_1, k=1)[0]
    pos2 = random.choices(vocab, weights=aa_2, k=1)[0]
    pos3 = random.choices(vocab, weights=aa_3, k=1)[0]

    return pos1 + pos2 + pos3




def temperature_sampling(logits, temperature):
    """
    Performs temperature-based sampling to select the next token from a set of logits.

    Args:
        - logits (Tensor): A tensor containing unnormalized log probabilities (logits) for each token.
        - temperature (float): A scaling factor that controls the randomness of the sampling process.
          - Higher temperature (>1) increases randomness by flattening the probability distribution.
          - Lower temperature (<1) makes the distribution sharper, favoring more likely tokens.

    Returns:
        - next_index (int): The index of the token selected by the sampling process."""

    padding_index = 21
    logits[:, padding_index] = float('-inf')
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    next_index = torch.multinomial(probabilities, num_samples=1).item()
    return next_index


def load_model(model_path, long_pep):
    """


    """
    model = LSTMPeptides(long_pep)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def gen_peptides(model, seed, number_aa, vocab, device, temperature=1.0):
    """
    Generates a peptide sequence using an LSTM-based model and temperature-based sampling.

    Args:
        - model (nn.Module): The trained LSTM model for peptide generation.
        - seed (str): The initial peptide sequence used as a starting point for generation.
        - number_aa (int): The total length of the desired peptide sequence, including the seed.
        - vocab (list): List of amino acids and special tokens (e.g., '_') representing the model's vocabulary.
        - device (torch.device): The device to run the generation on (e.g., 'cpu' or 'cuda').
        - temperature (float, optional): Controls randomness in sampling.
          - Higher values increase randomness, lower values make outputs more deterministic. Default is 1.0.

    Workflow:
        1. Convert the seed sequence into a list of corresponding indices using the vocabulary.
        2. Prepare the input tensor from the seed indices and move it to the specified device.
        3. Initialize the LSTM's hidden and cell states for a single batch.
        4. Iteratively generate amino acids until the desired sequence length (`number_aa`) is reached:
           - Use the model to predict the next token's logits.
           - Apply temperature sampling to select the next token index.
           - If the padding token (`_`) is selected, terminate the sequence early.
           - Append the selected token to the generated sequence and update the input tensor.
        5. Stop the loop if the generated sequence reaches the specified length.
        6. Return the final generated peptide sequence as a string.

    Returns:
        - gen_seq (str): The generated peptide sequence.

    Notes:
        - The function ensures that the padding token ('_') is excluded from the sequence and terminates generation if it is sampled.
        - This function is particularly useful for generating synthetic peptides with controlled randomness via the `temperature` parameter.
    """


    if seed == 'r':
        gen_seq = weighted_seed()
    else:
        gen_seq = seed

    to_index = {a: i for i, a in enumerate(vocab)}
    index_to_amino = {i: a for i, a in enumerate(vocab)}
    seed_indices = [to_index[aa] for aa in gen_seq if aa in to_index and to_index[aa] != 21]

    input_tensor = torch.LongTensor(seed_indices).unsqueeze(0).to(device)

    state_h, state_c = model.init_state(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    for _ in range(number_aa - len(seed)):
        with torch.no_grad():
            y_pred, (state_h, state_c) = model(input_tensor, (state_h, state_c))

        next_index = temperature_sampling(y_pred[:, -1, :], temperature)

        if next_index == 21:  # End sequence if padding token is generated
            break

        next_amino = index_to_amino[next_index]
        gen_seq += next_amino
        input_tensor = torch.cat((input_tensor, torch.tensor([[next_index]]).to(device)), dim=1)

        # Stop if generated sequence reaches the required length
        if len(gen_seq) >= number_aa:
            break

    return ''.join(gen_seq)


def seq_to_fasta(peptide: list, file_path: str):
    """
    Converts a list of peptide sequences into FASTA format and writes them to a specified file.

    Args:
        peptide (list): List of peptide sequences to be converted to FASTA format.
        file_path (str): The path of the file where the FASTA data will be stored.

    Returns:
        None: The function writes the FASTA data to the specified file and prints a confirmation message.
    """
    with open(file_path, 'w') as file:
        for i, seq in enumerate(peptide):
            # Format each peptide into FASTA format
            fasta_entry = f'>peptide_{i+1}\n{seq}\n'
            file.write(fasta_entry)  # Write each FASTA entry to the file
    print(f'Peptides stored in FASTA format at: {file_path}')


def main():
    """
    Main function to generate new peptide sequences and save them in FASTA format.

    This function loads a dataset of existing peptides, generates new peptide sequences using
    a trained model, and compares the generated sequences with the dataset to ensure uniqueness.
    The function uses a seed sequence and generates peptides of random lengths within a specified
    range. It removes any generated peptides that already exist in the dataset before saving the
    unique ones to a FASTA file.

    Args:
        None: The function takes no arguments and uses global configurations from `args`.

    Returns:
        None: The function prints information about the generated peptides, including their lengths,
        uniqueness, and writes them to a FASTA file.
    """

    peptides, long_pep = open_file(args.dataset_path)

    model = load_model(args.model_path, long_pep)

    vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
             'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']

    iteration = 0
    gen_sequences = set()
    while len(gen_sequences) < args.num_sequences:
        number_aa = random.randint(args.min_length, args.max_length)
        gen_pep = gen_peptides(model, args.seed, number_aa, vocab, device, args.temperature)
        gen_sequences.add(gen_pep)
        iteration += 1
        if iteration > (2 * args.num_sequences):
            break

    peptides_to_remove = set()

    print(f"--------------------------\n Generated {len(gen_sequences)} unique peptide sequences:")
    for pep in gen_sequences:
        print(pep)
        print('\n')
        if 'U' in peptides:
            print('Peptide contains non canonical amino acids \n Peptide marked for removal from generated sequences \n--------------------------\n')
            peptides_to_remove.add(pep)
        elif pep in peptides:
            print('Peptide already in dataset \n Peptide marked for removal from generated sequences \n--------------------------\n')
            peptides_to_remove.add(pep)

    gen_sequences -= peptides_to_remove

    print(f'--------------------------\n Generated {len(gen_sequences)} new unique peptide sequences \n')

    short_len = len(min(gen_sequences, key=len))
    long_len = len(max(gen_sequences, key=len))

    print(f'Shortest peptide: {short_len} amino acids')
    print(f'Longest peptide: {long_len} amino acids\n')

    gen_seq_list = list(gen_sequences)
    seq_to_fasta(gen_seq_list, args.output_path)

if __name__ == "__main__":
    main()




    
