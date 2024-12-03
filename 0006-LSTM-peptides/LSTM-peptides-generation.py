import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import csv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LSTMArgs:
    def __init__(self):
        self.dataset_path = '../0003b-RW-Lexicon/RW_lexicon.dat'
        self.output_path = 'generation_output.fasta'
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
        self.seed = ['R', 'W', 'W']

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
    parser.add_argument('--seed', type=str, default='RWW', help="Seed sequence for generation")
    args = parser.parse_args()

    return args

args = parse_args()

class LSTMPeptides(nn.Module):
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

            with torch.no_grad():
                predicted = y_pred.argmax(dim=2)
                correct = (predicted == y).float().sum()
                accuracy = correct / y.numel()

        return loss, accuracy


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


def temperature_sampling(logits, temperature):
    padding_index = 21
    logits[:, padding_index] = float('-inf')
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    next_index = torch.multinomial(probabilities, num_samples=1).item()
    return next_index


def load_model(model_path, long_pep):
    model = LSTMPeptides(long_pep)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def gen_peptides(model, seed, number_aa, vocab, device, temperature=1.0):
    to_index = {a: i for i, a in enumerate(vocab)}
    index_to_amino = {i: a for i, a in enumerate(vocab)}
    seed_indices = [to_index[aa] for aa in seed if aa in to_index and to_index[aa] != 21]

    input_tensor = torch.LongTensor(seed_indices).unsqueeze(0).to(device)

    state_h, state_c = model.init_state(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    gen_seq = seed

    for _ in range(number_aa - len(seed)):  # Ensures the loop stops at the specified length
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
    for i in range(len(peptide)):
        peptide_fasta = f'>peptide_{i}\n{peptide[i]}'

    with open(file_path, 'w') as file:
        for fasta in peptide_fasta:
            file.write(fasta + "\n")
    print(f'Peptides stored in fasta format at: {file_path}')


def main():

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
        if pep in peptides:
            print('Peptide already in dataset \n Peptide marked for removal from generated sequences \n--------------------------\n')
            peptides_to_remove.add(pep)

    gen_sequences -= peptides_to_remove

    print(f'--------------------------\n Generated {len(gen_sequences)} new unique peptide sequences \n')

    short_len = len(min(gen_sequences, key=len))
    long_len = len(max(gen_sequences, key=len))

    print(f'Shortest peptide: {short_len} amino acids')
    print(f'Longest peptide: {long_len} amino acids\n')

    gen_seq_list = list(gen_sequences)
    seq_to_fasta(gen_seq_list, args.output_file)

if __name__ == "__main__":
    main()


#python LSTM-peptides-generation.py --dataset_path /Users/igorgonteri/Documents/GitHub/Nornour/0003b-RW-Lexicon/RW_lexicon.dat --output_path /Users/igorgonteri/Desktop/Nornour/0006-LSTM-peptides/generation_output.fasta --model_path lstm_peptides_model.pt --output_size 22 --epochs 50 --batch_size 8 --learning_rate 0.00063 --hidden_size 256 --layers 2 --dropout 0.7 --temperature 1.5 --num_sequences 100 --min_length 2 --max_length 20 --seed RWWR




    
