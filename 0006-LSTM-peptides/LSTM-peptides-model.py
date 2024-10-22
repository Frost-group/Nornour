import random
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Reading the file containing peptides and splitting into lines
filepath = '../0003c-APD-Database/antimicrobial_peptides_database.txt'
with open(filepath, 'r') as f:
    peptides = [line.strip() for line in f.readlines() if line.strip()]

# Vocabulary based on amino acids and padding character '_'
vocab = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', '_']
len_vocab = len(vocab)

class LSTMpeptides(nn.Module):
    def __init__(self, input_d, hidden_d, len_vocab=22, layers=1):
        super(LSTMpeptides, self).__init__()
        self.input_d = input_d
        self.hidden_d = hidden_d
        self.len_vocab = len_vocab
        self.layers = layers

        self.embed = nn.Embedding(input_d, hidden_d)
        self.lstm = nn.LSTM(input_size=hidden_d, hidden_size=hidden_d, num_layers=self.layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_d, len_vocab)

    def forward(self, sequence, hidden, cell):
        out = self.embed(sequence)
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
        out = self.fc(out)
        return out, (hidden, cell)

    def init_hidden(self, batch):
        hidden = torch.zeros(self.layers, self.hidden_d).to(device)
        cell = torch.zeros(self.layers, self.hidden_d).to(device)
        return hidden, cell

# Hyperparameters
batch_size = 64
hidden_size = 256
n_epochs = 5000
num_layers = 2
lr = 0.0001

class Generator():
    def __init__(self, hidden_d, layers, num_epochs, batch, lr=1e-3):
        self.num_epoch = num_epochs
        self.batch = batch
        self.print_every = 50
        self.hidden_d = hidden_d
        self.layers = layers
        self.lr = lr

    def aa_tensor(self, string):
        """Convert peptide string to tensor of indices corresponding to vocabulary."""
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = vocab.index(string[c])
        return tensor

    def get_random_batch(self):
        """Fetches a random peptide from the dataset for training."""
        selected_peptides = random.choices(peptides, k=self.batch)

        # Prepare lists to store input and target tensors for each peptide
        text_input = []
        text_target = []

        for peptide in selected_peptides:
            peptide_len = len(peptide)

            # Input: peptide sequence up to the second-to-last character
            peptide_input = peptide[:-1]
            # Target: peptide sequence from the second character onward
            peptide_target = peptide[1:]

            # Convert to tensors
            peptide_input_tensor = self.aa_tensor(peptide_input)
            peptide_target_tensor = self.aa_tensor(peptide_target)

            # Append to batch
            text_input.append(peptide_input_tensor)
            text_target.append(peptide_target_tensor)

        # Return the list of tensors for input and target
        return text_input, text_target

    def generate(self, initial_str='A', predict_len=100, temp=1):
        hidden, cell = self.lstm.init_hidden(batch=self.batch)
        initial_input = self.aa_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.lstm(initial_input[p].view(1).to(device), hidden, cell)

        last_aa = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.lstm(last_aa.view(1).to(device), hidden, cell)
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
        writer = SummaryWriter(f'runs/peptide_model')

        print('===> Starting training')

        for epoch in range(1, self.num_epoch + 1):
            inp, target = self.get_random_batch()

            hidden, cell = self.lstm.init_hidden(batch=self.batch)
            self.lstm.zero_grad()
            loss = 0

            # Iterate over batch size (peptides in a batch)
            for i in range(len(inp)):
                inp_seq = inp[i].to(device)
                target_seq = target[i].to(device)

                # Initialize hidden state for each peptide sequence
                hidden, cell = self.lstm.init_hidden(batch=1)
                for c in range(inp_seq.size(0)):
                    output, (hidden, cell) = self.lstm(inp_seq[c].view(1), hidden, cell)
                    loss += criterion(output, target_seq[c].view(-1))

            loss.backward()
            optimizer.step()
            avg_loss = loss.item() / len(inp)

            if epoch % self.print_every == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss}')

            writer.add_scalar('Training Loss', avg_loss, global_step=epoch)

# Instantiate and train the generator
genpeptides = Generator(hidden_size, num_layers, n_epochs, batch_size, lr=lr)
genpeptides.train()
