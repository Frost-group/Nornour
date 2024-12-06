from Bio.SeqUtils import molecular_weight
import pandas as pd
from tqdm import tqdm
import math
from propy import PyPro

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def standardize_unit(unit):
    """
    Standardize the units in the database

    Args:
        - unit (str): unit of the MIC as a string

    Returns:
         - unit (str): the same unit standardized at the good format
    """
    # Replace µ with μ (both are representations of "micro")
    unit = unit.replace('µ', 'μ')

    unit = unit.lower()

    if unit == 'μg/m':
        unit = 'μg/ml'

    # Replace ml with mL (standardizing case)
    unit = unit.replace('ml', 'ml')

    return unit


def unit_converter(value, unit, MW):
    """
    Convert the MIC value from nm, μg/ml or mg/l to µM

    Args:
         - value (float): value of MIC in initial unit in the database
         - unit (str): unit associated with value
         - MW (float): molecular weight of the peptide sequence

    Returns:
        - value (float): MIC value converted from unit to µM if necessary
    """
    if unit == 'μg/ml':
        return float(value) * 1000 / float(MW)
    elif unit == 'mg/l':
        return float(value) / 1000 / float(MW)
    elif unit == 'nm':
        return float(value) / 1000
    else:
        return float(value)


def padding(peptide, long_pep):
    """
    Add padding ('_') or truncate the input peptide sequence to long_pep

    Args:
         - peptide (str): sequence of amino acids of the peptide (ex: 'RWWGLL')
         - long_pep (int): length to which the sequence must be padded or truncated to (ex: 9)

    Returns:
        - peptide (str): peptide sequence padded or truncated to long_pep ('ex: 'RWWGLL___')
    """
    pad_length = long_pep - len(peptide)

    if pad_length > 0:  #padding if  too short
        peptide = peptide + '_' * pad_length

    if pad_length < 0: #truncation if too long
        peptide = peptide[:long_pep]

    return peptide


def calculate_descriptors(sequence):
    """
    Calculate the molecular descriptors of a peptide sequence.

    Args:
        - sequence (str): Sequence of amino acids in the peptide (e.g., 'RWWGLL').

    Returns:
        - aa_comp (list): Amino acid composition as a list of normalized values.
        - ctd (list): Composition, transition, and distribution (CTD) descriptors.
        - qso (list): Quasi-sequence order (QSO) descriptors, capturing sequence order and physicochemical properties.
    """
    protein = PyPro.GetProDes(sequence)
    aa_comp = list(protein.GetAAComp().values())
    ctd = list(protein.GetCTD().values())
    qso = list(protein.GetQSO().values())

    return aa_comp, ctd, qso


def desc_to_df(df):
    """
    Add molecular descriptor features to a DataFrame containing peptide sequences.

    Args:
        - df (pd.DataFrame): A DataFrame with a column named 'Sequence', containing peptide sequences as strings.

    Returns:
        - pd.DataFrame: The input DataFrame with additional columns for normalized descriptor features:
            - 'AA_Composition': Amino acid composition descriptors (normalized).
            - 'CDT': Composition, transition, and distribution descriptors (scaled).
            - 'QSO': Quasi-sequence order descriptors (scaled).
    """
    aa_comp_list = []
    ctd_list = []
    qso_list = []

    #calculates the descriptors of each sequence in the df
    for sequence in df['Sequence']:
        aa_comp, ctd, qso = calculate_descriptors(sequence)
        aa_comp_list.append(aa_comp)
        ctd_list.append(ctd)
        qso_list.append(qso)

    # Add lists as columns and apply normalization
    df['AA_Composition'] = aa_comp_list
    df['AA_Composition'] = df['AA_Composition'].apply(lambda x: [val / 100 for val in x])

    #min-max normalization
    df['CTD'] = ctd_list
    ctd_min = df['CTD'].apply(lambda x: min(x)).min()
    ctd_max = df['CTD'].apply(lambda x: max(x)).max()
    df['CTD'] = df['CTD'].apply(lambda x: [(val - ctd_min) / (ctd_max - ctd_min) for val in x])

    #min-max normalization
    df['QSO'] = qso_list
    qso_min = df['QSO'].apply(lambda x: min(x)).min()
    qso_max = df['QSO'].apply(lambda x: max(x)).max()
    df['QSO'] = df['QSO'].apply(lambda x: [(val - qso_min) / (qso_max - qso_min) for val in x])

    return df


class PeptideDataset(Dataset):
    def __init__(self, dataframe, max_seq_len=50):
        """
        Initialize the PeptideDataset.

        Args:
            - dataframe (pd.DataFrame): DataFrame containing 'Sequence' and descriptor columns.
            - max_seq_len (int): Maximum sequence length for padding/truncation.
        """
        self.dataframe = dataframe
        self.max_seq_len = max_seq_len

        # Amino acid to index mapping
        self.char_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY_")}
        self.vocab_size = len(self.char_to_idx)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataframe)

    def encode_sequence(self, sequence):
        """
        Encode a peptide sequence as an integer tensor.

        Args:
            - sequence (str): Peptide sequence (e.g., "ACDE").

        Returns:
            - Tensor: Combined encoded sequence tensor
        """
        # Pad and encode the peptide sequence
        padded = padding(sequence, self.max_seq_len)
        encoded = [self.char_to_idx.get(aa, self.char_to_idx['_']) for aa in padded]

        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        """
        Retrieves the data sample at the specified index.

        Args:
            - idx (int): Index of the sample to retrieve.

        Returns:
            - sequence_tensor (Tensor): Encoded tensor representation of the peptide sequence.
            - aa_comp (Tensor): Tensor containing the normalized amino acid composition descriptors.
            - ctd (Tensor): Tensor containing the scaled CTD (composition, transition, distribution) descriptors.
            - qso (Tensor): Tensor containing the scaled quasi-sequence order descriptors.
            - label (Tensor): Target value (scaled log10 MIC) for regression.
        """
        # Fetch the row corresponding to the index
        row = self.dataframe.iloc[idx]
        sequence = row['Sequence']

        #Encode molecular descriptors to tensors
        aa_comp = torch.tensor(row['AA_Composition'], dtype=torch.float32)
        ctd = torch.tensor(row['CTD'], dtype=torch.float32)
        qso =  torch.tensor(row['QSO'], dtype=torch.float32)

        label = torch.tensor(row['scaled_log10_MIC'], dtype=torch.float32)  # Target value for regression

        # Encode the sequence
        sequence_tensor = self.encode_sequence(sequence)
        return sequence_tensor, aa_comp, ctd, qso, label


class LSTMActivityPredictor(nn.Module):
    """
    A deep learning model for predicting peptide antimicrobial activity using an LSTM for sequence modeling
    and feedforward layers for molecular descriptors.

    Args:
        - embedding_dim (int): Dimension of the embedding vectors for sequence tokens.
        - hidden_dim (int): Number of hidden units in the LSTM.
        - aa_comp_dim (int): Dimensionality of the amino acid composition feature vector.
        - ctd_dim (int): Dimensionality of the CTD (composition, transition, distribution) feature vector.
        - qso_dim (int): Dimensionality of the quasi-sequence order feature vector.
        - vocab_size (int): Size of the vocabulary (number of unique sequence tokens).
        - batch_size (int): Size of the input batch.
        - num_layers (int): Number of layers in the LSTM.
        - dropout (float): Dropout rate applied between LSTM layers.

    Attributes:
        - embedding (nn.Embedding): Embedding layer to encode sequence tokens into dense vectors.
        - lstm (nn.LSTM): Bi-directional LSTM for sequence modeling.
        - fc_aa_comp (nn.Sequential): Feedforward layers for amino acid composition descriptors.
        - fc_ctd (nn.Sequential): Feedforward layers for CTD descriptors.
        - fc_qso (nn.Sequential): Feedforward layers for QSO descriptors.
        - fc_intermediate (nn.Linear): Intermediate feedforward layer to combine descriptor outputs.
        - fc_final (nn.Linear): Final layer for MIC value prediction.
    """
    def __init__(self, embedding_dim, hidden_dim, aa_comp_dim, ctd_dim, qso_dim, vocab_size, batch_size, num_layers, dropout):
        super(LSTMActivityPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.aa_comp_dim = aa_comp_dim
        self.ctd_dim = ctd_dim
        self.qso_dim = qso_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.fc_aa_comp = nn.Sequential(
            nn.Linear(self.aa_comp_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        )
        self.fc_ctd = nn.Sequential(
            nn.Linear(self.ctd_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        )
        self.fc_qso = nn.Sequential(
            nn.Linear(self.qso_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        )
        self.fc_intermediate = nn.Linear(hidden_dim * 3 , 2 * hidden_dim)
        self.fc_final = nn.Linear(4 * hidden_dim, 1)

    def forward(self, sequence, aa_comp, ctd, qso):
        # Initialize hidden and cell states
        h_0 = torch.zeros(self.num_layers * 2, sequence.size(0), self.hidden_dim).to(sequence.device)
        c_0 = torch.zeros(self.num_layers * 2, sequence.size(0), self.hidden_dim).to(sequence.device)

        # Embedding layer
        embeds = self.embedding(sequence).permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]

        # LSTM for peptide sequence
        lstm_out, (h_f, c_f) = self.lstm(embeds, (h_0, c_0))
        lstm_out = lstm_out[-1, :, :]

        # FC layers for descriptors
        aa_comp_out = self.fc_aa_comp(aa_comp)
        ctd_out = self.fc_ctd(ctd)
        qso_out = self.fc_qso(qso)
        combined_desc = torch.cat([aa_comp_out, ctd_out, qso_out], dim=1)
        combined_desc_2 = self.fc_intermediate(combined_desc)

        #combining and FC layer to get MIC value
        combined = torch.cat([lstm_out, combined_desc_2], dim=1)
        MIC_val = self.fc_final(combined) # [batch_size, 1]

        return MIC_val, (h_f, c_f)


def train(dataloader, model, epochs, percentage=10):
    """
    Train the model and calculate accuracy based on a percentage range.

    Args:
    - train_dataloader: DataLoader object containing training data.
    - val_dataloader: DataLoader object containing validation data.
    - model: The model to train.
    - epochs: Number of training epochs.
    - percentage: The acceptable range for accuracy calculation, in percent.
    """
    model.train()
    criterion = nn.MSELoss() #Mean Squared Error loss for regression task
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print('\n-------Starting Training------- \n_________________________________')

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(dataloader)
        with tqdm(total=num_batches, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=True) as pbar:
            for batch in dataloader:

                sequence, aa_comp, ctd, qso, labels = batch
                # Forward pass
                preds, _ = model(sequence, aa_comp, ctd, qso)
                loss = criterion(preds.view(-1), labels.view(-1))  # Flatten for loss calculation

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

                # Accuracy Calculation based on percentage threshold
                tolerance_range = (percentage / 100) * labels
                lower_bound = labels - tolerance_range
                upper_bound = labels + tolerance_range

                correct_preds = (preds.view(-1) >= lower_bound) & (preds.view(-1) <= upper_bound)
                accuracy = correct_preds.float().sum() / correct_preds.numel()
                epoch_accuracy += accuracy.item()

                avg_epoch_loss = epoch_loss / num_batches
                avg_epoch_accuracy = epoch_accuracy / num_batches

                pbar.update(1)
                pbar.set_postfix(loss=avg_epoch_loss, accuracy=avg_epoch_accuracy)

    print('-------End of Training--------\n-----------------------------\n')


def activity_test(dataloader, model, percentage=10):
    """
    Test the model on the test dataset.

    Args:
        - dataloader: DataLoader object containing test data.
        - model: The trained model to test.
        - percentage (float): Percentage threshold for accuracy calculation.

    Returns:
        - test_loss: The average loss on the test dataset.
        - test_accuracy: The average accuracy on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss()  # Use Mean Squared Error Loss

    test_loss = 0
    test_accuracy = 0
    num_batches = len(dataloader)

    print("\n-------Starting Testing-------\n")

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch in dataloader:
            sequence, aa_comp, ctd, qso, labels = batch

            # Forward pass
            preds, _ = model(sequence, aa_comp, ctd, qso)

            # Calculate loss
            loss = criterion(preds.view(-1), labels.view(-1))
            test_loss += loss.item()

            # Accuracy Calculation (percentage tolerance)
            tolerance_range = (percentage / 100) * labels
            lower_bound = labels - tolerance_range
            upper_bound = labels + tolerance_range
            correct_preds = (preds.view(-1) >= lower_bound) & (preds.view(-1) <= upper_bound)
            accuracy = correct_preds.float().sum() / correct_preds.numel()
            test_accuracy += accuracy.item()

    avg_test_loss = test_loss / num_batches
    avg_test_accuracy = test_accuracy / num_batches

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy * 100:.2f}%")
    return avg_test_loss, avg_test_accuracy


def activity_predictor(sequence, model, max_seq_len, min_mic, max_mic, min_ctd, min_qso, max_ctd, max_qso, accuracy_percentage):
    """
    Predicts the Minimum Inhibitory Concentration (MIC) for a given peptide sequence using a trained model.

    Args:
        - sequence (str): Peptide sequence to evaluate.
        - model (torch.nn.Module): Pre-trained LSTM-based activity prediction model.
        - max_seq_len (int): Maximum sequence length for padding/truncation.
        - min_mic (float): Minimum MIC value in the training dataset (for normalization reversal).
        - max_mic (float): Maximum MIC value in the training dataset (for normalization reversal).
        - min_ctd (float): Minimum CTD descriptor value in the training dataset.
        - max_ctd (float): Maximum CTD descriptor value in the training dataset.
        - min_qso (float): Minimum QSO descriptor value in the training dataset.
        - max_qso (float): Maximum QSO descriptor value in the training dataset.
        - accuracy_percentage (float): Allowed percentage deviation for the confidence interval.

    Returns:
        - pred_mic (float): Predicted MIC value in µM.
        - lower_bound (float): Lower bound of the MIC confidence interval in µM.
        - upper_bound (float): Upper bound of the MIC confidence interval in µM.
    """
    # Calculate descriptors (AA composition, CTD, QSO)
    aa_comp, ctd, qso = calculate_descriptors(sequence)

    # Normalize AA composition (already normalized in the dataset preparation)
    aa_comp_normalized = [val / 100 for val in aa_comp]  # Normalization to sum to 1

    # Normalize CTD and QSO
    ctd_normalized = [(val - min_ctd) / (max_ctd - min_ctd) for val in ctd]
    qso_normalized = [(val - min_qso) / (max_qso - min_qso) for val in qso]

    # Convert normalized descriptors to tensors
    aa_comp_tensor = torch.tensor(aa_comp_normalized, dtype=torch.float32)
    ctd_tensor = torch.tensor(ctd_normalized, dtype=torch.float32)
    qso_tensor = torch.tensor(qso_normalized, dtype=torch.float32)

    # Encode the sequence
    char_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY_")}
    padded = padding(sequence, max_seq_len)
    encoded = [char_to_idx.get(aa, char_to_idx['_']) for aa in padded]
    sequence_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    # Make the model prediction
    model.eval()
    with torch.no_grad():
        pred_mic_log10_normalized, _ = model(sequence_tensor, aa_comp_tensor.unsqueeze(0), ctd_tensor.unsqueeze(0),
                                             qso_tensor.unsqueeze(0))

    # Denormalize the predicted MIC
    pred_mic_log10 = (pred_mic_log10_normalized * (max_mic - min_mic)) + min_mic
    pred_mic = 10 ** pred_mic_log10  # Convert log10 to MIC

    # Calculate tolerance range based on predicted MIC and accuracy percentage
    tolerance_range = (accuracy_percentage / 100) * pred_mic
    lower_bound = pred_mic - tolerance_range
    upper_bound = pred_mic + tolerance_range

    # Print prediction with uncertainty
    print(f"Predicted MIC for sequence {sequence}: {pred_mic:.4f} µM ± {tolerance_range:.4f} µM")
    print(f"Confidence Interval: {lower_bound:.4f} µM to {upper_bound:.4f} µM")
    return pred_mic, lower_bound, upper_bound


input_file = 'final_values.txt'

# Create an empty list to store rows
data = []

with open(input_file, "r") as infile:
    # Read the header line (if needed)
    headers = infile.readline().strip()

    # Process each subsequent line
    for line in infile:
        columns = line.strip().split(",")
        data.append(columns)

# Create a DataFrame with the collected data
df = pd.DataFrame(data, columns=["Sequence", "Average_MIC", "Unit"])

# Define valid amino acids
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

# Filter out rows with invalid sequences
df = df[df['Sequence'].apply(lambda seq: all(aa in valid_amino_acids for aa in seq))]

# Reset the index of the DataFrame after filtering
df = df.reset_index(drop=True)

df['Unit'] = df['Unit'].apply(standardize_unit)
df['Molecular_Weight[g/mol]'] = df['Sequence'].apply(lambda x: molecular_weight(x, seq_type='protein'))

df['Standardised_MIC[µM]'] = df.apply(
    lambda row: unit_converter(row['Average_MIC'], row['Unit'], row['Molecular_Weight[g/mol]']),
    axis=1
)
#taking log10 of MIC and scaling between min and max values
df['log10_Standardised_MIC[µM]'] = df['Standardised_MIC[µM]'].apply(lambda x: math.log10(x))
min_val = df['log10_Standardised_MIC[µM]'].min()
max_val = df['log10_Standardised_MIC[µM]'].max()
df['scaled_log10_MIC'] = (df['log10_Standardised_MIC[µM]'] - min_val) / (max_val - min_val)

df = desc_to_df(df)

aa_comp_dim = len(df['AA_Composition'].iloc[0])  # Dimension of AA_Comp
ctd_dim = len(df['CTD'].iloc[0])                # Dimension of CTD
qso_dim = len(df['QSO'].iloc[0])                # Dimension of QSO
min_ctd = df['CTD'].min()
max_ctd = df['CTD'].max()
min_qso = df['QSO'].min()
max_qso = df['QSO'].max()

model = LSTMActivityPredictor(
    embedding_dim=100,
    aa_comp_dim=aa_comp_dim,
    ctd_dim=ctd_dim,
    qso_dim=qso_dim,
    vocab_size=21,
    hidden_dim=256,
    batch_size=64,
    num_layers=4,
    dropout= 0.3
)

dataset = PeptideDataset(df, max_seq_len=35)
dataloader = DataLoader(dataset, batch_size = 64, shuffle=True)

dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train(
    train_loader,
    model,
    epochs=20,
    percentage=10
)

test_loss, test_accuracy = activity_test(test_loader, model, percentage=10)

test_sequence = 'RWWWWWRGLLRD'
activity_predictor(
    test_sequence,
    model,
    max_seq_len=35,
    min_mic=min_val,
    max_mic=max_val,
    min_ctd=min_ctd,
    min_qso=min_qso,
    max_ctd=max_ctd,
    max_qso=max_qso,
    accuracy_percentage=10
)



