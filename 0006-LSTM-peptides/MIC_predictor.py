import pandas as pd
from tqdm import tqdm
import math
from propy import PyPro
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class MICArgs:
    def __init__(self):
        self.data_path = '../0003e-DRAMP-MIC-database/DRAMP_MIC_p_aeruginosa.csv'
        self.batch_size = 8
        self.epochs = 50
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.num_layers = 2
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.max_seq_len = 15
        self.accuracy_percentage = 10.0
        self.train_ratio = 0.8
        self.vocab_size = 21
        self.train = True
        self.model_path = 'bi_lstm_peptides_model.pt'
        self.peptide_path = '../0006b-LSTM-data/sorted_peptides.fasta'
        self.prediction_path = '../0006b-LSTM-data/predictions.csv'


def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM-based peptide activity predictor.")

    parser.add_argument('--data_path', type=str, required=True, help="Path to the input dataset (CSV file).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimension of sequence embedding layer.")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Number of hidden units in LSTM.")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate between LSTM layers.")
    parser.add_argument('--max_seq_len', type=int, default=50, help="Maximum length of peptide sequences.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument('--accuracy_percentage', type=float, default=10.0, help="Percentage range for accuracy calculation.")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Proportion of data used for training.")
    parser.add_argument('--vocab_size', type=int, default=21, help='Vocabulary size (number of amino acids')
    parser.add_argument('--train', type=bool, default=True, help='Whether to train the model before prediction')

    args = parser.parse_args()
    return args


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


import numpy as np
import pandas as pd


def desc_to_df(df):
    """
    Add molecular descriptor features to a DataFrame containing peptide sequences.

    Args:
        - df (pd.DataFrame): A DataFrame with a column named 'Sequence', containing peptide sequences as strings.

    Returns:
        - pd.DataFrame: The input DataFrame with additional columns for normalized descriptor features:
            - 'AA_Composition': Amino acid composition descriptors (normalized).
            - 'CTD': Composition, transition, and distribution descriptors (scaled).
            - 'QSO': Quasi-sequence order descriptors (scaled).
        - dims (list): List containing the dimensions of each descriptor category.
        - mins_ctd, maxs_ctd: Min/max values for CTD features.
        - mins_qso, maxs_qso: Min/max values for QSO features.
    """

    aa_comp_list = []
    ctd_list = []
    qso_list = []

    # Calculate descriptors for each sequence in the df
    for sequence in df['Sequence']:
        aa_comp, ctd, qso = calculate_descriptors(sequence)
        aa_comp_list.append(aa_comp)
        ctd_list.append(ctd)
        qso_list.append(qso)

    # Convert descriptor lists into DataFrame columns
    df['AA_Composition'] = aa_comp_list
    df['CTD'] = ctd_list
    df['QSO'] = qso_list

    ctd_array = np.stack(ctd_list)
    qso_array = np.stack(qso_list)

    # Compute feature-wise min and max
    mins_ctd = ctd_array.min(axis=0)
    maxs_ctd = ctd_array.max(axis=0)
    mins_qso = qso_array.min(axis=0)
    maxs_qso = qso_array.max(axis=0)

    # Normalize AA_Composition
    df['AA_Composition'] = df['AA_Composition'].apply(lambda x: [val / 100 for val in x])

    # Normalize CTD descriptors
    df['CTD'] = list((ctd_array - mins_ctd) / (maxs_ctd - mins_ctd + 1e-8))

    # Normalize QSO descriptors
    df['QSO'] = list((qso_array - mins_qso) / (maxs_qso - mins_qso + 1e-8))

    aa_comp_dim = len(df['AA_Composition'].iloc[0])
    ctd_dim = len(df['CTD'].iloc[0])
    qso_dim = len(df['QSO'].iloc[0])
    dims = [aa_comp_dim, ctd_dim, qso_dim]

    return df, dims, mins_ctd, maxs_ctd, mins_qso, maxs_qso


def fasta_to_df(filepath):
    with open(filepath, 'r') as fasta_file:
        fasta = fasta_file.readlines()
        fasta = [pep.strip() for pep in fasta if pep.strip()]
        peptides = []
        for i, line in enumerate(fasta):
            if i % 2:
                peptides.append(line)

    peptides_df = pd.DataFrame(peptides, columns=['Sequence'])
    peptides_df = peptides_df[~peptides_df['Sequence'].str.contains('U')]  # "~" negates the condition

    return peptides_df


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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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


def activity_predictor(sequence, model, max_seq_len, min_mic, max_mic, mins_ctd, mins_qso, maxs_ctd, maxs_qso, accuracy_percentage):
    """
    Predicts the Minimum Inhibitory Concentration (MIC) for a given peptide sequence using a trained model.

    Args:
        - sequence (str): Peptide sequence to evaluate.
        - model (torch.nn.Module): Pre-trained LSTM-based activity prediction model.
        - max_seq_len (int): Maximum sequence length for padding/truncation.
        - min_mic (float): Minimum MIC value in the training dataset (for normalization reversal).
        - max_mic (float): Maximum MIC value in the training dataset (for normalization reversal).
        - mins_ctd (list): Minimum CTD descriptor value in the training dataset.
        - maxs_ctd (list): Maximum CTD descriptor value in the training dataset.
        - mins_qso (list): Minimum QSO descriptor value in the training dataset.
        - maxs_qso (list): Maximum QSO descriptor value in the training dataset.
        - accuracy_percentage (float): Allowed percentage deviation for the confidence interval.

    Returns:
        - pred_mic (float): Predicted MIC value in µM.
        - lower_bound (float): Lower bound of the MIC confidence interval in µM.
        - upper_bound (float): Upper bound of the MIC confidence interval in µM.
    """
    aa_comp, ctd, qso = calculate_descriptors(sequence)
    aa_comp_normalized = [val / 100 for val in aa_comp]

    ctd = np.array(ctd)
    qso = np.array(qso)

    ctd_normalized = (ctd - mins_ctd) / (maxs_ctd - mins_ctd + 1e-8)
    qso_normalized = (qso - mins_qso) / (maxs_qso - mins_qso + 1e-8)

    aa_comp_tensor = torch.tensor(aa_comp_normalized, dtype=torch.float32)
    ctd_tensor = torch.tensor(ctd_normalized, dtype=torch.float32)
    qso_tensor = torch.tensor(qso_normalized, dtype=torch.float32)

    char_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY_")}
    padded = padding(sequence, max_seq_len)
    encoded = [char_to_idx.get(aa, char_to_idx['_']) for aa in padded]
    sequence_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred_mic_log10_normalized, _ = model(sequence_tensor, aa_comp_tensor.unsqueeze(0), ctd_tensor.unsqueeze(0),
                                             qso_tensor.unsqueeze(0))
    print(pred_mic_log10_normalized)
    # Denormalize the predicted MIC
    pred_mic_log10 = (pred_mic_log10_normalized * (max_mic - min_mic)) + min_mic
    pred_mic = 10 ** pred_mic_log10  # Convert log10 to MIC

    # Calculate tolerance range based on predicted MIC and accuracy percentage
    tolerance_range = (accuracy_percentage / 100) * pred_mic
    lower_bound = pred_mic - tolerance_range
    upper_bound = pred_mic + tolerance_range
    print(pred_mic_log10_normalized)
    # Print prediction with uncertainty
    print(f"Predicted MIC for sequence {sequence}: {pred_mic.item():.4f} µM ± {tolerance_range.item():.4f} µM")
    print(f"Confidence Interval: {lower_bound.item():.4f} µM to {upper_bound.item():.4f} µM")

    return pred_mic, lower_bound, upper_bound, tolerance_range






def main():
    args = get_args()


    # Create an empty list to store rows
    data = []

    with open(args.data_path, "r") as infile:
    # Read the header line
        headers = infile.readline().strip()

         # Process each subsequent line
        for line in infile:
            columns = line.strip().split(",")
            data.append(columns)

    # Create a DataFrame with the collected data
    df = pd.DataFrame(data, columns=['Sequence', 'Molecular_Weight[g/mol]', 'Standardised_MIC[µM]'])
    df['Standardised_MIC[µM]']= df['Standardised_MIC[µM]'].apply(lambda x: float(x))

    #taking log10 of MIC and scaling between min and max values
    df['log10_Standardised_MIC'] = df['Standardised_MIC[µM]'].apply(lambda x: math.log10(x))
    min_val = df['log10_Standardised_MIC'].min()
    max_val = df['log10_Standardised_MIC'].max()
    df['scaled_log10_MIC'] = (df['log10_Standardised_MIC'] - min_val) / (max_val - min_val)

    df, dims, mins_ctd, maxs_ctd, mins_qso, maxs_qso = desc_to_df(df)
    aa_comp_dim, ctd_dim, qso_dim = dims[0], dims[1], dims[2]

    model = LSTMActivityPredictor(
            embedding_dim=args.embedding_dim,
            aa_comp_dim=aa_comp_dim,
            ctd_dim=ctd_dim,
            qso_dim=qso_dim,
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            dropout= args.dropout
        )
    if args.train:
        dataset = PeptideDataset(df, max_seq_len=args.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset_size = len(dataset)
        train_size = int(args.train_ratio * dataset_size)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        train(
            train_loader,
            model,
            epochs=args.epochs,
            percentage=args.accuracy_percentage
        )

        test_loss, test_accuracy = activity_test(test_loader, model, percentage=args.accuracy_percentage)

    else:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

    peptides_df = fasta_to_df(args.peptides_path)
    peptides_df[['MIC', 'Lower Bound', 'Upper Bound', 'Tolerance']] = df['Sequence'].apply(lambda x: activity_predictor(x ,model, max_seq_len=args.max_seq_len, min_mic=min_val, max_mic=max_val, mins_ctd=mins_ctd, mins_qso=mins_qso, maxs_ctd=maxs_ctd, maxs_qso=maxs_qso, accuracy_percentage=args.accuracy_percentage)).apply(pd.Series)
    df.to_csv(args.predictions_path, index=False)


if __name__ == "__main__":
    args = get_args()
    main()


