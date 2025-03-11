from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from modlamp.descriptors import GlobalDescriptor
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test LSTM model for peptide sequences')

    parser.add_argument('--input_path', type=str, required=True, help='Path to file with initial generated peptides (FASTA format)')
    parser.add_argument('--sorting_path', type=str, default='-d', help='Path to file in which to store the sorted peptides (FASTA format), default: -d')

    args = parser.parse_args()
    return args

args = parse_args()


def fasta_to_df(filepath):
    """
    Converts a FASTA file containing peptide sequences into a pandas DataFrame.

    Args:
        filepath (str): Path to the FASTA file.

    Returns:
        peptides_df (pandas.DataFrame): A DataFrame with a single column 'Sequence' containing the peptide sequences.
    """
    with open(filepath, 'r') as fasta_file:
        fasta = fasta_file.readlines()
        fasta = [pep.strip() for pep in fasta if pep.strip()]
        peptides = []
        for i, line in enumerate(fasta):
            if i % 2:
                peptides.append(line)

    peptides_df = pd.DataFrame(peptides, columns=['Sequence'])
    peptides_df = peptides_df[~peptides_df['Sequence'].str.contains('U')]

    return peptides_df


def df_to_fasta(filepath, df):
    """
    Writes a pandas Dataframe containing peptide sequences into a FASTA file.

    Args:
        - filepath (str): Path to the FASTA file.
        - df (pandas.Dataframe):

    Returns:
        peptides_df (pandas.DataFrame): A DataFrame with a single column 'Sequence' containing the peptide sequences.
    """
    with open(filepath, 'w') as fasta_file:
        for i, seq in enumerate(df['Sequence']):
            fasta_entry = f'>peptide_{i + 1}\n{seq}\n'
            fasta_file.write(fasta_entry)
    print(f'Peptides stored in FASTA format at: {filepath}')


def amphiphilicity_calculator(peptide):
    """
    Calculates the amphiphilicity of a peptide by projecting the peptide into a helix on a plane.

    Args:
        peptide (str): Amino acid sequence of the peptide.

    Returns:
        H_star (float): The amphiphilicity index of the peptide.
    """
    n_residues = len(peptide)
    angle_step = 100
    angles = np.arange(0, n_residues * angle_step, angle_step) % 360
    polar_AA = ['S', 'T', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']

    theta = np.arange(0, 360, 5)
    polar_AA_total = sum(1 for aa in peptide if aa in polar_AA)
    polar_AA_max = 0

    for j in theta:
        lower_limit = j
        upper_limit = (j + 180) % 360
        polar_AA_count = 0

        for residue, angle in zip(peptide, angles):
            if lower_limit < upper_limit:
                in_range = lower_limit <= angle <= upper_limit
            else:  # Case where range wraps around 360°
                in_range = (angle >= lower_limit or angle <= upper_limit)

            if in_range and residue in polar_AA:
                polar_AA_count += 1

        # Update max count if higher is found
        if polar_AA_count > polar_AA_max:
            polar_AA_max = polar_AA_count
            best_theta = j  # Store the best orientation

    # Compute H_star
    if polar_AA_total > 0:
        H_star_prime = polar_AA_max / polar_AA_total
        H_star = (H_star_prime - 0.5) * 2
    else:
        H_star = 0  # Avoid division by zero

    return H_star


def hydrophobic_percentage(peptide):
    """
    Calculates the percentage of hydrcophobic residues of a peptide.

    Args:
        peptide (str): Amino acid sequence of the peptide.

    Returns:
        percentage (float): Percentage of hydrophobic residues in the peptide.
    """
    hydrophobic_residues = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y', 'P'}
    total_length = len(peptide)
    hydrophobic_count = sum(1 for aa in peptide if aa in hydrophobic_residues)
    percentage = (hydrophobic_count / total_length)
    return round(percentage, 4)


def charge_calculator(peptide):
    """
    Calculates the charge of a peptide using the Bio package.

    Args:
        peptide (str): Amino acid sequence of the peptide

    Returns:
        charge (float): The charge of the amino acid at pH 7.4.
    """
    X = ProteinAnalysis(peptide)
    charge = X.charge_at_pH(7.4)
    return charge


def boman_index(peptide):
    """
    Calculates the binding index of a peptide (Boman index) using the modlamp package.

    Args:
        peptide (str): Amino acid sequence of the peptide

    Returns:
        desc.descriptor (float): The descriptor containing the Boman index of the peptide.
    """
    desc = GlobalDescriptor(peptide)
    desc.boman_index()
    return desc.descriptor


def isoelectric_point(peptide):
    """
    Calculates the isoelectric point of a peptide using the Bio package.

    Args:
        peptide (str): amino acid sequence of the peptide

    Returns:
        ip (float): Isoelectric point of the peptide.
    """

    X = IP(peptide)
    ip = X.pi()
    return ip


def amp_sorter(fasta_file_input, fasta_file_output):
    """
    Sorts the peptides from a FASTA file.

    Args:
        fasta_file_output (str): FASTA file with peptides to sort.

    Returns:
        fasta_file_output (str): FASTA file with sorted peptides.
    """
    #Calculate descriptors of the peptides in the FASTA file.
    peptide_df = fasta_to_df(fasta_file_input)
    peptide_df['Hydrophobic residues percentage'] = peptide_df['Sequence'].apply(lambda x: hydrophobic_percentage(x))
    peptide_df['Charge (pH 7.4)'] = peptide_df['Sequence'].apply(lambda x: charge_calculator(x))
    peptide_df['Amphiphilicity'] = peptide_df['Sequence'].apply(lambda x: amphiphilicity_calculator(x, False))
    peptide_df['Boman index'] = peptide_df['Sequence'].apply(lambda x: boman_index(x))
    peptide_df['Isoelectric point'] = peptide_df['Sequence'].apply(lambda x: isoelectric_point(x))

    initial_length = len(peptide_df)
    print(f'Initial number of peptides: {initial_length}')

    #Sort the peptides using threshold values based on DRAMP database and RW lexicon.
    peptide_df = peptide_df.query("`Hydrophobicity percentage` > 0.35")
    peptide_df = peptide_df.query("`Hydrophobicity percentage` < 0.7")
    peptide_df = peptide_df.query("`Charge (pH 7.4)` > 2")
    peptide_df = peptide_df.query("`Charge (pH 7.4)` < 6")
    peptide_df = peptide_df.query("`Amphiphilicity` > 0.33")
    peptide_df = peptide_df.query('`Boman index`> 0 ')
    peptide_df = peptide_df.query('`Boman index` < 8 ')
    peptide_df = peptide_df.query('`Isoelectric point` > 10 ')

    output_length = len(peptide_df)
    print(f'Number of peptides after sorting: {output_length}')

    #Puts the sorted peptides into the output file
    df_to_fasta(fasta_file_output, peptide_df)
    return initial_length, output_length


def main():
    input_filepath = args.input_path
    today = datetime.today().strftime("%d-%m-%Y")
    base_dir = f"../data/gen_{today}"
    gen_count = len([d for d in os.listdir(base_dir) if today in d]) + 1
    gen_dir = os.path.join(base_dir, f"gen_{today}", f"generated_peptides_{gen_count}")
    if args.output_filepath == '-d':
        output_filepath = os.path.join(gen_dir, 'sorted_peptides.fasta')
    else:
        output_filepath = args.sorting_path

    initial_length, output_length = amp_sorter(input_filepath, output_filepath)

    sum_path = os.path.join(gen_dir, "generation_summary.txt")
    with open(sum_path, 'a') as file:
        file.write(f'\n Sorting summary: \n')
        file.write(f'\t- Initial number of peptides: {initial_length} \n')
        file.write(f'\t- Number of peptides after sorting: {output_length} \n')

    sorting_input = input('Do you want to predict the MIC of the sorted peptides ? [y/n]: ')
    if sorting_input.lower() == 'y':
        print('MIC prediction command: \n')
        print(f'python MIC_predictor.py --data_path ../../0003e-DRAMP-MIC-database/DRAMP_MIC_p_aeruginosa.csv --batch_size 32 --epochs 50 --embedding_dim 128 --hidden_dim 128 --num_layers 1 --dropout 0.5 --max_seq_len 15 --learning_rate 1e-3 --weight_decay 1e-5 --accuracy_percentage 10.0 --train_ratio 0.8 --vocab_size 21 --train True --save_model False --model_path ../models/bi_lstm_peptides_model.pt --peptide_path -d --prediction_path -d\n')
        config_input = input('Keep configuration ? [y/n]: ')

        if config_input.lower() == 'n':
            data_path = input("Enter the path to the dataset: ")
            batch_size = int(input("Enter batch size (int): "))
            epochs = int(input("Enter number of epochs (int): "))
            embedding_dim = int(input("Enter embedding dimension (int): "))
            hidden_dim = int(input("Enter hidden dimension (int): "))
            num_layers = int(input("Enter number of LSTM layers (int): "))
            dropout = float(input("Enter dropout rate (float): "))
            max_seq_len = int(input("Enter max sequence length (int): "))
            learning_rate = float(input("Enter learning rate (float): "))
            weight_decay = float(input("Enter weight decay (float): "))
            accuracy_percentage = float(input("Enter accuracy percentage (float): "))
            train_ratio = float(input("Enter train ratio (float): "))
            vocab_size = int(input("Enter vocabulary size (int): "))
            train = input("Train the model? (True/False): ").strip().lower() == "true"
            save_model = input("Save the model? (True/False): ").strip().lower() == "true"
            model_path = input("Enter the model path: ")
            peptide_path = input("Enter the peptide path: ")
            prediction_path = input("Enter the prediction path: ")

        else:
            data_path = '../../0003e-DRAMP-MIC-database/DRAMP_MIC_p_aeruginosa.csv'
            batch_size =32
            epochs = 50
            embedding_dim = 128
            hidden_dim = 128
            num_layers = 1
            dropout = 0.5
            max_seq_len = 15
            learning_rate = 1e-3
            weight_decay = 1e-5
            accuracy_percentage = 10.0
            train_ratio = 0.8
            vocab_size = 21
            train = True
            save_model = False
            model_path = '../models/bi_lstm_peptides_model.pt'
            peptide_path = '-d'
            prediction_path = '-d'

        subprocess.run([
            "python",
            "MIC_predictor.py",
            "--data_path", data_path,
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--embedding_dim", str(embedding_dim),
            "--hidden_dim", str(hidden_dim),
            "--num_layers", str(num_layers),
            "--dropout", str(dropout),
            "--max_seq_len", str(max_seq_len),
            "--learning_rate", str(learning_rate),
            "--weight_decay", str(weight_decay),
            "--accuracy_percentage", str(accuracy_percentage),
            "--train_ratio", str(train_ratio),
            "--vocab_size", str(vocab_size),
            "--train", str(train),
            "--save_model", str(save_model),
            "--model_path", model_path,
            "--peptide_path", peptide_path,
            "--prediction_path", prediction_path
        ])
    else:
        pass
if __name__ == '__main__':
    main()

