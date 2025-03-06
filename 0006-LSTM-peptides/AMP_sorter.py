from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


class SorterArgs:
    def __init__(self):
        self.input_path = '../0006b-LSTM-data/generation_peptides.fasta'
        self.sorting_path = '../0006b-LSTM-data/sorted_peptides.fasta'
        self.output_path = '../0006b-LSTM-data/potential_amp.fasta'



def parse_args():
    parser = argparse.ArgumentParser(description='Train and test LSTM model for peptide sequences')

    parser.add_argument('--input_path', type=str, required=True, help='Path to file with initial generated peptides (FASTA format)')
    parser.add_argument('--sorting_path', type=str, required=True, help='Path to file in which to store the sorted peptides (FASTA format)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to file in which to store the 20 best potential AMPs (FASTA format)')

    args = parser.parse_args()

    return args

args = parse_args()


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


def df_to_fasta(filepath, df):
    with open(filepath, 'w') as fasta_file:
        for i, seq in enumerate(df['Sequence']):
            # Format each peptide into FASTA format
            fasta_entry = f'>peptide_{i + 1}\n{seq}\n'
            fasta_file.write(fasta_entry)  # Write each FASTA entry to the file
    print(f'Peptides stored in FASTA format at: {filepath}')


def plot_helical_wheel(angles, n_residues, peptide):
    polar_AA = ['S', 'T', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    colors = ['blue' if aa in polar_AA else 'orange' for aa in peptide]

    for i in range(n_residues):
        ax.scatter(np.radians(angles[i]), 1, c=colors[i], s=100, zorder=3)

    ax.plot(np.radians(angles), np.ones(n_residues), color='grey', lw=1, zorder=2)

    for i, (residue, angle) in enumerate(zip(peptide, angles)):
        ax.text(np.radians(angle), 1.1, residue, ha='center', va='center', fontsize=12, color='black', zorder=4)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.5)
    ax.set_title('Helical Wheel Projection', fontsize=16)
    plt.show()


def amphiphilicity_calculator(peptide, plot=False):
    n_residues = len(peptide)
    angle_step = 100
    angles = np.arange(0, n_residues * angle_step, angle_step) % 360
    polar_AA = ['S', 'T', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']

    if plot:
        plot_helical_wheel(angles, n_residues, peptide)

    theta = np.arange(0, 360, 5)
    polar_AA_total = sum(1 for aa in peptide if aa in polar_AA)

    polar_AA_max = 0
    best_theta = 0

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


def hydrophobicity_calculator(peptide):
    X = ProteinAnalysis(peptide)
    hydrophobicity = X.gravy(scale='KyteDoolitle')
    return hydrophobicity


def charge_calculator(peptide):
    X = ProteinAnalysis(peptide)
    charge = X.charge_at_pH(7.4)
    return charge


def amp_sorter(fasta_file_input, fasta_file_output):
    peptide_df = fasta_to_df(fasta_file_input)
    peptide_df['Hydrophobic index'] = peptide_df['Sequence'].apply(lambda x: hydrophobicity_calculator(x))
    peptide_df['Charge (pH 7.4)'] = peptide_df['Sequence'].apply(lambda x: charge_calculator(x))
    peptide_df['Amphiphilicity'] = peptide_df['Sequence'].apply(lambda x: amphiphilicity_calculator(x, False))

    peptide_df = peptide_df.query("`Hydrophobic index` > 0.5")
    peptide_df = peptide_df.query("`Charge (pH 7.4)` > 2")
    peptide_df = peptide_df.query("`Charge (pH 7.4)` < 5")
    peptide_df = peptide_df.query("`Amphiphilicity` > 0.33")

    df_to_fasta(fasta_file_output, peptide_df)


def main():

    input_filepath = args.input_path
    output_filepath = args.sorting_path

    amp_sorter(input_filepath, output_filepath)


if __name__ == '__main__':
    main()





