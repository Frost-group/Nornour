import requests
import argparse
import pandas as pd
from bs4 import BeautifulSoup

class SorterArgs:
    def __init__(self):
        self.sorting_path = '../0006b-LSTM-data/sorted_peptides.fasta'
        self.output_path = '../0006b-LSTM-data/predictions.csv'


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test LSTM model for peptide sequences')

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


def amp_proba_predictor(filepath):
    url = 'http://www.camp3.bicnirrh.res.in/predict/hii.php'

    files = {'userfile': open(filepath, 'rb')}
    payload = {'algo[]': 'rf'}
    r = requests.post(url, data=payload, files=files)

    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table')
    values = []

    if table:
        rows = table.find_all('tr')[5:]
        values = []

        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 2:
                try:
                    value =  float(cells[2].text.strip())
                    values.append(value)
                except ValueError:
                    pass

    pred_AMP_proba = [v for v in values if v is not None]
    return pred_AMP_proba


def main():
    sorting_filepath = args.sorting_path
    output_file = args.output_path

    output_df = pd.read_csv(output_file)
    pred_AMP_proba = amp_proba_predictor(sorting_filepath)
    if pred_AMP_proba:
        print('Values predicted')

    output_df['AMP probability'] = pd.DataFrame(pred_AMP_proba)

    peptide_df_sorted = output_df.sort_values(by='AMP probability', ascending=False)
    peptide_df_sorted.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()