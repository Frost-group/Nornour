# Peptide Generation and Analysis Pipeline

This part of the project provides a suite of Python scripts to train a Long Short-Term Memory (LSTM) model for peptide generation, generate novel peptide sequences, sort them based on physicochemical properties, predict their Antimicrobial Peptide (AMP) probability, and finally predict their Minimum Inhibitory Concentration (MIC) against target bacteria with a Bi-LSTM model.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Project Structure](#project-structure)
3.  [Workflow Overview](#workflow-overview)
4.  [Automated Directory Structure](#automated-directory-structure)
5.  [Usage](#usage)
    *   [Running the Full Pipeline (Semi-Automated)](#running-the-full-pipeline-semi-automated)
    *   [Running Scripts Individually](#running-scripts-individually)
        *   [1. `LSTM-peptide-model.py` (Train Generator)](#1-lstm-peptide-modelpy-train-generator)
        *   [2. `LSTM-peptides-generation.py` (Generate Peptides)](#2-lstm-peptides-generationpy-generate-peptides)
        *   [3. `AMP_sorter.py` (Sort Peptides)](#3-amp_sorterpy-sort-peptides)
        *   [4. `MIC_predictor.py` (Predict MIC)](#4-mic_predictorpy-predict-mic)
        *   [5. `AMP_proba_predictor.py` (Predict AMP Probability - Web Service)](#5-amp_proba_predictorpy-predict-amp-probability---web-service)
6.  [Workflow Summary File](#workflow-summary-file)
7.  [External Tools (amPEP)](#external-tools-ampep)

## Prerequisites

*   Python 3.7+
*   Install required Python packages:
    ```bash
    pip install torch pandas requests beautifulsoup4 biopython modlamp propy tqdm wandb
    ```
*   **Weights & Biases (wandb)**: You need to log in to wandb for experiment tracking during LSTM model training.
    ```bash
    wandb login
    ```
*   **Internet Connection**: Required for `AMP_proba_predictor.py` as it uses an external web service.
*   **(Optional) amPEP**: For AMP probability prediction triggered from `MIC_predictor.py`. See [amPEP GitHub](https://github.com/PseudoLab/amPEPpy) for installation. Ensure `ampep` is in your system PATH.

## Project Structure

(Assuming your scripts are in the root or a `scripts/` directory, and data/models are in relative paths like `../data` and `../models`)

```
your-repo/
├── LSTM-peptide-model.py
├── LSTM-peptides-generation.py
├── AMP_sorter.py
├── MIC_predictor.py
├── AMP_proba_predictor.py
├── data/ # Data directory (can be created by scripts)
│ └── gen_DD-MM-YYYY/
│   └── generated_peptides_X/
│     ├── gen_peptides.fasta
│     ├── sorted_peptides.fasta
│     ├── MIC_predictions.csv
│     ├── AMP_proba.tsv (from ampep CLI)
│     ├── predictions.csv (from AMP_proba_predictor.py)
│     └── generation_summary.txt
├── models/ # Model directory (can be created by scripts)
│ ├── lstm_peptides_model.pt
│ └── bi_lstm_peptides_model.pt
└── README.md
```


## Workflow Overview

The typical workflow is as follows:

1.  **Train Generator Model**: Use `LSTM-peptide-model.py` to train an LSTM model on a dataset of existing peptide sequences. This model learns the patterns of amino acid sequences.
2.  **Generate Peptides**: Use `LSTM-peptides-generation.py` with the trained model to generate novel peptide sequences.
3.  **Sort Peptides**: Use `AMP_sorter.py` to filter and sort the generated peptides based on various physicochemical properties relevant to AMPs (e.g., hydrophobicity, charge, amphiphilicity).
4.  **Predict MIC**: Use `MIC_predictor.py` to train a model (or use a pre-trained one) to predict the MIC values of the sorted peptides. This script can also trigger AMP probability prediction using the `amPEP` command-line tool.
5.  **Predict AMP Probability (Alternative)**: Use `AMP_proba_predictor.py` as an alternative or supplementary step to predict the AMP probability of peptides using an external web service.

Each script (except `AMP_proba_predictor.py`) can prompt you to run the next script in the sequence, allowing for a semi-automated pipeline execution.

## Automated Directory Structure

Several scripts create a timestamped and numbered directory structure to store outputs for each run:

*   Base directory: `../data/gen_DD-MM-YYYY/` (e.g., `../data/gen_25-12-2023/`)
*   Run-specific directory: `generated_peptides_X/` (e.g., `generated_peptides_1/`, `generated_peptides_2/`)

Output files like generated peptides, sorted peptides, predictions, and summaries are stored within these run-specific directories. When a script argument for an output path is set to `'d'`, it defaults to this structured path.

## Usage

### Running the Full Pipeline (Semi-Automated)

You can run the entire pipeline by executing the scripts one after another. Most scripts will ask if you want to proceed to the next step.

1.  **Train the peptide generation model:**
    ```bash
    python LSTM-peptide-model.py --dataset_path path/to/your/training_peptides.fasta --save_model True [other_lstm_model_args...]
    ```
    When prompted `Do you want to run peptide generation ? [y/n]:`, enter `y`. Configure generation parameters if needed, or accept defaults.

2.  **Peptide generation will run.**
    When prompted `Do you want to sort the peptides ? [y/n]:`, enter `y`.

3.  **Peptide sorting will run.**
    When prompted `Do you want to predict the MIC of the sorted peptides ? [y/n]:`, enter `y`. Configure MIC prediction parameters if needed.

4.  **MIC prediction will run.**
    When prompted `Run AMP probability prediciton ? [y/n]:` (this refers to `ampep` CLI via `MIC_predictor.py`), enter `y` if you have `amPEP` installed and want its predictions.

5.  **(Optional) Run `AMP_proba_predictor.py` separately if desired:**
    If you want to use the web service-based AMP probability prediction on the sorted peptides:
    ```bash
    python AMP_proba_predictor.py --sorting_path ../data/gen_DD-MM-YYYY/generated_peptides_X/sorted_peptides.fasta --output_path ../data/gen_DD-MM-YYYY/generated_peptides_X/predictions.csv
    ```
    (Replace `DD-MM-YYYY` and `X` with the actual date and run number).

### Running Scripts Individually

You can also run each script as a standalone tool.

---

#### 1. `LSTM-peptide-model.py` (Train Generator)

Trains an LSTM model for peptide sequence generation.

**Command:**
```bash
python LSTM-peptide-model.py --dataset_path <PATH> [OPTIONS]
```

**Arguments**
```
--dataset_path (str, required): Path to the dataset file (e.g., FASTA) containing peptide sequences for training. Example: ../../0003d-DBAASP-Database/Database_of_Antimicrobial_Activity_and_structure_of_Peptides
--output_size (int, default: 21): Size of the output layer (number of unique characters in vocab).
--epochs (int, default: 50): Number of epochs for training.
--batch_size (int, default: 16): Batch size for training and testing.
--learning_rate (float, default: 0.00063): Learning rate for the optimizer.
--hidden_size (int, default: 256): Hidden layer size of the LSTM.
--layers (int, default: 2): Number of LSTM layers.
--dropout (float, default: 0.7): Dropout rate.
--max_length (int, default: 15): Maximum peptide length to consider. Peptides longer will be truncated, shorter ones padded. If not set, uses the longest peptide in the dataset.
--vocab (str, default: "RHKDESTNQCGPAILMFWYV_"): String of amino acids and padding character defining the vocabulary.
--save_model (bool, default: False): If True, saves the trained model to ../models/lstm_peptides_model.pt.
```
**Outputs:**

* Trained model: `../models/lstm_peptides_model.pt (if --save_model True).
* Summary file: Appends training summary to `../data/gen_DD-MM-YYYY/generated_peptides_X/generation_summary.txt`.
* Prompts to run LSTM-peptides-generation.py.


#### 2. LSTM-peptides-generation.py (Generate Peptides)
Generates new peptide sequences using a pre-trained LSTM model.

**Command:**
```bash
python LSTM-peptides-generation.py --model_path <PATH> [OPTIONS]
```


**Arguments:**
```
--dataset_path (str, default: ../../0003d-DBAASP-Database/Database_of_Antimicrobial_Activity_and_structure_of_Peptides): Path to the original dataset used for training (used to check for novelty of generated sequences).
--output_path (str, default: 'd'): Path to the output FASTA file for generated sequences. If 'd', saves to the automated directory structure (e.g., ../data/gen_DD-MM-YYYY/generated_peptides_X/gen_peptides.fasta).
--model_path (str, default: ../models/lstm_peptides_model.pt): Path to the trained LSTM model (.pt file).
--output_size (int, default: 21): Size of the output layer (must match the trained model).
--epochs (int, default: 50): (Used for model architecture definition, not for retraining here).
--batch_size (int, default: 8): (Used for model architecture definition).
--learning_rate (float, default: 0.00063): (Used for model architecture definition).
--hidden_size (int, default: 256): Hidden layer size (must match the trained model).
--layers (int, default: 2): Number of LSTM layers (must match the trained model).
--dropout (float, default: 0.7): Dropout rate (must match the trained model).
--temperature (float, default: 1.0): Sampling temperature. Higher values increase randomness, lower values make generation more deterministic.
--num_sequences (int, default: 40000): Number of unique sequences to attempt to generate.
--min_length (int, default: 1): Minimum length of generated peptides.
--max_length (int, default: 15): Maximum length of generated peptides.
--seed (str, default: 'r'): Seed sequence to start generation. If 'r', a random seed is generated based on training data AA distribution.
```
**Outputs:**

* Generated peptides: FASTA file at output_path.
* Summary file: Appends generation summary to `../data/gen_DD-MM-YYYY/generated_peptides_X/generation_summary.txt`.
* Prompts to run `AMP_sorter.py`.

#### 3. AMP_sorter.py (Sort Peptides)
Filters and sorts peptides from a FASTA file based on physicochemical properties.

**Command:**
```
python MIC_predictor.py [OPTIONS]
```


**Arguments:**
```
--input_path (str, required): Path to the FASTA file containing peptides to be sorted (typically the output from LSTM-peptides-generation.py).
--sorting_path (str, default: 'd'): Path to the output FASTA file for sorted peptides. If 'd', saves to the automated directory structure (e.g., ../data/gen_DD-MM-YYYY/generated_peptides_X/sorted_peptides.fasta).
```
**Outputs:**

* Sorted peptides: FASTA file at sorting_path.
* Summary file: Appends sorting summary to `../data/gen_DD-MM-YYYY/generated_peptides_X/generation_summary.txt`.
* Prompts to run `MIC_predictor.py`.
#### 4. MIC_predictor.py (Predict MIC)
Trains or uses a pre-trained model to predict Minimum Inhibitory Concentration (MIC) for peptides.

**Command:**
```
python MIC_predictor.py [OPTIONS]
```

**Arguments:**
```
--data_path (str, default: ../../0003e-DRAMP-MIC-database/DRAMP_MIC_p_aeruginosa.csv): Path to the CSV dataset for training the MIC prediction model. Contains 'Sequence', 'Molecular_Weight[ g/mol ]', 'Standardised_MIC[µM]'.
--batch_size (int, default: 32): Batch size for training/evaluation.
--epochs (int, default: 50): Number of training epochs (if --train is active).
--embedding_dim (int, default: 128): Dimension of sequence embedding layer.
--hidden_dim (int, default: 128): Number of hidden units in LSTM.
--num_layers (int, default: 1): Number of LSTM layers.
--dropout (float, default: 0.5): Dropout rate.
--max_seq_len (int, default: 15): Maximum length of peptide sequences for the model.
--learning_rate (float, default: 1e-3): Learning rate for optimizer.
--weight_decay (float, default: 1e-5): Weight decay for optimizer.
--accuracy_percentage (float, default: 10.0): Percentage range for accuracy calculation (e.g., a prediction is "correct" if within +/- 10% of the true MIC).
--train_ratio (float, default: 0.8): Proportion of data for training (the rest is for testing).
--vocab_size (int, default: 21): Vocabulary size (number of amino acids + padding).
--train / --no-train (bool, default: True): Whether to train the MIC model. If --no-train, a pre-trained model specified by --model_path is loaded.
--save_model (bool, default: False): If True and --train is active, saves the trained MIC model.
--model_path (str, default: ../models/bi_lstm_peptides_model.pt): Path to save/load the MIC prediction model.
--peptide_path (str, default: 'd'): Path to the FASTA file of peptides for MIC prediction (typically the output from AMP_sorter.py). If 'd', uses ../data/gen_DD-MM-YYYY/generated_peptides_X/sorted_peptides.fasta.
--prediction_path (str, default: 'd'): Path to save the CSV file with MIC predictions. If 'd', saves to ../data/gen_DD-MM-YYYY/generated_peptides_X/MIC_predictions.csv.
```
**Outputs:**

* MIC predictions: CSV file at prediction_path.
* Trained model: model_path (if `--train` and `--save_model True`).
* Summary file: Appends MIC prediction summary to `../data/gen_DD-MM-YYYY/generated_peptides_X/generation_summary.txt`.
* Prompts to run ampep CLI for AMP probability prediction.

#### 5. AMP_proba_predictor.py (Predict AMP Probability - Web Service)
Predicts AMP probability for peptides using the CAMP3 web service.

**Command:**
```
python AMP_proba_predictor.py --sorting_path <PATH_TO_SORTED_FASTA> --output_path <PATH_TO_PREDICTIONS_CSV>
```

**Arguments:**
```
--sorting_path (str, required): Path to the FASTA file containing peptides for which AMP probability needs to be predicted (typically the output from AMP_sorter.py).
--output_path (str, required): Path to the output CSV file where peptides and their predicted AMP probabilities will be stored.
```
Note: The script defines a SorterArgs class with default paths, but the parse_args() function makes these arguments required from the command line.

**Outputs:**

* AMP probability predictions: CSV file at output_path. Example: `../data/old_generation/predictions.csv` (if using defaults from SorterArgs and if CLI args were not overriding, but CLI args are required).

## Workflow Summary File
A summary file named generation_summary.txt is created/appended to within the run-specific directory (e.g., `../data/gen_DD-MM-YYYY/generated_peptides_X/generation_summary.txt`). 

This file logs:

* LSTM training model details and performance.
* Peptide generation configuration and statistics.
* AMP sorter statistics.
* MIC predictor model details, performance (if trained), and prediction summary.

## External Tools (amPEP)
The MIC_predictor.py script includes an option to call the amPEP command-line tool for AMP probability prediction. If you choose this option:

`ampep train ...` will be called (using hardcoded paths to amPEP's example training data). This trains an amPEP model (amPEP.model in the current directory).
`ampep predict ...` will be called using the newly trained amPEP.model and the sorted_peptides.fasta from your current run.
Output from ampep predict is saved to `AMP_proba.tsv` in the run-specific directory (e.g., `../data/gen_DD-MM-YYYY/generated_peptides_X/AMP_proba.tsv`).
This is distinct from the `AMP_proba_predictor.py` script, which uses a different web-based prediction method.




