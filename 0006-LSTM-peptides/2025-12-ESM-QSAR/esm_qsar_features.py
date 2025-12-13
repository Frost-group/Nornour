import pandas as pd
import numpy as np
import torch
import h5py
from tqdm import tqdm
from propy import PyPro
from transformers import EsmTokenizer, EsmModel
import argparse
import os

# --- Configuration ---
# Mini 8M parameter model. 
# Most publications uses the medium 650M one, "facebook/esm2_t33_650M_UR50D"
# t6_8M, t12_35M, t30_150M, t33_650M, t36_3B, t48_15B
#    15B (!!!!)
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LEN = 64  # Maximum peptide length for full embeddings (covers most AMPs)
# 8M model super fast on my Mac cpu only
# 150M model on my Mac Studio CPU at work takes about 2 mins to run ~2000 sequences

def get_args():
    parser = argparse.ArgumentParser(description="Generate ESM and QSAR features.")
    parser.add_argument('--input_csv', type=str, required=True, help="input CSV containing 'Sequence' and 'MIC'.")
    parser.add_argument('--output_h5', type=str, default='peptides_featurised.h5', help="Output HDF5 file.")
    return parser.parse_args()

def calculate_qsar_descriptors(sequence):
    """
    Calculates raw descriptors using PyPro.
    NO NORMALISATION! 
    """
    try:
        protein = PyPro.GetProDes(sequence)
        
        # 1. Amino Acid Composition
        aa_comp = list(protein.GetAAComp().values())
        
        # 2. CTD (Composition, Transition, Distribution)
        ctd = list(protein.GetCTD().values())
        
        # 3. QSO (Quasi-sequence order)
        qso = list(protein.GetQSO().values())
        
        # Concatenate all QSAR features into one vector
        return np.concatenate([aa_comp, ctd, qso])
        
    except Exception as e:
        print(f"Error calculating QSAR for {sequence}: {e}")
        return None

def extract_esm_embeddings(sequences, tokenizer, model):
    """
    Generates ESM embeddings for a list of sequences using batch processing.

    Takes the SeqLength x HiddenDim set of vectors and projects it down to
    a SINGLE vector of HiidenDem using MeanPooling.  
    """
    model.eval()
    embeddings = []
    
    # Process in batches to avoid OOM
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Extracting ESM Embeddings"):
        batch_seqs = sequences[i : i + BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get the 'last_hidden_state' [Batch, Seq_Len, Hidden_Dim]
        # We perform Mean Pooling over the sequence length to get one vector per peptide
        # Mask out padding tokens for accurate mean
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_output = outputs.last_hidden_state * attention_mask
        sum_embeddings = masked_output.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_embeddings = sum_embeddings / counts
        
        embeddings.append(mean_embeddings.cpu().numpy())
        
    return np.vstack(embeddings)


def extract_esm_embeddings_full(sequences, tokenizer, model, max_len=MAX_SEQ_LEN):
    """
    Extract full ESM embeddings [N_samples, max_len, hidden_dim] for BiLSTM processing.
    
    Returns:
        embeddings: np.ndarray [N, max_len, hidden_dim] - padded full sequence embeddings
        lengths: np.ndarray [N] - actual sequence lengths (excluding special tokens)
    """
    model.eval()
    all_embeddings = []
    all_lengths = []
    
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Extracting Full ESM Embeddings"):
        batch_seqs = sequences[i : i + BATCH_SIZE]
        
        # Pad to max_len for consistent tensor sizes
        inputs = tokenizer(
            batch_seqs, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_len
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # last_hidden_state: [Batch, max_len, Hidden_Dim]
        hidden_states = outputs.last_hidden_state.cpu().numpy()
        
        # Compute actual lengths (sum of attention mask, minus 2 for <cls> and <eos>)
        # But for BiLSTM we include all non-pad tokens, so just sum attention_mask
        lengths = inputs['attention_mask'].sum(dim=1).cpu().numpy()
        
        all_embeddings.append(hidden_states)
        all_lengths.append(lengths)
    
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_lengths, axis=0)

def main():
    args = get_args()
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # Filter to canonical amino acids only (no U, X, Z, O, J, B, lowercase, or special chars)
    canonical_aa = set('ACDEFGHIKLMNPQRSTVWY')
    df = df[df['Sequence'].apply(lambda s: set(s).issubset(canonical_aa))]
    
    sequences = df['Sequence'].tolist()
    
    # Handle targets (MIC) - Log transform here for convenience
    mics = df['MIC'].astype(float).values
    log_mics = np.log10(mics)

    # 2. Extract QSAR Features
    print("Calculating QSAR features...")
    qsar_list = []
    valid_indices = []
    
    for idx, seq in enumerate(tqdm(sequences)):
        feat = calculate_qsar_descriptors(seq)
        if feat is not None:
            qsar_list.append(feat)
            valid_indices.append(idx)
            
    qsar_matrix = np.vstack(qsar_list)
    
    # Filter sequences/labels to match valid QSAR calculations
    sequences = [sequences[i] for i in valid_indices]
    log_mics = log_mics[valid_indices]

    # 3. Extract ESM Embeddings
    print(f"Loading ESM-2 Model: {ESM_MODEL_NAME}...")
    print("~Jarv: Nb:  Don't worry about the Pool warning below, we're not using the Pool layer as we're not predicting the CLS token, just using the embeddings and mean pooling.")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE)
    
    # Mean-pooled embeddings (for MLP baseline)
    esm_matrix = extract_esm_embeddings(sequences, tokenizer, model)
    
    # Full sequence embeddings (for BiLSTM)
    esm_full, seq_lengths = extract_esm_embeddings_full(sequences, tokenizer, model)

    # 4. Save to HDF5
    print(f"Saving features to {args.output_h5}...")
    with h5py.File(args.output_h5, 'w') as hf:
        # Save Features
        hf.create_dataset('esm_embeddings', data=esm_matrix)          # [N, hidden_dim] mean-pooled
        hf.create_dataset('esm_full', data=esm_full)                  # [N, max_len, hidden_dim] full
        hf.create_dataset('seq_lengths', data=seq_lengths)            # [N] actual lengths
        hf.create_dataset('qsar_features', data=qsar_matrix)
        
        # Save Labels
        hf.create_dataset('log_mic', data=log_mics)
        
        # Metadata
        hf.attrs['n_samples'] = len(sequences)
        hf.attrs['esm_dim'] = esm_matrix.shape[1]
        hf.attrs['qsar_dim'] = qsar_matrix.shape[1]
        hf.attrs['max_seq_len'] = esm_full.shape[1]
        
    print("Done! Data ready for training.")

if __name__ == "__main__":
    main()
