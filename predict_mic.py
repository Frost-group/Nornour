import pandas as pd
import numpy as np 

import torch 
import h5py 
import os
import argparse
from tqdm import tqdm
from typing import Tuple 

from train_mic import MLP, BiLSTM_QSAR

"""
predict_mic.py

Inference script for models:
- MLP (feature_type: "esm" | "qsar" | "both")
- BiLSTM (uses esm_full + qsar + lengths)
- BiLSTM_PURE (uses esm_full + lengths only)

- MIC in (µg/mL)

"""
#1. Load sequences

def load_h5_data(h5file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\n" + "="*60)
    print("STEP 1: Loading H5 Data")
    print("="*60)

    with tqdm(total=3, desc="Loading h5 file", unit ="dataset") as pbar:
        with h5py.File(h5file, "r") as f:
            if "esm_embeddings" not in f: 
                raise KeyError('H5 file missing key "esm_embeddings" required for inference.')
            if "qsar_features" not in f:
                raise KeyError('H5 file missing key "qsar_features" required for inference.')
            if "sequences" not in f:
                raise KeyError('H5 file missing key "sequences" required for output.')

            pbar.set_description("Loading ESM embeddings")
            esm = f["esm_embeddings"][:]
            pbar.update(1)

            pbar.set_description("Loading QSAR features")
            qsar = f["qsar_features"][:]
            pbar.update(1)

            pbar.set_description("Loading sequences")
            sequences = f["sequences"][:]
            pbar.update(1)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f" - ESM embeddings: {esm.shape}")
    print(f" - QSAR features: {qsar.shape}")

    return esm, qsar, sequences

def dimensions(esm, qsar):
    esm_dim = int(esm.shape[1])
    qsar_dim = int(qsar.shape[1])

    return esm_dim, qsar_dim

#2. Load checkpoint
def load_checkpoint(ckpt_path: str, device: str | torch.device) -> dict:
    with tqdm(total=1, desc="Loading checkpoint", unit="file") as pbar:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pbar.update(1)
    if "args" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError('Checkpoint must contain "args" and "model_state_dict".')
    
    print(f" Checkpoint loaded from: {ckpt_path}")
    if "best_val_loss" in ckpt:
        print(f"  - Best validation loss: {ckpt['best_val_loss']:.4f}")

    return ckpt

#3. Build Models 
def build_mlp(ckpt:dict,
    esm_dim: int,
    qsar_dim: int,
    device: str | torch.device = "cpu",
    ):

    print("\n" + "="*60)
    print("STEP 3: Building MLP Model")
    print("="*60)

    args = ckpt["args"]

    feature_type = args["feature_type"]
    if feature_type == "esm":
        input_dim = esm_dim
    elif feature_type == "qsar":
        input_dim = qsar_dim
    elif feature_type == "both":
        input_dim = esm_dim + qsar_dim
    else:
        raise ValueError(f"Unknown feature_type in checkpoint: {feature_type}")
    
    with tqdm(total=3, desc="Building model", unit="step") as pbar:
            pbar.set_description("Creating architecture")
            model = MLP(
                input_dim=input_dim, 
                hidden_dims=args["hidden_dims"],
                dropout=args["dropout"], 
                activation=args["activation"]
            ).to(device)
            pbar.update(1)
            
            pbar.set_description("Loading weights")
            model.load_state_dict(ckpt["model_state_dict"])
            pbar.update(1)
            
            pbar.set_description("Setting eval mode")
            model.eval()
            pbar.update(1)
        
    n_params = sum(p.numel() for p in model.parameters())
    print(f" MLP model built")
    print(f"  - Feature type: {feature_type}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden layers: {args['hidden_dims']}")
    print(f"  - Total parameters: {n_params:,}")

    return model

def build_bilstm_qsar(ckpt: dict, esm_full_dim: int, qsar_dim: int, device: str | torch.device) -> BiLSTM_QSAR:
    args = ckpt["args"]
    required = ["lstm_hidden", "lstm_layers", "mlp_hidden", "dropout", "activation"]
    missing = [k for k in required if k not in args]
    if missing:
        raise KeyError(f"Checkpoint args missing keys required for BiLSTM_QSAR: {missing}")

    print("\n" + "="*60)
    print("STEP 3: Building BiLSTM+QSAR Model")
    print("="*60)

    with tqdm(total=3, desc="Building model", unit="step") as pbar:
        pbar.set_description("Creating architecture")
        model = BiLSTM_QSAR(
            esm_dim=esm_full_dim,
            qsar_dim=qsar_dim,
            lstm_hidden=int(args["lstm_hidden"]),
            lstm_layers=int(args["lstm_layers"]),
            mlp_hidden=args["mlp_hidden"],
            dropout=float(args["dropout"]),
            activation=args["activation"]
        ).to(device)
        pbar.update(1)
        
        pbar.set_description("Loading weights")
        model.load_state_dict(ckpt["model_state_dict"])
        pbar.update(1)
        
        pbar.set_description("Setting eval mode")
        model.eval()
        pbar.update(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f" BiLSTM+QSAR model built")
    print(f"  - ESM dimension: {esm_full_dim}")
    print(f"  - QSAR dimension: {qsar_dim}")
    print(f"  - LSTM hidden: {args['lstm_hidden']} × {args['lstm_layers']} layers")
    print(f"  - MLP head: {args['mlp_hidden']}")
    print(f"  - Total parameters: {n_params:,}")

    return model


def build_bilstm_pure(ckpt: dict, esm_full_dim:int, device: str | torch.device) -> BiLSTM_QSAR:
    args = ckpt["args"]
    required = ["lstm_hidden", "lstm_layers", "mlp_hidden", "dropout", "activation"]
    missing = [k for k in required if k not in args]
    if missing:
        raise KeyError(f"Checkpoint args missing keys required for BiLSTM_PURE: {missing}")
        

    print("\n" + "="*60)
    print("STEP 3: Building Pure BiLSTM Model (No QSAR)")
    print("="*60)

    with tqdm(total=3, desc="Building model", unit="step") as pbar:
        pbar.set_description("Creating architecture")
        model = BiLSTM_QSAR(
            esm_dim=esm_full_dim,
            qsar_dim=0,  # CRITICAL: Must be 0 for pure BiLSTM
            lstm_hidden=int(args["lstm_hidden"]),
            lstm_layers=int(args["lstm_layers"]),
            mlp_hidden=args["mlp_hidden"],
            dropout=float(args["dropout"]),
            activation=args["activation"]
        ).to(device)
        pbar.update(1)
        
        pbar.set_description("Loading weights")
        model.load_state_dict(ckpt["model_state_dict"])
        pbar.update(1)
        
        pbar.set_description("Setting eval mode")
        model.eval()
        pbar.update(1)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f" Pure BiLSTM model built")
    print(f"  - ESM dimension: {esm_full_dim}")
    print(f"  - QSAR dimension: 0 (pure BiLSTM)")
    print(f"  - LSTM hidden: {args['lstm_hidden']} × {args['lstm_layers']} layers")
    print(f"  - MLP head: {args['mlp_hidden']}")
    print(f"  - Total parameters: {n_params:,}")

    return model

#4. Load Features
def load_mlp_features(h5file: str, ckpt: dict) -> np.ndarray:
    args = ckpt["args"]
    ftype = args["feature_type"]

    with tqdm(total=3, desc="Loading features", unit="step") as pbar:
        # Load from H5
        pbar.set_description("Reading H5 file")
        with h5py.File(h5file, "r") as f:
            if ftype in ("esm", "both") and "esm_embeddings" not in f:
                raise KeyError('H5 missing "esm_embeddings" required for MLP (esm/both).')
            if ftype in ("qsar", "both") and "qsar_features" not in f:
                raise KeyError('H5 missing "qsar_features" required for MLP (qsar/both).')

            esm = f["esm_embeddings"][:] if ftype in ("esm", "both") else None
            qsar = f["qsar_features"][:] if ftype in ("qsar", "both") else None
        pbar.update(1)

    print("\nNormalising Features...")    
    #Normalize features
    if qsar is not None:
        qsar = (qsar - ckpt["qsar_scaler_mean"]) / ckpt["qsar_scaler_scale"]

    if esm is not None:
        esm = (esm-ckpt["esm_scaler_mean"]) / ckpt["esm_scaler_scale"]

    if ftype == "esm":
        X = esm
    elif ftype == "qsar":
        X = qsar
    else:
        X = np.hstack([esm, qsar])

    print(f" Features prepared: {X.shape}")
    return X

def load_bilstm_common(h5file: str) -> Tuple[np.ndarray, np.ndarray]:
    with tqdm(total=2, desc="Loading BiLSTM data", unit="dataset") as pbar:
        with h5py.File(h5file, "r") as f:
            if "esm_full" not in f:
                raise KeyError('H5 missing "esm_full" required for BiLSTM inference.')
            if "seq_lengths" not in f:
                raise KeyError('H5 missing "seq_lengths" required for BiLSTM inference.')

            pbar.set_description("Loading ESM full embeddings")
            esm_full = f["esm_full"][:]
            pbar.update(1)

            pbar.set_description("Loading seqeunce lengths")
            lengths = f["seq_lengths"][:]
            pbar.update(1)

    if esm_full.ndim != 3:
        raise ValueError(f'Expected "esm_full" to have shape (N, L, D), got {esm_full.shape}')

    print(f"✓ BiLSTM data loaded")
    print(f"  - ESM full shape: {esm_full.shape}")
    print(f"  - Sequence lengths: {lengths.shape}")

    return esm_full.astype(np.float32), lengths.astype(np.int64)

def load_bilstm_qsar_features(h5file: str, ckpt: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    esm_full, lengths = load_bilstm_common(h5file)
    with tqdm(total=2, desc="Loading QSAR data", unit="step") as pbar:
        with h5py.File(h5file, "r") as f:
            if "qsar_features" not in f:
                raise KeyError('H5 missing "qsar_features" required for BiLSTM_QSAR.')
            qsar = f["qsar_features"][:].astype(np.float32)
        pbar.update(1)

        # BiLSTM training standardises QSAR; apply if scaler saved
        qsar = qsar = (qsar - ckpt["qsar_scaler_mean"]) / ckpt["qsar_scaler_scale"]
    
    print(f" QSAR features normalized: {qsar.shape}")
    return esm_full, qsar, lengths

def load_bilstm_pure_features(h5file: str) -> Tuple[np.ndarray, np.ndarray]:
    return load_bilstm_common(h5file)


#5. Prediction using saved model
@torch.no_grad()
def predict_mlp(model: MLP, X: np.ndarray, device: str | torch.device, batch_size: int) -> np.ndarray:
    print("\n" + "="*60)
    print("STEP 4: Running Predictions")
    print("="*60)

    N = X.shape[0]
    preds = np.zeros((N,), dtype=np.float32)

    with tqdm(total=N, desc="Predicting MIC", unit="peptide") as pbar:
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            x_t = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            out = model(x_t).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            preds[start:end] = out
            
            pbar.update(end - start)

    print(f" Predictions complete for {N} peptides")
    return preds


@torch.no_grad()
def predict_bilstm_qsar(model: BiLSTM_QSAR, esm_full: np.ndarray, qsar: np.ndarray, lengths: np.ndarray,
                        device: str | torch.device, batch_size: int) -> np.ndarray:
    print("\n" + "="*60)
    print("STEP 4: Running Predictions (BiLSTM+QSAR)")
    print("="*60)

    N = esm_full.shape[0]
    preds = np.zeros((N,), dtype=np.float32)

    # Many pack_padded_sequence implementations want lengths on CPU
    lengths_cpu = torch.tensor(lengths, dtype=torch.long, device="cpu")

    with tqdm(total=N, desc="Predicting MIC", unit="peptide") as pbar:
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            esm_t = torch.tensor(esm_full[start:end], dtype=torch.float32, device=device)
            qsar_t = torch.tensor(qsar[start:end], dtype=torch.float32, device=device)
            len_t = lengths_cpu[start:end]

            out = model(esm_t, qsar_t, len_t).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            preds[start:end] = out

            pbar.update(end - start)

    print(f" Predictions complete for {N} peptides")
    return preds


@torch.no_grad()
def predict_bilstm_pure(model: BiLSTM_QSAR, esm_full: np.ndarray, lengths: np.ndarray,
                        device: str | torch.device, batch_size: int) -> np.ndarray:
    N = esm_full.shape[0]
    preds = np.zeros((N,), dtype=np.float32)

    lengths_cpu = torch.tensor(lengths, dtype=torch.long, device="cpu")
    with tqdm(total=N, desc="Predicting MIC", unit="peptide") as pbar:
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            esm_t = torch.tensor(esm_full[start:end], dtype=torch.float32, device=device)
            len_t = lengths_cpu[start:end]

            qsar_t = torch.zeros((end - start, 0), dtype=torch.float32, device=device)

            out = model(esm_t, qsar_t, len_t).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            preds[start:end] = out
            pbar.update(end - start)

    print(f" Predictions complete for {N} peptides")
    return preds


def main(): 
    p = argparse.ArgumentParser()
    p.add_argument("--h5_file", required=True, help="Input H5 with esm_full/qsar_features/seq_lengths/sequences.")
    p.add_argument("--ckpt_path", required=True, help="Path to saved model checkpoint (.pt).")
    p.add_argument("--out_csv", required=True, help="Output CSV path.")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for inference.")
    p.add_argument("--device", default=None, help='Device override ("cpu" or "cuda"). Default auto.')
    p.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'bilstm'],
                        help="Model type: 'mlp' (mean-pooled ESM), 'bilstm' (full ESM sequence)")
    p.add_argument('--pure_bilstm', action='store_true',
                        help="Pure BiLSTM without QSAR features")
    
    args = p.parse_args()
    
    print("\n" + "="*60)
    print("PEPTIDE MIC PREDICTION PIPELINE")
    print("="*60)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    print(f"Model Type: {args.model}" + (" (pure)" if args.pure_bilstm else ""))
    print(f"Batch size: {args.batch_size}")
    
    # Load checkpoint
    ckpt = load_checkpoint(args.ckpt_path, device)

    # Load esm, qsar, sequences (globally required) 
    esm, qsar, sequences = load_h5_data(args.h5_file)

    # Load dimensions
    esm_dim, qsar_dim = dimensions(esm, qsar)
    
    # Build model and get predictions based on model type
    if args.model == 'bilstm':
        with h5py.File(args.h5_file, "r") as f:
            esm_full = f["esm_full"][:]
            if esm_full.ndim != 3:
                raise ValueError(f'Expected "esm_full" to have shape (N, L, D), got {esm_full.shape}')
        esm_full_dim = int(esm_full.shape[2])

        if args.pure_bilstm:
            print("Building Pure BiLSTM model (no QSAR)...")
            model = build_bilstm_pure(ckpt, esm_full_dim, device)
            esm_full, lengths = load_bilstm_pure_features(args.h5_file)
            log_mic = predict_bilstm_pure(model, esm_full, lengths, device, args.batch_size)
        else:
            print("Building BiLSTM+QSAR model...")
            model = build_bilstm_qsar(ckpt, esm_full_dim, qsar_dim, device)
            esm_full, qsar, lengths = load_bilstm_qsar_features(args.h5_file, ckpt)
            log_mic = predict_bilstm_qsar(model, esm_full, qsar, lengths, device, args.batch_size)
    
    elif args.model == 'mlp':
        print("Building MLP model...")
        model = build_mlp(ckpt, esm_dim, qsar_dim, device)
        X = load_mlp_features(args.h5_file, ckpt)
        print(f"Feature matrix shape: {X.shape}")
        log_mic = predict_mlp(model, X, device, args.batch_size)
    
    # Convert to MIC
    pred_mic = 10 ** log_mic
    
    # Decode sequences
    seq_list = [
        s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
        for s in sequences
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        "sequence": seq_list,
        "log_MIC": log_mic.astype(float),
        "MIC": pred_mic.astype(float),
    })
    
    # Save
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved predictions to: {args.out_csv}")
    
    # Print first 10 predictions
    print("\nFIRST 10 PREDICTIONS:")
    for i in range(min(10, len(sequences))):
        seq = sequences[i].decode() if isinstance(sequences[i], bytes) else str(sequences[i])
        print(f"{i+1:02d}. {seq} → log_MIC = {log_mic[i]:.3f}, MIC = {pred_mic[i]:.3f} µg/mL")
    
    print(f"\nSummary Statistics:")
    print(f"  log_MIC - Mean: {log_mic.mean():.3f}, Std: {log_mic.std():.3f}, Range: [{log_mic.min():.3f}, {log_mic.max():.3f}]")
    print(f"  MIC - Mean: {pred_mic.mean():.3f}, Std: {pred_mic.std():.3f}, Range: [{pred_mic.min():.3f}, {pred_mic.max():.3f}] µg/mL")


if __name__ == "__main__":
    main()


