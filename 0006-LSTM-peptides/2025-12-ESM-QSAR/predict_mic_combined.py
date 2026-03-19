#!/usr/bin/env python3
"""
predict_mic_ensemble.py
=======================
Ensemble inference script combining:
  - MLP (ESM+QSAR)   weight = 0.6
  - BiLSTM (+QSAR)   weight = 0.4

Predictions are combined in log-MIC space before back-transforming:
    log_MIC_ensemble = 0.6 * log_MIC_mlp + 0.4 * log_MIC_bilstm
    MIC_ensemble     = 10 ** log_MIC_ensemble

The output CSV contains individual and ensemble predictions for every peptide.

USAGE
-----
python predict_mic_ensemble.py \
    --h5_file       ecoli_featurised.h5 \
    --mlp_ckpt      model_mlp_both_40.pt \
    --bilstm_ckpt   model_bilstm_pure.pt \
    --out_csv       predictions/ensemble_predictions.csv \
    [--batch_size 128] \
    [--device cpu]
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch

from train_mic import MLP, BiLSTM_QSAR
from predict_mic import (
    build_bilstm_pure,
    load_bilstm_pure_features,
    load_h5_data,
    load_bilstm_qsar_features,
    dimensions,
    load_mlp_features,
    build_mlp,
    build_bilstm_qsar,
    predict_bilstm_pure,
    predict_mlp,
    predict_bilstm_qsar,
    load_checkpoint,
)

# ── Ensemble weights ──────────────────────────────────────────────────────────

MLP_WEIGHT    = 0.6
BILSTM_WEIGHT = 0.4
assert abs(MLP_WEIGHT + BILSTM_WEIGHT - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Ensemble MIC predictor: MLP(ESM+QSAR) x0.6 + BiLSTM(+QSAR) x0.4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5_file",     required=True,
                   help="Input HDF5 file with ESM embeddings, QSAR features, sequences.")
    p.add_argument("--mlp_ckpt",    required=True,
                   help="Path to MLP (ESM+QSAR) checkpoint (.pt).")
    p.add_argument("--bilstm_ckpt", required=True,
                   help="Path to BiLSTM (+QSAR) checkpoint (.pt).")
    p.add_argument("--out_csv",     required=True,
                   help="Output CSV path for predictions.")
    p.add_argument("--batch_size",  type=int, default=128,
                   help="Inference batch size.")
    p.add_argument("--device",      default=None,
                   help='Device override: "cpu" or "cuda". Default: auto-detect.')
    return p.parse_args()

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("ENSEMBLE MIC PREDICTION PIPELINE")
    print(f"  MLP (ESM+QSAR) weight : {MLP_WEIGHT}")
    print(f"  BiLSTM weight : {BILSTM_WEIGHT}")
    print(f"  Device                : {device}")
    print(f"  Batch size            : {args.batch_size}")
    print(SEP)

    # ------------------------------------------------------------------
    # Step 1: Load shared data from H5
    # ------------------------------------------------------------------
    print("\n[1] Loading H5 data...")
    esm, qsar, sequences = load_h5_data(args.h5_file)
    esm_dim, qsar_dim = dimensions(esm, qsar)

    # ------------------------------------------------------------------
    # Step 2: MLP (ESM+QSAR) predictions
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("[2] MLP (ESM+QSAR) - loading checkpoint & predicting")
    print(SEP)

    mlp_ckpt = load_checkpoint(args.mlp_ckpt, device)

    # Validate the checkpoint was trained with feature_type="both"
    mlp_ft = mlp_ckpt["args"].get("feature_type", "both")
    if mlp_ft != "both":
        raise ValueError(
            f"MLP checkpoint has feature_type='{mlp_ft}', "
            f"expected 'both' (ESM+QSAR). Please supply the correct checkpoint."
        )

    mlp_model   = build_mlp(mlp_ckpt, esm_dim, qsar_dim, device)
    X_mlp       = load_mlp_features(esm, qsar, mlp_ckpt)
    log_mic_mlp = predict_mlp(mlp_model, X_mlp, device, args.batch_size)

    print(
        f"\n  MLP log_MIC  "
        f"mean={log_mic_mlp.mean():.3f}  "
        f"std={log_mic_mlp.std():.3f}  "
        f"range=[{log_mic_mlp.min():.3f}, {log_mic_mlp.max():.3f}]"
    )

    # Free GPU memory before loading next model
    del mlp_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: BiLSTM predictions
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("[3] BiLSTM - loading checkpoint & predicting")
    print(SEP)

    bilstm_ckpt = load_checkpoint(args.bilstm_ckpt, device)

    esm_full, lengths = load_bilstm_pure_features(args.h5_file)

    esm_full_dim = int(esm_full.shape[2])   
    bilstm_model = build_bilstm_pure(bilstm_ckpt, esm_full_dim, device)
    log_mic_bilstm = predict_bilstm_pure(bilstm_model, esm_full, lengths, device, args.batch_size)

    print(
        f"\n  BiLSTM log_MIC  "
        f"mean={log_mic_bilstm.mean():.3f}  "
        f"std={log_mic_bilstm.std():.3f}  "
        f"range=[{log_mic_bilstm.min():.3f}, {log_mic_bilstm.max():.3f}]"
    )

    del bilstm_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Weighted ensemble combination (in log-MIC space)
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("[4] Computing Ensemble Predictions")
    print(SEP)
    print(f"  log_MIC_ensemble = {MLP_WEIGHT} x log_MIC_mlp + {BILSTM_WEIGHT} x log_MIC_bilstm")

    log_mic_ensemble = MLP_WEIGHT * log_mic_mlp + BILSTM_WEIGHT * log_mic_bilstm

    print(
        f"  Ensemble log_MIC  "
        f"mean={log_mic_ensemble.mean():.3f}  "
        f"std={log_mic_ensemble.std():.3f}  "
        f"range=[{log_mic_ensemble.min():.3f}, {log_mic_ensemble.max():.3f}]"
    )

    # ------------------------------------------------------------------
    # Step 5: Back-transform to MIC (ug/mL) and save CSV
    # ------------------------------------------------------------------
    mic_mlp      = 10 ** log_mic_mlp
    mic_bilstm   = 10 ** log_mic_bilstm
    mic_ensemble = 10 ** log_mic_ensemble

    seq_list = [
        s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
        for s in sequences
    ]

    df = pd.DataFrame({
        "Sequence":         seq_list,
        # Individual model outputs
        "log_MIC_mlp":      log_mic_mlp.astype(float),
        "MIC_mlp":          mic_mlp.astype(float),
        "log_MIC_bilstm":   log_mic_bilstm.astype(float),
        "MIC_bilstm":       mic_bilstm.astype(float),
        # Ensemble output
        "log_MIC_ensemble": log_mic_ensemble.astype(float),
        "MIC_ensemble":     mic_ensemble.astype(float),
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n  Saved {len(df)} predictions --> {args.out_csv}")

    # ------------------------------------------------------------------
    # Step 6: Print first 10 predictions
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("FIRST 10 ENSEMBLE PREDICTIONS")
    header = (
        f"{'No':>3}  {'Sequence':<30}  "
        f"{'log_MIC_mlp':>11}  {'log_MIC_bilstm':>14}  "
        f"{'log_MIC_ens':>11}  {'MIC_ens(ug/mL)':>14}"
    )
    print(header)
    print("-" * len(header))
    for i in range(min(10, len(seq_list))):
        print(
            f"{i+1:>3}. {seq_list[i]:<30}  "
            f"{log_mic_mlp[i]:>11.3f}  "
            f"{log_mic_bilstm[i]:>14.3f}  "
            f"{log_mic_ensemble[i]:>11.3f}  "
            f"{mic_ensemble[i]:>14.3f}"
        )
    
    print(f"\n{SEP}")
    print("Top 10 ENSEMBLE PREDICTIONS")
    print(header)
    print("-" * len(header))
    top_indices = np.argsort(log_mic_ensemble)[:10]
    for i in top_indices:
        print(
            f"{i+1:>3}. {seq_list[i]:<30}  "
            f"{log_mic_mlp[i]:>11.3f}  "
            f"{log_mic_bilstm[i]:>14.3f}  "
            f"{log_mic_ensemble[i]:>11.3f}  "
            f"{mic_ensemble[i]:>14.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()