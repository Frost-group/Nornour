#!/usr/bin/env bash
set -euo pipefail

H5="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/ecoli_val.h5"
OUTDIR="model_comparisons/ecoli_val_pred"
BATCH_SIZE=64

mkdir -p "$OUTDIR"

# Helper: only run if checkpoint exists
run_if_ckpt() {
  local name="$1"
  local model_flag="$2"     # mlp | bilstm | bilstm_seq
  local ckpt="$3"
  local outcsv="$4"
  local extra_flags="${5:-}"

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] $name (missing ckpt: $ckpt)"
    return 0
  fi

  echo "[RUN]  $name"
  python /home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/predict_mic.py \
    --h5_file "$H5" \
    --ckpt_path "$ckpt" \
    --out_csv "$outcsv" \
    --batch_size "$BATCH_SIZE" \
    --model "$model_flag" \
    $extra_flags
}

# -----------------------------
# EDIT THESE CHECKPOINT PATHS
# -----------------------------
CKPT_MLP_BOTH="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_mlp_both_40.pt"
CKPT_MLP_ESM="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_mlp_esm45epochs.pt"
CKPT_BILSTM_QSAR="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_bilstm_qsar.pt"
CKPT_BILSTM_PURE="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_bilstm_pure.pt"
CKPT_BILSTM_SEQ="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_bilstm_seq.pt"
CKPT_BILSTM_SEQ_QSAR="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_bilstm_seq_qsar.pt"
CKPT_BILSTM_SEQ_ESM="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_bilstm_seq_esm.pt"
CKPT_BILSTM_SEQ_ESM_QSAR="/home/amp/Nornour/0006-LSTM-peptides/2025-12-ESM-QSAR/model_bilstm_seq_esm_qsar.pt"

# -----------------------------
# RUNS (outputs are 20k only because H5 contains 20k)
# -----------------------------
run_if_ckpt "MLP (QSAR+ESM)"   "mlp"       "$CKPT_MLP_BOTH"      "$OUTDIR/mlp_both_rand20000.csv"
run_if_ckpt "MLP (ESM Only)"   "mlp"       "$CKPT_MLP_ESM"       "$OUTDIR/mlp_esm_rand20000.csv"

run_if_ckpt "BiLSTM (+QSAR)"   "bilstm"    "$CKPT_BILSTM_QSAR"   "$OUTDIR/bilstm_qsar_rand20000.csv"
run_if_ckpt "BiLSTM (pure)"    "bilstm"    "$CKPT_BILSTM_PURE"   "$OUTDIR/bilstm_pure_rand20000.csv"   "--pure_bilstm"

run_if_ckpt "BiLSTM Seq (pure)"        "bilstm_seq" "$CKPT_BILSTM_SEQ"      "$OUTDIR/bilstm_seq_rand20000.csv"       "--pure_bilstm"
run_if_ckpt "BiLSTM Seq (+QSAR)"       "bilstm_seq" "$CKPT_BILSTM_SEQ_QSAR" "$OUTDIR/bilstm_seq_qsar_rand20000.csv"

run_if_ckpt "BiLSTM Seq (+ESM)"        "bilstm_seq_esm" "$CKPT_BILSTM_SEQ_ESM"      "$OUTDIR/bilstm_seq_esm_rand20000.csv"       "--pure_bilstm"
run_if_ckpt "BiLSTM Seq (+ESM+QSAR)"       "bilstm_seq_esm" "$CKPT_BILSTM_SEQ_ESM_QSAR" "$OUTDIR/bilstm_seq_esm_qsar_rand20000.csv"

echo "[DONE] Outputs in: $OUTDIR"
