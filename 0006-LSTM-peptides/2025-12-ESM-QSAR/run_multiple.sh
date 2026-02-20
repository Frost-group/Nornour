#!/usr/bin/env bash
# =============================================================================
# run_multiple.sh
# Runs the top hyperparameter configurations from the hyperparameter sweeps,
# saving plots and stdout
# =============================================================================

# REMINDER! 
# Remove --pure_bilstm if QSAR required


# To run:
# chmod +x run_multiple.sh
# ./run_multiple.sh

set -euo pipefail

# --- Fixed hyperparameters ---
H5_FILE="gnegative_featurised.h5"
EPOCHS=40
BATCH_SIZE=64
SEED=42
OUT_DIR="bilstm_seq_esm_qsar_gn6-10"

mkdir -p "$OUT_DIR"

SUMMARY_ALL="${OUT_DIR}/summary.csv"

if [[ ! -f "$SUMMARY_ALL" ]]; then
  echo "rank,mlp_architecture,layers,hidden_layers,dropout,lr,embed_dim,epochs,ef10,spearman,score" > "$SUMMARY_ALL"
fi

# Use non-interactive backend (prevents plt.show() from blocking)
export MPLBACKEND=Agg

# --- Configuration table (Rank | mlp_hidden | lstm_layers | hidden(lstm) | dropout | lr | embed_dim) ---
# Columns: rank  mlp_hidden   lstm_layers  lstm_hidden  dropout   lr       embed_dim
CONFIGS=(
    "6 128-64 1 256 0.3 0.003 128"
    "7 128-64 1 256 0.3 0.0003 128"
    "8 64-32 1 128 0.3 0.003 32"
    "9 128-64 1 64 0.5 0.003 64"
    "10 64-32 1 64 0.4 0.003 64"
)

echo "=============================================="
echo " $OUT_DIR — $H5_FILE"
echo " Output directory : $OUT_DIR"
echo " Epochs: $EPOCHS | Batch: $BATCH_SIZE | Seed: $SEED"
echo "=============================================="
echo ""

for cfg in "${CONFIGS[@]}"; do
    # Parse config fields
    rank=$(    echo "$cfg" | awk '{print $1}')
    mh_raw=$(  echo "$cfg" | awk '{print $2}')   # e.g. "64-32"
    ly=$(      echo "$cfg" | awk '{print $3}')
    hd=$(      echo "$cfg" | awk '{print $4}')
    dr=$(      echo "$cfg" | awk '{print $5}')
    lr=$(      echo "$cfg" | awk '{print $6}')
    eb=$(      echo "$cfg" | awk '{print $7}')

    # Convert "64-32" → "64 32" for --mlp_hidden (space-separated)
    mh="${mh_raw//-/ }"

    # Output file names
    run_tag="rank${rank}_mh${mh_raw}_ly${ly}_hd${hd}_dr${dr}_lr${lr}_eb${eb}"
    plot_file="${OUT_DIR}/${run_tag}.png"
    stdout_file="${OUT_DIR}/${run_tag}.txt"

    # echo "----------------------------------------------"
    # echo " Rank ${rank}: mlp_hidden=[${mh}] lstm_layers=${ly} lstm_hidden=${hd}"
    # echo "          dropout=${dr} lr=${lr} embed_dim=${eb}"
    # echo "----------------------------------------------"

    echo -n "$rank ... "
                            
    # Skip if already completed
    if [[ -f "$stdout_file" && -f "$plot_file" ]]; then
            echo "(skipped)"
            continue
    fi
    python train_mic.py \
        --h5_file "$H5_FILE" \
        --model bilstm_seq_esm \
        --lstm_layers "$ly" \
        --lstm_hidden "$hd" \
        --embed_dim "$eb" \
        --mlp_hidden $mh \
        --dropout "$dr" \
        --lr "$lr" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --plot_path "$plot_file" \
        > "$stdout_file" 2>&1

    ef=$(grep "EF@10%" "$stdout_file" | awk '{print $2}')
    rho=$(grep "Spearman ρ:" "$stdout_file" | tail -1 | awk '{print $3}')
    score=$(awk -v ef="$ef" -v rho="$rho" 'BEGIN { printf "%.4f", ef + 2*rho }')
    
    echo "${rank},\"${mh_raw}\",${ly},${hd},${dr},${lr},${eb},${EPOCHS},${ef},${rho},${score}" >> "$SUMMARY_ALL"

    echo "EF=$ef ρ=$rho Score=$score"
done

echo "=============================================="
echo " All runs complete. Results saved to: $OUT_DIR"
echo ""
echo " Summary ..."
echo ""

# Sort summary.csv in place by score (descending)
{ head -1 "$SUMMARY_ALL"; tail -n +2 "$SUMMARY_ALL" | sort -t',' -k11 -rn; } > "${SUMMARY_ALL}.tmp" \
    && mv "${SUMMARY_ALL}.tmp" "$SUMMARY_ALL"

echo "  Rank | Run                                                        | EF@10% | Spearman |  Score"
echo "  -----|------------------------------------------------------------+--------+----------+--------"
echo "=============================================="

while IFS=',' read -r rank mla ly hd dr lr eb ep ef rho score; do
    tag="rank${rank}_mh${mla}_ly${ly}_hd${hd}_dr${dr}_lr${lr}_eb${eb}"
    printf "  %-4s | %-58s | %-6s | %-8s | %s\n" \
        "$rank" "$tag" "$ef" "$rho" "$score"
done < <(tail -n +2 "$SUMMARY_ALL")

echo ""
echo "  Sorted summary saved to: $SUMMARY_ALL"
echo "=============================================="

echo ""
echo " Stitching panel figure..."

python combine_images.py

echo "=============================================="

