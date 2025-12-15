#!/bin/bash
# model_sweep.sh - Sweep across datasets, model types, and key hyperparameters
#
# Generates plots for manual inspection rather than exhaustive grid search.
#
# Usage:
#   chmod +x model_sweep.sh
#   ./model_sweep.sh

set -e

RESULTS_DIR="sweep_results"
mkdir -p "$RESULTS_DIR"

# Prevent plt.show() from blocking
export MPLBACKEND=Agg

# Common training params
EPOCHS=40
BATCH_SIZE=64
SEED=42

run_experiment() {
    local name="$1"
    shift
    local args="$@"
    
    local stdout_file="${RESULTS_DIR}/${name}.log"
    local plot_file="${RESULTS_DIR}/${name}.png"
    
    # Skip if already completed
    if [[ -f "$plot_file" ]]; then
        echo "  [skip] $name"
        return
    fi
    
    echo -n "  [run]  $name ... "
    
    python train_mic.py $args \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --plot_path "$plot_file" \
        > "$stdout_file" 2>&1
    
    # Extract key metrics
    rho=$(grep "Spearman ρ:" "$stdout_file" | tail -1 | awk '{print $NF}')
    r2=$(grep "R²:" "$stdout_file" | tail -1 | awk '{print $2}' | tr -d ',')
    echo "R²=$r2 ρ=$rho"
}

echo "=== Model Sweep ==="
echo "Results: $RESULTS_DIR/"
echo ""

for h5 in peptides_gramnegative_featurised.h5 peptides_coli_featurised.h5 #peptides_featurised.h5 peptides_coli_featurised.h5
do
    case "$h5" in
        peptides_featurised.h5) ds="full" ;;
        peptides_coli_featurised.h5) ds="coli" ;;
        peptides_gramnegative_featurised.h5) ds="gramneg" ;;
        *) ds="unknown" ;;
    esac
    
    [[ ! -f "$h5" ]] && { echo "Skipping $h5 (not found)"; continue; }
    
    echo "Dataset: $h5 ($ds)"
    
    # -------------------------------------------------------------------------
    # MLP models
    # Input dims: ESM=640, QSAR≈267, Combined≈907
    # Rule of thumb: first layer ≤ 4x compression
    # -------------------------------------------------------------------------
    
    # Core baselines
    # MLP: ESM+QSAR (new default - ~4x compression)
    run_experiment "${ds}_mlp_esm_qsar" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.3 --lr 1e-3
    
    # MLP: ESM only (input=640)
    run_experiment "${ds}_mlp_esm" \
        --h5_file "$h5" --model mlp --feature_type esm \
        --hidden_dims 256 128 64 --dropout 0.3 --lr 1e-3
    
    # MLP: QSAR only (input≈267)
    run_experiment "${ds}_mlp_qsar" \
        --h5_file "$h5" --model mlp --feature_type qsar \
        --hidden_dims 128 64 --dropout 0.3 --lr 1e-3

    # --- High-performance MLP variants (EF/BEDROC-focused) ---

    # Smaller ESM+QSAR head
    run_experiment "${ds}_mlp_esm_qsar_small" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 128 64 --dropout 0.3 --lr 1e-3

    # Smaller ESM+QSAR head with stronger dropout
    run_experiment "${ds}_mlp_esm_qsar_small_highdrop" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 128 64 --dropout 0.5 --lr 1e-3

    # ESM-only, slightly smaller, higher dropout
    run_experiment "${ds}_mlp_esm_highdrop" \
        --h5_file "$h5" --model mlp --feature_type esm \
        --hidden_dims 256 128 --dropout 0.4 --lr 1e-3

    # ESM-only, higher dropout + lower LR
    run_experiment "${ds}_mlp_esm_highdrop_lowlr" \
        --h5_file "$h5" --model mlp --feature_type esm \
        --hidden_dims 256 128 --dropout 0.4 --lr 5e-4

    # ESM-only, higher dropout + lower LR + stronger weight decay
    run_experiment "${ds}_mlp_esm_highdrop_lowlr_wd" \
        --h5_file "$h5" --model mlp --feature_type esm \
        --hidden_dims 256 128 --dropout 0.4 --lr 5e-4 --weight_decay 1e-3
    

    # -------------------------------------------------------------------------
    # Attention-pooling models (ESM full sequence + QSAR)
    # Much lighter than BiLSTM; good for ~1.5k samples
    # -------------------------------------------------------------------------

    # Best-performing attention configuration (from previous sweep)
    run_experiment "${ds}_attn_esm_qsar_highdrop" \
        --h5_file "$h5" --model attn \
        --mlp_hidden 128 64 --dropout 0.5 --lr 1e-3


    # -------------------------------------------------------------------------
    # BiLSTM models - SLOW AS MOLASSES ON MY LITTLE CPU
    # MLP head input: 2×lstm_hidden + QSAR ≈ 256+267 ≈ 523 (with QSAR)
    #                 2×lstm_hidden ≈ 256 (pure)
    # -------------------------------------------------------------------------
    
    # Baby BiLSTM (small capacity, heavy dropout)
    run_experiment "${ds}_bilstm_and_qsar_small" \
        --h5_file "$h5" --model bilstm \
        --lstm_hidden 16 --lstm_layers 1 --mlp_hidden 128 64 \
        --dropout 0.5 --lr 1e-3
    
    # Best-performing BiLSTM + QSAR configuration (from previous sweep)
    run_experiment "${ds}_bilstm_and_qsar" \
        --h5_file "$h5" --model bilstm \
        --lstm_hidden 128 --lstm_layers 1 --mlp_hidden 128 64 \
        --dropout 0.3 --lr 1e-3

    # Pure BiLSTM (no QSAR)
    run_experiment "${ds}_bilstm_pure" \
        --h5_file "$h5" --model bilstm --pure_bilstm \
        --lstm_hidden 128 --lstm_layers 1 --mlp_hidden 128 64 \
        --dropout 0.3 --lr 1e-3
    
    echo ""
done

# Summary table
echo "=== Summary ==="
echo ""
printf "%-30s %8s %8s\n" "Experiment" "R²" "Spearman"
echo "------------------------------------------------"

for f in "$RESULTS_DIR"/*.log; do
    [[ ! -f "$f" ]] && continue
    name=$(basename "$f" .log)
    r2=$(grep "R²:" "$f" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d ',' || echo "N/A")
    rho=$(grep "Spearman ρ:" "$f" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
    printf "%-30s %8s %8s\n" "$name" "$r2" "$rho"
done | sort

echo ""
echo "Plots saved to: $RESULTS_DIR/*.png"

