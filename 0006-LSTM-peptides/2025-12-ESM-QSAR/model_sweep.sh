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

# Datasets: "h5_file:short_name"
DATASETS=(
    "peptides_featurised.h5:full"
    "peptides_coli_featurised.h5:coli"
    "peptides_gramnegative_featurised.h5:gramneg"
)

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
    
    python train_mlp.py $args \
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

for entry in "${DATASETS[@]}"; do
    h5="${entry%%:*}"
    ds="${entry##*:}"
    
    [[ ! -f "$h5" ]] && { echo "Skipping $h5 (not found)"; continue; }
    
    echo "Dataset: $h5 ($ds)"
    
    # -------------------------------------------------------------------------
    # MLP models
    # Input dims: ESM=640, QSAR≈267, Combined≈907
    # Rule of thumb: first layer ≤ 4x compression
    # -------------------------------------------------------------------------
    
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
    
    # --- Architecture variations (ESM+QSAR) ---
    
    # Wider first layer (~2x compression)
    run_experiment "${ds}_mlp_esm_qsar_wide" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 512 256 64 --dropout 0.3 --lr 1e-3
    
    # Deeper pyramid
    run_experiment "${ds}_mlp_esm_qsar_deep" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 32 --dropout 0.3 --lr 1e-3
    
    # Fat-then-thin
    run_experiment "${ds}_mlp_esm_qsar_fat" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 512 128 --dropout 0.3 --lr 1e-3
    
    # Single wide hidden layer
    run_experiment "${ds}_mlp_esm_qsar_shallow" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 --dropout 0.3 --lr 1e-3
    
    # Old narrow baseline (for comparison)
    run_experiment "${ds}_mlp_esm_qsar_narrow" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 64 32 16 --dropout 0.3 --lr 1e-3
    
    # --- Activation function ---
    
    run_experiment "${ds}_mlp_esm_qsar_silu" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.3 --lr 1e-3 --activation silu
    
    run_experiment "${ds}_mlp_esm_qsar_relu" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.3 --lr 1e-3 --activation relu
    
    # --- Regularisation sensitivity ---
    
    run_experiment "${ds}_mlp_esm_qsar_lowdrop" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.1 --lr 1e-3
    
    run_experiment "${ds}_mlp_esm_qsar_highdrop" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.5 --lr 1e-3
    
    # --- Learning rate sensitivity ---
    
    run_experiment "${ds}_mlp_esm_qsar_lowlr" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.3 --lr 1e-4
    
    run_experiment "${ds}_mlp_esm_qsar_highlr" \
        --h5_file "$h5" --model mlp --feature_type both \
        --hidden_dims 256 128 64 --dropout 0.3 --lr 5e-3
    

    # -------------------------------------------------------------------------
    # BiLSTM models - SLOW AS MOLASSES ON MY LITTLE CPU
    # MLP head input: 2×lstm_hidden + QSAR ≈ 256+267 ≈ 523 (with QSAR)
    #                 2×lstm_hidden ≈ 256 (pure)
    # -------------------------------------------------------------------------
    
    # BiLSTM + QSAR (wider head)
    run_experiment "${ds}_esm-bilstm_and_qsar" \
        --h5_file "$h5" --model bilstm \
        --lstm_hidden 128 --lstm_layers 1 --mlp_hidden 128 64 \
        --dropout 0.3 --lr 1e-3
    
    # Pure BiLSTM (no QSAR)
    run_experiment "${ds}_esm-bilstm_pure" \
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

