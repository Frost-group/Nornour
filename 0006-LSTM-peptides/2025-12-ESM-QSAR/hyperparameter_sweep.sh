#!/bin/bash
# hyperparameter_sweep.sh - Grid search over MLP hyperparameters
#
# Runs train_mlp.py with various configurations, saving:
#   - results/<name>.stdout  (terminal output with metrics)
#   - results/<name>.png     (training plots)
#
# Usage:
#   chmod +x hyperparameter_sweep.sh
#   ./hyperparameter_sweep.sh

set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Use non-interactive backend (prevents plt.show() from blocking)
export MPLBACKEND=Agg

# Fixed parameters
EPOCHS=150
BATCH_SIZE=64
SEED=42

# Hyperparameter grid
FEATURE_TYPES=("both" "esm" "qsar")
HIDDEN_DIMS=("128 64" "64 32" "256 128 64" "64 32 16")
DROPOUTS=(0.2 0.3 0.4)
LEARNING_RATES=(1e-3 5e-4 1e-4)

# Count total runs
total=0
for ft in "${FEATURE_TYPES[@]}"; do
    for hd in "${HIDDEN_DIMS[@]}"; do
        for dr in "${DROPOUTS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                ((total++))
            done
        done
    done
done

echo "=== Hyperparameter Sweep ==="
echo "Total configurations: $total"
echo "Results directory: $RESULTS_DIR/"
echo ""

run=0
for ft in "${FEATURE_TYPES[@]}"; do
    for hd in "${HIDDEN_DIMS[@]}"; do
        for dr in "${DROPOUTS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                ((run++))
                
                # Create descriptive filename from hyperparameters
                hd_str=$(echo "$hd" | tr ' ' '-')
                name="feat-${ft}_hd-${hd_str}_dr-${dr}_lr-${lr}"
                
                stdout_file="${RESULTS_DIR}/${name}.stdout"
                plot_file="${RESULTS_DIR}/${name}.png"
                
                echo -n "[$run/$total] $name ... "
                
                # Skip if already completed
                if [[ -f "$stdout_file" && -f "$plot_file" ]]; then
                    echo "(skipped)"
                    continue
                fi
                
                # Run training, capture stdout, generate plot
                python train_mlp.py \
                    --feature_type "$ft" \
                    --hidden_dims $hd \
                    --dropout "$dr" \
                    --lr "$lr" \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --seed $SEED \
                    --plot_path "$plot_file" \
                    > "$stdout_file" 2>&1
                
                # Extract key metrics
                ef=$(grep "Enrichment Factor" "$stdout_file" | awk '{print $3}')
                rho=$(grep "Spearman ρ:" "$stdout_file" | tail -1 | awk '{print $3}')
                echo "EF=$ef ρ=$rho"
                
            done
        done
    done
done

echo ""
echo "=== Sweep Complete ==="
echo ""

# Build summary table
echo "Results Summary (sorted by EF@10%):"
echo "===================================="
printf "%-50s %8s %8s\n" "Configuration" "EF@10%" "Spearman"
echo "--------------------------------------------------------------------"

for f in "$RESULTS_DIR"/*.stdout; do
    name=$(basename "$f" .stdout)
    ef=$(grep "Enrichment Factor" "$f" 2>/dev/null | awk '{print $3}' || echo "N/A")
    rho=$(grep "Spearman ρ:" "$f" 2>/dev/null | tail -1 | awk '{print $3}' || echo "N/A")
    printf "%-50s %8s %8s\n" "$name" "$ef" "$rho"
done | sort -t' ' -k2 -rn | head -20

echo ""
echo "All results in: $RESULTS_DIR/"
