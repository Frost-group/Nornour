#!/bin/bash
# hyperparameter_sweep.sh - Grid search over MLP hyperparameters
#
# Runs train_mic.py with various configurations, saving:
#   - results/<name>.stdout  (terminal output with metrics)
#   - results/<name>.png     (training plots)
#
# Usage:
#   chmod +x hyperparameter_sweep.sh
#   ./hyperparameter_sweep.sh

# Note: 
# remove --pure_bilstm if doing bilstm + QSAR 

set -e

RESULTS_DIR="BILSTM_seq_esm_sweep"
mkdir -p "$RESULTS_DIR"

SUMMARY_ALL="${RESULTS_DIR}/summary.csv"

if [[ ! -f "$SUMMARY_ALL" ]]; then
  echo "mlp_architecture,layers,hidden_layers,dropout,lr,embed_dim,epochs,ef10,spearman,score" > "$SUMMARY_ALL"
fi

# Use non-interactive backend (prevents plt.show() from blocking)
export MPLBACKEND=Agg

# Fixed parameters
EPOCHS=40
BATCH_SIZE=64
SEED=42

# Hyperparameter grid
LSTM_HIDDEN=(256 128 64)
LSTM_LAYERS=(1)
MLP_HIDDEN=("128 64" "64 32")
DROPOUTS=(0.3 0.4 0.5)
LEARNING_RATES=(1e-3 3e-3 3e-4 1e-4)
EMBEDDING_DIMS=(32 64 128)

# Count total runs
total=0
for ly in "${LSTM_LAYERS[@]}"; do
    for hd in "${LSTM_HIDDEN[@]}"; do
        for mh in "${MLP_HIDDEN[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for dr in "${DROPOUTS[@]}"; do
                    for eb in "${EMBEDDING_DIMS[@]}"; do
                        ((total++))
                    done
                done
            done
        done
    done
done

echo "=== Hyperparameter Sweep ==="

echo "Total configurations: $total"
echo "Results directory: $RESULTS_DIR/"
echo ""

run=0
for ly in "${LSTM_LAYERS[@]}"; do
    for hd in "${LSTM_HIDDEN[@]}"; do
        for mh in "${MLP_HIDDEN[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for dr in "${DROPOUTS[@]}"; do
                    for eb in "${EMBEDDING_DIMS[@]}"; do
                        ((run++))
                            
                        # Create descriptive filename from hyperparameters
                        mh_str=$(echo "$mh" | tr ' ' '-')
                        name="BILSTM_seq_esm_layers-${ly}_hd-${hd}_mlp-${mh_str}_lr-${lr}_dr-${dr}_eb-${eb}"
                        
                        stdout_file="${RESULTS_DIR}/${name}.stdout"
                        plot_file="${RESULTS_DIR}/${name}.png"
                            
                        echo -n "[$run/$total] $name ... "
                            
                            # Skip if already completed
                        if [[ -f "$stdout_file" && -f "$plot_file" ]]; then
                                echo "(skipped)"
                                continue
                        fi
                            
                        # Run training, capture stdout, generate plot
                        python train_mic.py \
                            --h5_file "gnegative_featurised.h5" \
                            --model bilstm_seq_esm \
                            --pure_bilstm \
                            --lstm_layers "$ly" \
                            --lstm_hidden "$hd" \
                            --embed_dim "$eb"\
                            --mlp_hidden $mh \
                            --dropout "$dr" \
                            --lr "$lr" \
                            --epochs $EPOCHS \
                            --batch_size $BATCH_SIZE \
                            --seed $SEED \
                            --plot_path "$plot_file" \
                            > "$stdout_file" 2>&1
                                
                        # Extract key metrics
                        ef=$(grep "EF@10%" "$stdout_file" | awk '{print $2}')
                        rho=$(grep "Spearman ρ:" "$stdout_file" | tail -1 | awk '{print $3}')
                        score=$(awk -v ef="$ef" -v rho="$rho" 'BEGIN { printf "%.4f", ef + 2*rho }')
                            
                        echo "\"${mh_str}\",${ly},${hd},${dr},${lr},${eb},${EPOCHS},${ef},${rho},${score}" >> "$SUMMARY_ALL"

                        echo "EF=$ef ρ=$rho Score=$score"
                    done
                done
            done
        done
    done
done

echo ""
echo "=== Sweep Complete ==="
echo ""

# Build summary table
echo "Results Summary (sorted by score):"
echo "===================================="
printf "%-50s %8s %8s %8s\n" "Configuration" "EF@10%" "Spearman" "Score"
echo "------------------------------------------------------------------------------"

for f in "$RESULTS_DIR"/*.stdout; do
    name=$(basename "$f" .stdout)
    ef=$(grep "EF@10%" "$f" 2>/dev/null | awk '{print $2}' || echo "N/A")
    rho=$(grep "Spearman ρ:" "$f" 2>/dev/null | tail -1 | awk '{print $3}' || echo "N/A")
    score=$(awk -v ef="$ef" -v rho="$rho" 'BEGIN { printf "%.4f", ef + 2*rho }')
    printf "%-50s %8s %8s %8s\n" "$name" "$ef" "$rho" "$score"
done | sort -t' ' -k4,4 -rn | head -20

echo ""
echo "All results in: $RESULTS_DIR/"



