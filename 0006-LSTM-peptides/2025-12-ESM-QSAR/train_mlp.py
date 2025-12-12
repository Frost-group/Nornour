#!/usr/bin/env python3
"""
train_mlp.py - Train an MLP to predict peptide MIC from ESM + QSAR features.

Uses weighted MSE loss to prioritise accurate prediction of low-MIC (potent) peptides.
Evaluation includes drug-discovery-relevant metrics: Enrichment Factor, Spearman ρ, Precision@k.

Usage:
    python train_mlp.py --h5_file peptides_featurised.h5 --epochs 200 --hidden_dims 256 128
"""

import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION & ARGUMENT PARSING
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Train MLP for MIC prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--h5_file', type=str, default='peptides_featurised.h5',
                        help="Input HDF5 file with features")
    parser.add_argument('--feature_type', type=str, default='both',
                        choices=['esm', 'qsar', 'both'],
                        help="Which features to use: 'esm', 'qsar', or 'both'")
    
    # Architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                        help="Hidden layer dimensions (e.g., --hidden_dims 128 64)")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'gelu', 'silu', 'tanh'],
                        help="Activation function")
    
    # Training
    parser.add_argument('--epochs', type=int, default=150,
                        help="Maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="L2 regularisation (weight decay)")
    
    # Data splits
    parser.add_argument('--val_frac', type=float, default=0.15,
                        help="Validation set fraction")
    parser.add_argument('--test_frac', type=float, default=0.15,
                        help="Test set fraction")
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--save_model', type=str, default=None,
                        help="Path to save best model (e.g., 'best_model.pt')")
    parser.add_argument('--no_plot', action='store_true',
                        help="Disable plotting")
    parser.add_argument('--plot_path', type=str, default='train_mlp_plots.png',
                        help="Path to save training plots")
    
    return parser.parse_args()

# ============================================================================
# DATASET
# ============================================================================

class PeptideDataset(Dataset):
    """
    PyTorch Dataset for peptide features with sample weights.
    
    Returns (features, target, weight) tuples where weight prioritises low-MIC samples.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

# ============================================================================
# MODEL
# ============================================================================

def get_activation(name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
    }
    return activations[name]

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for regression.
    Architecture: Input -> [Linear -> Activation -> Dropout] x N -> Linear -> Output
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2, 
                 activation: str = 'relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ============================================================================
# LOSS FUNCTION
# ============================================================================

def weighted_mse_loss(preds, targets, weights):
    """
    Weighted MSE: samples with higher weight contribute more to loss.
    Low MIC (potent) peptides get higher weights.
    """
    squared_errors = (preds - targets) ** 2
    return (weights * squared_errors).mean()

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, top_frac: float = 0.1):
    """
    Compute drug-discovery-relevant metrics.
    
    Args:
        y_true:   True log10(MIC) values
        y_pred:   Predicted log10(MIC) values
        top_frac: Fraction to consider as "top" for enrichment/precision (default 10%)
    
    Returns:
        Dictionary of metrics
    """
    n = len(y_true)
    k = max(1, int(n * top_frac))
    
    # Standard regression metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    
    # Spearman rank correlation (more relevant for ranking)
    spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    
    # Define "potent" as bottom top_frac% of true MIC (lowest MIC = most potent)
    potent_threshold = np.percentile(y_true, top_frac * 100)
    is_potent = y_true <= potent_threshold
    n_potent = is_potent.sum()
    
    # Top-k by prediction (lowest predicted MIC)
    top_k_pred_idx = np.argsort(y_pred)[:k]
    
    # Enrichment Factor: ratio of potent peptides in top-k vs random expectation
    hits_in_top_k = is_potent[top_k_pred_idx].sum()
    ef = (hits_in_top_k / k) / (n_potent / n) if n_potent > 0 else 0.0
    
    # Precision@k: fraction of top-k predictions that are truly potent
    precision_at_k = hits_in_top_k / k
    
    # Recall@k: fraction of truly potent peptides found in top-k predictions
    recall_at_k = hits_in_top_k / n_potent if n_potent > 0 else 0.0
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Spearman ρ': spearman_rho,
        f'EF@{int(top_frac*100)}%': ef,
        f'Precision@{int(top_frac*100)}%': precision_at_k,
        f'Recall@{int(top_frac*100)}%': recall_at_k,
        'n_potent': n_potent,
        'hits_in_top_k': hits_in_top_k,
    }

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    """Single training epoch with weighted MSE."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for X_batch, y_batch, w_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        w_batch = w_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = weighted_mse_loss(predictions, y_batch, w_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)
        
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model. Returns predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []
    
    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch)
        
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(y_batch.numpy())
        
    return np.vstack(all_preds).flatten(), np.vstack(all_targets).flatten()

# ============================================================================
# MAIN
# ============================================================================

def main():
    args = get_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print(f"\n[1] Loading data from: {args.h5_file}")
    
    with h5py.File(args.h5_file, 'r') as hf:
        esm_embeddings = hf['esm_embeddings'][:]
        qsar_features = hf['qsar_features'][:]
        log_mic = hf['log_mic'][:]
        
        print(f"    ESM embeddings:  {esm_embeddings.shape}")
        print(f"    QSAR features:   {qsar_features.shape}")
        print(f"    Targets (logMIC):{log_mic.shape}")
    
    if args.feature_type == 'esm':
        X = esm_embeddings
        print(f"    Using: ESM only -> {X.shape[1]} features")
    elif args.feature_type == 'qsar':
        X = qsar_features
        print(f"    Using: QSAR only -> {X.shape[1]} features")
    else:
        X = np.hstack([esm_embeddings, qsar_features])
        print(f"    Using: ESM + QSAR -> {X.shape[1]} features")
    
    y = log_mic
    
    # -------------------------------------------------------------------------
    # 2. COMPUTE SAMPLE WEIGHTS (prioritise low MIC)
    # -------------------------------------------------------------------------
    print("\n[2] Computing sample weights (low MIC → high weight)")
    
    # Weight = 1 / (MIC_value) → low MIC gets high weight
    # Shift to avoid division issues: weight ∝ 1 / (log_mic - min + 1)
    # Then normalise so mean weight = 1
    weights = 1.0 / (y - y.min() + 1.0)
    weights = weights / weights.mean()
    
    print(f"    Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"    Low MIC (potent) samples get ~{weights.max()/weights.min():.1f}x more weight")
    
    # -------------------------------------------------------------------------
    # 3. NORMALISE FEATURES
    # -------------------------------------------------------------------------
    print("\n[3] Normalising features (StandardScaler)")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # -------------------------------------------------------------------------
    # 4. TRAIN / VAL / TEST SPLIT
    # -------------------------------------------------------------------------
    print(f"\n[4] Splitting data: train={1-args.val_frac-args.test_frac:.0%}, "
          f"val={args.val_frac:.0%}, test={args.test_frac:.0%}")
    
    dataset = PeptideDataset(X, y, weights)
    n_total = len(dataset)
    n_test = int(n_total * args.test_frac)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val - n_test
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"    Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # -------------------------------------------------------------------------
    # 5. BUILD MODEL
    # -------------------------------------------------------------------------
    print(f"\n[5] Building MLP: {X.shape[1]} -> {args.hidden_dims} -> 1")
    print(f"    Activation: {args.activation}, Dropout: {args.dropout}")
    
    model = MLP(
        input_dim=X.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    print(f"    Loss: Weighted MSE (low MIC prioritised)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # -------------------------------------------------------------------------
    # 6. TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f"\n[6] Training for {args.epochs} epochs")
    
    train_losses = []
    val_spearman = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Quick validation check (Spearman is more informative than MSE for ranking)
        val_preds, val_targets = evaluate(model, val_loader, device)
        rho, _ = spearmanr(val_targets, val_preds)
        val_spearman.append(rho)
        
#        if epoch % 10 == 0 or epoch == 1:
        print(f"    Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_spearman={rho:.3f}")
    
    # -------------------------------------------------------------------------
    # 7. FINAL EVALUATION ON TEST SET
    # -------------------------------------------------------------------------
    print(f"\n[7] Final Evaluation on Test Set")
    
    test_preds, test_targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_targets, test_preds, top_frac=0.1)
    
    print(f"\n    --- Regression Metrics ---")
    print(f"    R²:         {metrics['R²']:.4f}")
    print(f"    RMSE:       {metrics['RMSE']:.4f}")
    print(f"    MAE:        {metrics['MAE']:.4f}")
    print(f"    Spearman ρ: {metrics['Spearman ρ']:.4f}")
    
    print(f"\n    --- Drug Discovery Metrics (top 10%) ---")
    print(f"    Enrichment Factor: {metrics['EF@10%']:.2f}x")
    print(f"    Precision@10%:     {metrics['Precision@10%']:.2%}")
    print(f"    Recall@10%:        {metrics['Recall@10%']:.2%}")
    print(f"    (Found {metrics['hits_in_top_k']:.0f} of {metrics['n_potent']:.0f} potent peptides in top 10%)")
    
    # -------------------------------------------------------------------------
    # 8. SAVE MODEL
    # -------------------------------------------------------------------------
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'args': vars(args),
            'metrics': metrics,
        }, args.save_model)
        print(f"\n    Model saved to: {args.save_model}")
    
    # -------------------------------------------------------------------------
    # 9. VISUALISATION
    # -------------------------------------------------------------------------
    if not args.no_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training curve (loss + spearman)
        ax1 = axes[0]
        ax1.plot(train_losses, 'b-', alpha=0.8, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weighted MSE Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_yscale('log')
        
        ax1b = ax1.twinx()
        ax1b.plot(val_spearman, 'orange', alpha=0.8, label='Val Spearman ρ')
        ax1b.set_ylabel('Spearman ρ', color='orange')
        ax1b.tick_params(axis='y', labelcolor='orange')
        ax1b.set_ylim(0, 1)
        
        ax1.set_title('Training Progress')
        
        # Predicted vs Actual
        ax2 = axes[1]
        ax2.scatter(test_targets, test_preds, alpha=0.5, edgecolors='none', s=20)
        lims = [min(test_targets.min(), test_preds.min()), max(test_targets.max(), test_preds.max())]
        ax2.plot(lims, lims, 'r--', alpha=0.8, label='Ideal')
        ax2.set_xlabel('Actual log₁₀(MIC)')
        ax2.set_ylabel('Predicted log₁₀(MIC)')
        ax2.set_title(f'Test Set: R² = {metrics["R²"]:.3f}, ρ = {metrics["Spearman ρ"]:.3f}')
        ax2.legend()
        ax2.set_aspect('equal', 'box')
        
        # Enrichment: highlight potent peptides
        ax3 = axes[2]
        potent_thresh = np.percentile(test_targets, 10)
        is_potent = test_targets <= potent_thresh
        
        # Sort by predicted MIC (ascending = most potent first)
        sort_idx = np.argsort(test_preds)
        cumulative_potent = np.cumsum(is_potent[sort_idx])
        x_frac = np.arange(1, len(test_preds) + 1) / len(test_preds)
        
        ax3.plot(x_frac * 100, cumulative_potent, 'b-', linewidth=2, label='Model')
        ax3.plot([0, 100], [0, is_potent.sum()], 'k--', alpha=0.5, label='Random')
        ax3.axvline(10, color='red', linestyle=':', alpha=0.7, label='Top 10%')
        ax3.set_xlabel('% of Peptides Screened (by predicted MIC)')
        ax3.set_ylabel('Cumulative Potent Peptides Found')
        ax3.set_title(f'Enrichment Curve (EF@10% = {metrics["EF@10%"]:.2f}x)')
        ax3.legend()
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, is_potent.sum() * 1.1)
        
        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=150)
        print(f"\n    Plot saved to: {args.plot_path}")
        plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
