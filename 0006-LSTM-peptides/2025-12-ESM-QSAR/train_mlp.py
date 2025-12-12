#!/usr/bin/env python3
"""
train_mlp.py - Train an MLP to predict peptide MIC from ESM + QSAR features.

Evaluation includes drug-discovery-relevant metrics: Enrichment Factor, Spearman ρ, BEDROC.

Usage:
    python train_mlp.py --h5_file peptides_featurised.h5 --epochs 150 --hidden_dims 64 32 16
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
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32, 16],
                        help="Hidden layer dimensions (e.g., --hidden_dims 64 32 16)")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'gelu', 'silu', 'tanh'],
                        help="Activation function")
    
    # Training
    parser.add_argument('--epochs', type=int, default=150,
                        help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
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
                        help="Path to save model (e.g., 'model.pt')")
    parser.add_argument('--no_plot', action='store_true',
                        help="Disable plotting")
    parser.add_argument('--plot_path', type=str, default='train_mlp_plots.png',
                        help="Path to save training plots")
    
    return parser.parse_args()

# ============================================================================
# DATASET
# ============================================================================

class PeptideDataset(Dataset):
    """PyTorch Dataset for peptide features."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# MODEL
# ============================================================================

def get_activation(name: str) -> nn.Module:
    return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh()}[name]

class MLP(nn.Module):
    """Multi-Layer Perceptron for regression."""
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2, 
                 activation: str = 'relu'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), get_activation(activation), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calc_bedroc(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 20.0) -> float:
    """
    BEDROC (Boltzmann-Enhanced Discrimination of ROC).
    Reference: Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508.
    """
    n = len(y_true)
    threshold = np.percentile(y_true, 10)
    is_active = y_true <= threshold
    n_actives = is_active.sum()
    
    if n_actives == 0 or n_actives == n:
        return 0.0
    
    order = np.argsort(y_pred)
    is_active_sorted = is_active[order]
    active_ranks = np.where(is_active_sorted)[0] + 1
    
    ra = n_actives / n
    s = np.sum(np.exp(-alpha * active_ranks / n))
    r_random = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    r_perfect = (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha / n))
    bedroc = (s - r_random) / (r_perfect - r_random)
    
    return max(0.0, min(1.0, bedroc))


def calc_enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, top_frac: float) -> dict:
    """Calculate enrichment factor at a given threshold."""
    n = len(y_true)
    k = max(1, int(n * top_frac))
    potent_threshold = np.percentile(y_true, 10)
    is_potent = y_true <= potent_threshold
    n_potent = is_potent.sum()
    
    if n_potent == 0:
        return {'ef': 0.0, 'hits': 0, 'precision': 0.0, 'recall': 0.0}
    
    top_k_idx = np.argsort(y_pred)[:k]
    hits = is_potent[top_k_idx].sum()
    
    return {
        'ef': (hits / k) / (n_potent / n),
        'hits': hits,
        'precision': hits / k,
        'recall': hits / n_potent
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression and drug-discovery metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    spearman_rho, _ = spearmanr(y_true, y_pred)
    bedroc = calc_bedroc(y_true, y_pred, alpha=20.0)
    
    ef_results = {f: calc_enrichment_factor(y_true, y_pred, f) for f in [0.01, 0.05, 0.10, 0.20]}
    
    potent_threshold = np.percentile(y_true, 10)
    n_potent = (y_true <= potent_threshold).sum()
    
    return {
        'R²': r2, 'RMSE': rmse, 'MAE': mae, 'Spearman ρ': spearman_rho, 'BEDROC': bedroc,
        'EF@1%': ef_results[0.01]['ef'], 'EF@5%': ef_results[0.05]['ef'],
        'EF@10%': ef_results[0.10]['ef'], 'EF@20%': ef_results[0.20]['ef'],
        'Precision@10%': ef_results[0.10]['precision'], 'Recall@10%': ef_results[0.10]['recall'],
        'hits@10%': ef_results[0.10]['hits'], 'n_potent': n_potent,
    }

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model. Returns predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        all_preds.append(model(X_batch.to(device)).cpu().numpy())
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
    
    # 1. Load data
    print(f"\n[1] Loading data from: {args.h5_file}")
    with h5py.File(args.h5_file, 'r') as hf:
        esm_embeddings = hf['esm_embeddings'][:]
        qsar_features = hf['qsar_features'][:]
        log_mic = hf['log_mic'][:]
        print(f"    ESM: {esm_embeddings.shape}, QSAR: {qsar_features.shape}, Targets: {log_mic.shape}")
    
    if args.feature_type == 'esm':
        X = esm_embeddings
    elif args.feature_type == 'qsar':
        X = qsar_features
    else:
        X = np.hstack([esm_embeddings, qsar_features])
    print(f"    Using: {args.feature_type} -> {X.shape[1]} features")
    
    y = log_mic
    
    # 2. Normalise features
    print("\n[2] Normalising features")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 3. Train/Val/Test split
    print(f"\n[3] Splitting data")
    dataset = PeptideDataset(X, y)
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
    
    # 4. Build model
    print(f"\n[4] Building MLP: {X.shape[1]} -> {args.hidden_dims} -> 1")
    model = MLP(X.shape[1], args.hidden_dims, args.dropout, args.activation).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 5. Training loop
    print(f"\n[5] Training for {args.epochs} epochs")
    train_losses, val_losses, val_spearman = [], [], []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        val_preds, val_targets = evaluate(model, val_loader, device)
        val_loss = ((val_preds - val_targets) ** 2).mean()
        val_losses.append(val_loss)
        rho, _ = spearmanr(val_targets, val_preds)
        val_spearman.append(rho)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, ρ={rho:.3f}")
    
    # 6. Final evaluation
    print(f"\n[6] Final Evaluation on Test Set")
    test_preds, test_targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_targets, test_preds)
    
    print(f"\n    --- Regression Metrics ---")
    print(f"    R²: {metrics['R²']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    print(f"    Spearman ρ: {metrics['Spearman ρ']:.4f}")
    
    print(f"\n    --- Drug Discovery Metrics ---")
    print(f"    BEDROC (α=20): {metrics['BEDROC']:.4f}")
    print(f"        EF@1%: {metrics['EF@1%']:.2f} x")
    print(f"        EF@5%: {metrics['EF@5%']:.2f} x")
    print(f"        EF@10%: {metrics['EF@10%']:.2f} x")
    print(f"        EF@20%: {metrics['EF@20%']:.2f} x")
    print(f"    Precision@10%: {metrics['Precision@10%']:.2%}")
    print(f"    Recall@10%: {metrics['Recall@10%']:.2%}")
    print(f"    (Found {metrics['hits@10%']:.0f} of {metrics['n_potent']:.0f} potent peptides in top 10%)")
 
    
    # 7. Save model
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'args': vars(args),
            'metrics': metrics,
        }, args.save_model)
        print(f"\n    Model saved to: {args.save_model}")
    
    # 8. Visualisation
    if not args.no_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curves
        ax1 = axes[0]
        ax1.plot(train_losses, 'b-', alpha=0.8, label='Train')
        ax1.plot(val_losses, 'r-', alpha=0.8, label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_yscale('log')
        ax1.legend(loc='upper right')
        ax1b = ax1.twinx()
        ax1b.plot(val_spearman, 'green', alpha=0.6, label='Val ρ')
        ax1b.set_ylabel('Spearman ρ', color='green')
        ax1b.tick_params(axis='y', labelcolor='green')
        ax1b.set_ylim(0, 1)
        ax1.set_title('Training Progress')
        
        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(test_targets, test_preds, alpha=0.5, edgecolors='none', s=20)
        lims = [min(test_targets.min(), test_preds.min()), max(test_targets.max(), test_preds.max())]
        ax2.plot(lims, lims, 'r--', alpha=0.8, label='Ideal')
        ax2.set_xlabel('Actual log₁₀(MIC)')
        ax2.set_ylabel('Predicted log₁₀(MIC)')
        ax2.set_title(f'Test Set: R² = {metrics["R²"]:.3f}, ρ = {metrics["Spearman ρ"]:.3f}')
        ax2.legend()
        ax2.set_aspect('equal', 'box')
        
        # Enrichment curve
        ax3 = axes[2]
        potent_thresh = np.percentile(test_targets, 10)
        is_potent = test_targets <= potent_thresh
        n_potent = is_potent.sum()
        n = len(test_preds)
        sort_idx = np.argsort(test_preds)
        cumulative_potent = np.cumsum(is_potent[sort_idx])
        x_pct = np.arange(1, n + 1) / n * 100
        
        ax3.plot(x_pct, cumulative_potent, 'b-', linewidth=2, label='Model')
        ax3.plot(x_pct, np.minimum(np.arange(1, n + 1), n_potent), 'g--', linewidth=1.5, alpha=0.7, label='Perfect')
        ax3.plot(x_pct, np.arange(1, n + 1) * (n_potent / n), 'k:', alpha=0.5, label='Random')
        
        for thresh, color in [(1, 'purple'), (5, 'red'), (10, 'orange'), (20, 'brown')]:
            k = max(1, int(n * thresh / 100))
            hits = cumulative_potent[k - 1]
            ef_val = metrics[f'EF@{thresh}%']
            ax3.scatter([thresh], [hits], color=color, s=60, zorder=5, edgecolors='white', linewidths=1.5)
            ax3.annotate(f'EF@{thresh}%={ef_val:.1f}x', xy=(thresh, hits), xytext=(thresh + 2, hits + 1),
                        fontsize=8, color=color)
        
        ax3.set_xlabel('% of Peptides Screened')
        ax3.set_ylabel('Cumulative Potent Peptides Found')
        ax3.set_title(f'Enrichment Curve (BEDROC={metrics["BEDROC"]:.3f})')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, n_potent * 1.1)
        
        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=150)
        print(f"\n    Plot saved to: {args.plot_path}")
        plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
