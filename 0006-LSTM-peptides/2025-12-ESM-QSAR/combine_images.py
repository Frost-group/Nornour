#!/usr/bin/env python3
"""
combine_images.py
Reads summary.csv (sorted by score) and combines the
corresponding 5 plot PNGs into a single 2x2 + 1-centred-bottom panel figure.

run_tag is reconstructed from the CSV columns using the same formula as
run_multiple.sh — no run_tag column needed in the CSV.
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT_DIR = "bilstm_seq_esm_qsar_gn6-10"
summary  = os.path.join(OUT_DIR, "summary.csv")

def build_run_tag(rank, mh_raw, ly, hd, dr, lr, eb):
    """Reconstruct the run_tag from hyperparameter fields — must match run_multiple.sh."""
    return f"rank{rank}_mh{mh_raw}_ly{ly}_hd{hd}_dr{dr}_lr{lr}_eb{eb}"

# Read CSV rows (already sorted by score descending)
with open(summary, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

run_tags = [
    build_run_tag(
        r["rank"], r["mlp_architecture"], r["layers"],
        r["hidden_layers"], r["dropout"], r["lr"], r["embed_dim"]
    )
    for r in rows
]

img_paths = [os.path.join(OUT_DIR, f"{tag}.png") for tag in run_tags]
missing   = [p for p in img_paths if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(f"Missing plot files: {missing}")

imgs   = [mpimg.imread(p) for p in img_paths]
labels = list("fghij")

fig = plt.figure(figsize=(20, 14))

# 2x2 top grid + 1 centred at bottom
axes_list = [
    fig.add_axes([0.01, 0.56, 0.48, 0.44]),  # a – top-left
    fig.add_axes([0.51, 0.56, 0.48, 0.44]),  # b – top-right
    fig.add_axes([0.01, 0.31, 0.48, 0.44]),  # c – mid-left
    fig.add_axes([0.51, 0.31, 0.48, 0.44]),  # d – mid-right
    fig.add_axes([0.26, 0.05, 0.48, 0.44]),  # e – bottom-centre
]

for ax, img, label in zip(axes_list, imgs, labels):
    ax.imshow(img)
    ax.axis("off")
    ax.text(-0.02, 1.03, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right")

plt.suptitle("Performance of Top 5 Hyperparameter Settings on Top 6-10 Gram-negative MIC Prediction for BiLSTM Sequence Model (+ESM+QSAR)", fontsize=18)
plt.tight_layout
panel_path = os.path.join(OUT_DIR, "panel_all.png")
fig.savefig(panel_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Panel saved to: {panel_path}")