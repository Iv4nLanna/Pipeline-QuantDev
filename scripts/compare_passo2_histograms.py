"""Plot comparativo lado a lado de dois histogramas de pf_perm (Passo 2).

Lê dois arquivos .npy salvos pelo run_passo2_btc.py, recebe os pf_real e
gera uma figura com 2 subplots compartilhando o eixo X — para comparação
direta da distribuição nula de duas estratégias avaliadas sobre o mesmo
dataset (mesmas permutações via seed_base=0).

Uso:
    .venv/bin/python scripts/compare_passo2_histograms.py \\
        --left  results/pf_perm_<ts>.npy --left-real 1.0419 --left-label "donchian" \\
        --right results/pf_perm_donchian_filtered_regime_<ts>.npy --right-real 1.0583 \\
        --right-label "donchian_filtered_regime" \\
        --out    results/comparacao_passo2.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _p_value(pf_perm, pf_real):
    n = len(pf_perm)
    return (int((pf_perm >= pf_real).sum()) + 1) / (n + 1)


def _draw(ax, pf_perm, pf_real, label):
    p = _p_value(pf_perm, pf_real)
    ax.hist(pf_perm, bins=30, color="#4a7", alpha=0.7, edgecolor="black")
    ax.axvline(pf_real, color="red", linestyle="--", lw=2,
               label=f"pf_real={pf_real:.4f}")
    ax.set_xlabel("profit_factor (permutação)")
    ax.set_ylabel("frequência")
    ax.set_title(f"{label}\np-valor={p:.4f} ({len(pf_perm)} perms)")
    ax.legend(loc="upper right")
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--left-real", type=float, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--right-real", type=float, required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pf_left = np.load(args.left)
    pf_right = np.load(args.right)

    # eixo X compartilhado, com folga
    xmin = min(pf_left.min(), pf_right.min(), args.left_real, args.right_real)
    xmax = max(pf_left.max(), pf_right.max(), args.left_real, args.right_real)
    pad = (xmax - xmin) * 0.05

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    p_l = _draw(ax_l, pf_left, args.left_real, args.left_label)
    p_r = _draw(ax_r, pf_right, args.right_real, args.right_label)
    ax_l.set_xlim(xmin - pad, xmax + pad)
    fig.suptitle("Passo 2 — distribuição nula vs PF real (mesmas 500 permutações; "
                 "seed_base=0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"escrito: {args.out}")
    print(f"  {args.left_label}: pf_real={args.left_real:.4f}, p={p_l:.4f}")
    print(f"  {args.right_label}: pf_real={args.right_real:.4f}, p={p_r:.4f}")


if __name__ == "__main__":
    main()
