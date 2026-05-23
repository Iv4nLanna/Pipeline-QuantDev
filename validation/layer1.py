import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from utils.contract import make_result, save_result
from validation.engine import evaluate_grid


def _plot_param_heatmap(grid_df, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    pf = grid_df["profit_factor"].replace([np.inf, -np.inf], np.nan)
    pivot = pf.unstack()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(pivot.columns.name)
    ax.set_ylabel(pivot.index.name)
    ax.set_title("Profit factor in-sample por parâmetro")
    fig.colorbar(im, ax=ax, label="profit_factor")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"param_heatmap_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return caminho


def in_sample_excellence(strategy_func, df, param_grid, results_dir="results"):
    grid_df = evaluate_grid(strategy_func, df, param_grid)
    best_idx = grid_df["profit_factor"].idxmax()
    best_row = grid_df.loc[best_idx]
    idx_vals = best_idx if isinstance(best_idx, tuple) else (best_idx,)
    best_param = {k: int(v) for k, v in zip(grid_df.index.names, idx_vals)}

    pf = float(best_row["profit_factor"])
    win_rate = float(best_row["win_rate"])
    n_trades = int(best_row["n_trades"])
    n_bars = int(best_row["n_bars"])
    heatmap = _plot_param_heatmap(grid_df, results_dir)

    metricas = {
        "best_param": best_param,
        "profit_factor": pf,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "n_bars": n_bars,
        "heatmap": heatmap,
    }

    red_flags = []
    if win_rate > config.REDFLAG_WIN_RATE:
        red_flags.append(f"win_rate {win_rate:.2%} > {config.REDFLAG_WIN_RATE:.0%}")
    if n_trades < config.REDFLAG_MIN_TRADES:
        red_flags.append(f"n_trades {n_trades} < {config.REDFLAG_MIN_TRADES}")
    if pf > config.REDFLAG_MAX_PF:
        red_flags.append(f"profit_factor {pf} > {config.REDFLAG_MAX_PF}")

    # Passo 1 é qualitativo: aprova se não houver red-flags de overfit ou problema
    # técnico. A magnitude do PF não é critério aqui — significância estatística
    # é o trabalho do Passo 2 (permutation_test_is).
    if not red_flags:
        status = "aprovado"
        motivo = f"PF={pf:.3f} com {n_trades} trades; sem red-flags."
        proximo = "Avançar para o Passo 2 (permutation_test_is)."
    else:
        status = "reprovado"
        motivo = "Red-flags: " + "; ".join(red_flags)
        proximo = "Revisar estratégia/param_grid; não avançar."

    resultado = make_result("in_sample_excellence", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
