import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from utils.contract import make_result, save_result
from validation.engine import evaluate_grid
from validation.permutation import get_permutation


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


def _plot_pf_histogram(pf_perm, pf_real, p_value, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pf_perm, bins=30, color="#4a7", alpha=0.7, edgecolor="black")
    ax.axvline(pf_real, color="red", linestyle="--", lw=2,
               label=f"pf_real={pf_real:.3f}")
    ax.set_xlabel("profit_factor (permutação)")
    ax.set_ylabel("frequência")
    ax.set_title(f"Distribuição nula vs. PF real — p-valor={p_value:.4f} "
                 f"({len(pf_perm)} permutações)")
    ax.legend(loc="upper right")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"pf_perm_histograma_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return caminho


def permutation_test_is(strategy_func, df, param_grid, best_param,
                        pf_real, n_permutations=None, seed_base=0,
                        results_dir="results"):
    """Passo 2 da Camada 1: teste de permutação in-sample.

    Em cada uma das n_permutations permutações, re-otimiza param_grid completo
    e usa o MAX PF como estatística. p_value Monte Carlo não-viesado:
    (count(pf_perm >= pf_real) + 1) / (n_permutations + 1).
    """
    if n_permutations is None:
        n_permutations = config.N_PERMUTATIONS
    if not math.isfinite(pf_real):
        raise ValueError("pf_real infinito; rever red-flag REDFLAG_MAX_PF")

    pf_perm = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        df_perm = get_permutation(df, start_index=0, seed=seed_base + i)
        grid_df = evaluate_grid(strategy_func, df_perm, param_grid)
        pf_finito = grid_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
        pf_perm[i] = float(pf_finito.max()) if not pf_finito.empty else 0.0

    n_extremos = int((pf_perm >= pf_real).sum())
    p_value = (n_extremos + 1) / (n_permutations + 1)

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    array_path = os.path.join(results_dir, f"pf_perm_{ts}.npy")
    np.save(array_path, pf_perm)
    histograma = _plot_pf_histogram(pf_perm, pf_real, p_value, results_dir)

    metricas = {
        "best_param": best_param,
        "profit_factor_real": float(pf_real),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "pf_perm_array_path": array_path,
        "histograma": histograma,
    }
    if p_value < config.P_VALUE_THRESHOLD:
        status = "aprovado"
        motivo = (f"p_value={p_value:.4f} < {config.P_VALUE_THRESHOLD} "
                  f"em {n_permutations} permutações.")
        proximo = "Avançar para o Passo 3 (walk_forward_test)."
    else:
        status = "reprovado"
        motivo = (f"p_value={p_value:.4f} >= {config.P_VALUE_THRESHOLD} "
                  f"em {n_permutations} permutações.")
        proximo = "Estratégia indistinguível de ruído otimizado; não avançar."

    resultado = make_result("permutation_test_is", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
