"""Executa Passo 1 + Passo 2 sobre BTC real com paralelização das permutações.

Uso:
    .venv/bin/python scripts/run_passo2_btc.py --strategy donchian
    .venv/bin/python scripts/run_passo2_btc.py --strategy donchian_filtered_regime

Não modifica validation/layer1.py — `permutation_test_is` lá fica serial como
referência limpa. Este runner replica o cálculo do p_value em paralelo via
multiprocessing.Pool (12 cores → ~4 min para N=500), reusando _plot_pf_histogram
+ make_result + save_result para preservar o mesmo contrato de output.

Cada estratégia testada deve ser registrada manualmente em
docs/STRATEGIES_TESTED.md para rastreamento do selection bias acumulado.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from data.layer0 import load_clean_ohlc
from strategies.donchian import donchian
from strategies.donchian_regime import donchian_filtered_regime
from utils.contract import make_result, save_result
from validation.engine import evaluate_grid
from validation.layer1 import _plot_pf_histogram, in_sample_excellence
from validation.permutation import get_permutation


STRATEGIES = {
    "donchian": donchian,
    "donchian_filtered_regime": donchian_filtered_regime,
}


_DF = None
_PARAM_GRID = None
_STRATEGY = None


def _worker_init(df, param_grid, strategy_name):
    global _DF, _PARAM_GRID, _STRATEGY
    _DF = df
    _PARAM_GRID = param_grid
    _STRATEGY = STRATEGIES[strategy_name]


def _worker_pf_for_seed(seed):
    df_perm = get_permutation(_DF, start_index=0, seed=seed)
    grid_df = evaluate_grid(_STRATEGY, df_perm, _PARAM_GRID)
    pf_finito = grid_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    return float(pf_finito.max()) if not pf_finito.empty else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="donchian")
    parser.add_argument("--n-perms", type=int, default=None,
                        help="default: config.N_PERMUTATIONS")
    args = parser.parse_args()

    strategy_name = args.strategy
    strategy_func = STRATEGIES[strategy_name]
    param_grid = config.DONCHIAN_PARAM_GRID
    n_perms = args.n_perms if args.n_perms is not None else config.N_PERMUTATIONS
    seed_base = 0

    print(f">> estratégia: {strategy_name}")
    t0 = time.time()
    df = load_clean_ohlc(config.TICKER, config.TIMEFRAME, config.START, config.END,
                         gap_threshold=config.GAP_THRESHOLD, cache_dir=config.CACHE_DIR,
                         exchange_name=config.EXCHANGE)
    print(f">> dataset: {len(df)} barras [{time.time()-t0:.1f}s]")

    t1 = time.time()
    r1 = in_sample_excellence(strategy_func, df, param_grid,
                              results_dir=config.RESULTS_DIR)
    print(f">> Passo 1 [{time.time()-t1:.1f}s] status={r1['status']}")
    print(f"   best_param={r1['metricas']['best_param']}")
    print(f"   pf_real={r1['metricas']['profit_factor']:.6f}")
    assert r1["status"] == "aprovado", f"Passo 1 reprovado para {strategy_name}"

    best_param = r1["metricas"]["best_param"]
    pf_real = float(r1["metricas"]["profit_factor"])
    if not math.isfinite(pf_real):
        raise ValueError("pf_real infinito; rever red-flag REDFLAG_MAX_PF")

    seeds = [seed_base + i for i in range(n_perms)]
    n_workers = max(1, mp.cpu_count() - 2)
    print(f">> Passo 2: {n_perms} permutações em {n_workers} workers (seed_base=0)...")

    t2 = time.time()
    # spawn em vez de fork: o fork enrosca numpy._dtype._legacy em Python 3.14
    # quando o worker chama pd.Series.diff() — bug de import lazy não resolvido
    # em workers forkados. Spawn re-importa tudo, paga overhead único na criação
    # do worker mas funciona em qualquer estratégia.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers,
                  initializer=_worker_init,
                  initargs=(df, param_grid, strategy_name)) as pool:
        pf_perm = np.array(pool.map(_worker_pf_for_seed, seeds), dtype=float)
    print(f">> Passo 2 [{time.time()-t2:.1f}s]")

    n_extremos = int((pf_perm >= pf_real).sum())
    p_value = (n_extremos + 1) / (n_perms + 1)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    array_path = os.path.join(config.RESULTS_DIR,
                              f"pf_perm_{strategy_name}_{ts}.npy")
    np.save(array_path, pf_perm)
    histograma = _plot_pf_histogram(pf_perm, pf_real, p_value, config.RESULTS_DIR)

    metricas = {
        "strategy": strategy_name,
        "best_param": best_param,
        "profit_factor_real": pf_real,
        "p_value": p_value,
        "n_permutations": n_perms,
        "pf_perm_array_path": array_path,
        "histograma": histograma,
    }
    if p_value < config.P_VALUE_THRESHOLD:
        status = "aprovado"
        motivo = (f"p_value={p_value:.4f} < {config.P_VALUE_THRESHOLD} "
                  f"em {n_perms} permutações.")
        proximo = "Avançar para o Passo 3 (walk_forward_test)."
    else:
        status = "reprovado"
        motivo = (f"p_value={p_value:.4f} >= {config.P_VALUE_THRESHOLD} "
                  f"em {n_perms} permutações.")
        proximo = "Estratégia indistinguível de ruído otimizado; não avançar."

    r2 = make_result("permutation_test_is", status, metricas, motivo, proximo)
    save_result(r2, results_dir=config.RESULTS_DIR)

    print()
    print("=== PASSO 2 RESULT ===")
    print(json.dumps(r2, indent=2, ensure_ascii=False, default=str))
    print()
    print(f"Tempo total: {time.time()-t0:.1f}s")
    print(f">> registre este resultado em docs/STRATEGIES_TESTED.md")


if __name__ == "__main__":
    main()
