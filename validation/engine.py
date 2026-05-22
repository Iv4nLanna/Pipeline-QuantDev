import itertools

import pandas as pd

from validation.metrics import bar_returns, profit_factor, trade_stats


def evaluate_grid(strategy_func, df, param_grid):
    if not param_grid:
        raise ValueError("param_grid vazio")
    keys = list(param_grid.keys())
    rows = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        sig = strategy_func(df, **params)
        rets = bar_returns(sig, df["close"])
        n_trades, win_rate = trade_stats(sig, df["close"])
        rows.append({
            **params,
            "profit_factor": profit_factor(rets),
            "win_rate": win_rate,
            "n_trades": n_trades,
            "n_bars": len(df),
        })
    return pd.DataFrame(rows).set_index(keys)
