import numpy as np
import pandas as pd


def _permute_single(df, perm_idx, start_index):
    """Aplica um vetor de permutação a um único DataFrame.

    perm_idx tem comprimento (n-1) - start_index e reordena os log-retornos
    de índice >= start_index. Os primeiros `start_index` log-retornos ficam
    intactos, o que preserva close[0..start_index] exatamente.
    """
    log_close = np.log(df["close"].to_numpy())
    r = np.diff(log_close)                                # len n-1
    r_head = r[:start_index]
    r_tail_perm = r[start_index:][perm_idx]
    r_final = np.concatenate([r_head, r_tail_perm])
    n = len(df)
    close_perm = np.empty(n)
    close_perm[0] = df["close"].iloc[0]
    close_perm[1:] = close_perm[0] * np.exp(np.cumsum(r_final))
    multiplier = close_perm / df["close"].to_numpy()
    out = df.copy()
    out["open"] = df["open"].to_numpy() * multiplier
    out["high"] = df["high"].to_numpy() * multiplier
    out["low"] = df["low"].to_numpy() * multiplier
    out["close"] = close_perm
    return out


def get_permutation(dfs, start_index=0, seed=None):
    """Embaralha log-retornos do close preservando momentos e estrutura OHLC.

    dfs: DataFrame único ou lista. Se lista, todos com o mesmo índice — o mesmo
         vetor de permutação é aplicado a todos (preserva correlação cruzada
         entre ativos).
    start_index: barras [0, start_index] do close são intocadas; a permutação
                 atua nos log-retornos com índice >= start_index.
    seed: int opcional para reprodutibilidade.

    Retorna o mesmo tipo da entrada (DF único ou lista). Volume é preservado.
    """
    is_list = isinstance(dfs, list)
    dfs_list = dfs if is_list else [dfs]
    if not dfs_list:
        raise ValueError("dfs vazio")
    n = len(dfs_list[0])
    for d in dfs_list[1:]:
        if len(d) != n or not d.index.equals(dfs_list[0].index):
            raise ValueError("DFs da lista devem ter o mesmo índice")
    if not (0 <= start_index <= n - 2):
        raise ValueError(f"start_index fora do range: {start_index} (n={n})")

    rng = np.random.default_rng(seed)
    tail_len = (n - 1) - start_index
    perm_idx = rng.permutation(tail_len)

    out = [_permute_single(d, perm_idx, start_index) for d in dfs_list]
    return out if is_list else out[0]
