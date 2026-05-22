# tests/test_metrics.py
import numpy as np
import pandas as pd
from validation.metrics import bar_returns


def test_bar_returns_long_e_short_com_shift():
    close = pd.Series([100.0, 110.0, 99.0])
    signals = pd.Series([1, 1, -1])
    # pos = signals.shift(1) = [0, 1, 1] -> barra0 flat; barra1 long; barra2 long
    r = bar_returns(signals, close)
    assert r[0] == 0.0
    assert abs(r[1] - 0.10) < 1e-9          # long: 110/100 - 1
    assert abs(r[2] - (99 / 110 - 1)) < 1e-9  # long: 99/110 - 1


def test_bar_returns_short():
    close = pd.Series([100.0, 90.0])
    signals = pd.Series([-1, -1])
    # pos = [0, -1] -> barra1 short: 100/90 - 1
    r = bar_returns(signals, close)
    assert r[0] == 0.0
    assert abs(r[1] - (100 / 90 - 1)) < 1e-9


from validation.metrics import profit_factor


def test_profit_factor_valor_exato():
    r = np.array([0.1, -0.05, 0.2, -0.05])
    # pos = 0.3 ; neg = -0.1 ; PF = 3.0
    assert abs(profit_factor(r) - 3.0) < 1e-9


def test_profit_factor_sem_perdas_e_sem_ganhos():
    assert profit_factor(np.array([0.1, 0.2])) == float("inf")
    assert profit_factor(np.array([-0.1, -0.2])) == 0.0


from validation.metrics import trade_stats


def test_trade_stats_conta_trades_e_win_rate():
    signals = pd.Series([0, 1, 1, 0, -1, -1, 0])
    close = pd.Series([100.0, 100.0, 110.0, 121.0, 121.0, 121.0, 100.0])
    # posição efetiva (shift1): [0,0,1,1,0,-1,-1]
    # trade1 (long, barras 2-3): (1.10*1.10)-1 = +0.21 -> win
    # trade2 (short, barras 5-6): (1.0*1.21)-1 = +0.21 -> win
    n_trades, win_rate = trade_stats(signals, close)
    assert n_trades == 2
    assert abs(win_rate - 1.0) < 1e-9


def test_trade_stats_sem_trades():
    signals = pd.Series([0, 0, 0])
    close = pd.Series([100.0, 101.0, 102.0])
    assert trade_stats(signals, close) == (0, 0.0)
