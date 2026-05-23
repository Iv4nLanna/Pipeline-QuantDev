TICKER = "BTC/USDT"        # formato ccxt/Binance
TICKER_LABEL = "BTC-USD"   # rótulo de exibição
TIMEFRAME = "1h"
START = "2017-08-17"       # primeiro 1h real de BTC/USDT na Binance
END = "2023-12-31"
PAPER_START = "2024-01-01"
FEE = 0.001
MAX_DRAWDOWN_LIMIT = 0.25
KELLY_FRACTION = 0.25

EXCHANGE = "binance"
CACHE_DIR = "data/cache"
RESULTS_DIR = "results"
REGIME_MA_PERIOD = 200
GAP_THRESHOLD = 0.5        # |retorno| > 50% em 1h = candle anômalo

# Camada 1 — validação
DONCHIAN_PARAM_GRID = {
    "entry_lookback": [20, 30, 40, 60, 80, 100],
    "exit_lookback": [10, 20, 30, 40],
}
REDFLAG_WIN_RATE = 0.95      # win_rate acima disto = overfit suspeito
REDFLAG_MIN_TRADES = 30      # menos trades que isto = sem poder estatístico
REDFLAG_MAX_PF = 5.0         # PF acima disto = implausível em dados reais
