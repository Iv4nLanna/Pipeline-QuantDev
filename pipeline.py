import json

import config
from data.layer0 import run_layer0


def run_pipeline():
    resultado = run_layer0(
        config.TICKER, config.TIMEFRAME, config.START, config.END,
        ma_period=config.REGIME_MA_PERIOD,
        results_dir=config.RESULTS_DIR, cache_dir=config.CACHE_DIR,
        exchange_name=config.EXCHANGE,
    )
    print("=" * 60)
    print(f"CAMADA: {resultado['camada']}  STATUS: {resultado['status'].upper()}")
    print("=" * 60)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    return resultado


if __name__ == "__main__":
    run_pipeline()
