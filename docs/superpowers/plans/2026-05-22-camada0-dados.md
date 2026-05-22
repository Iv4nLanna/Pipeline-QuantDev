# Camada 0 (Dados) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Criar o scaffolding do `trading_pipeline`, o `config.py`, o contrato padronizado de camada e implementar/validar a Camada 0 (coleta via ccxt/Binance, limpeza e detecção de regime).

**Architecture:** Funções puras sem estado global. Toda camada retorna o dict padronizado (`make_result`) e salva JSON em `results/`. A Camada 0 baixa OHLCV 1h de BTC/USDT (Binance) com paginação e cache em parquet, limpa os dados e classifica o regime via MA200.

**Tech Stack:** Python 3.14, venv, ccxt, pandas, numpy, scipy, matplotlib, pyarrow, pytest.

---

## File Structure

- `requirements.txt` — dependências fixadas.
- `config.py` — parâmetros globais.
- `utils/contract.py` — `make_result`, `save_result`.
- `data/layer0.py` — `fetch_data`, `clean_data`, `detect_regime`, `run_layer0`.
- `pipeline.py` — stub que roda só a Camada 0.
- `tests/test_contract.py`, `tests/test_layer0.py` — testes.
- `__init__.py` em todos os pacotes; `results/.gitkeep`.

---

## Task 1: Ambiente, scaffolding e dependências

**Files:**
- Create: `requirements.txt`, `results/.gitkeep`
- Create: `__init__.py` em `data/ validation/ costs/ robustness/ risk/ portfolio/ paper/ monitor/ propfirm/ strategies/ utils/ tests/`

- [ ] **Step 1: Criar requirements.txt**

```
ccxt>=4.2
pandas>=2.2
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
pyarrow>=15.0
pytest>=8.0
```

- [ ] **Step 2: Criar a árvore de pacotes e arquivos vazios**

Run:
```bash
cd /home/ivan/trading_pipeline
for d in data validation costs robustness risk portfolio paper monitor propfirm strategies utils tests; do mkdir -p "$d"; touch "$d/__init__.py"; done
mkdir -p results data/cache && touch results/.gitkeep
```

- [ ] **Step 3: Criar venv e instalar dependências**

Run:
```bash
cd /home/ivan/trading_pipeline
python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import ccxt, pandas, numpy, scipy, matplotlib, pyarrow, pytest; print('deps ok')"
```
Expected: imprime `deps ok` (se algum wheel faltar para Python 3.14, reportar qual pacote e parar).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt results/.gitkeep data/__init__.py validation/__init__.py costs/__init__.py robustness/__init__.py risk/__init__.py portfolio/__init__.py paper/__init__.py monitor/__init__.py propfirm/__init__.py strategies/__init__.py utils/__init__.py tests/__init__.py
git commit -m "chore: scaffolding de pacotes, requirements e venv"
```

---

## Task 2: config.py

**Files:**
- Create: `config.py`

- [ ] **Step 1: Escrever config.py**

```python
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
```

- [ ] **Step 2: Verificar import**

Run: `.venv/bin/python -c "import config; print(config.TICKER, config.START)"`
Expected: `BTC/USDT 2017-08-17`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: config.py com parâmetros globais (ccxt/Binance)"
```

---

## Task 3: Contrato padronizado — make_result

**Files:**
- Create: `utils/contract.py`
- Test: `tests/test_contract.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# tests/test_contract.py
import pytest
from utils.contract import make_result


def test_make_result_retorna_todas_as_chaves():
    r = make_result("camada0", "aprovado", {"n": 1}, "ok", "seguir")
    assert set(r.keys()) == {"camada", "status", "metricas", "motivo", "proximo_passo"}
    assert r["camada"] == "camada0"
    assert r["status"] == "aprovado"
    assert r["metricas"] == {"n": 1}


def test_make_result_rejeita_status_invalido():
    with pytest.raises(ValueError):
        make_result("camada0", "talvez", {}, "x", "y")
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_contract.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'utils.contract'` ou `ImportError`.

- [ ] **Step 3: Implementar make_result**

```python
# utils/contract.py
import json
import os
from datetime import datetime

_STATUS_VALIDOS = {"aprovado", "reprovado", "revisar"}


def make_result(camada, status, metricas, motivo, proximo_passo):
    if status not in _STATUS_VALIDOS:
        raise ValueError(f"status inválido: {status!r}; use um de {_STATUS_VALIDOS}")
    return {
        "camada": camada,
        "status": status,
        "metricas": metricas,
        "motivo": motivo,
        "proximo_passo": proximo_passo,
    }
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_contract.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/contract.py tests/test_contract.py
git commit -m "feat: make_result com validação de status"
```

---

## Task 4: Contrato padronizado — save_result

**Files:**
- Modify: `utils/contract.py`
- Test: `tests/test_contract.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# anexar em tests/test_contract.py
import json
from utils.contract import save_result, make_result


def test_save_result_cria_arquivo_com_timestamp(tmp_path):
    r = make_result("camada0", "aprovado", {"n": 5}, "ok", "seguir")
    caminho = save_result(r, results_dir=str(tmp_path))
    assert caminho.startswith(str(tmp_path))
    assert "camada0_" in caminho and caminho.endswith(".json")
    with open(caminho) as f:
        carregado = json.load(f)
    assert carregado == r
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_contract.py::test_save_result_cria_arquivo_com_timestamp -v`
Expected: FAIL com `ImportError: cannot import name 'save_result'`.

- [ ] **Step 3: Implementar save_result**

```python
# anexar em utils/contract.py
def save_result(result, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"{result['camada']}_{ts}.json")
    with open(caminho, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return caminho
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_contract.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/contract.py tests/test_contract.py
git commit -m "feat: save_result salva dict como JSON com timestamp"
```

---

## Task 5: clean_data

**Files:**
- Create: `data/layer0.py`
- Test: `tests/test_layer0.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# tests/test_layer0.py
import pandas as pd
import numpy as np
from data.layer0 import clean_data


def _df(idx, close, volume):
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume},
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def test_clean_data_remove_duplicatas_e_volume_zero():
    idx = ["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    df = _df(idx, close=[100, 100, 101, 102], volume=[10, 10, 0, 5])
    out = clean_data(df)
    # duplicata removida e candle de volume zero removido -> 2 linhas
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert (out["volume"] > 0).all()


def test_clean_data_remove_gap_anomalo():
    idx = pd.date_range("2020-01-01", periods=3, freq="1h")
    # segundo candle dobra de preço (+100%) -> anômalo, removido
    df = _df(idx, close=[100, 200, 201], volume=[5, 5, 5])
    out = clean_data(df, gap_threshold=0.5)
    assert 200 not in out["close"].values
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'data.layer0'`.

- [ ] **Step 3: Implementar clean_data (e cabeçalho do módulo)**

```python
# data/layer0.py
import os
import time

import numpy as np
import pandas as pd


def clean_data(df, gap_threshold=0.5):
    if df.empty or "close" not in df.columns:
        raise ValueError("clean_data requer DataFrame não-vazio com coluna 'close'")
    out = df[~df.index.duplicated(keep="first")].copy()
    out = out.sort_index()
    out = out[out["volume"] > 0]
    ret = out["close"].pct_change().abs()
    out = out[(ret <= gap_threshold) | ret.isna()]
    return out
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add data/layer0.py tests/test_layer0.py
git commit -m "feat: clean_data remove duplicatas, volume zero e gaps anômalos"
```

---

## Task 6: detect_regime

**Files:**
- Modify: `data/layer0.py`
- Test: `tests/test_layer0.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# anexar em tests/test_layer0.py
from data.layer0 import detect_regime


def test_detect_regime_classifica_tendencias():
    n = 250
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    subida = pd.DataFrame({"close": np.linspace(100, 300, n)}, index=idx)
    out_bull = detect_regime(subida, ma_period=200)
    assert (out_bull["regime"].dropna() == "bull").mean() > 0.8

    descida = pd.DataFrame({"close": np.linspace(300, 100, n)}, index=idx)
    out_bear = detect_regime(descida, ma_period=200)
    assert (out_bear["regime"].dropna() == "bear").mean() > 0.8


def test_detect_regime_adiciona_coluna_e_nan_no_aquecimento():
    idx = pd.date_range("2020-01-01", periods=250, freq="1h")
    df = pd.DataFrame({"close": np.linspace(100, 300, 250)}, index=idx)
    out = detect_regime(df, ma_period=200)
    assert "regime" in out.columns
    assert out["regime"].iloc[:199].isna().all()
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k detect_regime -v`
Expected: FAIL com `ImportError: cannot import name 'detect_regime'`.

- [ ] **Step 3: Implementar detect_regime**

```python
# anexar em data/layer0.py
def detect_regime(df, ma_period=200):
    if df.empty or "close" not in df.columns:
        raise ValueError("detect_regime requer DataFrame não-vazio com coluna 'close'")
    out = df.copy()
    ma = out["close"].rolling(ma_period).mean()
    ma_slope = ma.diff()
    regime = pd.Series(np.nan, index=out.index, dtype="object")
    bull = (out["close"] > ma) & (ma_slope > 0)
    bear = (out["close"] < ma) & (ma_slope < 0)
    valido = ma.notna()
    regime[valido] = "lateral"
    regime[valido & bull] = "bull"
    regime[valido & bear] = "bear"
    out["regime"] = regime
    return out
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k detect_regime -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add data/layer0.py tests/test_layer0.py
git commit -m "feat: detect_regime classifica bull/bear/lateral via MA200"
```

---

## Task 7: fetch_data com paginação e cache

**Files:**
- Modify: `data/layer0.py`
- Test: `tests/test_layer0.py`

- [ ] **Step 1: Escrever o teste de cache (sem rede)**

```python
# anexar em tests/test_layer0.py
from data import layer0


def test_fetch_data_le_do_cache_sem_rede(tmp_path, monkeypatch):
    # prepara um parquet de cache cobrindo o intervalo pedido
    idx = pd.date_range("2020-01-01", periods=48, freq="1h", tz="UTC")
    cache_df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx
    )
    cache_df.index.name = "datetime"
    cache_path = tmp_path / "btcusdt_1h.parquet"
    cache_df.to_parquet(cache_path)

    # se _download_ccxt for chamado, o teste falha (não deve haver rede)
    def _boom(*a, **k):
        raise AssertionError("não deveria baixar da rede quando o cache cobre o intervalo")

    monkeypatch.setattr(layer0, "_download_ccxt", _boom)

    out = layer0.fetch_data(
        "BTC/USDT", "1h", "2020-01-01", "2020-01-02",
        use_cache=True, cache_dir=str(tmp_path),
    )
    assert len(out) >= 24
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k fetch_data -v`
Expected: FAIL com `AttributeError`/`ImportError` (sem `fetch_data`/`_download_ccxt`).

- [ ] **Step 3: Implementar fetch_data + _download_ccxt**

```python
# anexar em data/layer0.py
import ccxt


def _download_ccxt(ticker, timeframe, start_ms, end_ms, exchange_name="binance"):
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = start_ms
    linhas = []
    while since < end_ms:
        for tentativa in range(3):
            try:
                lote = ex.fetch_ohlcv(ticker, timeframe, since=since, limit=1000)
                break
            except ccxt.NetworkError:
                if tentativa == 2:
                    raise
                time.sleep(2 ** tentativa)
        if not lote:
            break
        linhas.extend(lote)
        since = lote[-1][0] + tf_ms
        if len(lote) < 1000:
            continue
    df = pd.DataFrame(linhas, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns="ts")
    return df[df.index <= pd.to_datetime(end_ms, unit="ms", utc=True)]


def fetch_data(ticker, timeframe, start, end, use_cache=True,
               cache_dir="data/cache", exchange_name="binance"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "btcusdt_1h.parquet")
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    if use_cache and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts),
                              ["open", "high", "low", "close", "volume"]]

    df = _download_ccxt(ticker, timeframe, int(start_ts.value // 1_000_000),
                        int(end_ts.value // 1_000_000), exchange_name)
    if use_cache:
        df.to_parquet(cache_path)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts),
                  ["open", "high", "low", "close", "volume"]]
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k fetch_data -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add data/layer0.py tests/test_layer0.py
git commit -m "feat: fetch_data com paginação ccxt, retry e cache em parquet"
```

---

## Task 8: run_layer0 (orquestração, dict padronizado e gráfico)

**Files:**
- Modify: `data/layer0.py`
- Test: `tests/test_layer0.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# anexar em tests/test_layer0.py
def test_run_layer0_retorna_dict_padronizado(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=300, freq="1h", tz="UTC")
    fake = pd.DataFrame(
        {"open": np.linspace(100, 200, 300), "high": np.linspace(100, 200, 300),
         "low": np.linspace(100, 200, 300), "close": np.linspace(100, 200, 300),
         "volume": np.full(300, 5.0)}, index=idx)
    fake.index.name = "datetime"

    monkeypatch.setattr(layer0, "fetch_data", lambda *a, **k: fake)

    r = layer0.run_layer0(
        "BTC/USDT", "1h", "2020-01-01", "2020-01-13",
        ma_period=200, results_dir=str(tmp_path), cache_dir=str(tmp_path),
    )
    assert set(r.keys()) == {"camada", "status", "metricas", "motivo", "proximo_passo"}
    assert r["status"] in {"aprovado", "reprovado", "revisar"}
    assert r["camada"] == "camada0_dados"
    assert "distribuicao_regimes" in r["metricas"]
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k run_layer0 -v`
Expected: FAIL com `AttributeError: module 'data.layer0' has no attribute 'run_layer0'`.

- [ ] **Step 3: Implementar run_layer0 + _plot_regime**

```python
# anexar em data/layer0.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from utils.contract import make_result, save_result


def _plot_regime(df, ma_period, results_dir):
    from datetime import datetime
    os.makedirs(results_dir, exist_ok=True)
    ma = df["close"].rolling(ma_period).mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["close"], lw=0.8, label="close")
    ax.plot(df.index, ma, lw=1.0, label=f"MA{ma_period}")
    cores = {"bull": "#c8f7c5", "bear": "#f7c5c5", "lateral": "#eeeeee"}
    for reg, cor in cores.items():
        mask = df["regime"] == reg
        ax.fill_between(df.index, df["close"].min(), df["close"].max(),
                        where=mask, color=cor, alpha=0.3, step="mid")
    ax.legend(); ax.set_title("Camada 0 — preço, MA e regime")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"regime_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight"); plt.close(fig)
    return caminho


def run_layer0(ticker, timeframe, start, end, ma_period=200,
               results_dir="results", cache_dir="data/cache", exchange_name="binance"):
    bruto = fetch_data(ticker, timeframe, start, end,
                       cache_dir=cache_dir, exchange_name=exchange_name)
    n_bruto = len(bruto)
    limpo = clean_data(bruto)
    com_regime = detect_regime(limpo, ma_period=ma_period)

    pct_removido = 0.0 if n_bruto == 0 else round(100 * (n_bruto - len(limpo)) / n_bruto, 3)
    regimes = com_regime["regime"].dropna()
    dist = {} if regimes.empty else (regimes.value_counts(normalize=True) * 100).round(2).to_dict()
    ret = limpo["close"].pct_change().dropna()
    metricas = {
        "n_candles": int(len(limpo)),
        "intervalo": [str(limpo.index.min()), str(limpo.index.max())] if len(limpo) else [],
        "pct_removido_limpeza": pct_removido,
        "distribuicao_regimes": dist,
        "retorno_medio": round(float(ret.mean()), 8) if len(ret) else 0.0,
        "retorno_std": round(float(ret.std()), 8) if len(ret) else 0.0,
        "skew": round(float(skew(ret)), 4) if len(ret) > 2 else 0.0,
        "kurtosis": round(float(kurtosis(ret)), 4) if len(ret) > 2 else 0.0,
    }

    if len(limpo) >= ma_period:
        grafico = _plot_regime(com_regime, ma_period, results_dir)
        metricas["grafico_regime"] = grafico
        status, motivo = "aprovado", f"{len(limpo)} candles limpos, regimes detectados."
        proximo = "Avançar para Camada 1 (validação estatística)."
    else:
        status = "reprovado"
        motivo = f"Apenas {len(limpo)} candles após limpeza; mínimo {ma_period}."
        proximo = "Ampliar intervalo ou revisar fonte de dados."

    resultado = make_result("camada0_dados", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k run_layer0 -v`
Expected: 1 passed.

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv/bin/python -m pytest -v`
Expected: todos passam (testes que não exigem rede).

- [ ] **Step 6: Commit**

```bash
git add data/layer0.py tests/test_layer0.py
git commit -m "feat: run_layer0 orquestra camada 0, gera métricas e gráfico de regime"
```

---

## Task 9: pipeline.py stub e execução com dados reais

**Files:**
- Create: `pipeline.py`

- [ ] **Step 1: Escrever o stub do orquestrador**

```python
# pipeline.py
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
```

- [ ] **Step 2: Executar com dados reais do BTC (faz rede; baixa ~55k candles, pode demorar)**

Run: `.venv/bin/python pipeline.py`
Expected: imprime dict padronizado com `status: aprovado`, distribuição de regimes, e cria `results/camada0_dados_*.json` + `results/regime_*.png`.

- [ ] **Step 3: Inspecionar outputs**

Run: `ls -la results/ && cat results/camada0_dados_*.json | head -40`
Expected: JSON com métricas reais e PNG do gráfico de regime presentes.

- [ ] **Step 4: Commit**

```bash
git add pipeline.py
git commit -m "feat: pipeline.py stub executando a Camada 0 com dados reais"
```

- [ ] **Step 5: Apresentar output real ao usuário**

Mostrar o resumo estatístico (métricas do JSON) e o gráfico de regime gerado. Aguardar aprovação antes de iniciar a Camada 1.

---

## Self-Review

- **Cobertura do spec:** scaffolding (T1), config (T2), contrato `make_result`/`save_result` (T3–T4), `clean_data` (T5), `detect_regime` (T6), `fetch_data` com paginação/cache/retry (T7), `run_layer0` com métricas + gráfico + dict padronizado (T8), pipeline stub + execução real e apresentação (T9). Todos os requisitos da Seção 6 do spec cobertos.
- **Placeholders:** nenhum — todo passo tem código/comando concreto.
- **Consistência de tipos:** `make_result(camada, status, metricas, motivo, proximo_passo)` e `save_result(result, results_dir)` usados de forma idêntica em T3, T4 e T8. `fetch_data`/`_download_ccxt`/`detect_regime`/`clean_data` com assinaturas consistentes entre tasks. Camada nomeada `camada0_dados` consistente em T8 e teste.
