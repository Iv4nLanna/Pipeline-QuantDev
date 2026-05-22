# Camada 1 — Sub-ciclo 1A (Fundação da validação estatística)

**Data:** 2026-05-22
**Status:** aprovado para implementação
**Escopo:** primeiro de dois sub-ciclos da Camada 1. Entrega a estratégia-cobaia (Donchian), o motor reutilizável de sinais/retornos/profit factor e o primeiro portão de validação — `in_sample_excellence` (Passo 1 da metodologia). Os Passos 2–4, o orquestrador `run_validation` e as visualizações restantes são o sub-ciclo **1B**.

---

## 1. Contexto e objetivo

A Camada 1 implementa o framework de validação estatística por **permutação** (metodologia Timothy Masters) cujo objetivo é detectar *data mining bias*: a tendência de qualquer estratégia otimizada em dados históricos parecer boa só por ter sido ajustada àqueles dados. A camada completa tem 4 passos em sequência rígida (um passo reprovado encerra o pipeline). Este spec cobre apenas a **fundação (1A)**: a infraestrutura compartilhada por todos os passos e o Passo 1.

A Camada 0 (já implementada) entrega OHLCV de BTC/USDT 1h limpo. A Camada 1 consome esse DataFrame, roda uma estratégia sobre uma grade de parâmetros e mede a qualidade in-sample antes dos testes de robustez.

Princípio de arquitetura (herdado da C0): funções puras, sem estado global, cada uma testável isoladamente; toda função-portão retorna o dict padronizado de `utils/contract.py`.

## 2. Decisões fechadas no brainstorming

1. **Engine: VectorBT.** Adotado de forma estrutural já no 1A, mesmo usando só parte agora, para suportar robustez futura. Usado para **vetorizar a geração de sinais sobre o `param_grid`**. O `profit_factor` é calculado por nós a partir dos **retornos barra-a-barra** (não do `Portfolio` do VBT), para bater exatamente com a metodologia. Verificado: `vectorbt 1.0.0` + `numba 0.65.1` (cp314) + `llvmlite 0.47.0` (cp314) resolvem em Python 3.14.
2. **Estratégia-cobaia: Donchian de 2 parâmetros, long+short** (estilo Turtle). Dois parâmetros permitem um heatmap 2D real e usam todo o range de sinal `[-1, 0, 1]`.
3. **Decomposição em 2 sub-ciclos.** 1A = fundação + Passo 1; 1B = Passos 2–4 + orquestrador + visualizações.
4. **Convenção de timing anti-lookahead:** o sinal de uma barra vale para a barra **seguinte** (`posicao = signal.shift(1)`) antes de aplicar a fórmula de retorno. Garante que nenhuma barra usa o próprio fechamento para lucrar com o movimento que o produziu.
5. **TDD** para todas as funções.

## 3. Conceito central: retornos barra-a-barra

Todas as métricas usam **retornos barra-a-barra**, não por trade (em estratégias com poucos trades, métricas por trade têm alta variância e pouco poder estatístico).

Seja `pos[t] = signal[t-1]` (sinal da barra anterior — convenção da decisão 4):

- `pos == 1`  (long):  `retorno_barra[t] = close[t] / close[t-1] - 1`
- `pos == -1` (short): `retorno_barra[t] = close[t-1] / close[t] - 1`
- `pos == 0`  (flat):  `retorno_barra[t] = 0`

`profit_factor = Σ(retornos positivos) / |Σ(retornos negativos)|`.

## 4. Fronteira de escopo 1A × 1B

| Componente | Sub-ciclo |
|---|---|
| Interface de estratégia + Donchian | 1A |
| `metrics.py` (bar_returns, profit_factor, trade_stats) | 1A |
| `engine.py` (evaluate_grid via VBT) | 1A |
| Passo 1 `in_sample_excellence` + heatmap de parâmetros | 1A |
| Passo 2 `permutation_test_is` | 1B |
| Passo 3 `walk_forward_test` | 1B |
| Passo 4 `wf_permutation_test` | 1B |
| Orquestrador `run_validation` | 1B |
| Visualizações 1–3 (histogramas, equity curve) | 1B |

## 5. Estrutura de arquivos (criada/modificada neste sub-ciclo)

```
trading_pipeline/
├── strategies/
│   └── donchian.py           # donchian(df, entry_lookback, exit_lookback) -> Series[-1,0,1]
├── validation/
│   ├── metrics.py            # bar_returns, profit_factor, trade_stats  (núcleo reusado pelo 1B)
│   ├── engine.py             # evaluate_grid(strategy_func, df, param_grid) via VectorBT
│   └── layer1.py             # in_sample_excellence + _plot_param_heatmap
├── data/
│   └── layer0.py             # + load_clean_ohlc(...) helper (reusa fetch_data + clean_data)
├── config.py                 # + DONCHIAN_PARAM_GRID, PF_MIN, red-flags
├── requirements.txt          # + vectorbt, numba
└── tests/
    ├── test_donchian.py
    ├── test_metrics.py
    ├── test_engine.py
    └── test_layer1.py
```

## 6. Especificação funcional

### 6.1 `strategies/donchian.py`

**Interface de estratégia (contrato para todas as estratégias futuras):**
`strategy_func(df, **params) -> pd.Series` com valores em `{-1, 0, 1}`, alinhada ao índice de `df`.

**`donchian(df, entry_lookback, exit_lookback) -> pd.Series`** (long+short, Turtle):
- Canal de entrada: máxima/mínima dos `entry_lookback` candles **anteriores** (`high.rolling(entry_lookback).max().shift(1)` e `low.rolling(...).min().shift(1)`).
- Canal de saída: máxima/mínima dos `exit_lookback` candles anteriores (mesmo `shift(1)`).
- Regra de posição (com estado, propagado para frente):
  - Entra **long** (`+1`) quando `close > canal_entrada_alto`.
  - Entra **short** (`-1`) quando `close < canal_entrada_baixo`.
  - Sai para **flat** (`0`) quando, estando long, `close < canal_saida_baixo`; ou, estando short, `close > canal_saida_alto`.
  - Uma entrada oposta inverte a posição diretamente.
- As primeiras `entry_lookback` barras (sem canal) recebem `0`.
- **Sem lookahead na construção dos canais:** o `shift(1)` garante que o canal em `t` usa apenas barras `< t`. (A defasagem de execução de 1 barra é aplicada depois, em `bar_returns`, conforme decisão 4.)

### 6.2 `validation/metrics.py`

- **`bar_returns(signals, close) -> np.ndarray`** — aplica `pos = signals.shift(1)` e a fórmula da Seção 3. Primeira barra = 0.
- **`profit_factor(returns) -> float`** — `Σ pos / |Σ neg|`. Se não houver retornos negativos, retorna um sentinela alto (`float("inf")`), que é tratado pela red-flag `PF > 5.0` no Passo 1. Se não houver positivos, retorna `0.0`.
- **`trade_stats(signals) -> tuple[int, float]`** — `n_trades` = nº de **entradas** (transições de `0`/sinal-oposto para um sinal não-nulo); `win_rate` = fração de trades cujo **retorno acumulado** (soma dos `bar_returns` durante a vigência da posição) é positivo. Sem trades → `(0, 0.0)`.

### 6.3 `validation/engine.py`

**`evaluate_grid(strategy_func, df, param_grid) -> pd.DataFrame`**
- `param_grid`: dict de listas (ex.: `{"entry_lookback": [...], "exit_lookback": [...]}`); o produto cartesiano define as combinações.
- Para cada combinação gera o sinal e calcula `profit_factor`, `win_rate`, `n_trades`, `n_bars`.
- Usa o VectorBT para vetorizar a varredura de parâmetros onde couber; o `profit_factor` é sempre calculado via `metrics.bar_returns` (não pelo `Portfolio` do VBT).
- Retorna DataFrame indexado pela combinação de parâmetros, colunas `[profit_factor, win_rate, n_trades, n_bars]`.

### 6.4 `validation/layer1.py`

**`in_sample_excellence(strategy_func, df, param_grid, results_dir="results") -> dict`** (Passo 1):
- Chama `evaluate_grid`; `best_param` = combinação com **maior `profit_factor`**.
- **Critério de aprovação:** `profit_factor > PF_MIN (1.05)` **E não** dispara nenhuma red-flag:
  - `win_rate > 0.95`
  - `n_trades < 30`
  - `profit_factor > 5.0`
- Gera o **heatmap de parâmetros** (`_plot_param_heatmap`): `profit_factor` in-sample para cada combinação `entry × exit`, salvo em `results/param_heatmap_<TIMESTAMP>.png`. Mostra se há "colina suave" (robusto) ou "pico isolado" (overfit).
- Retorna via `make_result(camada="in_sample_excellence", ...)` e salva via `save_result`. Métricas: `best_param`, `profit_factor`, `win_rate`, `n_trades`, `n_bars`, `heatmap`.

### 6.5 `data/layer0.py` — helper de fluxo de dados

**`load_clean_ohlc(ticker, timeframe, start, end, **kwargs) -> pd.DataFrame`** — reusa `fetch_data` + `clean_data` e devolve o DataFrame OHLCV limpo. Permite que a C1 obtenha dados reais sem alterar o contrato de retorno de `run_layer0`. As funções da C1 continuam recebendo `df` diretamente (testes usam `df` sintético).

## 7. `config.py` — adições

```python
DONCHIAN_PARAM_GRID = {
    "entry_lookback": [20, 30, 40, 60, 80, 100],
    "exit_lookback": [10, 20, 30, 40],
}
PF_MIN = 1.05                     # profit factor mínimo para aprovar in-sample
REDFLAG_WIN_RATE = 0.95           # win_rate acima disto = overfit suspeito
REDFLAG_MIN_TRADES = 30           # menos trades que isto = sem poder estatístico
REDFLAG_MAX_PF = 5.0              # PF acima disto = implausível em dados reais
```

## 8. Tratamento de erros

- `donchian`/`metrics`/`engine`: validam que o DataFrame não está vazio e tem as colunas necessárias (`donchian` exige `high`, `low`, `close`); senão `ValueError` com mensagem clara. Validação só na fronteira de entrada.
- `evaluate_grid`: `param_grid` vazio ou combinação inválida → `ValueError`.
- Funções internas confiam no contrato.

## 9. Estratégia de testes (TDD)

- **`test_donchian.py`:** série sintética com rompimento claro de máxima → `+1` no ponto certo; rompimento de mínima → `-1`; saída pelo canal → `0`. Teste explícito de **ausência de lookahead** (sinal em `t` não depende de dados `> t-1` na construção do canal).
- **`test_metrics.py`:** `bar_returns` com sinais/preços conhecidos (long, short, flat) → retornos esperados, incluindo o efeito do `shift(1)`. `profit_factor` com retornos conhecidos → valor exato; casos sem negativos (`inf`) e sem positivos (`0.0`). `trade_stats` com sequência de sinais conhecida → `n_trades` e `win_rate` corretos.
- **`test_engine.py`:** `evaluate_grid` em `df` sintético com grade pequena → shape do DataFrame, colunas e valores coerentes; confirma que o VBT/numba **executa** (não só importa).
- **`test_layer1.py`:** caso em que uma combinação claramente vence → `status: aprovado` e `best_param` correto; caso degenerado (PF < 1.05 ou red-flag) → `status: reprovado` com motivo. Verifica todas as chaves do contrato.
- Sem necessidade de rede (dados sintéticos). O teste do VBT roda offline.

## 10. Critério de pronto deste sub-ciclo

- `pytest` verde (offline).
- Smoke-test do VectorBT/numba passa em Python 3.14.
- `in_sample_excellence(donchian, <df real do BTC via load_clean_ohlc>, DONCHIAN_PARAM_GRID)` roda, imprime/salva o dict padronizado e gera o heatmap de parâmetros em `results/`.
- Output real (dict + heatmap) apresentado ao usuário antes de iniciar o sub-ciclo 1B.

## 11. Fora de escopo (sub-ciclo 1B e além)

Passos 2–4 (`permutation_test_is`, `walk_forward_test`, `wf_permutation_test`), orquestrador `run_validation`, histogramas de permutação e equity curve walk-forward, integração da Camada 1 ao `pipeline.py`. Camadas 2–8.
