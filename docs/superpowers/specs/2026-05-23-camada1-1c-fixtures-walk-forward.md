# Camada 1 — Sub-ciclo 1C (Fixtures sintéticos + Passo 3: walk_forward_test)

**Data:** 2026-05-23
**Status:** rascunho — aguardando revisão
**Escopo:** primeiro sub-ciclo sob o novo objetivo de "construir a fábrica". Entrega
(a) os três fixtures de cenário sob `tests/fixtures/` que serão usados para validar
TODAS as camadas existentes e futuras, e (b) o Passo 3 da Camada 1
(`walk_forward_test`). Passo 4 (`wf_permutation_test`) é sub-ciclo 1D.

---

## 1. Contexto e pivô de objetivo

Até 1B, o critério de "pronto" de cada camada era *"a Donchian passa neste gate?"*.
A partir de 1C o critério muda — formalmente:

> **Camada está pronta quando: calcula corretamente, trata casos de borda,
> retorna o dict padronizado `make_result`, e produz o status esperado
> em cada um dos três fixtures de cenário canônicos.**

Donchian deixa de ser o produto e vira **fixture do mundo real**. O resultado
do Passo 2 sobre Donchian (`reprovado`, `p=0.26`) deixa de ser "uma falha do
projeto" e vira "evidência de que o caminho de rejeição está funcionando".

Implicação operacional: **todo gate desta camada (e futuras) deve passar nos
três fixtures**, provando que aprova o que deve aprovar e reprova o que deve
reprovar. Sem fixtures de "edge real plantado" e "ruído puro", não há como
distinguir uma camada que funciona de uma que aprova/reprova tudo por
acidente.

## 2. Contrato dos fixtures (escopo desta sub-ciclo)

`tests/fixtures/synthetic_signals.py` expõe três cenários canônicos, cada um
estruturado como `Scenario(name, df, strategy, param_grid, expected_passo1,
expected_passo2, expected_passo3)`. As camadas iteram sobre os cenários e
asseguram o status esperado.

### 2.1 `scenario_edge_real`

**Mecanismo:** série de log-retornos com **drift condicional plantado**:
após N=3 barras de retorno positivo consecutivas, a barra seguinte tem drift
`+0.002`; após 3 barras negativas consecutivas, drift `-0.002`; caso
contrário drift `0`. Ruído gaussiano `σ=0.005` sempre presente.

**Estratégia que captura:** `strategy_momentum_consec(df, lookback)` —
long quando as últimas `lookback` barras foram todas positivas, short quando
todas negativas, flat caso contrário. Grid `{lookback: [2, 3, 4]}`.

**Justificativa:** o drift condicional **persiste** sob diferentes janelas
de treino (estável no tempo, sem regime change), portanto walk-forward
preserva o sinal. **Não persiste** sob permutação dos log-retornos (que
destrói a relação temporal entre eles).

**Status esperado:**
| Camada | Status | Por quê |
|---|---|---|
| Passo 1 | aprovado | n_trades >> 30; PF finito > 1; sem red-flags |
| Passo 2 | aprovado | permutação destrói drift condicional → PF cai → p < 0.01 |
| Passo 3 | aprovado | drift estável no tempo → OOS PF > 1.0 com trades suficientes |

### 2.2 `scenario_ruido_puro`

**Mecanismo:** random walk geométrico puro, log-retornos `~N(0, σ=0.005)`,
sem drift, sem autocorrelação, sem padrão.

**Estratégia:** mesma `strategy_momentum_consec` (importante usar a mesma
estratégia que captura o cenário com edge — assim a diferença vem dos dados,
não da estratégia).

**Status esperado:**
| Camada | Status | Por quê |
|---|---|---|
| Passo 1 | aprovado | sem red-flags (n_trades OK, PF finito, win_rate ~0.5) |
| Passo 2 | reprovado | strategy PF idêntico em real e permutado → p ~ 0.5 |
| Passo 3 | reprovado | OOS PF ≤ 1.0 (zero edge fora da amostra) |

Nota sobre a fragilidade de Passo 3 reprovar ruído: como ruído tem
`OOS PF ≈ 1.0 ± noise`, em algumas seeds pode ficar marginal. A spec **fixa
a seed** (`seed=0`) e o fixture é calibrado para que essa seed produza
`OOS PF < 1.0` (verificável; se na implementação a seed=0 falhar, a spec
admite escolher outra seed canônica e documentá-la em comentário do
fixture — **sem reotimizar** estratégia ou Passo).

### 2.3 `scenario_donchian_btc`

**Mecanismo:** OHLCV real do BTC/USDT 1h via `load_clean_ohlc` (cache parquet
existente). `strategy = donchian`, `param_grid = DONCHIAN_PARAM_GRID`.

**Status esperado:** (já medido em 1A/1B)
| Camada | Status |
|---|---|
| Passo 1 | aprovado |
| Passo 2 | reprovado |
| Passo 3 | desconhecido — depende da implementação, **não há expectativa fixa** |

Cenário "mundo real" — fica como sanity check + integração com cache,
sem assertiva rígida no Passo 3. A spec **não** trata "Donchian reprova
no Passo 3" como falha: é informação.

## 3. Especificação do Passo 3: `walk_forward_test`

### 3.1 Algoritmo (anchored walk-forward)

```
Entrada: strategy_func, df, param_grid, train_size, test_size, step

trades_oos = []
janelas = []
inicio = 0
enquanto (inicio + train_size + test_size) <= len(df):
    is_slice  = df.iloc[inicio : inicio + train_size]
    oos_slice = df.iloc[inicio + train_size : inicio + train_size + test_size]

    grid_is = evaluate_grid(strategy_func, is_slice, param_grid)
    best_param_janela = grid_is["profit_factor"].idxmax()  # com nome dos params

    sig_oos = strategy_func(oos_slice, **best_param_janela)
    rets_oos = bar_returns(sig_oos, oos_slice["close"])
    trades_oos.append(rets_oos)
    janelas.append({inicio, train_size, test_size, best_param_janela, pf_is, pf_oos})

    inicio += step

retornos_concat = concat(trades_oos)
pf_oos_total = profit_factor(retornos_concat)
n_oos_trades = soma de n_trades por janela (recalculada via trade_stats)
```

### 3.2 Defaults

- `train_size`: default `int(len(df) * 0.4)` — janela de treino de 40% do df.
- `test_size`: default `int(len(df) * 0.1)` — OOS de 10%.
- `step`: default igual a `test_size` (janelas OOS não sobrepostas → trades OOS
  independentes).
- Resulta em ~6 janelas para BTC (55k barras): IS=22k, OOS=5.5k cada.

### 3.3 Anti-lookahead

- Cada `oos_slice` começa **após** o fim do `is_slice` correspondente. Nenhuma
  re-otimização vê dados que ainda não aconteceram.
- A convenção `signals.shift(1)` continua aplicada via `bar_returns` em cada
  slice OOS.
- A primeira barra de cada `oos_slice` recebe `bar_return=0` por definição da
  convenção (não tem barra anterior dentro do slice). **Isso descarta a 1ª
  barra de cada OOS** — perda de ~`(n_janelas / len(df)) * 100`%, aceitável.

### 3.4 Gate (qualitativo, mesma filosofia de Passo 1)

Aprovado se **TODAS** as condições:
- `pf_oos_total > 1.0` (out-of-sample net positivo)
- `n_oos_trades >= REDFLAG_MIN_TRADES (30)` (poder estatístico)

Reprovado caso contrário. **Não há gate de magnitude** (PF > 1.05 etc) — a
magnitude é trabalho do Passo 4.

### 3.5 Output (`make_result`)

```python
{
  "camada": "walk_forward_test",
  "status": "aprovado" | "reprovado",
  "metricas": {
    "pf_oos_total": float,
    "n_oos_trades": int,
    "n_janelas": int,
    "train_size": int,
    "test_size": int,
    "step": int,
    "janelas": [
      {"inicio": int, "train_size": int, "test_size": int,
       "best_param": dict, "pf_is": float, "pf_oos": float, "n_trades_oos": int},
      ...
    ],
    "equity_curve_path": str,   # PNG da curva OOS concatenada
  },
  "motivo": str,
  "proximo_passo": "Avançar para o Passo 4 (wf_permutation_test)."
}
```

### 3.6 `_plot_equity_curve_oos`

Plot da curva de equity OOS concatenada (`(1 + retornos).cumprod()`), com
linhas verticais marcando o início de cada janela OOS. Salvo em
`results/equity_curve_wf_<TIMESTAMP>.png`.

## 4. Estratégia auxiliar: `strategy_momentum_consec`

Onde fica: `strategies/momentum_consec.py` (nova).

```python
def strategy_momentum_consec(df, lookback):
    """Long se as últimas `lookback` barras tiveram retorno positivo;
    short se todas negativas; flat caso contrário. Sem lookahead na construção
    (mas o sinal de t vê close[t]; defasagem de execução continua aplicada
    em metrics.bar_returns via signals.shift(1))."""
```

## 5. Estrutura de arquivos

```
trading_pipeline/
├── strategies/
│   └── momentum_consec.py        # NOVO: estratégia auxiliar dos fixtures
├── validation/
│   └── walk_forward.py           # NOVO: walk_forward_test + helpers de split
├── validation/layer1.py          # + _plot_equity_curve_oos (anexo)
└── tests/
    ├── fixtures/                  # NOVA pasta
    │   ├── __init__.py
    │   └── synthetic_signals.py   # fixture_edge_real, fixture_ruido_puro,
    │                              # strategy_momentum_consec re-export,
    │                              # Scenario namedtuple, get_scenarios()
    ├── test_fixtures.py           # NOVO: valida propriedades dos fixtures
    ├── test_momentum_consec.py    # NOVO: TDD da estratégia auxiliar
    ├── test_walk_forward.py       # NOVO: TDD do Passo 3
    ├── test_layer1.py             # + 3 testes de cenário no Passo 1
    ├── test_layer1_passo2_scenarios.py  # NOVO: 3 cenários no Passo 2
    └── test_walk_forward_scenarios.py   # NOVO: 3 cenários no Passo 3
```

## 6. Estratégia de testes (TDD)

### 6.1 `test_momentum_consec.py`
- Sinal `+1` quando últimas N barras positivas; `-1` quando todas negativas; `0` caso misto ou warmup.
- Sem lookahead (altera futuro não muda sinais passados).
- Valida coluna `close`.

### 6.2 `test_fixtures.py` (propriedades intrínsecas dos fixtures)
- `scenario_edge_real.df` tem `len >= 5000`, OHLC sem NaN, close positivo.
- `strategy_momentum_consec(scenario_edge_real.df, lookback=3)` produz
  `n_trades >= 30` e `profit_factor > 1.1` (sinal capturado).
- Mesma estratégia em `scenario_ruido_puro.df` produz `0.85 < pf < 1.15`
  (PF perto de 1, sem edge).
- Reprodutibilidade: chamar `get_scenarios()` 2x → DFs idênticos (seeds fixas).

### 6.3 `test_layer1.py` — extensão com cenários
- Roda `in_sample_excellence` sobre cada cenário; asserta `status` esperado
  do contrato (Seção 2). Itera com `pytest.parametrize`.

### 6.4 `test_layer1_passo2_scenarios.py`
- Roda `permutation_test_is` sobre cada cenário com `n_permutations=50`
  (compromisso entre tempo de teste e poder estatístico). Asserta `status`
  esperado. Donchian é skipado por tempo (`@pytest.mark.slow`).

### 6.5 `test_walk_forward.py` (TDD do Passo 3)
- Helper `_split_windows(n, train_size, test_size, step)` retorna lista de
  `(is_start, is_end, oos_start, oos_end)`; testa shape e ausência de
  sobreposição IS/OOS dentro da mesma janela.
- `walk_forward_test` num df sintético pequeno: confirma estrutura do dict
  retornado, presença de `equity_curve_path`, e que `pf_oos_total` bate com
  `profit_factor(concat(rets_oos_por_janela))`.

### 6.6 `test_walk_forward_scenarios.py`
- Roda `walk_forward_test` em `scenario_edge_real` e `scenario_ruido_puro`.
  Asserta `status` esperado. Donchian (BTC) opcional, sem assertiva rígida.

## 7. Tratamento de erros

- `walk_forward_test`: `train_size + test_size > len(df)` → `ValueError("dataframe muito curto")`.
- `walk_forward_test`: `train_size <= 0` ou `test_size <= 0` ou `step <= 0` → `ValueError`.
- `walk_forward_test`: zero janelas geradas → `ValueError("nenhuma janela")`.
- `strategy_momentum_consec`: ausência de `close` → `ValueError`.
- Fixtures: assumem ambiente em ordem (cache parquet do BTC presente para
  `scenario_donchian_btc`); fixture levanta `pytest.skip` se cache ausente.

## 8. Critério de pronto

- `pytest -q` verde, incluindo os 3 cenários em Passo 1, Passo 2 e Passo 3.
- Donchian em Passo 3 roda sem crash; status registrado mas sem assertiva
  fixa.
- Diretório `tests/fixtures/` existe e é importável (`tests/fixtures/__init__.py`).
- `docs/STRATEGIES_TESTED.md` atualizado com resultado de Donchian no Passo 3
  (sob a nova interpretação: **registro neutro, não falha do projeto**).

## 9. Decisões fechadas vs deferidas

**Fechadas nesta spec:**
- Anchored walk-forward (não rolling) — IS começa em 0 e desliza; OOS sempre
  após IS. Mais simples e mais conservador (treina sobre mais história).
- Gate qualitativo (PF > 1, n_trades >= 30) — sem magnitude. Significância
  fica para Passo 4.
- Drift condicional como mecanismo do edge_real — testável, estável no tempo,
  destruído pela permutação.
- Janelas OOS não sobrepostas — `step = test_size`. Maximiza independência
  estatística do conjunto OOS.

**Deferidas (próximos sub-ciclos):**
- **Passo 4 (`wf_permutation_test`)** — sub-ciclo 1D. Aplica permutação ao
  walk-forward, p-valor sobre `pf_oos_total`.
- **Sequência das Camadas 2-8** — pergunta aberta para o usuário. Inferência
  baseada em diretórios + constantes em config.py sugere:
  `2=risk → 3=costs → 4=portfolio → 5=robustness → 6=paper → 7=propfirm → 8=monitor`.
  Mas o usuário mencionou um `CLAUDE.md` que **não existe no projeto**. Antes
  de iniciar 2+, precisamos travar essa sequência.
- **`pipeline.py` orquestrador** — só após pelo menos uma camada 2+ pronta.

## 10. Fora de escopo

- Otimização de tempo do walk_forward (paralelização — viável via
  multiprocessing, mas adiado).
- Métricas além de profit_factor (Sharpe, Sortino, max drawdown OOS) — Passo 3
  produz apenas PF + n_trades. Camada 5 (robustness) adiciona o resto.
- Walk-forward com re-treino contínuo (todo step uma nova otimização) —
  o algoritmo atual já faz isso; nada a discutir.
- Mudança do contrato de Passo 1 ou Passo 2 — 1C **não modifica** Passos 1 e 2.
  Apenas adiciona testes contra os cenários.
