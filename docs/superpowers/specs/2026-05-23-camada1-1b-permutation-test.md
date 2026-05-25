# Camada 1 — Sub-ciclo 1B (Passo 2: permutation_test_is)

**Data:** 2026-05-23
**Status:** rascunho — aguardando revisão
**Escopo:** segundo sub-ciclo da Camada 1. Entrega o **Passo 2** da metodologia (Aronson/Masters): teste de permutação in-sample que separa "sinal real" de "ruído ajustado a dados históricos". Os Passos 3–4, orquestrador `run_validation` e integração ao `pipeline.py` ficam fora — são sub-ciclo 1C+.

---

## 1. Contexto

O sub-ciclo 1A entregou a fundação (`donchian`, `bar_returns`, `profit_factor`, `evaluate_grid`, `in_sample_excellence`). A execução real do Passo 1 sobre BTC/USDT 1h aprovou a estratégia com `entry_lookback=100, exit_lookback=10` e `profit_factor=1.042` — sem red-flags de overfit. A pergunta que o Passo 1 **não responde** é: *"esse PF é distinto do que se obteria otimizando os mesmos parâmetros sobre ruído?"*

O Passo 2 responde isso por **permutação**: embaralha os log-retornos do close (destruindo padrões sequenciais reais mas preservando momentos estatísticos), re-otimiza o `param_grid` completo sobre cada permutação, e compara o `profit_factor` real com a distribuição de PFs sob a hipótese nula de ausência de estrutura. Um p-valor baixo significa que estratégias treinadas sobre ruído raramente alcançam o PF real — o sinal é estatisticamente improvável de ser fruto de data mining.

---

## 2. Contratos travados (não-negociáveis)

Estes são os requisitos duros que vieram da revisão do Passo 1. Qualquer desvio invalida o pipeline.

### 2.1 Herança da convenção anti-lookahead

1B **não redefine** `bar_returns`, `profit_factor`, `trade_stats` ou a convenção de timing. Toda permutação reusa `validation/metrics.bar_returns` (já aplica `signals.shift(1)`) e `validation/engine.evaluate_grid`. Se houver bug na convenção, é bug do 1A, não do 1B.

### 2.2 Algoritmo de permutação OHLC

`get_permutation(dfs, start_index=0, seed=None)`:

```text
Para cada DataFrame d em dfs (lista única ou um DF):
  1. r = log(d.close).diff()                              # log-retornos, len(d)
  2. r_fixed = r.iloc[:start_index]                       # cabeçalho preservado
  3. r_perm  = r.iloc[start_index:].sample(frac=1, seed)  # cauda embaralhada
                                                          # MESMO permutation index
                                                          # entre TODOS os DFs (cross-corr OK)
  4. r_final = concat([r_fixed, r_perm])
  5. close_perm[t] = close[0] * exp(cumsum(r_final))      # primeiro close intocado
  6. multiplier = close_perm / d.close                    # razão barra-a-barra
  7. open_perm  = d.open  * multiplier
     high_perm  = d.high  * multiplier
     low_perm   = d.low   * multiplier
     volume_perm = d.volume                               # volume não é permutado
  8. retorna DataFrame com mesmo índice, mesmas colunas
```

**Propriedades obrigatórias** (testadas no TDD):
- `mean`, `std`, `skew`, `kurtosis` dos log-retornos do close permutado estão dentro de **5%** dos valores reais.
- `autocorr(lag=1)` dos log-retornos do close permutado é `|ρ_perm| < 0.05` em série com `|ρ_real| > 0.10` (autocorrelação destruída).
- `close_perm[:start_index] == close[:start_index]` (cabeçalho intacto).
- Para lista de DFs com mesmo `start_index`, o mesmo vetor de índices é aplicado a todos — preservando correlação cruzada entre ativos.

**Sementes:** parâmetro `seed` opcional para reprodutibilidade nos testes. Em produção, cada permutação usa uma seed diferente derivada de `seed_base + i`.

### 2.3 Cálculo do p-valor

```python
n_extremos = sum(1 for pf in pf_perm if pf >= pf_real)
p_value = (n_extremos + 1) / (n_permutations + 1)
```

Estimador Monte Carlo não-viesado (Davison-Hinkley). O `+1` no numerador e denominador trata `pf_real` como uma observação a mais sob a hipótese nula — impede `p_value=0` quando nenhuma permutação atinge `pf_real`, o que seria incorreto com amostra finita. Mínimo possível com `N=500`: `p_value = 1/501 ≈ 0.002`. **Re-otimização completa** do `param_grid` em cada permutação — `best_pf_perm` é o **máximo** sobre todas as combinações, espelhando o que `in_sample_excellence` fez no real.

### 2.4 Configuração

- `N_PERMUTATIONS = 500` — default de desenvolvimento.
- Flag `n_permutations` na assinatura permite override para `1000` em validação final ou `100` em smoke-test.
- `P_VALUE_THRESHOLD = 0.01` — gate: `p_value < 0.01` ⇒ aprovado.

### 2.5 Output

`permutation_test_is` retorna o dict padronizado `make_result("permutation_test_is", ...)`. Métricas:
- `best_param` (recebido do Passo 1, não re-otimizado fora das permutações)
- `profit_factor_real` (PF do `best_param` no dado real)
- `pf_perm_array_path` (caminho do `.npy` salvo em `results/` com os N PFs permutados)
- `p_value`
- `n_permutations`
- `histograma` (caminho do PNG: distribuição de pf_perm + linha vertical no pf_real + anotação do p-valor)

Status:
- `aprovado` se `p_value < P_VALUE_THRESHOLD`
- `reprovado` caso contrário (motivo cita `p_value` e n permutações)

---

## 3. Decisões de localização

- **`validation/permutation.py`** (novo) — `get_permutation` mora aqui. É uma ferramenta de validação, não dados raw (não pertence a `data/`).
- **`validation/layer1.py`** — `permutation_test_is` e `_plot_pf_histogram` anexados. `in_sample_excellence` continua aqui sem alteração.
- **`config.py`** — `N_PERMUTATIONS`, `P_VALUE_THRESHOLD`.
- **`tests/test_permutation.py`** (novo) — TDD do `get_permutation`, incluindo o teste estatístico obrigatório.
- **`tests/test_layer1.py`** — anexa testes de `permutation_test_is`.

## 4. Estrutura de arquivos

```
trading_pipeline/
├── validation/
│   ├── permutation.py   # NOVO: get_permutation(dfs, start_index, seed)
│   └── layer1.py        # + permutation_test_is + _plot_pf_histogram
├── config.py            # + N_PERMUTATIONS, P_VALUE_THRESHOLD
└── tests/
    ├── test_permutation.py   # NOVO
    └── test_layer1.py        # + 2-3 testes de permutation_test_is
```

## 5. Especificação funcional detalhada

### 5.1 `validation/permutation.py`

```python
def get_permutation(dfs, start_index=0, seed=None):
    """Embaralha log-retornos do close preservando momentos e estrutura OHLC.

    Args:
        dfs: DataFrame único ou lista de DataFrames. Se lista, todos devem ter o
             mesmo índice (mesmo número de barras). O mesmo vetor de permutação é
             aplicado a todos os DFs — preserva correlação cruzada entre ativos.
        start_index: barras [0, start_index) não são permutadas (cabeçalho intacto).
                     Default 0 = permuta a série inteira (exceto a barra 0, que é
                     a âncora do cumsum).
        seed: int opcional para reprodutibilidade.

    Returns:
        Mesmo tipo da entrada (DF único ou lista), mesmas colunas, mesmo índice.
        Apenas open/high/low/close são modificados; volume é preservado.

    Raises:
        ValueError: se dfs vazio, se índices divergem entre DFs da lista, ou se
                    start_index não está em [0, len(df)-2].
    """
```

### 5.2 `validation/layer1.py` — anexos

```python
def permutation_test_is(strategy_func, df, param_grid, best_param,
                        pf_real, n_permutations=None, seed_base=0,
                        results_dir="results"):
    """Passo 2: teste de permutação in-sample.

    Args:
        strategy_func, df, param_grid: contrato da Camada 1.
        best_param: dict com a combinação eleita no Passo 1.
        pf_real: profit_factor de best_param no df real (passado pelo orquestrador
                 para evitar recálculo).
        n_permutations: default config.N_PERMUTATIONS.
        seed_base: cada permutação i usa seed = seed_base + i.
        results_dir: onde salvar histograma e array .npy.

    Returns:
        dict padronizado (make_result). Status 'aprovado' se p_value < threshold.
    """
```

```python
def _plot_pf_histogram(pf_perm, pf_real, p_value, results_dir):
    """Histograma de pf_perm com linha vertical em pf_real e anotação do p-valor.

    Retorna caminho do PNG salvo em results_dir.
    """
```

## 6. Estratégia de testes (TDD)

### 6.1 `test_permutation.py`

1. **`test_get_permutation_preserva_momentos`**: gera AR(1) com `phi=0.6` (autocorr forte), 5000 barras. Permuta. Asserta:
   - `|mean_perm - mean_real| / |mean_real| < 0.05`
   - `|std_perm - std_real| / std_real < 0.05`
   - `|skew_perm - skew_real| < 0.1` (skew tem variância alta em série finita)
   - `|kurt_perm - kurt_real| < 0.5`
2. **`test_get_permutation_destroi_autocorrelacao`**: mesma série AR(1). Asserta `abs(autocorr_real(lag=1)) > 0.5` e `abs(autocorr_perm(lag=1)) < 0.05`.
3. **`test_get_permutation_start_index_preserva_cabecalho`**: `start_index=100` numa série de 500 barras. Asserta `close_perm[:100] == close[:100]` exatamente.
4. **`test_get_permutation_multi_df_mesmo_indice`**: lista de 2 DFs sintéticos. Asserta que a sequência de log-retornos permutados é **idêntica** entre os dois DFs após permutação (preserva correlação cruzada).
5. **`test_get_permutation_multi_df_indices_divergentes_levanta`**: 2 DFs com `len` diferente → `ValueError`.
6. **`test_get_permutation_reprodutivel_com_seed`**: chamar 2x com mesma seed → resultado idêntico; sem seed → resultado diferente.
7. **`test_get_permutation_preserva_ohlc_via_multiplier`**: assert `(open_perm/close_perm == open/close).all()` (a razão O/C, H/C, L/C é invariante por construção).

### 6.2 `test_layer1.py` (anexar)

1. **`test_permutation_test_is_aprovado_quando_estrategia_real`**: estratégia *fake_persistente* (sempre long) sobre dado com tendência clara → `pf_real >> pf_perm` (em ruído, sem tendência, fake long fica perto de PF=1). Asserta `status="aprovado"` e `p_value < 0.1`. Usa `n_permutations=50` para velocidade.
2. **`test_permutation_test_is_reprovado_quando_estrategia_ruido`**: estratégia que ignora o df e retorna sinais aleatórios fixos. Real e permutado têm distribuições semelhantes → `p_value > 0.1`, `status="reprovado"`. `n_permutations=50`.
3. **`test_permutation_test_is_salva_array_e_histograma`**: confirma que `pf_perm_array_path` aponta para `.npy` existente, `histograma` aponta para `.png` existente, e `len(np.load(path)) == n_permutations`.

### 6.3 Determinismo

Todos os testes usam `seed_base` fixo. `numpy.random.default_rng(seed)` no `get_permutation`.

## 7. Tratamento de erros

- `get_permutation`: valida `dfs` não-vazio; valida índices coincidem se lista; valida `start_index` em range; senão `ValueError`.
- `permutation_test_is`: valida `pf_real` finito (se `inf`, p-valor não é bem definido — `ValueError` "PF real infinito; rever red-flag REDFLAG_MAX_PF").

## 8. Critério de pronto

- `pytest` verde (offline, sem rede).
- `permutation_test_is` rodado sobre o mesmo dataset do Passo 1 (`BTC/USDT 1h 2017-08-17 → 2023-12-31`), com `best_param={"entry_lookback":100,"exit_lookback":10}` e `pf_real=1.042`.
- Histograma + array `.npy` salvos em `results/`.
- Output (dict + histograma com linha do PF real + anotação do p-valor) apresentado ao usuário.

## 9. Fora de escopo

- Passos 3 e 4 (`walk_forward_test`, `wf_permutation_test`).
- Orquestrador `run_validation` encadeando os 4 passos.
- Integração ao `pipeline.py`.
- Aceleração via VectorBT do laço de re-otimização (500× é viável em loop direto; otimização ficará para 1C se medirmos tempo > 5 min).
- Suporte multi-mercado em produção (a infraestrutura está pronta via `get_permutation` aceitando lista, mas o `permutation_test_is` só recebe um `df` neste sub-ciclo).
