# Registro de estratégias testadas no Passo 2 — rastreabilidade de selection bias

**Dataset:** BTC/USDT 1h, 2017-08-17 → 2023-12-31 (55.713 barras)
**Threshold por estratégia:** `p_value < 0.01`
**N permutações:** 500

> **Por que este registro existe:** cada estratégia testada sobre o MESMO conjunto
> de dados aumenta a chance de pegar um falso positivo (multiple-comparisons /
> selection bias). Sob a hipótese nula, K estratégias têm probabilidade
> aproximadamente `1 - (1-α)^K` de pelo menos uma passar com α=0.01. Para K=2
> isso é ~2%; para K=10, ~10%. O threshold Bonferroni-ajustado seria `0.01/K`.
> Este arquivo lista TODAS as estratégias avaliadas — incluindo as reprovadas —
> para que o leitor possa decidir se a evidência acumulada é robusta.

## Resultados

| # | Estratégia | param_grid | best_param | pf_real | p_value | n_perm | status | data |
|---|------------|------------|------------|---------|---------|--------|--------|------|
| 1 | `donchian` (puro) | `entry∈{20,30,40,60,80,100} × exit∈{10,20,30,40}` (24 combos) | `entry=100, exit=10` | 1.0419 | 0.2575 | 500 | reprovado | 2026-05-23 |
| 2 | `donchian_filtered_regime` (filtro MA200 bull/bear) | mesma grade Donchian | _pendente_ | _pendente_ | _pendente_ | 500 | _pendente_ | 2026-05-23 |

## Threshold Bonferroni acumulado

| K (nº de estratégias testadas até aqui) | α individual | α Bonferroni (α/K) |
|---|---|---|
| 1 | 0.01 | 0.0100 |
| 2 | 0.01 | 0.0050 |
| 3 | 0.01 | 0.0033 |

## Convenção dos seeds

Cada execução usa `seed_base=0` com `seed[i] = seed_base + i` para `i ∈ [0, n_perm)`.
Isso significa que estratégias diferentes são avaliadas sobre **os mesmos DataFrames
permutados**, tornando os p-valores diretamente comparáveis (mesma realização da
distribuição nula).
