# Ciclo 1 — Scaffolding + config.py + Camada 0 (Dados)

**Data:** 2026-05-22
**Status:** aprovado para implementação
**Escopo:** primeiro incremento do `trading_pipeline`. Cria a estrutura completa do projeto, o `config.py` e implementa e valida totalmente a Camada 0 (coleta, limpeza e detecção de regime). As demais camadas (1–8) serão ciclos próprios de spec → plano → implementação.

---

## 1. Contexto e objetivo

O projeto completo está descrito no `CLAUDE.md` (fonte de verdade). Este spec cobre apenas o Ciclo 1.

A Camada 0 entrega dados OHLCV limpos e classificados por regime de mercado, no formato de dict padronizado, para alimentar a Camada 1 (validação estatística). Princípio da arquitetura: cada camada recebe um dict padronizado e retorna um dict padronizado, sem estado global e com funções puras e testáveis isoladamente.

## 2. Decisões de design (fechadas no brainstorming)

1. **Fonte de dados: ccxt + Binance.** yfinance não fornece 1h histórico desde 2016 (limite ~730 dias para intraday). Usamos ccxt/Binance, par `BTC/USDT` spot, 1h, cujo histórico real começa em ~2017-08-17.
2. **Ambiente: venv + requirements.txt.** Virtualenv local `.venv`, dependências fixadas em `requirements.txt`. Evita PEP 668 do Python 3.14 do sistema.
3. **Cache de dados em disco.** O download bruto (~55k candles) é cacheado em `data/cache/btcusdt_1h.parquet`. Execuções seguintes leem do cache; re-download só com flag explícita.
4. **TDD.** Testes antes da implementação para todas as funções da camada.

## 3. Contrato padronizado de camada (Seção 3 do CLAUDE.md)

Toda função de camada retorna exatamente:

```python
{
    "camada": "nome_da_camada",
    "status": "aprovado" | "reprovado" | "revisar",
    "metricas": { ... },
    "motivo": "string explicando a decisão",
    "proximo_passo": "string com recomendação"
}
```

Implementado como helper compartilhado em `utils/contract.py`:
- `make_result(camada, status, metricas, motivo, proximo_passo) -> dict` — valida que `status` é um dos três valores permitidos e que todas as chaves estão presentes.
- `save_result(result, results_dir="results") -> str` — salva o dict como `results/<camada>_<TIMESTAMP>.json` e retorna o caminho. Timestamp em formato `YYYYMMDD_HHMMSS`.

## 4. Estrutura de pastas criada neste ciclo

```
trading_pipeline/
├── data/
│   ├── __init__.py
│   ├── layer0.py          # fetch_data, clean_data, detect_regime, run_layer0
│   └── cache/             # parquet do download bruto (gitignored)
├── validation/__init__.py
├── costs/__init__.py
├── robustness/__init__.py
├── risk/__init__.py
├── portfolio/__init__.py
├── paper/__init__.py
├── monitor/__init__.py
├── propfirm/__init__.py
├── strategies/__init__.py
├── utils/
│   ├── __init__.py
│   └── contract.py        # make_result, save_result
├── results/               # JSONs + PNGs de output (gitignored exceto .gitkeep)
├── tests/
│   ├── __init__.py
│   ├── test_contract.py
│   └── test_layer0.py
├── config.py
├── requirements.txt
├── .gitignore
└── pipeline.py            # stub neste ciclo (orquestrador completo vem depois)
```

`pipeline.py` neste ciclo é um stub mínimo que chama apenas a Camada 0, para validar o fluxo end-to-end do primeiro incremento.

## 5. config.py

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
```

## 6. Camada 0 — especificação funcional

Arquivo `data/layer0.py`. Funções puras, sem estado global.

### 6.1 `fetch_data(ticker, timeframe, start, end, use_cache=True) -> pd.DataFrame`
- Usa ccxt (`binance`) `fetch_ohlcv` com **paginação**: a API retorna ~1000 candles por chamada; itera avançando o `since` até cobrir `[start, end]`, respeitando `rateLimit`.
- Retorna DataFrame com índice `datetime` (UTC) e colunas `open, high, low, close, volume`.
- **Cache:** se `use_cache` e o parquet existir e cobrir o intervalo, lê do disco. Caso contrário baixa, salva em `CACHE_DIR/btcusdt_1h.parquet` e retorna.
- Erros de rede: retry com backoff (até 3 tentativas) antes de falhar com mensagem clara.

### 6.2 `clean_data(df) -> pd.DataFrame`
- Remove duplicatas de índice (mantém primeira ocorrência).
- Remove candles com `volume == 0`.
- Remove gaps anômalos: candles cujo retorno absoluto `|close/close_prev - 1|` excede um limite (default 0.5 = 50% em 1h, claramente anômalo). Loga quantas linhas foram removidas em cada etapa.
- Garante índice ordenado e crescente.

### 6.3 `detect_regime(df, ma_period=200) -> pd.DataFrame`
- Calcula MA de `ma_period` períodos sobre `close`.
- Classifica cada período: `bull` se `close > MA` e MA crescente; `bear` se `close < MA` e MA decrescente; `lateral` caso contrário.
- Adiciona coluna `regime`. As primeiras `ma_period-1` linhas (sem MA) recebem `NaN`/`indefinido` e não entram no resumo.

### 6.4 `run_layer0(...) -> dict`
- Orquestra: fetch → clean → detect_regime.
- Monta `metricas`: n_candles, intervalo coberto, % removido na limpeza, distribuição de regimes (% bull/bear/lateral), estatísticas de retorno (média, std, skew, kurtosis).
- Gera **gráfico de regime** (`results/regime_<TIMESTAMP>.png`): preço + MA200 com fundo colorido por regime.
- `status`: `aprovado` se houver dados suficientes (ex.: ≥ `ma_period` candles após limpeza e cobertura razoável do intervalo); senão `reprovado` com motivo.
- Retorna dict padronizado via `make_result` e salva via `save_result`.

## 7. Tratamento de erros

- `fetch_data`: retry/backoff em erro de rede; falha explícita se o intervalo não puder ser coberto.
- `clean_data`/`detect_regime`: validam que o DataFrame tem as colunas esperadas e não está vazio; senão `ValueError` com mensagem clara.
- Validação só nas fronteiras (entrada de dados externos). Funções internas confiam no contrato.

## 8. Estratégia de testes (TDD)

- `tests/test_contract.py`: `make_result` rejeita status inválido e exige chaves; `save_result` cria arquivo com timestamp e conteúdo correto (usa `tmp_path`).
- `tests/test_layer0.py`:
  - `clean_data` remove duplicatas, volume zero e gaps — DataFrames sintéticos determinísticos.
  - `detect_regime` classifica corretamente séries sintéticas (tendência de alta → bull, de baixa → bear, ruído plano → lateral).
  - `fetch_data`: um teste de integração curto (intervalo de poucos dias) marcável como `network`, e um teste de que o cache é lido sem rede na segunda chamada.
  - `run_layer0`: retorna dict com todas as chaves do contrato e `status` válido.

## 9. Critério de pronto deste ciclo

- `pytest` verde (testes que não exigem rede passam offline).
- `python pipeline.py` roda a Camada 0 com dados reais do BTC, imprime o dict padronizado, salva JSON em `results/` e gera o PNG de regime.
- Output real (resumo estatístico + gráfico) apresentado ao usuário para aprovação antes de iniciar a Camada 1.

## 10. Fora de escopo (ciclos futuros)

Camadas 1–8, orquestrador completo (`run_pipeline`), estratégia Donchian (`strategies/donchian.py`) — cada uma em seu próprio ciclo de spec → plano → implementação.
