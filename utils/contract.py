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


def save_result(result, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"{result['camada']}_{ts}.json")
    with open(caminho, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return caminho
