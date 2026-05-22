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
