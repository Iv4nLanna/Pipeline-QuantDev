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
