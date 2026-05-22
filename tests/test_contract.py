import json
import pytest
from utils.contract import make_result, save_result


def test_make_result_retorna_todas_as_chaves():
    r = make_result("camada0", "aprovado", {"n": 1}, "ok", "seguir")
    assert set(r.keys()) == {"camada", "status", "metricas", "motivo", "proximo_passo"}
    assert r["camada"] == "camada0"
    assert r["status"] == "aprovado"
    assert r["metricas"] == {"n": 1}


def test_make_result_rejeita_status_invalido():
    with pytest.raises(ValueError):
        make_result("camada0", "talvez", {}, "x", "y")


def test_save_result_cria_arquivo_com_timestamp(tmp_path):
    r = make_result("camada0", "aprovado", {"n": 5}, "ok", "seguir")
    caminho = save_result(r, results_dir=str(tmp_path))
    assert caminho.startswith(str(tmp_path))
    assert "camada0_" in caminho and caminho.endswith(".json")
    with open(caminho) as f:
        carregado = json.load(f)
    assert carregado == r
