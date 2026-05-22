import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Executa testes marcados com 'network' (acesso real à Binance).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: testes que acessam a rede (Binance); rodam só com --run-network",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="precisa de --run-network")
        for item in items:
            if item.get_closest_marker("network"):
                item.add_marker(skip_network)
