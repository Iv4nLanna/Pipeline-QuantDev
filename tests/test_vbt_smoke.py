# tests/test_vbt_smoke.py
def test_numba_jit_executa():
    from numba import njit

    @njit
    def soma(a, b):
        return a + b

    assert soma(2, 3) == 5


def test_vectorbt_importa():
    import vectorbt as vbt

    assert hasattr(vbt, "Portfolio")
