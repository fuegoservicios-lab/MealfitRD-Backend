"""[P1-PLAN-LOTE-1085 · 2026-11-18] Abrir un día del diario tardaba 0,4-3,4 s: los micros de cada comida se recalculaban
(con embeddings de pago para los nombres sin coincidencia) en CADA visita. Ahora se recuerdan por la huella de los
ingredientes, en memoria y en Redis, y se calientan al registrar."""
from pathlib import Path

import pytest

import diary_micros
import micros_memo

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    micros_memo._vaciar_memoria()
    monkeypatch.setattr(micros_memo, "ACTIVO_EN_TESTS", True)
    import cache_manager
    monkeypatch.setattr(cache_manager, "redis_client", None, raising=False)
    yield
    micros_memo._vaciar_memoria()


def _contar(monkeypatch, resultado):
    llamadas = []

    def falso(ingredients, db):
        llamadas.append(list(ingredients or []))
        return resultado() if callable(resultado) else resultado
    monkeypatch.setattr(diary_micros, "micros_de_ingredientes", falso)
    return llamadas


def test_la_huella_es_estable_y_nada_sin_ingredientes():
    assert micros_memo.huella(["100 g de arroz", "1 huevo"]) == micros_memo.huella(["100 g de arroz", "1 huevo"])
    assert micros_memo.huella(["1 huevo", "100 g de arroz"]) != micros_memo.huella(["100 g de arroz", "1 huevo"])
    assert micros_memo.huella([]) is None and micros_memo.huella(None) is None and micros_memo.huella("x") is None


def test_la_misma_comida_se_calcula_una_vez_y_devuelve_copias(monkeypatch):
    llamadas = _contar(monkeypatch, lambda: {"values": {"fibra_g": 3.0}, "resolved": 1, "total": 1})
    a = micros_memo.micros_de_ingredientes_memo(["1 taza de avena"], db=None)
    a["values"]["fibra_g"] = 999          # quien lo usa lo toca: el guardado no cambia
    b = micros_memo.micros_de_ingredientes_memo(["1 taza de avena"], db=None)
    assert len(llamadas) == 1
    assert b["values"]["fibra_g"] == 3.0
    micros_memo.micros_de_ingredientes_memo(["2 tazas de avena"], db=None)
    assert len(llamadas) == 2


def test_sin_micros_tambien_se_recuerda_y_sin_ingredientes_no_se_guarda(monkeypatch):
    llamadas = _contar(monkeypatch, None)
    assert micros_memo.micros_de_ingredientes_memo(["algo raro"], db=None) is None
    assert micros_memo.micros_de_ingredientes_memo(["algo raro"], db=None) is None
    assert len(llamadas) == 1
    micros_memo.micros_de_ingredientes_memo([], db=None)
    micros_memo.micros_de_ingredientes_memo([], db=None)
    assert len(llamadas) == 3                # sin huella: siempre al cálculo de siempre


class _RedisFalso:
    def __init__(self, roto=False):
        self.d, self.roto = {}, roto

    def get(self, k):
        if self.roto:
            raise ConnectionError("caído")
        return self.d.get(k)

    def setex(self, k, ttl, v):
        if self.roto:
            raise ConnectionError("caído")
        assert ttl == micros_memo.TTL_S
        self.d[k] = v


def test_redis_sobrevive_al_proceso_y_si_se_cae_no_rompe(monkeypatch):
    import cache_manager
    r = _RedisFalso()
    monkeypatch.setattr(cache_manager, "redis_client", r, raising=False)
    llamadas = _contar(monkeypatch, lambda: {"values": {"hierro_mg": 2.0}})
    micros_memo.micros_de_ingredientes_memo(["1 huevo"], db=None)
    micros_memo._vaciar_memoria()           # otro proceso / tras un despliegue
    assert micros_memo.micros_de_ingredientes_memo(["1 huevo"], db=None) == {"values": {"hierro_mg": 2.0}}
    assert len(llamadas) == 1
    monkeypatch.setattr(cache_manager, "redis_client", _RedisFalso(roto=True), raising=False)
    micros_memo._vaciar_memoria()
    assert micros_memo.micros_de_ingredientes_memo(["1 huevo"], db=None) == {"values": {"hierro_mg": 2.0}}
    assert len(llamadas) == 2


def test_lo_usan_el_dia_del_diario_y_el_informe_y_se_calienta_al_registrar():
    diary = (RAIZ / "routers" / "diary.py").read_text(encoding="utf-8")
    assert '__import__("micros_memo").micros_de_ingredientes_memo(ingredients, _ndb)' in diary
    informe = (RAIZ / "informe_medico.py").read_text(encoding="utf-8")
    assert '__import__("micros_memo").micros_de_ingredientes_memo(m.get("ingredients"), ndb)' in informe
    facts = (RAIZ / "db_facts.py").read_text(encoding="utf-8")
    assert 'target=__import__("micros_memo").calentar' in facts


def test_fuera_de_sus_tests_no_recuerda_nada(monkeypatch):
    monkeypatch.setattr(micros_memo, "ACTIVO_EN_TESTS", False)
    llamadas = _contar(monkeypatch, {"values": {}})
    micros_memo.micros_de_ingredientes_memo(["1 huevo"], db=None)
    micros_memo.micros_de_ingredientes_memo(["1 huevo"], db=None)
    assert len(llamadas) == 2


def test_calentar_no_corre_en_tests_ni_lanza(monkeypatch):
    llamadas = _contar(monkeypatch, {"values": {}})
    micros_memo.calentar(["1 huevo"])
    assert llamadas == []
    micros_memo.calentar(None)
