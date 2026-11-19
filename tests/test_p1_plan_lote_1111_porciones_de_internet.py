"""[P1-PLAN-LOTE-1111 · 2026-11-19] El Multi+ Phyto del dueño quedó con su tabla de internet pero con 0 porciones: la
búsqueda SÍ traía «30 servings» (120 cápsulas, 4 por porción) y `completar_pote` no las escribía, así que el pote no
descontaba ni avisaba. Ahora, si el pote no las tenía, se ponen; si ya las tenía, mandan las suyas."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import etiqueta_web_vitaminas as v  # noqa: E402

HALLAZGO = {"porciones": 30.0, "fuente": "https://x.com/p", "supplement_facts": {
    "serving_text": "4 Capsules", "units_per_serving": 4, "serving_unit": "capsule", "amount_basis": "per_serving",
    "table_readable": True, "nutrients": [{"name": "Vitamin C", "amount": 250, "unit": "mg"}]}}


def _completar(monkeypatch, hallazgo):
    import suplementos
    import db_core
    w = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p: w.append((q, p)) or True)
    monkeypatch.setattr(suplementos, "buscar", lambda u, n: {"id": "p1", "serving_label": None})
    assert v.completar_pote("u1", "Multi + Phyto", hallazgo) is True
    return w[0]


def test_las_porciones_del_envase_se_ponen_si_el_pote_no_las_tenia(monkeypatch):
    q, p = _completar(monkeypatch, HALLAZGO)
    assert "COALESCE(quantity, 0) <= 0" in q and p[1:3] == (30.0, 30.0) and p[-2:] == ("p1", "u1")


def test_sin_porciones_o_absurdas_no_se_toca_la_cantidad(monkeypatch):
    for malo in (None, 0, 5000, "30"):
        _, p = _completar(monkeypatch, {**HALLAZGO, "porciones": malo})
        assert p[1:3] == (None, None)


def test_una_porcion_de_4_capsulas_cuenta_porciones_no_capsulas(monkeypatch):
    _, p = _completar(monkeypatch, HALLAZGO)
    assert p[3] == "porción"
    _, p = _completar(monkeypatch, {**HALLAZGO, "supplement_facts": {**HALLAZGO["supplement_facts"], "units_per_serving": 1}})
    assert p[3] is None, "1 cápsula por porción: se queda su unidad"


def _con_pote(monkeypatch, pote):
    import suplementos
    import db_core
    w = []
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p: w.append((q, p)) or True)
    monkeypatch.setattr(suplementos, "buscar", lambda u, n: pote)
    return w


LEGIBLE = {"supplement_facts": {"table_readable": True, "nutrients": [{"name": "Zinc", "amount": 11}]}}


def test_la_foto_legible_no_se_pisa_pero_recibe_las_porciones_que_le_faltaban(monkeypatch):
    w = _con_pote(monkeypatch, {"id": "p1", "quantity": 0, "label_source": "foto", "serving_label": LEGIBLE})
    assert v.completar_pote("u1", "x", HALLAZGO) is True
    assert "serving_label" not in w[0][0] and w[0][1] == (30.0, "p1", "u1")
    w = _con_pote(monkeypatch, {"id": "p1", "quantity": 12, "label_source": "foto", "serving_label": LEGIBLE})
    assert v.completar_pote("u1", "x", HALLAZGO) is False and not w, "con sus porciones, mandan las suyas"


def test_una_tabla_de_internet_si_se_actualiza(monkeypatch):
    # el Multi+ Phyto del dueño: completado sin porciones antes del 1111; la segunda pasada pone las 30
    w = _con_pote(monkeypatch, {"id": "p1", "quantity": 0, "label_source": "web", "serving_label": LEGIBLE})
    assert v.completar_pote("u1", "x", HALLAZGO) is True and "serving_label" in w[0][0] and w[0][1][1] == 30.0
