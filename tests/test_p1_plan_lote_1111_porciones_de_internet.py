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
