# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-231 · 2026-09-25] Lo tecleado en «Otra alergia» / «Otro alimento que no te gusta» llega a todas las
superficies que reescriben comida fuera del grafo, y la dieta del perfil manda sobre la del cuerpo.

Auditoría de solo lectura del 25-sep: `_enrich_clinical_from_profile` (swap, regenerar día, expandir, persistir) y
`build_clinical_form_from_profile` (escudo pre-INSERT y cadena de calidad) unían solo los chips; el abaratador del bloque
cambiaba almendras por maní mirando solo los chips (y el maní no tiene chip).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

HP = {"allergies": ["Mariscos"], "otherAllergies": "fresa, maní", "dislikes": ["Cilantro"],
      "otherDislikes": "remolacha", "dietType": "vegan", "country": "DO"}


def _fold(xs):
    return {str(x).strip().lower() for x in (xs or [])}


def test_el_swap_y_regenerar_dia_ven_el_texto_libre(monkeypatch):
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": dict(HP)})
    data = {"allergies": [], "dislikes": [], "dietType": "balanced", "otherAllergies": "kiwi"}
    rp._enrich_clinical_from_profile(data, "u-231")
    assert {"mariscos", "fresa", "maní", "kiwi"} <= _fold(data["allergies"]), data["allergies"]
    assert {"cilantro", "remolacha"} <= _fold(data["dislikes"]), data["dislikes"]
    assert data["diet_type"] == "vegan", "la dieta del perfil manda sobre el 'balanced' por defecto del cliente"


def test_la_ninguna_del_formulario_sigue_mandando(monkeypatch):
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {
        "allergies": ["Ninguna"], "otherAllergies": "fresa"}})
    data = {"allergies": []}
    rp._enrich_clinical_from_profile(data, "u-231")
    assert "fresa" not in _fold(data["allergies"])   # P0-FORM-1: el centinela es exclusivo, igual que en el generador


def test_el_escudo_pre_insert_ve_el_texto_libre(monkeypatch):
    import db_profiles as dp
    monkeypatch.setattr(dp, "get_user_profile", lambda uid: {"health_profile": dict(HP)})
    ctx = dp.build_clinical_form_from_profile("u-231")
    assert {"mariscos", "fresa", "maní"} <= _fold(ctx["allergies"]), ctx["allergies"]
    assert {"cilantro", "remolacha"} <= _fold(ctx["dislikes"]), ctx["dislikes"]
    assert HP["allergies"] == ["Mariscos"], "el perfil guardado no se toca (lo repinta el formulario)"


def test_el_abaratador_del_bloque_ve_el_texto_libre(monkeypatch):
    vistos = []
    monkeypatch.setattr(go, "BUDGET_CONVERGENCE_FUTURE_ONLY", False)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_ENABLED", False)

    def _captura(days, fd, **kw):
        vistos.append(fd)
        return 0
    monkeypatch.setattr(go, "_apply_budget_cheapen_pass", _captura)
    go.apply_budget_convergence_for_days({"days": [{"day": 1, "meals": []}]},
                                         {"allergies": [], "otherAllergies": "maní"})
    assert vistos and "maní" in _fold(vistos[0].get("allergies")), vistos


def test_anclas():
    for f in ("routers/plans.py", "db_profiles.py", "graph_orchestrator.py"):
        assert "tooltip-anchor: P1-PLAN-LOTE-231-TEXTO-LIBRE" in (_BACKEND / f).read_text(encoding="utf-8"), f


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 231
