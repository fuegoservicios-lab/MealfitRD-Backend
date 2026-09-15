# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-54 · 2026-09-15] El panel «solicitaste / aplicamos / por qué» dice cuándo el plan pasa del tiempo de
cocina que el usuario marcó. El revisor ya lo medía (`_fidelity_report.issues`, `prep_time_over_budget`), pero el panel
sólo leía el `mode`. En las 5 pruebas del dueño con «Nada» (unos 10 min) salían de 6 a 10 comidas por plan de hasta
70 min y la pantalla no lo decía.

Aquí vive el contrato entre los dos repos: el código, los campos `minutes` y `budget`, y el tope de 10 del backend (con
10, el panel dice «al menos 10»).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"

import horizon  # noqa: E402

_NADA = {"cookingTime": "none"}


def _cena(minutos: int) -> dict:
    return {"meal": "Cena", "prep_time": f"{minutos} min", "_prep_time_source": "biblioteca"}


def test_el_revisor_mide_el_tiempo_con_los_campos_que_lee_el_panel():
    issues, medidos, _ = horizon._prep_time_issues([{"meals": [_cena(65), _cena(8)]}], _NADA)
    assert medidos == ["prep_time"]
    assert [i["code"] for i in issues] == ["prep_time_over_budget"]
    assert issues[0]["minutes"] == 65 and issues[0]["budget"] == 10 and issues[0]["day"] == 1


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_el_tope_del_backend_es_el_que_supone_el_panel():
    issues, _, _ = horizon._prep_time_issues([{"meals": [_cena(60) for _ in range(15)]}], _NADA)
    tope = re.search(r"PREP_TIME_ISSUES_CAP = (\d+)", _front("src/config/planPolicy.js"))
    assert tope and len(issues) == int(tope.group(1)) == 10


def test_el_panel_lee_el_dato_del_revisor():
    pol = _front("src/config/planPolicy.js")
    panel = _front("src/components/dashboard/PlanPolicyPanel.jsx")
    assert "'prep_time_over_budget'" in pol and "export const prepTimeFact" in pol
    assert "prepTimeFact(fidelity)" in panel and "P1-PLAN-LOTE-54" in panel
    assert "fidelity={planData._fidelity_report}" in _front("src/pages/Dashboard.jsx")


def test_las_cadenas_nuevas_en_los_cuatro_catalogos():
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        cat = json.loads(_front(f"src/i18n/locales/{loc}.json"))
        assert cat.get("Tiempo de cocina") and cat.get("1 ajuste"), loc
        assert len([k for k in cat if k.startswith("Con el tiempo que marcaste")]) == 3, loc


def test_marcador_y_documentos():
    assert "P1-PLAN-LOTE-54" in (_BACKEND / "app.py").read_text(encoding="utf-8")
    for doc in ("plan_policy_f4.md", "plan_pendientes_2026_09_11.md"):
        assert "P1-PLAN-LOTE-54" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
