# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-103 · 2026-09-18] «Micros de hoy» + la pestaña «Progreso» del modo plan.

El contador de micros del día: por comida registrada, sus `ingredients` guardados se resuelven contra el catálogo
con el MISMO resolutor que el informe de micros del plan; sin ingredientes (foto, macros propias) no hay micros,
y el total dice con cuántas comidas se calculó. Metas: los ocho del dueño desde `micronutrients.dri_targets`
(sodio como techo). Ancla cross-repo de la división Plan / Progreso."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


class _DBFalsa:
    """Resolutor de mentira: conoce dos ingredientes; el resto no resuelve."""

    def micros_from_ingredient_string(self, s: str):
        if "Pechuga" in s:
            return {"grams": 150, "fiber": 0.0, "sodium_mg": 100.0, "potassium_mg": 400.0, "calcium_mg": 10.0,
                    "iron_mg": 1.0, "vit_c_mg": 0.0, "vit_a_mcg": 10.0, "vit_d_mcg": None}
        if "Guineo" in s:
            return {"grams": 100, "fiber": 2.6, "sodium_mg": 1.0, "potassium_mg": 358.0, "calcium_mg": 5.0,
                    "iron_mg": 0.3, "vit_c_mg": 8.7, "vit_a_mcg": 3.0, "vit_d_mcg": 0.0}
        return None


def test_micros_de_una_comida_suman_y_cuentan_lo_resuelto():
    from diary_micros import CLAVES, micros_de_ingredientes
    r = micros_de_ingredientes(["150 g de Pechuga de pollo", "100 g de Guineo maduro", "1 pizca de Polvo mágico"], _DBFalsa())
    assert r["resolved"] == 2 and r["total"] == 3
    assert set(r["values"]) == set(CLAVES)
    assert r["values"]["potassium_mg"] == 758.0 and r["values"]["fiber_g"] == 2.6
    # un micro sin dato (None) no suma ni cuenta como cero medido
    assert r["values"]["vit_d_mcg"] == 0.0


def test_sin_ingredientes_no_hay_micros_ni_cero_disfrazado():
    from diary_micros import micros_de_ingredientes, resumen_micros
    assert micros_de_ingredientes(None, _DBFalsa()) is None
    assert micros_de_ingredientes([], _DBFalsa()) is None
    assert micros_de_ingredientes(["", "   "], _DBFalsa()) is None
    con = {"micros": micros_de_ingredientes(["100 g de Guineo maduro"], _DBFalsa())}
    sin = {"micros": None}
    r = resumen_micros([con, sin])
    assert r["micros_coverage"] == {"con_datos": 1, "total": 2}
    assert r["micros"]["potassium_mg"] == 358.0


def test_los_ocho_del_dueno_y_el_sodio_como_techo():
    from diary_micros import CLAVES, metas_micros
    assert CLAVES == ("fiber_g", "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vit_c_mg", "vit_a_mcg", "vit_d_mcg")
    m = metas_micros("male", 31)
    assert set(m) == set(CLAVES)
    assert m["sodium_mg"] == {"target": 2000.0, "kind": "ceiling", "unit": "mg"}
    assert m["fiber_g"]["kind"] == "floor" and m["fiber_g"]["target"] == 38.0
    # sexo/edad cuentan (DRI): hierro 18 mg en mujer joven, 8 en hombre
    assert metas_micros("female", 30)["iron_mg"]["target"] == 18.0 and m["iron_mg"]["target"] == 8.0


def test_el_diario_de_hoy_lleva_micros_y_no_los_ingredientes():
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert 'm["micros"] = micros_de_ingredientes(m.pop("ingredients", None), _ndb)' in diary
    assert '"micros_coverage": _resumen["micros_coverage"],' in diary
    dbf = (_BACKEND / "db_facts.py").read_text(encoding="utf-8")
    assert '"consumed_at, meal_type, ingredients"' in dbf
    ud = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    assert '"micros": _metas_micros_de(hp),' in ud


def test_frontend_progreso_en_modo_plan_y_micros_en_los_dos_modos():
    nav = _front("src/config/dashboardNav.js")
    assert "{ key: 'progress', label: t('Progreso'), path: '/dashboard/progress' }" in nav
    assert '<Route path="/dashboard/progress" element={<ProgressPage />} />' in _front("src/App.jsx")
    dash = _front("src/pages/Dashboard.jsx")
    assert "<TrackingProgress" not in dash and "<WaterTracker" not in dash
    assert "useTodaysConsumedMeals(" in dash
    dt = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "<MicrosTracker userId={userProfile?.id} flatOnMobile />" in dt
    mt = _front("src/components/dashboard/MicrosTracker.jsx")
    for k in ("fiber_g", "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "vit_c_mg", "vit_a_mcg", "vit_d_mcg"):
        assert f"key: '{k}'" in mt
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        d = json.loads(_front(f"src/i18n/locales/{loc}.json"))
        for k in ("Micros de hoy", "Fibra", "Sodio", "Potasio", "Calcio", "Hierro", "Vitamina C", "Vitamina A", "Vitamina D"):
            assert d.get(k), f"{loc}: {k}"


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 103
    assert "P1-PLAN-LOTE-103" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
