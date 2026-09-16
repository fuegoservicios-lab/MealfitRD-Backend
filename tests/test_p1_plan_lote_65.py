# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-65 · 2026-09-16] F8: cada arnés declara el knob que necesita, y la lista del paso con el perfil de
producción baja a la mitad.

Los 35 ficheros marcados con `MEALFIT_VERIFIED_INGREDIENTS_ONLY` en las dos listas parchean el catálogo a vacío
(`get_master_ingredients → []`) para aislar SU tema; con el knob en el valor de producción el filtro de verificados los
dropea todos y no queda nada que medir. Ahora lo declaran ellos (`_f8_verified_only_off`) en vez de heredar el
`setdefault` global de `conftest.py`. Re-medida la suite CON base: 109 fallos en 34 ficheros → **19 en 17** de 25.332.

(a) los 35 declaran el knob, con motivo, y ninguno sigue en la lista general;
(b) la lista general es exactamente la del artefacto nuevo (el contrato lo fija `test_p1_plan_lote_33`) y baja a 17;
(c) la lista SIN BASE no se tocó: se mide en la CI, y aquí un árbol sin `frontend/` da falsos de layout;
(d) docs y marker ≥ 65.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_TESTS = _BACKEND / "tests"
_MARCA = "_f8_verified_only_off"
_GENERAL = _TESTS / "prod_profile_excluded.txt"
_SIN_BASE = _TESTS / "prod_profile_excluded_sin_base.txt"
_ARTEFACTO = _BACKEND / "scripts" / "data" / "f8_prod_profile_2026_09_16.json"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _ficheros(lista: Path) -> list:
    return [l.split()[0] for l in _src(lista).splitlines() if l.startswith("tests/")]


def _con_fixture() -> list:
    # la DEFINICIÓN del fixture, no su mención: este mismo fichero nombra la marca y no es un arnés
    return sorted(p for p in _TESTS.glob("test_*.py") if f"def {_MARCA}(" in _src(p))


# ─────────────────────────────── (a) el arnés declara lo que necesita

def test_los_arneses_declaran_el_knob_con_su_motivo():
    ficheros = _con_fixture()
    assert len(ficheros) == 35, [p.name for p in ficheros]
    for p in ficheros:
        src = _src(p)
        assert 'monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "false")' in src, p.name
        assert "@pytest.fixture(autouse=True)" in src, p.name
        assert "P1-PLAN-LOTE-65" in src, f"{p.name}: el fixture debe decir POR QUÉ lo necesita"


def test_ninguno_de_ellos_sigue_en_la_lista_general():
    con_fixture = {f"tests/{p.name}" for p in _con_fixture()}
    assert con_fixture & set(_ficheros(_GENERAL)) == set(), "declara el knob y sigue excluido: o sobra el fixture o sobra la línea"


# ─────────────────────────────── (b) la lista general baja con la medición

def test_la_lista_general_es_la_del_artefacto_nuevo():
    art = json.loads(_src(_ARTEFACTO))
    fallan = {f for f, d in art["ficheros"].items() if d["fallos"] > 0}
    assert art["con_base"] is True and art["corrida"]["tests"] > 25_000
    assert (art["corrida"]["fallos"], len(fallan)) == (19, 17), art["corrida"]
    assert set(_ficheros(_GENERAL)) == fallan
    assert _ARTEFACTO.name == sorted(p.name for p in _ARTEFACTO.parent.glob("f8_prod_profile_*.json"))[-1], \
        "el artefacto del lote debe ser el más reciente: es el que leen los tests del lote 33"


def test_lo_que_queda_son_knobs_de_conducta_no_arneses():
    art = json.loads(_src(_ARTEFACTO))
    for f, d in art["ficheros"].items():
        if d["fallos"] <= 0:
            continue
        a = d.get("atribucion") or {}
        assert a.get("veredicto") in {"knob", "combinacion"}, (f, a)
        assert "MEALFIT_VERIFIED_INGREDIENTS_ONLY" not in (a.get("knobs") or []) or a["veredicto"] == "combinacion", f


# ─────────────────────────────── (c) la lista sin base, intacta y con su razón

def test_la_lista_sin_base_no_se_toco_y_dice_quien_la_mide():
    sb = _ficheros(_SIN_BASE)
    assert len(sb) == 19
    cabecera = "\n".join(l for l in _src(_SIN_BASE).splitlines() if l.startswith("#"))
    assert "CI" in cabecera and "sin base" in cabecera.lower()
    assert sorted(p.name for p in (_BACKEND / "scripts" / "data").glob("f8_sin_base_*.json"))[-1] == \
        "f8_sin_base_2026_09_13.json", "si aparece una medición nueva sin base, esta lista tiene que re-medirse con ella"


# ─────────────────────────────── (d) docs y marker

def test_docs_y_marker():
    knobs = _src(_BACKEND / "docs" / "knobs_reference.md")
    assert "P1-PLAN-LOTE-65" in knobs and "19 fallos en 17 ficheros" in knobs
    assert "1.611 fallos en 344 ficheros" in knobs, "el falso rojo de layout queda escrito: es la trampa que costó una medición"
    assert "P1-PLAN-LOTE-65" in _src(_BACKEND / "docs" / "plan_pendientes_2026_09_11.md")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src(_BACKEND / "app.py"), re.M)
    assert m and int(m.group(1)) >= 65 and m.group(2) >= "2026-09-16"
    cabecera = "\n".join(l for l in _src(_GENERAL).splitlines() if l.startswith("#"))
    assert "P1-PLAN-LOTE-65" in cabecera and "19 fallos" in cabecera
