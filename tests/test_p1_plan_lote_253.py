# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-253 · 2026-09-25] Las comidas que el revisor le sugiere al corrector respetan el perfil.

Batería rd252 (alérgico a lácteos y mariscos): «SOBREUSO DE HUEVO … reemplaza el huevo por … queso de freír, yogur
griego …». Los textos de crudos y de transformación proponían «revoltillo» y «panqueques de avena» sin mirar a quién.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _huevo(country, fd):
    return go._review_country_feedback(country, "egg_overuse", egg=6, total=12, cap=3, form_data=fd)


def test_alergico_a_lacteos_y_mariscos_sin_queso_ni_yogur():
    t = _huevo("DO", {"allergies": ["Lacteos", "Mariscos"]})
    assert "queso" not in t and "yogur" not in t, t
    assert "pollo guisado" in t and "pescado" in t and "habichuelas" in t


def test_alergico_al_pescado_sin_pescado_atun_ni_sardina():
    t = _huevo("DO", {"allergies": ["Pescado"]})
    assert not any(x in t for x in ("pescado", "atún", "sardina")), t


def test_vegetariana_y_vegana():
    t = _huevo("DO", {"dietType": "vegetarian"})
    assert not any(x in t for x in ("pollo", "pescado", "atún", "sardina", "res molida")), t
    assert "queso de freír" in t and "habichuelas" in t
    t = _huevo("DO", {"dietType": "vegan"})
    assert "queso" not in t and "yogur" not in t and "habichuelas" in t, t
    t = _huevo("ES", {"dietType": "vegetarian"})
    assert "aves" not in t and "pescado" not in t and "legumbres" in t, t


def test_rechazo_escrito_a_mano():
    # (con el chip «Ninguno» el formulario bloquea el texto libre y el backend lo descarta: P0-FORM-1)
    t = _huevo("DO", {"dislikes": ["Cilantro"], "otherDislikes": "habichuelas"})
    assert "habichuelas" not in t, t


def test_revoltillo_y_panqueques():
    raw = go._review_country_feedback("DO", "raw_staples", count=3, sample="x", form_data={"allergies": ["Huevo"]})
    assert "revoltillo" not in raw and "guisos" in raw, raw
    tr = go._review_country_feedback("DO", "transform_minimum", form_data={"allergies": ["Gluten"]})
    assert "panqueques de avena" not in tr and "guiso" in tr, tr
    tr = go._review_country_feedback("MX", "transform_minimum", form_data={"dietType": "vegan"})
    assert "revoltillo" not in tr and tr.count(" o ") <= 1, tr


def test_sin_perfil_el_texto_de_siempre():
    # El byte a byte de P1-REVIEW-RETRY-FEEDBACK-DO sigue en su test; aquí, que un perfil sin restricciones no cambia nada.
    for kind, kw in (("egg_overuse", dict(egg=6, total=12, cap=3)), ("raw_staples", dict(count=2, sample="x")),
                     ("transform_minimum", {})):
        for country in ("DO", "ES"):
            base = go._review_country_feedback(country, kind, **kw)
            assert go._review_country_feedback(country, kind, form_data={"allergies": ["Ninguna"]}, **kw) == base


def test_el_revisor_pasa_el_perfil():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    for kind in ("egg_overuse", "raw_staples", "transform_minimum"):
        assert f'"{kind}", form_data=form_data,' in src, kind
    assert "P1-PLAN-LOTE-253-SUGERENCIAS" in (_BACKEND / "sugerencias_perfil.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 253
