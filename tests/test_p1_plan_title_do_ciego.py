"""[P1-PLAN-TITLE-DO-CIEGO · 2026-08-23]

El título es parte visible del plan y debe usar el país estampado. DO conserva
su prompt histórico; los países beta no reciben referencias dominicanas.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]
_BETA = ("ES", "MX", "CO", "PR", "US")
_DOMINICAN_EXAMPLES = (
    "Energía Tropical al Máximo",
    "Fuerza y Balance Criollo",
    "Ruta Fit Dominicana",
)


def _plan(country=None):
    data = {
        "days": [{"day": 1, "meals": [{"name": "Avena con Frutas"}]}],
        "calories": 2100,
        "goal": "maintain",
    }
    if country is not None:
        data["_country"] = country
    return data


def _render_title_prompt(monkeypatch, country=None) -> str:
    import ai_helpers

    captured = []

    class FakeResponse:
        content = "Energía Serena"

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, prompt):
            captured.append(prompt)
            return FakeResponse()

    monkeypatch.setattr(ai_helpers, "ChatDeepSeek", FakeLLM)
    assert ai_helpers.generate_plan_title(_plan(country)) == "Energía Serena"
    assert len(captured) == 1
    return captured[0]


def test_do_y_plan_legacy_conservan_las_dos_lineas_historicas_byte_a_byte(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    legacy = _render_title_prompt(monkeypatch, None)
    dominicano = _render_title_prompt(monkeypatch, "DO")

    assert dominicano == legacy
    assert "- Puede ser metafórico o usar referencias dominicanas sutiles" in dominicano
    assert (
        '- Ejemplos de buenos títulos: "Energía Tropical al Máximo", "Sabor Sin Culpa", '
        '"Fuerza y Balance Criollo", "Combustible Para Tu Meta", "Ruta Fit Dominicana", '
        '"Poder Verde y Proteína"'
    ) in dominicano


@pytest.mark.parametrize("country", _BETA)
def test_titulo_beta_nombra_su_contexto_y_no_hereda_referencias_do(monkeypatch, country):
    from constants import COUNTRY_PROFILES

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    prompt = _render_title_prompt(monkeypatch, country)

    assert COUNTRY_PROFILES[country]["name_es"] in prompt
    assert not re.search(r"dominican\w*", prompt, flags=re.IGNORECASE)
    for example in _DOMINICAN_EXAMPLES:
        assert example not in prompt


def test_marker_movil_del_gap():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    helpers = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
    assert "P1-PLAN-TITLE-DO-CIEGO" in helpers
