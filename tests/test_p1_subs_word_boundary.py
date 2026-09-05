# -*- coding: utf-8 -*-
"""[P1-SUBS-WORD-BOUNDARY · 2026-09-05] La sustitución por dieta comparaba sus tokens por SUBCADENA, y sus
tokens son cortos. Medido sobre el plan vivo 18326457 (vegetariano):

    «res»   ⊂ queso f-res-co · f-res-as · ciruelas f-res-cas · berenjena f-res-ca
    «pollo» ⊂ re-pollo
    «mero»  ⊂ nú-mero

Eso «sustituyó» 4 comidas del intento 1 y 2 del intento 2: el queso fresco de un vegetariano cambiado por soya
texturizada, con sus macros y su receta detrás, y el ingrediente «4 ciruelas Soya texturizada» que lo destapó.
Es la decimotercera vez que esta clase de fallo aparece en el repo («sal»⊂«salsa», «pollo»⊂«repollo»,
«res»⊂«fresco»), así que el ancla no es el token: es que el modo de palabra completa siga puesto."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402

_VEGETARIANO = {"dietType": "vegetariana"}


def _ings_tras_sustituir(*ingredientes):
    plan = {"days": [{"meals": [{"name": "Plato", "ingredients": list(ingredientes),
                                 "ingredients_raw": list(ingredientes)}]}]}
    go._apply_diet_substitutions(plan, _VEGETARIANO)
    return plan["days"][0]["meals"][0]["ingredients"]


@pytest.mark.parametrize("inocente", [
    "150 g de queso fresco",
    "100 g de queso fresco rallado",
    "4 ciruelas frescas",
    "200 g de fresas",
    "1 berenjena fresca",
    "1 taza de repollo",
    "2 tazas de repollo morado",
    "1 número de porciones",
])
def test_la_comida_inocente_sobrevive(inocente):
    """Ninguno de estos es carne ni pescado; todos casaban antes por subcadena."""
    assert _ings_tras_sustituir(inocente) == [inocente]


@pytest.mark.parametrize("carne, esperado", [
    ("120 g de pechuga de pollo", "Soya texturizada"),
    ("150 g de carne de res", "Soya texturizada"),
    ("2 lonjas de jamón", "Soya texturizada"),
    ("100 g de pavo molido", "Soya texturizada"),
    ("120 g de tilapia", "Garbanzos cocidos"),
    ("150 g de camarones", "Garbanzos cocidos"),
    ("1 lata de atún", "Garbanzos cocidos"),
])
def test_la_carne_y_el_pescado_siguen_cayendo(carne, esperado):
    out = _ings_tras_sustituir(carne)[0]
    assert esperado in out, f"«{carne}» debía sustituirse y quedó «{out}»"


def test_el_plural_sigue_casando():
    """El sufijo `(?:e?s)?` conserva lo que la subcadena daba gratis: reses, camarones, pavos."""
    for plural in ("200 g de reses", "150 g de camarones", "2 pavos"):
        assert "texturizada" in _ings_tras_sustituir(plural)[0] or "Garbanzos" in _ings_tras_sustituir(plural)[0]


def test_solo_la_dieta_usa_palabra_completa():
    """Los tokens de condición y alérgenos son largos y NECESITAN la subcadena: no se les cambia el modo."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _apply_diet_substitutions")
    assert '"word_match": True' in src[i:i + 2500], "la dieta pide palabra completa"
    j = src.index("def _apply_allergen_substitutions")
    assert '"word_match"' not in src[j:j + 2000], "los alérgenos conservan la subcadena (sobre-detectar es seguro)"


def test_el_veto_sigue_siendo_por_subcadena():
    """Un negativo es un veto: ahí sobre-detectar es lo seguro, así que NO se le pone frontera."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _match(ing_norm):", src.index("def _apply_substitutions_core"))
    cuerpo = src[i:i + 600]
    assert "any(neg in ing_norm for neg in" in cuerpo
    assert "carne vegetal" not in cuerpo  # el veto vive en la tabla, no aquí
    assert _ings_tras_sustituir("100 g de carne vegetal de soya") == ["100 g de carne vegetal de soya"]


def test_el_cerrador_del_coach_mira_la_dieta():
    """Los cinco call sites del generador ya pasaban `diet=`; el del swap del coach se había quedado atrás."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("_cands = _safe_high_density_proteins(allergies, _cl_db")
    assert "diet=diet_type" in src[i:i + 200], "sin dieta, el coach le mete pollo al plato de un vegetariano"
