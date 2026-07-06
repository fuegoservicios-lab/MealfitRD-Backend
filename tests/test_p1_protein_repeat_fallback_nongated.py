"""[P1-PROTEIN-REPEAT-FALLBACK-NONGATED + P1-PROTEIN-REPEAT-ATUN · 2026-07-06] Residual del gate
same-day-protein medido EN VIVO en el plan del owner (semana 2): un día con "Tilapia Salteada"
(pescado) + "Pechuga de pollo al Vapor" + "Pollo en Salsa de Vegetales" → pollo×2, pero el
pescado ya estaba tomado por la tilapia y pavo/cerdo/res fuera (dislikes) → la escalera de carnes
se agotaba (`tgt None`) → el autofix no reasignaba → 3 intentos quemados en esa semana.

Fix: cuando la escalera de CARNES se agota, cae a proteínas NO-gated (legumbres/queso), que no
cuentan en el gate same-day y son platos DR coherentes. Diet-aware (vegano excluye queso). Además,
atún dejó de ser backstop-only (ahora tiene escalera de swap propia).
"""
import pytest

import graph_orchestrator as go


class _StubDB:
    def macros_from_ingredient_string(self, s):
        return None


@pytest.fixture()
def _go(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return go


def _meal(name, ings):
    return {"name": name, "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": [f"Prepara {name}."]}


def _count(day, label):
    import re
    from constants import strip_accents as sa
    aliases = go._MAIN_PROTEIN_ALIASES.get(label, (label,))
    n = 0
    for m in day["meals"]:
        blob = sa((str(m.get("name", "")) + " "
                   + " ".join(str(i) for i in m.get("ingredients") or [])).lower())
        if any(re.search(r"\b" + re.escape(sa(a)) + r"\b", blob) for a in aliases):
            n += 1
    return n


# ─────────── el caso vivo: carnes agotadas → legumbre ───────────

def test_ladder_exhausted_falls_to_legume(_go):
    days = [{"day": 1, "meals": [
        _meal("Tilapia Salteada con Piña y Quinoa", ["150 g de filete de tilapia", "quinoa"]),
        _meal("Pechuga de pollo al Vapor con Ensalada", ["150 g de pechuga de pollo"]),
        _meal("Pollo en Salsa de Vegetales Naturales", ["150 g de pechuga de pollo"]),
    ]}]
    fd = {"dislikes": ["pavo", "cerdo", "res"]}
    n = _go._protein_repeat_autofix(days, fd, db=_StubDB())
    assert n == 1, "el 2º pollo se reasigna al fallback (pescado tomado, pavo/cerdo/res fuera)"
    assert _count(days[0], "pollo") == 1, "pollo queda en 1 comida → gate pasa"
    _swapped = days[0]["meals"][2]
    assert "habichuela" in _swapped["name"].lower() or "lenteja" in _swapped["name"].lower() \
        or "queso" in _swapped["name"].lower(), f"cayó a proteína no-gated: {_swapped['name']}"


def test_heavy_target_preferred_over_fallback(_go):
    # pavo disponible (no disliked) → usa pavo, NO el fallback legumbre.
    days = [{"day": 1, "meals": [
        _meal("Pechuga de pollo al Vapor", ["150 g de pechuga de pollo"]),
        _meal("Pollo Guisado con Arroz", ["150 g de pechuga de pollo"]),
    ]}]
    assert _go._protein_repeat_autofix(days, {}, db=_StubDB()) == 1
    assert days[0]["meals"][1]["_protein_autofix_applied"] == "pollo->pavo", \
        "con pavo disponible, NO se usa el fallback no-gated"


def test_vegan_excludes_queso_from_fallback(_go):
    # vegano + carnes agotadas → legumbre sí, queso NO.
    days = [{"day": 1, "meals": [
        _meal("Pescado al Horno", ["150 g de filete de pescado blanco"]),
        _meal("Pollo a la Plancha", ["150 g de pechuga de pollo"]),
        _meal("Pollo Guisado", ["150 g de pechuga de pollo"]),
    ]}]
    fd = {"dislikes": ["pavo", "cerdo", "res"], "dietType": "vegano"}
    _go._protein_repeat_autofix(days, fd, db=_StubDB())
    _swapped = days[0]["meals"][2]
    assert "queso" not in _swapped["name"].lower(), f"vegano: queso fuera del fallback: {_swapped['name']}"


def test_legume_allergy_falls_to_queso(_go):
    days = [{"day": 1, "meals": [
        _meal("Pescado al Horno", ["150 g de filete de pescado blanco"]),
        _meal("Pollo a la Plancha", ["150 g de pechuga de pollo"]),
        _meal("Pollo Guisado", ["150 g de pechuga de pollo"]),
    ]}]
    # dislikes bloquean carnes + legumbres → único fallback seguro = queso
    fd = {"dislikes": ["pavo", "cerdo", "res", "habichuela", "lenteja"]}
    _go._protein_repeat_autofix(days, fd, db=_StubDB())
    _swapped = days[0]["meals"][2]
    assert "queso" in _swapped["name"].lower(), f"legumbres fuera → cae a queso: {_swapped['name']}"


def test_knob_off_reverts_to_gate(_go, monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_REPEAT_FALLBACK_NONGATED", False)
    days = [{"day": 1, "meals": [
        _meal("Tilapia Salteada", ["150 g de filete de tilapia"]),
        _meal("Pechuga de pollo al Vapor", ["150 g de pechuga de pollo"]),
        _meal("Pollo en Salsa", ["150 g de pechuga de pollo"]),
    ]}]
    fd = {"dislikes": ["pavo", "cerdo", "res"]}
    assert _go._protein_repeat_autofix(days, fd, db=_StubDB()) == 0, \
        "knob off → escalera de carnes agotada → gate/retry (comportamiento previo)"


def test_marker_anchored_in_source():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-PROTEIN-REPEAT-FALLBACK-NONGATED" in src
