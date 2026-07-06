"""[P1-EGG-CLARAS-PHRASE · 2026-07-06] Residual del gate same-day-protein medido en vivo tras
P1-EGG-PROTAGONIST-SURPLUS: un día con "Tortilla de Claras con Queso Parmesano" (intrínseca) +
"Bowl de Claras Revueltas con Berro Salteado" el mismo día → el gate seguía viendo HUEVO×2.

Dos huecos del fix anterior:
1. Solo reconocía "huevo", no 'claras'/'yema' (aliases de huevo del gate).
2. "Bowl de Claras Revueltas" no es intrínseco (cabeza 'bowl') pero el `_strip_egg_from_name`
   previo solo reconocía el patrón "con huevo", no "de Claras".

Fix: `_replace_egg_phrase_in_name` reemplaza la FRASE de huevo NOMBRADA (sustantivo
huevo/claras/yema + modo de cocción opcional) IN-PLACE por la proteína nueva → "Bowl de Queso
blanco con Berro" (limpio, sin manglear el multi-componente).
"""
import pytest

import graph_orchestrator as go


@pytest.fixture()
def _go(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return go


def _meal(slot, name, ings):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": [f"Prepara {name}."]}


def _count_egg_meals(day):
    import re
    from constants import strip_accents as sa
    cnt = 0
    for m in day["meals"]:
        blob = sa((str(m.get("name", "")) + " "
                   + " ".join(str(i) for i in m.get("ingredients") or [])).lower())
        if re.search(r"\bhuevos?\b|\bclaras?\b", blob):
            cnt += 1
    return cnt


# ─────────── reemplazo de frase in-place ───────────

def test_replace_egg_phrase_huevo():
    assert go._replace_egg_phrase_in_name(
        "Puré de Batata con Huevos Revueltos", "Queso blanco") == "Puré de Batata con Queso blanco"


def test_replace_egg_phrase_claras_de_form():
    assert go._replace_egg_phrase_in_name(
        "Bowl de Claras Revueltas con Berro", "Queso blanco") == "Bowl de Queso blanco con Berro"


def test_replace_egg_phrase_yema_and_duro():
    assert go._replace_egg_phrase_in_name(
        "Ensalada Fresca con Huevo Duro", "Pechuga de pollo") == "Ensalada Fresca con Pechuga de pollo"


def test_replace_egg_phrase_none_when_no_egg():
    assert go._replace_egg_phrase_in_name("Croquetas de Yuca", "Queso blanco") is None


# ─────────── el caso vivo end-to-end ───────────

def test_claras_bowl_reassigned_keeps_intrinsic_tortilla(_go):
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Tortilla de Claras con Queso Parmesano",
              ["4 claras", "30 g de queso parmesano"]),
        _meal("Cena", "Bowl de Claras Revueltas con Berro Salteado",
              ["3 claras", "50 g de berro"]),
    ]}]
    assert _go._protein_repeat_autofix(days, {}, db=object()) == 1
    assert _count_egg_meals(days[0]) == 1, "el día queda con 1 comida-huevo (la tortilla)"
    _bowl = days[0]["meals"][1]
    assert "clara" not in _bowl["name"].lower() and "huevo" not in _bowl["name"].lower(), (
        f"'Bowl de Claras' renombrado a la proteína nueva: {_bowl['name']}"
    )
    assert "clara" not in " ".join(_bowl["ingredients"]).lower()
    assert "Tortilla de Claras" in days[0]["meals"][0]["name"], "tortilla intrínseca intacta"


def test_two_intrinsic_claras_left_to_gate(_go):
    # 'Tortilla de Claras' + 'Revoltillo de Claras' — ambos intrínsecos → gate/retry (identidad).
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Tortilla de Claras", ["4 claras"]),
        _meal("Cena", "Revoltillo de Claras con Espinaca", ["3 claras", "espinaca"]),
    ]}]
    assert _go._protein_repeat_autofix(days, {}, db=object()) == 0
    assert _count_egg_meals(days[0]) == 2


def test_marker_anchored_in_source():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-EGG-CLARAS-PHRASE" in src
