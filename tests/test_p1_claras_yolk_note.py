"""[P1-CLARAS-YOLK-NOTE + P1-WHOLE-EGGS-FIRST · 2026-07-30] "¿La otra parte del huevo se bota?"

Pregunta literal del owner ante una receta de '4 claras de huevo'. El motor elige claras por
aritmética legítima (proteína sin la grasa/kcal de la yema cuando la banda del día va apretada: 1
clara ≈ 3,6 g prot / 0 g grasa; 1 huevo entero suma ~5 g de grasa), pero la receta no decía qué
hacer con las 4 yemas. Dos capas:

1. Motor (determinista): receta de SOLO claras → nota sin numerar "no botes las yemas, guárdalas…".
2. Prompt (preferencia): huevos ENTEROS por defecto; claras solo para el excedente que la grasa del
   día no permite, y entonces la MEZCLA ("2 huevos + 2 claras") antes que claras puras.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_DG = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")


def _meal(ings, recipe=None):
    return {"name": "Revoltillo de Claras con Ñame", "meal": "cena",
            "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": recipe if recipe is not None else
            ["Mise en place: ralla el ñame.", "El Toque de Fuego: cuaja las claras 2-3 min.",
             "Montaje: sirve en plato llano."]}


# ─────────────── la nota ───────────────

def test_receta_de_solo_claras_recibe_la_nota():
    m = _meal(["4 claras de huevo", "½ pedazo de ñame (≈147 g)", "Sal al gusto"])
    assert go._note_claras_save_yolks(m) == 1
    notas = [s for s in m["recipe"] if "yemas" in str(s).lower()]
    assert len(notas) == 1
    assert "NO botes" in notas[0] and "Nota del Nutricionista AI" in notas[0], (
        "el prefijo es lo que el frontend (P2-RECIPE-NOTES-NOT-STEPS) deja SIN numerar")
    # va antes del Montaje, no después de emplatar
    idx_nota = m["recipe"].index(notas[0])
    idx_montaje = next(i for i, s in enumerate(m["recipe"]) if "montaje" in str(s).lower())
    assert idx_nota < idx_montaje


def test_mezcla_con_huevo_entero_no_lleva_nota():
    """'2 huevos + 2 claras' ya consume ambas partes: la nota sobraría."""
    m = _meal(["2 huevos", "2 claras de huevo", "½ cebolla"])
    assert go._note_claras_save_yolks(m) == 0


def test_sin_claras_no_hay_nota():
    for ings in (["3 huevos", "1 taza de arroz"], ["150 g de pollo"], []):
        assert go._note_claras_save_yolks(_meal(ings)) == 0


def test_idempotente():
    m = _meal(["4 claras de huevo", "Sal al gusto"])
    assert go._note_claras_save_yolks(m) == 1
    assert go._note_claras_save_yolks(m) == 0, "segunda pasada no duplica la nota"
    assert sum("yemas" in str(s).lower() for s in m["recipe"]) == 1


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(go, "CLARAS_YOLK_NOTE_ENABLED", False)
    m = _meal(["4 claras de huevo"])
    assert go._note_claras_save_yolks(m) == 0


def test_cableado_en_ambas_surfaces():
    """Mismo par de callsites que el cheese-display pass: plan-level (form-gen/chunks) y el
    finalizador single-meal (swap/chat-modify). Si falta uno, la nota existe solo a veces."""
    assert _GO.count("_note_claras_save_yolks(_m)") == 1, "falta el callsite plan-level"
    assert _GO.count("_note_claras_save_yolks(meal)") == 1, "falta el callsite del update finalizer"


# ─────────────── la preferencia del prompt ───────────────

def test_regla_huevos_enteros_primero():
    i = _DG.index("P1-WHOLE-EGGS-FIRST")
    seg = _DG[i:i + 600]
    assert "huevos ENTEROS por defecto" in seg
    assert "2 huevos + 2 claras" in seg, "la regla debe dar la forma preferida, no solo prohibir"
    assert "excedente de proteína" in seg, (
        "las claras siguen PERMITIDAS cuando la grasa del día no da — la regla es preferencia, "
        "no prohibición: sin esa válvula el solver no puede cerrar proteína en días apretados")
