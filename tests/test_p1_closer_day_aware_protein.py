"""[P1-CLOSER-DAY-AWARE-PROTEIN · 2026-07-10] El chooser del closer de proteína
(`_close_protein_gap_for_meal`) evita INTRODUCIR una proteína cuyo label del gate
same-day ya está usado por OTRA comida del mismo día.

Root cause (renovación corr=2451c8ac): `_autofix_same_day_protein_repeats` limpió el
día a las 11:30:00, pero a las 11:30:03 el closer (rama P1-CLOSER-NO-DUP-CHEESE) eligió
'Huevo' como complemento para 'Tostadas de Queso Blanco...' — y el detector del gate
(`build_variety_report.same_day_protein_repeats`) escanea nombre+INGREDIENTES del estado
FINAL → rechazo "MISMA PROTEÍNA REPETIDA EL MISMO DÍA" a las 11:31:46 → retry completo
(day-gen escalado a PRO = costo). El autofix corre ANTES de los closers: cualquier
introducción tardía era invisible.

Fix: los callers del closer (FASE A repair + loop del solver) computan los labels de
proteína del gate presentes en las DEMÁS comidas del día y los pasan como
`day_used_proteins`; el chooser los evita en las ramas que INTRODUCEN proteína nueva
(categoría/fallback/_alt del no-dup-cheese). La rama de CONGRUENCIA no filtra (escala
una proteína que YA está en ESA comida → no crea label nuevo). El piso de proteína
SIEMPRE gana: si todos los candidatos colisionan, mantiene la elección original.

tooltip-anchor: P1-CLOSER-DAY-AWARE-PROTEIN
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Firma: el chooser acepta day_used_proteins
# ---------------------------------------------------------------------------

def test_chooser_signature_has_day_used_proteins():
    import graph_orchestrator as go
    sig = inspect.signature(go._close_protein_gap_for_meal)
    assert "day_used_proteins" in sig.parameters, (
        "P1-CLOSER-DAY-AWARE-PROTEIN: `_close_protein_gap_for_meal` debe aceptar "
        "`day_used_proteins` (labels del gate usados por OTRAS comidas del día). "
        "Sin él, el closer reintroduce proteínas repetidas same-day que el autofix "
        "ya limpió → rechazo del reviewer → retry en PRO (costo)."
    )
    assert sig.parameters["day_used_proteins"].default is None, (
        "day_used_proteins debe ser opcional (None = comportamiento legacy)"
    )


# ---------------------------------------------------------------------------
# 2. Helper de labels (SSOT con el detector del gate)
# ---------------------------------------------------------------------------

def test_gate_label_helper_exists_and_matches_gate_ssot():
    import graph_orchestrator as go
    fn = getattr(go, "_protein_gate_labels_in_text", None)
    assert callable(fn), (
        "P1-CLOSER-DAY-AWARE-PROTEIN: falta `_protein_gate_labels_in_text` — el "
        "helper que mapea un blob (nombre+ingredientes, lower/sin acento) a labels "
        "del gate usando el MISMO SSOT (_SAME_DAY_PROTEIN_GATE_LABELS + "
        "_MAIN_PROTEIN_ALIASES + _name_has_token). Detectores asimétricos = la clase "
        "de bug que causó el rechazo del 2026-07-10."
    )
    assert fn("revoltillo de huevos con tomate") == {"huevo"}
    assert fn("60g de pechuga de pollo cocido") == {"pollo"}
    # 'res' NO matchea dentro de 'fresas' (word-boundary del SSOT)
    assert fn("batido de fresas") == set()
    assert fn("ensalada de lechuga y tomate") == set()


# ---------------------------------------------------------------------------
# 3. Funcional: el chooser evita el label usado; None = legacy
# ---------------------------------------------------------------------------

def _mk_info(name, protein=20.0, carbs=1.0, fats=5.0, kcal=140.0):
    return SimpleNamespace(name=name, protein=protein, carbs=carbs, fats=fats, kcal=kcal)


def _mk_meal():
    return {
        "meal": "Merienda",
        "name": "Sándwich integral de vegetales",
        "protein": 5, "carbs": 30, "fats": 8, "cals": 212,
        "ingredients": ["2 rebanadas de pan integral", "1/2 taza de vegetales salteados"],
        "ingredients_raw": ["2 rebanadas de pan integral", "1/2 taza de vegetales salteados"],
        "recipe": ["MISE EN PLACE: Prepara los vegetales.",
                   "EL TOQUE DE FUEGO: Saltea 5 min.",
                   "MONTAJE: Arma el sándwich."],
    }


@pytest.fixture()
def _isolated_chooser(monkeypatch):
    import graph_orchestrator as go
    # aislar la unidad: sin scale-first ni line-merge (dependen de db real)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    return go


def test_functional_avoids_day_used_label(_isolated_chooser):
    go = _isolated_chooser
    candidates = [
        (1.0, "Huevo entero", _mk_info("Huevo entero", protein=13.0, kcal=155.0)),
        (2.0, "Yogurt griego", _mk_info("Yogurt griego", protein=19.0, kcal=97.0)),
    ]
    meal = _mk_meal()
    g = go._close_protein_gap_for_meal(
        meal, 22.0, None, candidates,
        day_used_proteins={"huevo"}, enforce_min_threshold=False)
    assert g > 0, "el closer debe seguir cerrando el piso (nunca sacrificarlo)"
    blob = " ".join(meal["ingredients"]).lower()
    assert "yogurt" in blob and "huevo" not in blob, (
        f"day_used_proteins={{'huevo'}} → debe elegir el candidato SIN colisión "
        f"(yogurt). ingredients={meal['ingredients']}"
    )


def test_functional_legacy_behavior_when_none(_isolated_chooser):
    go = _isolated_chooser
    candidates = [
        (1.0, "Huevo entero", _mk_info("Huevo entero", protein=13.0, kcal=155.0)),
        (2.0, "Yogurt griego", _mk_info("Yogurt griego", protein=19.0, kcal=97.0)),
    ]
    meal = _mk_meal()
    g = go._close_protein_gap_for_meal(
        meal, 22.0, None, candidates,
        day_used_proteins=None, enforce_min_threshold=False)
    assert g > 0
    blob = " ".join(meal["ingredients"]).lower()
    assert "huevo" in blob, (
        f"sin day_used_proteins el orden legacy manda (huevo primero). "
        f"ingredients={meal['ingredients']}"
    )


def test_functional_floor_wins_when_all_candidates_collide(_isolated_chooser):
    go = _isolated_chooser
    candidates = [
        (1.0, "Huevo entero", _mk_info("Huevo entero", protein=13.0, kcal=155.0)),
    ]
    meal = _mk_meal()
    g = go._close_protein_gap_for_meal(
        meal, 22.0, None, candidates,
        day_used_proteins={"huevo"}, enforce_min_threshold=False)
    assert g > 0, (
        "si TODOS los candidatos colisionan, el piso de proteína GANA (se mantiene "
        "la elección original; el gate graceful/autofix aguas abajo la maneja)"
    )
    blob = " ".join(meal["ingredients"]).lower()
    assert "huevo" in blob


# ---------------------------------------------------------------------------
# 4. Callers: ambos callsites del closer pasan day_used_proteins
# ---------------------------------------------------------------------------

def test_both_callsites_pass_day_context():
    calls = []
    idx = 0
    while True:
        i = _GO_SRC.find("_close_protein_gap_for_meal(", idx)
        if i < 0:
            break
        if "def _close_protein_gap_for_meal" not in _GO_SRC[max(0, i - 4):i + 40]:
            calls.append(_GO_SRC[i: i + 700])
        idx = i + 10
    assert len(calls) >= 2, f"se esperaban ≥2 callsites del closer, hay {len(calls)}"
    for c in calls:
        assert "day_used_proteins=" in c, (
            "P1-CLOSER-DAY-AWARE-PROTEIN: TODO callsite del closer debe pasar "
            f"day_used_proteins (labels de las demás comidas del día). Callsite sin "
            f"contexto:\n{c[:200]}"
        )


def test_no_dup_cheese_alt_picker_is_day_aware():
    i = _GO_SRC.find("[P1-CLOSER-NO-DUP-CHEESE] plato ya tiene queso")
    assert i > 0
    region = _GO_SRC[i - 2500: i]
    assert "day_used_proteins" in region, (
        "P1-CLOSER-DAY-AWARE-PROTEIN: el picker _alt del no-dup-cheese fue "
        "exactamente el que eligió 'Huevo' el 2026-07-10 reintroduciendo el repeat "
        "same-day — debe preferir alternativas cuyo label no esté usado en el día."
    )
