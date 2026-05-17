"""[P2-PROTEIN-VIOLATION-SANITIZE · 2026-05-16] Regression guard: cuando el
bounded regen NO logra eliminar las menciones de proteínas prohibidas en el
`name` o `recipe` de un meal, el sistema DEBE aplicar sanitización defensiva
(text replacement) en lugar de aceptar silenciosamente.

Bug observado en plan_id=`fbd014b2-594d-4ad9-aa08-db7bf027a099` (2026-05-16 02:08:52 → 02:15:38):
  - LLM eligió "Chuleta al Airfryer con Tostones" para Día 2.
  - PROTEIN-RECIPE-VIOLATION strippeó chuleta del ingredients.
  - Bounded regen falló (LLM repitió chuleta).
  - Sistema aceptó "para evitar loop" → meal final tenía:
      name="Chuleta al Airfryer..." (mentía sobre la chuleta)
      ingredients=[tostones, salsa, ...] (sin chuleta)
  - Reviewer médico rechazó: "almuerzo sin proteína adecuada".

Fix: helper `_sanitize_unauthorized_protein_text` que reemplaza las palabras
prohibidas en name/recipe por placeholders neutros + marca el meal con
`_protein_violation_sanitized` para audit.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


def test_sanitize_helper_exists():
    """`_sanitize_unauthorized_protein_text` debe estar definido como función
    independiente para que pueda ser testeada/reutilizada."""
    text = _read_graph()
    assert "def _sanitize_unauthorized_protein_text(" in text, (
        "Falta helper `_sanitize_unauthorized_protein_text` en "
        "graph_orchestrator.py. P2-PROTEIN-VIOLATION-SANITIZE requiere este "
        "helper para cerrar el gap del LLM que persiste con proteína prohibida."
    )


def test_helper_returns_replacements_count():
    """El helper debe retornar count (int) para que el caller pueda loguear "
    "el alcance del fix."""
    text = _read_graph()
    m = re.search(
        r"def _sanitize_unauthorized_protein_text\([^)]*\)\s*->\s*int",
        text,
    )
    assert m, (
        "`_sanitize_unauthorized_protein_text` debe declarar return type `int` "
        "(count of replacements). Sin esto, el log del caller no puede "
        "comunicar el alcance del fix."
    )


def test_sanitize_invoked_post_failed_regen():
    """El caller (en `generate_days_parallel_node` o similar) debe invocar "
    "el sanitize CUANDO el bounded regen falla — no antes (sería overkill) "
    "ni nunca (sería el bug pre-fix)."""
    text = _read_graph()
    # Buscar el callsite que invoca el sanitize después de _violations_post > 0.
    pattern = re.compile(
        r"if\s+_violations_post\s*>\s*0\s*:.*?_sanitize_unauthorized_protein_text\(",
        re.DOTALL,
    )
    assert pattern.search(text), (
        "El sanitize NO se invoca dentro de `if _violations_post > 0:` "
        "(post-regen-failure path). Sin esto, el fix NO se aplica en el caso "
        "real del bug."
    )


def test_meal_marked_with_audit_trail():
    """Cada meal sanitizado debe quedar marcado con `_protein_violation_sanitized` "
    "para que downstream (reviewer, persist) tenga audit trail."""
    text = _read_graph()
    fn_match = re.search(
        r"def _sanitize_unauthorized_protein_text\([^)]*\)[^:]*:(.*?)(?=^def |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match, "No se encontró cuerpo de `_sanitize_unauthorized_protein_text`."
    body = fn_match.group(1)
    assert "_protein_violation_sanitized" in body, (
        "Helper NO marca el meal con `_protein_violation_sanitized`. "
        "Sin audit trail, debugging post-incidente es opaco."
    )


def test_fallback_name_per_meal_type():
    """Si el name queda vacío/muy corto tras strip de la palabra prohibida, "
    "el helper debe usar fallback por meal_type (Desayuno/Almuerzo/Merienda/Cena)."""
    text = _read_graph()
    fn_match = re.search(
        r"def _sanitize_unauthorized_protein_text\([^)]*\)[^:]*:(.*?)(?=^def |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match
    body = fn_match.group(1)
    for meal_type in ("desayuno", "almuerzo", "merienda", "cena"):
        assert meal_type in body.lower(), (
            f"Fallback para meal_type=`{meal_type}` no presente en helper. "
            f"Sin fallback, un meal con name='Chuleta' (1 palabra) queda con "
            f"name=' ' (vacío) tras strip."
        )


def test_recipe_steps_also_sanitized():
    """Los steps del array `recipe` también deben sanitizarse (no solo el "
    "name). El reviewer médico lee ambos."""
    text = _read_graph()
    fn_match = re.search(
        r"def _sanitize_unauthorized_protein_text\([^)]*\)[^:]*:(.*?)(?=^def |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match
    body = fn_match.group(1)
    assert "recipe" in body, (
        "Helper NO procesa el array `recipe` del meal. El reviewer escanea "
        "tanto name como recipe — sanitizar solo name deja inconsistencias."
    )
    # Verificar que hay placeholder para steps (no solo strip total).
    assert "ingrediente alternativo" in body or "alternativ" in body.lower(), (
        "Helper no usa placeholder en recipe steps. Strip total puede romper "
        "la gramática del paso; placeholder ('ingrediente alternativo') "
        "preserva la estructura."
    )
