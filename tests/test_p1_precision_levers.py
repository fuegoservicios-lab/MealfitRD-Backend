"""[P1-PRECISION-LEVERS · 2026-07-04] Los 3 levers de precisión post-monitoreo en vivo.

Evidencia (generaciones reales monitoreadas 2026-07-04):
  1. SODIO: un intento salió con 4,261 mg/día → gate lo rechazó → retry completo pagado.
     El prompt no tenía números. → §17 con presupuesto CUANTITATIVO (estático, cache-safe).
  2. CROSS-DÍA: "revoltillo ×3" y "salteado ×3" — los días se generan en PARALELO y no se
     ven entre sí; solo el retry del gate de variedad lo frenaba. → el dispatch inyecta
     `_other_days_brief` (técnica + desayuno de los otros días) y el builder lo convierte
     en instrucción negativa explícita.
  3. MAX_ATTEMPTS 2→3 en prod: cambio de ENV en el VPS (MEALFIT_MAX_ATTEMPTS), no de
     código — documentado aquí; el default de código ya era 3.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")
_DG = _read(os.path.join("prompts", "day_generator.py"))


# ---------------------------------------------------------------------------
# lever 1 · presupuesto de sodio en el system prompt (estático)
# ---------------------------------------------------------------------------

def test_sodium_budget_in_system_prompt():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as _P
    assert "PRESUPUESTO DE SODIO" in _P
    assert "2000 mg" in _P, "el techo OMS debe ser un número explícito, no prosa"
    assert "UN ítem enlatado" in _P, "regla de conteo accionable (los enlatados eran el driver medido)"
    assert "cubitos" in _P, "el cubito/sazón completo es el reventador típico del presupuesto RD"


def test_sodium_block_is_static_append():
    # Import-time append (string estático) → prompt-cache del SystemMessage intacto.
    i = _DG.index("17. PRESUPUESTO DE SODIO")
    blk = _DG[max(0, i - 900):i]
    assert "P1-PRECISION-LEVERS" in blk
    assert "import-time" in blk or "estático" in blk.lower() or "ESTÁTICO" in blk


# ---------------------------------------------------------------------------
# lever 2 · anti-repetición cross-día (brief inyectado + builder)
# ---------------------------------------------------------------------------

def _mk_skel_day(with_brief=True):
    sd = {
        "brief_concept": "Criollo ligero",
        "assigned_technique": "guisado",
        "breakfast_category": "avena",
        "protein_pool": ["Pollo", "Huevos"],
        "meal_types": ["Desayuno", "Almuerzo", "Cena"],
    }
    if with_brief:
        sd["_other_days_brief"] = [
            {"technique": "salteado", "breakfast": "revoltillo"},
            {"technique": "al horno", "breakfast": "mangú"},
        ]
    return sd


def test_builder_emits_cross_day_block():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_mk_skel_day(), 2)
    assert "ANTI-REPETICIÓN ENTRE DÍAS" in ctx
    assert "salteado" in ctx and "al horno" in ctx
    assert "revoltillo" in ctx, "el desayuno de los otros días debe ser visible"
    # determinista (prompt-cache per-día estable).
    assert ctx == build_day_assignment_context(_mk_skel_day(), 2)


def test_builder_backcompat_without_brief():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_mk_skel_day(with_brief=False), 2)
    assert "ANTI-REPETICIÓN ENTRE DÍAS" not in ctx, \
        "sin brief inyectado el bloque no aparece (fixtures/paths legacy intactos)"


def test_dispatch_injects_other_days_brief():
    i = _GO.index('_sd["_other_days_brief"]') if '_sd["_other_days_brief"]' in _GO else _GO.index('_abd_sd["_other_days_brief"]')
    win = _GO[max(0, i - 1600):i + 800]
    assert "assigned_technique" in win and "breakfast_category" in win
    # corre ANTES del dispatch paralelo (el candidato lo consume).
    assert _GO.index('_abd_sd["_other_days_brief"]') < _GO.index("async def _generate_candidate")


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------

def test_marker_anchored_in_source():
    # NO pineamos _LAST_KNOWN_PFIX (cada P-fix posterior lo bumpea y rompería en cadena);
    # el contrato marker↔test lo enforza test_p2_hist_audit_14. Anclamos el marker en el SOURCE.
    assert "P1-PRECISION-LEVERS" in _GO and "P1-PRECISION-LEVERS" in _DG
