"""[P1-CLOSER-LINE-CLAMP · 2026-07-05] Techo ABSOLUTO de gramos por línea escalada del closer.

Forense del plan fdfeba33 (banner "1 de 3 días se pasa del techo, peor: Día 2"): la línea
"47.5 tallos de apio" (≈1.9 kg, 1,520 mg de sodio — 49% del sodio del día) nació del escalado
COMPUESTO del closer de micros: el factor por-pasada (MAX_SCALE ~1.6) se re-aplica en cada
pasada (pre-motor + recheck P2-2 + assemble post-surgical) y los días RECICLADOS entre attempts
re-pasan por todo → 1.6^N sin techo. Además de romper el techo de sodio, produce recetas
absurdas (nadie corta 47 tallos de apio).

El clamp corta el crecimiento VITALICIO sin tocar la lógica por-pasada: una línea escalada
jamás supera su techo físico — semillas/frutos secos 40 g (densos), resto
MICRO_CLOSER_LINE_MAX_G (250 g default ≈ 6 tallos de apio ≈ 190 mg de sodio, inocuo).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _FakeDB:
    """apio: potasio por línea proporcional a gramos; grams parsea el prefijo 'NN g'."""

    def micros_from_ingredient_string(self, s):
        import re
        low = str(s).lower()
        if "apio" in low:
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
            g = float(m.group(1).replace(",", ".")) if m else 40.0
            return {"potassium_mg": 2.6 * g}
        return {"potassium_mg": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 10.0}  # apio: kcal despreciable → el budget kcal NO frena el escalado

    def grams_from_ingredient_string(self, s):
        import re
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else None

    def rescale_ingredient_string(self, s, factor):
        import re
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return s
        g = float(m.group(1).replace(",", ".")) * factor
        return re.sub(r"^\s*\d+(?:[.,]\d+)?\s*g", f"{g:.0f} g", s, count=1)


def _mk_plan(apio_g=200):
    return {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada",
         "ingredients": [f"{apio_g} g de apio", "50 g de lechuga"],
         "ingredients_raw": [f"{apio_g} g de apio", "50 g de lechuga"],
         "recipe": ["Sirve."]},
    ]}]}


def _run(go, monkeypatch, plan, piso=4000.0):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": "potassium_mg", "piso": piso, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": "potassium_mg", "piso": piso, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    go._close_micro_gaps_for_plan(plan, {}, _FakeDB())
    return plan


def _apio_g(plan):
    import re
    line = next(s for s in plan["days"][0]["meals"][0]["ingredients"] if "apio" in s)
    return float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))


# ---------------------------------------------------------------------------

def test_knob_default():
    assert '_env_int("MEALFIT_MICRO_CLOSER_LINE_MAX_G", 250' in _GO


def test_scaling_clamped_to_absolute_cap(go, monkeypatch):
    # 200 g de apio, piso ENORME (4000 mg K) → sin clamp el factor 1.6 lo lleva a 320 g;
    # con clamp queda en ≤250 g.
    plan = _run(go, monkeypatch, _mk_plan(apio_g=200))
    assert _apio_g(plan) <= 250.0 + 0.5, f"la línea escaló sobre el techo: {_apio_g(plan)} g"


def test_line_at_cap_not_scaled_further(go, monkeypatch):
    # 260 g (ya sobre el cap) → NO se escala más (el compuesto entre pasadas queda cortado).
    plan = _run(go, monkeypatch, _mk_plan(apio_g=260))
    assert _apio_g(plan) == 260.0, "línea en su techo vitalicio → intocada"


def test_normal_scaling_below_cap_unaffected(go, monkeypatch):
    # 100 g con deficit chico → escala normal (el clamp no interfiere bajo el techo).
    plan = _run(go, monkeypatch, _mk_plan(apio_g=100), piso=300.0)
    g = _apio_g(plan)
    assert 100.0 < g <= 160.0, f"escala por-pasada normal esperada (~115 g): {g} g"


def test_seed_lines_have_tighter_cap():
    i = _GO.index("[P1-CLOSER-LINE-CLAMP · 2026-07-05] techo ABSOLUTO")
    win = _GO[i:i + 900]
    assert "40.0" in win and "linaza" in win and "girasol" in win, \
        "semillas/frutos secos (densos en kcal) llevan techo propio de 40 g"


def test_marker_anchored_in_source():
    assert "P1-CLOSER-LINE-CLAMP" in _GO
