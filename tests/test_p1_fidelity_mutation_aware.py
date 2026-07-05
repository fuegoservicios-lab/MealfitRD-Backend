"""[P1-FIDELITY-MUTATION-AWARE · 2026-07-05] El skeleton-fidelity gate era ciego a dos
realidades del pipeline (caso vivo corr=38719947, quemó el retry que el quirúrgico ahorraba):

1. **Días RECICLADOS** (surgical fix del retry re-usa días buenos del attempt anterior): el
   planner RE-planifica en cada attempt → el día reciclado se comparaba contra un skeleton
   que nunca vio ("Día 2 omitió arenque/almendras" — jamás le fueron asignadas). Fix: marker
   `_recycled_from_prior_attempt` en el recycle + skip SOLO del fidelity check (coherence y
   schema siguen aplicando).
2. **Mutaciones DETERMINISTAS legítimas** (presupuesto "salmón → pescado blanco", autofix de
   proteína repetida "pescado->pollo", sodio enlatado→fresco): la proteína asignada SÍ fue
   honrada en generación y luego reemplazada a propósito por el sistema. Fix: descuento de
   `missing_proteins` cuando el matcher tolerante encuentra la proteína en el blob de
   "reemplazados" (mismo `_skeleton_protein_present`, sin heurística nueva).

El gate conserva su propósito original intacto: cazar al day-gen que IGNORA la asignación.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def _meal(name, ings, **extra):
    m = {"meal": "Almuerzo", "name": name, "ingredients": list(ings),
         "ingredients_raw": list(ings), "recipe": ["Sirve."],
         "cals": 500, "protein": 30, "carbs": 50, "fats": 15}
    m.update(extra)
    return m


def _mk(result_day_extra=None, meal_extra=None):
    day = {"day": 1, "meals": [
        _meal("Arroz con Ensalada", ["150 g de arroz blanco", "50 g de lechuga"],
              **(meal_extra or {})),
    ]}
    day.update(result_day_extra or {})
    result = {"days": [day]}
    skeleton = {"days": [{"day": 1, "protein_pool": [
        "pollo (proteína principal)", "huevos (desayuno)"]}]}
    return result, skeleton


# ---------------------------------------------------------------------------
# baseline: el gate sigue cazando la desobediencia real del day-gen
# ---------------------------------------------------------------------------

def test_baseline_still_fires_on_real_omission(go):
    result, skeleton = _mk()
    go._run_assembly_validations(result, skeleton, set())
    errs = result.get("_skeleton_fidelity_errors") or []
    assert errs and "omitió" in errs[0], \
        "día sin NINGUNA proteína asignada y sin mutaciones → el gate debe seguir cazándolo"


# ---------------------------------------------------------------------------
# 1) días reciclados exentos
# ---------------------------------------------------------------------------

def test_recycled_day_skips_fidelity(go):
    result, skeleton = _mk(result_day_extra={"_recycled_from_prior_attempt": True})
    go._run_assembly_validations(result, skeleton, set())
    assert not result.get("_skeleton_fidelity_errors"), \
        "día reciclado valida contra el skeleton de SU attempt, no contra el re-planificado"


def test_recycle_site_sets_marker():
    i = _GO.index("recycled_day = recycled_days_cache.get(day_num)")
    win = _GO[i:i + 900]
    assert 'recycled_day["_recycled_from_prior_attempt"] = True' in win


# ---------------------------------------------------------------------------
# 2) descuento por mutaciones deterministas
# ---------------------------------------------------------------------------

def test_budget_substitution_discounts_missing(go):
    result, skeleton = _mk(meal_extra={
        "_budget_substitutions": ["pollo → Filete de pescado blanco"]})
    go._run_assembly_validations(result, skeleton, set())
    assert not result.get("_skeleton_fidelity_errors"), \
        "pollo fue REEMPLAZADO por presupuesto (no ignorado) → queda 1 missing < threshold 2"


def test_protein_autofix_discounts_missing(go):
    result, skeleton = _mk(meal_extra={"_protein_autofix_applied": "pollo->pavo"})
    go._run_assembly_validations(result, skeleton, set())
    assert not result.get("_skeleton_fidelity_errors"), \
        "el autofix de proteína repetida documenta el swap → la asignación fue honrada"


def test_unrelated_mutation_does_not_discount(go):
    # la sustitución fue de OTRA proteína (res) → pollo y huevos siguen missing → gate dispara.
    result, skeleton = _mk(meal_extra={
        "_budget_substitutions": ["res → Filete de pescado blanco"]})
    go._run_assembly_validations(result, skeleton, set())
    assert result.get("_skeleton_fidelity_errors"), \
        "el descuento es por-proteína (matcher), no un bypass general del gate"


def test_marker_anchored_in_source():
    assert "P1-FIDELITY-MUTATION-AWARE" in _GO
