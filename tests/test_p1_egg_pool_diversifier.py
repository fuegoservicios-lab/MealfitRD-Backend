"""[P1-EGG-POOL-DIVERSIFIER · 2026-07-05] La RAÍZ del sobreuso de huevo vive en el PLANNER.

Forense corr=1b2d7696 (2026-07-05 06:21): el planner asignó 'Huevos'/'Claras' al pool de LOS
TRES días → el day-gen cocinó huevo a diario como protagonista ('revoltillo' ×3) → 3 gates
distintos dispararon (sobreuso 6/12, cross-day dish, same-day repeat) y quemaron 2 intentos.
Los correctores downstream (egg-cap autofix, protein-repeat) respetan protagonistas POR DISEÑO
— la corrección correcta es upstream: no darle huevo al day-gen todos los días.

`_diversify_egg_pools`: huevo permitido en ≤ EGG_POOL_MAX_DAYS pools (default 2); en los días
excedentes la entry huevo/claras se reemplaza por una proteína ligera verificada rotada
(diet-aware: vegano → legumbres; alergias vía scan SSOT) que no esté ya en ese pool. Corre
ANTES de day-gen → prompt, PROTEIN-POOL-SCRUB y skeleton-fidelity leen el MISMO pool mutado.
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


def _mk_skel(egg_days=3):
    days = []
    for n in range(1, 4):
        pool = ["Pollo", "Queso ricotta"]
        if n <= egg_days:
            pool.insert(1, "Huevos (desayuno)")
        days.append({"day": n, "protein_pool": pool})
    return days


# ---------------------------------------------------------------------------

def test_knobs_defaults():
    assert '_env_bool("MEALFIT_EGG_POOL_DIVERSIFIER", True)' in _GO
    assert '_env_int("MEALFIT_EGG_POOL_MAX_DAYS", 2' in _GO


def test_third_egg_day_diversified(go):
    skel = _mk_skel(egg_days=3)
    n = go._diversify_egg_pools(skel, {})
    assert n == 1, "huevo en 3 pools, cap 2 → se diversifica exactamente 1"
    # días 1-2 conservan huevo; día 3 lo pierde por una proteína ligera verificada.
    assert any("huevo" in str(p).lower() for p in skel[0]["protein_pool"])
    assert any("huevo" in str(p).lower() for p in skel[1]["protein_pool"])
    _p3 = " ".join(str(p) for p in skel[2]["protein_pool"]).lower()
    assert "huevo" not in _p3 and "clara" not in _p3
    assert "queso blanco" in _p3 or "yogurt griego" in _p3 or "habichuelas" in _p3


def test_replacement_not_already_in_pool(go):
    skel = _mk_skel(egg_days=3)
    skel[2]["protein_pool"] = ["Queso blanco fresco", "Huevos", "Pollo"]
    go._diversify_egg_pools(skel, {})
    _p3 = [str(p).lower() for p in skel[2]["protein_pool"]]
    # 'Queso blanco' ya estaba (substring) → la rotación salta a la siguiente opción.
    assert sum("queso blanco" in p for p in _p3) == 1


def test_multiple_egg_entries_collapse(go):
    skel = _mk_skel(egg_days=3)
    skel[2]["protein_pool"] = ["Huevos", "Claras de huevo", "Pollo"]
    go._diversify_egg_pools(skel, {})
    _p3 = [str(p).lower() for p in skel[2]["protein_pool"]]
    assert not any("huevo" in p or "clara" in p for p in _p3)
    assert len(_p3) == 2, "Huevos + Claras colapsan en UN reemplazo (no se infla el pool)"


def test_vegan_rotation_uses_legumes(go):
    skel = _mk_skel(egg_days=3)
    go._diversify_egg_pools(skel, {"dietType": "vegano"})
    _p3 = " ".join(str(p) for p in skel[2]["protein_pool"]).lower()
    assert "queso" not in _p3.replace("queso ricotta", "") and "yogurt" not in _p3
    assert any(t in _p3 for t in ("habichuelas", "lentejas", "garbanzos"))


def test_under_cap_untouched_and_knob_off(go, monkeypatch):
    skel = _mk_skel(egg_days=2)
    assert go._diversify_egg_pools(skel, {}) == 0
    monkeypatch.setattr(go, "EGG_POOL_DIVERSIFIER_ENABLED", False)
    skel3 = _mk_skel(egg_days=3)
    assert go._diversify_egg_pools(skel3, {}) == 0


def test_callsite_before_day_gen():
    i = _GO.index("_diversify_egg_pools(skeleton_days[:days_in_chunk], form_data)")
    j = _GO.index("async def _generate_candidate(temp_override=None):")
    assert i < j, "el diversificador corre ANTES de generar los días (prompt/scrub/fidelity coherentes)"


def test_marker_anchored_in_source():
    assert "P1-EGG-POOL-DIVERSIFIER" in _GO
