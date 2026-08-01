"""[P1-CULINARY-GOLDEN · 2026-07-31] Golden set de coherencia culinaria (F0 del
spec 2026-07-31-culinary-coherence-design.md).

10 fixtures ESTÁTICOS commiteados (5 buenos = miden falsos positivos; 5 mutados
con defectos etiquetados = miden falsos negativos) + manifest ground-truth.
Estáticos a propósito: el ground truth no debe reescribirse en silencio cuando
cambie la DB. Anti-caducidad: test de slugs vivos contra el catálogo real
(skip limpio sin DB).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIX = Path(__file__).resolve().parent / "fixtures" / "culinary_golden"
_CLASES = {"verbo_imposible", "estado_imposible", "ingrediente_huerfano",
           "combo_absurdo", "tecnica_impropia", "nombre_no_corresponde"}
_SLOTS = {"Desayuno", "Merienda", "Almuerzo", "Cena"}


def _load(name):
    return json.loads((_FIX / f"{name}.json").read_text(encoding="utf-8"))


def test_existen_los_11_archivos():
    esperados = [f"golden_{i:02d}_bueno" for i in range(1, 6)] + \
                [f"golden_{i:02d}_mutado" for i in range(1, 6)] + ["golden_manifest"]
    faltan = [n for n in esperados if not (_FIX / f"{n}.json").exists()]
    assert not faltan, f"faltan fixtures: {faltan}"


@pytest.mark.parametrize("i", range(1, 6))
def test_shape_de_plan_valido(i):
    for suf in ("bueno", "mutado"):
        plan = _load(f"golden_{i:02d}_{suf}")
        days = plan.get("days")
        assert isinstance(days, list) and days, f"golden_{i:02d}_{suf}: days vacío"
        for d in days:
            assert isinstance(d.get("day"), int)
            for m in d.get("meals") or []:
                assert m.get("meal") in _SLOTS, f"slot raro: {m.get('meal')}"
                assert isinstance(m.get("name"), str) and m["name"]
                assert isinstance(m.get("ingredients"), list) and m["ingredients"]
                assert isinstance(m.get("recipe"), list) and m["recipe"]
                assert all(isinstance(s, str) for s in m["recipe"])


def test_cobertura_de_slots_y_vegetariano():
    """Los 5 buenos cubren los 4 slots, meal-counts distintos y ≥1 vegetariano."""
    slots, counts, veg = set(), set(), 0
    for i in range(1, 6):
        plan = _load(f"golden_{i:02d}_bueno")
        n_meals = {len(d["meals"]) for d in plan["days"]}
        counts |= n_meals
        for d in plan["days"]:
            slots |= {m["meal"] for m in d["meals"]}
        if plan.get("_meta", {}).get("vegetariano"):
            veg += 1
    assert slots == _SLOTS
    assert len(counts) >= 2, f"todos los planes tienen el mismo meal-count: {counts}"
    assert veg >= 1, "ningún plan bueno es vegetariano"


def test_trampa_fp_plural_singular():
    """≥1 bueno declara el par plural-en-ingrediente/singular-en-paso (FP del
    dry-run 2026-07-31: '2½ tomates' vs 'Ralla el tomate')."""
    con_trampa = [i for i in range(1, 6)
                  if _load(f"golden_{i:02d}_bueno").get("_meta", {}).get("trampa_plural")]
    assert con_trampa, "ningún bueno incluye la trampa plural↔singular"


def test_manifest_cruza_con_fixtures():
    man = _load("golden_manifest")
    assert set(man["mutados"].keys()) == {f"golden_{i:02d}_mutado" for i in range(1, 6)}
    clases_vistas = set()
    for nombre, entry in man["mutados"].items():
        assert entry["base"].endswith("_bueno")
        defects = entry["defects"]
        assert 4 <= len(defects) <= 6, f"{nombre}: {len(defects)} defectos (esperado 4-6)"
        plan = _load(nombre)
        dias = {d["day"] for d in plan["days"]}
        for df in defects:
            assert df["class"] in _CLASES, f"clase desconocida: {df['class']}"
            assert df["expected_by"].startswith(("capa1:", "juez"))
            assert df["day"] in dias, f"{nombre}: defecto apunta a día inexistente {df['day']}"
            clases_vistas.add(df["class"])
    assert clases_vistas == _CLASES, f"clases sin mutación: {_CLASES - clases_vistas}"


def test_slugs_de_catalogo_vivos():
    """Anti-caducidad: los alimentos que el golden set usa siguen en el catálogo.
    Sin pool DB → skip (no flakiness en CI sin red)."""
    import db_core
    if not getattr(db_core, "connection_pool", None):
        pytest.skip("sin pool DB")
    try:
        db_core.connection_pool.open()
    except Exception:
        pass
    from shopping_calculator import get_master_ingredients
    catalogo = get_master_ingredients()
    if not catalogo:
        pytest.skip("catálogo vacío (sin DB)")
    nombres = {str(r.get("name", "")).strip().lower() for r in catalogo}
    man = _load("golden_manifest")
    muertos = [f for f in man["catalog_foods_used"] if f.strip().lower() not in nombres]
    assert not muertos, f"alimentos del golden set ya no existen en el catálogo: {muertos}"
