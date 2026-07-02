"""[P0-FALLBACK-CENA-ARROZ · 2026-07-01] (audit objetivo v3, gap P0 único) El pool curado del fallback
bariátrico contenía "Habichuelas Guisadas con Arroz" (80g arroz blanco cocido) en la CENA — la violación
cultural #1 del slot-appropriateness ("arroz de noche") — servida DETERMINISTA el día 3/7 de todo plan
fallback multi-día por la rotación P2-FALLBACK-DAY-ROTATION, a población clínica, en un pool documentado
como "clínicamente VETADO por construcción". El fallback crítico bypassa assemble_plan_node → nunca corría
_night_rice_autofix ni el gate P1-SLOT-APPROPRIATENESS.

Fix en dos capas:
1. Pool: el plato Cena legume ahora es "Habichuelas Guisadas con Yuca" (tubérculo nocturno, ya verificado
   en el catálogo por este mismo pool).
2. Defensa-en-profundidad: _apply_deterministic_clinical_layer corre _night_rice_autofix (Guard 0.5, antes
   del Guard 1 renal) — cualquier day que llegue a la capa clínica SIN pasar por assemble hereda la misma
   corrección. En el happy path es no-op idempotente. Rollback: NIGHT_RICE_AUTOFIX_ENABLED.

tooltip-anchor: P0-FALLBACK-CENA-ARROZ
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")

# Slots donde el arroz es culturalmente inaceptable (Almuerzo es el ÚNICO slot donde sí se acepta).
_NO_RICE_SLOTS_RE = re.compile(r"(?i)\b(desayuno|cena|merienda)")
_RICE_RE = re.compile(r"(?i)\barroz\b")


def _pool_items(pool: dict):
    for slot, templates in (pool or {}).items():
        for tmpl in templates or []:
            name, _tokens, desc, ingredients = tmpl
            yield slot, name, desc, list(ingredients)


# ---------------------------------------------------------------------------
# 1) Blanket estático: ningún pool fallback (genérico NI bariátrico) puede tener
#    arroz fuera del Almuerzo. Cubre nombre, descripción e ingredientes.
# ---------------------------------------------------------------------------

def test_fallback_pools_no_rice_outside_almuerzo():
    for pool_name, pool in (("_FALLBACK_MEAL_POOLS", g._FALLBACK_MEAL_POOLS),
                            ("_FALLBACK_MEAL_POOLS_BARIATRIC", g._FALLBACK_MEAL_POOLS_BARIATRIC)):
        for slot, name, desc, ings in _pool_items(pool):
            if not _NO_RICE_SLOTS_RE.search(slot):
                continue  # Almuerzo: arroz culturalmente OK
            haystack = " ".join([name, desc] + [str(i) for i in ings])
            assert not _RICE_RE.search(haystack), (
                f"[P0-FALLBACK-CENA-ARROZ] {pool_name}[{slot!r}] contiene arroz en {name!r} — "
                f"el fallback bypassa assemble (sin gate ni autofix): arroz solo se acepta en Almuerzo. "
                f"Sustituye por tubérculo nocturno (batata/yuca/casabe)."
            )


def test_replacement_dish_anchored():
    """Ancla el reemplazo concreto: la Cena legume del pool bariátrico es yuca, no arroz."""
    cena = g._FALLBACK_MEAL_POOLS_BARIATRIC["Cena"]
    legume = [t for t in cena if "legume" in t[1]]
    assert legume, "el pool Cena bariátrico perdió su opción legume (fail-safe multi-alergia)"
    name, _tokens, _desc, ings = legume[0]
    assert "yuca" in name.lower()
    assert any("yuca cocida" in str(i).lower() for i in ings)


# ---------------------------------------------------------------------------
# 2) Funcional: _build_fallback_day bariátrico jamás sirve arroz en cena/desayuno/
#    meriendas, en NINGÚN día de la rotación, incluso forzando la opción legume
#    (fish+egg restringidos elimina tilapia y huevos del pool Cena).
# ---------------------------------------------------------------------------

def _bariatric_form():
    return {"medicalConditions": ["Cirugía Bariátrica"], "allergies": []}


def _nutr():
    return {"target_calories": 1400, "macros": {"protein_g": 90, "carbs_g": 130, "fats_g": 45}}


def test_build_fallback_day_bariatric_never_rice_at_night(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_BARIATRIC_CURATED_ENABLED", True)
    monkeypatch.setattr(g, "FALLBACK_PHYSICAL_MACROS_ENABLED", False)  # hermético: sin DB de rescale
    for restricted in (frozenset(), frozenset({"fish", "egg"})):
        for day in range(1, 8):
            d = g._build_fallback_day(_nutr(), day, restricted_tokens=restricted,
                                      form_data=_bariatric_form())
            for m in d.get("meals", []):
                slot = str(m.get("meal", ""))
                if not _NO_RICE_SLOTS_RE.search(slot):
                    continue
                haystack = " ".join([str(m.get("name", ""))] +
                                    [str(i) for i in (m.get("ingredients") or [])])
                assert not _RICE_RE.search(haystack), (
                    f"día {day}, restricted={set(restricted)}: arroz en {slot!r} → {m.get('name')!r}"
                )


# ---------------------------------------------------------------------------
# 3) Estructural: la capa clínica corre _night_rice_autofix (Guard 0.5) ANTES del
#    Guard 1 renal, para que renal/quantize/carb-floor/micro-closer dimensionen el
#    tubérculo final. Anchor textual para que un refactor falle acá primero.
# ---------------------------------------------------------------------------

def test_clinical_layer_wired_before_guard1():
    i = _SRC.index("def _apply_deterministic_clinical_layer")
    j = _SRC.index('plan["_clinical_layer_applied"] = True', i)
    body = _SRC[i:j]
    assert "P0-FALLBACK-CENA-ARROZ" in body
    i_nr = body.index("_night_rice_autofix(")
    i_g1 = body.index('enforce_one("renal"')
    assert i_nr < i_g1, "el autofix debe correr ANTES del Guard 1 (renal) para operar sobre ingredientes pre-trim"


# ---------------------------------------------------------------------------
# 4) Funcional: un plan que llega a la capa clínica SIN assemble (el modo de fallo
#    real de los paths fallback 25620/26930/27053) sale sin arroz en la cena.
# ---------------------------------------------------------------------------

def _plan_cena_arroz():
    return {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Pollo con Arroz Blanco",
        "ingredients": ["120 g de Pollo", "arroz blanco 150g"],
        "recipe": ["Cocina el arroz blanco 20 minutos.", "Sirve con el pollo."],
    }]}], "calories": 2000}


def test_clinical_layer_rewrites_cena_rice(monkeypatch):
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    plan = _plan_cena_arroz()
    g._apply_deterministic_clinical_layer(plan, {}, {})
    meal = plan["days"][0]["meals"][0]
    haystack = " ".join([meal["name"]] + [str(i) for i in meal["ingredients"]])
    assert not _RICE_RE.search(haystack), f"la capa clínica no reescribió el arroz nocturno: {meal}"


def test_clinical_layer_night_rice_respects_knob_off(monkeypatch):
    """Rollback sin redeploy: NIGHT_RICE_AUTOFIX_ENABLED=False apaga también el Guard 0.5."""
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    monkeypatch.setattr(g, "NIGHT_RICE_AUTOFIX_ENABLED", False)
    plan = _plan_cena_arroz()
    g._apply_deterministic_clinical_layer(plan, {}, {})
    assert "arroz" in plan["days"][0]["meals"][0]["name"].lower()
