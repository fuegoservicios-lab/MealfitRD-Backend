"""[P0-SKELETON-FIDELITY-MATCH · 2026-06-13] El check de skeleton-fidelity daba
FALSOS POSITIVOS de "omitió proteínas clave": comparaba la etiqueta COMPLETA del
`protein_pool` del skeleton ("lentejas (proteína principal)", "huevos enteros /
claras", "maní / mantequilla de maní") como substring directo de los ingredientes
del día ("lentejas", "huevos", "maní"). Como la etiqueta con descriptor/alternativas
nunca aparece verbatim, el día se marcaba como omitiendo proteínas que SÍ tenía →
rechazo HIGH del revisor → reintentos agotados → entrega degradada con alerta
`plan_quality_degraded` (observado en prod 2026-06-13, plan 3abe27cd: día 1 con
P154/100% pero rechazado por "omitió lentejas/huevos/maní" que estaban presentes).

`_skeleton_protein_present` corrige el matching (quita "(...)", parte "/", matchea
token-núcleo con frontera de palabra) preservando la detección de omisiones reales.

Anchor: P0-SKELETON-FIDELITY-MATCH.
"""
from graph_orchestrator import _skeleton_protein_present, _run_assembly_validations


# Texto real de ingredientes del día 1 del plan de prod (3abe27cd) — incluía las
# 3 proteínas asignadas, pero el matcher viejo las daba por omitidas.
_REAL_DAY1 = (
    "3/4 taza de yogurt griego natural 0.5 taza de avena 2 cdas de maní tostado "
    "1 taza de sandía 1 manzana roja 1 cdta de miel canela "
    "1 taza de lentejas cocidas 1/2 taza de arroz integral 1 papa pequeña "
    "2 huevos enteros 3 claras de huevo aceite de oliva"
).lower()


def test_strips_parenthetical_descriptor():
    assert _skeleton_protein_present("lentejas (proteína principal)", _REAL_DAY1)


def test_handles_slash_alternatives():
    assert _skeleton_protein_present("huevos enteros / claras", _REAL_DAY1)
    assert _skeleton_protein_present("maní / mantequilla de maní", _REAL_DAY1)


def test_genuinely_absent_protein_not_matched():
    assert not _skeleton_protein_present("pescado (proteína principal)", _REAL_DAY1)
    assert not _skeleton_protein_present("pollo / pechuga de pollo", _REAL_DAY1)


def test_word_boundary_avoids_substring_false_match():
    # "pollo" NO debe matchear "repollo".
    assert not _skeleton_protein_present("pollo", "1 taza de repollo morado picado")


def test_assembly_no_false_skeleton_error_when_proteins_present():
    """E2E del validador: día con las proteínas presentes (nombres comerciales)
    NO debe producir _skeleton_fidelity_errors (el bug de prod)."""
    result = {"days": [{
        "day": 1,
        "meals": [
            {"ingredients": ["1 taza de lentejas cocidas", "1/2 taza de arroz integral"]},
            {"ingredients": ["2 cdas de maní tostado", "1 manzana"]},
            {"ingredients": ["2 huevos enteros", "3 claras de huevo", "vegetales salteados"]},
        ],
    }]}
    skeleton = {"days": [{
        "day": 1,
        "protein_pool": ["lentejas (proteína principal)", "huevos enteros / claras", "maní / mantequilla de maní"],
    }]}
    _run_assembly_validations(result, skeleton, affected_days_set=set())
    assert "_skeleton_fidelity_errors" not in result, result.get("_skeleton_fidelity_errors")


def test_assembly_still_flags_real_omission():
    """No romper la detección legítima: día que de verdad omite 2+ proteínas
    asignadas SÍ debe flagearse (bug del day_generator que ignora el skeleton)."""
    result = {"days": [{
        "day": 1,
        "meals": [{"ingredients": ["1 taza de arroz blanco", "ensalada verde", "aceite de oliva"]}],
    }]}
    skeleton = {"days": [{
        "day": 1,
        "protein_pool": ["pollo (proteína principal)", "pescado / atún", "res molida"],
    }]}
    _run_assembly_validations(result, skeleton, affected_days_set=set())
    assert result.get("_skeleton_fidelity_errors"), "debe flagear la omisión real"
