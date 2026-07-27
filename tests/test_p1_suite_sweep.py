"""[P1-SUITE-SWEEP · 2026-07-27] Barrido de los 15 rojos genuinos del subconjunto p1_ (19→4).

Los 4 que quedan son contaminación de orden documentada (pasan solos; ver
project_suite_triage_2026_07_26). De los 15 genuinos, 12 eran tests caducados (ventanas de bytes,
conteos exactos, constantes migradas, decisiones superadas) y **3 destaparon defectos REALES de
producción**, anclados aquí para que el barrido no se pueda revertir en silencio:

1. **P1-PREP-HEAD-GUARD** — "TORTILLA de harina de trigo (wrap, 60g)" resolvía a HARINA cruda:
   el guard de preparaciones casaba "harina de X" en cualquier posición. ~17% de drift kcal y
   nombre falso en micros. (destapado por test_p1_resolver_coverage)
2. **P1-BLENDER-FUSION-EXEMPT** — la fusión al Montaje (07-26) dejaba la espinaca de un batido
   verde como "Termina con espinacas frescas" DESPUÉS de licuar: sin licuar. En licuados el
   ingrediente va a la licuadora. (destapado por test_p1_blender_step_coherence)
3. **P1-CLOSER-OWN-LINE-EXEMPT** — el detector precocido-desde-línea (07-25) leía la línea que
   el PROPIO closer acababa de insertar ("58g de pechuga de pollo cocido", base de peso) y
   convertía pollo crudo en "Incorpora (ya viene cocido)": nadie lo cocinaba.
   (destapado por test_p1_protein_step_soft_dairy)

tooltip-anchor: P1-SUITE-SWEEP
"""
from __future__ import annotations

import pathlib

import graph_orchestrator as g
import shopping_calculator as sc

_GO_SRC = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8")


def test_prep_head_guard_vivo():
    assert sc.resolve_preparation_distinct("tortilla de harina de trigo (wrap, 60g)") == (False, None)
    assert sc.resolve_preparation_distinct("1 taza de harina de trigo") == (True, "Harina de trigo")


def test_blender_fusion_exempt_vivo():
    """La fusión al Montaje no aplica a licuados (el complemento DEBE licuarse)."""
    assert "not _blended and isinstance(recipe, list)" in _GO_SRC, (
        "la exención de licuados salió de la condición de fusión — vuelve 'Termina con espinacas' "
        "sin licuar en los batidos"
    )


def test_closer_own_line_exempt_vivo():
    assert "P1-CLOSER-OWN-LINE-EXEMPT" in _GO_SRC
    m = {"name": "Revoltillo", "protein": 5, "carbs": 10, "fats": 5, "cals": 150,
         "ingredients": ["2 huevos", "58g de pechuga de pollo cocido"],
         "ingredients_raw": ["2 huevos"],
         "recipe": ["Bate los huevos y cocínalos."]}
    g._append_closer_protein_step(m, "pechuga de pollo", False)
    blob = " ".join(str(s) for s in m["recipe"]).lower()
    assert "ya viene cocido" not in blob, (
        "la línea del propio closer volvió a contar como precocido: el pollo crudo queda sin cocinar"
    )
