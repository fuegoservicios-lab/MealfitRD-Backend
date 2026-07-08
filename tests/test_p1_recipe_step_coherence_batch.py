"""[P1-RECIPE-STEP-COHERENCE-BATCH · 2026-07-08] Anchor de 3 fixes de calidad de receta encontrados
en la revisión en vivo del plan 830d9aaa (12 recetas, 6/12 con defectos de texto visibles):

1. P1-STEM-SHORT-FOOD-NOUN: "pan" (3 chars) se filtraba del stem de `_ensure_ingredients_used_in_recipe`
   → "pan integral familiar" ganaba un paso "complemento" espurio pese a que "pan" SÍ aparecía en el
   paso real (Tostadas Francesas con Piña). Cobertura funcional: test_p2_stem_filler_tokens.py.
2. P1-COMPLEMENT-STEP-MERGE (+ fusión del paso "El Toque de Fuego (complemento)" de reverse-coherence):
   `_integrate_complement_steps` solo fusionaba el 💪 del closer, no el paso complemento de
   `_ensure_ingredients_used_in_recipe` (quedaba como 3er paso con título casi-duplicado — Atún
   Salteado Cantonés, Arepitas de Harina de Negrito), y concatenaba 2 proteínas del mismo template
   como 2 oraciones casi-idénticas en vez de fusionarlas (Catibías pollo+camarones). Cobertura
   funcional: test_p1_closer_step_integrate.py.
3. P1-EGG-STEP-SCRUB: `_substitute_blended_raw_egg` reemplazaba el huevo crudo por yogur a nivel de
   ingredientes pero no limpiaba pasos previos que instruían separar claras de yemas (Batido
   Refrescante de Lechosa y Arándano). Cobertura funcional: test_p2_raw_egg_substitute.py.

Este archivo NO duplica esas pruebas — solo ancla el marker (contrato P2-HIST-AUDIT-14) a los 3
sub-markers para que un futuro grep encuentre las 3 causas raíz desde un solo punto de entrada.
"""
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_three_submarkers_present_in_source():
    for marker in ("P1-STEM-SHORT-FOOD-NOUN", "P1-COMPLEMENT-STEP-MERGE", "P1-EGG-STEP-SCRUB"):
        assert marker in _GO_SRC, f"falta el marker {marker} en graph_orchestrator.py"


def test_last_known_pfix_matches_this_batch():
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-RECIPE-STEP-COHERENCE-BATCH" in app_src, \
        "_LAST_KNOWN_PFIX debe apuntar a este batch tras el commit"
