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
import re as _re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_three_submarkers_present_in_source():
    for marker in ("P1-STEM-SHORT-FOOD-NOUN", "P1-COMPLEMENT-STEP-MERGE", "P1-EGG-STEP-SCRUB"):
        assert marker in _GO_SRC, f"falta el marker {marker} en graph_orchestrator.py"


def test_last_known_pfix_matches_this_batch():
    """El marker debe estar bien formado y tener su test de regresión enlazado.

    ⚠️ [P1-FALSE-STEP-GRAFTS · 2026-07-25] Antes exigía que `_LAST_KNOWN_PFIX` contuviera
    literalmente `P1-RECIPE-STEP-COHERENCE-BATCH`. Eso sólo puede ser cierto en el único commit
    en que este batch fue el último P-fix: desde el siguiente, el test quedó **rojo para siempre**
    y por una razón que no es la que vigila. Un test que no puede volver a pasar deja de ser
    señal y pasa a ser ruido que esconde fallos reales.

    La invariante que sí importa ya es del repo (`test_p2_hist_audit_14_marker_test_link`): el
    slug del marker vivo tiene que resolver a un `tests/test_<slug>*.py`. Aquí se comprueba lo
    que le toca a ESTE archivo: que el batch conserve su propio test enlazado.
    """
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = _re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', app_src)
    assert m, "app.py debe declarar _LAST_KNOWN_PFIX"
    marker = m.group(1)
    assert _re.match(r"^P\d[A-Z0-9-]*.*·\s*\d{4}-\d{2}-\d{2}$", marker), marker
    slug = marker.split("·")[0].strip().lower().replace("-", "_")
    assert list(_BACKEND.glob(f"tests/test_{slug}*.py")), \
        f"el marker vivo {marker!r} no tiene test de regresión enlazado"
    assert (_BACKEND / "tests" / "test_p1_recipe_step_coherence_batch.py").exists()
