"""[P1-SPICES-CATALOG-SYNC · 2026-07-01] (audit creatividad G1)

Los prompts prohibían como EJEMPLO especias que el lote 2 del catálogo (2026-06-26) verificó
(comino/cúrcuma/laurel/tomillo/curry) → instrucción auto-contradictoria (aparecían en la lista
"USA EXCLUSIVAMENTE" Y como ejemplo de prohibido) → el LLM omitía sazones legítimas y los guisos/
locrios salían desabridos, contra el objetivo de apetecibilidad.

Fix: ejemplos de prohibido = solo especias genuinamente off-catálogo (achiote, sazón en polvo,
clavo dulce, pimienta de olor); las verificadas se promueven explícitamente como sazonadores.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_DAYGEN = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

_VERIFIED_SPICES = ("comino", "cúrcuma", "laurel", "tomillo", "curry")


def _prohibited_example_segments(src: str) -> list:
    """Segmentos '(ej. ...)' que siguen a la instrucción de omitir off-catálogo."""
    return re.findall(r"pide algo que no está (?:aquí|en el catálogo)\s*\(ej\.([^)]*)\)", src)


def test_daygen_prohibited_examples_exclude_verified_spices():
    segs = _prohibited_example_segments(_DAYGEN)
    assert segs, "no se encontró el segmento de ejemplos prohibidos en day_generator"
    for seg in segs:
        for sp in _VERIFIED_SPICES:
            assert sp not in seg.lower(), (
                f"'{sp}' es VERIFICADO (lote 2 · 2026-06-26) pero aparece como ejemplo de "
                f"prohibido en day_generator — instrucción auto-contradictoria (P1-SPICES-CATALOG-SYNC)"
            )


def test_graph_catalog_block_examples_exclude_verified_spices():
    segs = _prohibited_example_segments(_GRAPH)
    assert segs, "no se encontró el segmento de ejemplos prohibidos en el bloque catálogo de g_o"
    for seg in segs:
        for sp in _VERIFIED_SPICES:
            assert sp not in seg.lower(), (
                f"'{sp}' es VERIFICADO pero aparece como prohibido en el bloque catálogo "
                f"(P1-SPICES-CATALOG-SYNC)"
            )


def test_verified_spices_promoted_as_seasonings():
    assert "P1-SPICES-CATALOG-SYNC" in _DAYGEN and "P1-SPICES-CATALOG-SYNC" in _GRAPH
    for sp in _VERIFIED_SPICES:
        assert sp in _DAYGEN.lower(), f"day_generator ya no promueve '{sp}' como sazonador verificado"
        assert sp in _GRAPH.lower(), f"el bloque catálogo ya no promueve '{sp}' como sazonador verificado"
