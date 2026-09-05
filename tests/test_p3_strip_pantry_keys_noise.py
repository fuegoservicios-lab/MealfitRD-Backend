"""[P3-STRIP-PANTRY-KEYS-NOISE · 2026-09-03] Los 4 metadatos de la despensa que `_enqueue_plan_chunk`
sella en `pipeline_snapshot["form_data"]` están en la whitelist del orquestador.

Medido en el journal (02→03 sep): 31 WARNING «P0-A2: stripped 4 clave(s)» en 24 h — uno por chunk,
el aviso más frecuente del backend — sin efecto alguno: esos campos los lee el worker desde el
snapshot (staleness de la nevera, ancla del huso, modo de cantidades), no el grafo.
"""
from __future__ import annotations

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
GO = (BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
CT = (BACKEND / "cron_tasks.py").read_text(encoding="utf-8")

KEYS = ("_pantry_captured_at", "_chunk_anchor_source", "_pantry_quantity_mode", "_pantry_quantity_hybrid_tolerance")


def _whitelist_block() -> str:
    m = re.search(r"_TRUSTED_INTERNAL_FORM_KEYS: frozenset = frozenset\(\{(.*?)\n\}\)", GO, re.DOTALL)
    assert m, "whitelist _TRUSTED_INTERNAL_FORM_KEYS no encontrada"
    return m.group(1)


def test_the_four_pantry_keys_are_whitelisted():
    block = _whitelist_block()
    for k in KEYS:
        assert f'"{k}",' in block, k


def test_the_enqueuer_still_stamps_them_into_the_snapshot():
    """Si el encolador deja de sellarlos, la whitelist queda sin sentido — y al revés."""
    for k in KEYS:
        assert f'pipeline_snapshot["form_data"]["{k}"]' in CT or f'"{k}" not in pipeline_snapshot["form_data"]' in CT, k


def test_strict_router_strip_is_untouched():
    """El strip ESTRICTO del router (allow_set=None) sigue: un cliente no puede colar estas claves."""
    assert GO.count("allow_set=None") >= 1 or "allow_set=None" in (BACKEND / "generation_inputs.py").read_text(encoding="utf-8")
    assert 'allow_set=_TRUSTED_INTERNAL_FORM_KEYS' in GO
