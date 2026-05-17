"""[P2-COHERENCE-GUARD-PERF · 2026-05-16] Regression guard: el alias_map
construido desde `get_master_ingredients()` para `_canonicalize_for_coherence`
DEBE estar cacheado con TTL.

Bug observado en test E2E del 2026-05-15 21:51:54 (plan_id=ae29c7a9):
  - `[P2-COHERENCE-GUARD-PERF] guard tardó 3323ms (umbral 1000ms). recipes=35
    ingredients=33 divergences=61 mode=warn. Posible regresión perf — investigar.`

Root cause: `_canonicalize_for_coherence` reconstruía el alias_map en CADA
call iterando ~100-200 items del master_list + sus aliases. Y
`_canonicalize_food_dict_for_coherence` la llamaba N+1 veces:
  1. UNA bulk para computar `canonical_set` (todas las food_names).
  2. UNA por food_name para deducir el mapping inverso raw→canonical
     (truco: pasar `[raw]` y leer `next(iter(...))`).

Para 33 ingredientes × 2 dicts × 2 sets = ~70 builds del alias_map de ~150 items
cada uno → 3323ms.

Fix: helper `_get_coherence_alias_map_cached()` con cache module-level + TTL=300s.
master_ingredients rara vez cambia en runtime (dataset estático); restart natural
invalida el cache. Las invocaciones subsecuentes son O(1) lookup + O(N) iter.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PATH = _BACKEND_ROOT / "shopping_calculator.py"


def _read_shopping() -> str:
    return _SHOPPING_PATH.read_text(encoding="utf-8")


def test_alias_map_cache_helper_exists():
    """Helper `_get_coherence_alias_map_cached()` debe estar definido."""
    text = _read_shopping()
    assert "def _get_coherence_alias_map_cached" in text, (
        "Falta helper `_get_coherence_alias_map_cached` en shopping_calculator.py. "
        "P2-COHERENCE-GUARD-PERF requiere cache del alias_map para evitar O(n²)."
    )


def test_alias_map_cache_globals_declared():
    """Variables module-level del cache deben existir."""
    text = _read_shopping()
    for sym in (
        "_COHERENCE_ALIAS_MAP_CACHE",
        "_COHERENCE_ALIAS_MAP_CACHE_AT",
        "_COHERENCE_ALIAS_MAP_TTL_S",
    ):
        assert re.search(rf"^{re.escape(sym)}\b", text, re.MULTILINE), (
            f"Falta variable module-level `{sym}`. Cache no podrá persistir "
            f"entre invocaciones del guard."
        )


def test_alias_map_cache_ttl_reasonable():
    """TTL del cache debe ser >=60s (no tan corto que se invalide en cada guard
    call) y <=3600s (no tan largo que ignore updates legítimos del master_list)."""
    text = _read_shopping()
    m = re.search(r"_COHERENCE_ALIAS_MAP_TTL_S\s*=\s*([\d.]+)", text)
    assert m, "TTL `_COHERENCE_ALIAS_MAP_TTL_S` no definido como literal."
    ttl = float(m.group(1))
    assert 60.0 <= ttl <= 3600.0, (
        f"TTL del cache alias_map = {ttl}s fuera del rango razonable [60, 3600]. "
        f"Si necesitas algo distinto, justificar en comentario inline."
    )


def test_canonicalize_for_coherence_uses_cache():
    """`_canonicalize_for_coherence` debe usar `_get_coherence_alias_map_cached()`,
    NO reconstruir alias_map inline. Regresar al inline build reintroduce O(n²)."""
    text = _read_shopping()
    fn_match = re.search(
        r"def _canonicalize_for_coherence\([^)]*\)[^:]*:(.*?)(?=^def |\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert fn_match, "No se encontró `def _canonicalize_for_coherence(...)`."
    body = fn_match.group(1)
    assert "_get_coherence_alias_map_cached" in body, (
        "`_canonicalize_for_coherence` no invoca el helper cacheado. "
        "Refactor incompleto — verás de nuevo 3000+ ms en el guard bajo carga."
    )
    # Defensiva: el bloque inline original DEBE estar removido.
    inline_build_pattern = re.compile(
        r"master_list\s*=\s*get_master_ingredients\(\)",
    )
    assert not inline_build_pattern.search(body), (
        "`_canonicalize_for_coherence` aún contiene la construcción inline "
        "del alias_map (`master_list = get_master_ingredients()`). El helper "
        "cacheado debe ser la única ruta."
    )
