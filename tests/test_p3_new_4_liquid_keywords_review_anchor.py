"""[P3-NEW-4 · 2026-05-10] Lock-the-doc: comentario en
`_get_coherence_liquid_keywords` (shopping_calculator.py) anchora el review
trimestral del knob `MEALFIT_COHERENCE_LIQUID_KEYWORDS`.

Bug temido (audit 2026-05-10 — no accionable sin evidencia):
  La lista default de keywords líquidos puede quedar obsoleta a medida
  que los planes incorporan nuevos condimentos es-DO (agrio, mojo,
  leche de coco). Sin process review, el cron de coherencia reporta
  false positives `cap_swallowed_modifier` para esos items.

Fix:
  No code change inmediato (requeriría evidencia de telemetría). En su
  lugar, comentario en el código que documenta:
    - Cómo añadir nuevos keywords via env var (no redeploy).
    - 4 candidatos a vigilar (agrio, mojo, miel, leche de coco).
    - Frecuencia sugerida de review (trimestral).
    - Owner (SRE de pipeline_metrics).

Cobertura:
  1. El bloque `[P3-NEW-4` sigue presente en `_get_coherence_liquid_keywords`.
  2. Menciona el knob `MEALFIT_COHERENCE_LIQUID_KEYWORDS`.
  3. Lista al menos 3 candidatos es-DO a vigilar.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PY = _BACKEND_ROOT / "shopping_calculator.py"


def _read_function_docstring(fn_name: str) -> str:
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    fn_match = re.search(
        rf"def\s+{fn_name}\s*\(.*?\)\s*(?:->\s*[\w\[\]\| ]+)?\s*:\s*\n\s*\"\"\"",
        src,
        re.DOTALL,
    )
    assert fn_match is not None, f"No encuentro `def {fn_name}`."
    start = fn_match.end()
    end_match = re.search(r'\"\"\"', src[start:])
    assert end_match is not None
    return src[start:start + end_match.start()]


def test_p3_new_4_block_present():
    doc = _read_function_docstring("_get_coherence_liquid_keywords")
    assert "[P3-NEW-4" in doc, (
        "Falta anchor `[P3-NEW-4` en docstring de "
        "_get_coherence_liquid_keywords."
    )


def test_documents_env_var_override():
    doc = _read_function_docstring("_get_coherence_liquid_keywords")
    block = doc[doc.find("[P3-NEW-4"):]
    assert "MEALFIT_COHERENCE_LIQUID_KEYWORDS" in block, (
        "El bloque P3-NEW-4 no nombra el knob — falta el mecanismo de "
        "override sin redeploy."
    )


def test_lists_at_least_three_candidate_keywords():
    """El bloque debe listar al menos 3 candidatos es-DO a vigilar
    (anchor cultural-context para futuros devs no familiarizados con
    cocina dominicana)."""
    doc = _read_function_docstring("_get_coherence_liquid_keywords")
    block = doc[doc.find("[P3-NEW-4"):]
    # Heurística: candidatos esperados.
    expected_at_least = {"agrio", "mojo", "leche de coco"}
    found = sum(1 for kw in expected_at_least if kw in block.lower())
    assert found >= 3, (
        f"El bloque P3-NEW-4 perdió candidatos cultural-context. "
        f"Esperados al menos 3 de {expected_at_least}, encontrados: {found}. "
        f"Sin contexto cultural, devs no-locales pueden ignorar nuevos "
        f"condimentos típicos en el review trimestral."
    )
