"""[P3-NEW-2 · 2026-05-10] Lock-the-contract: docstring de
`/{plan_id}/lifetime-lessons` documenta cuándo `last_chunk_learning` y
`consecutive_zero_log_chunks` están populated y cuándo `null` operacional
vs. bug.

Bug temido (audit 2026-05-10):
  El endpoint devolvía esos 2 campos sin explicar dependencia del cron
  (`_persist_last_chunk_learning` en cron_tasks.py al final del merge T1).
  Un frontend dev viendo `last_chunk_learning=null` no podía discernir si
  era estado normal (plan nuevo, pausado) o bug (cron crashed).

Fix:
  Bloque `[P3-NEW-2 · 2026-05-10] Contrato cron-dependiente —` en el
  docstring del endpoint enumera:
    - Cuál cron lo escribe (cron_tasks.py merge T1).
    - 4 razones operacionales para `null` (plan nuevo, failed,
      legacy, pausado).
    - Cuándo `null` ES bug (N >=1 chunks completados sin pausa) +
      métrica a vigilar.

Cobertura:
  1. El bloque P3-NEW-2 sigue presente.
  2. Menciona `cron_tasks.py` (anchor a producer).
  3. Menciona `_persist_last_chunk_learning` (anchor a función específica).
  4. Enumera las 4 razones operacionales para `null`.
  5. Menciona el caso BUG (N chunks completados + null).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


def _read_endpoint_docstring() -> str:
    """Extrae el docstring de api_plan_lifetime_lessons."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    # La signature spans multi-line con type annotations — usar pattern
    # más permisivo que matchea hasta el primer `"""` posterior.
    fn_match = re.search(
        r"def\s+api_plan_lifetime_lessons\s*\(.+?\)\s*:\s*\n\s*\"\"\"",
        src,
        re.DOTALL,
    )
    assert fn_match is not None, (
        "No encuentro `def api_plan_lifetime_lessons`."
    )
    # El docstring va desde el opening `"""` hasta el siguiente `"""`.
    start = fn_match.end()  # justo después de la apertura `"""`
    end_match = re.search(r'\"\"\"', src[start:])
    assert end_match is not None
    return src[start:start + end_match.start()]


def test_p3_new_2_block_present():
    """El bloque `[P3-NEW-2 · 2026-05-10] Contrato cron-dependiente —`
    debe seguir en el docstring."""
    docstring = _read_endpoint_docstring()
    assert "[P3-NEW-2" in docstring, (
        "Falta el anchor `[P3-NEW-2` en el docstring de "
        "`/lifetime-lessons`. Un futuro refactor que lo borre dejaría al "
        "frontend dev sin contrato escrito."
    )


def test_mentions_cron_producer():
    """El docstring debe nombrar `cron_tasks.py` y
    `_persist_last_chunk_learning` (anchor del productor)."""
    docstring = _read_endpoint_docstring()
    assert "cron_tasks.py" in docstring, (
        "Docstring P3-NEW-2 perdió la referencia a `cron_tasks.py` — sin "
        "anchor a producer, el contrato es huérfano."
    )
    assert "_persist_last_chunk_learning" in docstring, (
        "Docstring P3-NEW-2 perdió la referencia a "
        "`_persist_last_chunk_learning`. Anchor a la función del worker "
        "es crítico para diagnóstico."
    )


def test_enumerates_four_operational_null_reasons():
    """El docstring debe explicar las 4 razones operacionales para
    `null` (plan nuevo, failed/dead_letter, legacy, pausado)."""
    docstring = _read_endpoint_docstring()
    # Heurística: el bloque enumera con `1.`, `2.`, `3.`, `4.` patterns.
    null_block = docstring[docstring.find("[P3-NEW-2"):]
    for marker in ("1. ", "2. ", "3. ", "4. "):
        assert marker in null_block, (
            f"Falta enumeración `{marker}` en el bloque P3-NEW-2 — la "
            f"lista de 4 razones para `null` debe ser explícita."
        )


def test_documents_when_null_is_bug():
    """El docstring debe diferenciar `null` operacional de `null` BUG."""
    docstring = _read_endpoint_docstring()
    null_block = docstring[docstring.find("[P3-NEW-2"):]
    assert "BUG" in null_block.upper() or "bug" in null_block, (
        "Falta diferenciación `null` operacional vs `null` BUG. Sin esto, "
        "frontend dev confunde estado normal con incidente."
    )
    # También debe nombrar `pipeline_metrics` como métrica a vigilar.
    assert "pipeline_metrics" in null_block, (
        "Docstring no menciona `pipeline_metrics` como métrica a vigilar "
        "cuando el null es bug."
    )
