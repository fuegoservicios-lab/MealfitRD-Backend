"""[P0-PROD-1 · 2026-05-12] `pipeline_metrics` INSERTs no deben incluir
`is_guest` en su column-list.

Production-readiness audit 2026-05-12 reveló que la columna
`pipeline_metrics.is_guest` es `GENERATED ALWAYS AS (user_id IS NULL)`,
mientras 12 INSERTs (1 en `backend/app.py` + 11 en `backend/cron_tasks.py`)
la incluían explícitamente en su column list, causando rechazo silencioso
de cada emit:

    cannot insert a non-DEFAULT value into column "is_guest"

Modo de fallo silencioso:
  - Cada tick observable de los runbooks P0-LIVE-1 / P2-LIVE-9 /
    P1-LIVE-4 / P3-LIVE-1 está envuelto en `try/except` con
    `logger.debug(...)` para que el INSERT fallido no rompa el cron
    host. Los `tick observable SIEMPRE` del repo se volvían silentes:
    SRE pierde la señal de que el cron corrió.
  - El error log de postgres se contamina (~10 errores/min en audit
    2026-05-12), enmascarando errores genuinos de otros subsistemas.
  - No hay alert porque el `try/except` lo come.

Fix (P0-A del audit):
  Quitar `is_guest` del column-list y `true` de los VALUES. La columna
  GENERATED se computa automáticamente desde `user_id IS NULL`.

Lo que este test enforza:
  A) Cualquier INSERT INTO pipeline_metrics en `app.py` o `cron_tasks.py`
     NO debe mencionar `is_guest` en la column list.
  B) Defensa-en-profundidad VALUES: el patrón
     `0, %s::jsonb, true)` (exacto del INSERT legacy) NO debe aparecer.
     Discrimina contra `jsonb_set(...,'{path}',%s::jsonb,true)` que
     legítimamente usa `true` como flag `create_missing`.

Tooltip-anchor: P0-PROD-1-PIPELINE-METRICS-NO-IS-GUEST
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TARGET_FILES = [
    _BACKEND_ROOT / "app.py",
    _BACKEND_ROOT / "cron_tasks.py",
]

# Patron para localizar bloques `INSERT INTO pipeline_metrics (cols)`.
# DOTALL para soportar column-list multi-línea (todos los INSERTs reales
# están splitted en 2 líneas).
_INSERT_PATTERN = re.compile(
    r"INSERT\s+INTO\s+pipeline_metrics\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)


def test_no_pipeline_metrics_insert_lists_is_guest():
    """A) Ningún INSERT INTO pipeline_metrics debe nombrar `is_guest` en
    su column list. La columna es GENERATED ALWAYS — postgres rechaza
    el INSERT con error.
    """
    violations = []
    for fp in _TARGET_FILES:
        text = fp.read_text(encoding="utf-8")
        for m in _INSERT_PATTERN.finditer(text):
            cols = m.group(1)
            if "is_guest" in cols:
                line_no = text[: m.start()].count("\n") + 1
                violations.append(f"{fp.name}:{line_no} — INSERT incluye is_guest")

    assert not violations, (
        "P0-PROD-1 violation: INSERT INTO pipeline_metrics no debe "
        "mencionar `is_guest` en la column list. La columna es "
        "`GENERATED ALWAYS AS (user_id IS NULL)` y postgres rechaza "
        "cualquier INSERT explícito con:\n"
        "  cannot insert a non-DEFAULT value into column \"is_guest\"\n\n"
        "Violations encontradas:\n  " + "\n  ".join(violations) + "\n\n"
        "Fix: quitar `is_guest` del column-list y el último argumento "
        "`true` de VALUES. Patrón correcto:\n"
        "  INSERT INTO pipeline_metrics\n"
        "    (user_id, session_id, node, duration_ms, retries,\n"
        "     tokens_estimated, confidence, metadata)\n"
        "  VALUES (NULL, NULL, %s, 0, 0, 0, 0, %s::jsonb)"
    )


def test_no_pipeline_metrics_legacy_values_pattern():
    """B) Defense-in-depth: si alguien refactorea quitando `is_guest`
    del column-list pero olvida quitar el `true` correspondiente de
    VALUES, el INSERT tendría 9 valores para 8 columnas. Postgres
    daría diferent error pero igualmente silent (mismo try/except).

    Patrón canónico VALUES post-fix: `(NULL, NULL, %s, 0, 0, 0, 0, %s::jsonb)`.

    El patrón legacy a bloquear es `0, %s::jsonb, true)` — exacto del
    INSERT pre-P0-PROD-1. NO confundir con `jsonb_set(pipeline_snapshot,
    '{path}', %s::jsonb, true)` cuyo token previo es path-string (NO `0`).
    """
    bad_pattern = re.compile(r"0,\s*%s::jsonb,\s*true\s*\)")
    violations = []
    for fp in _TARGET_FILES:
        text = fp.read_text(encoding="utf-8")
        for m in bad_pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            violations.append(f"{fp.name}:{line_no}")

    assert not violations, (
        "P0-PROD-1 violation (variante VALUES): patrón legacy "
        "`0, %s::jsonb, true)` encontrado en:\n  "
        + "\n  ".join(violations)
        + "\n\nEste patrón es el VALUES list del INSERT INTO "
        "pipeline_metrics pre-P0-PROD-1 (con `is_guest=true` al "
        "final). Quitar el `, true)` — la columna GENERATED computa "
        "is_guest automáticamente desde user_id IS NULL."
    )


def test_insert_target_files_exist():
    """Sanity: los archivos target del scan siguen existiendo.

    Si alguien mueve el código a otro módulo (e.g. extrae
    observability_ticks.py durante el refactor planeado en
    `runbook_mega_files_refactor_plan_2026_05_12.md`), actualizar
    `_TARGET_FILES` arriba antes del merge para que este test siga
    cubriendo todos los callsites.
    """
    missing = [str(fp) for fp in _TARGET_FILES if not fp.exists()]
    assert not missing, (
        f"Archivos target ausentes: {missing}. Si los moviste al "
        "refactor P1-A documentado, actualizar `_TARGET_FILES` en este "
        "test para que siga escaneando los nuevos paths."
    )
