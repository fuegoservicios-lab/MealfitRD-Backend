"""[P1-A-COH-UPDATED-AT · 2026-05-10] El cron horario
`_aggregate_coherence_block_history_metrics` (watchdog del invariante
P2-2 sobre `_shopping_coherence_block_history`) DEBE filtrar por
`meal_plans.updated_at`, no por `created_at`.

Bug original (audit 2026-05-10):
    P0-2 (2026-05-10) añadió la columna `meal_plans.updated_at` + trigger
    BEFORE UPDATE + índice `idx_meal_plans_user_updated_at`. La intención
    explícita de la migración (ver `migrations/p0_2_meal_plans_updated_at.sql`
    líneas 13-23 y 89-100) era permitir que este cron volviese a usar
    `updated_at` después del workaround P0-OBS-1 que había caído a
    `created_at` cuando la columna no existía.

    Sin embargo, el cron quedó filtrando por `created_at`. Resultado:
      - Regeneraciones de planes >`MEALFIT_COHERENCE_METRICS_LOOKBACK_H`
        horas (default 1) NO se contaban en el agregado horario, pese a
        que pueden appendear nuevas entries al `_shopping_coherence_block_history`
        vía `_recompute_aggregates_after_swap` (P2-B 2026-05-08), retry
        del coherence guard, o cualquier flujo del orquestador que
        re-ejecute `assemble_plan_node`.
      - El índice `idx_meal_plans_user_updated_at` no tenía un solo
        consumer en código → trabajo de la migración sin uso.

Fix (P1-A · 2026-05-10):
    `backend/cron_tasks.py:_aggregate_coherence_block_history_metrics`
    cambia `.gte("created_at", cutoff)` → `.gte("updated_at", cutoff)`.
    El trigger `trg_meal_plans_set_updated_at` (BEFORE UPDATE FOR EACH
    ROW) garantiza que cualquier mutación al plan bumpee la columna,
    así que cualquier append al history cae dentro del lookback.

[P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: el cron migró de
PostgREST (builder `.gte(col, cutoff)`) a SQL directo via
`execute_sql_query`. El filtro vive ahora en el literal SQL
`WHERE updated_at >= %s` dentro de la misma función — los regex de este
test parsean esa forma. Misma propiedad, transporte nuevo.

Cobertura de este test (parser-based, no DB):
    1. La función usa `updated_at` en el filtro `WHERE <col> >= %s` de la
       query a `meal_plans` (no `created_at`).
    2. El comentario obsoleto que afirmaba "meal_plans NO tiene columna
       updated_at" desapareció (defense-in-depth contra revertir el fix
       sin actualizar la documentación).
    3. La migración P0-2 SSOT existe (sanity check del prerequisito).
    4. El marker `_LAST_KNOWN_PFIX` de app.py está bumpeado al cierre P1-A.

Out of scope:
    - Validación schema runtime contra DB real: `test_p0_2_meal_plans_updated_at.py`
      cubre que la migración declara la columna.
    - Reemplazo del fallback diario `_shopping_coherence_alert_job` (cron 04:00 UTC
      que escanea ventana 24h): sigue usando `created_at` deliberadamente
      porque su lookback es lo bastante amplio para no perder regeneraciones.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


# Raíz del repo: este test vive en backend/tests/, subir 2 niveles.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRON_PATH = _REPO_ROOT / "backend" / "cron_tasks.py"
_APP_PATH = _REPO_ROOT / "backend" / "app.py"
_MIGRATION_PATH = _REPO_ROOT / "migrations" / "p0_2_meal_plans_updated_at.sql"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


def _extract_function_block(src: str, fn_name: str) -> str:
    """Extrae el cuerpo textual de una función top-level por su `def`.

    Heurística: desde `def <fn_name>(` hasta el siguiente `^def ` o EOF.
    Suficientemente robusta para módulos donde las funciones top-level
    no están anidadas (caso de `cron_tasks.py`).
    """
    m = re.search(
        rf"^def\s+{re.escape(fn_name)}\s*\(.*?(?=^def\s|\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"Función `{fn_name}` no encontrada en el módulo."
    return m.group(0)


# ---------------------------------------------------------------------------
# 1. La query del cron filtra por `updated_at`, no `created_at`.
# ---------------------------------------------------------------------------
def test_cron_filters_by_updated_at_not_created_at():
    """Núcleo del fix: el filtro temporal sobre `meal_plans` debe usar
    `updated_at`. Si esto falla, el cron volvió a perder regeneraciones
    de planes viejos (regresión de P1-A → P0-OBS-1).

    [P1-NEON-DB-MIGRATION] El filtro es ahora el literal SQL
    `WHERE updated_at >= %s` dentro del `execute_sql_query` de la función
    (antes builder PostgREST `.gte("updated_at", ...)`)."""
    src = _read(_CRON_PATH)
    block = _extract_function_block(src, "_aggregate_coherence_block_history_metrics")

    # Debe haber al menos un `WHERE updated_at >= %s` — el SQL ejecutable
    # del fetch (el placeholder %s ancla a código real, no comentarios).
    has_updated_at_filter = bool(re.search(
        r"WHERE\s+updated_at\s*>=\s*%s", block,
    ))
    assert has_updated_at_filter, (
        "El cron debe filtrar `meal_plans` por `updated_at` (post-P0-2) "
        "via `WHERE updated_at >= %s` en el SQL del fetch. Si revertiste "
        "a `created_at`, perdiste regeneraciones de planes viejos — la "
        "intención explícita de la migración P0-2 era cerrar ese gap."
    )

    # Y NO debe haber filtro por `created_at` en el mismo bloque (el
    # trade-off documentado quedó obsoleto post-migración).
    has_created_at_filter = bool(re.search(
        r"WHERE\s+created_at\s*>=\s*%s", block,
    ))
    assert not has_created_at_filter, (
        "El cron NO debe filtrar por `created_at` post-P1-A. Si necesitas "
        "ambos filtros (e.g. cobertura ampliada), documenta el rationale "
        "en el comentario del bloque y actualiza este test para reflejar "
        "el nuevo contrato."
    )


# ---------------------------------------------------------------------------
# 2. El comentario obsoleto desapareció.
# ---------------------------------------------------------------------------
def test_obsolete_comment_about_missing_column_is_gone():
    """El comentario pre-P0-2 afirmaba "meal_plans NO tiene columna
    updated_at". Si vuelve a aparecer, alguien revertió el fix sin
    actualizar la documentación — el siguiente auditor leerá el
    comentario falso y replicará el workaround."""
    src = _read(_CRON_PATH)
    block = _extract_function_block(src, "_aggregate_coherence_block_history_metrics")

    # Patrón flexible: cualquier afirmación de que la columna no existe.
    bad_phrases = [
        "meal_plans NO tiene",
        "meal_plans no tiene",
        "columna `updated_at` (verificado contra information_schema",
    ]
    for phrase in bad_phrases:
        assert phrase not in block, (
            f"Comentario obsoleto presente: {phrase!r}. "
            f"Post-P0-2 la columna `updated_at` SÍ existe. Si necesitas "
            f"documentar limitaciones nuevas, redacta sin reproducir la "
            f"afirmación falsa."
        )


# ---------------------------------------------------------------------------
# 3. Migración prerequisito existe.
# ---------------------------------------------------------------------------
def test_p0_2_migration_prerequisite_exists():
    """Sanity check: el fix P1-A asume que la migración P0-2 está
    aplicada (columna + trigger + índice). Si la migración fue removida
    accidentalmente, el cron crasheará con 400 — abort el merge."""
    assert _MIGRATION_PATH.exists(), (
        f"Migración prerequisito P0-2 no encontrada en {_MIGRATION_PATH}. "
        f"Sin ella, el filtro `.gte('updated_at', ...)` produce 400 en "
        f"PostgREST. Restaurar antes de mergear este fix."
    )


# ---------------------------------------------------------------------------
# 4. Marker tiene formato válido.
# ---------------------------------------------------------------------------
# NOTA: el contrato es "el marker apunta al ÚLTIMO P-fix mergeado", no a
# uno específico. Tests `test_p3_1_last_known_pfix_freshness` (formato +
# floor de fecha) y `test_p2_hist_audit_14_marker_test_link` (cross-link
# slug↔test file) cubren la integridad del marker. Aquí solo
# verificamos que existe y tiene formato — sin atarlo a este P-fix
# concreto, porque P-fixes posteriores rotarán el valor legítimamente.
def test_last_known_pfix_marker_exists_and_has_valid_format():
    """El marker debe estar presente con formato `Pn-... · YYYY-MM-DD`.
    Validación más detallada (floor de fecha, cross-link a test) vive
    en P3-1 y P2-HIST-AUDIT-14."""
    src = _read(_APP_PATH)
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]',
        src,
    )
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    assert re.match(
        r"^P\d+(?:-[A-Z0-9]+)+\s+·\s+\d{4}-\d{2}-\d{2}$",
        marker,
    ), (
        f"Marker `_LAST_KNOWN_PFIX = {marker!r}` no matchea el formato "
        f"canónico `Pn-X[-Y...] · YYYY-MM-DD`."
    )
