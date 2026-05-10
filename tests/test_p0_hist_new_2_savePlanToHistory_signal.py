"""[P0-HIST-NEW-2 · 2026-05-09] Regression guard: gap de invalidación
entre `Plan.jsx::savePlanToHistory` y `History.jsx`.

Bug observado en el audit 2026-05-09:
    `Plan.jsx::savePlanToHistory` (~línea 398) inserta directo a
    `meal_plans` vía supabase.js (no pasa por el backend), pero NO
    señalizaba a `History.jsx` que el listado quedó stale. El listener
    de `visibilitychange` de `History.jsx` tiene un threshold de 60s
    para evitar fetches espurios por alt-tabs cortos. Resultado: un
    usuario que guarda un plan en /plan y vuelve a /history en otra
    pestaña dentro de la ventana de 60s ve el listado pre-insert
    (sin el plan recién guardado) hasta refresh manual o re-mount.

Además: P1-HIST-NEW-7 (mismo audit) recreó el índice
`idx_chunk_lesson_telemetry_plan_week` que estaba ausente en prod
aunque la migración p1_3 lo declaraba — la nueva migración
`p1_hist_new_7_recreate_chunk_lesson_telemetry_plan_week_idx.sql`
es la SSOT del fix.

Fix:
    1. `Plan.jsx`: tras insert exitoso, escribe `mealfit_history_dirty_at`
       en `localStorage` con `Date.now()`.
    2. `History.jsx`: el listener `visibilitychange` lee la key y
       si su valor supera `_lastFetchedAtRef.current`, fuerza
       `fetchHistory` aunque `_stale < 60s`.
    3. Migración FK index garantiza que `/lifetime-lessons` y el cron
       `_record_chunk_lesson_telemetry` no degraden a seq-scan.

Este test parsea los sources (mismo patrón que los meta-tests de
formValidation) — drift detection cross-language frontend↔migrations.
NO ejecuta browser; solo valida que los anchors textuales estén
en su sitio.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLAN_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Plan.jsx"
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_MIGRATIONS_DIR = _REPO_ROOT / "supabase" / "migrations"
_INDEX_MIGRATION = (
    _MIGRATIONS_DIR
    / "p1_hist_new_7_recreate_chunk_lesson_telemetry_plan_week_idx.sql"
)

_STORAGE_KEY = "mealfit_history_dirty_at"


# ---------------------------------------------------------------------------
# 1. Plan.jsx: señaliza inserción exitosa
# ---------------------------------------------------------------------------
def test_plan_jsx_writes_localstorage_on_successful_insert():
    """`Plan.jsx::savePlanToHistory` debe escribir
    `mealfit_history_dirty_at` en `localStorage` tras un insert
    exitoso. Sin esto, `History.jsx` no sabe que el listado está
    stale y aplica su threshold de 60s ciegamente.
    """
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # La key debe aparecer textualmente.
    assert _STORAGE_KEY in src, (
        f"`{_STORAGE_KEY}` no encontrado en Plan.jsx. El handshake con "
        f"History.jsx se rompe — restaurar el `localStorage.setItem` en "
        f"el bloque `else` de savePlanToHistory."
    )
    # Debe estar en una llamada `setItem`.
    pattern = re.compile(
        r"localStorage\s*\.\s*setItem\s*\(\s*['\"]" + re.escape(_STORAGE_KEY) + r"['\"]"
    )
    assert pattern.search(src), (
        f"La key `{_STORAGE_KEY}` aparece en Plan.jsx pero no como "
        f"argumento de `localStorage.setItem`. El handshake requiere un "
        f"setItem explícito."
    )


def test_plan_jsx_signal_lives_in_success_branch():
    """El `setItem` debe vivir en el ELSE de `if (saveError)` — si
    señaliza incluso ante error, History refetcha y vuelve a no ver
    el plan (porque no se insertó), añadiendo ruido.
    """
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # Bloque desde `if (saveError)` hasta el cierre del catch (heurística:
    # buscar el tramo que contiene la key y verificar que esté después
    # de `} else {` y antes del próximo `} catch`).
    block_match = re.search(
        r"if\s*\(\s*saveError\s*\)\s*\{[^}]*\}\s*else\s*\{(?P<else_body>[\s\S]*?)\}\s*\}\s*catch",
        src,
    )
    assert block_match is not None, (
        "Estructura `if (saveError) {...} else {...} } catch` no encontrada "
        "en Plan.jsx::savePlanToHistory. Posible refactor que rompió el "
        "anchor — revisar y actualizar este test."
    )
    else_body = block_match.group("else_body")
    assert _STORAGE_KEY in else_body, (
        f"La key `{_STORAGE_KEY}` está en Plan.jsx pero NO en la rama "
        f"`else` de saveError. Debe ejecutarse SOLO en éxito — un signal "
        f"en error path causa fetches inútiles."
    )


def test_plan_jsx_signal_is_try_wrapped():
    """El `setItem` debe estar en un try/catch: localStorage tira
    `SecurityError` en modo private/incógnito de algunos browsers.
    Sin try, el side-effect rompe el flujo de guardado del plan.
    """
    src = _PLAN_JSX.read_text(encoding="utf-8")
    # Buscar el setItem y verificar que la línea/líneas anteriores
    # contengan un `try {`.
    setitem_pos = src.find(f"setItem('{_STORAGE_KEY}'")
    if setitem_pos < 0:
        setitem_pos = src.find(f'setItem("{_STORAGE_KEY}"')
    assert setitem_pos > 0, (
        "setItem call no encontrado — test_plan_jsx_writes_localstorage_on_successful_insert "
        "ya falló previamente con mejor mensaje."
    )
    # Buscar el `try {` más cercano hacia atrás (dentro de ~500 chars).
    preceding = src[max(0, setitem_pos - 500): setitem_pos]
    assert "try {" in preceding, (
        "El `setItem` de la señal NO está envuelto en try/catch. "
        "localStorage puede tirar SecurityError en modo privado — "
        "envolver en try { ... } catch { /* silent */ }."
    )


# ---------------------------------------------------------------------------
# 2. History.jsx: bypass del threshold cuando hay señal fresca
# ---------------------------------------------------------------------------
def test_history_jsx_reads_dirty_signal():
    """`History.jsx` debe leer `mealfit_history_dirty_at` desde
    localStorage en el listener de `visibilitychange`. Sin esto, el
    bypass del threshold no ocurre y el bug original persiste."""
    src = _HISTORY_JSX.read_text(encoding="utf-8")
    assert _STORAGE_KEY in src, (
        f"`{_STORAGE_KEY}` no encontrado en History.jsx. El bypass del "
        f"threshold de 60s nunca dispara — restaurar el helper "
        f"`_isHistoryDirtySinceLastFetch`."
    )
    pattern = re.compile(
        r"getItem\s*\(\s*['\"]" + re.escape(_STORAGE_KEY) + r"['\"]"
    )
    assert pattern.search(src), (
        f"La key `{_STORAGE_KEY}` aparece en History.jsx pero no en un "
        f"`localStorage.getItem`. El handshake requiere un read explícito."
    )


def test_history_jsx_dirty_bypasses_stale_threshold():
    """El listener `visibilitychange` debe forzar `fetchHistory()`
    cuando la señal es más nueva que `_lastFetchedAtRef`, aunque
    `_stale < _STALE_MS`. Verificamos que el `return` early del
    threshold dependa AMBOS: stale Y NOT dirty.
    """
    src = _HISTORY_JSX.read_text(encoding="utf-8")
    # Buscar el patrón `if (_stale < _STALE_MS && !_dirty) return;` —
    # estricto a la lógica del fix.
    pattern = re.compile(
        r"if\s*\(\s*_stale\s*<\s*_STALE_MS\s*&&\s*!\s*_dirty\s*\)\s*return"
    )
    assert pattern.search(src), (
        "El early-return del listener no combina `_stale < _STALE_MS` "
        "con `!_dirty`. Sin el `&& !_dirty`, una señal fresca de Plan.jsx "
        "queda ignorada por el threshold de 60s. Restaurar el bypass."
    )


def test_history_jsx_compares_signal_to_last_fetch_ref():
    """La señal solo debe forzar refetch si su timestamp supera
    `_lastFetchedAtRef.current`. Comparar contra `Date.now()` directo
    haría que el flag quede perpetuamente "dirty" tras el primer
    setItem y dispare refetch en cada visibilitychange.
    """
    src = _HISTORY_JSX.read_text(encoding="utf-8")
    # Buscar `ts > _lastFetchedAtRef.current` cerca de la lectura de la key.
    # Heurística: la comparación debe estar dentro de 800 chars del getItem.
    getitem_match = re.search(
        r"getItem\s*\(\s*['\"]" + re.escape(_STORAGE_KEY) + r"['\"]",
        src,
    )
    assert getitem_match is not None, (
        "test_history_jsx_reads_dirty_signal ya falló — fix ese primero."
    )
    region = src[getitem_match.start(): getitem_match.start() + 800]
    assert re.search(r"_lastFetchedAtRef\s*\.\s*current", region), (
        "La comparación de la señal NO referencia `_lastFetchedAtRef.current`. "
        "Comparar contra Date.now() directo crearía bucle de refetch — "
        "la señal solo es válida si supera el último fetch exitoso."
    )


# ---------------------------------------------------------------------------
# 3. Migración FK index recreada (P1-HIST-NEW-7 incluido en este fix)
# ---------------------------------------------------------------------------
def test_recreate_plan_week_index_migration_exists():
    """La migración que recrea `idx_chunk_lesson_telemetry_plan_week`
    debe estar en `supabase/migrations/`. Sin SSOT en source, un
    rollback/replay de la DB pierde el índice y el advisor
    `unindexed_foreign_keys` reaparece."""
    assert _INDEX_MIGRATION.exists(), (
        f"Migración faltante: {_INDEX_MIGRATION.name}. La SSOT del FK "
        f"index recreado debe vivir en `supabase/migrations/` para "
        f"sobrevivir a un replay."
    )


def test_recreate_plan_week_index_migration_idempotent():
    """La migración debe usar `CREATE INDEX IF NOT EXISTS` — un
    re-run no debe fallar si el índice ya existe en algún ambiente."""
    src = _INDEX_MIGRATION.read_text(encoding="utf-8")
    assert "CREATE INDEX IF NOT EXISTS" in src, (
        "La migración debe ser idempotente con `CREATE INDEX IF NOT EXISTS`. "
        "Sin eso, un replay falla con `relation already exists`."
    )
    assert "idx_chunk_lesson_telemetry_plan_week" in src, (
        "El nombre del índice no aparece en la migración. Verificar el "
        "DDL — el advisor de Supabase mira por nombre."
    )
    assert "(meal_plan_id, week_number)" in src.replace(" ", "").replace("\n", "").replace("(meal_plan_id,week_number)", "(meal_plan_id, week_number)") \
        or "meal_plan_id" in src and "week_number" in src, (
        "Las columnas del índice deben ser `(meal_plan_id, week_number)` "
        "para cubrir la FK Y servir la query del cron."
    )


# ---------------------------------------------------------------------------
# 4. Sanity: ambos archivos del frontend siguen existiendo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_PLAN_JSX, _HISTORY_JSX])
def test_frontend_source_paths_exist(path: Path):
    """Si Plan.jsx o History.jsx se mueven a otra ruta, este test falla
    loud antes de que los demás tests den falsos negativos."""
    assert path.exists(), (
        f"{path} no existe. Si el archivo se renombró/movió, actualizar "
        f"este test ANTES de mergear el rename — los demás tests del "
        f"P-fix dependen de estas rutas."
    )
