"""[P0-HIST-NEW-2 · 2026-05-09 · migrado P3-DOC-1 · 2026-05-11] Regression
guard: handshake `mealfit_history_dirty_at` post-plan-save.

Histórico:
    Bug original (P0-HIST-NEW-2 · 2026-05-09): `Plan.jsx::savePlanToHistory`
    insertaba directo a `meal_plans` y NO señalizaba a `History.jsx` que
    el listado quedó stale. Threshold de 60s del listener
    `visibilitychange` ocultaba el plan recién guardado a usuarios que
    volvían a /history dentro de esa ventana. Fix original: setItem
    `mealfit_history_dirty_at` post-insert + bypass del threshold cuando
    `_dirty=true && ts > _lastFetchedAtRef.current`.

    Migrado en P3-DOC-1 (2026-05-11): audit confirmó que
    `Plan.jsx::savePlanToHistory` era DEAD CODE (0 callers cross-codebase).
    El backend ya persiste vía `services._save_plan_and_track_background`
    post-SSE-completion. La función se eliminó de Plan.jsx; el setItem se
    movió al callsite real `AssessmentContext.jsx::saveGeneratedPlan`
    (invocado cuando el usuario acepta el plan generado, justo tras la
    persistencia backend). El handshake con `History.jsx` se preserva
    intacto — sólo cambió quién emite la señal.

    Además: P1-HIST-NEW-7 (mismo audit 2026-05-09) recreó el índice
    `idx_chunk_lesson_telemetry_plan_week` — los tests de migración
    siguen vivos al final de este archivo.

Este test parsea los sources (drift detection cross-language
frontend↔migrations) — NO ejecuta browser; solo valida anchors textuales.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLAN_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Plan.jsx"
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_ASSESSMENT_CTX = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"
_MIGRATIONS_DIR = _REPO_ROOT / "supabase" / "migrations"
_INDEX_MIGRATION = (
    _MIGRATIONS_DIR
    / "p1_hist_new_7_recreate_chunk_lesson_telemetry_plan_week_idx.sql"
)

_STORAGE_KEY = "mealfit_history_dirty_at"


# ---------------------------------------------------------------------------
# 1. AssessmentContext.jsx::saveGeneratedPlan: emite la señal post-SSE-success
# ---------------------------------------------------------------------------
def test_assessment_context_writes_localstorage_signal():
    """`AssessmentContext.jsx::saveGeneratedPlan` debe escribir
    `mealfit_history_dirty_at` en localStorage. Es el callsite real
    post-SSE-completion (el backend ya persistió el plan en
    `_save_plan_and_track_background`). Sin esto, `History.jsx` no
    sabe que el listado está stale y aplica su threshold de 60s
    ciegamente.

    [P3-DOC-1 · 2026-05-11] Migrado desde Plan.jsx::savePlanToHistory
    (eliminada por dead code, 0 callers).
    """
    src = _ASSESSMENT_CTX.read_text(encoding="utf-8")
    assert _STORAGE_KEY in src, (
        f"`{_STORAGE_KEY}` no encontrado en AssessmentContext.jsx. El "
        f"handshake con History.jsx se rompe — restaurar el "
        f"`localStorage.setItem` en `saveGeneratedPlan` post-comentario "
        f"`NOTA: NO guardamos en Supabase aquí`."
    )
    pattern = re.compile(
        r"localStorage\s*\.\s*setItem\s*\(\s*['\"]" + re.escape(_STORAGE_KEY) + r"['\"]"
    )
    assert pattern.search(src), (
        f"La key `{_STORAGE_KEY}` aparece en AssessmentContext.jsx pero "
        f"no como argumento de `localStorage.setItem`. El handshake requiere "
        f"un setItem explícito."
    )


def test_assessment_context_signal_lives_in_save_generated_plan():
    """El `setItem` debe estar dentro de `saveGeneratedPlan` específicamente,
    no en cualquier otro punto del context. saveGeneratedPlan es el callsite
    invocado tras SSE-success; otros callers (e.g. restorePlan) no deben
    emitir la señal."""
    src = _ASSESSMENT_CTX.read_text(encoding="utf-8")
    # Localizar saveGeneratedPlan body
    fn_idx = src.find("const saveGeneratedPlan = async")
    assert fn_idx > 0, (
        "`const saveGeneratedPlan = async` no encontrado. Si renombraste "
        "la función o cambiaste su declaración, actualizar este test."
    )
    next_const = re.search(r"\n    const \w+\s*=", src[fn_idx + 20:])
    end = fn_idx + 20 + (next_const.start() if next_const else len(src) - fn_idx - 20)
    body = src[fn_idx: end]
    assert _STORAGE_KEY in body, (
        f"La key `{_STORAGE_KEY}` está en AssessmentContext.jsx pero NO "
        f"dentro de `saveGeneratedPlan`. Debe vivir ahí — otros callsites "
        f"emitirían la señal en momentos equivocados (e.g., restorePlan "
        f"con plan ya viejo causaría refetch innecesario)."
    )


def test_assessment_context_signal_is_try_wrapped():
    """El `setItem` debe estar en try/catch: localStorage tira
    `SecurityError` en private/incógnito. Sin try, el side-effect rompe
    el flujo de aceptación del plan."""
    src = _ASSESSMENT_CTX.read_text(encoding="utf-8")
    setitem_pos = src.find(f"setItem('{_STORAGE_KEY}'")
    if setitem_pos < 0:
        setitem_pos = src.find(f'setItem("{_STORAGE_KEY}"')
    assert setitem_pos > 0, (
        "setItem call no encontrado — test_assessment_context_writes_localstorage_signal "
        "ya falló previamente con mejor mensaje."
    )
    preceding = src[max(0, setitem_pos - 500): setitem_pos]
    assert "try {" in preceding, (
        "El `setItem` de la señal NO está envuelto en try/catch. "
        "localStorage puede tirar SecurityError en modo privado — "
        "envolver en try { ... } catch { /* silent */ }."
    )


def test_plan_jsx_save_plan_to_history_removed():
    """[P3-DOC-1 · 2026-05-11] La función `savePlanToHistory` fue eliminada
    de Plan.jsx (dead code, 0 callers). Anti-regression: si alguien la
    re-introduce con un INSERT directo, este test falla.

    Si genuinamente necesitas un fallback frontend-side de persist (e.g.,
    backend persist falló silente), crear endpoint backend
    `POST /api/plans/persist-from-stream` con auth + dedupe + INSERT
    atómico bajo `acquire_meal_plan_advisory_lock(purpose='general')`.
    NO restaurar el patrón directo `supabase.from('meal_plans').insert()`.
    """
    src = _PLAN_JSX.read_text(encoding="utf-8")
    assert "const savePlanToHistory = async" not in src, (
        "P3-DOC-1 regresión: la función `savePlanToHistory` reaparecio en "
        "Plan.jsx. Era dead code (0 callers cross-codebase) + violaba I6 "
        "(direct INSERT a meal_plans desde frontend). Si necesitas un "
        "fallback, crear endpoint backend (ver tooltip-anchor "
        "`P3-DOC-1-DEAD-CODE-REMOVED` en Plan.jsx)."
    )
    # Verificar que el INSERT directo no aparece en CÓDIGO ejecutable.
    # Ignoramos líneas que comienzan con `//` (comentarios) o `*` (JSDoc),
    # porque mi propio bloque de explicación en Plan.jsx menciona el pattern
    # como warning anti-regresión. La invariante real es: el CÓDIGO no debe
    # tener un INSERT directo. Test blanket P1-NEW-A
    # (test_p1_new_a_frontend_no_direct_meal_plans_write.py) cubre este
    # contrato a nivel global; acá solo hacemos sanity check específico.
    code_lines = [
        ln for ln in src.split("\n")
        if "supabase.from('meal_plans').insert" in ln
        and not ln.strip().startswith("//")
        and not ln.strip().startswith("*")
    ]
    assert not code_lines, (
        f"P3-DOC-1 regresión: encontradas {len(code_lines)} líneas con "
        f"`supabase.from('meal_plans').insert(...)` que NO son comentarios "
        f"en Plan.jsx. Violación de invariante I6. Líneas:\n  "
        + "\n  ".join(code_lines)
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
