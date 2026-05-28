"""[P2-PROD-AUDIT-BUNDLE · 2026-05-27] Tests parser-based + funcionales para
el bundle P2 del audit prod-readiness 2026-05-27 (post-P2-CRONS-HEALTH-AGGREGATE).

3 sub-P2 cerrados:

  - **P2-1 (DELETE-EXPIRED-TEMPORAL-FACTS-DEBOUNCE)**: debounce in-process
    per-user en `delete_expired_temporal_facts` (`db_facts.py`). Pre-fix 3
    callsites side-effect emitían 3 DELETEs idénticos en ms durante el mismo
    pipeline run. Knob `MEALFIT_TEMPORAL_FACTS_CLEANUP_DEBOUNCE_S` default 60s
    clamp [5, 3600].

  - **P2-2 (USER-FACTS-AUTOVACUUM)**: migración SSOT que extiende el patrón
    P1-B (scale_factor=0.05, threshold=25) a `user_facts` (la única tabla
    user-scoped del audit P1-B que quedó fuera). pg_stat_user_tables 2026-05-27:
    6 live / 20 dead = 333% dead_pct.

  - **P2-3 (DEPLOY-LAG-AGE-CHECK)**: script CLI standalone
    `backend/scripts/check_deploy_lag_age.py`. Compara `_LAST_KNOWN_PFIX`
    local vs `app_kv_store.expected_last_known_pfix`. Exit 1 si delta
    en días > threshold (default 14). Útil para el operador local-only
    cuya deuda de deploy no es visible al cron del backend.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_DB_FACTS = _BACKEND_ROOT / "db_facts.py"
_APP_PY = _BACKEND_ROOT / "app.py"
_MIGRATIONS_ROOT = _REPO_ROOT / "supabase" / "migrations"
_MIGRATIONS_BACKEND = _BACKEND_ROOT / "supabase" / "migrations"
_MIGRATION_FILENAME = "p2_prod_audit_bundle_user_facts_autovacuum_2026_05_27.sql"
_SCRIPT_PATH = _BACKEND_ROOT / "scripts" / "check_deploy_lag_age.py"

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ============================================================================
# Cross-link anchor (test_p2_hist_audit_14_marker_test_link)
# ============================================================================

def test_marker_anchor_present_in_db_facts():
    src = _DB_FACTS.read_text(encoding="utf-8")
    assert "P2-PROD-AUDIT-BUNDLE" in src, (
        "Falta anchor `P2-PROD-AUDIT-BUNDLE` en `db_facts.py`. Sin anchor, "
        "un refactor cosmético borraría la convención del debounce y este "
        "test fallaría buscando regresiones que no entendería."
    )


# ============================================================================
# P2-1 · DEBOUNCE delete_expired_temporal_facts
# ============================================================================

def test_p2_1_debounce_helper_exists():
    """El módulo expone `_should_skip_temporal_cleanup(user_id)`."""
    src = _DB_FACTS.read_text(encoding="utf-8")
    assert re.search(
        r"def\s+_should_skip_temporal_cleanup\s*\(", src
    ), (
        "P2-1: helper `_should_skip_temporal_cleanup` ausente en `db_facts.py`. "
        "Sin el helper, el debounce no aplica."
    )


def test_p2_1_debounce_called_in_delete_expired():
    """`delete_expired_temporal_facts` debe consultar el debounce ANTES de
    construir el query — un debounce que corre después no ahorra DB round-trip."""
    src = _DB_FACTS.read_text(encoding="utf-8")
    # Aislar el cuerpo de la función:
    body_match = re.search(
        r"def\s+delete_expired_temporal_facts\s*\([^)]*\)\s*:(.*?)(?=\ndef\s|\Z)",
        src,
        re.DOTALL,
    )
    assert body_match, "No se encontró def `delete_expired_temporal_facts`."
    body = body_match.group(1)

    # El skip-check debe aparecer antes de cualquier construcción de query
    # (`from datetime` / `supabase.table` / `_do_delete`).
    skip_idx = body.find("_should_skip_temporal_cleanup")
    assert skip_idx >= 0, (
        "P2-1: `delete_expired_temporal_facts` no consulta "
        "`_should_skip_temporal_cleanup`. El callsite se mantuvo a la antigua."
    )
    query_marker_idx = body.find("_do_delete")
    assert query_marker_idx > skip_idx, (
        "P2-1: el debounce check debe aparecer ANTES de construir `_do_delete` "
        "— si no, igual se paga el round-trip al DB."
    )


def test_p2_1_knob_clamp_5_to_3600():
    """El knob env tiene clamp defensivo [5, 3600] — typos del operador
    (`0`, `-1`, `999999`) caen al default 60 o al clamp."""
    src = _DB_FACTS.read_text(encoding="utf-8")
    assert "MEALFIT_TEMPORAL_FACTS_CLEANUP_DEBOUNCE_S" in src, (
        "P2-1: knob `MEALFIT_TEMPORAL_FACTS_CLEANUP_DEBOUNCE_S` ausente."
    )
    # Clamp inferior 5
    assert re.search(r"raw\s*<\s*5", src), (
        "P2-1: clamp inferior `raw < 5` ausente en `_temporal_cleanup_debounce_seconds`."
    )
    # Clamp superior 3600
    assert re.search(r"raw\s*>\s*3600", src), (
        "P2-1: clamp superior `raw > 3600` ausente."
    )


def test_p2_1_debounce_skips_within_window():
    """Funcional: dos calls consecutivos al mismo user dentro de la ventana
    → el segundo debe retornar True (skip)."""
    import db_facts as df  # type: ignore

    importlib.reload(df)
    # Reset state in case otro test lo ensució.
    df._TEMPORAL_CLEANUP_LAST_RUN.clear()

    user_id = "11111111-1111-1111-1111-111111111111"
    with patch.dict(os.environ, {"MEALFIT_TEMPORAL_FACTS_CLEANUP_DEBOUNCE_S": "60"}):
        first = df._should_skip_temporal_cleanup(user_id)
        second = df._should_skip_temporal_cleanup(user_id)
    assert first is False, "Primera llamada debe ejecutar (return False)."
    assert second is True, "Segunda llamada dentro de la ventana debe skip."


def test_p2_1_debounce_does_not_skip_for_global_cleanup():
    """`user_id=None` (cleanup global del cron) NO debouncea: esos son
    invocaciones explícitas, no side-effects de search."""
    import db_facts as df  # type: ignore

    importlib.reload(df)
    df._TEMPORAL_CLEANUP_LAST_RUN.clear()

    first = df._should_skip_temporal_cleanup(None)
    second = df._should_skip_temporal_cleanup(None)
    assert first is False
    assert second is False, (
        "`user_id=None` (cleanup global) NUNCA debe debouncearse. Pre-fix "
        "habría hecho que el cron diario sobre `sintoma_temporal` se skippeara "
        "después de cualquier search que tocara el helper."
    )


def test_p2_1_debounce_isolates_per_user():
    """Dos users distintos no comparten el debounce — el skip de user A no
    afecta el primer call de user B."""
    import db_facts as df  # type: ignore

    importlib.reload(df)
    df._TEMPORAL_CLEANUP_LAST_RUN.clear()

    user_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    user_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert df._should_skip_temporal_cleanup(user_a) is False
    assert df._should_skip_temporal_cleanup(user_b) is False
    assert df._should_skip_temporal_cleanup(user_a) is True
    assert df._should_skip_temporal_cleanup(user_b) is True


# ============================================================================
# P2-2 · USER-FACTS AUTOVACUUM TUNING
# ============================================================================

def test_p2_2_migration_exists_in_both_dirs():
    """SSOT [P3-MIGRATIONS-SSOT · 2026-05-20]: migration debe vivir en AMBOS
    dirs sincronizados (`supabase/migrations/` y `backend/supabase/migrations/`)."""
    workspace_path = _MIGRATIONS_ROOT / _MIGRATION_FILENAME
    backend_path = _MIGRATIONS_BACKEND / _MIGRATION_FILENAME
    assert workspace_path.exists(), (
        f"P2-2: migración SSOT ausente en {workspace_path}. "
        f"Sin SSOT, el tuning vive solo en memoria del operador."
    )
    assert backend_path.exists(), (
        f"P2-2: migración SSOT ausente en {backend_path}. "
        f"P3-MIGRATIONS-SSOT exige copia en ambos dirs (workspace + backend)."
    )


def test_p2_2_migration_content_identical_across_dirs():
    """Ambas copias deben ser byte-identical."""
    workspace_text = (_MIGRATIONS_ROOT / _MIGRATION_FILENAME).read_text(encoding="utf-8")
    backend_text = (_MIGRATIONS_BACKEND / _MIGRATION_FILENAME).read_text(encoding="utf-8")
    assert workspace_text == backend_text, (
        "P2-2: las 2 copias de la migración divergieron. "
        "P3-MIGRATIONS-SSOT exige byte-identical."
    )


def test_p2_2_migration_alters_user_facts():
    """`ALTER TABLE public.user_facts SET (...)` presente."""
    src = (_MIGRATIONS_ROOT / _MIGRATION_FILENAME).read_text(encoding="utf-8")
    assert re.search(
        r"ALTER\s+TABLE\s+public\.user_facts\s+SET\s*\(",
        src,
        re.IGNORECASE,
    ), (
        "P2-2: `ALTER TABLE public.user_facts SET (...)` ausente. "
        "El autovacuum tuning no aplicaría."
    )


def test_p2_2_migration_sets_four_autovacuum_params():
    """Patrón P1-B: 4 params autovacuum (vacuum + analyze × scale_factor + threshold)."""
    src = (_MIGRATIONS_ROOT / _MIGRATION_FILENAME).read_text(encoding="utf-8")
    expected = (
        "autovacuum_vacuum_scale_factor",
        "autovacuum_vacuum_threshold",
        "autovacuum_analyze_scale_factor",
        "autovacuum_analyze_threshold",
    )
    for param in expected:
        assert param in src, (
            f"P2-2: parameter `{param}` ausente. Patrón P1-B exige los 4 "
            f"(vacuum y analyze tienen umbrales independientes)."
        )


def test_p2_2_migration_uses_p1_b_thresholds():
    """Valores `scale_factor=0.05` + `threshold=25` (mismos que P1-B para
    coherencia operacional)."""
    src = (_MIGRATIONS_ROOT / _MIGRATION_FILENAME).read_text(encoding="utf-8")
    assert "scale_factor = 0.05" in src or "scale_factor=0.05" in src, (
        "P2-2: scale_factor distinto a 0.05 — perder coherencia con P1-B "
        "complicaría el debugging operacional."
    )
    assert "threshold = 25" in src or "threshold=25" in src, (
        "P2-2: threshold distinto a 25 — mismo problema de coherencia."
    )


def test_p2_2_migration_has_comment_anchor():
    """`COMMENT ON TABLE public.user_facts IS '[P2-PROD-AUDIT-BUNDLE ...]'`
    descubrible vía `\\d+ user_facts` post-deploy."""
    src = (_MIGRATIONS_ROOT / _MIGRATION_FILENAME).read_text(encoding="utf-8")
    assert re.search(
        r"COMMENT\s+ON\s+TABLE\s+public\.user_facts\s+IS\s+'\[P2-PROD-AUDIT-BUNDLE",
        src,
        re.IGNORECASE,
    ), (
        "P2-2: `COMMENT ON TABLE public.user_facts IS '[P2-PROD-AUDIT-BUNDLE ...'` "
        "ausente. Sin el comment, el operador no entiende por qué los defaults "
        "están tuneados al revisar `\\d+ user_facts`."
    )


# ============================================================================
# P2-3 · DEPLOY-LAG-AGE CLI SCRIPT
# ============================================================================

def test_p2_3_script_exists():
    assert _SCRIPT_PATH.exists(), (
        f"P2-3: script CLI ausente en {_SCRIPT_PATH}. "
        f"Sin el script, el operador local-only no tiene check on-demand."
    )


def test_p2_3_script_docstring_anchor():
    src = _SCRIPT_PATH.read_text(encoding="utf-8")
    assert "P2-PROD-AUDIT-BUNDLE" in src, (
        "P2-3: docstring del script sin anchor `P2-PROD-AUDIT-BUNDLE`. "
        "Anchor obligatorio para grep + cross-link al test."
    )


def test_p2_3_script_zero_backend_runtime_deps():
    """Como check_unpushed_age, este script NO debe importar runtime
    backend (`cron_tasks`, `db`, `services`, etc.). Pensado para correr
    pre-deploy / en git hook, sin spinning up el backend.

    Permitidos: stdlib + `httpx` (opcional, fetch del KV)."""
    src = _SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = (
        "from cron_tasks",
        "import cron_tasks",
        "from services",
        "import services",
        "from graph_orchestrator",
        "from app ",  # `from app import …`
    )
    for needle in forbidden:
        assert needle not in src, (
            f"P2-3: import prohibido `{needle}` en script CLI. "
            f"Romperia el principio 'zero deps backend runtime'."
        )


def test_p2_3_script_help_exits_zero():
    """Smoke: `--help` retorna 0 y menciona el bundle.

    [P2-OPS-BUNDLE pattern · 2026-05-26] encoding="utf-8", errors="replace"
    obligatorio en Windows (cp1252 no maneja `·` ni emojis)."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, (
        f"P2-3: `--help` no retorna 0 (rc={result.returncode}). "
        f"stderr: {result.stderr[:200]}"
    )
    assert "--max-age-days" in result.stdout, (
        "P2-3: `--help` no documenta `--max-age-days` — UX rota."
    )
    assert "--local-only" in result.stdout, (
        "P2-3: `--help` no documenta `--local-only`."
    )


def test_p2_3_script_local_only_exits_zero_when_marker_well_formed():
    """`--local-only --quiet` no consulta Supabase. Si el marker local es válido,
    debe exit 0 sin requerir SUPABASE_URL/SUPABASE_KEY."""
    # Borrar env vars Supabase temporalmente para asegurar que `--local-only` no
    # intenta fetch. Usamos patch.dict con clear=True sobre las keys relevantes.
    safe_env = dict(os.environ)
    for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY"):
        safe_env.pop(k, None)

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--local-only", "--quiet"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        env=safe_env,
    )
    assert result.returncode == 0, (
        f"P2-3: `--local-only --quiet` debe exit 0 con marker local válido "
        f"(rc={result.returncode}). stderr: {result.stderr[:300]}"
    )


def test_p2_3_script_fails_when_delta_exceeds_threshold():
    """Dry-run con `--kv-marker` antiguo + `--max-age-days 1` → exit 1."""
    # KV marker viejo: usamos una fecha 1995 garantizada >>1 día por debajo del
    # marker local actual.
    result = subprocess.run(
        [
            sys.executable, str(_SCRIPT_PATH),
            "--kv-marker", "P1-LEGACY · 1995-01-01",
            "--max-age-days", "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 1, (
        f"P2-3: con kv-marker antiguo + threshold 1d debe exit 1 "
        f"(rc={result.returncode}). stdout: {result.stdout[:300]} "
        f"stderr: {result.stderr[:300]}"
    )


def test_p2_3_script_passes_when_delta_within_threshold():
    """Dry-run con `--kv-marker` igual al local + threshold 0 → exit 0
    (delta=0 NO es > 0)."""
    # Leer el marker local actual para construir un kv-marker idéntico.
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']', text)
    assert m, "no se pudo leer _LAST_KNOWN_PFIX del app.py"
    local_marker = m.group(1)

    result = subprocess.run(
        [
            sys.executable, str(_SCRIPT_PATH),
            "--kv-marker", local_marker,
            "--max-age-days", "0",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, (
        f"P2-3: con kv-marker == local + threshold 0d debe exit 0 "
        f"(rc={result.returncode}). delta=0 NO es > 0. "
        f"stdout: {result.stdout[:300]} stderr: {result.stderr[:300]}"
    )


def test_p2_3_script_rejects_malformed_kv_marker():
    """Marker remoto mal formado → exit 2 (no 1: distingue 'lag' de 'data corrupta')."""
    result = subprocess.run(
        [
            sys.executable, str(_SCRIPT_PATH),
            "--kv-marker", "garbage-not-a-marker",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 2, (
        f"P2-3: kv-marker mal formado debe exit 2 (no 1) — distingue 'lag' de "
        f"'data corrupta'. rc={result.returncode}."
    )


# ============================================================================
# Smoke: el bundle no rompe imports de `db_facts`
# ============================================================================

def test_db_facts_still_imports_cleanly():
    """Después del bundle, `import db_facts` debe seguir funcionando.
    Un typo en el nuevo bloque rompería todo el módulo."""
    import db_facts as df  # noqa: F401
    importlib.reload(df)
    assert hasattr(df, "delete_expired_temporal_facts")
    assert hasattr(df, "_should_skip_temporal_cleanup")
    assert hasattr(df, "_temporal_cleanup_debounce_seconds")
