"""[P1-A · 2026-05-08] Cross-module drift detection contra runtime DDL.

Bug observado en el re-audit 2026-05-08:
  Logs Postgres (MCP get_logs) mostraban statements `CREATE TABLE IF NOT EXISTS
  system_alerts` y `ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS
  quality_alert_at` ejecutándose cada ~5 min, contradiciendo P2-NEW-G que
  dejó `_ensure_quality_alert_schema` como no-op stub. Causa-raíz hipotética:
  binary desplegado pre-P2-NEW-G o cron externo invocando script standalone
  (4 archivos: migrate_quality_alerts, alter_db, add_price_cols,
  migrate_subscriptions).

Fix:
  - 4 scripts standalone renombrados a `_deprecated_*.py.bak` (no cargables
    como módulo Python).
  - Migración SSOT `p1_a_consolidate_remaining_runtime_ddl.sql` cubre las
    columnas residuales (price_per_*, paypal_*).
  - Endpoint `/health/version` permite verificar el commit hash desplegado.

Este test es la red de seguridad: garantiza que NO se reintroduzca DDL en
runtime fuera de migrations/ y de scripts ya archivados. Mismo patrón que
P0/P1 cross-language drift detection (test_p0_form_6_required_fields_sync,
test_p3_5_bio_ranges_parity).

Cobertura:
  - Backend activo (excluyendo tests/, scripts/, _deprecated_*) sin
    `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`,
    `CREATE INDEX IF NOT EXISTS`.
  - Si se detecta match: el test reporta archivo:línea para diagnóstico
    rápido y guía al fix (consolidar a `supabase/migrations/<P-fix>.sql`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent

# Patrones de DDL runtime que debe vivir SOLO en supabase/migrations/.
# `(?i)` case-insensitive — algunos scripts viejos usaban `create table`.
_DDL_PATTERNS = [
    re.compile(r"(?i)CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS"),
    re.compile(r"(?i)ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS"),
    re.compile(r"(?i)CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS"),
]

# Directorios y archivos exentos:
#   - tests/                      : aserciones sobre DDL strings son válidas
#   - scripts/                    : helpers de ops one-off
#   - _deprecated_*.py.bak        : scripts archivados (no ejecutables)
#   - supabase/migrations/        : SSOT autorizado
#   - .venv/, node_modules/, etc. : deps externas
_EXCLUDED_DIR_NAMES = {
    "tests",
    "scripts",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".git",
    ".pytest_cache",
}

_EXCLUDED_FILE_GLOBS = (
    "_deprecated_*.py.bak",
    "_deprecated_*.py",
    "*.bak",
    "pyright_results.json",
    "conftest.py",
)


def _iter_active_backend_files():
    """Yields *.py bajo backend/ excluyendo dirs y archivos archivados."""
    for path in _BACKEND_ROOT.rglob("*.py"):
        # Excluir si cualquier ancestro está en _EXCLUDED_DIR_NAMES
        if any(part in _EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        # Excluir patterns archivados
        if any(path.match(g) for g in _EXCLUDED_FILE_GLOBS):
            continue
        yield path


_TRIPLE_DOUBLE = '"' * 3
_TRIPLE_SINGLE = "'" * 3


def _is_comment_or_docstring_only(line: str) -> bool:
    """Heurística pragmática (no parser AST): True si la línea es comentario
    `#` o un docstring de una sola línea con delimitadores triples a ambos
    lados. Línea con código + comentario inline devuelve False — para ese
    caso `_strip_inline_comment` quita el comentario antes del scan.
    """
    stripped = line.lstrip().rstrip()
    if stripped.startswith("#"):
        return True
    if stripped.startswith(_TRIPLE_DOUBLE) and stripped.endswith(_TRIPLE_DOUBLE) and len(stripped) >= 6:
        return True
    if stripped.startswith(_TRIPLE_SINGLE) and stripped.endswith(_TRIPLE_SINGLE) and len(stripped) >= 6:
        return True
    return False


def _strip_inline_comment(line: str) -> str:
    """Devuelve la línea sin comentario inline, conservativo:
    encuentra el primer `#` que NO esté entre comillas. Si no se puede
    determinar con seguridad (comilla sin cerrar), devuelve la línea entera.
    """
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _scan_pattern(pattern):
    """Escanea backend activo y retorna lista de hits formateados."""
    matches = []
    for path in _iter_active_backend_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if _is_comment_or_docstring_only(line):
                continue
            code_only = _strip_inline_comment(line)
            if pattern.search(code_only):
                matches.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()[:120]}")
    return matches


def test_no_runtime_ddl_create_table_in_active_backend():
    """Ningún archivo activo debe contener `CREATE TABLE IF NOT EXISTS`.

    Si este test falla: el statement debe migrarse a un archivo bajo
    `supabase/migrations/<P-fix>_<descripcion>.sql` y el call site Python
    debe reemplazarse por un no-op (preservando call sites para no romper
    tests con `patch(...)`) o eliminarse si no hay test dependency.
    """
    matches = _scan_pattern(_DDL_PATTERNS[0])
    assert not matches, (
        "Runtime DDL `CREATE TABLE IF NOT EXISTS` detectado en código activo.\n"
        "Consolidar a supabase/migrations/. Hallazgos:\n  - "
        + "\n  - ".join(matches)
    )


def test_no_runtime_ddl_add_column_in_active_backend():
    """Ningún archivo activo debe contener `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`."""
    matches = _scan_pattern(_DDL_PATTERNS[1])
    assert not matches, (
        "Runtime DDL `ALTER TABLE ADD COLUMN IF NOT EXISTS` detectado en código activo.\n"
        "Consolidar a supabase/migrations/. Hallazgos:\n  - "
        + "\n  - ".join(matches)
    )


def test_no_runtime_ddl_create_index_in_active_backend():
    """Ningún archivo activo debe contener `CREATE INDEX IF NOT EXISTS`.

    Cierra patrón P1-NEW-A 2026-05-08 (runtime DDL recreaba dup indexes
    cada startup, deshacía P1-A drop).
    """
    matches = _scan_pattern(_DDL_PATTERNS[2])
    assert not matches, (
        "Runtime DDL `CREATE INDEX IF NOT EXISTS` detectado en código activo.\n"
        "Consolidar a supabase/migrations/. Hallazgos:\n  - "
        + "\n  - ".join(matches)
    )


def test_deprecated_scripts_are_archived():
    """Verifica que los 4 scripts standalone DDL están archivados.

    Si reaparece un `<name>.py` sin sufijo `.bak`, el test falla y guía
    al renombrado correcto. Garantiza que el fix de P1-A no se revierte
    por accidente (ej. un git revert de un commit cosmético).
    """
    expected_archived = [
        "_deprecated_migrate_quality_alerts.py.bak",
        "_deprecated_alter_db.py.bak",
        "_deprecated_add_price_cols.py.bak",
        "_deprecated_migrate_subscriptions.py.bak",
        # [P3-2 · 2026-05-08] Archivo `add_semantic_cache.py` cuyo DDL fue
        # consolidado a `supabase/migrations/p1_1_consolidate_semantic_cache_ddl.sql`.
        "_deprecated_add_semantic_cache.py.bak",
    ]
    missing = [
        name for name in expected_archived
        if not (_BACKEND_ROOT / name).is_file()
    ]
    assert not missing, (
        f"Scripts DDL deprecated faltan en backend/: {missing}. "
        "Si los renombraste, actualiza la lista en este test. "
        "Si fueron borrados, el test debe eliminarse junto con la entrada del registry."
    )

    # Y también: NO deben re-aparecer las versiones sin .bak
    forbidden_active = [
        "migrate_quality_alerts.py",
        "alter_db.py",
        "add_price_cols.py",
        "migrate_subscriptions.py",
    ]
    leaked = [
        name for name in forbidden_active
        if (_BACKEND_ROOT / name).is_file()
    ]
    assert not leaked, (
        f"Scripts standalone DDL reaparecieron sin sufijo .bak: {leaked}. "
        "Renombrar a `_deprecated_<name>.py.bak`. Ver P1-A 2026-05-08."
    )


def test_deprecated_scripts_have_deprecation_header_and_exec_guard():
    """[P2-C · 2026-05-08] Cada `.bak` archivado debe tener:
      1. Header `[DEPRECATED · P1-A 2026-05-08]` que apunte a la SSOT.
      2. `sys.exit(1)` antes del cuerpo histórico para refusar ejecución
         si alguien resuelve la ruta absoluta y hace `python <ruta>.bak`.

    Sin estas dos defensas, un cron externo configurado con la ruta
    absoluta podría seguir ejecutando el cuerpo legacy aunque el rename
    rompa el path lookup por nombre. P1-A hipótesis 2 ("cron externo
    ejecuta el script") quedaría sólo parcialmente mitigado.
    """
    archived_with_ssot_pointer = {
        "_deprecated_migrate_quality_alerts.py.bak":
            "supabase/migrations/p2_new_e_consolidate_runtime_ddl.sql",
        "_deprecated_alter_db.py.bak":
            "supabase/migrations/p1_2_missing_user_telemetry_tables.sql",
        "_deprecated_add_price_cols.py.bak":
            "supabase/migrations/p1_a_consolidate_remaining_runtime_ddl.sql",
        "_deprecated_migrate_subscriptions.py.bak":
            "supabase/migrations/p1_a_consolidate_remaining_runtime_ddl.sql",
        # [P3-2 · 2026-05-08] Pointer a la migración SSOT del semantic cache.
        "_deprecated_add_semantic_cache.py.bak":
            "supabase/migrations/p1_1_consolidate_semantic_cache_ddl.sql",
    }
    failures: list[str] = []
    for fname, ssot_path in archived_with_ssot_pointer.items():
        path = _BACKEND_ROOT / fname
        if not path.is_file():
            failures.append(f"{fname}: archivo no existe (ver test_deprecated_scripts_are_archived)")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            failures.append(f"{fname}: lectura falló: {e}")
            continue

        # Defensa 1: header DEPRECATED visible en las primeras 30 líneas
        header = "\n".join(content.splitlines()[:30])
        if "[DEPRECATED" not in header:
            failures.append(f"{fname}: falta header `[DEPRECATED · P1-A 2026-05-08]` en las primeras 30 líneas")
        if ssot_path not in content:
            failures.append(f"{fname}: falta puntero a SSOT (esperado: `{ssot_path}`)")

        # Defensa 2: sys.exit(1) antes del primer statement de cuerpo
        # (heurística: aparece en las primeras 40 líneas Y antes de cualquier
        # `def `, `class ` o llamada a psycopg.connect / cursor.execute).
        first_40 = content.splitlines()[:40]
        if not any("sys.exit(" in line and "1" in line for line in first_40):
            failures.append(f"{fname}: falta `sys.exit(1)` en las primeras 40 líneas para refusar ejecución")

    assert not failures, (
        "Defensas P2-C en `.bak` deprecated incompletas:\n  - "
        + "\n  - ".join(failures)
        + "\n\nReparar añadiendo el header y `sys.exit(1)` al inicio del archivo. "
        "Patrón: ver `_deprecated_migrate_quality_alerts.py.bak` como referencia."
    )


def test_deprecated_scripts_actually_refuse_execution():
    """[P2-C · 2026-05-08] Smoke run: invoca cada .bak con `python <path>` y
    verifica exit code != 0. Detecta regresiones donde alguien edite el
    header y rompa accidentalmente el `sys.exit` (ej. comentándolo).

    Skip en CI sin conda env / Python interpreter accesible, pero local
    debe ejecutar.
    """
    import subprocess
    import sys

    archived = [
        "_deprecated_migrate_quality_alerts.py.bak",
        "_deprecated_alter_db.py.bak",
        "_deprecated_add_price_cols.py.bak",
        "_deprecated_migrate_subscriptions.py.bak",
        # [P3-2 · 2026-05-08] Subprocess test cubre también el script del
        # semantic cache: `python <path>.bak` debe exit code != 0.
        "_deprecated_add_semantic_cache.py.bak",
    ]
    failures: list[str] = []
    for fname in archived:
        path = _BACKEND_ROOT / fname
        if not path.is_file():
            continue  # cubierto por test_deprecated_scripts_are_archived
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                timeout=10,
                cwd=str(_BACKEND_ROOT),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            failures.append(f"{fname}: invocación falló inesperadamente: {e}")
            continue
        if result.returncode == 0:
            failures.append(
                f"{fname}: terminó con exit code 0 — el guard `sys.exit(1)` "
                f"NO se está ejecutando. stdout={result.stdout[:200]!r} "
                f"stderr={result.stderr[:200]!r}"
            )
        if b"DEPRECATED" not in (result.stderr or b""):
            failures.append(
                f"{fname}: stderr NO menciona `DEPRECATED`. ops no recibe "
                f"señal clara de por qué falló: stderr={result.stderr[:300]!r}"
            )

    assert not failures, (
        "Defensas runtime P2-C fallaron:\n  - "
        + "\n  - ".join(failures)
    )


def test_health_version_endpoint_signature():
    """Smoke: el endpoint `/health/version` existe y devuelve los 6 campos
    documentados (git_sha, last_known_pfix, knobs_count, etc.).

    Este test es la única fuente de verdad sobre la API del endpoint.
    Si cambia la signature, debe actualizarse aquí — el SOP de diagnóstico
    en P1-A se basa en estos campos. Sin invocación HTTP: importamos la
    función directo para evitar dependencia del cliente Starlette.
    """
    import sys
    backend_path = str(_BACKEND_ROOT)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Import diferido: el import top-level de app dispara cron schedulers.
    # Importamos solo la función. Si app falla por env vars de DB en CI,
    # marcamos skip en lugar de error: el test de schema textual de los 3
    # patrones DDL ya cubre lo crítico.
    try:
        from app import health_version, _LAST_KNOWN_PFIX, _PROCESS_START_ISO  # noqa: F401
    except Exception as e:  # pragma: no cover (env-dependent)
        pytest.skip(f"app no importable en este entorno: {type(e).__name__}: {str(e)[:80]}")
        return

    result = health_version()
    expected_keys = {
        "git_sha",
        "git_short_sha",
        "deploy_timestamp",
        "process_started_at",
        "last_known_pfix",
        "knobs_count",
        "knobs_sample",
    }
    assert expected_keys.issubset(result.keys()), (
        f"`/health/version` cambió signature. Esperado ⊇ {expected_keys}, "
        f"got {set(result.keys())}. Actualizar SOP de diagnóstico en P1-A "
        "memory entry si esto es intencional."
    )
    # last_known_pfix debe seguir el formato `P<n>(-X)+ · YYYY-MM-DD`
    # [P3-1 · 2026-05-08] Regex permisivo para suffixes multi-segmento:
    # `P1-1`, `P2-A`, `P2-NEW-A`, `P3-CANDIDATE-B`, etc. La validación
    # estricta de freshness vive en `test_p3_1_last_known_pfix_freshness`.
    assert re.match(r"^P\d+(-[A-Z0-9]+)+\s+·\s+\d{4}-\d{2}-\d{2}$", result["last_known_pfix"]), (
        f"`_LAST_KNOWN_PFIX` no sigue el formato `Pn-X · YYYY-MM-DD`: "
        f"{result['last_known_pfix']!r}"
    )
    assert isinstance(result["knobs_count"], int)
    assert isinstance(result["knobs_sample"], list)
