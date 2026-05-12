"""[P3-NEXT-3 · 2026-05-11] Defensa contra bypass del helper
`update_meal_plan_data` (que adquiere advisory lock interno via P1-NEXT-1):
ningún archivo Python del backend (fuera de la whitelist) puede contener
`UPDATE meal_plans SET plan_data = …`. Cualquier mutación full-overwrite
de plan_data DEBE pasar por el helper o por uno de los callsites
canónicos en la whitelist (que tienen su propio advisory lock probado
por P1-NEW-B / P1-NEW-C / P1-OPEN-1 / P1-NEXT-1).

Cierra el gap residual del audit 2026-05-11:
    P1-NEW-B scanea `cron_tasks.py` con regex `::jsonb` literal.
    P1-NEW-C scanea `routers/plans.py` con la misma regex.
    P1-NEXT-1 ancla el lock interno en `db_plans.py`.
    Pero un archivo NUEVO (por ejemplo: `routers/new_feature.py`
    o `worker_service.py`) que añadiera `UPDATE meal_plans SET
    plan_data = %s WHERE id = %s` plano NO sería detectado por
    los tests anteriores. Este test cierra ese hueco.

Whitelist canónica (archivos donde el patrón es legítimo, cada uno
con su propio mecanismo de lock probado):
    - `db_plans.py` — el helper `update_meal_plan_data` y
      `update_plan_data_atomic`. Lock advisory interno (P1-NEXT-1)
      o FOR UPDATE row lock (update_plan_data_atomic).
    - `cron_tasks.py` — `_chunk_worker` T1/T2,
      `_background_shift_plan_for_user`. Lock cubierto por P1-NEW-B.
    - `routers/plans.py` — `api_shift_plan`, `api_restore_plan_local`.
      Lock cubierto por P1-NEW-C / P1-OPEN-1.

Cualquier otro callsite que necesite full-overwrite de plan_data DEBE:
  - Migrar a `jsonb_set` o `||` jsonb merge (atómicos, exentos I7), O
  - Llamar `update_meal_plan_data(plan_id, plan_data, user_id=...)`
    (que aplica el lock automáticamente por P1-NEXT-1).

Si un caso legítimo necesita full-overwrite directo (e.g., script
de mantenimiento one-shot con su propio lock), añadir whitelist
inline `# [P3-NEXT-3 WHITELIST: <razón ≥1 char>]` en las 30 líneas
previas al UPDATE.

Tooltip-anchor: P3-NEXT-3-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent

# Archivos donde el patrón es legítimo (cubiertos por otros tests).
_ALLOWED_FILES = {
    _BACKEND / "db_plans.py",
    _BACKEND / "cron_tasks.py",
    _BACKEND / "routers" / "plans.py",
}

# Directorios excluidos del scan (tests, deprecated, vendor).
_EXCLUDE_DIRS = {
    "tests",
    "__pycache__",
    "venv",
    "test_venv",
    "scratch",
    "scripts",
    "uploads",
    ".pytest_cache",
}

# Patrón material: `UPDATE meal_plans SET plan_data = …`.
# Cubre ambos estilos: `= %s::jsonb` y `= %s` (adapter Jsonb).
_RAW_UPDATE_RE = re.compile(
    r"UPDATE\s+meal_plans\s+SET\s+plan_data\s*=\s*%s",
    re.IGNORECASE,
)

# Whitelist marker inline.
_WHITELIST_RE = re.compile(
    r"#\s*\[P3-NEXT-3\s+WHITELIST\s*:\s*(?P<reason>.+?)\]",
)

# Ventana de lookback para el marker.
_WHITELIST_LOOKBACK_LINES = 30


def _iter_backend_py_files():
    for path in _BACKEND.rglob("*.py"):
        # Saltar archivos absolutos en la whitelist.
        if path in _ALLOWED_FILES:
            continue
        # Saltar directorios excluidos.
        if any(part in _EXCLUDE_DIRS for part in path.parts):
            continue
        # Saltar archivos backup `.bak` (aunque rglob `*.py` ya los excluye).
        if path.suffix != ".py":
            continue
        yield path


def _find_raw_updates(lines: list[str]) -> list[tuple[int, str]]:
    """Devuelve [(line_no, snippet)] de matches al patrón."""
    matches: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Saltar comentarios y docstrings de una línea.
        if stripped.startswith("#"):
            continue
        if _RAW_UPDATE_RE.search(line):
            matches.append((idx, stripped))
    return matches


def _has_whitelist_marker(lines: list[str], target_line: int) -> str | None:
    """¿Hay marker whitelist en las N líneas previas?"""
    start = max(0, target_line - 1 - _WHITELIST_LOOKBACK_LINES)
    end = target_line - 1
    window = "\n".join(lines[start:end])
    m = _WHITELIST_RE.search(window)
    return m.group("reason").strip() if m else None


# ---------------------------------------------------------------------------
# 1. Contrato principal
# ---------------------------------------------------------------------------
def test_no_raw_update_meal_plans_outside_allowed_files():
    """Ningún archivo Python del backend (fuera de la whitelist) puede
    contener `UPDATE meal_plans SET plan_data = …`. Migrar al helper
    `update_meal_plan_data` (que ahora aplica lock interno por P1-NEXT-1)
    o a `jsonb_set` / `||` jsonb merge atómicos.
    """
    offenders: list[str] = []
    scanned_count = 0

    for path in _iter_backend_py_files():
        scanned_count += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            pytest.fail(f"No pudo leer {path}: {e}")

        sites = _find_raw_updates(lines)
        if not sites:
            continue

        for line_no, snippet in sites:
            whitelist_reason = _has_whitelist_marker(lines, line_no)
            if whitelist_reason:
                continue
            rel_path = path.relative_to(_BACKEND).as_posix()
            offenders.append(f"  {rel_path}:{line_no} → {snippet[:140]}")

    # Sanity: el scan cubre archivos (no es un no-op silencioso si
    # `_BACKEND` cambia).
    assert scanned_count >= 5, (
        f"P3-NEXT-3 sanity: solo {scanned_count} archivos Python escaneados "
        f"en {_BACKEND}. ¿Cambió la estructura del backend? Verifica que "
        f"`_BACKEND` y `_EXCLUDE_DIRS` siguen vigentes."
    )

    assert not offenders, (
        f"P3-NEXT-3 violation: {len(offenders)} archivo(s) tienen "
        "`UPDATE meal_plans SET plan_data = …` FUERA de la whitelist "
        "canónica (db_plans.py, cron_tasks.py, routers/plans.py).\n\n"
        "Sin estar en la whitelist, el UPDATE bypassa el advisory lock "
        "interno del helper (P1-NEXT-1) → race lost-update silente.\n\n"
        "Offenders:\n"
        + "\n".join(offenders)
        + "\n\nFix (orden de preferencia):\n"
        "  1. Migrar a `jsonb_set` o `||` jsonb merge (atómicos, exentos I7).\n"
        "  2. Llamar `update_meal_plan_data(plan_id, plan_data, user_id=...)` "
        "(lock interno automático).\n"
        "  3. Si one-shot legítimo con su propio lock, añadir marker:\n"
        "       # [P3-NEXT-3 WHITELIST: razón explícita]\n"
        "     en las 30 líneas previas al UPDATE."
    )


# ---------------------------------------------------------------------------
# 2. Whitelist canónica de archivos sigue presente (anchor)
# ---------------------------------------------------------------------------
def test_allowed_files_anchor_present():
    """Defensa contra renombre/movimiento de los 3 archivos canónicos.
    Si alguno desaparece, el scan dejaría de cubrir su patrón legítimo —
    convirtiéndolo en falso positivo masivo o falso negativo masivo."""
    missing: list[str] = []
    for f in _ALLOWED_FILES:
        if not f.exists():
            missing.append(str(f.relative_to(_BACKEND)))
    assert not missing, (
        f"P3-NEXT-3 anchor lost: archivos canónicos de la whitelist "
        f"no encontrados: {missing}. ¿Renombre? Actualizar `_ALLOWED_FILES`."
    )


# ---------------------------------------------------------------------------
# 3. Test self-check: la whitelist NO debe contener archivos vacíos
#     (regresión contra refactor que mueve el patrón sin actualizar test)
# ---------------------------------------------------------------------------
def test_each_allowed_file_actually_has_the_pattern():
    """Defensa contra drift: si uno de los 3 archivos whitelisted YA NO
    contiene el patrón (todos migraron a jsonb_set), podemos remover la
    excepción. Sin esto, la whitelist crece mientras el código se
    sanea — perdemos detección."""
    files_without_pattern: list[str] = []
    for f in _ALLOWED_FILES:
        if not f.exists():
            continue  # cubierto por test #2
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if not _RAW_UPDATE_RE.search(text):
            files_without_pattern.append(str(f.relative_to(_BACKEND)))

    # Si TODOS los 3 perdieron el patrón, el test es no-op residual y se
    # puede borrar manualmente. Si solo algunos, alertamos para reducir
    # la whitelist y aumentar la cobertura del scan.
    if files_without_pattern and len(files_without_pattern) < len(_ALLOWED_FILES):
        pytest.fail(
            f"P3-NEXT-3 drift: {len(files_without_pattern)} archivo(s) "
            f"en la whitelist ya NO contienen el patrón `UPDATE meal_plans "
            f"SET plan_data = …`: {files_without_pattern}.\n\n"
            "Removerlos de `_ALLOWED_FILES` aumenta la cobertura del scan. "
            "Si el patrón fue migrado a jsonb_set/helper, ya no necesita "
            "estar en la whitelist."
        )


# ---------------------------------------------------------------------------
# 4. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p3_next_3"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "Filename debe contener slug `p3_next_3` para cross-link con "
        "test_p2_hist_audit_14."
    )
