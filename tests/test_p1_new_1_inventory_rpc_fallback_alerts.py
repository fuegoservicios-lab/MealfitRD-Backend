"""[P1-NEW-1 · 2026-05-10] `apply_inventory_delta` RPC fallback DEBE:
  1. Emitir `system_alerts.inventory_rpc_fallback` con severity=critical.
  2. Consultar knob `MEALFIT_INVENTORY_RPC_STRICT` (via `_env_bool` para
     auto-registro en `_KNOBS_REGISTRY`).
  3. Re-raise la excepción RPC si strict=True (fail-loud), antes de caer
     al UPDATE legacy no-atómico.

Bug original (audit 2026-05-10):
    El except block del fallback solo emitía `logger.error`. Producción
    podía operar bajo lost-update race (path SELECT-MODIFY-WRITE legacy
    re-introducido) silenciosamente — el fix P0-4 quedaría desactivado
    sin que nadie lo notara.

Estrategia del test (parser estático sobre db_inventory.py):
    1. Localizar el except block para `rpc_err` en `add_or_update_inventory_item`.
    2. Verificar `supabase.table('system_alerts').upsert(...)` con
       `alert_key='inventory_rpc_fallback'` Y `severity='critical'`.
    3. Verificar `_env_bool('MEALFIT_INVENTORY_RPC_STRICT', False)`.
    4. Verificar `if _strict: raise` antes del UPDATE legacy.
    5. Verificar que `upsert` está dentro de try/except (no crashea la
       transacción principal si la inserción de alert falla).

Drift detection:
    - Si alguien quita el alert → falla `test_fallback_emits_system_alert`.
    - Si alguien borra el knob → falla `test_strict_knob_consulted`.
    - Si alguien deja el strict siempre off → falla `test_strict_mode_reraises`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_INVENTORY_PY = _REPO_ROOT / "backend" / "db_inventory.py"


@pytest.fixture(scope="module")
def db_inventory_src() -> str:
    return _DB_INVENTORY_PY.read_text(encoding="utf-8")


def test_fallback_emits_system_alert(db_inventory_src: str):
    """El except block debe upsert a `system_alerts` con
    `alert_key='inventory_rpc_fallback'` y severity crítica."""
    # Buscar upsert con el alert_key específico cerca de la marker P1-NEW-1.
    p1_block_match = re.search(
        r"P1-NEW-1.*?inventory_rpc_fallback",
        db_inventory_src,
        re.DOTALL,
    )
    assert p1_block_match, (
        "P1-NEW-1 regresión: no se encontró el anchor `P1-NEW-1` cerca "
        "de `inventory_rpc_fallback`. El alert system para el fallback "
        "RPC desapareció. Restaurar el upsert a system_alerts."
    )

    alert_pattern = re.compile(
        r'["\']alert_key["\']\s*:\s*["\']inventory_rpc_fallback["\']',
    )
    assert alert_pattern.search(db_inventory_src), (
        "P1-NEW-1 regresión: el upsert a system_alerts NO usa "
        "`alert_key='inventory_rpc_fallback'`. Sin este key, las alerts "
        "no aparecen agrupadas en `/admin/cron-health` ni se pueden "
        "monitorear desde Supabase."
    )

    severity_pattern = re.compile(
        r'["\']severity["\']\s*:\s*["\']critical["\']',
    )
    assert severity_pattern.search(db_inventory_src), (
        "P1-NEW-1 regresión: el alert no tiene severity=critical. La "
        "RPC NO debería fallar en producción; cuando falla, implica "
        "regresión y debe escalar al máximo nivel para que ops actúe."
    )


def test_strict_knob_consulted(db_inventory_src: str):
    """El knob `MEALFIT_INVENTORY_RPC_STRICT` debe leerse vía `_env_bool`
    (auto-registra en `_KNOBS_REGISTRY` consistente con P3-NEW-D).
    """
    pattern = re.compile(
        r'_env_bool\(\s*["\']MEALFIT_INVENTORY_RPC_STRICT["\']\s*,\s*(False|True)',
    )
    assert pattern.search(db_inventory_src), (
        "P1-NEW-1 regresión: el knob `MEALFIT_INVENTORY_RPC_STRICT` ya "
        "no se lee vía `_env_bool`. Si alguien lo migró a "
        "`os.environ.get` directo, perdemos el auto-registro en "
        "`_KNOBS_REGISTRY` y `/admin/knobs` no lo expone."
    )


def test_strict_default_is_false(db_inventory_src: str):
    """Default del knob debe ser `False` — la migración a STRICT debe
    ser un opt-in operacional explícito tras verificar que la RPC es
    estable. Default True forzaría a TODOS los deploys hacer fail-loud
    incluso durante el deploy lag inicial post-rollout de la RPC.
    """
    pattern = re.compile(
        r'_env_bool\(\s*["\']MEALFIT_INVENTORY_RPC_STRICT["\']\s*,\s*False\b',
    )
    assert pattern.search(db_inventory_src), (
        "P1-NEW-1 regresión: el default de `MEALFIT_INVENTORY_RPC_STRICT` "
        "ya no es `False`. Si pasó a True, todos los deploys posteriores "
        "fallan loud incluso en deploy-lag legítimo. Cambiar a False y "
        "documentar el opt-in en CLAUDE.md."
    )


def test_strict_mode_reraises(db_inventory_src: str):
    """El bloque debe `raise` si `_strict` es True, ANTES del UPDATE
    legacy. Sin esto el knob es decorativo (alert se emite pero el
    fallback no-atómico corre igual)."""
    # Patrón: `if _strict:\n raise` cerca del marker P1-NEW-1.
    pattern = re.compile(
        r"if\s+_strict\s*:\s*[\r\n]+\s*(?:#[^\r\n]*[\r\n]+\s*)*\s*raise\b",
    )
    assert pattern.search(db_inventory_src), (
        "P1-NEW-1 regresión: no se encontró `if _strict: raise` después "
        "del upsert. Sin este re-raise, el knob STRICT no tiene efecto "
        "y el código siempre cae al UPDATE legacy no-atómico (lost-update "
        "race re-introducido del path P0-4)."
    )


def test_upsert_wrapped_in_try_except(db_inventory_src: str):
    """El upsert a system_alerts debe estar dentro de try/except. Si
    Supabase está caído o el schema cambió, no debemos crashear el
    flujo principal de la deducción (que ya está manejando un error
    del path primario).
    """
    # Encontrar el upsert con alert_key='inventory_rpc_fallback' y
    # verificar que hacia arriba (en las últimas ~30 líneas) hay un
    # `try:` y hacia abajo (en las próximas ~30 líneas) hay un
    # `except Exception as _alert_err`.
    upsert_match = re.search(
        r'supabase\.table\(\s*["\']system_alerts["\']\s*\)\.upsert',
        db_inventory_src,
    )
    assert upsert_match, (
        "P1-NEW-1: no se encontró el upsert a system_alerts en "
        "db_inventory.py. Si fue movido a otro módulo, actualizar el test."
    )

    upsert_start = upsert_match.start()
    # Mirar las 1000 chars previas: debe haber un `try:` cercano.
    preceding = db_inventory_src[max(0, upsert_start - 1000):upsert_start]
    assert re.search(r"\btry\s*:", preceding), (
        "P1-NEW-1 regresión: el upsert a `system_alerts` no está "
        "precedido por un `try:` en su entorno inmediato. Si Supabase "
        "está caído cuando la RPC falla, perdemos el flujo primario. "
        "Restaurar el try defensivo alrededor del upsert."
    )

    # Mirar las 3500 chars siguientes: debe haber un `except Exception as
    # _alert_err`. Buffer amplio porque el upsert tiene un dict de metadata
    # multi-línea (~80 líneas con indentación).
    following = db_inventory_src[upsert_start:upsert_start + 3500]
    assert re.search(r"except\s+Exception\s+as\s+_alert_err", following), (
        "P1-NEW-1 regresión: el upsert a `system_alerts` no tiene un "
        "`except Exception as _alert_err` siguiente. Sin ese catch, "
        "una falla del upsert tira el flujo primario."
    )
