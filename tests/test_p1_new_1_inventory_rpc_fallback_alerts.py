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
    2. Verificar el upsert SQL `INSERT INTO system_alerts ... ON CONFLICT
       (alert_key) DO UPDATE` con param `"inventory_rpc_fallback"` Y
       severity `'critical'` ([P1-NEON-DB-MIGRATION · 2026-06-12]: antes era
       `supabase.table('system_alerts').upsert(...)` — misma propiedad).
    3. Verificar `_env_bool('MEALFIT_INVENTORY_RPC_STRICT', False)`.
    4. Verificar `if _strict: raise` antes del UPDATE legacy.
    5. Verificar que el upsert está dentro de try/except (no crashea la
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


def _find_rpc_fallback_alert_insert(src: str):
    """Localiza el `INSERT INTO system_alerts` (upsert por alert_key) cuyo
    primer param es `"inventory_rpc_fallback"`. db_inventory tiene OTRO insert
    a system_alerts (P3-1 partial rollback) — el match exige que el literal del
    alert_key aparezca en la tupla de params inmediatamente después del SQL.

    [P1-NEON-DB-MIGRATION · 2026-06-12] Antes el upsert era
    `supabase.table('system_alerts').upsert({...})`; ahora es SQL directo
    `INSERT ... ON CONFLICT (alert_key) DO UPDATE` con params posicionales.
    Devuelve (start_pos, block_text) o (None, None)."""
    for m in re.finditer(r"INSERT INTO system_alerts", src):
        block = src[m.start():m.start() + 1500]
        if '"inventory_rpc_fallback"' in block:
            return m.start(), block
    return None, None


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

    insert_start, block = _find_rpc_fallback_alert_insert(db_inventory_src)
    assert insert_start is not None, (
        "P1-NEW-1 regresión: no hay `INSERT INTO system_alerts` cuyo param "
        "sea `\"inventory_rpc_fallback\"`. Sin este key, las alerts no "
        "aparecen agrupadas en `/admin/cron-health` ni se pueden monitorear."
    )

    # Upsert idempotente por alert_key (no flooding) + re-arm de resolved_at.
    assert "ON CONFLICT (alert_key) DO UPDATE" in block, (
        "P1-NEW-1 regresión: el INSERT del alert ya no es upsert por "
        "alert_key — cada fallo de RPC floodearía system_alerts."
    )
    assert re.search(r"resolved_at\s*=\s*NULL", block), (
        "P1-NEW-1 regresión: el DO UPDATE no re-arma `resolved_at = NULL` — "
        "una alert resuelta no volvería a dispararse en recurrencia."
    )

    assert re.search(r"VALUES\s*\(\s*%s,\s*'inventory',\s*'critical'", block), (
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
    """El upsert (INSERT ... ON CONFLICT) a system_alerts debe estar dentro de
    try/except. Si la DB está caída o el schema cambió, no debemos crashear el
    flujo principal de la deducción (que ya está manejando un error del path
    primario).
    """
    # Encontrar el INSERT con alert_key='inventory_rpc_fallback' y
    # verificar que hacia arriba hay un `try:` cercano y hacia abajo hay un
    # `except Exception as _alert_err`.
    upsert_start, _block = _find_rpc_fallback_alert_insert(db_inventory_src)
    assert upsert_start is not None, (
        "P1-NEW-1: no se encontró el upsert (INSERT ... ON CONFLICT) a "
        "system_alerts en db_inventory.py. Si fue movido a otro módulo, "
        "actualizar el test."
    )

    # Mirar las 2500 chars previas: debe haber un `try:` cercano (el bloque
    # construye _alert_message/_alert_metadata multi-línea antes del INSERT).
    preceding = db_inventory_src[max(0, upsert_start - 2500):upsert_start]
    assert re.search(r"\btry\s*:", preceding), (
        "P1-NEW-1 regresión: el upsert a `system_alerts` no está "
        "precedido por un `try:` en su entorno inmediato. Si la DB "
        "está caída cuando la RPC falla, perdemos el flujo primario. "
        "Restaurar el try defensivo alrededor del upsert."
    )

    # Mirar las 3500 chars siguientes: debe haber un `except Exception as
    # _alert_err`. Buffer amplio porque el INSERT lleva SQL + params
    # multi-línea con indentación.
    following = db_inventory_src[upsert_start:upsert_start + 3500]
    assert re.search(r"except\s+Exception\s+as\s+_alert_err", following), (
        "P1-NEW-1 regresión: el upsert a `system_alerts` no tiene un "
        "`except Exception as _alert_err` siguiente. Sin ese catch, "
        "una falla del upsert tira el flujo primario."
    )
