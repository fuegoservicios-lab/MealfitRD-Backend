"""[P2-NEW-E · 2026-05-11] Cuarto sweep en `_resolve_stale_scheduler_alerts`:
hard-cap del parent `scheduler_cascade_missed` por edad absoluta
(default `MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS=6`), INDEPENDIENTE
de la condición de estabilización del 3er sweep (P2-LIVE-1).

Gap cerrado:
    El 3er sweep (P2-LIVE-1) cierra el parent SOLO cuando 0 children
    abiertos AND 0 nuevos MISSED en
    `MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN` (default 60min).
    Bajo churn sostenido (23+ crons en bursts cada minuto-0,
    alguno cruzando grace), cada tick re-emite `scheduler_missed_<job>`
    antes de que la ventana se cumpla → el parent queda abierto
    indefinidamente (alert fatigue).

    Observado live 2026-05-11 18:08 UTC: 9 children re-emitiéndose
    cada hora durante 24h+, cascade parent abierto >24h sin que el
    3er sweep nunca pudiera disparar.

Drift detection:
    - Sweep hard-cap eliminado del cuerpo de
      `_resolve_stale_scheduler_alerts` → falla.
    - Filtro `triggered_at < NOW() - hard_cap_h` pierde el operador
      `<` (ej. cambia a `>` por error) → falla.
    - Knob `MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS` no leído via
      `_env_int` → falla.
    - Sentry breadcrumb `[CASCADE_HARD_CAP_REACHED]` desaparece → falla
      (SRE pierde señal de saturación persistente).
    - Tick observable (`_scheduler_alerts_sweep_tick`) deja de
      incluir `swept_cascade_hard_cap` → falla (post-mortem ciego).

Tooltip-anchor: P2-NEW-E-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"


def _read_function_body(source: str, fn_name: str) -> str:
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    next_def_pattern = re.compile(r"^(def |class |@)", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[m.start():next_def.start()]
    return source[m.start():]


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. _resolve_stale_scheduler_alerts contiene el hard-cap block
# ---------------------------------------------------------------------------
def test_hard_cap_block_present(cron_source: str):
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    assert body, "_resolve_stale_scheduler_alerts no encontrada."
    assert "P2-NEW-E" in body, (
        "P2-NEW-E violation: bloque hard-cap (marker `P2-NEW-E`) ausente "
        "en `_resolve_stale_scheduler_alerts`. Sin el sweep, parent "
        "`scheduler_cascade_missed` puede quedar abierto indefinidamente "
        "bajo churn sostenido — el 3er sweep (P2-LIVE-1) nunca cierra "
        "porque sus 60min sin nuevos MISSED nunca se cumplen."
    )


# ---------------------------------------------------------------------------
# 2. UPDATE force-resolve por edad
# ---------------------------------------------------------------------------
def test_force_resolve_filters_by_age(cron_source: str):
    """El UPDATE hard-cap DEBE filtrar `triggered_at < NOW() - make_interval(hours => HARDCAP)`
    SIN dependencia del estado de children. Mismo filtro `alert_key=
    'scheduler_cascade_missed' AND resolved_at IS NULL` del 3er sweep."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    upd = re.search(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)"
        r".*?alert_key\s*=\s*'scheduler_cascade_missed'"
        r".*?resolved_at\s+IS\s+NULL"
        r".*?triggered_at\s*<\s*NOW\(\)\s*-\s*make_interval\s*\(\s*hours\s*=>\s*%s",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    assert upd, (
        "P2-NEW-E violation: UPDATE hard-cap no contiene el filtro "
        "esperado `alert_key='scheduler_cascade_missed' AND resolved_at "
        "IS NULL AND triggered_at < NOW() - make_interval(hours => %s)`. "
        "Sin el operador `<` el sweep nunca cerraría; sin "
        "`scheduler_cascade_missed` literal escalaría alerts ortogonales."
    )


# ---------------------------------------------------------------------------
# 3. Knob MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS leído via _env_int
# ---------------------------------------------------------------------------
def test_hard_cap_knob_registered(cron_source: str):
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    assert re.search(
        r'_env_int\s*\(\s*["\']MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS["\']',
        body,
    ), (
        "P2-NEW-E violation: `MEALFIT_SCHEDULER_CASCADE_HARD_CAP_HOURS` "
        "no leído via `_env_int` (auto-registro `_KNOBS_REGISTRY`). "
        "Sin esto, `/health/version` no muestra el override del operador."
    )


# ---------------------------------------------------------------------------
# 4. Sentry breadcrumb [CASCADE_HARD_CAP_REACHED]
# ---------------------------------------------------------------------------
def test_sentry_breadcrumb_on_force_resolve(cron_source: str):
    """Cuando el hard-cap dispara DEBE emitir un breadcrumb a Sentry con
    el literal `[CASCADE_HARD_CAP_REACHED]` para que SRE pueda decidir
    si escalar capacidad."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    assert "[CASCADE_HARD_CAP_REACHED]" in body, (
        "P2-NEW-E violation: el sweep no emite breadcrumb "
        "`[CASCADE_HARD_CAP_REACHED]` a Sentry. Sin este señal, SRE no "
        "tiene forma de detectar saturación persistente del cluster "
        "(parent forzado a cerrarse N veces sin que la cascada cese)."
    )
    # Debe estar dentro de un try/except con import lazy de sentry_sdk
    # (patrón espejo P0-NEW-2-AUTOHEAL).
    assert re.search(
        r"import\s+sentry_sdk.{0,300}CASCADE_HARD_CAP_REACHED",
        body,
        re.DOTALL,
    ), (
        "P2-NEW-E violation: el breadcrumb Sentry no está envuelto en "
        "`import sentry_sdk` lazy + try/except. Patrón espejo de "
        "`_alert_scheduler_cascade_missed` (cron_tasks.py:1767-1774)."
    )


# ---------------------------------------------------------------------------
# 5. Tick observable extendido con swept_cascade_hard_cap
# ---------------------------------------------------------------------------
def test_tick_metadata_includes_hard_cap(cron_source: str):
    """El tick `_scheduler_alerts_sweep_tick` DEBE incluir
    `swept_cascade_hard_cap` y `cascade_hard_cap_children_open` en su
    metadata para que post-mortem pueda separar resoluciones por edad
    (saturación sostenida) vs por estabilización (recuperación natural)."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    assert '"swept_cascade_hard_cap"' in body, (
        "P2-NEW-E violation: metadata del tick observable no incluye "
        "`swept_cascade_hard_cap`. Sin este field, post-mortem no puede "
        "diferenciar resoluciones por hard-cap vs por estabilización."
    )
    assert '"cascade_hard_cap_children_open"' in body, (
        "P2-NEW-E violation: metadata no incluye "
        "`cascade_hard_cap_children_open`. Sin este field, post-mortem "
        "ciego al estado del cluster en el momento del force-resolve."
    )


# ---------------------------------------------------------------------------
# 6. Hard-cap es INDEPENDIENTE del 3er sweep (no requiere NOT EXISTS children)
# ---------------------------------------------------------------------------
def test_hard_cap_independent_of_children(cron_source: str):
    """Por diseño, el hard-cap fuerza cierre IGNORANDO el estado de los
    children — ese es exactamente el caso que el 3er sweep (P2-LIVE-1)
    no puede cubrir. Por eso el bloque marcado `P2-NEW-E` NO debe
    contener `NOT EXISTS` filtros sobre `scheduler_missed_%` / `scheduler_error_%`
    dentro de su UPDATE."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    # Encontrar el bloque P2-NEW-E (entre los markers que comienzan con
    # `[P2-NEW-E` y antes del siguiente `[Px-...` o el tick observable).
    m_start = body.find("[P2-NEW-E ")
    assert m_start >= 0, "Marker P2-NEW-E no encontrado en función."
    # Buscar el UPDATE específico del hard-cap (con make_interval(hours => %s))
    # y verificar que NO tiene NOT EXISTS dentro del mismo statement.
    upd_pattern = re.compile(
        r"UPDATE\s+system_alerts\s+SET\s+resolved_at\s*=\s*NOW\(\)\s+WHERE\s+"
        r"alert_key\s*=\s*'scheduler_cascade_missed'\s+AND\s+resolved_at\s+IS\s+NULL\s+"
        r"AND\s+triggered_at\s*<\s*NOW\(\)\s*-\s*make_interval\s*\(\s*hours\s*=>\s*%s\s*\)\s+"
        r"RETURNING",
        re.IGNORECASE | re.DOTALL,
    )
    hard_cap_section = body[m_start:m_start + 6000]
    assert upd_pattern.search(hard_cap_section), (
        "P2-NEW-E violation: UPDATE hard-cap dentro de la sección "
        "P2-NEW-E no es independiente de children (debería ser un "
        "single statement con SOLO 3 filtros: alert_key, resolved_at, "
        "triggered_at < edad). Si contiene NOT EXISTS, duplica la "
        "lógica del 3er sweep P2-LIVE-1 — el hard-cap pierde su "
        "propósito de cubrir el caso 'children re-emitiéndose'."
    )
