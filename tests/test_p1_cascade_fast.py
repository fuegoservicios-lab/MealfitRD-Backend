"""[P1-CASCADE-FAST · 2026-05-12] Sweep fast-resolve del parent
`scheduler_cascade_missed` cuando los children estabilizan en ventana corta.

Caso real (audit live 2026-05-12):
    05:41:33 — 3 children scheduler_missed_* triggered (post-deploy burst).
    05:43:48 — los 3 children resueltos via listener P1-NEW-2.
    05:53:07 — detector cron crea parent `scheduler_cascade_missed`.
    06:13:29 — parent cerrado MANUALMENTE (~20min total).

El sweep #3 (P2-LIVE-1) hubiera esperado hasta 06:41:33 UTC para cerrar
(60min default desde último triggered) — 50min de alert critical visible
pese a que la cascada estaba estabilizada a las 05:43:48.

Diseño del fast-resolve (sweep #7):
    Cierra parent inmediato si:
      (1) parent abierto.
      (2) 0 children abiertos.
      (3) 0 children TRIGGERED en últimos
          MEALFIT_SCHEDULER_CASCADE_FAST_RESOLVE_MIN (default 10).
      (4) parent edad ≥ MEALFIT_SCHEDULER_CASCADE_FAST_RESOLVE_MIN_AGE_MIN
          (default 5).

Diferencia con sweep #3:
    #3 = ventana 60min desde último triggered (cascadas largas).
    #7 = ventana 10min + edad parent 5min (post-burst limpio).

Test parser-based: verifica anchor, knobs con clamps correctos, SQL del
sweep cubre las 4 condiciones, tick observable extendido.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_text() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor presente — load-bearing para cross-link P2-HIST-AUDIT-14
# ---------------------------------------------------------------------------
def test_anchor_present(cron_text):
    assert "P1-CASCADE-FAST" in cron_text, (
        "Anchor `P1-CASCADE-FAST` removido de cron_tasks.py — el cross-link "
        "del marker (`test_p2_hist_audit_14_marker_test_link`) y este test "
        "dependen del anchor para localizar el bloque del sweep."
    )


# ---------------------------------------------------------------------------
# 2. Knobs presentes con clamps correctos
# ---------------------------------------------------------------------------
def test_fast_resolve_min_knob_with_clamp(cron_text):
    """Knob `MEALFIT_SCHEDULER_CASCADE_FAST_RESOLVE_MIN` (default 10,
    clamp [5, 30])."""
    assert 'MEALFIT_SCHEDULER_CASCADE_FAST_RESOLVE_MIN", 10' in cron_text, (
        "Knob default 10min no encontrado. Si lo bumpeas, actualiza el doc."
    )
    # Clamp `max(5, min(..., 30))`.
    assert re.search(
        r"fast_resolve_min\s*=\s*max\(\s*5\s*,\s*min\(\s*fast_resolve_min\s*,\s*30\s*\)\s*\)",
        cron_text,
    ), (
        "Clamp [5, 30] no encontrado para fast_resolve_min — sin clamp un "
        "operador puede setear 1 (race con detector) o 999 (defeats the "
        "purpose, regresa al sweep #3)."
    )


def test_fast_resolve_min_age_knob_with_clamp(cron_text):
    """Knob `MEALFIT_SCHEDULER_CASCADE_FAST_RESOLVE_MIN_AGE_MIN` (default 5,
    clamp [1, 30])."""
    assert (
        'MEALFIT_SCHEDULER_CASCADE_FAST_RESOLVE_MIN_AGE_MIN", 5' in cron_text
    ), "Knob default 5min para edad mínima no encontrado."
    assert re.search(
        r"fast_resolve_min_age\s*=\s*max\(\s*1\s*,\s*min\(\s*fast_resolve_min_age\s*,\s*30\s*\)\s*\)",
        cron_text,
    ), "Clamp [1, 30] no encontrado para fast_resolve_min_age."


# ---------------------------------------------------------------------------
# 3. SQL del sweep cubre las 4 condiciones
# ---------------------------------------------------------------------------
def _extract_fast_resolve_block(text: str) -> str:
    """Localiza el bloque del fast-resolve sweep entre el anchor
    `[P1-CASCADE-FAST · 2026-05-12]` y el siguiente sweep / tick."""
    m = re.search(
        r"\[P1-CASCADE-FAST · 2026-05-12\].*?(?=\[P2-B-OBS|\Z)",
        text,
        re.DOTALL,
    )
    assert m, "Bloque P1-CASCADE-FAST no localizable — anchor o estructura cambió."
    return m.group(0)


def test_sql_filters_parent_alert_key(cron_text):
    block = _extract_fast_resolve_block(cron_text)
    assert "p.alert_key = 'scheduler_cascade_missed'" in block, (
        "Sweep debe filtrar por `p.alert_key = 'scheduler_cascade_missed'` "
        "(NO por LIKE genérico — eso cerraría children también)."
    )


def test_sql_only_open_parent(cron_text):
    block = _extract_fast_resolve_block(cron_text)
    assert "p.resolved_at IS NULL" in block, (
        "Sweep debe ignorar parents ya cerrados (idempotencia)."
    )


def test_sql_min_age_condition(cron_text):
    """Condición (4): parent triggered hace ≥ fast_resolve_min_age."""
    block = _extract_fast_resolve_block(cron_text)
    assert (
        "p.triggered_at < NOW() - make_interval(mins => %s)" in block
    ), (
        "Condición edad mínima del parent ausente. Sin esto el detector "
        "que justo emitió el parent en este tick puede ser cerrado en el "
        "mismo tick antes de ser visible (race)."
    )


def test_sql_no_children_open_condition(cron_text):
    """Condición (2): NOT EXISTS children abiertos."""
    block = _extract_fast_resolve_block(cron_text)
    assert re.search(
        r"NOT EXISTS\s*\(\s*SELECT 1 FROM system_alerts c.*?c\.resolved_at IS NULL.*?scheduler_missed_%%.*?scheduler_error_%%",
        block,
        re.DOTALL,
    ), "Subquery `NOT EXISTS` para children abiertos ausente o malformada."


def test_sql_no_recent_children_condition(cron_text):
    """Condición (3): NOT EXISTS children TRIGGERED en últimos
    fast_resolve_min."""
    block = _extract_fast_resolve_block(cron_text)
    assert re.search(
        r"NOT EXISTS\s*\(\s*SELECT 1 FROM system_alerts r.*?r\.alert_key <> 'scheduler_cascade_missed'.*?r\.triggered_at > NOW\(\) - make_interval\(mins => %s\)",
        block,
        re.DOTALL,
    ), (
        "Subquery `NOT EXISTS` para children recientes (en ventana corta) "
        "ausente o malformada. Sin esta condición el sweep cerraría "
        "prematuramente cuando una nueva cascada acaba de iniciar."
    )


def test_sql_excludes_self_from_recent_check(cron_text):
    """La subquery de `r.triggered_at > NOW() - X` debe excluir el propio
    parent — sin `r.alert_key <> 'scheduler_cascade_missed'`, el SELECT
    encuentra al propio parent (cuya triggered_at puede estar dentro de la
    ventana corta) y BLOQUEA el cierre eternamente."""
    block = _extract_fast_resolve_block(cron_text)
    # El exclude del propio cascade key debe estar dentro del segundo NOT EXISTS.
    assert (
        "r.alert_key <> 'scheduler_cascade_missed'" in block
    ), (
        "Falta `r.alert_key <> 'scheduler_cascade_missed'` en la subquery "
        "de children recientes — el sweep nunca cerraría porque encuentra "
        "su propio parent en la ventana."
    )


# ---------------------------------------------------------------------------
# 4. Best-effort: try/except + log warning, NO raise
# ---------------------------------------------------------------------------
def test_sweep_is_best_effort(cron_text):
    block = _extract_fast_resolve_block(cron_text)
    assert "fast_resolve_failed = True" in block, (
        "Flag `fast_resolve_failed` ausente — el tick observable lo necesita "
        "para registrar fallos del sweep sin abortar los demás sweeps."
    )
    assert "best-effort" in block.lower(), (
        "Comentario `best-effort` ausente — patrón establecido en los 6 "
        "sweeps existentes; futuros readers asumen que la excepción aborta "
        "el cron entero si NO se documenta."
    )


# ---------------------------------------------------------------------------
# 5. Tick observable extendido con keys nuevas
# ---------------------------------------------------------------------------
def test_tick_includes_swept_count_in_total(cron_text):
    """`swept_count` total debe sumar `fast_resolve_swept`."""
    assert "+ fast_resolve_swept" in cron_text, (
        "El total `swept_count` del tick observable NO incluye "
        "`fast_resolve_swept` — sub-totales pierden 1 categoría."
    )


def test_tick_includes_fast_resolve_keys(cron_text):
    """3 keys nuevas en metadata del tick observable."""
    for key in (
        "swept_cascade_fast_resolve",
        "cascade_fast_resolve_min",
        "cascade_fast_resolve_min_age_min",
        "fast_resolve_sweep_failed",
    ):
        assert f'"{key}"' in cron_text, (
            f"Key `{key}` ausente en metadata del tick observable. "
            f"Sin esta señal, post-mortem no puede correlacionar fast-resolve "
            f"vs hard-cap vs estabilización en pipeline_metrics."
        )
