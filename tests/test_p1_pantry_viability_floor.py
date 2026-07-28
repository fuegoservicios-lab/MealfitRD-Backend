"""[P1-PANTRY-VIABILITY-FLOOR · 2026-07-28] Nevera de 10 ítems en modo estricto = insatisfacible.

## El caso vivo (madrugada del 28-07, cazado por el loop de logs)

    01:54  PANTRY GUARD RECHAZO | unauthorized=31   (intento 1)
    01:56  PANTRY GUARD RECHAZO | unauthorized=33   (intento 2 — EMPEORA)
    01:59  RECHAZADO (intento 3)

Nevera del usuario: **10 ítems**. Componer 4 comidas/día de una semana con 10 ingredientes no es
desobediencia del modelo — es insatisfacible (la reflexión SÍ le pasaba la lista de faltantes y
aun así empeoró). Tres generaciones quemadas en 5 minutos, camino directo a chunk degradado. La
clase de P1-FRUIT-SEEDER-GATE-CONTRACT: la restricción imposible no produce mejores planes,
produce reintentos.

## El hueco

Las válvulas existentes de flexibilización miran FRESCURA (snapshot stale, live degradado), no
TAMAÑO; y la nevera VACÍA tiene carril propio (P2-CHUNK-AUTONOMY: vacía no es error). El hueco
era exactamente `0 < items < piso`.

## El arreglo

Wrapper sobre `_refresh_chunk_pantry` (cuerpo original intacto en `_inner`): cubre los 3
callsites y todos los returns internos con UNA pieza. Con `0 < len(items) <
CHUNK_PANTRY_STRICT_MIN_ITEMS` (knob, default 12) → `flex + advisory_only` con reason
`pantry_below_viability_floor`. Rollback: knob = 0.

tooltip-anchor: P1-PANTRY-VIABILITY-FLOOR
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import cron_tasks as ct


def _stub(user_id, form_data, snapshot_form_data=None, task_id=None, week_number=None):
    return form_data


def _run(items, **extra):
    fd = {"current_pantry_ingredients": [{"name": f"x{i}"} for i in range(items)]}
    fd.update(extra)
    with patch.object(ct, "_refresh_chunk_pantry_inner", _stub):
        return ct._refresh_chunk_pantry("u-test", fd, None)


# ───────────── 1. el caso vivo ─────────────

def test_diez_items_activan_flex_advisory():
    out = _run(10)
    assert out.get("_pantry_flexible_mode") is True
    assert out.get("_pantry_advisory_only") is True
    assert out.get("_pantry_degraded_reason") == "pantry_below_viability_floor"


@pytest.mark.parametrize("n", [1, 5, 11])
def test_bajo_el_piso_flexibiliza(n):
    assert _run(n).get("_pantry_flexible_mode") is True


# ───────────── 2. los carriles que NO debe pisar ─────────────

def test_nevera_suficiente_sigue_estricta():
    out = _run(20)
    assert not out.get("_pantry_flexible_mode"), (
        "con nevera suficiente el modo estricto es correcto y valioso — no flexibilizar de más"
    )


def test_nevera_vacia_conserva_su_carril():
    """P2-CHUNK-AUTONOMY: vacía no es error y tiene su propio manejo — el piso NO la toca."""
    assert not _run(0).get("_pantry_flexible_mode")


def test_no_pisa_el_pause():
    out = _run(3, _pantry_paused=True)
    assert not out.get("_pantry_flexible_mode"), (
        "un chunk pausado espera al usuario; flexibilizarlo saltaría la pausa"
    )


def test_no_duplica_si_ya_venia_flexible():
    out = _run(3, _pantry_flexible_mode=True, _pantry_degraded_reason="live_fetch_degraded")
    assert out.get("_pantry_degraded_reason") == "live_fetch_degraded", (
        "si otra válvula ya flexibilizó, su reason se conserva (no sobrescribir el diagnóstico)"
    )


# ───────────── 3. knob y estructura ─────────────

def test_knob_cero_desactiva():
    with patch.object(ct, "CHUNK_PANTRY_STRICT_MIN_ITEMS", 0):
        assert not _run(3).get("_pantry_flexible_mode")


def test_wrapper_cubre_todos_los_callsites():
    """El floor vive en el WRAPPER: los 3 callsites llaman `_refresh_chunk_pantry` y el cuerpo
    original quedó en `_inner`. Si alguien llama al inner directo, esquiva el piso."""
    import pathlib
    src = pathlib.Path(ct.__file__).with_suffix(".py").read_text(encoding="utf-8")
    directos = src.count("_refresh_chunk_pantry_inner(")
    # 1 def + 1 llamada del wrapper — cualquier extra es un callsite esquivando el piso
    assert directos == 2, (
        f"{directos - 2} callsite(s) llaman a _refresh_chunk_pantry_inner directo y esquivan "
        f"el piso de viabilidad"
    )
