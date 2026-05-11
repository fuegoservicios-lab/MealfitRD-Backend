"""[P3-LIVE-1 . 2026-05-11] Baseline tick observable para
`_alert_chunk_pantry_snapshots_stale`.

Bug observado en audit live (mpoodlmnzaeuuazsazbj, 2026-05-11):
    El cron `_alert_chunk_pantry_snapshots_stale` no emitia nada a
    `pipeline_metrics` cuando corria — solo `logger.warning` cuando
    detectaba >=min_count chunks stale. Resultado: un audit live no
    podia responder "el cron estuvo corriendo durante la ventana
    del incidente?" porque la ausencia de filas en pipeline_metrics
    no distinguia "cron caido" de "cron corriendo sin alerts".

    Patron consistente con P2-B-OBS:
      - `_alert_scheduler_cascade_missed` emite
        `_scheduler_cascade_check_tick` siempre.
      - `_resolve_stale_scheduler_alerts` emite
        `_scheduler_alerts_sweep_tick` siempre.
    Falta cubrir `_alert_chunk_pantry_snapshots_stale` con el mismo
    pattern.

Fix:
    Refactor de la funcion para acumular flags de outcome
    (stale_count, select_failed, auto_resolved_attempted,
    cooldown_skipped, alert_emitted) y emitir `pipeline_metrics`
    en `finally`, capturando el outcome de cualquier path tomado.
    Knob kill-switch `MEALFIT_PANTRY_STALE_BASELINE_EMIT` (default
    True) por si el volumen de ticks resulta problematico.

Estrategia del test (parser estatico sobre cron_tasks.py):
    1. Funcion `_alert_chunk_pantry_snapshots_stale` existe.
    2. Lee knob `MEALFIT_PANTRY_STALE_BASELINE_EMIT` via _env_bool
       con default True.
    3. Contiene un INSERT a `pipeline_metrics` con node
       `_alert_chunk_pantry_snapshots_stale_tick`.
    4. El INSERT esta dentro de un bloque `finally:` (corre en
       todos los paths del cron, incluyendo early-return).
    5. El INSERT esta envuelto en try/except (best-effort — un
       fallo del tick NO debe propagar al scheduler).
    6. La metadata del tick incluye los 7 flags clave:
       stale_count, threshold_hours, min_count, select_failed,
       auto_resolved_attempted, cooldown_skipped, alert_emitted.
    7. Docstring contiene marker `P3-LIVE-1` para trazabilidad.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    """Cuerpo de una funcion top-level hasta el siguiente `def`."""
    m = re.search(
        rf"^def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"No se encontro funcion top-level `{name}`."
    return m.group(1)


def test_function_exists(cron_src: str):
    """Sanity."""
    assert re.search(
        r"^def\s+_alert_chunk_pantry_snapshots_stale\s*\(",
        cron_src,
        re.MULTILINE,
    ), "P3-LIVE-1: funcion productora desaparecio."


def test_baseline_emit_knob_read(cron_src: str):
    """Knob `MEALFIT_PANTRY_STALE_BASELINE_EMIT` debe leerse via
    `_env_bool` con default True (kill-switch sin redeploy)."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    assert re.search(
        r"_env_bool\(\s*[\"']MEALFIT_PANTRY_STALE_BASELINE_EMIT[\"']\s*,\s*True\s*\)",
        body,
    ), (
        "P3-LIVE-1 regresion: knob "
        "`MEALFIT_PANTRY_STALE_BASELINE_EMIT` no se lee via "
        "`_env_bool(..., True)`. Sin ese knob el operador no "
        "puede silenciar el tick sin redeploy."
    )


def test_tick_node_name_present(cron_src: str):
    """El tick debe usar node `_alert_chunk_pantry_snapshots_stale_tick`
    (mismo namespace que la funcion productora)."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    assert '"_alert_chunk_pantry_snapshots_stale_tick"' in body, (
        "P3-LIVE-1 regresion: el tick no usa el node "
        "`_alert_chunk_pantry_snapshots_stale_tick` esperado. "
        "Sin ese nombre, queries de observability post-incidente "
        "(WHERE node = ...) fallan silenciosamente."
    )


def test_tick_emitted_in_finally_block(cron_src: str):
    """El tick debe estar dentro de `finally:` (no en una rama
    regular). Solo asi corre en todos los paths: select_failed,
    auto-resolve, cooldown, emit."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    # Buscamos la region desde `finally:` hasta el siguiente bloque
    # top-level (cierre de la funcion).
    finally_match = re.search(
        r"^\s{4}finally\s*:\s*\n(.*?)(?=^def\s|\Z)",
        body,
        re.DOTALL | re.MULTILINE,
    )
    assert finally_match, (
        "P3-LIVE-1 regresion: no se encontro bloque `finally:` "
        "a 4-espacios de indent en la funcion. El tick puede "
        "estar fuera del try/finally — paths con `return` no lo "
        "emitirian."
    )
    finally_body = finally_match.group(1)
    assert "_alert_chunk_pantry_snapshots_stale_tick" in finally_body, (
        "P3-LIVE-1 regresion: el bloque `finally:` no contiene "
        "el INSERT del tick. El tick esta fuera del finally y "
        "no se emite en todos los paths."
    )


def test_tick_is_best_effort(cron_src: str):
    """El INSERT del tick debe estar dentro de try/except — un
    fallo de DB NO debe propagar al scheduler."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    finally_match = re.search(
        r"^\s{4}finally\s*:\s*\n(.*?)(?=^def\s|\Z)",
        body,
        re.DOTALL | re.MULTILINE,
    )
    assert finally_match
    finally_body = finally_match.group(1)
    assert re.search(r"try\s*:", finally_body) and re.search(
        r"except\s+Exception", finally_body
    ), (
        "P3-LIVE-1 regresion: el INSERT del tick en `finally:` "
        "no esta envuelto en try/except. Un blip de Supabase "
        "haria que el cron entero falle."
    )


@pytest.mark.parametrize("flag", [
    "stale_count",
    "threshold_hours",
    "min_count",
    "select_failed",
    "auto_resolved_attempted",
    "cooldown_skipped",
    "alert_emitted",
])
def test_tick_metadata_includes_outcome_flags(cron_src: str, flag: str):
    """La metadata JSON del tick debe incluir los 7 flags de outcome.
    Sin estos, queries de observability no pueden distinguir cron sano
    sin alerts (`stale_count=0, alert_emitted=False`) de cron sano
    con alert (`stale_count=N, alert_emitted=True`) de cron con SELECT
    caido (`select_failed=True`)."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    finally_match = re.search(
        r"^\s{4}finally\s*:\s*\n(.*?)(?=^def\s|\Z)",
        body,
        re.DOTALL | re.MULTILINE,
    )
    assert finally_match
    finally_body = finally_match.group(1)
    # El flag debe aparecer como key del json.dumps en la metadata.
    pattern = re.compile(rf'["\']?{re.escape(flag)}["\']?\s*:')
    assert pattern.search(finally_body), (
        f"P3-LIVE-1 regresion: metadata del tick NO incluye `{flag}`. "
        f"Sin este field, queries de observability post-incidente "
        f"quedan ciegas sobre ese aspecto del outcome."
    )


def test_docstring_documents_p3_live_1(cron_src: str):
    """Docstring contiene marker `P3-LIVE-1` para trazabilidad."""
    body = _extract_function_body(cron_src, "_alert_chunk_pantry_snapshots_stale")
    assert "P3-LIVE-1" in body, (
        "P3-LIVE-1 regresion: marker ausente del docstring. Un "
        "auditor no puede confirmar que el tick corresponde a "
        "este P-fix."
    )
