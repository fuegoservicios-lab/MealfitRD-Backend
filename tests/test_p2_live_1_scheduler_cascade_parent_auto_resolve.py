"""[P2-LIVE-1 . 2026-05-11] Auto-cierre del parent
`scheduler_cascade_missed` cuando la cascada se estabiliza.

Bug observado en produccion (mpoodlmnzaeuuazsazbj, 2026-05-11 03:08 UTC):
    El cron `_alert_scheduler_cascade_missed` upsertea un unico
    `scheduler_cascade_missed` (severity critical) cuando detecta >=N
    jobs distintos MISSED en la ventana. Las alerts hijas
    `scheduler_missed_<job_id>` se auto-resuelven via:
      - listener `EVENT_JOB_EXECUTED` (P1-NEW-2, app.py)
      - sweep `_resolve_stale_scheduler_alerts` (P0-AUDIT-1) con TTL=24h
      - sweep one-off P3-CLEANUP con TTL=12h

    Pero el padre `scheduler_cascade_missed` quedaba abierto
    indefinidamente hasta cleanup manual SRE. En prod 2026-05-11:
      - 03:08 UTC: cron detector dispara el padre.
      - 05:46 UTC: las 25 alerts hijas se resuelven (listener + sweep).
      - 08:08 UTC: padre sigue abierto (3h+ post-estabilizacion).

    Una alerta `critical` visible en el dashboard horas/dias tras
    estabilizacion efectiva entrena al operador a ignorar criticals.
    CLAUDE.md tabla "Politica system_alerts" documentaba el modelo
    como "Auto (implicit) + Manual cleanup" — esta version
    convierte la parte manual en automatica con guardrails.

Fix:
    Anadir un tercer sweep dentro de `_resolve_stale_scheduler_alerts`
    que cierra el padre `scheduler_cascade_missed` SOLO si:
      (1) Hay parent con `resolved_at IS NULL`.
      (2) NO hay ninguna `scheduler_missed_%` ni `scheduler_error_%`
          con `resolved_at IS NULL` (cascada estabilizada).
      (3) NO hay ninguna `scheduler_missed_%` ni `scheduler_error_%`
          con `triggered_at > NOW() - MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN`
          (ningun MISSED reciente — no en flanco descendente).

    Ventana de estabilizacion default 60min, alineada con el lookback
    del detector (1h): si en una hora no re-dispara, la cascada esta
    cerrada de hecho.

Estrategia del test (parser estatico sobre cron_tasks.py):
    1. La funcion `_resolve_stale_scheduler_alerts` existe.
    2. Contiene un UPDATE que filtra
       `alert_key = 'scheduler_cascade_missed'`.
    3. Ese UPDATE filtra `resolved_at IS NULL` (preserva manuales).
    4. Contiene dos `NOT EXISTS` (children abiertos + nuevos missed
       en ventana).
    5. Lee el knob `MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN`
       via _env_int (con default 60).
    6. El sweep esta dentro de try/except (best-effort).
    7. El tick (`pipeline_metrics`) incluye `swept_cascade_parent`.
    8. Docstring de la funcion contiene marker `P2-LIVE-1`.
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
    """Sanity: la funcion sweep sigue existiendo."""
    assert re.search(
        r"^def\s+_resolve_stale_scheduler_alerts\s*\(",
        cron_src,
        re.MULTILINE,
    ), "P2-LIVE-1: funcion sweep desaparecio."


def test_cascade_parent_update_targets_correct_alert_key(cron_src: str):
    """El UPDATE del parent debe targetear `alert_key = 'scheduler_cascade_missed'`
    especificamente — wildcard seria peligroso (mataria otros critical alerts).
    """
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    pattern = re.compile(
        r"alert_key\s*=\s*'scheduler_cascade_missed'",
        re.IGNORECASE,
    )
    assert pattern.search(body), (
        "P2-LIVE-1 regresion: no se encontro UPDATE/WHERE con "
        "`alert_key = 'scheduler_cascade_missed'`. Sin este predicate "
        "el sweep no cierra el parent del cascade."
    )


def test_cascade_parent_update_preserves_manual_resolutions(cron_src: str):
    """El UPDATE debe filtrar `resolved_at IS NULL` para no pisar
    resoluciones manuales."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # Localizar el bloque que contiene `alert_key = 'scheduler_cascade_missed'`
    # — debe estar acompañado de `resolved_at IS NULL` cercano.
    cascade_block_match = re.search(
        r"alert_key\s*=\s*'scheduler_cascade_missed'[\s\S]{0,800}",
        body,
        re.IGNORECASE,
    )
    assert cascade_block_match, (
        "P2-LIVE-1 setup: bloque del parent sweep no encontrado."
    )
    block = cascade_block_match.group(0)
    assert re.search(r"resolved_at\s+IS\s+NULL", block, re.IGNORECASE), (
        "P2-LIVE-1 regresion: UPDATE no filtra `resolved_at IS NULL`. "
        "Cada run del sweep pisaria timestamps de resoluciones manuales."
    )


def test_cascade_parent_update_has_two_not_exists_guards(cron_src: str):
    """El UPDATE debe tener DOS `NOT EXISTS`: uno para children abiertos,
    otro para nuevos MISSED en ventana de estabilizacion. Sin ambos, el
    sweep cerraria el parent antes de tiempo (durante cascada activa)."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    cascade_block_match = re.search(
        r"alert_key\s*=\s*'scheduler_cascade_missed'[\s\S]{0,2000}?RETURNING",
        body,
        re.IGNORECASE,
    )
    assert cascade_block_match, (
        "P2-LIVE-1 setup: bloque del parent sweep no encontrado "
        "(falta sub-string `RETURNING` que delimita el final del UPDATE)."
    )
    block = cascade_block_match.group(0)
    not_exists_count = len(re.findall(r"NOT\s+EXISTS", block, re.IGNORECASE))
    assert not_exists_count >= 2, (
        f"P2-LIVE-1 regresion: encontre {not_exists_count} `NOT EXISTS`, "
        f"esperaba >=2 (children abiertos + nuevos missed en ventana). "
        f"Sin ambos, el sweep puede cerrar el parent durante cascada activa."
    )


def test_cascade_parent_update_filters_children_namespaces(cron_src: str):
    """Los `NOT EXISTS` deben filtrar children por los namespaces
    `scheduler_missed_%` y `scheduler_error_%` (ambos). Sin filtrar
    `scheduler_error_*` el sweep cerraria el parent aunque sigan
    apareciendo errores activos."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    cascade_block_match = re.search(
        r"alert_key\s*=\s*'scheduler_cascade_missed'[\s\S]{0,2000}?RETURNING",
        body,
        re.IGNORECASE,
    )
    assert cascade_block_match
    block = cascade_block_match.group(0)
    assert "scheduler_missed_%" in block, (
        "P2-LIVE-1 regresion: los NOT EXISTS no incluyen "
        "`scheduler_missed_%`. El parent se cerraria pese a children "
        "abiertos."
    )
    assert "scheduler_error_%" in block, (
        "P2-LIVE-1 regresion: los NOT EXISTS no incluyen "
        "`scheduler_error_%`. Errores activos NO cuentan como cascada "
        "no-estabilizada — riesgo de cerrar parent prematuramente."
    )


def test_cascade_stabilization_knob_read(cron_src: str):
    """El knob `MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN` debe leerse
    en la funcion (default 60). Asegura que el operador puede ajustar
    la ventana sin redeploy."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert re.search(
        r'_env_int\(\s*[\'"]MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN[\'"]\s*,\s*60\s*\)',
        body,
    ), (
        "P2-LIVE-1 regresion: knob "
        "`MEALFIT_SCHEDULER_CASCADE_STABILIZATION_MIN` no se lee via "
        "`_env_int(..., 60)`. El operador no podria ajustar la "
        "ventana sin redeploy."
    )


def test_cascade_parent_sweep_is_best_effort(cron_src: str):
    """El parent sweep debe estar dentro de try/except — un fallo de DB
    NO debe propagar al scheduler ni pausar el resto del sweep."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # Localizar la region que contiene el UPDATE del parent + verificar
    # que esta envuelta en try/except.
    parent_region = re.search(
        r"P2-LIVE-1[\s\S]*?except\s+Exception\s+as\s+\w+\s*:",
        body,
    )
    assert parent_region, (
        "P2-LIVE-1 regresion: el bloque del parent sweep no esta "
        "envuelto en try/except. Un blip de Supabase haria que el "
        "cron entero falle."
    )


def test_tick_metric_includes_cascade_parent_field(cron_src: str):
    """El `pipeline_metrics` tick emitido al final del sweep debe
    incluir `swept_cascade_parent` (counter del nuevo sweep). Sin esto,
    post-mortems no pueden correlacionar 'cron corrio sin barrer
    cascade' vs 'cron barrio cascade'."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "swept_cascade_parent" in body, (
        "P2-LIVE-1 regresion: tick metric no incluye "
        "`swept_cascade_parent` en metadata. Post-mortem queda ciego "
        "sobre cuantas veces el sweep cerro parents."
    )


def test_docstring_documents_p2_live_1(cron_src: str):
    """Docstring contiene marker `P2-LIVE-1` para trazabilidad."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "P2-LIVE-1" in body, (
        "P2-LIVE-1 regresion: marker ausente del docstring. "
        "Un auditor no puede confirmar que el tercer sweep "
        "corresponde a este P-fix."
    )
