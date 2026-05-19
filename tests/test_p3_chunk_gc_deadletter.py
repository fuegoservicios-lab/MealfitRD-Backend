r"""[P3-CHUNK-GC-DEADLETTER · 2026-05-18] GC de chunks dead-lettered + chunk_kind en telemetría.

Bundle de 2 fixes derivados del audit del sistema de chunks 2026-05-18:

**Fix #1: GC de chunks dead-lettered**

Pre-fix, `_alert_new_dead_lettered_chunks` (cron_tasks.py:16096) ALERTA sobre
filas `dead_lettered_at IS NOT NULL` pero NO purga las viejas. Sin GC, la
tabla `plan_chunk_queue` acumulaba filas terminales indefinidamente:
- Index `plan_chunk_queue_status_idx` y `plan_chunk_queue_user_id_idx` crecían.
- Pickup `SELECT … FOR UPDATE SKIP LOCKED LIMIT 1` degradaba O(N).
- Queries forensics (admin endpoint `/admin/chunks/dead-lettered`) se
  ralentizaban.

Fix: cron `_gc_dead_lettered_chunks` (cron_tasks.py:16253) corre cada
`CHUNK_GC_DEAD_LETTER_INTERVAL_HOURS` (default 24h), purga rows con
`dead_lettered_at < NOW() - INTERVAL '<TTL> days'` (TTL default 30d,
knob `CHUNK_GC_DEAD_LETTER_TTL_DAYS`), en batches de
`CHUNK_GC_DEAD_LETTER_BATCH` (default 1000) para evitar lock contention
con worker pickups.

Patrón espejo de la purga existente "cancelled >48h" (cron_tasks.py:20039)
pero con TTL mayor (forensic interest). Emite tick observable
`_gc_dead_lettered_chunks_tick` SIEMPRE (patrón P2-LIVE-9).

**Fix #3: chunk_kind en telemetría**

Pre-fix, el campo `chunk_kind` de `plan_chunk_queue` (initial_plan vs
rolling_refill vs catchup) se persistía pero NO se propagaba a:
1. `_alert_new_dead_lettered_chunks` metadata — solo agrupaba por
   `dead_letter_reason`. SRE no podía distinguir "planes nuevos fallan
   pero rolling están OK" o viceversa.
2. `_chunk_heartbeat_baseline` / `_chunk_heartbeat_lag` metadata — solo
   tenía `chunk_id`. Dashboards p50/p95 de heartbeat lag no podían filtrar
   por chunk_kind.

Fix: añadir `by_chunk_kind` (alert) + `chunk_kind` (heartbeat metrics)
al metadata. Cero cost (chunk_kind ya está en el RETURNING del pickup
y en task dict).

**Drift detection:**
- Si alguien borra el cron `gc_dead_lettered_chunks` del registro → falla
  `test_gc_cron_registered`.
- Si alguien remueve los knobs → falla `test_knobs_defined`.
- Si alguien borra el `by_chunk_kind` del alert metadata → falla
  `test_alert_includes_by_chunk_kind`.
- Si alguien remueve `chunk_kind` del state del heartbeat thread → falla
  `test_heartbeat_state_includes_chunk_kind`.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
_CONSTANTS = (_BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Knobs y cron registration
# ─────────────────────────────────────────────────────────────────────────────


def test_knobs_defined():
    """Los 3 knobs del GC deben estar definidos en constants.py con defaults y clamps."""
    assert "CHUNK_GC_DEAD_LETTER_TTL_DAYS" in _CONSTANTS, (
        "Knob CHUNK_GC_DEAD_LETTER_TTL_DAYS ausente — sin él, el cron no tiene "
        "TTL configurable y se hardcodea, violando convención del repo."
    )
    assert "CHUNK_GC_DEAD_LETTER_INTERVAL_HOURS" in _CONSTANTS
    assert "CHUNK_GC_DEAD_LETTER_BATCH" in _CONSTANTS

    # Verificar clamps razonables.
    m = re.search(
        r"CHUNK_GC_DEAD_LETTER_TTL_DAYS\s*=\s*max\((\d+),\s*min\((\d+),",
        _CONSTANTS,
    )
    assert m, "TTL knob no usa max+min clamp — valores patológicos no se protegen."
    lower, upper = int(m.group(1)), int(m.group(2))
    assert lower >= 7, f"TTL floor demasiado bajo ({lower}d), perdería forensic context"
    assert upper <= 365, f"TTL cap demasiado alto ({upper}d), eso no es GC sino archivado"


def test_gc_function_defined():
    """La función `_gc_dead_lettered_chunks` debe estar definida en cron_tasks.py."""
    assert "def _gc_dead_lettered_chunks(" in _CRON_TASKS, (
        "Función _gc_dead_lettered_chunks no definida en cron_tasks.py. "
        "Sin ella, el knob existe pero no hace nada."
    )


def test_gc_cron_registered():
    """El cron debe estar registrado en `register_plan_chunk_scheduler`."""
    # Anchor: el block dentro de register_plan_chunk_scheduler.
    register_block_idx = _CRON_TASKS.find("def register_plan_chunk_scheduler")
    assert register_block_idx > 0, "register_plan_chunk_scheduler no encontrado"
    register_block = _CRON_TASKS[register_block_idx:register_block_idx + 30000]

    assert 'id="gc_dead_lettered_chunks"' in register_block, (
        "Cron 'gc_dead_lettered_chunks' no registrado en register_plan_chunk_scheduler. "
        "Sin registro, el handler existe pero APScheduler nunca lo invoca."
    )
    assert "_gc_dead_lettered_chunks," in register_block, (
        "Handler _gc_dead_lettered_chunks no se pasa a _add_job_jittered."
    )


def test_gc_uses_status_failed_and_dead_lettered_at():
    """Filtro: `status='failed' AND dead_lettered_at IS NOT NULL` (no `cancelled`
    ni `completed` por error)."""
    body_idx = _CRON_TASKS.find("def _gc_dead_lettered_chunks(")
    body = _CRON_TASKS[body_idx:body_idx + 6000]
    assert "status = 'failed'" in body, (
        "GC no filtra por status='failed'. Sin este filtro, podría borrar chunks "
        "transient (en retry) o non-terminal."
    )
    assert "dead_lettered_at IS NOT NULL" in body, (
        "GC no filtra por `dead_lettered_at IS NOT NULL`. Sin este filtro, podría "
        "borrar failures transient en backoff que aún se pueden recuperar."
    )
    assert "make_interval(days =>" in body, (
        "GC no aplica el TTL via make_interval(days => ...). Si está hardcodeado o "
        "usa unidad distinta, el knob no funciona."
    )


def test_gc_uses_batch_limit():
    """Para evitar lock contention con worker pickup, el DELETE debe estar limitado."""
    body_idx = _CRON_TASKS.find("def _gc_dead_lettered_chunks(")
    body = _CRON_TASKS[body_idx:body_idx + 6000]
    # Debe usar CTE con LIMIT (psycopg no soporta LIMIT directo en DELETE).
    assert "LIMIT %s" in body or "LIMIT " in body, (
        "GC DELETE no tiene LIMIT — un dead-letter spike masivo podría hacer un "
        "DELETE de 100k+ filas, contendiendo con worker pickup por minutos."
    )
    assert "WITH victims AS" in body or "WITH " in body, (
        "GC no usa CTE para limitar el DELETE. psycopg/postgres no soportan "
        "LIMIT en DELETE directo."
    )


def test_gc_emits_tick_metric():
    """Tick observable SIEMPRE — patrón P2-LIVE-9. Sin esto, SRE no puede confirmar
    que el cron corrió cuando purged_count=0."""
    body_idx = _CRON_TASKS.find("def _gc_dead_lettered_chunks(")
    body = _CRON_TASKS[body_idx:body_idx + 6000]
    assert "_gc_dead_lettered_chunks_tick" in body, (
        "GC no emite tick observable a pipeline_metrics. SRE no podrá confirmar "
        "que el cron corrió cuando no hay nada que purgar."
    )
    assert "INSERT INTO pipeline_metrics" in body


# ─────────────────────────────────────────────────────────────────────────────
# Fix #3a: by_chunk_kind en _alert_new_dead_lettered_chunks
# ─────────────────────────────────────────────────────────────────────────────


def test_alert_includes_by_chunk_kind():
    """El alert `dead_lettered_chunks_recent` debe incluir `by_chunk_kind` además
    de `by_reason` en su metadata. Sin esto, SRE no puede distinguir si los
    dead-letters son de planes nuevos vs rolling refills."""
    body_idx = _CRON_TASKS.find("def _alert_new_dead_lettered_chunks(")
    assert body_idx > 0
    body = _CRON_TASKS[body_idx:body_idx + 8000]

    assert "by_chunk_kind" in body, (
        "Alert metadata no incluye `by_chunk_kind`. Pre-fix, ambos chunk_kinds "
        "caían en el mismo bucket y la causa raíz quedaba invisible en dashboards."
    )
    # Verificar query GROUP BY chunk_kind.
    assert "GROUP BY COALESCE(chunk_kind" in body, (
        "Query de breakdown por chunk_kind ausente. El metadata key existe pero "
        "no se computa."
    )
    # Verificar marker.
    assert "P3-CHUNK-KIND-TELEMETRY" in body, (
        "Marker P3-CHUNK-KIND-TELEMETRY ausente en _alert_new_dead_lettered_chunks. "
        "Un revert silente reintroduciría la zona ciega."
    )


def test_alert_message_includes_chunk_kinds():
    """El message del alert debe mencionar chunk_kinds para que el operador lea
    la señal sin abrir el metadata."""
    body_idx = _CRON_TASKS.find("def _alert_new_dead_lettered_chunks(")
    body = _CRON_TASKS[body_idx:body_idx + 8000]
    assert "kinds_str" in body, (
        "El message del alert no incluye un kinds_str compuesto desde by_chunk_kind. "
        "El operador tendría que abrir el metadata para diagnosticar."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fix #3b: chunk_kind en heartbeat metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_heartbeat_state_includes_chunk_kind():
    """El state del thread daemon de heartbeat debe propagar chunk_kind para que
    el metadata final incluya la dimensión."""
    # Anchor del state dict.
    state_idx = _CRON_TASKS.find('"lock_chunk_id": task_id,')
    assert state_idx > 0
    block = _CRON_TASKS[state_idx:state_idx + 1500]
    assert '"chunk_kind"' in block, (
        "_heartbeat_state no propaga chunk_kind. Las metrics emitidas al cierre "
        "tendrán solo chunk_id, perdiendo la dimensión más útil para dashboards."
    )


def test_heartbeat_metrics_include_chunk_kind():
    """Tanto el metric anómalo (`_chunk_heartbeat_lag`) como el baseline
    (`_chunk_heartbeat_baseline`) deben incluir `chunk_kind` en su metadata."""
    # Anchor: bloque del finally del thread daemon donde emite stats.
    finally_idx = _CRON_TASKS.find("[P3-3 · 2026-05-08] Stats finales del thread")
    assert finally_idx > 0, "Bloque de stats del heartbeat no encontrado"
    # Ventana grande para cubrir ambos metrics (lag + baseline emit, ~3500 chars
    # entre ellos).
    block = _CRON_TASKS[finally_idx:finally_idx + 8000]

    # Lag metric
    lag_idx = block.find('"_chunk_heartbeat_lag"')
    assert lag_idx > 0
    lag_block = block[lag_idx:lag_idx + 1500]
    assert '"chunk_kind"' in lag_block, (
        "_chunk_heartbeat_lag metadata NO incluye chunk_kind. Dashboards de lag "
        "anómalo no pueden filtrar por initial_plan vs rolling_refill."
    )

    # Baseline metric
    baseline_idx = block.find('"_chunk_heartbeat_baseline"')
    assert baseline_idx > 0, (
        "Block window no llega al baseline metric. Ampliar la ventana o anclar "
        "más arriba (el orden es lag emit primero, baseline después)."
    )
    baseline_block = block[baseline_idx:baseline_idx + 1500]
    assert '"chunk_kind"' in baseline_block, (
        "_chunk_heartbeat_baseline metadata NO incluye chunk_kind. Sin esto, "
        "queries p50/p95 de heartbeat lag por chunk_kind no funcionan."
    )


def test_chunk_kind_metric_var_uses_unknown_fallback():
    """Defensa: si el state legacy no tiene chunk_kind, fallback a 'unknown' (no None
    ni KeyError). Evita que un upgrade in-place con threads viejos rompa el emit."""
    finally_idx = _CRON_TASKS.find("[P3-3 · 2026-05-08] Stats finales del thread")
    assert finally_idx > 0
    block = _CRON_TASKS[finally_idx:finally_idx + 5000]
    assert "_chunk_kind_metric" in block
    # Debe usar `or "unknown"` fallback.
    assert re.search(r'state\.get\("chunk_kind"\)\s*or\s*"unknown"', block), (
        "Fallback a 'unknown' ausente — un state legacy sin chunk_kind rompería "
        "el emit de metrics con None en JSON o KeyError."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bump marker
# ─────────────────────────────────────────────────────────────────────────────


def test_marker_bumped_at_least_to_this_fix_date():
    """El marker P-fix global debe tener fecha >= 2026-05-18 (este fix).
    Validamos fecha y no literal para que P-fixes posteriores en el mismo día
    bumpeen al suyo sin romper este test."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"[^"]+·\s*(\d{4}-\d{2}-\d{2})"',
        _APP_PY,
    )
    assert m, "_LAST_KNOWN_PFIX no encontrado o sin fecha ISO"
    from datetime import date
    assert date.fromisoformat(m.group(1)) >= date(2026, 5, 18), (
        f"_LAST_KNOWN_PFIX stale ({m.group(1)}); este P-fix es 2026-05-18."
    )
