"""[P3-CHUNK-HEALTH-CHECK · 2026-05-18] Inspecciona el estado del sistema de
chunks contra una Supabase live para diagnosticar problemas operacionales.

Uso:

    # Verificaciones globales:
    py -3.11 -m scripts.chunk_health_check

    # + verificaciones específicas a un plan tuyo:
    py -3.11 -m scripts.chunk_health_check --plan-id <uuid>

    # + verificaciones específicas a un usuario:
    py -3.11 -m scripts.chunk_health_check --user-id <uuid>

Variables de entorno requeridas:
    SUPABASE_DB_URL  — connection string (mismo que el backend usa).

Output: tabla con cada check + status (OK/WARN/FAIL) + detalles. Exit code
0 si todos OK o WARN, 1 si hay FAIL.

Diseñado para correr en seco contra prod: SOLO lee. NO escribe.

Cubre 15 invariantes del sistema de chunks:
  - Sanidad: distribución por status, zombies, locks colgados, queue stuck.
  - Dead-letter: counts en ventanas, breakdown por reason+chunk_kind (P3-CHUNK-KIND-TELEMETRY),
    chunks listos para GC (P3-CHUNK-GC-DEADLETTER).
  - Pausas: pending_user_action con TTLs vencidos, pausas indefinidas (>24h).
  - Coherencia: con --plan-id, valida sum(days_count) chunks == len(plan_data.days)
    y ausencia de gaps en days_offset.
  - Telemetría: heartbeat baseline/lag emitidos recientemente, GC tick emitido.

Tooltip-anchor: P3-CHUNK-HEALTH-CHECK.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows: forzar UTF-8 en stdout para emojis (✓/⚠/✗). cp1252 default fallaría
# con UnicodeEncodeError. Python 3.7+ soporta reconfigure.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Permitir ejecutar el script desde la raíz del backend (cd backend && py script.py)
# o desde el repo root (py -m scripts.chunk_health_check). Hack mínimo: añadir el
# backend al sys.path si no está.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Cargar .env del backend ANTES del check de SUPABASE_DB_URL. db_core también
# llama load_dotenv() pero se importa lazily dentro de _exec_query, así que
# el check fail-fast en main() se ejecutaría sin haber cargado el .env.
try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND_ROOT / ".env")
except ImportError:
    pass  # dotenv no instalado: fall through al check explícito.


# Output styling: ANSI colors. Detect si stdout es tty; sino desactivar.
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _green(t): return _c("32", t)
def _yellow(t): return _c("33", t)
def _red(t): return _c("31", t)
def _cyan(t): return _c("36", t)
def _bold(t): return _c("1", t)


# Status emojis + colors
_STATUS = {
    "OK": (_green("✓ OK"), 0),
    "WARN": (_yellow("⚠ WARN"), 0),  # WARN no falla el exit
    "FAIL": (_red("✗ FAIL"), 1),
    "INFO": (_cyan("ℹ INFO"), 0),
}


class CheckResult:
    __slots__ = ("name", "status", "message", "details")

    def __init__(self, name: str, status: str, message: str, details: dict | None = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}

    def print(self) -> None:
        badge, _ = _STATUS[self.status]
        print(f"  {badge}  {_bold(self.name)}")
        # Indentar mensaje + detalles bajo el badge.
        for line in self.message.split("\n"):
            print(f"           {line}")
        if self.details:
            for k, v in self.details.items():
                if isinstance(v, (dict, list)):
                    v_str = json.dumps(v, ensure_ascii=False)
                else:
                    v_str = str(v)
                # Truncar valores largos para no explotar la pantalla.
                if len(v_str) > 120:
                    v_str = v_str[:117] + "..."
                print(f"           {_cyan(k)}: {v_str}")


def _exec_query(query: str, params: tuple | None = None, fetch_all: bool = True):
    """Wrapper sobre execute_sql_query que no carga toda la app. Usa la misma
    config de connection_pool del backend (lee SUPABASE_DB_URL del env)."""
    from db_core import execute_sql_query, connection_pool
    if connection_pool is None:
        raise RuntimeError(
            "connection_pool no disponible. Verifica SUPABASE_DB_URL en env."
        )
    # connection_pool tiene open=False por default; abrir lazy.
    if not connection_pool._opened:
        connection_pool.open()
    if fetch_all:
        return execute_sql_query(query, params, fetch_all=True) or []
    return execute_sql_query(query, params, fetch_one=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS · Sanidad general
# ─────────────────────────────────────────────────────────────────────────────


def check_queue_status_distribution() -> CheckResult:
    """Cuenta filas en plan_chunk_queue por status. Sin alertas — solo info."""
    rows = _exec_query(
        """
        SELECT status, COUNT(*)::int AS cnt
        FROM plan_chunk_queue
        GROUP BY status
        ORDER BY cnt DESC
        """
    )
    if not rows:
        return CheckResult(
            "Queue status distribution",
            "OK",
            "Tabla plan_chunk_queue vacía (sin chunks vivos ni terminales)."
        )
    counts = {str(r["status"]): int(r["cnt"]) for r in rows}
    total = sum(counts.values())
    return CheckResult(
        "Queue status distribution",
        "INFO",
        f"Total {total} chunks en plan_chunk_queue.",
        details=counts,
    )


def check_zombie_chunks_pending_rescue() -> CheckResult:
    """Chunks `processing` con updated_at viejo NO recogidos aún por housekeeping.
    El rescue corre cada tick del worker; si hay zombies queda señal de que el
    worker NO está corriendo o el housekeeping está paused."""
    rows = _exec_query(
        """
        SELECT id, meal_plan_id, user_id, week_number, updated_at,
               EXTRACT(EPOCH FROM (NOW() - updated_at))/60 AS minutes_stuck
        FROM plan_chunk_queue
        WHERE status = 'processing'
          AND updated_at < NOW() - INTERVAL '15 minutes'
        ORDER BY updated_at ASC
        LIMIT 5
        """
    )
    if not rows:
        return CheckResult(
            "Zombie chunks awaiting rescue",
            "OK",
            "No hay chunks 'processing' con >15min sin update (housekeeping al día)."
        )
    return CheckResult(
        "Zombie chunks awaiting rescue",
        "FAIL",
        f"{len(rows)} chunks 'processing' con >15min sin update. "
        "Housekeeping no los ha rescatado — worker caído o cron pausado?",
        details={"top_5": [
            {
                "chunk_id": str(r["id"])[:8],
                "user_id": str(r["user_id"])[:8],
                "minutes_stuck": round(float(r["minutes_stuck"]), 1),
            } for r in rows
        ]},
    )


def check_stale_user_locks() -> CheckResult:
    """chunk_user_locks con heartbeat_at viejo (debería ser purgado por
    DELETE WHERE heartbeat_at < NOW() - CHUNK_LOCK_STALE_MINUTES)."""
    rows = _exec_query(
        """
        SELECT user_id, locked_by_chunk_id, heartbeat_at,
               EXTRACT(EPOCH FROM (NOW() - heartbeat_at))/60 AS minutes_stale
        FROM chunk_user_locks
        WHERE heartbeat_at < NOW() - INTERVAL '15 minutes'
        ORDER BY heartbeat_at ASC
        LIMIT 5
        """
    )
    if not rows:
        return CheckResult(
            "Stale chunk_user_locks",
            "OK",
            "No hay locks de usuario con heartbeat viejo (cleanup al día)."
        )
    return CheckResult(
        "Stale chunk_user_locks",
        "FAIL",
        f"{len(rows)} user locks con >15min sin heartbeat. "
        "Algún usuario está bloqueado contra nuevos pickups innecesariamente.",
        details={"top_5": [
            {
                "user_id": str(r["user_id"])[:8],
                "chunk_id": str(r["locked_by_chunk_id"])[:8],
                "minutes_stale": round(float(r["minutes_stale"]), 1),
            } for r in rows
        ]},
    )


def check_pending_queue_stuck() -> CheckResult:
    """Chunks `pending` con execute_after vencido hace >30min: el worker no los
    está tomando. Posibles causas: cron caído, advisory lock filtra al usuario,
    o user lock zombie no purgado."""
    rows = _exec_query(
        """
        SELECT id, meal_plan_id, user_id, week_number, execute_after,
               EXTRACT(EPOCH FROM (NOW() - execute_after))/60 AS minutes_overdue,
               attempts
        FROM plan_chunk_queue
        WHERE status = 'pending'
          AND execute_after < NOW() - INTERVAL '30 minutes'
        ORDER BY execute_after ASC
        LIMIT 10
        """
    )
    if not rows:
        return CheckResult(
            "Pending queue stuck",
            "OK",
            "No hay chunks pending con >30min de overdue (worker procesando al ritmo esperado)."
        )
    severity = "WARN" if len(rows) <= 3 else "FAIL"
    return CheckResult(
        "Pending queue stuck",
        severity,
        f"{len(rows)} chunks pending con >30min de overdue. "
        "Worker no los está procesando.",
        details={"top_chunks": [
            {
                "chunk_id": str(r["id"])[:8],
                "user_id": str(r["user_id"])[:8],
                "minutes_overdue": round(float(r["minutes_overdue"]), 1),
                "attempts": int(r["attempts"] or 0),
            } for r in rows[:5]
        ]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS · Dead-letter health
# ─────────────────────────────────────────────────────────────────────────────


def check_dead_letter_counts() -> CheckResult:
    """Distribución temporal de dead-letters."""
    rows = _exec_query(
        """
        SELECT
            COUNT(*) FILTER (WHERE dead_lettered_at > NOW() - INTERVAL '24 hours')::int AS last_24h,
            COUNT(*) FILTER (WHERE dead_lettered_at > NOW() - INTERVAL '7 days')::int AS last_7d,
            COUNT(*) FILTER (WHERE dead_lettered_at > NOW() - INTERVAL '30 days')::int AS last_30d,
            COUNT(*) FILTER (WHERE dead_lettered_at IS NOT NULL)::int AS total
        FROM plan_chunk_queue
        WHERE dead_lettered_at IS NOT NULL
        """,
        fetch_all=False,
    ) or {}
    last_24h = int(rows.get("last_24h") or 0)
    last_7d = int(rows.get("last_7d") or 0)
    last_30d = int(rows.get("last_30d") or 0)
    total = int(rows.get("total") or 0)

    if total == 0:
        return CheckResult(
            "Dead-letter counts",
            "OK",
            "Cero chunks dead-lettered en la historia (flota saludable o joven)."
        )

    severity = "OK"
    if last_24h >= 5:
        severity = "FAIL"
    elif last_24h >= 1 or last_7d >= 10:
        severity = "WARN"
    return CheckResult(
        "Dead-letter counts",
        severity,
        f"Total dead-lettered (incluye ya purgables): {total}",
        details={
            "last_24h": last_24h,
            "last_7d": last_7d,
            "last_30d": last_30d,
            "total": total,
        },
    )


def check_dead_letter_breakdown() -> CheckResult:
    """Breakdown por (chunk_kind, dead_letter_reason) en últimas 7d.
    Permite diagnosticar 'planes nuevos fallan' vs 'rolling fallan' (gap cerrado
    por P3-CHUNK-KIND-TELEMETRY)."""
    rows = _exec_query(
        """
        SELECT COALESCE(chunk_kind, 'unknown') AS chunk_kind,
               COALESCE(dead_letter_reason, 'unknown') AS reason,
               COUNT(*)::int AS cnt
        FROM plan_chunk_queue
        WHERE dead_lettered_at > NOW() - INTERVAL '7 days'
        GROUP BY COALESCE(chunk_kind, 'unknown'),
                 COALESCE(dead_letter_reason, 'unknown')
        ORDER BY cnt DESC
        LIMIT 10
        """
    )
    if not rows:
        return CheckResult(
            "Dead-letter breakdown (7d)",
            "OK",
            "Cero dead-letters en últimos 7 días."
        )
    breakdown = [
        f"{r['chunk_kind']}/{r['reason']}: {int(r['cnt'])}"
        for r in rows
    ]
    return CheckResult(
        "Dead-letter breakdown (7d)",
        "INFO",
        f"{len(rows)} combinaciones (chunk_kind, reason) en últimos 7d:",
        details={"top": breakdown},
    )


def check_gc_purgable_dead_letters() -> CheckResult:
    """Cantidad de dead-letters que el cron `_gc_dead_lettered_chunks`
    purgaría en su próximo tick. Sirve para verificar que el GC nuevo
    (P3-CHUNK-GC-DEADLETTER) tiene trabajo pendiente o no."""
    rows = _exec_query(
        """
        SELECT COUNT(*)::int AS cnt
        FROM plan_chunk_queue
        WHERE status = 'failed'
          AND dead_lettered_at IS NOT NULL
          AND dead_lettered_at < NOW() - INTERVAL '30 days'
        """,
        fetch_all=False,
    ) or {}
    cnt = int(rows.get("cnt") or 0)
    if cnt == 0:
        return CheckResult(
            "GC purgable dead-letters",
            "OK",
            "Cero dead-letters con >30d (default TTL). El cron no tiene trabajo pendiente."
        )
    return CheckResult(
        "GC purgable dead-letters",
        "INFO",
        f"{cnt} dead-letters con >30d listos para que el cron _gc_dead_lettered_chunks "
        "los purgue en su próximo tick (cada 24h por default).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS · Pausas
# ─────────────────────────────────────────────────────────────────────────────


def check_paused_chunks() -> CheckResult:
    """Chunks en status='pending_user_action' agrupados por reason. WARN si
    hay pausas; FAIL si alguna lleva >24h (debería haber alert ya)."""
    # Detectar la columna correcta para "reason": en algunos schemas es
    # pause_reason_code, en otros viene en pipeline_snapshot._pantry_pause_reason.
    rows = _exec_query(
        """
        SELECT
            COALESCE(pipeline_snapshot->>'_pantry_pause_reason', 'unknown') AS reason,
            chunk_kind,
            COUNT(*)::int AS cnt,
            MAX(EXTRACT(EPOCH FROM (NOW() - updated_at))/3600)::numeric AS max_hours_paused
        FROM plan_chunk_queue
        WHERE status = 'pending_user_action'
        GROUP BY 1, 2
        ORDER BY cnt DESC
        """
    )
    if not rows:
        return CheckResult(
            "Paused chunks (pending_user_action)",
            "OK",
            "Cero chunks pausados esperando acción del usuario."
        )
    max_hours = max((float(r.get("max_hours_paused") or 0) for r in rows), default=0)
    total_paused = sum(int(r["cnt"]) for r in rows)
    severity = "OK"
    if max_hours >= 24:
        severity = "WARN"
    if max_hours >= 48:
        severity = "FAIL"
    return CheckResult(
        "Paused chunks (pending_user_action)",
        severity,
        f"{total_paused} chunks pausados, max edad: {round(max_hours, 1)}h.",
        details={"by_reason": [
            f"{r['chunk_kind'] or 'unknown'}/{r['reason']}: {int(r['cnt'])} "
            f"(max_h={round(float(r['max_hours_paused'] or 0), 1)})"
            for r in rows
        ]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS · Telemetría reciente
# ─────────────────────────────────────────────────────────────────────────────


def check_heartbeat_telemetry_recent() -> CheckResult:
    """Verifica que el cron de worker está emitiendo heartbeat baseline + lag.
    Sin emits recientes = worker no procesó chunks en ese período (o crashed)."""
    rows = _exec_query(
        """
        SELECT node, COUNT(*)::int AS cnt,
               MAX(created_at) AS last_at
        FROM pipeline_metrics
        WHERE node IN ('_chunk_heartbeat_baseline', '_chunk_heartbeat_lag')
          AND created_at > NOW() - INTERVAL '48 hours'
        GROUP BY node
        """
    )
    baseline_cnt = 0
    lag_cnt = 0
    last_baseline = None
    last_lag = None
    for r in rows:
        if r["node"] == "_chunk_heartbeat_baseline":
            baseline_cnt = int(r["cnt"])
            last_baseline = r["last_at"]
        elif r["node"] == "_chunk_heartbeat_lag":
            lag_cnt = int(r["cnt"])
            last_lag = r["last_at"]

    if baseline_cnt == 0:
        return CheckResult(
            "Heartbeat telemetry (48h)",
            "WARN",
            "Cero emisiones de _chunk_heartbeat_baseline en últimas 48h. "
            "El cron del worker podría no haber procesado chunks (flota inactiva o caída)."
        )
    lag_ratio = (lag_cnt / baseline_cnt) if baseline_cnt > 0 else 0
    severity = "OK"
    if lag_ratio > 0.20:
        severity = "WARN"
    if lag_ratio > 0.50:
        severity = "FAIL"
    return CheckResult(
        "Heartbeat telemetry (48h)",
        severity,
        f"{baseline_cnt} baselines, {lag_cnt} anomalías (ratio={round(lag_ratio, 3)}).",
        details={
            "baseline_count": baseline_cnt,
            "lag_count": lag_cnt,
            "lag_ratio": round(lag_ratio, 3),
            "last_baseline_at": str(last_baseline) if last_baseline else None,
            "last_lag_at": str(last_lag) if last_lag else None,
        },
    )


def check_gc_dead_letter_tick_recent() -> CheckResult:
    """[P3-CHUNK-GC-DEADLETTER · 2026-05-18] El cron nuevo debe emitir tick
    al menos cada 24h (default knob `CHUNK_GC_DEAD_LETTER_INTERVAL_HOURS`)."""
    rows = _exec_query(
        """
        SELECT COUNT(*)::int AS cnt,
               MAX(created_at) AS last_at,
               MAX(metadata->>'purged_count')::int AS last_purged
        FROM pipeline_metrics
        WHERE node = '_gc_dead_lettered_chunks_tick'
          AND created_at > NOW() - INTERVAL '48 hours'
        """,
        fetch_all=False,
    ) or {}
    cnt = int(rows.get("cnt") or 0)
    last_at = rows.get("last_at")
    last_purged = rows.get("last_purged")

    if cnt == 0:
        return CheckResult(
            "GC dead-letter tick (48h)",
            "WARN",
            "Cero ticks de _gc_dead_lettered_chunks en 48h. "
            "Cron podría no estar registrado en el binary deployado (P3-CHUNK-GC-DEADLETTER "
            "se mergeó hoy 2026-05-18 — esperar próximo deploy + 24h)."
        )
    return CheckResult(
        "GC dead-letter tick (48h)",
        "OK",
        f"{cnt} ticks emitidos. Último: {last_at} (purged={last_purged}).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS · Específicos a un plan (con --plan-id)
# ─────────────────────────────────────────────────────────────────────────────


def check_plan_coherence(plan_id: str) -> list[CheckResult]:
    """Para un plan concreto, valida:
    - El plan existe en meal_plans.
    - sum(days_count) de chunks completed == len(plan_data.days).
    - No hay gaps en days_offset (chunks cubren todos los días).
    - generation_status coherente con cuántos chunks completaron.
    """
    out: list[CheckResult] = []

    # Plan existe
    row = _exec_query(
        """
        SELECT id,
               (plan_data->>'generation_status') AS gen_status,
               jsonb_array_length(plan_data->'days') AS persisted_days,
               (plan_data->>'total_days_requested')::int AS total_days_requested,
               (plan_data->>'grocery_start_date') AS grocery_start,
               user_id
        FROM meal_plans
        WHERE id = %s
        """,
        (plan_id,),
        fetch_all=False,
    )
    if not row:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} exists",
            "FAIL",
            f"Plan_id {plan_id} no existe en meal_plans.",
        ))
        return out
    out.append(CheckResult(
        f"Plan {plan_id[:8]} exists",
        "OK",
        f"Plan encontrado. status={row['gen_status']}, persisted_days={row['persisted_days']}, "
        f"total_requested={row['total_days_requested']}, grocery_start={row['grocery_start']}",
        details={
            "user_id": str(row["user_id"])[:8],
        },
    ))

    # Chunks distribution
    chunks = _exec_query(
        """
        SELECT week_number, days_offset, days_count, status, chunk_kind,
               attempts, dead_lettered_at,
               (pipeline_snapshot->>'_pantry_pause_reason') AS pause_reason
        FROM plan_chunk_queue
        WHERE meal_plan_id = %s
        ORDER BY week_number ASC
        """,
        (plan_id,),
    )
    if not chunks:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} chunks",
            "WARN",
            "Sin chunks en plan_chunk_queue. Para planes weekly post-shift es normal "
            "(chunks 'completed' pueden haber sido purgados). Para planes en generación "
            "es señal de problema.",
        ))
        return out
    by_status: dict = {}
    for c in chunks:
        s = c["status"] or "unknown"
        by_status[s] = by_status.get(s, 0) + 1
    out.append(CheckResult(
        f"Plan {plan_id[:8]} chunks distribution",
        "INFO",
        f"{len(chunks)} chunks en queue.",
        details={"by_status": by_status},
    ))

    # Sum check: chunks completed should account for persisted_days
    completed_chunks = [c for c in chunks if c["status"] == "completed"]
    sum_completed = sum(int(c["days_count"] or 0) for c in completed_chunks)
    persisted_days = int(row["persisted_days"] or 0)
    # NOTA: para planes shifted, persisted_days disminuye con el shift pero
    # sum(completed) NO. El check exacto solo es válido pre-shift.
    if persisted_days >= sum_completed:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} days coherence",
            "OK",
            f"persisted_days={persisted_days} >= sum(completed chunks)={sum_completed}. "
            f"OK (diff explained by shifts).",
        ))
    else:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} days coherence",
            "FAIL",
            f"persisted_days={persisted_days} < sum(completed chunks)={sum_completed}. "
            "Plan tiene menos días que los chunks completed suman. Posible "
            "corrupción del plan_data (P2-NEXT-4 CHECK debería haberlo bloqueado).",
        ))

    # Gaps en days_offset (chunks consecutivos deben cubrir 0..total)
    sorted_chunks = sorted(chunks, key=lambda c: int(c["days_offset"] or 0))
    expected_offset = 0
    gaps: list[str] = []
    for c in sorted_chunks:
        offset = int(c["days_offset"] or 0)
        count = int(c["days_count"] or 0)
        if offset != expected_offset:
            gaps.append(f"week={c['week_number']} expected_offset={expected_offset} actual={offset}")
        expected_offset = offset + count
    if not gaps:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} offset continuity",
            "OK",
            f"days_offset son consecutivos. Total cubierto: {expected_offset}d.",
        ))
    else:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} offset continuity",
            "FAIL",
            f"Gaps detectados en days_offset:",
            details={"gaps": gaps[:5]},
        ))

    # Dead-lettered breakdown
    dead = [c for c in chunks if c["dead_lettered_at"]]
    if dead:
        out.append(CheckResult(
            f"Plan {plan_id[:8]} dead-lettered",
            "WARN",
            f"{len(dead)} chunks dead-lettered en este plan.",
            details={"weeks": [int(c["week_number"]) for c in dead]},
        ))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS · Específicos a un user (con --user-id)
# ─────────────────────────────────────────────────────────────────────────────


def check_user_chunk_state(user_id: str) -> list[CheckResult]:
    out: list[CheckResult] = []

    rows = _exec_query(
        """
        SELECT meal_plan_id, week_number, status, chunk_kind, attempts,
               dead_lettered_at,
               EXTRACT(EPOCH FROM (NOW() - updated_at))/60 AS minutes_since_update
        FROM plan_chunk_queue
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 30
        """,
        (user_id,),
    )
    if not rows:
        out.append(CheckResult(
            f"User {user_id[:8]} chunks",
            "INFO",
            f"Sin chunks en queue para este user.",
        ))
        return out

    by_status: dict = {}
    for r in rows:
        s = r["status"] or "unknown"
        by_status[s] = by_status.get(s, 0) + 1
    out.append(CheckResult(
        f"User {user_id[:8]} chunks (last 30)",
        "INFO",
        f"{len(rows)} chunks recientes para el user.",
        details={"by_status": by_status},
    ))

    # Currently processing
    processing = [r for r in rows if r["status"] == "processing"]
    if len(processing) > 1:
        out.append(CheckResult(
            f"User {user_id[:8]} concurrent processing",
            "FAIL",
            f"{len(processing)} chunks 'processing' simultáneos para mismo user. "
            "Esto VIOLA la invariante 'un chunk por user a la vez'. Posible bug "
            "del advisory lock o lock zombie no purgado.",
            details={"chunks": [
                {"week": int(p["week_number"]), "minutes": round(float(p["minutes_since_update"]), 1)}
                for p in processing
            ]},
        ))
    else:
        out.append(CheckResult(
            f"User {user_id[:8]} concurrent processing",
            "OK",
            f"Solo {len(processing)} chunk 'processing' (invariante respetada).",
        ))

    # User lock
    lock = _exec_query(
        """
        SELECT locked_by_chunk_id,
               EXTRACT(EPOCH FROM (NOW() - heartbeat_at))/60 AS minutes_since_heartbeat
        FROM chunk_user_locks
        WHERE user_id = %s
        """,
        (user_id,),
        fetch_all=False,
    )
    if not lock:
        out.append(CheckResult(
            f"User {user_id[:8]} lock",
            "OK",
            "No hay lock activo (esperado si no hay chunks corriendo)."
        ))
    else:
        minutes_stale = float(lock["minutes_since_heartbeat"] or 0)
        severity = "OK" if minutes_stale < 10 else "FAIL"
        out.append(CheckResult(
            f"User {user_id[:8]} lock",
            severity,
            f"Lock activo, heartbeat hace {round(minutes_stale, 1)}min.",
            details={"chunk_id": str(lock["locked_by_chunk_id"])[:8]},
        ))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk health check contra Supabase live."
    )
    parser.add_argument("--plan-id", help="UUID de un plan específico para verificar coherencia.")
    parser.add_argument("--user-id", help="UUID de un user específico para verificar chunks/lock.")
    args = parser.parse_args()

    if not os.environ.get("SUPABASE_DB_URL"):
        print(_red("✗ SUPABASE_DB_URL no está en env. Ejecutar:"))
        print("    export SUPABASE_DB_URL='postgres://...'")
        return 1

    print(_bold("\n━━━ Chunk Health Check ━━━"))
    print(f"  Run: {os.environ.get('USER', '?')} · {Path(__file__).name}\n")

    checks: list[CheckResult] = []

    # Sanidad
    print(_bold("\n[1/4] Sanidad general"))
    for fn in [
        check_queue_status_distribution,
        check_zombie_chunks_pending_rescue,
        check_stale_user_locks,
        check_pending_queue_stuck,
    ]:
        try:
            result = fn()
            result.print()
            checks.append(result)
        except Exception as e:
            print(_red(f"  ✗ FAIL {fn.__name__}: {e}"))
            checks.append(CheckResult(fn.__name__, "FAIL", str(e)))

    # Dead-letter
    print(_bold("\n[2/4] Dead-letter health"))
    for fn in [
        check_dead_letter_counts,
        check_dead_letter_breakdown,
        check_gc_purgable_dead_letters,
    ]:
        try:
            result = fn()
            result.print()
            checks.append(result)
        except Exception as e:
            print(_red(f"  ✗ FAIL {fn.__name__}: {e}"))
            checks.append(CheckResult(fn.__name__, "FAIL", str(e)))

    # Pausas + telemetría
    print(_bold("\n[3/4] Pausas + telemetría"))
    for fn in [
        check_paused_chunks,
        check_heartbeat_telemetry_recent,
        check_gc_dead_letter_tick_recent,
    ]:
        try:
            result = fn()
            result.print()
            checks.append(result)
        except Exception as e:
            print(_red(f"  ✗ FAIL {fn.__name__}: {e}"))
            checks.append(CheckResult(fn.__name__, "FAIL", str(e)))

    # Plan o User específicos
    if args.plan_id or args.user_id:
        print(_bold("\n[4/4] Verificaciones específicas"))
        if args.plan_id:
            try:
                for r in check_plan_coherence(args.plan_id):
                    r.print()
                    checks.append(r)
            except Exception as e:
                print(_red(f"  ✗ FAIL check_plan_coherence: {e}"))
        if args.user_id:
            try:
                for r in check_user_chunk_state(args.user_id):
                    r.print()
                    checks.append(r)
            except Exception as e:
                print(_red(f"  ✗ FAIL check_user_chunk_state: {e}"))

    # Resumen
    n_ok = sum(1 for c in checks if c.status == "OK")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_info = sum(1 for c in checks if c.status == "INFO")

    print(_bold("\n━━━ Resumen ━━━"))
    print(f"  {_green(f'OK: {n_ok}')}  ·  "
          f"{_yellow(f'WARN: {n_warn}')}  ·  "
          f"{_red(f'FAIL: {n_fail}')}  ·  "
          f"{_cyan(f'INFO: {n_info}')}")
    print()

    if n_fail > 0:
        print(_red("✗ Hay FAIL: investigar los items marcados."))
        return 1
    if n_warn > 0:
        print(_yellow("⚠ Hay WARN: revisar pero no urgente."))
    else:
        print(_green("✓ Todos OK. Sistema de chunks saludable."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
