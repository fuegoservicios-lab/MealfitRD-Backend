# -*- coding: utf-8 -*-
"""[P1-PAUSE-GC-SURVIVAL · 2026-08-12] La cola pausada debe SOBREVIVIR al propio
sistema hasta que el usuario reanude.

La secuela del primer ciclo real pausa→reanuda (plan f380821a): el revive de
P1-RESUME-REVIVES-QUEUE devolvía las filas a pending, pero tres mecanismos del
worker las destruían igual, cada uno por su lado:

  1. El GC eager (GAP 7) vacía `pipeline_snapshot` de TODO cancelled en el
     siguiente tick (1 min) — las filas revividas quedaban HUECAS: pending sin
     form_data, imposibles de generar.
  2. El TZ-SYNC (P0-5) leía ese snapshot vacío con `COALESCE(..., 0)` —
     fabricando "tz 0" de la AUSENCIA — y sumaba +240min por tick, DOS crons a
     la vez. Su fix-forward era un no-op silencioso: `jsonb_set` con path
     '{form_data,tzOffset}' no puede crear el padre `form_data` sobre '{}'.
     ~220 ticks después exec estaba en septiembre y el dead-letter cron
     (correctamente) escalaba `execute_after_beyond_plan_window`.
  3. La purga definitiva (GAP 11) BORRA cancelled >48h — reanudar al tercer
     día no encontraba nada que revivir (ventana de reanudación: 30 días).

Tres cierres, uno por mecanismo: el TZ-SYNC ya no inventa el 0 (ausencia ⇒
NULL ⇒ rama defensiva que NO mueve exec) y sus escrituras construyen el padre;
la purga exime a las filas firmadas por la pausa dentro de la ventana; y el
revive RECONSTRUYE los snapshots desde el perfil (los vacíos son la norma, no
la excepción — el GC llega antes que cualquier reanudación humana).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import cron_tasks as ct
import plan_mode as pm

SRC_CRON = Path(ct.__file__).read_text(encoding="utf-8")
SRC_PM = Path(pm.__file__).read_text(encoding="utf-8")


def _cuerpo(src: str, desde: str, hasta: str) -> str:
    i = src.index(desde)
    return src[i:src.index(hasta, i)]


# ── 1. TZ-SYNC no fabrica certeza de la ausencia ────────────────────────────

TZSYNC = _cuerpo(SRC_CRON, "def _sync_chunk_queue_tz_offsets", "\ndef _read_inventory_live_failure_log")


def test_tzsync_snapshot_sin_tz_es_null_no_cero():
    """El SELECT del TZ-SYNC: un snapshot sin form_data/tz debe salir NULL para
    caer en la rama defensiva (persiste live_tz SIN mover execute_after). El
    `COALESCE(..., 0)` original convertía "no sé" en "tz 0" ⇒ drift=240 ⇒
    +4h/tick en bucle infinito (medido: exec de agosto empujado a septiembre)."""
    m = re.search(r"COALESCE\((.*?)\)\s+AS snapshot_tz", TZSYNC, re.DOTALL)
    assert m, "no se encontró el COALESCE de snapshot_tz en el SELECT"
    brazos = m.group(1)
    assert not re.search(r",\s*0\s*$", brazos.strip()), (
        "el COALESCE de snapshot_tz termina en `, 0`: eso fabrica 'tz 0' de un "
        "snapshot VACÍO (el GC los vacía al minuto) y revive el bucle +4h/tick"
    )


def test_tzsync_escrituras_construyen_el_padre_form_data():
    """Las 3 escrituras del TZ-SYNC persisten el tz al snapshot. Con path
    '{form_data,tzOffset}' sobre un snapshot '{}' jsonb_set NO crea el padre y
    la escritura es un no-op SILENCIOSO — el próximo tick vuelve a leer la
    ausencia y el bucle no converge jamás. Deben escribir '{form_data}' entero
    (COALESCE del padre || jsonb_build_object)."""
    assert "'{form_data,tzOffset}'" not in TZSYNC and "'{form_data,tz_offset_minutes}'" not in TZSYNC, (
        "queda una escritura con path hijo '{form_data,...}': no-op sobre snapshot vacío"
    )
    escrituras = re.findall(r"COALESCE\(pipeline_snapshot->'form_data',\s*'\{\}'::jsonb\)\s*\|\|\s*jsonb_build_object", TZSYNC)
    assert len(escrituras) >= 3, (
        f"esperaba las 3 escrituras (defensiva, force_now, shift) construyendo el "
        f"padre form_data; encontradas {len(escrituras)}"
    )


def test_tzsync_rama_defensiva_no_mueve_exec():
    """La rama snapshot_tz IS NULL persiste live_tz pero JAMÁS toca execute_after:
    el delta original es incognoscible (el snapshot que lo sabía fue vaciado)."""
    rama = _cuerpo(TZSYNC, "if snapshot_tz is None:", "continue")
    assert not re.search(r"execute_after\s*=", rama), (
        "la rama defensiva ASIGNA execute_after: sin snapshot_tz el delta es incognoscible"
    )


# ── 2. La purga de 48h respeta la ventana de reanudación ────────────────────

def test_purga_48h_exime_a_las_filas_firmadas_por_la_pausa():
    """GAP 11 borra cancelled >48h, pero la reanudación tiene 30 días
    (MEALFIT_PLAN_PAUSE_MAX_RESUME_DAYS): purgar las filas firmadas por la pausa
    al segundo día dejaba resume>2d sin cola que revivir — plan partial PARA
    SIEMPRE. La firma sobrevive la ventana entera; vencida, se purga igual."""
    purga = _cuerpo(SRC_CRON, "[GAP 11 FIX", "Error purgando chunks cancelados")
    assert "dead_letter_reason = %s" in purga, "la purga no distingue la firma de la pausa"
    assert "dead_lettered_at IS NULL" in purga
    assert re.search(r"make_interval\(days\s*=>\s*%s\)", purga), (
        "la exención debe expirar con la ventana de reanudación (make_interval days => %s)"
    )
    assert "PAUSE_CANCEL_REASON" in purga and "_resume_max_days" in purga, (
        "la exención debe leer firma y ventana del SSOT plan_mode, no duplicarlas"
    )


# ── 3. El revive reconstruye los snapshots desde el perfil ──────────────────

def _pool_con(cursor):
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value.transaction.return_value.__enter__.return_value = MagicMock()
    pool.connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cursor
    return pool


def test_revive_reconstruye_snapshot_desde_el_perfil(monkeypatch):
    """Funcional: las filas revividas SIEMPRE están huecas (el GC del worker vació
    su snapshot al minuto de la pausa). El revive debe reconstruirlo con la forma
    del catch-up: form_data = {**hp, user_id, totalDays, _plan_start_date=d0} +
    previous_meals de los días vivos + _is_rolling_refill."""
    hp = {"age": 30, "tz_offset_minutes": 240, "weight": 80}
    cursor = MagicMock()
    ejecutadas = []

    def _exec(sql, params=None):
        ejecutadas.append((" ".join(str(sql).split()), params))
    cursor.execute.side_effect = _exec
    cursor.fetchone.side_effect = [
        {"health_profile": hp},                      # perfil
        {"d": 3, "d0": "2026-08-12",                  # info del plan
         "days": [{"meals": [{"name": "Mangú"}, {"name": "Pollo guisado"}]}]},
    ]
    cursor.fetchall.return_value = [
        {"id": "c1", "meal_plan_id": "plan-aa", "days_count": 4},
        {"id": "c2", "meal_plan_id": "plan-aa", "days_count": 3},
    ]
    monkeypatch.setattr("db_core.connection_pool", _pool_con(cursor))
    monkeypatch.setattr(ct, "_rebase_pending_chunk_offsets_sql", lambda cur, pid, dias: 0)

    out = pm._revive_paused_chunks("u-hueco")
    assert out["revived"] == 2

    rebuilds = [(s, p) for s, p in ejecutadas if "SET pipeline_snapshot = %s::jsonb" in s]
    assert len(rebuilds) == 2, "cada fila revivida debe recibir su snapshot reconstruido"
    snap = json.loads(rebuilds[0][1][0])
    fd = snap["form_data"]
    assert fd["age"] == 30 and fd["user_id"] == "u-hueco", "form_data debe nacer del health_profile + user_id"
    assert fd["totalDays"] == 4 and snap["totalDays"] == 4, "totalDays = days_count DE ESA fila"
    assert fd["_plan_start_date"] == "2026-08-12", "_plan_start_date = d0 vivo del plan (ancla actual)"
    assert "Mangú" in snap["previous_meals"], "previous_meals sale de los días vivos (anti-repetición)"
    assert snap["_is_rolling_refill"] is True
    # la 2ª fila lleva SU days_count, no el de la 1ª
    assert json.loads(rebuilds[1][1][0])["totalDays"] == 3


def test_revive_sin_perfil_no_revive_nada(monkeypatch):
    """Sin health_profile no hay materia prima: revivir produciría pending
    imposibles de generar que el dead-letter escala a `_user_action_required`
    (peor que seguir en pausa). Cero filas tocadas, error GRITADO."""
    cursor = MagicMock()
    cursor.fetchone.return_value = {"health_profile": None}
    cursor.fetchall.return_value = []
    monkeypatch.setattr("db_core.connection_pool", _pool_con(cursor))

    out = pm._revive_paused_chunks("u-sin-perfil")
    assert out == {"revived": 0, "plans": 0}
    updates = [c for c in cursor.execute.call_args_list if "SET status = 'pending'" in str(c)]
    assert not updates, "sin perfil NO debe flipear filas a pending"
