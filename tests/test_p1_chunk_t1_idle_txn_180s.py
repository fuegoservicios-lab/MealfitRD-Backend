"""[P1-CHUNK-T1-IDLE-TXN-180S · 2026-09-04] El T1 del chunk worker vive más de 60 s sin hablar con la DB.

Bloque 2 del plan 05fb9d22 (dueño, gain_muscle, 4 días), dos intentos seguidos, mismo final:
revisor aprueba (16/16 en banda) → waiver post-merge → coherence-finalize del T1 → a los ~60 s
del último statement del cursor, `psycopg.errors.IdleInTransactionSessionTimeout` en el
`UPDATE meal_plans SET plan_data` final (cron_tasks.py ~32654). El chunk vuelve a la cola y
repite los 10 minutos.

La transacción del T1 abre con `set_meal_plan_for_update_timeouts` (P0-PERSIST-TXN-IDLE,
2026-07-10: override de 60 s sobre los 15 s de sesión, cuando el tramo medía 10-20 s). Entre el
último `cursor.execute` (lecciones) y el UPDATE final corre CPU-only: finalize + quantize +
consolidación + json.dumps de 4 días; los `execute_sql_write` intermedios van por OTRA conexión
y no reinician el reloj de inactividad. Hoy ese tramo supera los 60 s. Default → 180 s en las
DOS lecturas del knob (helper FOR UPDATE y `update_plan_data_atomic`). El fondo (mover el CPU
fuera del lock) queda como follow-up de perf.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_both_readers_default_to_180s():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert '_env_int("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", 180000)' in src
    assert '_env_int_atomic("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", 180000)' in src
    # ninguna lectura del knob conserva el default viejo
    assert not re.search(r'IDLE_TXN_TIMEOUT_MS",\s*60000\)', src)


def test_chunk_worker_t1_uses_the_helper_before_for_update():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index('cursor.execute("SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE", (meal_plan_id,))')
    before = src[max(0, i - 3000):i]
    assert "set_meal_plan_for_update_timeouts(cursor)" in before


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-CHUNK-T1-IDLE-TXN-180S · 2026-09-04" in app
