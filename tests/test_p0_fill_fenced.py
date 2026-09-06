# -*- coding: utf-8 -*-
"""[P0-FILL-FENCED · 2026-09-05] El worker desplazado podía RELLENAR el plan.

`run_initial_chunk` llamaba al postprocesado —que escribe el plan y lanza sus tareas de fondo— y comprobaba el
fencing DESPUÉS, en el CAS de `completed`. El propio mensaje de error lo admitía: «Plan ya persistido». Impedir
un SEGUNDO llenado no impide que el PRIMERO lo haga un worker que ya perdió su lease.

Ahora el token del claim viaja dentro del plan y se comprueba con `FOR UPDATE` sobre la fila de la cola, dentro
de la misma transacción que publica los días.

LO QUE ESTE TEST NO HACE: la carrera real contra PostgreSQL con dos workers. Aquí la conexión es falsa y lo que
se prueba es que, cuando la fila de la cola dice que el trabajo ya no es nuestro, **no se ejecuta ni un UPDATE
sobre meal_plans**. La carrera contra una base efímera sigue pendiente y no se da por hecha."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import db_plans  # noqa: E402


class _Cursor:
    """Cursor falso: registra el SQL ejecutado y responde a las dos SELECT del camino."""

    def __init__(self, queue_row, plan_row):
        self.queue_row = queue_row
        self.plan_row = plan_row
        self.sql = []
        self._last = None
        self.rowcount = 1

    def execute(self, q, params=None):
        self.sql.append(" ".join(str(q).split()))
        low = self.sql[-1].lower()
        if "from plan_chunk_queue" in low:
            self._last = self.queue_row
        elif "select plan_data from meal_plans" in low:
            self._last = self.plan_row
        else:
            self._last = None

    def fetchone(self):
        return self._last

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self, **kw):
        return self._cur

    def transaction(self):
        return self._cur          # sirve de context manager

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Pool:
    def __init__(self, cur):
        self._cur = cur

    def connection(self):
        return _Conn(self._cur)


def _preparar(monkeypatch, queue_row):
    cur = _Cursor(queue_row, {"plan_data": {"generation_status": "generating"}})
    monkeypatch.setattr(db_plans, "connection_pool", _Pool(cur))
    monkeypatch.setattr(db_plans, "set_meal_plan_for_update_timeouts", lambda c: None)
    monkeypatch.setattr(db_plans, "acquire_meal_plan_advisory_lock", lambda c, p, purpose=None: None)
    monkeypatch.setattr(db_plans, "_apply_inherited_lifetime_lessons", lambda u, d, cursor=None: None)
    monkeypatch.setattr(db_plans, "_finalize_plan_data_for_insert", lambda d: None)
    return cur


def _insert_data(attempts=3, task="11111111-1111-1111-1111-111111111111"):
    return {"plan_data": {"generation_status": "complete", "days": [{"meals": []}],
                          "_chunk_fence": {"task_id": task, "attempts": attempts}},
            "name": "Plan"}


def _escribio_el_plan(cur):
    return any(s.lower().startswith("update meal_plans") for s in cur.sql)


def test_el_worker_desplazado_no_escribe_el_plan(monkeypatch):
    """Otro worker reclamó: la fila de la cola ya va por attempts=4 y nosotros creíamos ser el 3."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 4})
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=3))
    assert out is None
    assert not _escribio_el_plan(cur), f"escribió pese a estar desplazado: {cur.sql}"


def test_el_chunk_cancelado_no_escribe_el_plan(monkeypatch):
    cur = _preparar(monkeypatch, {"status": "cancelled", "attempts": 3})
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=3))
    assert out is None
    assert not _escribio_el_plan(cur)


def test_el_chunk_desaparecido_no_escribe_el_plan(monkeypatch):
    """Si la fila de la cola ya no existe, no hay trabajo que publicar."""
    cur = _preparar(monkeypatch, None)
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=3))
    assert out is None
    assert not _escribio_el_plan(cur)


def test_el_dueño_legitimo_si_escribe(monkeypatch):
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 3})
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=3))
    assert out == "plan-1"
    assert _escribio_el_plan(cur), cur.sql


def test_sin_token_el_camino_de_siempre_no_cambia(monkeypatch):
    """El path HTTP no pasa por la cola y no trae token: no puede quedarse sin escribir por eso."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 3})
    data = _insert_data()
    data["plan_data"].pop("_chunk_fence")
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", data)
    assert out == "plan-1"
    assert not any("plan_chunk_queue" in s for s in cur.sql), "sin token no se consulta la cola"


def test_el_token_no_llega_a_la_base(monkeypatch):
    """`_chunk_fence` es transporte: se saca del plan antes de construir el UPDATE."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 3})
    db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=3))
    assert not any("_chunk_fence" in s for s in cur.sql), cur.sql


def test_la_comprobacion_va_antes_del_update_y_bloquea_la_fila():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("def fill_placeholder_meal_plan_atomic")
    # 8.000 y no 6.000: entre el fencing (offset ~4.000) y el UPDATE (~6.040) van el lock, la lectura del
    # placeholder y el constructor de columnas. Medido, no estimado — un margen corto deja el UPDATE fuera de
    # la ventana y el test pasa a comprobar otra cosa.
    cuerpo = src[i:i + 8000]
    a = cuerpo.index("FROM plan_chunk_queue WHERE id = %s FOR UPDATE")
    b = cuerpo.index("UPDATE meal_plans SET")
    assert a < b, "el fencing tiene que preceder a la escritura"
    assert "FOR UPDATE" in cuerpo[a:a + 60], "sin FOR UPDATE es una comprobación optimista, no una garantía"


def test_el_orquestador_pone_el_token_antes_del_postprocesado():
    src = (_BACKEND / "generation_lifecycle.py").read_text(encoding="utf-8")
    i = src.index('result["_chunk_fence"]')
    j = src.index("_postprocess_pipeline_result(", i - 4000 if i > 4000 else 0)
    assert i < src.index("_postprocess_pipeline_result(", i), "el token se pone antes de postprocesar"
    bloque = src[i:i + 200]
    assert '"attempts": int(pickup_attempts)' in bloque and '"task_id": str(task_id)' in bloque


def test_el_primer_intento_escribe_aunque_attempts_sea_cero(monkeypatch):
    """[corregido 2026-09-06] El bug que este archivo NO cazaba porque todos sus casos usaban attempts=3.

    `int(x or -1)` convierte 0 en -1 —cero es falso en Python— y el primer intento de TODO chunk lleva
    attempts=0, así que el fence rechazaba la escritura legítima de cada plan. Visto en producción a los 40
    minutos de desplegarlo: «attempts=-1 vs esperado 0» en el plan c5ba1681. El plan se salvó solo porque el
    recovery re-reclamó el chunk y en el segundo intento attempts=1 ya es verdadero.

    Ausente y cero son estados distintos. Es la misma invariante que el roadmap 2.6 llama I20."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": 0})
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=0))
    assert out == "plan-1", "el primer intento de un chunk tiene que poder escribir"
    assert _escribio_el_plan(cur), cur.sql


def test_la_fila_sin_attempts_no_escribe(monkeypatch):
    """Ausente sigue siendo distinto de cero: si la columna viene NULL, no sabemos de quién es el trabajo."""
    cur = _preparar(monkeypatch, {"status": "processing", "attempts": None})
    out = db_plans.fill_placeholder_meal_plan_atomic("plan-1", "user-1", _insert_data(attempts=0))
    assert out is None
    assert not _escribio_el_plan(cur)
