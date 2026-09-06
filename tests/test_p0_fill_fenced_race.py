# -*- coding: utf-8 -*-
"""[P0-FILL-FENCED · carrera real · 2026-09-06] La prueba que faltaba: DOS workers contra un PostgreSQL de
verdad, no una conexión falsa.

`tests/test_p0_fill_fenced.py` comprueba la lógica con un cursor simulado, y eso deja fuera lo único que
importa de un fence: que el `FOR UPDATE` serialice de verdad a dos escritores concurrentes. Aquí se monta el
caso: el worker A tiene el claim, B se lo roba mientras A prepara el plan, y A intenta escribir.

NO CORRE POR DEFECTO. Necesita una base PostgreSQL desechable y se salta si no hay ninguna:

    docker run -d --rm --name pg-race -e POSTGRES_PASSWORD=race -p 55432:5432 postgres:16-alpine
    MEALFIT_TEST_PG_URL=postgresql://postgres:race@localhost:55432/postgres pytest tests/test_p0_fill_fenced_race.py
    docker rm -f pg-race

Se hizo así, y no añadiendo Docker a las dependencias del repo, porque el repo nunca lo ha usado y una
dependencia nueva sobrevive al test que la trajo. Contra cualquier PostgreSQL sirve: la URL la pones tú.

⚠️ NUNCA apuntes `MEALFIT_TEST_PG_URL` a producción: el test CREA y BORRA tablas."""
from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_PG = os.environ.get("MEALFIT_TEST_PG_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not _PG, reason="carrera real: define MEALFIT_TEST_PG_URL con un PostgreSQL desechable (ver docstring)")

_ESQUEMA = """
DROP TABLE IF EXISTS meal_plans;
DROP TABLE IF EXISTS plan_chunk_queue;
CREATE TABLE meal_plans (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    plan_data jsonb NOT NULL,
    name text
);
CREATE TABLE plan_chunk_queue (
    id uuid PRIMARY KEY,
    status text NOT NULL,
    attempts int NOT NULL
);
"""


@pytest.fixture()
def entorno(monkeypatch):
    """Base limpia + `db_plans` apuntando a ella, con los ayudantes que no aportan nada a la carrera
    neutralizados (el lock consultivo y el finalize de plan_data son de producción, no de este contrato)."""
    import psycopg
    import psycopg_pool
    import db_plans

    plan_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    with psycopg.connect(_PG, autocommit=True) as c:
        with c.cursor() as cur:
            cur.execute(_ESQUEMA)
            cur.execute("INSERT INTO plan_chunk_queue (id, status, attempts) VALUES (%s, 'processing', 0)",
                        (task_id,))
            cur.execute("INSERT INTO meal_plans (id, user_id, plan_data, name) VALUES (%s, %s, %s, %s)",
                        (plan_id, "u1", '{"generation_status": "generating"}', "Placeholder"))

    pool = psycopg_pool.ConnectionPool(_PG, min_size=2, max_size=6, open=True)
    monkeypatch.setattr(db_plans, "connection_pool", pool)
    monkeypatch.setattr(db_plans, "set_meal_plan_for_update_timeouts", lambda c: None)
    monkeypatch.setattr(db_plans, "acquire_meal_plan_advisory_lock", lambda c, p, purpose=None: None)
    monkeypatch.setattr(db_plans, "_apply_inherited_lifetime_lessons", lambda u, d, cursor=None: None)
    monkeypatch.setattr(db_plans, "_finalize_plan_data_for_insert", lambda d: None)
    try:
        yield {"plan_id": plan_id, "task_id": task_id, "pool": pool, "db_plans": db_plans}
    finally:
        pool.close()


def _datos(task_id, attempts, dias):
    return {"plan_data": {"generation_status": "complete", "days": dias,
                          "_chunk_fence": {"task_id": task_id, "attempts": attempts}},
            "name": "Plan escrito por el worker"}


def _dias_persistidos(pool, plan_id):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT plan_data->'days', name FROM meal_plans WHERE id = %s", (plan_id,))
        return cur.fetchone()


def test_el_worker_desplazado_no_escribe_ni_una_fila(entorno):
    """A pierde el lease mientras prepara; B reclama. A NO puede publicar."""
    dbp, pool = entorno["db_plans"], entorno["pool"]
    plan_id, task_id = entorno["plan_id"], entorno["task_id"]

    # B reclama: mismo chunk, siguiente intento.
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE plan_chunk_queue SET attempts = 1 WHERE id = %s", (task_id,))
        conn.commit()

    assert dbp.fill_placeholder_meal_plan_atomic(plan_id, "u1", _datos(task_id, 0, [{"a": 1}])) is None
    dias, nombre = _dias_persistidos(pool, plan_id)
    assert dias is None, f"el desplazado escribió días: {dias}"
    assert nombre == "Placeholder", f"el desplazado pisó el nombre: {nombre}"


def test_el_dueno_legitimo_publica(entorno):
    dbp, pool = entorno["db_plans"], entorno["pool"]
    plan_id, task_id = entorno["plan_id"], entorno["task_id"]
    assert dbp.fill_placeholder_meal_plan_atomic(plan_id, "u1", _datos(task_id, 0, [{"a": 1}])) == plan_id
    dias, nombre = _dias_persistidos(pool, plan_id)
    assert dias == [{"a": 1}] and nombre == "Plan escrito por el worker"


def test_dos_workers_a_la_vez_solo_uno_publica(entorno):
    """La carrera de verdad: los dos creen ser el dueño y arrancan a la vez. El `FOR UPDATE` los serializa,
    y el segundo encuentra la fila ya cambiada por el primero."""
    dbp, pool = entorno["db_plans"], entorno["pool"]
    plan_id, task_id = entorno["plan_id"], entorno["task_id"]
    resultados = {}
    barrera = threading.Barrier(2)

    def worker(nombre, attempts, dias):
        barrera.wait()
        try:
            resultados[nombre] = dbp.fill_placeholder_meal_plan_atomic(plan_id, "u1",
                                                                      _datos(task_id, attempts, dias))
        except Exception as e:                       # noqa: BLE001
            resultados[nombre] = f"error:{type(e).__name__}"

    # A cree ser el intento 0; B ya reclamó y es el 1. Solo uno de los dos coincide con la fila.
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE plan_chunk_queue SET attempts = 1 WHERE id = %s", (task_id,))
        conn.commit()

    hilos = [threading.Thread(target=worker, args=("A", 0, [{"de": "A"}])),
             threading.Thread(target=worker, args=("B", 1, [{"de": "B"}]))]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=30)

    assert resultados.get("A") is None, f"el desplazado publicó: {resultados}"
    assert resultados.get("B") == plan_id, f"el dueño legítimo no publicó: {resultados}"
    dias, _ = _dias_persistidos(pool, plan_id)
    assert dias == [{"de": "B"}], dias


def test_el_chunk_cancelado_no_publica(entorno):
    dbp, pool = entorno["db_plans"], entorno["pool"]
    plan_id, task_id = entorno["plan_id"], entorno["task_id"]
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
        conn.commit()
    assert dbp.fill_placeholder_meal_plan_atomic(plan_id, "u1", _datos(task_id, 0, [{"a": 1}])) is None
    assert _dias_persistidos(pool, plan_id)[0] is None
