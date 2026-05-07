import pytest
from unittest.mock import patch, MagicMock

# [test fix] Antes este archivo stub-eaba langgraph parcialmente (solo
# langgraph.graph y langgraph.graph.message) para entornos donde el paquete no
# estuviera instalado. Ahora cron_tasks importa transitiva langgraph.checkpoint.memory
# vía agent.py, así que el stub parcial rompía la resolución del submódulo. El
# entorno conda 'mealfit' que corre los tests SÍ tiene langgraph instalado, así que
# los stubs son innecesarios — los quitamos y dejamos que Python cargue la versión real.

from cron_tasks import process_plan_chunk_queue

def test_chunk_user_race_condition():
    """
    Valida que _chunk_worker hace `INSERT INTO chunk_user_locks ... ON CONFLICT
    DO NOTHING RETURNING user_id` y, cuando la fila ya existe (otro worker tiene
    el lock), defiere el chunk vía `UPDATE plan_chunk_queue SET status = 'pending'
    WHERE id = %s` y retorna sin procesar.

    [test rewrite] La versión anterior usaba ThreadPoolExecutor mockeado con
    `mock_map` que corría tasks sequencialmente: task 1 adquiría el lock, corría
    a completar, su `finally` liberaba el lock, y task 2 lo encontraba libre →
    sin deferral observable. El test no probaba lo que decía probar.

    Aquí simulamos el escenario directo: mockeamos `INSERT INTO chunk_user_locks`
    para devolver SIEMPRE `[]` (lock ya tomado por un worker externo). Con un solo
    task corriendo, esperamos UN intento de INSERT y UN UPDATE de deferral. No
    necesitamos ThreadPoolExecutor — corremos `_chunk_worker` directamente.
    """
    from cron_tasks import process_plan_chunk_queue

    mock_tasks = [
        {"id": 1, "user_id": "uuid-123", "meal_plan_id": "plan-1", "week_number": 1,
         "days_offset": 0, "days_count": 7, "pipeline_snapshot": "{}"},
    ]

    def side_effect_sql_write(query, params=None, returning=False):
        # Pickup batch de tareas (retorna las tasks mockeadas).
        if "UPDATE plan_chunk_queue" in query and "FOR UPDATE SKIP LOCKED" in query:
            return mock_tasks
        # Lock siempre ocupado por un "tercero" → INSERT no devuelve fila → defer.
        if "INSERT INTO chunk_user_locks" in query:
            return []
        # Las demás escrituras devuelven [] benignamente.
        return []

    with patch("cron_tasks.execute_sql_write", side_effect=side_effect_sql_write) as mock_write, \
         patch("cron_tasks.execute_sql_query", return_value={"id": "plan-1", "status": "active"}), \
         patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:

        # Corremos _chunk_worker sincrónicamente para no introducir threads en el test.
        def mock_map(func, iterable):
            for i in iterable:
                func(i)

        mock_exec_instance = mock_executor.return_value.__enter__.return_value
        mock_exec_instance.map = mock_map

        process_plan_chunk_queue()

        # 1 intento de adquirir el lock.
        lock_inserts = [
            call for call in mock_write.call_args_list
            if "INSERT INTO chunk_user_locks" in call[0][0]
        ]
        assert len(lock_inserts) == 1, \
            f"Debería intentar adquirir el lock 1 vez, hubo {len(lock_inserts)}"

        # [P0-5] El chunk fue diferido vía CAS-protected `_cas_update_chunk_status`
        # con new_status='pending'. La query es ahora:
        #   UPDATE plan_chunk_queue
        #   SET status = %s, updated_at = NOW()
        #   WHERE id = %s AND attempts = %s AND status = %s
        #   RETURNING id
        # El test antes buscaba `SET status = 'pending'` literal con valor inline,
        # pero el CAS helper usa parámetros (%s) para evitar SQL-injection y para
        # transitar atómicamente. Filtramos por la firma `SET status = %s ... AND
        # attempts = %s AND status = %s` y comprobamos que `'pending'` es el primer
        # parámetro pasado.
        deferrals = [
            call for call in mock_write.call_args_list
            if "UPDATE plan_chunk_queue" in call[0][0]
            and "SET status = %s" in call[0][0]
            and "AND attempts = %s" in call[0][0]
            and call[0][1][0] == "pending"
        ]
        assert len(deferrals) == 1, \
            f"El chunk debió ser diferido al fallar el lock, hubo {len(deferrals)} deferrals"

        # Como nunca adquirimos el lock para el task, NO debe haber DELETE específico
        # `DELETE FROM chunk_user_locks WHERE locked_by_chunk_id = %s` (el del finally).
        # El housekeeping de orphan cleanup `DELETE ... WHERE heartbeat_at < NOW() - ...`
        # SÍ se ejecuta como parte del cron y es legítimo — lo filtramos.
        worker_lock_deletes = [
            call for call in mock_write.call_args_list
            if "DELETE FROM chunk_user_locks" in call[0][0]
            and "locked_by_chunk_id" in call[0][0]
        ]
        assert len(worker_lock_deletes) == 0, \
            "No debe liberarse un lock que nunca se adquirió"
