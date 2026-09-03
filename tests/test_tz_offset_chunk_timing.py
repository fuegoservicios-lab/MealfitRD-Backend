import datetime
from unittest.mock import patch, MagicMock

@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_timezone_alignment_rd(mock_query, mock_write):
    import cron_tasks
    # [test-drift fix] Tras [P0-1] el INSERT del chunk dejó de hacerse vía
    # `execute_sql_write` y ahora se hace con un UPSERT atómico
    # (`execute_sql_query(upsert_sql, params, fetch_one=True)`), cuyo
    # `RETURNING id, status, (xmax=0) AS inserted` distingue insert/update/skip.
    # El `execute_after` (lo que verifica este test) viaja en los params de ese
    # UPSERT. Devolvemos un dict truthy con `inserted=True` para reflejar un
    # INSERT real (return None dispararía la rama "skip-active"). El mismo mock
    # responde al SELECT health_profile de `_get_user_tz_live`; como el dict no
    # trae `health_profile`, ese lookup cae al fallback tzOffset=240 del snapshot.
    mock_query.return_value = {"id": "chunk-1", "status": "pending", "inserted": True}

    snapshot = {
        "form_data": {
            "_plan_start_date": "2024-05-01T04:00:00+00:00", # Midnight local time RD (UTC-4) = 04:00 UTC
            "tzOffset": 240 # RD is 240 minutes behind UTC
        }
    }

    with patch("cron_tasks.datetime") as mock_dt:
        mock_dt.now.return_value = datetime.datetime(2024, 5, 2, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_dt.combine = datetime.datetime.combine
        mock_dt.min = datetime.datetime.min

        cron_tasks._enqueue_plan_chunk(
            user_id="u1",
            meal_plan_id="m1",
            week_number=2,
            days_offset=3,  # Delay days should be 3
            days_count=3,
            pipeline_snapshot=snapshot
        )

    # Asserting that the atomic UPSERT was issued with correct execute_after.
    assert mock_query.called
    args, kwargs = mock_query.call_args  # último call == el UPSERT del chunk
    # Verify execute_after.
    # start_dt_midnight_utc = 2024-05-01 00:00:00 UTC
    # execute_dt_target = 2024-05-01 00:00:00 UTC + 3 days (2024-05-04) + 270 mins (4h30m) = 2024-05-04 04:30:00 UTC
    # 04:30 UTC is 00:30 RD time.
    query = args[0]
    params = args[1]

    # [test-drift fix] prod serializa el execute_after con `.isoformat()`
    # (separador 'T'): `fresh_execute_dt.isoformat()` → "2024-05-04T04:30:00+00:00".
    # El VALOR es idéntico al esperado; solo cambia el separador ' ' → 'T'.
    assert "2024-05-04T04:30:00+00:00" in str(params)

@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_timezone_alignment_asia(mock_query, mock_write):
    import cron_tasks
    # [test-drift fix] Ver test_timezone_alignment_rd: el INSERT del chunk ahora
    # es un UPSERT atómico [P0-1] vía execute_sql_query con RETURNING; el
    # execute_after viaja en sus params. Dict truthy con inserted=True para no
    # caer en la rama "skip-active".
    mock_query.return_value = {"id": "chunk-1", "status": "pending", "inserted": True}

    snapshot = {
        "form_data": {
            "_plan_start_date": "2024-05-01T16:00:00+00:00", # Midnight local time Manila (UTC+8) = 16:00 UTC (day before)
            "tzOffset": -480 # Manila is 480 minutes ahead of UTC (JS offset is negative)
        }
    }

    with patch("cron_tasks.datetime") as mock_dt:
        mock_dt.now.return_value = datetime.datetime(2024, 5, 2, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_dt.combine = datetime.datetime.combine
        mock_dt.min = datetime.datetime.min

        cron_tasks._enqueue_plan_chunk(
            user_id="u1",
            meal_plan_id="m1",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot=snapshot
        )

    assert mock_query.called
    args, kwargs = mock_query.call_args  # último call == el UPSERT del chunk
    #
    # [reconvertido · P1-CHUNK-ANCHOR-LOCAL-DATE · 2026-08-21] Este test esperaba
    # `2024-05-03T16:30`, y para llegar ahí partía de «start_dt_midnight_utc = 2024-05-01 00:00
    # UTC»: la fecha del ancla leída EN UTC. Pero el propio comentario del snapshot dice que
    # `2024-05-01T16:00Z` es «midnight local time Manila (day before)» — o sea que el plan empieza
    # el **2 de mayo** en Manila, no el 1. Tomar la fecha UTC de un instante que ya es del día
    # siguiente en local es el off-by-one que este P-fix cerró, aquí en su versión ESPEJO: al este
    # de UTC el bloque se programaba un día ANTES del tramo que cubre; al oeste (República
    # Dominicana, offset 240) un día DESPUÉS — medido en la cola viva, con el día 11 programado
    # para el 22 en vez del 21.
    #
    # Aritmética correcta: ancla local = 2024-05-02 (Manila) → +3 días = 2024-05-05 local →
    # 00:30 Manila del 5 = 2024-05-04 16:30 UTC.
    params = args[1]
    # [test-drift fix] prod serializa con `.isoformat()` (separador 'T').
    assert "2024-05-04T16:30:00+00:00" in str(params), (
        "el ancla volvió a leerse en UTC en vez de en la hora local del usuario"
    )
