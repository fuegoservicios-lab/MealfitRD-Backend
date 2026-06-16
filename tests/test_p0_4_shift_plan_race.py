"""[P0-4] Race condition entre worker merge y `/shift-plan`.

Antes del fix:
    Worker tiene dos transacciones (T1 mergea days; T2 escribe shopping_list +
    learning_metrics + status='completed'). Entre T1 y T2 corren ~5-15s de
    cálculo de shopping list (3 retries con backoff). Si `/shift-plan` corría
    en esa ventana, modificaba `days` (shift + renumber) y `grocery_start_date`,
    pero T2 sobrescribía plan_data completo con el dict en memoria del worker
    (que tenía days PRE-shift). Resultado: usuario veía números de día stale,
    grocery_start_date stale, y posibles días duplicados en el dashboard.

Después del fix:
    1. Worker T1 + T2 + /shift-plan adquieren `acquire_meal_plan_advisory_lock(
       cursor, plan_id, purpose='general')` antes de tocar plan_data. Esto los
       serializa: no pueden ejecutarse simultáneamente.
    2. T2 RE-LEE plan_data con FOR UPDATE y aplica SOLO los campos incrementales
       (P0_4_T2_INCREMENTAL_KEYS: learning + shopping + quality), preservando
       cualquier modificación que /shift-plan haya commiteado entre T1 y T2.

NOTA: El advisory lock POR SÍ SOLO no cierra la ventana T1→T2 (porque
pg_advisory_xact_lock auto-releases al cerrar transacción), por eso el re-read
en T2 es lo que verdaderamente preserva los cambios de /shift-plan.
"""
from unittest.mock import patch, MagicMock

import pytest

from cron_tasks import P0_4_T2_INCREMENTAL_KEYS, P0_1_DEFERRED_LEARNING_KEYS


def test_t2_incremental_keys_includes_learning_shopping_quality():
    """Contrato: los campos que SÓLO T2 persiste deben estar listados aquí.
    Si se añade un nuevo campo (e.g. nueva métrica de calidad o nueva variante
    de shopping_list) sin agregarlo, T2 lo perderá durante el re-read porque
    el merge solo aplica los keys listados.
    """
    expected = {
        # Learning (deferidos de T1 por P0-1).
        '_last_chunk_learning',
        '_recent_chunk_lessons',
        '_critical_lessons_permanent',
        '_lifetime_lessons_history',
        '_lifetime_lessons_summary',
        '_chunk_learning_stub_count',
        # Shopping list (4 variantes: weekly, biweekly, monthly, active).
        'aggregated_shopping_list',
        'aggregated_shopping_list_weekly',
        'aggregated_shopping_list_biweekly',
        'aggregated_shopping_list_monthly',
        # Quality flags (calculadas post-merge).
        'quality_warning',
        'quality_degraded_ratio',
        # [P0-5] Pantry quantity violation annotation (advisory/hybrid mode):
        # set in T1 by the worker but must survive T2's fresh-read overlay so
        # the UI/admin sees the chunk-level annotation.
        '_pantry_quantity_violations',
        # [P2-CHUNK-6] Degradación a nivel plan-result (fallback / review-failed
        # entregada igualmente). El worker las setea post-merge cuando
        # run_plan_pipeline degradó; deben sobrevivir T2 para que el cron de
        # auto-resolve de I5 vea el plan como NO-limpio.
        '_is_fallback',
        '_review_failed_but_delivered',
        # [P2-10 · 2026-06-16] Flags user-facing de degradación de calidad (band-score/
        # panel/clinical-layer/resolution gates) que el worker ahora propaga en semana 2+.
        '_quality_degraded',
        '_quality_degraded_reason',
        '_quality_degraded_severity',
        '_quality_degraded_attempts',
        '_quality_degraded_band_score',
        '_quality_degraded_panel_detail',
        '_quality_degraded_clinical_detail',
        '_quality_degraded_resolution_pct',
    }
    assert set(P0_4_T2_INCREMENTAL_KEYS) == expected, (
        "Si añadiste un nuevo campo que el worker calcula entre T1 y T2 sin "
        "actualizarlo aquí, T2 lo perderá durante el merge incremental. "
        "Alternativamente, si hay un campo aquí que YA se persiste en T1, "
        "removelo (sólo deberían listarse los campos exclusivos a T2)."
    )


def test_t2_incremental_keys_includes_p0_1_deferred_learning():
    """Los campos diferidos por P0-1 (learning) deben estar entre los
    incrementales de T2. Sin esto, el aprendizaje quedaría escrito por T1
    no-aplicado en T2, rompiendo la atomicidad con plan_chunk_queue.learning_metrics.
    """
    for k in P0_1_DEFERRED_LEARNING_KEYS:
        assert k in P0_4_T2_INCREMENTAL_KEYS, (
            f"El campo diferido por P0-1 {k!r} no está en la lista de "
            f"incrementales de T2 — quedaría sin persistir."
        )


def test_t2_merge_preserves_shift_plan_modifications_to_days():
    """Simula el escenario crítico de la race:
    1. Worker T1 mergea days [1..10] al plan.
    2. /shift-plan corre entre T1 y T2 → shifta a days [1..9] con renumeración.
    3. Worker T2 corre, debe RE-LEER fresh plan_data (post /shift-plan) y
       aplicar SOLO sus campos incrementales (learning, shopping, quality)
       SIN sobrescribir days.

    Verifica que el merge incremental preserva los days de /shift-plan.
    """
    # Estado en DB DESPUÉS de T1 + /shift-plan: days fueron modificados, y
    # /shift-plan también actualizó grocery_start_date.
    fresh_plan_data_in_db = {
        'days': [{'day': i, 'meals': [{'name': f'Comida-{i}'}]} for i in range(1, 10)],
        'grocery_start_date': '2026-05-03',  # /shift-plan actualizó esto
        'generation_status': 'partial',
        '_merged_chunk_ids': ['chunk-1'],
        'last_technique': 'Horneado',
    }

    # Estado en memoria del worker (con days PRE-shift + incrementales nuevos).
    worker_full_plan_data = {
        'days': [{'day': i, 'meals': []} for i in range(1, 11)],  # 10 days, stale
        'grocery_start_date': '2026-05-02',  # Stale
        '_last_chunk_learning': {'chunk': 1, 'rejection_violations': 0},
        '_lifetime_lessons_history': [{'chunk': 1, 'timestamp': '...'}],
        'aggregated_shopping_list': [{'name': 'Pollo', 'qty': '500g'}],
        'aggregated_shopping_list_weekly': [{'name': 'Pollo', 'qty': '500g'}],
        'aggregated_shopping_list_biweekly': [],
        'aggregated_shopping_list_monthly': [],
        'quality_warning': False,
        'quality_degraded_ratio': 0.0,
    }

    # Replica del merge que hace T2 (líneas 14380-14403 cron_tasks.py):
    merged = dict(fresh_plan_data_in_db)
    for k in P0_4_T2_INCREMENTAL_KEYS:
        if k in worker_full_plan_data:
            merged[k] = worker_full_plan_data[k]

    # Días de /shift-plan preservados (NO sobrescritos con worker stale).
    assert len(merged['days']) == 9
    assert merged['days'][0]['meals'][0]['name'] == 'Comida-1', (
        "El merge incremental sobrescribió days con la versión stale del worker. "
        "T2 debe preservar lo que /shift-plan persistió entre T1 y T2."
    )

    # grocery_start_date preservada (NO sobrescrita).
    assert merged['grocery_start_date'] == '2026-05-03'

    # Otros campos persistidos por T1 (NO incrementales) preservados.
    assert merged['_merged_chunk_ids'] == ['chunk-1']
    assert merged['last_technique'] == 'Horneado'

    # Campos incrementales del worker (learning + shopping + quality) aplicados.
    assert merged['_last_chunk_learning']['chunk'] == 1
    assert merged['_lifetime_lessons_history'][0]['chunk'] == 1
    assert merged['aggregated_shopping_list'][0]['name'] == 'Pollo'
    assert merged['quality_warning'] is False


def test_t2_merge_handles_fresh_plan_with_no_overlap():
    """Si fresh plan_data no tiene los keys incrementales (caso normal: T1
    persistió todo lo demás, los incrementales no estaban porque P0-1 los
    difirió), el merge los añade limpiamente."""
    fresh = {
        'days': [{'day': 1}],
        '_merged_chunk_ids': ['c1'],
    }
    worker = {
        '_last_chunk_learning': {'chunk': 1},
        'aggregated_shopping_list': [],
    }
    merged = dict(fresh)
    for k in P0_4_T2_INCREMENTAL_KEYS:
        if k in worker:
            merged[k] = worker[k]
    assert merged['days'] == [{'day': 1}]
    assert merged['_last_chunk_learning'] == {'chunk': 1}
    assert merged['aggregated_shopping_list'] == []


def test_t2_merge_skips_keys_worker_didnt_compute():
    """Si el worker NO calculó algún incremental (e.g. shopping list falló
    todos sus retries y el chunk se re-encoló — pero hipotéticamente llegamos
    a T2), no debe escribirse un valor None ni vacío. Solo aplicar los keys
    que el worker SÍ tiene en memoria."""
    fresh = {
        'days': [{'day': 1}],
        'aggregated_shopping_list': [{'name': 'fresh-list-from-prior'}],  # valor stale OK
    }
    worker = {
        '_last_chunk_learning': {'chunk': 1},
        # No aggregated_shopping_list en worker (e.g. shopping calc falló)
    }
    merged = dict(fresh)
    for k in P0_4_T2_INCREMENTAL_KEYS:
        if k in worker:
            merged[k] = worker[k]
    # Worker NO sobrescribió aggregated_shopping_list porque no estaba en memoria.
    assert merged['aggregated_shopping_list'] == [{'name': 'fresh-list-from-prior'}]
    # Pero sí aplicó learning.
    assert merged['_last_chunk_learning'] == {'chunk': 1}


def test_advisory_lock_helper_is_imported_in_worker_paths():
    """Smoke test: verifica que el helper canónico `acquire_meal_plan_advisory_lock`
    está disponible en cron_tasks (worker T1+T2) y routers/plans (shift-plan).
    Si por refactor se renombrara o moviera, los locks dejarían de adquirirse y
    la race volvería silenciosamente.
    """
    from db_plans import acquire_meal_plan_advisory_lock
    assert callable(acquire_meal_plan_advisory_lock)
    # El helper debe aceptar `purpose='general'` sin warning.
    # (Verificación de string pertenecer al set conocido en db_plans._MEAL_PLAN_LOCK_PURPOSES.)
    from db_plans import _MEAL_PLAN_LOCK_PURPOSES
    assert "general" in _MEAL_PLAN_LOCK_PURPOSES


def test_advisory_lock_uses_correct_purpose_in_t1():
    """Regression guard: el código de T1 (worker merge en cron_tasks) debe usar
    purpose='general'. Si pasara a 'catchup' u otro, no se serializaría con
    /shift-plan que también usa 'general'.
    """
    import cron_tasks
    import inspect
    src = inspect.getsource(cron_tasks)
    # Buscar la primera ocurrencia de _p04_acquire_lock con purpose= en T1
    # (cerca del comentario "[P0-4] Advisory lock por meal_plan ANTES del FOR UPDATE").
    assert '_p04_acquire_lock(cursor, meal_plan_id, purpose="general")' in src, (
        "T1 del worker no está usando purpose='general' en el advisory lock. "
        "Sin esto, no se serializa con /shift-plan."
    )


def test_advisory_lock_uses_correct_purpose_in_shift_plan():
    """Regression guard: /shift-plan debe usar purpose='general' (no solo
    'catchup' que es para una rama interna específica)."""
    from routers import plans as plans_router
    import inspect
    src = inspect.getsource(plans_router)
    assert '_p04_acquire_lock(cursor, plan_id, purpose="general")' in src, (
        "/shift-plan no está usando purpose='general'. Sin esto, no se "
        "serializa con el merge del worker."
    )


def _collect_subscript_assign_string_keys(source: str, target_name: str) -> set[str]:
    """[P0-4/STATIC] AST scan helper: encuentra todas las keys string asignadas
    via `target_name['KEY'] = ...` (Subscript Assign con clave Constant str).

    Cubre también `target_name.update({'K': v, ...})` donde la dict literal
    tiene claves Constant str. NO cubre claves dinámicas (variables, expresiones)
    — esas se loguean implícitamente por el caller si el set retornado no
    matchea expectativas.

    Args:
        source: código fuente Python como string.
        target_name: nombre de la variable a auditar (ej. 'full_plan_data').

    Returns:
        Set de string keys mutadas en `target_name`. Vacío si target no aparece.
    """
    import ast

    keys: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Pattern 1: target['K'] = ...
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == target_name
                    and isinstance(tgt.slice, ast.Constant)
                    and isinstance(tgt.slice.value, str)
                ):
                    keys.add(tgt.slice.value)
        # Pattern 2: target.update({'K': v, ...})
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == target_name
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            for k in node.args[0].keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    return keys


def test_all_full_plan_data_writes_are_in_incremental_keys():
    """[P0-4/STATIC] Forward guard: cualquier key que el worker mute en
    `full_plan_data` entre T1 y T2 DEBE estar en P0_4_T2_INCREMENTAL_KEYS,
    sino el merge incremental de T2 la pierde silenciosamente.

    El test existente `test_t2_incremental_keys_includes_learning_shopping_quality`
    asegura que el SET sea exactamente el esperado, pero NO verifica que TODAS
    las mutaciones del código real estén cubiertas. Si un futuro contribuidor
    añade `full_plan_data['nueva_metrica'] = ...` sin añadir `nueva_metrica` al
    constant, ese campo se cae en la próxima ejecución de T2 sin alertas.

    Este scan AST captura ese caso al test time.
    """
    import inspect
    import cron_tasks

    src = inspect.getsource(cron_tasks)
    written_keys = _collect_subscript_assign_string_keys(src, "full_plan_data")

    # Sanity: los 6 keys conocidos (shopping × 4, quality × 2) deben aparecer.
    expected_minimum = {
        "aggregated_shopping_list",
        "aggregated_shopping_list_weekly",
        "aggregated_shopping_list_biweekly",
        "aggregated_shopping_list_monthly",
        "quality_warning",
        "quality_degraded_ratio",
    }
    assert expected_minimum.issubset(written_keys), (
        f"AST scan no encontró todas las escrituras conocidas a full_plan_data. "
        f"Encontradas: {written_keys}. Faltantes: {expected_minimum - written_keys}. "
        f"¿Cambió la sintaxis del worker o se movió el código?"
    )

    # Invariante: todas las keys escritas deben estar en INCREMENTAL_KEYS.
    leaked = written_keys - set(P0_4_T2_INCREMENTAL_KEYS)
    assert not leaked, (
        f"[P0-4 BUG] El worker escribe a full_plan_data{[k for k in sorted(leaked)]} "
        f"pero esas claves NO están en P0_4_T2_INCREMENTAL_KEYS. T2 las descartará "
        f"silenciosamente al hacer el merge incremental. "
        f"Solución: añadir cada clave faltante a P0_4_T2_INCREMENTAL_KEYS en "
        f"cron_tasks.py:123 (y al expected set de "
        f"test_t2_incremental_keys_includes_learning_shopping_quality)."
    )


def test_no_shift_plan_keys_overlap_with_incremental_keys():
    """[P0-4/STATIC] Inverse guard: ningún campo que `/shift-plan` (HTTP en
    routers/plans.py o BG en cron_tasks._background_shift_plan_for_user) muta
    debe estar en P0_4_T2_INCREMENTAL_KEYS.

    Si un campo está en ambos lados:
      - /shift-plan lo escribe entre T1 y T2 del worker.
      - T2 lo sobreescribe con la versión stale del worker (pre-shift).
      - El cambio del shift se pierde — la race que P0-4 quiso cerrar vuelve.

    Hoy las shift-keys son `days`, `grocery_start_date`, `generation_status`,
    `pending_user_action`, `_plan_modified_at` — NINGUNA está en INCREMENTAL_KEYS.
    Este guard previene que un futuro refactor introduzca el solapamiento.
    """
    import inspect
    import cron_tasks
    from routers import plans as plans_router

    incremental = set(P0_4_T2_INCREMENTAL_KEYS)

    # Path BG (cron diario): muta `shifted_data` dentro de _background_shift_plan_for_user.
    bg_src = inspect.getsource(cron_tasks._background_shift_plan_for_user)
    bg_writes = _collect_subscript_assign_string_keys(bg_src, "shifted_data")
    bg_overlap = bg_writes & incremental
    assert not bg_overlap, (
        f"[P0-4 BUG] _background_shift_plan_for_user muta {sorted(bg_overlap)} "
        f"y esas keys ALSO están en P0_4_T2_INCREMENTAL_KEYS. T2 sobreescribirá "
        f"los cambios del BG shift con la versión stale del worker."
    )

    # Path HTTP (api_shift_plan): muta `shifted_data` también.
    http_src = inspect.getsource(plans_router)
    http_writes = _collect_subscript_assign_string_keys(http_src, "shifted_data")
    http_overlap = http_writes & incremental
    assert not http_overlap, (
        f"[P0-4 BUG] /shift-plan HTTP muta {sorted(http_overlap)} y esas keys "
        f"ALSO están en P0_4_T2_INCREMENTAL_KEYS. T2 sobreescribirá los "
        f"cambios del shift con la versión stale del worker."
    )


def test_advisory_lock_uses_correct_purpose_in_background_shift():
    """[P0-2/BG-LOCK] Regression guard: el path background `_background_shift_plan_for_user`
    (cron diario que dispara shift-plan para usuarios inactivos) debe adquirir el
    mismo advisory lock `purpose='general'` que el endpoint HTTP `/shift-plan`.

    Sin esto, el cron BG puede mutar plan_data sin coordinarse con el worker T2,
    rompiendo el invariante "todo escritor de plan_data toma purpose='general'"
    que el resto del sistema asume. El FOR UPDATE actual mitiga vía row-lock,
    pero el advisory lock es la garantía explícita y sobrevive a refactors.
    """
    import inspect
    from cron_tasks import _background_shift_plan_for_user

    src = inspect.getsource(_background_shift_plan_for_user)
    assert 'acquire_meal_plan_advisory_lock' in src, (
        "_background_shift_plan_for_user no importa acquire_meal_plan_advisory_lock. "
        "Debe usar el helper canónico para serializar contra el worker."
    )
    assert 'purpose="general"' in src, (
        "_background_shift_plan_for_user no está usando purpose='general' al "
        "adquirir el advisory lock. Sin eso, no se serializa con el worker T2 "
        "(que también usa 'general') ni con /shift-plan HTTP."
    )
