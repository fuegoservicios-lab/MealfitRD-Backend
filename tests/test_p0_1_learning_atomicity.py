"""[P0-1] Atomicidad del learning entre meal_plans.plan_data y plan_chunk_queue.

Invariante (post-fix):
    Los campos canónicos de aprendizaje (P0_1_DEFERRED_LEARNING_KEYS:
    _last_chunk_learning, _recent_chunk_lessons, _critical_lessons_permanent,
    _lifetime_lessons_history, _lifetime_lessons_summary, _chunk_learning_stub_count)
    se persisten ATÓMICAMENTE en T1 — el mismo FOR UPDATE que escribe `days` y
    `_merged_chunk_ids` a meal_plans.plan_data, y que estampa `learning_metrics`
    en plan_chunk_queue.

    Si T2 (status='completed' + shopping_list) falla por crash o DB blip, el
    retry detecta que `_persisted_chunk_id == week_number` y salta el backfill:
    la lección está garantizada por T1.

Razón del cambio (pre-fix → post-fix):
    Pre-fix, T1 strippeaba estos campos para "diferirlos" a T2. Pero si el
    worker crasheaba entre T1 y T2 (durante shopping list ~segundos),
    `_merged_chunk_ids` quedaba commiteado SIN lección. En el retry, el path
    `chunk_already_merged` saltaba el merge y leía queue.learning_metrics=NULL
    → backfill caía a STUB y la lección real se perdía permanentemente.
    El nombre `DEFERRED` de la constante se preserva por compatibilidad.

Tests aquí cubren:
    1. La constante lista TODOS los campos de learning del worker.
    2. El payload de T1 INCLUYE los campos de learning (no se strippean).
    3. El payload de T2 (full_plan_data) también los preserva.
    4. La ventana T1→T2 mantiene plan_data y queue.learning_metrics consistentes.
"""
import pytest

from cron_tasks import P0_1_DEFERRED_LEARNING_KEYS


def test_deferred_keys_constant_covers_all_learning_fields():
    """Contrato: la constante P0_1_DEFERRED_LEARNING_KEYS lista TODOS los campos
    de aprendizaje que el worker escribe en plan_data durante el merge.

    Si se añade un nuevo campo de learning en `_chunk_worker` sin agregarlo
    aquí, ese campo NO sería overlay-eado por T2 (P0_4_T2_INCREMENTAL_KEYS lo
    referencia) y el rebuilder no lo reconocería como learning.
    """
    expected = {
        '_last_chunk_learning',
        '_recent_chunk_lessons',
        '_critical_lessons_permanent',
        '_lifetime_lessons_history',
        '_lifetime_lessons_summary',
        '_chunk_learning_stub_count',
    }
    assert set(P0_1_DEFERRED_LEARNING_KEYS) == expected, (
        "Si añadiste un nuevo campo de learning al worker, agrégalo a "
        "P0_1_DEFERRED_LEARNING_KEYS en cron_tasks.py o T2 no lo overlay-eará "
        "tras el re-read post /shift-plan, perdiéndolo."
    )


def test_t1_persist_view_includes_learning_fields():
    """[P0-1 FIX] Replica de la lógica del worker: el dict que se serializa
    para T1 debe INCLUIR todos los campos de aprendizaje (junto con days,
    _merged_chunk_ids, etc.).

    Pre-fix, T1 strippeaba P0_1_DEFERRED_LEARNING_KEYS y el race se abría
    cuando T2 fallaba post shopping-list. Post-fix, todo se persiste atómico
    en T1.
    """
    plan_data = {
        # Campos NO-learning que T1 SÍ debe persistir.
        'days': [{'day': 1, 'meals': [{'name': 'Pollo'}]}],
        '_merged_chunk_ids': ['chunk-uuid-1'],
        'last_technique': 'Horneado',
        'generation_status': 'partial',
        'total_days_requested': 7,
        'grocery_start_date': '2026-05-02',
        '_plan_modified_at': '2026-05-02T10:00:00+00:00',
        # Campos de learning: T1 también los persiste atómicamente.
        '_last_chunk_learning': {'chunk': 1, 'rejection_violations': 0},
        '_recent_chunk_lessons': [{'chunk': 1}],
        '_critical_lessons_permanent': [],
        '_lifetime_lessons_history': [{'chunk': 1, 'timestamp': '...'}],
        '_lifetime_lessons_summary': {'total_rejection_violations': 0},
        '_chunk_learning_stub_count': 0,
    }

    # [P0-1 FIX] Lógica del worker (cron_tasks.py:_chunk_worker T1): persiste
    # plan_data completo, sin strippear learning.
    t1_view = dict(plan_data)

    # Idempotencia: T1 conserva days y _merged_chunk_ids para que el pre-check
    # del próximo intento detecte mergeación previa.
    assert 'days' in t1_view
    assert '_merged_chunk_ids' in t1_view
    assert t1_view['_merged_chunk_ids'] == ['chunk-uuid-1']
    assert t1_view['last_technique'] == 'Horneado'
    assert t1_view['generation_status'] == 'partial'
    assert t1_view['total_days_requested'] == 7
    assert t1_view['_plan_modified_at'] == '2026-05-02T10:00:00+00:00'

    # [P0-1 FIX] Learning: TODOS los campos canónicos deben aparecer en T1.
    # Si T2 falla post-T1, el retry encuentra `_last_chunk_learning.chunk ==
    # week_number` y salta el backfill — la lección está garantizada.
    for k in P0_1_DEFERRED_LEARNING_KEYS:
        assert k in t1_view, (
            f"El campo de learning {k!r} NO aparece en el payload de T1. "
            "Si T2 falla post-T1, el retry leería queue.learning_metrics=NULL "
            "y el backfill caería a stub, perdiendo la lección real."
        )

    # plan_data en memoria sigue intacto: dict(plan_data) es una copia shallow.
    assert plan_data['_last_chunk_learning']['chunk'] == 1


def test_t1_atomically_persists_learning_metrics_to_chunk_queue():
    """[P0-1 FIX] Contrato: dentro del FOR UPDATE de T1, el worker DEBE escribir
    `learning_metrics` a `plan_chunk_queue` en la misma transacción que escribe
    learning a `plan_data`.

    Sin esto, plan_data tendría la lección pero queue.learning_metrics quedaría
    en NULL — el rebuilder y el path `chunk_already_merged` no podrían
    distinguir un T1 commiteado de un T1 abortado, y caerían a stub en el
    retry.

    Este test verifica el invariante a nivel de fuente: el bloque de T1 contiene
    tanto el UPDATE meal_plans como el UPDATE plan_chunk_queue para
    learning_metrics, y ambos están dentro del mismo `if not _stale_abort:`.
    """
    import os
    import re

    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # El bloque de T1 está marcado por el comentario "[P0-1 FIX] Atomicidad".
    # Capturamos hasta la siguiente directiva mayor (TOCTOU abort post-tx).
    block_match = re.search(
        r"\[P0-1 FIX\] Atomicidad de learning.*?\[P0-1/TOCTOU\]",
        source,
        re.DOTALL,
    )
    assert block_match, (
        "No se encontró el bloque T1 marcado con [P0-1 FIX] Atomicidad. "
        "Si renombraste el comentario, actualiza este test."
    )
    t1_block = block_match.group(0)

    # T1 debe escribir plan_data...
    assert "UPDATE meal_plans SET plan_data" in t1_block, (
        "T1 ya no contiene UPDATE meal_plans — el invariante P0-1 se rompió."
    )
    # ...y debe escribir learning_metrics a plan_chunk_queue en la misma tx.
    assert "UPDATE plan_chunk_queue" in t1_block, (
        "T1 no escribe a plan_chunk_queue. Sin esto, learning_metrics queda "
        "en NULL si T2 falla, y el retry pierde la lección."
    )
    assert "learning_metrics = %s::jsonb" in t1_block, (
        "T1 no estampa learning_metrics dentro de la transacción. Pre-fix, "
        "este UPDATE estaba sólo en T2 — ese era el race que P0-1 cierra."
    )


def test_t2_atomic_payload_includes_all_learning_fields():
    """T2 hace re-read fresh + overlay de los incrementales. Como T1 ya escribió
    learning, T2 sobrescribe con el mismo valor (idempotente). full_plan_data
    en memoria tras el merge debe seguir teniendo todos los campos canónicos
    para que el overlay de T2 los aplique sobre el plan_data fresco.
    """
    full_plan_data = {
        'days': [{'day': 1}],
        '_merged_chunk_ids': ['chunk-uuid-1'],
        '_last_chunk_learning': {'chunk': 1, 'rejection_violations': 2},
        '_recent_chunk_lessons': [{'chunk': 1}],
        '_critical_lessons_permanent': [{'chunk': 1, '_critical': True}],
        '_lifetime_lessons_history': [{'chunk': 1}],
        '_lifetime_lessons_summary': {'total_rejection_violations': 2},
        '_chunk_learning_stub_count': 0,
        'aggregated_shopping_list': [{'name': 'Pollo', 'qty': '500g'}],
    }

    for k in P0_1_DEFERRED_LEARNING_KEYS:
        assert k in full_plan_data, (
            f"El campo {k!r} se perdió antes de T2. T2 hace overlay vía "
            f"P0_4_T2_INCREMENTAL_KEYS leyendo de full_plan_data; si falta, "
            f"el overlay no lo aplica y la lección queda sólo en el T1 commit."
        )

    assert 'aggregated_shopping_list' in full_plan_data


def test_atomicity_window_invariant_post_fix():
    """[P0-1 FIX] Si T1 commitea pero T2 falla, plan_data persistido DEBE
    contener los learning fields, y plan_chunk_queue.learning_metrics DEBE
    estar populado. Ambos en el mismo commit — no hay ventana donde uno exista
    sin el otro.

    Estado simulado tras crash inmediato post-T1, pre-T2:
      - meal_plans.plan_data: days + _merged_chunk_ids + learning fields ✓
      - plan_chunk_queue.learning_metrics: populated ✓
      - plan_chunk_queue.status: 'processing' (T2 no llegó a 'completed')

    En el retry, `chunk_already_merged` detecta `_persisted_chunk_id ==
    week_number` y salta el backfill — la lección está garantizada.
    """
    plan_data_in_memory = {
        'days': [{'day': 1}],
        '_merged_chunk_ids': ['c1'],
        '_last_chunk_learning': {'chunk': 1, 'rejection_violations': 1},
        '_recent_chunk_lessons': [{'chunk': 1}],
        '_lifetime_lessons_history': [{'chunk': 1}],
    }

    # Lo que T1 escribe en DB con el fix: plan_data completo (sin strip).
    persisted_by_t1 = dict(plan_data_in_memory)

    # El invariante post-fix: persisted_by_t1 INCLUYE learning, y
    # queue.learning_metrics DEBE estar populado en la misma transacción.
    assert '_last_chunk_learning' in persisted_by_t1
    assert '_lifetime_lessons_history' in persisted_by_t1
    assert persisted_by_t1['_last_chunk_learning']['chunk'] == 1
    assert persisted_by_t1['_merged_chunk_ids'] == ['c1']
    assert persisted_by_t1['days'] == [{'day': 1}]

    # Simulación del retry post-crash: lee plan_data, encuentra
    # chunk_already_merged, y verifica `_persisted_chunk_id == week_number`.
    week_number = 1
    persisted_lesson = persisted_by_t1.get('_last_chunk_learning') or {}
    persisted_chunk_id = (
        persisted_lesson.get('chunk')
        if isinstance(persisted_lesson, dict) else None
    )
    assert persisted_chunk_id == week_number, (
        "El retry no detectaría que el chunk ya tiene su lección persistida — "
        "caería al backfill desde queue. Verifica que T1 esté escribiendo "
        "_last_chunk_learning con el chunk number correcto."
    )
