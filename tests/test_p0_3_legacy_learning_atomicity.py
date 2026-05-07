"""[P0.3] Atomicidad de _last_chunk_learning en paths legacy.

El worker tiene un T2 atómico que sincroniza plan_data._last_chunk_learning con
plan_chunk_queue.learning_metrics + status='completed' en una transacción
(cron_tasks.py:16799). Pero hay 4 paths LEGACY que también escriben
_last_chunk_learning a meal_plans.plan_data fuera de ese T2:

  1. routers/plans.py: seed inicial de chunk 1 (sync) — context='seed_chunk1_sync'.
  2. routers/plans.py: seed inicial via SSE — context='seed_chunk1_sse'.
  3. cron_tasks.py: P0-3 auto-recovery desde plan_chunk_queue — context='rebuild_from_queue'.
  4. cron_tasks.py: P0-4 last-resort synthesis desde plan_data.days — context='synthesis_from_days'.

Antes de P0.3, cada uno duplicaba el patrón SQL `jsonb_set` y dos de cuatro
NO sellaban `_plan_modified_at` (CAS broken). Tras P0.3, todos pasan por
`persist_legacy_learning_to_plan_data`, que:

  - Sella `_plan_modified_at` con NOW() en el mismo UPDATE atómico.
  - Valida que el campo persistido esté en P0_1_DEFERRED_LEARNING_KEYS.
  - Rechaza contexts no canónicos (defense-in-depth contra nuevos paths).

Ejecutar:
    cd backend && python -m pytest tests/test_p0_3_legacy_learning_atomicity.py -v
"""
import sys
import os
import re
import types
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs necesarios para importar cron_tasks aislado.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_kw: None,
    execute_sql_write=lambda *_a, **_kw: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_kw: None,
    get_inventory_activity_since=lambda *_a, **_kw: [],
    get_raw_user_inventory=lambda *_a, **_kw: [],
    get_user_inventory_net=lambda *_a, **_kw: [],
    release_chunk_reservations=lambda *_a, **_kw: None,
    reserve_plan_ingredients=lambda *_a, **_kw: 0,
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_recent_plans=lambda *_a, **_kw: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_kw: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_kw: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_kw: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_kw: {})

apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg

import cron_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# 1. La constante de contextos canónicos cubre exactamente los 4 paths legacy.
# ---------------------------------------------------------------------------
def test_legacy_contexts_cover_exactly_four_paths():
    expected = {
        "seed_chunk1_sync",
        "seed_chunk1_sse",
        "rebuild_from_queue",
        "synthesis_from_days",
    }
    assert set(cron_tasks.P0_3_LEGACY_LEARNING_CONTEXTS) == expected, (
        "Si añadiste un nuevo path legacy de persistencia de learning, "
        "regístralo en P0_3_LEGACY_LEARNING_CONTEXTS o usa el worker T2 atómico."
    )


# ---------------------------------------------------------------------------
# 2. El helper rechaza contexts inválidos.
# ---------------------------------------------------------------------------
def test_helper_rejects_unknown_context():
    captured = []
    with patch.object(cron_tasks, "execute_sql_write", side_effect=lambda *a, **kw: captured.append(a)):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", {"chunk": 1}, context="not_a_canonical_context",
            user_id="u1",  # [P0-9] requerido aunque el context check aborte antes
        )
    assert ok is False
    assert captured == []  # no SQL emitido


# ---------------------------------------------------------------------------
# 3. El UPDATE incluye sello CAS de `_plan_modified_at`.
# ---------------------------------------------------------------------------
def test_helper_seals_plan_modified_at_in_update_sql():
    captured = {}

    def _capture(query, params=None, returning=False):
        captured["query"] = query
        captured["params"] = params
        # [P0-9] el helper ahora exige RETURNING id; simulamos match exitoso.
        return [{"id": "plan-1"}] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_capture):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1",
            {"chunk": 1, "rejection_violations": 0},
            context="rebuild_from_queue",
            user_id="u1",
        )

    assert ok is True
    sql = captured["query"]
    # Contrato: el UPDATE debe escribir _last_chunk_learning Y sellar _plan_modified_at.
    assert "_last_chunk_learning" in sql
    assert "_plan_modified_at" in sql
    assert "to_jsonb(NOW()::text)" in sql
    # [P0-9] El WHERE debe filtrar por user_id, no solo id.
    assert "user_id" in sql, "P0-9: el WHERE debe filtrar por user_id"
    assert "RETURNING id" in sql, "P0-9: debe usar RETURNING para detectar mismatch"


# ---------------------------------------------------------------------------
# 4. Si se pasa recent_chunk_lessons, ambos campos van en el mismo UPDATE.
# ---------------------------------------------------------------------------
def test_helper_persists_recent_lessons_atomically():
    captured = {}

    def _capture(query, params=None, returning=False):
        captured["query"] = query
        captured["params"] = params
        return [{"id": "plan-1"}] if returning else None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_capture):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1",
            {"chunk": 1},
            recent_chunk_lessons=[{"chunk": 1}],
            context="seed_chunk1_sync",
            user_id="u1",
        )

    assert ok is True
    sql = captured["query"]
    assert "_last_chunk_learning" in sql
    assert "_recent_chunk_lessons" in sql
    assert "_plan_modified_at" in sql
    # [P0-9] params: 2 jsonb payloads + meal_plan_id + user_id = 4.
    assert len(captured["params"]) == 4


# ---------------------------------------------------------------------------
# 5. El helper devuelve False y no crashea si execute_sql_write falla.
# ---------------------------------------------------------------------------
def test_helper_returns_false_on_db_error():
    def _raise(*_a, **_kw):
        raise Exception("DB blip")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=_raise):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", {"chunk": 1}, context="rebuild_from_queue",
            user_id="u1",
        )
    assert ok is False


# ---------------------------------------------------------------------------
# 6. Args inválidos no llegan al SQL.
# ---------------------------------------------------------------------------
def test_helper_rejects_invalid_args():
    captured = []
    with patch.object(cron_tasks, "execute_sql_write", side_effect=lambda *a, **kw: captured.append(a)):
        # meal_plan_id vacío.
        assert cron_tasks.persist_legacy_learning_to_plan_data(
            "", {"chunk": 1}, context="rebuild_from_queue",
            user_id="u1",
        ) is False
        # last_chunk_learning no es dict.
        assert cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", "not_a_dict", context="rebuild_from_queue",  # type: ignore
            user_id="u1",
        ) is False
        # [P0-9] user_id vacío rechaza antes del SQL.
        assert cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-1", {"chunk": 1}, context="rebuild_from_queue",
            user_id="",
        ) is False
    assert captured == []


# ---------------------------------------------------------------------------
# 7. Convención: ningún call-site legacy emite SQL inline `jsonb_set` con
#    `_last_chunk_learning` fuera del helper. Si alguien añade un nuevo path
#    duplicando el patrón, este test lo bloquea.
# ---------------------------------------------------------------------------
def test_no_inline_jsonb_set_for_last_chunk_learning_outside_helper():
    """Escanea cron_tasks.py y routers/plans.py buscando el patrón
    `jsonb_set(...'{_last_chunk_learning}'...)`. La única ocurrencia legítima
    es dentro de `persist_legacy_learning_to_plan_data` (cron_tasks.py).
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cron_path = os.path.join(repo_root, "backend", "cron_tasks.py")
    routers_plans_path = os.path.join(repo_root, "backend", "routers", "plans.py")

    pattern = re.compile(r"jsonb_set\([^)]*_last_chunk_learning")

    def _count_inline_sites(filepath, allow_inside_function=None):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # Si se especifica allow_inside_function, removemos el cuerpo de esa
        # función antes de buscar (es donde el patrón ES legítimo).
        if allow_inside_function:
            # Heurística simple: encontrar `def <fn>(` y eliminar hasta la
            # próxima línea que empiece con `def ` o `class ` al margen 0.
            fn_re = re.compile(
                rf"^def {re.escape(allow_inside_function)}\b.*?(?=^def |^class |\Z)",
                re.DOTALL | re.MULTILINE,
            )
            text = fn_re.sub("", text)
        return len(pattern.findall(text))

    cron_inline = _count_inline_sites(
        cron_path,
        allow_inside_function="persist_legacy_learning_to_plan_data",
    )
    routers_inline = _count_inline_sites(routers_plans_path)

    assert cron_inline == 0, (
        f"cron_tasks.py contiene {cron_inline} usos inline de "
        f"`jsonb_set(...'_last_chunk_learning'...)` fuera de "
        f"persist_legacy_learning_to_plan_data. Migrarlos al helper."
    )
    assert routers_inline == 0, (
        f"routers/plans.py contiene {routers_inline} usos inline de "
        f"`jsonb_set(...'_last_chunk_learning'...)`. Usar "
        f"`persist_legacy_learning_to_plan_data` desde cron_tasks."
    )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
