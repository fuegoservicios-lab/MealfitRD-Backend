"""
Shared fixtures for backend E2E tests.

Provides `seeded_user_profile`: inserts a synthetic user into auth.users,
user_profiles, and user_inventory so that E2E chunk tests never skip
due to missing DB data.
"""
# [P0-5] Eagerly resolve real `langgraph` BEFORE any test module loads. Several
# test files do `sys.modules.setdefault('langgraph', MagicMock())` to support
# environments without the package, but `setdefault` only checks if the key is
# already in `sys.modules` — it cannot tell whether the existing entry is the
# real package or a previously-installed MagicMock. When the alphabetically-first
# test (e.g. test_chunk_learning_appears_in_prompt.py) ran its `setdefault`
# without `langgraph` yet in `sys.modules`, it installed a MagicMock, and every
# subsequent `from langgraph.checkpoint.memory import MemorySaver` (transitively
# pulled in by `cron_tasks` → `agent`) raised
# `ModuleNotFoundError: 'langgraph' is not a package`. Importing it here primes
# `sys.modules` with the real package so all later `setdefault`s become no-ops.
# Only stub if the real submodule path is genuinely unimportable (CI without
# the dependency installed).
try:
    import langgraph  # noqa: F401
    import langgraph.graph  # noqa: F401
    import langgraph.graph.message  # noqa: F401
    import langgraph.checkpoint.memory  # noqa: F401
except Exception:
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault("langgraph", MagicMock())
    sys.modules.setdefault("langgraph.graph", MagicMock())
    sys.modules.setdefault("langgraph.graph.message", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint.memory", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint.postgres", MagicMock())

# [P0-5 · P0-DEEPSEEK-MIGRATION 2026-06-12] Same eager-import for
# `langchain_openai` (base client of the DeepSeek provider, see
# `llm_provider.py`). If a test file installs a partial stub first, a later
# import of the real surface (e.g. via cron_tasks → ai_helpers →
# llm_provider) raises ImportError. Importing the real package here primes
# sys.modules with the full surface, and subsequent stub `setdefault` /
# `_install_stub` calls become no-ops because the key is already populated.
try:
    import langchain_openai  # noqa: F401
    from langchain_openai import (  # noqa: F401
        ChatOpenAI,
        OpenAIEmbeddings,
    )
except Exception:
    import sys
    from unittest.mock import MagicMock
    if "langchain_openai" not in sys.modules:
        _stub = MagicMock()
        _stub.ChatOpenAI = object
        _stub.OpenAIEmbeddings = object
        sys.modules["langchain_openai"] = _stub

import uuid
import json
import pytest
from datetime import datetime, timezone

from db_core import execute_sql_write, execute_sql_query, connection_pool


# ---------------------------------------------------------------------------
# Ensure the connection pool is open for test sessions
# ---------------------------------------------------------------------------
if connection_pool and not getattr(connection_pool, '_opened', False):
    connection_pool.open()


# ---------------------------------------------------------------------------
# Marker registration (also declared in pytest.ini at backend root)
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: End-to-end tests requiring a live database")


# ---------------------------------------------------------------------------
# Core fixture: synthetic user + plan_id, with full teardown
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_user_profile():
    """Create a throwaway user in auth.users → user_profiles → user_inventory.

    Yields (user_id, plan_id).  Teardown removes all traces in FK-safe order.
    """
    user_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    email = f"e2e-test-{user_id[:8]}@test.local"

    # --- Setup -----------------------------------------------------------
    # Pre-clean any leftover data from a previously interrupted test with
    # the same UUID (astronomically unlikely, but handles partial teardowns).
    for tbl in ("plan_chunk_queue", "meal_plans", "user_inventory"):
        execute_sql_write(f"DELETE FROM {tbl} WHERE user_id = %s", (user_id,))
    execute_sql_write("DELETE FROM user_profiles WHERE id = %s", (user_id,))
    execute_sql_write("DELETE FROM auth.users WHERE id = %s", (user_id,))

    # 1. auth.users  (minimal row that satisfies the FK from user_profiles)
    execute_sql_write(
        "INSERT INTO auth.users (id, aud, role, email) "
        "VALUES (%s, 'authenticated', 'authenticated', %s) "
        "ON CONFLICT (id) DO NOTHING",
        (user_id, email),
    )

    # 2. user_profiles
    health_profile = {
        "age": 30,
        "weight": 75,
        "height": 170,
        "gender": "M",
        "goal": "maintain",
        "activityLevel": "moderate",
        "dietType": "Omnívora",
        "allergies": [],
        "budget": "medium",
        "householdSize": 1,
        "tz_offset_minutes": -240,
    }
    execute_sql_write(
        "INSERT INTO user_profiles (id, health_profile) VALUES (%s, %s::jsonb) "
        "ON CONFLICT (id) DO UPDATE SET health_profile = EXCLUDED.health_profile",
        (user_id, json.dumps(health_profile, ensure_ascii=False)),
    )

    # 3. user_inventory  (enough staples so pantry checks pass)
    pantry_items = [
        ("Pechuga de Pollo", 1000, "g"),
        ("Arroz", 2000, "g"),
        ("Habichuelas", 500, "g"),
        ("Res", 800, "g"),
        ("Pescado", 600, "g"),
        ("Huevos", 12, "unidad"),
        ("Aceite de Oliva", 500, "ml"),
        ("Cebolla", 500, "g"),
        ("Ajo", 100, "g"),
        ("Tomate", 400, "g"),
    ]
    for name, qty, unit in pantry_items:
        execute_sql_write(
            "INSERT INTO user_inventory (user_id, ingredient_name, quantity, unit) "
            "VALUES (%s, %s, %s, %s)",
            (user_id, name, qty, unit),
        )

    yield user_id, plan_id

    # --- Teardown (FK-safe order) ----------------------------------------
    execute_sql_write(
        "DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,)
    )
    execute_sql_write(
        "DELETE FROM meal_plans WHERE id = %s", (plan_id,)
    )
    # Also clean any plans created with this user_id outside the fixture plan_id
    execute_sql_write(
        "DELETE FROM plan_chunk_queue WHERE user_id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM meal_plans WHERE user_id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM user_inventory WHERE user_id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM user_profiles WHERE id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM auth.users WHERE id = %s", (user_id,)
    )
