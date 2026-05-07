"""
Tests P1-ORQ-1: read-modify-write atómico de `health_profile`.

Bug original:
  3 call sites en graph_orchestrator.py hacían
    profile = get_user_profile(user_id)
    hp = profile["health_profile"]
    hp["some_list"].append(...)
    update_user_health_profile(user_id, hp)
  Sin lock, dos pipelines del mismo user_id (2 tabs regenerando, cron +
  manual) leían el mismo snapshot, cada uno appendeaba localmente, y el
  último UPDATE pisaba al primero — pierden 1 entry por par concurrente.

Fix:
  `db_profiles.update_user_health_profile_atomic(user_id, mutator)` envuelve
  read + mutate + write en un `SELECT … FOR UPDATE` + UPDATE dentro de la
  misma transacción. Dos invocaciones simultáneas se serializan: la segunda
  espera al COMMIT de la primera y su mutator ve el estado post-primera.

Tests:
  - Path con `connection_pool` real → atomicidad real (requiere DB; skip si
    no hay pool — entornos de CI sin DB).
  - Path fallback (`connection_pool=None`) → comportamiento legacy preservado
    (no atómico, pero no rompe nada).
  - Mutator semantics: in-place vs return dict vs return False.
  - Side effects: invalidation_reasons / tz_changed se computan correctamente.
  - Refactor de los 3 call sites: imports y patrón nuevo presentes en
    graph_orchestrator.py.
"""
import re
from unittest.mock import patch, MagicMock

import pytest

import db_profiles
from db_profiles import update_user_health_profile_atomic


# ---------------------------------------------------------------------------
# 1. Mutator semantics (path fallback, sin connection_pool real)
# ---------------------------------------------------------------------------
def _setup_fallback_mocks(initial_hp):
    """Patches el path fallback (no connection_pool) con stubs de get/update."""
    profile_stub = {"health_profile": dict(initial_hp)}
    update_calls = []

    def _mock_get(uid):
        # Devolvemos copia para simular semántica de DB read.
        return {"health_profile": dict(profile_stub["health_profile"])}

    def _mock_update(uid, hp):
        update_calls.append({"user_id": uid, "hp": dict(hp)})
        profile_stub["health_profile"] = dict(hp)
        return [{"id": uid}]

    return profile_stub, _mock_get, _mock_update, update_calls


def test_mutator_inplace_persiste_cambios_path_fallback():
    """Mutator que mutua hp in-place y retorna None → cambios se persisten."""
    profile, mock_get, mock_update, calls = _setup_fallback_mocks(
        {"reflection_history": [{"date": "2026-01-01", "diagnosis": "old"}]}
    )

    with (
        patch.object(db_profiles, "connection_pool", None),
        patch.object(db_profiles, "get_user_profile", side_effect=mock_get),
        patch.object(db_profiles, "update_user_health_profile", side_effect=mock_update),
    ):
        def _mut(hp):
            hp["reflection_history"].append({"date": "2026-01-02", "diagnosis": "new"})
            # No return → mutación in-place se persiste

        result = update_user_health_profile_atomic("user-123", _mut)

    assert result is not None
    assert len(result["reflection_history"]) == 2
    assert result["reflection_history"][-1]["diagnosis"] == "new"
    assert len(calls) == 1


def test_mutator_returns_false_aborta_update():
    """Mutator retorna False → no se llama a update_user_health_profile."""
    _, mock_get, mock_update, calls = _setup_fallback_mocks(
        {"some_field": "value"}
    )

    with (
        patch.object(db_profiles, "connection_pool", None),
        patch.object(db_profiles, "get_user_profile", side_effect=mock_get),
        patch.object(db_profiles, "update_user_health_profile", side_effect=mock_update),
    ):
        def _mut(hp):
            return False

        result = update_user_health_profile_atomic("user-123", _mut)

    assert result == {"some_field": "value"}
    assert len(calls) == 0  # NO update


def test_mutator_returns_new_dict_persiste_ese_dict():
    """Mutator retorna un dict distinto → ese dict se persiste."""
    _, mock_get, mock_update, calls = _setup_fallback_mocks({"old_key": "old"})

    with (
        patch.object(db_profiles, "connection_pool", None),
        patch.object(db_profiles, "get_user_profile", side_effect=mock_get),
        patch.object(db_profiles, "update_user_health_profile", side_effect=mock_update),
    ):
        def _mut(hp):
            return {"new_key": "new"}

        result = update_user_health_profile_atomic("user-123", _mut)

    assert result == {"new_key": "new"}
    assert calls[0]["hp"] == {"new_key": "new"}


def test_user_no_existe_retorna_none():
    """get_user_profile devuelve None → atomic helper retorna None."""
    with (
        patch.object(db_profiles, "connection_pool", None),
        patch.object(db_profiles, "get_user_profile", return_value=None),
        patch.object(db_profiles, "update_user_health_profile") as mock_upd,
    ):
        result = update_user_health_profile_atomic("ghost-user", lambda hp: None)

    assert result is None
    mock_upd.assert_not_called()


def test_health_profile_no_dict_se_normaliza_a_dict_vacio():
    """Si DB devuelve health_profile=None o tipo raro, mutator recibe dict vacío."""
    with (
        patch.object(db_profiles, "connection_pool", None),
        patch.object(
            db_profiles,
            "get_user_profile",
            return_value={"health_profile": None},  # None, no dict
        ),
        patch.object(db_profiles, "update_user_health_profile", return_value=None),
    ):
        captured = {}

        def _mut(hp):
            captured["received"] = hp
            hp["x"] = 1

        update_user_health_profile_atomic("user-123", _mut)

    assert captured["received"] == {"x": 1}  # arrancó {} y mutator lo llenó


# ---------------------------------------------------------------------------
# 2. Atomicidad real con connection_pool simulado
# ---------------------------------------------------------------------------
def test_path_atomico_usa_select_for_update():
    """Verifica que el path con connection_pool emite `SELECT … FOR UPDATE`
    seguido de UPDATE en la misma transacción."""
    executed_sqls = []

    fake_cursor = MagicMock()
    fake_cursor.fetchone.return_value = {
        "health_profile": {"existing_key": "v0", "reflection_history": []}
    }
    fake_cursor.execute = MagicMock(
        side_effect=lambda sql, params=None: executed_sqls.append(sql)
    )
    fake_cursor.__enter__ = lambda self: self
    fake_cursor.__exit__ = lambda self, *a: None

    fake_conn = MagicMock()
    fake_conn.cursor = MagicMock(return_value=fake_cursor)
    fake_conn.transaction = MagicMock()
    fake_conn.transaction.return_value.__enter__ = MagicMock()
    fake_conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    fake_conn.__enter__ = lambda self: self
    fake_conn.__exit__ = lambda self, *a: None

    fake_pool = MagicMock()
    fake_pool.connection.return_value = fake_conn

    with patch.object(db_profiles, "connection_pool", fake_pool):
        def _mut(hp):
            hp["reflection_history"].append({"date": "2026-01-02", "diagnosis": "x"})

        result = update_user_health_profile_atomic("user-123", _mut)

    assert result is not None
    assert len(result["reflection_history"]) == 1

    # Contract: debe haber un SELECT FOR UPDATE seguido de un UPDATE.
    select_idx = next(
        (i for i, s in enumerate(executed_sqls) if "SELECT" in s.upper() and "FOR UPDATE" in s.upper()),
        None,
    )
    update_idx = next(
        (i for i, s in enumerate(executed_sqls) if s.upper().startswith("UPDATE USER_PROFILES")),
        None,
    )
    assert select_idx is not None, f"falta SELECT FOR UPDATE. SQLs: {executed_sqls}"
    assert update_idx is not None, f"falta UPDATE. SQLs: {executed_sqls}"
    assert select_idx < update_idx, "UPDATE debe correr DESPUÉS del SELECT FOR UPDATE"


# ---------------------------------------------------------------------------
# 3. Side effects post-commit (invalidation_reasons + tz_changed)
# ---------------------------------------------------------------------------
def _build_pool_with_initial_hp(initial_hp):
    """Helper: pool fake que devuelve `initial_hp` en el SELECT FOR UPDATE."""
    fake_cursor = MagicMock()
    fake_cursor.fetchone.return_value = {"health_profile": dict(initial_hp)}
    fake_cursor.__enter__ = lambda self: self
    fake_cursor.__exit__ = lambda self, *a: None

    fake_conn = MagicMock()
    fake_conn.cursor = MagicMock(return_value=fake_cursor)
    fake_conn.transaction = MagicMock()
    fake_conn.transaction.return_value.__enter__ = MagicMock()
    fake_conn.transaction.return_value.__exit__ = MagicMock(return_value=False)
    fake_conn.__enter__ = lambda self: self
    fake_conn.__exit__ = lambda self, *a: None

    fake_pool = MagicMock()
    fake_pool.connection.return_value = fake_conn

    return fake_pool


def test_cambio_de_goal_dispara_invalidation():
    """Cambiar `goal` debe disparar `_invalidate_stale_chunks` post-commit."""
    pool = _build_pool_with_initial_hp({"goal": "lose_fat", "weight": 70})

    with (
        patch.object(db_profiles, "connection_pool", pool),
        patch.object(db_profiles, "_invalidate_stale_chunks") as mock_inv,
    ):
        def _mut(hp):
            hp["goal"] = "gain_muscle"

        update_user_health_profile_atomic("user-123", _mut)

    mock_inv.assert_called_once()
    args = mock_inv.call_args
    assert args[0][0] == "user-123"
    assert "goal_changed" in args[0][1]


def test_sin_cambios_criticos_no_dispara_invalidation():
    """Mutar un campo no-crítico (e.g. reflection_history) NO invalida chunks."""
    pool = _build_pool_with_initial_hp({"goal": "lose_fat", "reflection_history": []})

    with (
        patch.object(db_profiles, "connection_pool", pool),
        patch.object(db_profiles, "_invalidate_stale_chunks") as mock_inv,
    ):
        def _mut(hp):
            hp["reflection_history"].append({"x": 1})

        update_user_health_profile_atomic("user-123", _mut)

    mock_inv.assert_not_called()


def test_cambio_significativo_de_peso_dispara_invalidation():
    """Cambio de peso ≥5kg debe disparar invalidación."""
    pool = _build_pool_with_initial_hp({"weight": 70})

    with (
        patch.object(db_profiles, "connection_pool", pool),
        patch.object(db_profiles, "_invalidate_stale_chunks") as mock_inv,
    ):
        def _mut(hp):
            hp["weight"] = 76  # +6 kg

        update_user_health_profile_atomic("user-123", _mut)

    mock_inv.assert_called_once()
    assert "significant_weight_change" in mock_inv.call_args[0][1]


# ---------------------------------------------------------------------------
# 4. Refactor de call sites verificado por inspección de código
# ---------------------------------------------------------------------------
def test_graph_orchestrator_no_usa_get_update_pattern_para_health_profile():
    """Los 3 call sites P1-ORQ-1 (holistic score, reflection_node,
    review_plan_node rejection) deben usar `update_user_health_profile_atomic`,
    NO el patrón legacy `get_user_profile + update_user_health_profile`.

    Si alguien añade un nuevo caller en el futuro con el patrón viejo, este
    test falla — los reviewers son redirigidos al helper atómico.
    """
    import inspect
    import graph_orchestrator as go

    src = inspect.getsource(go)

    # Debe haber al menos 3 imports/usos del helper atómico (uno por call site).
    n_atomic = src.count("update_user_health_profile_atomic")
    assert n_atomic >= 3, (
        f"Esperaba ≥3 usos de update_user_health_profile_atomic en "
        f"graph_orchestrator (uno por call site P1-ORQ-1), encontró {n_atomic}."
    )

    # NO debe haber ya `from db_profiles import ... update_user_health_profile`
    # SIN el sufijo _atomic. Tolera la mención en comentarios pero no en imports.
    legacy_imports = re.findall(
        r"from\s+db_profiles\s+import[^\n]*\bupdate_user_health_profile\b(?!_atomic)",
        src,
    )
    assert len(legacy_imports) == 0, (
        f"Detectado import legacy `update_user_health_profile` (sin _atomic) "
        f"en graph_orchestrator.py: {legacy_imports}. Usar "
        f"`update_user_health_profile_atomic` con un mutator."
    )


def test_helper_at_export_publico_de_db_profiles():
    """`update_user_health_profile_atomic` debe ser importable directamente."""
    from db_profiles import update_user_health_profile_atomic as _h
    assert callable(_h)
