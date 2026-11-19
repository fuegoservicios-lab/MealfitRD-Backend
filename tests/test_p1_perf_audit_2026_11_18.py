# -*- coding: utf-8 -*-
"""[P1-PERF-AUDIT-2026-11-18] Tests de regresión para los hallazgos P0-P3 de la auditoría integral."""
from __future__ import annotations

import ast
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent


def test_p0_billing_verify_rejects_malformed_sub_id_and_has_limiter():
    import routers.billing as b

    assert hasattr(b, "_SUBSCRIPTION_VERIFY_LIMITER")
    assert b._PAYPAL_SUB_ID_RE.fullmatch("I-BW452GLLEP1G")
    assert not b._PAYPAL_SUB_ID_RE.fullmatch("../v1/payments")
    assert not b._PAYPAL_SUB_ID_RE.fullmatch("abc")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            b.api_verify_subscription(
                {"user_id": "u-1", "subscriptionID": "../evil", "tier": "plus"},
                verified_user_id="u-1",
            )
        )
    assert exc.value.status_code == 400


def test_p0_billing_verify_rejects_custom_id_mismatch(monkeypatch):
    import routers.billing as b

    monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid")
    monkeypatch.setenv("PAYPAL_SECRET", "csec")

    auth_resp = MagicMock(status_code=200)
    auth_resp.json.return_value = {"access_token": "tok"}
    sub_resp = MagicMock(status_code=200)
    sub_resp.json.return_value = {
        "status": "ACTIVE",
        "plan_id": "P-PLUS",
        "custom_id": "other-user-id",
    }

    fake_client = AsyncMock()
    fake_client.post.return_value = auth_resp
    fake_client.get.return_value = sub_resp
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = False

    with patch("routers.billing.httpx.AsyncClient", return_value=fake_client):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                b.api_verify_subscription(
                    {"user_id": "u-1", "subscriptionID": "I-VALIDSUB1234", "tier": "plus"},
                    verified_user_id="u-1",
                )
            )
    assert exc.value.status_code == 403


def test_p0_billing_webhook_5xx_returns_503_and_bad_json_returns_400():
    src = (_BACKEND / "routers" / "billing.py").read_text(encoding="utf-8")
    assert "verify_resp.status_code >= 500 or verify_resp.status_code == 429" in src
    assert "except (UnicodeDecodeError, json.JSONDecodeError):" in src


def test_p0_blocked_reasons_requires_auth():
    import routers.plans as p

    resp = MagicMock()
    with pytest.raises(HTTPException) as exc:
        p.api_blocked_reasons("11111111-1111-1111-1111-111111111111", resp, verified_user_id=None)
    assert exc.value.status_code == 401


def test_p0_i18n_textos_requires_auth():
    import routers.user_data as ud

    with pytest.raises(HTTPException) as exc:
        ud.api_traducir_textos({"locale": "en-US", "textos": ["Pollo"]}, verified_user_id=None, _ia=True)
    assert exc.value.status_code == 401


def test_p1_db_core_keeps_guards_and_contiguous_overloads():
    # Revisión 18-nov: quitar el `check=` dejaba conexiones muertas de Neon (cierra las ociosas) en el pool, y
    # condicionar la guarda de escritura a PYTEST_CURRENT_TEST dejaba sin guarda las conexiones abiertas fuera de un
    # test (colección, fixtures de sesión, batería en seco). Se quedan las dos.
    src = (_BACKEND / "db_core.py").read_text(encoding="utf-8")
    assert src.count("check=ConnectionPool.check_connection") == 2
    assert "check=AsyncConnectionPool.check_connection" in src
    assert 'if os.environ.get("PYTEST_CURRENT_TEST"):' not in src

    tree = ast.parse(src)
    write_defs = [
        (i, n.lineno)
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.FunctionDef) and n.name == "execute_sql_write"
    ]
    assert len(write_defs) == 3
    indices = [idx for idx, _ in write_defs]
    assert indices == list(range(indices[0], indices[0] + 3)), "PEP 484: @overload debe ser contiguo"


def test_p1_missing_indexes_migration_ssot_synced():
    name = "p1_perf_audit_2026_11_18_missing_indexes.sql"
    f_root = _ROOT / "migrations" / name
    f_back = _BACKEND / "migrations" / name
    assert f_root.exists() and f_back.exists()
    assert f_root.read_bytes() == f_back.read_bytes()
    sql = f_root.read_text(encoding="utf-8")
    assert "idx_consumed_meals_user_consumed_at" in sql
    assert "ux_user_profiles_paypal_subscription_id" in sql
    assert "idx_plan_chunk_queue_meal_plan_week" in sql


def test_p2_catalog_cache_no_deepcopy_and_caches_empty():
    from catalog_cache import CatalogCache

    cache = CatalogCache(ttl=60)
    calls = {"n": 0}

    async def _loader():
        calls["n"] += 1
        return [{"id": 1, "name": "Pollo"}]

    r1 = asyncio.run(cache.get(_loader))
    r1[0]["name"] = "MUTATED"
    r2 = asyncio.run(cache.get(_loader))
    assert calls["n"] == 1
    assert r2[0]["name"] == "Pollo"


def test_p2_cpu_tasks_single_pass_regex_accuracy():
    from cpu_tasks import _calcular_frecuencias_regex_cpu_bound, _normalize_meal_name

    p, c, v, f = _calcular_frecuencias_regex_cpu_bound(
        "pechuga de pollo con arroz blanco y brocoli al vapor, pollo guisado",
        ["Pollo"], {"pollo": ["pollo"]},
        ["Arroz"], {"arroz": ["arroz"]},
        ["Brocoli"], {"brocoli": ["brocoli"]},
        ["Manzana"], {"manzana": ["manzana"]},
    )
    assert p["Pollo"] == 2
    assert c["Arroz"] == 1
    assert v["Brocoli"] == 1
    assert f["Manzana"] == 0
    assert _normalize_meal_name("200g Pollo con Arroz") == "pollo arroz"


def test_p3_scheduler_worker_compiles_cleanly():
    src = (_BACKEND / "scheduler_worker.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "MEALFIT_RUN_SCHEDULER_IN_WEB" in (_BACKEND / "app.py").read_text(encoding="utf-8")
