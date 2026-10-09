import asyncio
import threading
import uuid

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import auth
import admin_cuentas as ac
import db_core
import db_profiles
import ios_free as free
import llm_provider
import regalos_cuenta as rc
import routers.admin as admin

UID = '33333333-3333-3333-3333-333333333333'
ADMIN = '11111111-1111-1111-1111-111111111111'


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv('MEALFIT_IOS_FREE_ENABLED', 'true')


def test_paid_and_courtesy_entitlements_do_not_reach_ios(monkeypatch):
    scopes = []
    def gifts(uid, usage_scope='web'):
        scopes.append(usage_scope)
        return ([{'kind': 'plan', 'plan': 'ultra'}, {'kind': 'creditos_coach', 'amount': 900}]
                if usage_scope == 'web' else [{'kind': 'creditos_coach', 'amount': 1000}])
    monkeypatch.setattr(rc, 'regalos_vigentes', gifts)
    stored = {'id': UID, 'plan_tier': 'plus'}
    with free.using('ios_free'):
        profile = rc.superponer(dict(stored))
        assert profile['plan_tier'] == profile['plan_tier_pagado'] == 'gratis'
        assert profile['cortesia'] is None
        assert profile['creditos_extra'] == {'generacion': 0, 'coach': 1000}
        monkeypatch.setattr(auth, 'get_user_profile', lambda uid: profile)
        monkeypatch.setattr(auth, 'get_monthly_api_usage', lambda *a, **k: 1999)
        assert auth.verify_coach_quota(UID) == UID
        assert auth.coach_quota_snapshot(UID)['limit'] == 2000
    web = rc.superponer(dict(stored))
    assert web['plan_tier'] == 'ultra' and web['plan_tier_pagado'] == 'plus'
    assert stored['plan_tier'] == 'plus'
    assert scopes == ['ios_free', 'web']


def test_ios_exhaustion_has_no_payment_prompt(monkeypatch):
    monkeypatch.setattr(auth, 'get_user_profile', lambda uid: {'plan_tier': 'admin'})
    monkeypatch.setattr(auth, 'get_monthly_api_usage', lambda *a, **k: 1000)
    with free.using('ios_free'):
        for gate in (auth.verify_coach_quota, auth.verify_api_quota):
            with pytest.raises(HTTPException) as error:
                gate(UID)
            assert error.value.status_code == 402
            assert 'gratuita' in error.value.detail
            assert 'Mejora' not in error.value.detail
    assert auth.verify_api_quota(UID) == UID


def test_paid_model_cache_cannot_override_free_access(monkeypatch):
    llm_provider._TIER_CACHE[UID] = ('ultra', float('inf'))
    with free.using('ios_free'):
        assert llm_provider.get_user_tier(UID) == 'gratis'
        assert db_profiles.get_user_plan_tier(UID) == 'gratis'
    assert llm_provider.get_user_tier(UID) == 'ultra'
    llm_provider._TIER_CACHE.pop(UID, None)


def test_usage_is_recorded_and_counted_in_its_scope(monkeypatch):
    monkeypatch.setattr(db_core, 'connection_pool', object())
    writes, queries = [], []
    monkeypatch.setattr(db_profiles, 'execute_sql_write', lambda q, p: writes.append((q, p)))
    def query(q, p, **kwargs):
        queries.append((q, p))
        return {'total': 8 if p[-1] == 'ios_free' else 31}
    monkeypatch.setattr(db_profiles, 'execute_sql_query', query)
    with free.using('ios_free'):
        db_profiles.log_api_usage(UID, 'llm_chat')
        assert db_profiles.get_monthly_api_usage(UID, 'coach') == 8
    assert db_profiles.get_monthly_api_usage(UID, 'coach') == 31
    assert writes[0][1] == (UID, 'llm_chat', 'ios_free')
    assert all('usage_scope = %s' in q for q, _ in queries)


def test_plan_origin_is_server_owned_for_future_refills():
    from db_plans import _build_meal_plan_insert_sql
    data = {'user_id': UID, 'name': 'Ejemplo', 'usage_scope': 'ios_free'}
    sql, values = _build_meal_plan_insert_sql(data, skip_plan_data_finalize=True)
    assert 'usage_scope' in sql and values[-1] == 'web'
    with free.using('ios_free'):
        sql, values = _build_meal_plan_insert_sql({**data, 'usage_scope': 'web'}, skip_plan_data_finalize=True)
        assert values[-1] == 'ios_free'


def test_requests_threads_and_queued_jobs_keep_scope_without_leaking():
    observed = []
    async def downstream(request, receive, send):
        await asyncio.sleep(0)
        observed.append(free.scope())
    async def exercise():
        middleware = free.FreeIOSMiddleware(downstream)
        await asyncio.gather(middleware({'headers': [(b'origin', b'capacitor://localhost')]}, None, None),
                             middleware({'headers': [(b'origin', b'https://app.bioboros.com')]}, None, None))
    asyncio.run(exercise())
    assert sorted(observed) == ['ios_free', 'web']
    with free.using('ios_free'):
        thread = threading.Thread(target=free.inherit_context(lambda: observed.append(free.scope())))
        thread.start()
        thread.join()
    @free.isolated_worker
    def worker():
        free.restore_snapshot({'_usage_scope': 'ios_free'})
        assert free.is_free()
        free.restore_snapshot({})
        assert free.scope() == 'web'
    worker()
    assert observed[-1] == 'ios_free' and free.scope() == 'web'


def test_recharge_is_free_atomic_repeatable_and_retry_safe(monkeypatch):
    monkeypatch.setattr(ac, '_perfil', lambda uid: {'plan_tier': 'plus'})
    monkeypatch.setattr(ac, '_exigir_activo', lambda: None)
    events, rows = [], {}
    monkeypatch.setattr(ac, '_anotar', lambda *a: events.append('audit'))
    def transaction(queries):
        assert events[-1] == 'audit'
        assert len(queries) == 2
        for sql, params in queries:
            assert 'ON CONFLICT (id) DO NOTHING' in sql and "'ios_free'" in sql
            rows.setdefault(params[0], params)
        events.append('write')
    monkeypatch.setattr(ac, 'execute_sql_transaction', transaction)
    request = uuid.uuid4()
    first = ac.recargar_ios_gratis(ADMIN, UID, request, 'Recarga gratuita de continuidad')
    second = ac.recargar_ios_gratis(ADMIN, UID, request, 'Recarga gratuita de continuidad')
    assert first['grant_ids'] == second['grant_ids'] and len(rows) == 2
    ac.recargar_ios_gratis(ADMIN, UID, uuid.uuid4(), 'Nueva recarga gratuita')
    assert len(rows) == 4
    assert sorted(p[3] for p in rows.values()) == [100, 100, 1000, 1000]


def test_recharge_requires_admin_and_anti_csrf(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[admin.require_admin] = lambda: ADMIN
    with TestClient(app) as client:
        response = client.post(f'/api/admin/cuentas/{UID}/ios-gratis/recargar',
                               json={'request_id': str(uuid.uuid4()), 'motivo': 'Recarga gratuita'})
        assert response.status_code == 403
    def forbidden():
        raise HTTPException(status_code=404)
    app.dependency_overrides[admin.require_admin] = forbidden
    with TestClient(app) as client:
        response = client.post(f'/api/admin/cuentas/{UID}/ios-gratis/recargar', headers={'X-Admin-Accion': '1'},
                               json={'request_id': str(uuid.uuid4()), 'motivo': 'Recarga gratuita'})
        assert response.status_code == 404
