"""[P1-DEDUP-RECENT-PLAN · 2026-06-25] Idempotencia de generación: si el cliente pierde la conexión
SSE DESPUÉS de que el plan ya se generó (caso real: se cae el internet, el usuario reintenta 2-3 min
después), el guard de "pipeline activo" (KV='generating') NO lo cubre (la KV ya está en 'complete') →
se generaba un 2º plan DUPLICADO (2× costo LLM). El endpoint /analyze/stream ahora, si ya existe un
plan USABLE creado hace < N min, devuelve 409 `plan_recently_created` con su plan_id (el frontend lo
adopta) en vez de regenerar. Knob `MEALFIT_RECENT_PLAN_DEDUP_MINUTES` (default 5, 0 = off).
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_BACKEND)


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def test_dedup_guard_present_y_ordenado_antes_del_upsert():
    src = _read(_BACKEND, "routers", "plans.py")
    assert 'MEALFIT_RECENT_PLAN_DEDUP_MINUTES' in src, "falta el knob del dedup"
    assert '"code": "plan_recently_created"' in src, "falta el 409 de plan reciente"
    assert 'get_latest_meal_plan_with_id' in src, "el dedup debe usar el helper de plan reciente"
    # ORDEN crítico: el dedup va DESPUÉS del check de pipeline-activo y ANTES del upsert a
    # 'generating' → si retorna 409 NO deja una KV 'generating' stale (que bloquearía 15 min).
    pos_inflight = src.find('"code": "pipeline_already_running"')
    pos_dedup = src.find('"code": "plan_recently_created"')
    pos_upsert = src.find('upsert_pending_pipeline(_deep_search_user_id, status="generating")')
    assert pos_inflight >= 0 and pos_dedup >= 0 and pos_upsert >= 0
    assert pos_inflight < pos_dedup < pos_upsert, "el dedup debe ir entre el check de pipeline activo y el upsert"


def test_dedup_solo_cuenta_planes_usables():
    # Debe verificar que el plan reciente tiene días (no un plan vacío/fallido) antes de deduplicar.
    src = _read(_BACKEND, "routers", "plans.py")
    # ancla del chequeo de usabilidad (days) en el bloque del dedup.
    seg = src[src.find("P1-DEDUP-RECENT-PLAN"): src.find("Reservar slot")]
    assert '.get("days")' in seg or "'days'" in seg, "el dedup debe exigir que el plan tenga días"


def test_frontend_maneja_plan_recently_created():
    src = _read(_ROOT, "frontend", "src", "pages", "Plan.jsx")
    assert "plan_recently_created" in src, "Plan.jsx debe manejar el 409 plan_recently_created"
    assert "err.planId" in src, "Plan.jsx debe extraer plan_id del 409"
    # no debe reintentar ese 409 (no se resuelve con backoff).
    assert "err.code === 'plan_recently_created'" in src
