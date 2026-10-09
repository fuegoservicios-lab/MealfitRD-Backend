"""Free iOS allowance, independent of web purchases. Never changes billing records."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import os

GENERATION = 100
COACH = 1000
_scope = ContextVar("usage_scope", default="web")


def enabled():
    return os.environ.get("MEALFIT_IOS_FREE_ENABLED", "false").lower() == "true"


def scope():
    return _scope.get()


def is_free():
    return enabled() and scope() == "ios_free"


@contextmanager
def using(value):
    token = _scope.set("ios_free" if value == "ios_free" else "web")
    try:
        yield
    finally:
        _scope.reset(token)


def inherit_context(fn):
    context = copy_context()
    @wraps(fn)
    def run(*args, **kwargs):
        return context.run(fn, *args, **kwargs)
    return run


def isolated_worker(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        with using("web"):
            return fn(*args, **kwargs)
    return run


def restore_snapshot(snapshot):
    # Only the server writes this top-level key; form fields never decide entitlements.
    _scope.set("ios_free" if snapshot.get("_usage_scope") == "ios_free" else "web")


class FreeIOSMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, request, receive, send):
        headers = dict(request.get("headers", []))
        value = "ios_free" if enabled() and headers.get(b"origin", b"").lower() == b"capacitor://localhost" else "web"
        with using(value):
            await self.app(request, receive, send)


def project_profile(profile):
    if not profile or not is_free():
        return profile
    from regalos_cuenta import regalos_vigentes, extra_de
    gifts = regalos_vigentes(profile.get("id"), usage_scope="ios_free")
    return {**profile, "plan_tier": "gratis", "plan_tier_pagado": "gratis", "cortesia": None,
            "access_model": "ios_free", "creditos_extra": {
                "generacion": extra_de(gifts, "generacion"), "coach": extra_de(gifts, "coach")}}


def allowance(user_id):
    from db_profiles import get_monthly_api_usage
    from regalos_cuenta import regalos_vigentes, extra_de
    with using("ios_free"):
        gifts = regalos_vigentes(user_id, usage_scope="ios_free")
        def meter(kind, base):
            bonus = extra_de(gifts, kind)
            used = get_monthly_api_usage(user_id, kind="coach" if kind == "coach" else "generation")
            return {"usados": used, "plan": base, "regalo": bonus, "tope": base + bonus,
                    "restantes": max(0, base + bonus - used)}
        return {"creditos": meter("generacion", GENERATION), "coach": meter("coach", COACH)}
