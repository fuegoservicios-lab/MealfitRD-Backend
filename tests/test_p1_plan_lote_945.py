# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-945 · 2026-09-30] Cada limitador cuenta lo suyo en Redis.

La clave era rl:{max}:{periodo}:{uid}: los 23 limitadores 30/60 compartían UN cupo por usuario entre endpoints
distintos (y los 23 de 10/60, los 17 de 20/60) — las metas del contador, el medidor del coach y la proyección de la
lista se gastaban el cupo entre sí. En memoria nunca pasó (cada instancia tiene su `_hits`), así que el límite real
dependía de si Redis está configurado. e7 lo pisó en el panel de admin (90/60 → 96/60).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import rate_limiter as rl  # noqa: E402


class _Pipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def zremrangebyscore(self, k, _a, hasta):
        self.ops.append(lambda: self.r.purga(k, hasta))
        return self

    def zcard(self, k):
        self.ops.append(lambda: len(self.r.z.get(k, [])))
        return self

    def zrange(self, k, _a, _b, withscores=False):
        self.ops.append(lambda: [(str(s), s) for s in sorted(self.r.z.get(k, []))[:1]])
        return self

    def expire(self, _k, _t):
        self.ops.append(lambda: True)
        return self

    def execute(self):
        return [op() for op in self.ops]


class _Redis:
    def __init__(self):
        self.z = {}

    def purga(self, k, hasta):
        self.z[k] = [s for s in self.z.get(k, []) if s > hasta]
        return 0

    def pipeline(self):
        return _Pipe(self)

    def zadd(self, k, miembros):
        self.z.setdefault(k, []).extend(miembros.values())


def _agota(lim, uid="u1"):
    for _ in range(lim.max_calls):
        lim(None, verified_user_id=uid)
    with pytest.raises(HTTPException) as e:
        lim(None, verified_user_id=uid)
    assert e.value.status_code == 429


def test_dos_limitadores_con_el_mismo_par_no_comparten_cupo(monkeypatch):
    monkeypatch.setattr(rl, "redis_client", _Redis())
    metas = rl.RateLimiter(max_calls=3, period_seconds=60)
    medidor = rl.RateLimiter(max_calls=3, period_seconds=60)
    _agota(metas)
    assert medidor(None, verified_user_id="u1") == "u1"      # antes: 429 con el cupo gastado por «metas»


def test_un_mismo_limitador_sigue_contando_entre_endpoints(monkeypatch):
    monkeypatch.setattr(rl, "redis_client", _Redis())
    compartido = rl.RateLimiter(max_calls=2, period_seconds=60)
    _agota(compartido)                                       # el mismo objeto en dos rutas = un cupo, como siempre


def test_la_identidad_es_estable_entre_workers():
    sitios = {rl.RateLimiter(max_calls=3, period_seconds=60).scope for _ in range(2)}
    assert len(sitios) == 1 and sitios.pop().startswith(__name__ + ":")
    assert rl.RateLimiter(max_calls=3, period_seconds=60, scope="panel").scope == "panel"


def test_knob_apagado_vuelve_a_la_clave_compartida(monkeypatch):
    monkeypatch.setenv("MEALFIT_RATE_LIMIT_KEY_PER_LIMITER", "false")
    monkeypatch.setattr(rl, "redis_client", _Redis())
    a = rl.RateLimiter(max_calls=3, period_seconds=60)
    b = rl.RateLimiter(max_calls=3, period_seconds=60)
    _agota(a)
    with pytest.raises(HTTPException):
        b(None, verified_user_id="u1")


def test_ancla():
    assert "tooltip-anchor: P1-PLAN-LOTE-945" in (_BACKEND / "rate_limiter.py").read_text(encoding="utf-8")
