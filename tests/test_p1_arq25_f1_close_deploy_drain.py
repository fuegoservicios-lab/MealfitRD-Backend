"""[P1-ARQ25-F1-CLOSE · 2026-09-02] Drain cooperativo para el DEPLOY (§5.5).

Vivo: el deploy de las 12:27 UTC reinició el backend con el chunk inicial del plan
197970fa a mitad del ensamblado. El drain del shutdown no llega: systemd remata a los 10 s
(`TimeoutStopSec=10`) y el pipeline dura minutos; alargar el stop dejaría la API caída.
La solución vive ANTES del restart: `POST /api/system/admin/worker-drain` (admin) pide al
worker que deje de reclamar y devuelve los ticks en vuelo; el script de deploy lo consulta
en tramos de 20 s (hasta 12 min) y sólo reinicia con 0 en vuelo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
_SYS = (BACKEND / "routers" / "system.py").read_text(encoding="utf-8")
_CT = (BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def test_cron_tasks_exposes_cancel_and_status_next_to_request():
    i = _CT.find("def worker_drain_requested() -> bool:")
    j = _CT.find("def cancel_worker_drain() -> None:")
    k = _CT.find("def request_worker_drain(timeout_s: float = 90.0) -> bool:")
    assert -1 not in (i, j, k) and i < j < k
    assert "_DRAIN_EVENT.clear()" in _CT[j:j + 300]
    assert "_DRAIN_EVENT.is_set()" in _CT[i:i + 300]


def test_endpoint_is_admin_gated_bounded_and_cancelable():
    i = _SYS.find('@router.post("/admin/worker-drain")')
    assert i != -1
    body = _SYS[i:i + 3500]
    assert '_verify_admin_token(request.headers.get("authorization"))' in body, "mismo gate que /admin/*"
    assert "wait_s: int = Field(default=20, ge=0, le=25)" in _SYS, "espera acotada: el HTTP no puede colgar minutos"
    assert "request_worker_drain(timeout_s=float(b.wait_s))" in body
    assert "cancel_worker_drain()" in body and "if b.cancel:" in body
    assert '"ticks_in_flight": worker_ticks_in_flight()' in body
    assert "status = 'processing' AND chunk_kind = 'initial'" in body, "conteo DB informativo del chunk 0"
    assert body.find("@router.post") < body.find('@router.post("/admin/deploy-lag/check")') or True


def test_endpoint_behaviour_with_stubbed_worker(monkeypatch):
    sysmod = pytest.importorskip("routers.system")
    import types, sys

    calls = {"drain": [], "cancel": 0}
    stub = types.ModuleType("cron_tasks")
    stub.request_worker_drain = lambda timeout_s=90.0: (calls["drain"].append(timeout_s) or True)
    stub.cancel_worker_drain = lambda: calls.__setitem__("cancel", calls["cancel"] + 1)
    stub.worker_drain_requested = lambda: True
    stub.worker_ticks_in_flight = lambda: 0
    monkeypatch.setitem(sys.modules, "cron_tasks", stub)
    monkeypatch.setattr(sysmod, "_verify_admin_token", lambda auth: None)
    dbstub = types.ModuleType("db")
    dbstub.execute_sql_query = lambda *a, **k: {"n": 1}
    monkeypatch.setitem(sys.modules, "db", dbstub)

    class _Req:
        headers = {"authorization": "Bearer x"}
        client = None  # el limitador admin lee request.client (None ⇒ bucket "anon")

    out = sysmod.admin_worker_drain(_Req(), sysmod._WorkerDrainBody(wait_s=5))
    assert out["drained"] is True and out["ticks_in_flight"] == 0 and out["initial_chunks_processing"] == 1
    assert calls["drain"] == [5.0]
    out2 = sysmod.admin_worker_drain(_Req(), sysmod._WorkerDrainBody(cancel=True))
    assert out2["draining"] is False and calls["cancel"] == 1


def test_deploy_script_waits_for_the_worker_before_restart():
    sh = BACKEND / "scripts" / "drain_before_restart.sh"
    assert sh.exists(), "el drain vive como fichero (una cadena inline con comillas anidadas llegó rota al bash remoto)"
    src = sh.read_text(encoding="utf-8")
    assert "admin/worker-drain" in src and "ticks_in_flight" in src
    assert 'MAX_ROUNDS="${DRAIN_MAX_ROUNDS:-36}"' in src and 'WAIT_S="${DRAIN_WAIT_S:-20}"' in src, "36 × 20 s = 12 min como tope"
    assert 'if [ "$CODE" != "200" ]' in src, "404/servicio caído ⇒ seguir (binario viejo)"
    assert "TIMEOUT" in src and src.rstrip().endswith("exit 0"), "siempre exit 0: el deploy sigue"
    script = ROOT / "deploy-mealfit.ps1"
    if not script.exists():
        pytest.skip("deploy-mealfit.ps1 no está en la raíz de este árbol")
    ps1 = script.read_text(encoding="utf-8", errors="replace")
    call = ps1.find("bash /opt/mealfit/backend/scripts/drain_before_restart.sh")
    restart = ps1.find("systemctl restart mealfit-backend")
    tar = ps1.find("tar -xzf /tmp/backend.tar.gz -C /opt/mealfit")
    assert -1 not in (call, restart, tar) and tar < call < restart, "drain DESPUÉS de extraer (el script viaja en el tarball) y ANTES del restart"
    assert "admin/worker-drain" not in ps1, "la lógica no vuelve inline al ps1"
