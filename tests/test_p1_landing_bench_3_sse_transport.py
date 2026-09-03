"""[P1-LANDING-BENCH-3 · 2026-08-07] El benchmark remoto viaja por SSE, con diagnóstico a bordo.

QUÉ PASÓ. La verificación post-deploy de P1-DAYGEN-DIET-CONVERGE quedó CIEGA en 5/8
perfiles: el endpoint síncrono `/analyze` corta generaciones largas (proxy_read_timeout
de nginx), y el retry informado del fix alarga el pipeline justo por encima. El frontend
usa `/analyze/stream` (SSE con heartbeats) precisamente por esto — el benchmark ahora
también. Y como el header X-Bioboros-Review-Diag no puede viajar en un stream ya abierto,
el evento `error` de un rechazo crítico lleva `review_issues` DENTRO del payload (paridad
con el header del síncrono, P1-LANDING-BENCH-2).

QUÉ ANCLA:
  1. El evento SSE de `critical_restriction` incluye `review_issues` + `fallback_reason`
     (sin ellos, un 422 vía stream vuelve a ser indiagnosticable desde fuera).
  2. El runner tiene el cliente SSE (`_remote_generate_stream`), default `--transport sse`,
     y cae al síncrono si el deploy no sirve event-stream (mismo fallback del frontend).
  3. Funcional contra stub: heartbeats tolerados, `complete` → plan, `error` → excepción
     CON el diagnóstico, respuesta no-SSE → fallback señalizado.

tooltip-anchor: P1-LANDING-BENCH-3
"""
from __future__ import annotations

import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_RUNNER_PATH = _BACKEND / "scripts" / "landing_benchmark.py"
_RUNNER_SRC = _RUNNER_PATH.read_text(encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("landing_benchmark_cli", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. El evento SSE crítico lleva el diagnóstico
# ---------------------------------------------------------------------------
def test_sse_critical_event_carries_review_issues():
    m = re.search(r"FALLBACK-GUARD/SSE.*?yield f\"data: \{_json\.dumps\(\{'event': 'error', "
                  r"'data': \{'code': 'critical_restriction'.*?\n", _PLANS_SRC, re.DOTALL)
    assert m, "no encuentro el yield del rechazo crítico en /analyze/stream"
    block = m.group(0)
    assert "'review_issues'" in block and "'fallback_reason'" in block, (
        "P1-LANDING-BENCH-3: el evento SSE de critical_restriction perdió el diagnóstico "
        "(review_issues/fallback_reason) — un 422 vía stream vuelve a ser indiagnosticable, "
        "el mismo agujero que el header cerró para el síncrono (P1-LANDING-BENCH-2)."
    )
    assert "[:160]" in block and "[:5]" in block, "el diagnóstico SSE debe truncar igual que el header"


# ---------------------------------------------------------------------------
# 2. El runner: cliente SSE por default con fallback al síncrono
# ---------------------------------------------------------------------------
def test_runner_defaults_to_sse_with_sync_fallback():
    assert "_remote_generate_stream" in _RUNNER_SRC
    assert re.search(r'"--transport",\s*choices=\("sse",\s*"sync"\),\s*default="sse"', _RUNNER_SRC), (
        "P1-LANDING-BENCH-3: el modo remote debe usar SSE por default")
    assert "stream no disponible" in _RUNNER_SRC, (
        "el runner debe caer al síncrono cuando el deploy no sirve event-stream")


# ---------------------------------------------------------------------------
# 3. Funcional contra stub SSE
# ---------------------------------------------------------------------------
class _SSEStub(BaseHTTPRequestHandler):
    mode = "complete"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if type(self).mode == "plain":
            body = json.dumps({"detail": "sin sse"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"event": "heartbeat"}\n\n')
        self.wfile.flush()
        if type(self).mode == "complete":
            payload = {"event": "complete", "data": {"days": [{"meals": []}], "calories": 2000}}
        else:
            payload = {"event": "error", "data": {
                "code": "critical_restriction", "message": "rechazo de prueba",
                "fallback_reason": "medical_critical",
                "review_issues": ["DIETA INCOMPATIBLE: atun en plan vegetariano"]}}
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture()
def sse_stub():
    server = HTTPServer(("127.0.0.1", 0), _SSEStub)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_stream_client_returns_plan_on_complete(sse_stub):
    mod = _load_runner()
    _SSEStub.mode = "complete"
    plan = mod._remote_generate_stream(sse_stub, {"x": 1}, timeout_s=30)
    assert plan["calories"] == 2000 and plan["days"]


def test_stream_client_raises_with_diag_on_critical(sse_stub):
    mod = _load_runner()
    _SSEStub.mode = "error"
    with pytest.raises(RuntimeError) as exc:
        mod._remote_generate_stream(sse_stub, {"x": 1}, timeout_s=30)
    msg = str(exc.value)
    assert "critical_restriction" in msg and "DIETA INCOMPATIBLE" in msg, (
        "el error SSE debe llevar el diagnóstico — sin él la corrida vuelve a ser ciega")


def test_stream_client_flags_non_sse_response(sse_stub):
    mod = _load_runner()
    _SSEStub.mode = "plain"
    with pytest.raises(RuntimeError) as exc:
        mod._remote_generate_stream(sse_stub, {"x": 1}, timeout_s=30)
    assert "stream no disponible" in str(exc.value), (
        "una respuesta no-SSE debe señalizar el fallback al síncrono, no colgar")
