"""[P1-QUEUE-5XX-NO-LEGACY · 2026-09-02] Ancla backend del guard del cliente.

Medido en prod (18:26→18:35 UTC): el usuario envió el formulario justo durante el reinicio
de un deploy; `POST /generation-runs` devolvió 502 y el cliente cayó por el camino «no es
event-stream» al endpoint SÍNCRONO legado `/api/plans/analyze` — un plan entero fuera de la
cola, sin run ni drain (exactamente lo que la Fase 1 quiere retirar). Ahora el cliente
reintenta la creación del run con esperas 4/8/12 s, lanza `queue_unavailable` (reintentable)
si sigue caído, y si el tail del run no responde ok reanuda por estado del run. El legado
queda SOLO para el 404 «cola apagada».

Tooltip-anchor: P1-QUEUE-5XX-NO-LEGACY | QUEUE_CREATE_RETRY_MS
"""
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def _plan() -> str:
    p = FRONTEND / "src/pages/Plan.jsx"
    if not p.exists():
        pytest.skip("frontend no visible desde este checkout")
    return p.read_text(encoding="utf-8")


def test_client_retries_queue_and_never_falls_to_legacy_on_5xx():
    src = _plan()
    assert "const QUEUE_CREATE_RETRY_MS = [4_000, 8_000, 12_000];" in src
    assert "eQ.code = 'queue_unavailable';" in src
    assert src.index("eQ.code = 'queue_unavailable';") < src.index("if (runResp.status === 404) {")


def test_client_resumes_run_when_tail_not_ok():
    src = _plan()
    i = src.index("/events`, {")
    assert "return await resumeQueueRunUntilReady(run.run_id" in src[i:i + 900]


def test_vitest_anchor_exists():
    p = FRONTEND / "src/__tests__/Plan.p1_arq25_f1_lifecycle.test.js"
    if not p.exists():
        pytest.skip("frontend no visible")
    assert "P1-QUEUE-5XX-NO-LEGACY" in p.read_text(encoding="utf-8")
