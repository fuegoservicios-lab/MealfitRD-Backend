"""[P1-PLAN-LOTE-50 · 2026-09-14] El lote de la auditoría del coach (agente de chat).

El marcador de deploy sigue la numeración de lotes (los tests de los LOTE-24…49 exigen
`P1-PLAN-LOTE-N` con N ≥ el suyo). Este lote agrupa cinco P-fix, cada uno con su test:

- P0-CHAT-IDENTITY-FROM-TOKEN — `test_p0_chat_identity_from_token.py`
- núcleo del agente (historial saneado, alergias con el valor guardado…) — `test_p1_chat_core_audit.py`
- P1-CHAT-FACTS-AUDIT — `test_p1_chat_facts_audit.py`
- P1-CHAT-TOOLS-AUDIT — `test_p1_chat_tools_audit.py`
- P1-CHAT-ORPHAN-SESSIONS — `test_p1_chat_orphan_sessions.py`, `test_p1_chat_orphan_sessions_callsites.py`

Aquí solo se ancla el marcador y que los cinco tests del lote existen.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_marcador_del_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 50 and m.group(2) >= "2026-09-14"
    assert "[P1-PLAN-LOTE-50 · 2026-09-14] = [P0-CHAT-IDENTITY-FROM-TOKEN]" in app


def test_los_tests_del_lote_existen():
    for nombre in (
        "test_p0_chat_identity_from_token.py",
        "test_p1_chat_core_audit.py",
        "test_p1_chat_facts_audit.py",
        "test_p1_chat_tools_audit.py",
        "test_p1_chat_orphan_sessions.py",
        "test_p1_chat_orphan_sessions_callsites.py",
    ):
        assert (_BACKEND / "tests" / nombre).exists(), nombre
