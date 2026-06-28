"""[P1-REGEN-DAY-IDEMPOTENCY-JSONFIX · 2026-06-28] Descubierto en vivo (corr=23a1b8f3): `mark_regen_day_done`
usaba `_json.dumps(...)` pero el `import json as _json` vivía LOCAL en otra función → NameError SIEMPRE → la marca
de idempotencia (P2-REGEN-DAY-IDEMPOTENCY) fallaba en cada regenerar-día, dejando la puerta a que un retry del
cliente tras corte de red re-corriera el loop LLM y RE-COBRARA. Fix: import local dentro de la función.
"""
from __future__ import annotations

import db_plans


def test_mark_regen_day_done_no_nameerror(monkeypatch):
    """Con execute_sql_write mockeado, la función debe completar (True) — antes reventaba con NameError(_json)."""
    captured = {}

    def fake_write(sql, params):
        captured["params"] = params
        return True

    monkeypatch.setattr(db_plans, "execute_sql_write", fake_write)
    ok = db_plans.mark_regen_day_done("user-123", "plan-abc", 0)
    assert ok is True, "mark_regen_day_done debe retornar True (no NameError)"
    # el value serializado debe ser JSON válido con day_index
    assert captured.get("params") and "day_index" in str(captured["params"][1])


def test_mark_regen_day_done_serializes_json(monkeypatch):
    captured = {}
    monkeypatch.setattr(db_plans, "execute_sql_write", lambda sql, params: captured.update({"params": params}) or True)
    db_plans.mark_regen_day_done("u", "p", 3)
    import json
    payload = json.loads(captured["params"][1])  # debe parsear sin error
    assert payload == {"day_index": 3}
