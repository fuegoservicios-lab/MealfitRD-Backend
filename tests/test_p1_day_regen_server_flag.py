"""[P1-DAY-REGEN-SERVER-FLAG · 2026-07-10] El backend declara el regen-day in-flight en el
propio plan (jsonb_set quirurgico) para que el resume cross-refresh del frontend NO dependa
del marker localStorage escrito por el bundle del click — un cliente con bundle stale (SW del
PWA pre-deploy, caso vivo 2026-07-10 15:47) no escribia nada y el refresh perdia el overlay
aunque el backend siguiera generando. tooltip-anchor: P1-DAY-REGEN-SERVER-FLAG
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_CTX = (_BACKEND.parent / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")


def test_server_flag_declares_inflight_regen():
    # [P1-DAY-REGEN-SERVER-FLAG · 2026-07-10] el resume no puede depender SOLO del marker
    # local: un cliente con bundle stale (SW del PWA pre-deploy, caso vivo 15:47) no escribe
    # nada al click → el refresh pierde el overlay aunque el backend siga generando.
    i_set = _PLANS.find("P1-DAY-REGEN-SERVER-FLAG")
    assert i_set > 0, "falta el flag server-side _day_regen_inflight en el regen-day"
    blk = _PLANS[i_set: i_set + 2600]
    assert "jsonb_set(plan_data, '{_day_regen_inflight}'" in blk, "set quirúrgico (I7-exempt) al arrancar"
    assert "AND user_id = %s" in blk, "I2: el UPDATE filtra ownership"
    assert 'pd.pop("_day_regen_inflight", None)' in _PLANS, "el _day_mutator retira el flag en el persist"
    assert "plan_data - '_day_regen_inflight'" in _PLANS, "los soft-fail (regenerated==0) lo retiran quirúrgicamente"
    assert "_day_regen_inflight" in _CTX, "el resume del frontend sondea el flag cuando no hay marker local"
