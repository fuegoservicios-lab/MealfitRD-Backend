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


def test_resume_completion_has_no_client_clock_comparison():
    # [P2-REGEN-RESUME-NO-CLOCKS · 2026-07-10] la completion NO compara reloj del cliente vs
    # timestamp del servidor (con drift de minutos declaraba éxito instantáneo con el plan sin
    # cambiar — reporte del owner: "dijo que había creado el plan" en <10s): flag-presence =
    # "sigue corriendo"; flag ausente + baseline server-vs-server (baseModifiedAt) = terminó.
    assert "baseModifiedAt" in _CTX, "baseline server-vs-server para la completion (sin relojes cliente)"
    assert "modifiedAt && modifiedAt > marker.startedAt" not in _CTX, (
        "P2-REGEN-RESUME-NO-CLOCKS: reapareció la comparación reloj-cliente vs servidor "
        "(fuente del falso '¡Día actualizado!' en <10s)"
    )
    assert "flagFresh" in _CTX, "la presencia del flag es la señal primaria de 'sigue corriendo'"


def test_credits_refresh_live_after_regen_success():
    # [P1-CREDITS-LIVE-REFRESH · 2026-07-10] el regen-day cobra 1 crédito server-side; el
    # contador debe bajar EN VIVO (invalidate + re-fetch) — antes solo al refrescar la web.
    assert "invalidatePlanCountCache" in _CTX, (
        "P1-CREDITS-LIVE-REFRESH: el éxito del regen-day (en-sesión y resume) debe invalidar "
        "el cache de cuota y re-fetch del contador"
    )
    i_ok = _CTX.find("[P1-CREDITS-LIVE-REFRESH · 2026-07-10] el regen exitoso cobró")
    assert i_ok > 0 and "checkPlanLimit()" in _CTX[i_ok: i_ok + 600], (
        "el path ok:true de regenerateDay debe re-fetch el contador"
    )
