"""[P2-PLAN-POLL-DORMANT-SLEEP · 2026-09-04] El sondeo del plan confundía «dormido» (siguiente bloque
programado para más adelante) con «rendido»: mataba el loop, avisaba give-up y el banner «Dejamos de
revisar si llegaron tus próximas semanas…» salía en cada plan SANO al dejar la pestaña un rato; solo se
iba refrescando la página. Ancla: dormir = latido largo + despertar al volver a la pestaña; give-up solo
para la pantalla muda (0 días); el reloj de give-up no cuenta el sueño; «Revisar ahora» reinicia el loop.
"""
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_CR = chr(13)


def _frontend_file(*parts: str) -> str:
    for base in (_HERE.parents[2], _HERE.parents[1].parent):
        p = base.joinpath("frontend", "src", *parts)
        if p.exists():
            return p.read_text(encoding="utf-8").replace(_CR, "")
    pytest.skip("frontend hermano no disponible")


def test_a_dormido_no_es_rendido():
    hook = _frontend_file("hooks", "usePlanPollLoop.js")
    assert "scheduleNext(dormantMs);" in hook
    assert "onGiveUpChangeRef.current?.(!(Number(snapshot.daysCount) > 0));" in hook, "give-up solo para la pantalla muda"
    assert "document.addEventListener('visibilitychange', onVisible);" in hook
    assert "document.removeEventListener('visibilitychange', onVisible);" in hook


def test_b_el_reloj_de_give_up_no_cuenta_el_sueno():
    hook = _frontend_file("hooks", "usePlanPollLoop.js")
    reset = "if (dormant) { activeSinceMs = nowBeforeFetch; dormant = false; }"
    assert reset in hook
    # el reset va ANTES del tope wall-clock: si no, dos sueños de 15 min disparan el give-up de 30
    assert hook.index(reset) < hook.index("if (hasPollGivenUp(activeSinceMs, nowBeforeFetch, giveUpMs)) {")


def test_c_revisar_ahora_reinicia_el_loop_de_verdad():
    ctx = _frontend_file("context", "AssessmentContext.jsx")
    assert "const restartPlanPoll = useCallback(() => {" in ctx
    assert "resetKey: `${planData?.id ?? ''}#${pollRestartNonce}`," in ctx
    hook = _frontend_file("hooks", "usePlanPollLoop.js")
    # un (re)arranque limpia el give-up: la llamada va justo antes de arrancar el estado del loop
    start = hook.index("let cancelled = false;")
    clear = hook.rindex("onGiveUpChangeRef.current?.(false);", 0, start)
    assert 0 < start - clear < 200
    dash = _frontend_file("pages", "Dashboard.jsx")
    assert "hydrateLatestPlan?.({ force: true, src: 'give-up-retry' }); restartPlanPoll?.();" in dash


def test_d_knob_del_latido():
    backoff = _frontend_file("utils", "planPollBackoff.js")
    assert backoff.count("export const PLAN_POLL_DORMANT_MS") == 1
    assert "_env.VITE_PLAN_POLL_DORMANT_MS, 15 * 60 * 1000" in backoff
