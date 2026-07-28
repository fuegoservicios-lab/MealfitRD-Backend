"""[P2-BRAND-APPLY-FEEDBACK · 2026-07-06] Feedback en vivo al elegir marca.

Owner: "seleccioné Quaker en avena y tuve que refrescar la página para que la
lista se actualizara — ¿no se puede en tiempo real?". Los logs mostraron que el
recalc SÍ corrió y devolvió 200 — pero tarda 15-40s (pipeline completo + cola
tras el auto-refresh serializado) y NO había ninguna señal visible: el owner
refrescó antes de que llegara y el F5 mató el fetch.

Fix: (1) toast 'Aplicando tu marca a la lista…' INSTANTÁNEO al elegir
(onPrefPending, antes del debounce de 900ms) que vive hasta el resultado
(loading→success/error con id compartido 'brand-apply'); (2) retry 1× tras 2s
si el recalc falla; (3) error honesto si aún falla (la pref queda guardada).
"""
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
BRANDS_JSX = (BACKEND.parent / "frontend" / "src" / "components" / "dashboard"
              / "SupermarketBrands.jsx").read_text(encoding="utf-8")
DASH_JSX = (BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


def test_pending_signal_fires_before_debounce():
    assert "P2-BRAND-APPLY-FEEDBACK" in BRANDS_JSX
    assert "onPrefPending" in BRANDS_JSX
    i = BRANDS_JSX.index("onPrefPending === 'function'")
    j = BRANDS_JSX.index("applyTimerRef.current = setTimeout")
    assert i < j, "la señal pending dispara ANTES del debounce de 900ms (feedback instantáneo)"


def test_dashboard_shows_living_toast():
    # [reapuntado 2026-07-28 · d270c3b] El flujo se volvió OPTIMISTA: el patch visual +
    # "Marca aplicada a tu lista" ocurren AL INSTANTE del pick (antes el success llegaba
    # tras el recalc de 15-40s — el copy "Lista actualizada con tu marca" murió con esa
    # espera). El id compartido 'brand-apply' sigue siendo el contrato del toast vivo.
    assert "Aplicando tu marca a la lista" in DASH_JSX  # fallback sin patch optimista
    assert "Marca aplicada a tu lista" in DASH_JSX      # camino optimista instantáneo
    assert DASH_JSX.count("id: 'brand-apply'") >= 3, (
        "quitar/aplicar/fallback comparten id — el toast VIVE del pick al resultado"
    )


def test_recalc_retries_once_then_fails_honest():
    # [reapuntado 2026-07-28 · d270c3b] La honestidad post-fallo cambió de forma: ya no hay
    # toast de error porque la UI NO miente — el patch optimista dejó la marca aplicada
    # visualmente, la pref quedó guardada server-side y el próximo recalc/recarga trae el
    # costo exacto (documentado inline). El contrato que queda: patch optimista ANTES del
    # recalc + retry 1× + el estado optimista NO se revierte en fallo.
    assert "applyBrandToPlanOptimistic(" in DASH_JSX, (
        "el patch optimista desapareció — sin él, el fallo del recalc vuelve a dejar al "
        "user sin feedback (el F5 del owner)."
    )
    i = DASH_JSX.index("_applyOnce")
    win = DASH_JSX[i:i + 3000]
    assert "setTimeout(res, 2000)" in win, "retry 1× tras 2s en blips transitorios"
    assert "quedó guardada" in win, (
        "el rationale del fallo-silencioso-seguro (pref server-side + próximo recalc) "
        "desapareció del bloque — si además quitaste el patch optimista, restaurar el "
        "toast de error honesto."
    )
