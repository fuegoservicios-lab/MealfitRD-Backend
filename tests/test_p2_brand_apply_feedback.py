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
    assert "Aplicando tu marca a la lista" in DASH_JSX
    assert DASH_JSX.count("id: 'brand-apply'") >= 3, (
        "loading, success y error comparten id — el toast VIVE del pick al resultado"
    )
    assert "Lista actualizada con tu marca" in DASH_JSX


def test_recalc_retries_once_then_fails_honest():
    i = DASH_JSX.index("_applyOnce")
    win = DASH_JSX[i:i + 3000]
    assert "setTimeout(res, 2000)" in win, "retry 1× tras 2s en blips transitorios"
    assert "tu marca quedó guardada y se aplicará al recargar" in win, (
        "si aún falla, error HONESTO (nada de silencio — eso causó el F5 del owner)"
    )
