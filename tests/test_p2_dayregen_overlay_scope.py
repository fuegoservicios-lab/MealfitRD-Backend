"""[P2-DAYREGEN-OVERLAY-SCOPE · 2026-07-12] El overlay "cocinando" del regen-día se muestra
SOLO en el tab del día que se regenera.

Vivo (screenshots del owner, regen del domingo): la animación de carga aparecía también en
los tabs de lunes y martes — `dayRegenInFlight` era un boolean sin índice de día, y el
overlay per-card del Dashboard no podía escopar.

Contrato (parser sobre el frontend):
1. El contexto trackea `dayRegenIndex` (número | null) — seteado en el start del request,
   en el arm del resume (desde el marker), y limpiado en finally/finish.
2. El Dashboard escopa el overlay per-card: `dayRegenIndex === activeDayIndex` (con
   fallback null → visible, para la ventana de 1 render pre-set).
3. Los estados DISABLED de los botones siguen globales a propósito (un regen a la vez —
   protege créditos): solo el OVERLAY visual se escopa.

tooltip-anchor: P2-DAYREGEN-OVERLAY-SCOPE
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CTX = (_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
_DASH = (_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


def test_context_tracks_day_index():
    assert "const [dayRegenIndex, setDayRegenIndex] = useState(null)" in _CTX
    assert _CTX.count("setDayRegenIndex(") >= 4, (
        "4 sitios: start del request, finally, arm del resume (marker.dayIndex), finish"
    )
    assert "setDayRegenIndex(dayIndex)" in _CTX, "start en-sesión"
    assert "marker?.dayIndex" in _CTX, "arm del resume desde el marker"


def test_context_exposes_index():
    assert "dayRegenIndex," in _CTX.split("}), [")[0].split("dayRegenInFlight,")[1][:400], (
        "dayRegenIndex expuesto en el value del provider junto a dayRegenInFlight"
    )


def test_dashboard_scopes_overlay_to_active_tab():
    i = _DASH.find("P2-DAYREGEN-OVERLAY-SCOPE] el overlay del día solo en SU tab")
    assert i != -1, "el gate del overlay desapareció del Dashboard"
    win = _DASH[i:i + 600]
    assert "dayRegenIndex === activeDayIndex" in win
    assert "dayRegenIndex == null ||" in win, (
        "fallback null → visible (ventana de 1 render antes del set; jamás ocultar un "
        "regen real por una carrera de estado)"
    )


def test_v2_spin_and_daybutton_scoped():
    """[v2] El giro del ícono de 'Cambiar Plato' y el spinner/label del botón del día
    también se escopan (quedaban girando en los otros tabs como residual visual)."""
    assert _DASH.count("dayRegenIndex == null || dayRegenIndex === activeDayIndex") >= 4, (
        "4 gates: overlay per-card + spin del ícono Cambiar Plato + spinner del botón "
        "del día + label 'Actualizando…'"
    )


def test_button_disable_stays_global():
    assert "aria-busy={isDayUpdating}" in _DASH, (
        "los disabled/busy de los botones quedan GLOBALES a propósito: un regen a la vez "
        "(doble regen = doble cobro); solo el overlay visual se escopa"
    )
