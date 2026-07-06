"""[P3-BUDGET-BANNER-POST-RESTOCK · 2026-07-06] Banner de presupuesto se oculta post-compra.

Pedido del owner: al meter los alimentos del PDF a la Nevera ("Ya compré la
lista"), el banner "Dentro de tu presupuesto…" debe ocultarse — su trabajo era
guiar la COMPRA; con la compra hecha es ruido post-hoc. Señal = la misma del
RestockNudge (`!!planData?.is_restocked || sessionRestocked`); render condicional
→ el layout colapsa sin hueco; reaparece al renovar el ciclo (is_restocked se
resetea con el plan nuevo — CYCLE-RESET-ON-REGEN).
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DASH = os.path.join(_BACKEND, "..", "frontend", "src", "pages", "Dashboard.jsx")

with open(_DASH, encoding="utf-8") as f:
    _SRC = f.read()


def test_banner_hides_when_restocked():
    assert "P3-BUDGET-BANNER-POST-RESTOCK" in _SRC
    i = _SRC.index("P3-BUDGET-BANNER-POST-RESTOCK")
    win = _SRC[i:i + 1200]
    assert "planData?.is_restocked || sessionRestocked" in win, (
        "misma señal que el RestockNudge — el hero entero cuenta una sola historia"
    )
    assert "_restockedNow" in win and "return null" in win, (
        "render condicional: el layout colapsa sin hueco (adaptación dinámica pedida)"
    )


def test_signal_matches_restock_nudge():
    # La señal del nudge existe y es la MISMA expresión (consistencia, no copia drift).
    assert _SRC.count("planData?.is_restocked || sessionRestocked") >= 2, (
        "banner y RestockNudge comparten la señal de 'compra hecha'"
    )
