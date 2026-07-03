"""[P4-UPDATE-DISHES-STRICT-ALL · 2026-06-23] Los botones de actualizar platos
cocinan SOLO con la Nevera para TODOS los motivos (incl. cravings/weekend) cuando el
knob `MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS` está ON. Dark-ship: OFF por default
(preserva la leniencia legacy de cravings/weekend + sus tests). Test parser-based
sobre la derivación de `strict_pantry`/`_external_tolerance` en `swap_meal`."""
import re
from pathlib import Path

_AGENT = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")


def test_knob_gates_strict_pantry():
    assert "MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS" in _AGENT, "falta el knob strict-all"
    # _strict_all deriva del knob.
    assert re.search(
        r'_strict_all\s*=\s*os\.environ\.get\(\s*"MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS"',
        _AGENT,
    ), "_strict_all no deriva del knob"
    # strict_pantry = True si strict_all, si no legacy.
    assert re.search(
        r"strict_pantry\s*=\s*True if _strict_all else \(swap_reason not in \(.cravings., .weekend.\)\)",
        _AGENT,
    ), "strict_pantry no respeta el knob strict-all"


def test_default_on_in_code():
    # [P2-AUDIT-V5-BATCH · 2026-07-02] (GAP-14) Invertido: default 'true' en código (patrón
    # P1-VERIFIED-ONLY-DEFAULT-ON — el dark-ship OFF-en-código era la regresión silenciosa
    # ".env reseteado ⇒ cravings/weekend vuelven a comprar"). El baseline legacy de los tests
    # de cravings/weekend vive en tests/conftest.py (setdefault "false").
    assert re.search(
        r'os\.environ\.get\(\s*"MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS"\s*,\s*"true"',
        _AGENT,
    ), "el knob debe default a 'true' en código (rollback sin redeploy via .env)"


def test_external_tolerance_zero_when_strict_all():
    # El bloque de tolerancia externa para cravings/weekend se desactiva con strict_all.
    assert 'if swap_reason in ("cravings", "weekend") and not _strict_all:' in _AGENT, \
        "la tolerancia de ingredientes externos debe quedar en 0 cuando strict_all está ON"
