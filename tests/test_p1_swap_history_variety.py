"""[P1-SWAP-HISTORY-VARIETY · 2026-07-12] El swap conoce el historial + pulido de display.

Vivo (owner): el swap propuso panqueques/avena EN BUCLE con 64 items en la
Nevera — la señal de variedad solo miraba el plan ACTIVO (2 días = 1 nombre).
Ahora `_cross_day_meal_names_for_swap` mira el mismo slot en los ÚLTIMOS 3
PLANES + lo que el usuario REGISTRÓ comer (consumed_meals, 14 días), cap 12,
y la señal viaja a AMBOS paths (botón vía agent.swap_meal y chat vía
execute_modify_single_meal — paridad).

Mismo turno (receta viva de panqueques): dos manchas de display — "1 tazas
(240 ml) de leche" y "1 cdta de (4 g) de mantequilla" — corregidas en
`_shared_clean` del display polish (todas las superficies con finalize).
tooltip-anchor: P1-SWAP-HISTORY-VARIETY
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PL = f.read()
with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as f:
    _TL = f.read()
with open(os.path.join(_BACKEND, "agent.py"), encoding="utf-8") as f:
    _AG = f.read()


def test_helper_reads_history_and_diary():
    i = _PL.find("def _cross_day_meal_names_for_swap")
    body = _PL[i:i + 4200]
    assert "LIMIT 3" in body, "últimos 3 planes, no solo el activo"
    assert "consumed_meals" in body and "interval '14 days'" in body, \
        "lo que el usuario registró comer también cuenta como 'reciente'"
    assert "cap: int = 12" in body


def test_signal_reaches_both_paths():
    # Botón (swap_meal en agent.py) — slice ampliado al cap nuevo.
    assert "_cross_day_names[:12]" in _AG
    # Chat (execute_modify_single_meal) — paridad, lazy import sin ciclo.
    assert "from routers.plans import _cross_day_meal_names_for_swap" in _TL
    assert _TL.count("VARIEDAD (preferencia FUERTE)") >= 1


def test_display_polish_singular_and_double_de():
    import graph_orchestrator as g

    def _clean(s):
        s2 = g._SINGULAR_ONE_RE.sub(
            lambda m: f"{m.group(1)} {g._SINGULAR_UNIT_MAP.get(m.group(2).lower(), m.group(2))}", s
        )
        return g._DE_PAREN_DE_RE.sub(r"\1 de", s2)

    assert _clean("1 tazas (240 ml) de leche descremada") == "1 taza (240 ml) de leche descremada"
    assert _clean("1 cdta de (4 g) de mantequilla de almendras") == "1 cdta (4 g) de mantequilla de almendras"
    # No tocar plurales legítimos ni singulares correctos:
    assert _clean("2 tazas de arroz") == "2 tazas de arroz"
    assert _clean("0.25 taza (29 g) de avena") == "0.25 taza (29 g) de avena"
