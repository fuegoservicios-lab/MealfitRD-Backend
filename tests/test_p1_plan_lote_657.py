# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-657 · 2026-09-27] (G90) El doc canónico de países cita símbolos, no números de línea.

`docs/country_system_f1.md` es lo primero que abre quien va a operar el flip o el rollback, y sus 25 anclas
`fichero.py:NNNN` apuntaban a código no relacionado (la que prometía `pricing_mode_for_country` caía en un literal de
salsa). Un nombre no envejece con un insert de 200 líneas; un número sí. Cada fila ya nombra su símbolo: el enlace
queda al fichero.

tooltip-anchor: P1-PLAN-LOTE-657
"""
import re
from pathlib import Path

_DOC = Path(__file__).resolve().parents[1] / "docs" / "country_system_f1.md"


def test_sin_numeros_de_linea():
    txt = _DOC.read_text(encoding="utf-8")
    assert not re.findall(r"\.py:\d+", txt), re.findall(r"\S*\.py:\d+\S*", txt)[:5]
    assert not re.findall(r"\.py#L\d+", txt)


def test_los_simbolos_siguen_nombrados():
    txt = _DOC.read_text(encoding="utf-8")
    for simbolo in ("country_for_form_data", "slot_coherence_backstop_for_meal", "build_meal_timing_rules",
                    "compute_shopping_cost_summary", "user_tz_offset_min"):
        assert simbolo in txt, simbolo
