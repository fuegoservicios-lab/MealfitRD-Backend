"""[P2-I18N-DISPLAY-REASON-NULA-CON-CERO-COMIDAS · 2026-08-23] Si se escribía el nombre del
plan o los insights pero NINGUNA comida, la fila de telemetría decía `reason: null` —éxito
limpio— y no se emitía la alerta.

El `reason` del resumen se calcula así:

    "partial_loss" si total_written > 0 y targets_perdidos > 0
    None           si total_written > 0
    last_skip_reason en otro caso

y `last_skip_reason` se pone a `None` «si se escribió algo», donde «algo» incluye el nombre
del plan y los insights. Así que con `no_valid_meals` en todos los lotes de COMIDA pero el
nombre del plan escrito, `last_skip_reason` se limpia y el resumen sale sin razón: el plan
queda con el título en francés y los doce platos en español, y la telemetría lo registra
como un ciclo perfecto. Sin alerta.

Es el estado exacto que se encontró en producción el 22-ago —el único plan enriquecido:
nombre + insights traducidos, 0/12 platos— y la telemetría lo habría dado por bueno.

LA REGLA: que se haya escrito el nombre o los insights NO limpia la razón de que las comidas
no se escribieran. Son tres campos del mismo contrato, pero un ciclo que pidió comidas y no
escribió ninguna ha fallado en lo que pidió.

tooltip-anchor: P2-I18N-DISPLAY-REASON-NULA-CON-CERO-COMIDAS
"""
from __future__ import annotations

import re
from pathlib import Path

_MARKER = "P2-I18N-DISPLAY-REASON-NULA-CON-CERO-COMIDAS"
_MOD = Path(__file__).resolve().parents[1] / "plan_display_i18n.py"


def test_escribir_el_nombre_o_los_insights_no_limpia_la_razon_de_las_comidas() -> None:
    """Parser-based sobre la línea que decide, con su forma exacta: la limpieza de
    `last_skip_reason` tiene que depender SÓLO de `written` (comidas), no de
    `_plan_name_written` ni `_insights_written`."""
    src = _MOD.read_text(encoding="utf-8")
    m = re.search(r"if (.+?):\s*\n\s*last_skip_reason = None", src)
    assert m, f"desapareció la limpieza de `last_skip_reason` [{_MARKER}]"
    condicion = m.group(1)
    assert "_plan_name_written" not in condicion and "_insights_written" not in condicion, (
        f"la razón de las comidas se limpia cuando se escribe el NOMBRE o los INSIGHTS "
        f"(`{condicion}`): un ciclo con el título en francés y 0/12 platos sale como "
        f"`reason: null`, éxito limpio, sin alerta. Es el estado que se encontró en "
        f"producción el 22-ago. [{_MARKER}]"
    )
    assert re.search(r"\bwritten\b", condicion), (
        f"la limpieza ya no mira `written` (comidas escritas): `{condicion}` [{_MARKER}]"
    )
