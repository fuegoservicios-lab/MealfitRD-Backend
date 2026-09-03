"""[P2-I18N-DISPLAY-EXCEPCION-NO-DEJA-RASTRO · 2026-08-23] El `except` final del motor —el
único camino de salida que significa «algo está ROTO»— era el único sin telemetría.

`enrich_plan_display` tiene siete salidas y seis dejan fila en `pipeline_metrics` (y alerta
si la razón no es benigna): `disabled`, `dedupe_locked`, `circuit_breaker_open`,
`no_meals`, `no_valid_meals`, `partial_loss`… Todas son estados ESPERADOS. El `except
Exception` de abajo del todo es el que captura lo que nadie previó —un `KeyError` en el
parser, un cambio de forma en `plan_data`, un proveedor devolviendo algo nuevo— y ahí sólo
había `logger.warning` + `return {"skipped": "exception"}`. Cero fila, cero alerta.

O sea: si la capa empieza a reventar por un bug nuevo, la telemetría dice que NO SE HA
EJECUTADO (cero filas) en vez de que se ejecuta y falla. Es la diferencia entre «no le toca a
nadie» y «está roto», y es exactamente la confusión que la tercera auditoría tardó una
sesión en despejar midiendo contra Neon.

El mecanismo ya existía: `_emit_result_telemetry` escribe la fila y, con una `reason` no
benigna, emite la alerta por locale. Sólo faltaba llamarlo desde el `except`.

tooltip-anchor: P2-I18N-DISPLAY-EXCEPCION-NO-DEJA-RASTRO
"""
from __future__ import annotations

from unittest.mock import patch

import plan_display_i18n as pdi

_MARKER = "P2-I18N-DISPLAY-EXCEPCION-NO-DEJA-RASTRO"


def _plan_con_una_comida() -> dict:
    return {"days": [{"meals": [{"name": "Pollo", "description": "x", "ingredients": ["100 g de Pollo"],
                                  "recipe": ["Cocina."]}]}]}


def test_una_excepcion_no_prevista_deja_fila_y_razon(monkeypatch) -> None:
    """La CONDUCTA: se provoca un reventón en medio del motor y se mira qué telemetría
    sale. Sin este arreglo, `_emit_result_telemetry` no se llamaba."""
    emitidas = []
    monkeypatch.setattr(pdi, "_emit_result_telemetry",
                        lambda plan_id, user_id, locale, resumen: emitidas.append(resumen))
    monkeypatch.setattr(pdi, "_plan_display_i18n_enabled", lambda: True)

    # Se revienta lo primero que el motor toca tras los gates: cargar el plan.
    def _revienta(*a, **k):
        raise KeyError("forma nueva de plan_data que nadie previó")
    monkeypatch.setattr(pdi, "_fetch_plan_data", _revienta)

    out = pdi.enrich_plan_display("plan-x", "user-x", "fr-FR")

    assert out.get("skipped") == "exception", "premisa: el except final es el que sale"
    razones = [e.get("reason") for e in emitidas]
    assert "exception" in razones, (
        f"el `except` final NO emitió telemetría (razones emitidas: {razones}). La capa "
        f"puede estar reventando por un bug nuevo y `pipeline_metrics` dirá que no se "
        f"ejecuta — «no le toca a nadie» y «está roto» se vuelven indistinguibles. [{_MARKER}]"
    )


def test_exception_no_es_razon_benigna() -> None:
    """Si alguien la mete en `_RAZONES_BENIGNAS`, la fila se escribe pero la alerta no: el
    operador no se entera. Una excepción no prevista nunca es benigna."""
    assert "exception" not in pdi._RAZONES_BENIGNAS, (
        f"`exception` está en `_RAZONES_BENIGNAS`: el reventón del motor no alertaría [{_MARKER}]"
    )
