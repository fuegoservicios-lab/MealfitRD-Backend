"""[P2-PANTRY-FLOOR-VS-NUDGE · 2026-06-23] Regression guard: separar el PISO DURO
de la nevera (gatea la pausa de mantenimiento) del OBJETIVO RECOMENDADO (nudge visual).

Contexto: el owner pidió subir el "mínimo de nevera" de 3 a 25 porque 3 se veía muy
poco. Pero `CHUNK_MIN_FRESH_PANTRY_ITEMS` NO es un objetivo de nevera surtida — es el
piso bajo el cual `_should_pause_chunk_for_insufficient_pantry` PAUSA la generación de
mantenimiento (no se puede cocinar "desde la nevera" si está casi vacía). Subirlo a 20+
congelaría el mantenimiento de casi todos los usuarios reales (pocos tienen 20 ítems
distintos). La solución: mantener el piso BAJO y añadir `PANTRY_RECOMMENDED_ITEMS` como
meta aspiracional desacoplada que SOLO afecta el copy del banner.

Este test ancla esa separación para que un futuro cambio no re-acople ambos conceptos
ni suba el piso duro a un valor que rompa el mantenimiento.
"""
import re
from pathlib import Path

import constants


_PLANS_SRC = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")
_CONSTANTS_SRC = (Path(__file__).resolve().parent.parent / "constants.py").read_text(encoding="utf-8")


def test_hard_floor_stays_low():
    """El piso duro gatea la pausa de mantenimiento → DEBE quedarse bajo. Si alguien lo
    sube a 20+, congela el mantenimiento de casi todos. Tope de seguridad: 10."""
    assert constants.CHUNK_MIN_FRESH_PANTRY_ITEMS <= 10, (
        f"CHUNK_MIN_FRESH_PANTRY_ITEMS={constants.CHUNK_MIN_FRESH_PANTRY_ITEMS} es demasiado "
        "alto: gatea la PAUSA de mantenimiento; >10 congelaría a casi todos los usuarios. "
        "Si quieres una meta más alta usa PANTRY_RECOMMENDED_ITEMS (nudge, no bloquea)."
    )
    assert constants.CHUNK_MIN_FRESH_PANTRY_ITEMS >= 1


def test_recommended_target_exists_and_decoupled():
    """El objetivo recomendado (nudge visual) existe, es >= el piso, y por default ~20."""
    assert hasattr(constants, "PANTRY_RECOMMENDED_ITEMS"), (
        "Falta PANTRY_RECOMMENDED_ITEMS: es el número aspiracional del banner, "
        "desacoplado del piso duro."
    )
    assert constants.PANTRY_RECOMMENDED_ITEMS >= constants.CHUNK_MIN_FRESH_PANTRY_ITEMS, (
        "El objetivo recomendado nunca debe ser menor que el piso duro."
    )
    # Default aspiracional: claramente por encima del piso para que se lea como meta.
    assert constants.PANTRY_RECOMMENDED_ITEMS >= 10


def test_recommended_knob_is_overridable():
    """El objetivo recomendado se lee desde un knob para A/B sin redeploy."""
    assert "MEALFIT_PANTRY_RECOMMENDED_ITEMS" in _CONSTANTS_SRC, (
        "PANTRY_RECOMMENDED_ITEMS debe leer del knob MEALFIT_PANTRY_RECOMMENDED_ITEMS."
    )


def test_pantry_status_endpoint_exposes_recommended_target():
    """El endpoint /pantry-status DEBE devolver `recommended_target` en sus 3 ramas
    (guest, ok, except) — el banner lo consume para el nudge."""
    # Localizar el cuerpo de api_pantry_status (hasta el próximo decorator @router).
    start = _PLANS_SRC.find("async def api_pantry_status")
    assert start != -1, "No se encontró api_pantry_status en plans.py"
    nxt = _PLANS_SRC.find("\n@router", start)
    body = _PLANS_SRC[start:nxt] if nxt != -1 else _PLANS_SRC[start:]
    # Las 3 ramas de return deben incluir recommended_target.
    returns = re.findall(r"return \{[^}]*\}", body)
    assert len(returns) >= 3, f"Se esperaban >=3 returns en api_pantry_status, hubo {len(returns)}"
    for r in returns:
        assert "recommended_target" in r, (
            f"Una rama de return de api_pantry_status no expone recommended_target: {r}"
        )
    # Y debe importar PANTRY_RECOMMENDED_ITEMS.
    assert "PANTRY_RECOMMENDED_ITEMS" in body, (
        "api_pantry_status debe importar PANTRY_RECOMMENDED_ITEMS."
    )
