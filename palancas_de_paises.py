# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-651 · 2026-09-27] (G89) Las palancas de rollback del sistema de países, en el registro desde el arranque.

El registro de knobs solo conoce uno cuando se LEE, y estas se leían al armar la primera lista o el primer presupuesto:
tras arrancar, el registro mostraba UNA de las siete palancas del flip, justo lo que un operador consulta en un
incidente. `registrar()` las lee una vez; `app.py` la llama al arrancar. Vive aquí y no al final de
`shopping_calculator.py`/`nutrition_calculator.py` porque esos god-files tienen el tope de líneas congelado. La lectura
sigue siendo por llamada: esto solo las hace visibles.

tooltip-anchor: P1-PLAN-LOTE-651
"""
import logging

logger = logging.getLogger(__name__)


def registrar() -> int:
    """Lee cada palanca una vez (así entra en `_KNOBS_REGISTRY`). Devuelve cuántas se leyeron. Nunca lanza."""
    leidas = 0
    try:
        import shopping_calculator as sc
        import nutrition_calculator as nc
        palancas = (sc._country_catalog_unpriced_keep_enabled, sc._country_keep_respect_recipe_qty_enabled,
                    sc._unit_system_by_country_enabled, sc._baking_staples_keep_enabled,
                    sc._seasoning_catalog_keep_enabled, nc._budget_floor_enabled)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-651] palancas de países no registradas: {e!r}")
        return 0
    for palanca in palancas:
        try:
            palanca()
            leidas += 1
        except Exception:  # noqa: BLE001 — registrar no puede tumbar el arranque
            pass
    return leidas
