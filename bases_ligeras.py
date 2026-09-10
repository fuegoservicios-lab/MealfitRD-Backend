# -*- coding: utf-8 -*-
"""[P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA · 2026-09-10] SSOT de las bases ligeras de desayuno y merienda.

Vivía como una tupla dentro de `graph_orchestrator.py`, que está en su tope de 53.100 líneas
(«extraer, no subir el cap»). Aquí la leen los dos caminos que arman días: la autocrítica del modelo
(`_detect_light_base_repeats`) y el día determinista (`deterministic_day._bases_ligeras_de`).

La FAMILIA existe porque la regla casa por PREFIJO de palabra y «arepitas» no empieza por «arepa»
(a-r-e-p-I): ni la etiqueta «arepa» veía las arepitas, ni una etiqueta «arepita» aparte las juntaba
con la arepa. Un desayuno de arepa y una merienda de arepitas son la MISMA base.
"""

LIGHT_BASE_TOKENS = ("avena", "granola", "cereal", "hojuelas", "muesli", "pan integral", "casabe",
                     "tostada", "arepa", "arepita", "tortilla")

_FAMILIA = {"arepita": "arepa"}


def familia_base_ligera(token: str) -> str:
    """La base a la que pertenece un token: «arepita» -> «arepa»; el resto, él mismo."""
    return _FAMILIA.get(token, token)
