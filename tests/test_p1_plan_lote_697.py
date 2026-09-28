# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-697 · 2026-09-28] La respuesta ESCRITA a las dudas de la foto se recalcula; la tocada, tal cual.

Batería en seco del 694 (caso P5, DeepSeek, US$0,002): «Mi cena / 4 huevos con 3 yemas · Verde» registró 550 kcal y
20 g de proteína —las cifras de la foto con 2 huevos— porque la regla afirmaba que la «Estimación» ya incluía las
respuestas. Solo es cierto cuando se tocaron opciones (su `ajuste` se aplica). Las pruebas viven en
test_p1_plan_lote_694.py (`test_697_*`); este fichero ancla el marcador."""


def test_marker():
    import app
    # [P1-PLAN-LOTE-713] «el marker no bajó de 697», no «el marker empieza por 69»: el siguiente
    # bloque de otra sesión lo sube a 7xx y el prefijo fallaba sin que nada se hubiera roto.
    import re
    m = re.match(r"P1-PLAN-LOTE-(\d+) ", app._LAST_KNOWN_PFIX)
    assert m and int(m.group(1)) >= 697, app._LAST_KNOWN_PFIX
