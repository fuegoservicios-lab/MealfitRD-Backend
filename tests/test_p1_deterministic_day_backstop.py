# -*- coding: utf-8 -*-
"""[P1-DETERMINISTIC-DAY-BACKSTOP · 2026-09-08] El backstop del día sin LLM, anclado aparte.

Los contratos del ensamblador y de su enganche viven en
`test_p1_deterministic_day.py` y `test_p1_deterministic_day_wired.py`. Éste ancla el tercero: que
un día armado sin pasar por el grafo **se verifique**, y que los filtros clínicos viajen al
selector en vez de dejarle todo el trabajo al backstop.
"""
from tests.test_p1_deterministic_day_wired import (  # noqa: F401
    test_el_backstop_clinico_se_invoca_de_verdad,
    test_el_rechazo_dice_el_MOTIVO_con_las_dos_formas,
    test_los_filtros_clinicos_VIAJAN_al_selector,
)
