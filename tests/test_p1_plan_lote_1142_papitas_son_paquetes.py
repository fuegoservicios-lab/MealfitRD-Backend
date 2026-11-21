# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1142 · 2026-11-21] «Me voy a comer 3 papitas de las que te envié» eran 3 PAQUETES de Lay's; el
escáner lo anotó como 3 papas sueltas (≈4 g). En RD «una papita» es un paquetico de papitas."""
def test_papitas_son_paquetes():
    import fotos_recientes as fr
    import vision_agent as va
    with fr.scope("«1 paquete de papas fritas Lay's Clásicas»"):
        p = va._prompt_de_escaneo("3 papitas de las que te envie")
    assert "N UNIDADES ENTERAS como la de esa foto" in p
    assert "'3 papitas' = 3 paquetes" in va._MEAL_VISION_PROMPT
