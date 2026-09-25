# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-267 · 2026-09-25] Tras una sustitución determinista, la DESCRIPCIÓN tampoco nombra lo que se quitó.

Verificación con IA real del lote 264 (DM2 + HTA, metformina): la lista ya traía tayota en vez de cundeamor, el título y
los pasos estaban reescritos… y la descripción seguía prometiendo «con cundeamor salteado». Es texto que el usuario lee
(y el mismo defecto le diría «con pan tostado» a un celíaco cuyo pan pasó a casabe). El reescritor del título
(`_rewrite_meal_name_after_subs`: acentos, artículo, token más largo primero, sin duplicar el reemplazo) sirve tal cual
para la descripción; va con el mismo knob (`MEALFIT_SUBST_NAME_REWRITE`). tooltip-anchor: P1-PLAN-LOTE-267-DESC
"""
from __future__ import annotations


def titulo_y_desc(meal, token_subs) -> bool:
    """Reescribe el título (como siempre) y la descripción tras una sustitución. True si cambió alguno. Nunca lanza."""
    import graph_orchestrator as go
    cambio = False
    try:
        cambio = bool(go._rewrite_meal_name_after_subs(meal, token_subs))
    except Exception:
        pass
    try:
        desc = meal.get("desc") if isinstance(meal, dict) else None
        if isinstance(desc, str) and desc.strip():
            tmp = {"name": desc}
            if go._rewrite_meal_name_after_subs(tmp, token_subs):
                meal["desc"] = tmp["name"]
                cambio = True
    except Exception:
        pass
    return cambio
