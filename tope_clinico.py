# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-915 · 2026-09-29] La cola de identidad no deshace un tope clínico.

Corpus del VPS (566 planes entregados, 219 con alguna condición): la cola de identidad (`identidad_plato`, lotes 46-179)
corre DESPUÉS de los topes clínicos del escudo y sube lo que da nombre al plato a su ración sin saber que ese plan tiene
topes: «↑50→100 g de Guineo» en un plan de cirugía bariátrica, cuyo tope de fruta de alto índice glucémico es de 50 g
(`cap_bariatric_portions`), o «↑40→60 g de Atún en agua» en lactancia, con el tope semanal de pescado ya repartido
(`embarazo_pescado`, lote 187). De las 5.739 subidas del corpus, 2.811 son de planes con condición.

Aquí, el tope en gramos de una LÍNEA con las condiciones del plan, leído de las mismas tablas y knobs que aplican los
topes (no hay una tabla nueva):
  · cirugía bariátrica: queso, yogurt, fruta de alto índice glucémico, fruta, aguacate y frutos secos;
  · diabetes: víveres de alto índice glucémico y fruta dulce;
  · embarazo y lactancia: el pescado y el marisco no suben (0: el total de la semana es del lote 187).
Y en cirugía bariátrica la comida no pasa su VOLUMEN (`tope_por_volumen`: 300 g de sólidos, 200 g la merienda).
`identidad_plato._subir_linea` sube hasta el menor de los dos, su piso o su tope. Sin condiciones conocidas no hay tope
(la conducta de siempre). Knob `MEALFIT_IDENTITY_RAISE_CLINICAL_CAP` (True). tooltip-anchor: P1-PLAN-LOTE-915-TOPE-CLINICO
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_IDENTITY_RAISE_CLINICAL_CAP", True)
    except Exception:                                                          # noqa: BLE001
        return True


def condiciones_de(plan_data) -> Optional[list]:
    """Las condiciones clínicas que la política del plan declara; `None` si el plan no las trae (no se sabe)."""
    try:
        pol = plan_data.get("_plan_policy") if isinstance(plan_data, dict) else None
        clin = ((pol or {}).get("effective") or {}).get("clinical") if isinstance(pol, dict) else None
        conds = clin.get("conditions") if isinstance(clin, dict) else None
        return [str(c) for c in conds if c] if isinstance(conds, (list, tuple)) else None
    except Exception:                                                          # noqa: BLE001
        return None


def es_renal(condiciones) -> bool:
    try:
        from micronutrients import _has_condition
        import graph_orchestrator as go
        return bool(condiciones) and bool(_has_condition(list(condiciones), go._RENAL_CONDITION_TERMS))
    except Exception:                                                          # noqa: BLE001
        return bool(condiciones)             # sin poder comprobarlo, como si lo fuera


def tope_de_linea(linea, condiciones) -> Optional[float]:
    """Gramos hasta los que esa línea puede subir con esas condiciones; `None` = sin tope. Nunca lanza."""
    try:
        if not condiciones or not activo():
            return None
        from micronutrients import _has_condition
        from constants import BARIATRIC_CONDITION_TERMS, DIABETES_CONDITION_TERMS
        from nutrition_calculator import _is_pregnancy_or_lactation
        import embarazo_pescado as ep
        import graph_orchestrator as go
        conds = [str(c) for c in condiciones]
        low = go._norm_text(str(linea))

        def _nombra(tokens) -> bool:
            return any(go._name_has_token(t, low) for t in tokens)

        topes = []
        if _has_condition(conds, BARIATRIC_CONDITION_TERMS):
            for tokens, cap in ((go._BARIATRIC_CHEESE_TOKENS, go.BARIATRIC_CHEESE_CAP_G),
                                (go._BARIATRIC_YOGURT_TOKENS, go.BARIATRIC_YOGURT_CAP_G),
                                (go._BARIATRIC_HIGHGI_FRUIT_TOKENS, go.BARIATRIC_HIGHGI_FRUIT_CAP_G),
                                (go._BARIATRIC_FRUIT_TOKENS, go.BARIATRIC_FRUIT_CAP_G),
                                (go._BARIATRIC_FAT_TOKENS, go.BARIATRIC_AVOCADO_CAP_G),
                                (go._BARIATRIC_NUT_TOKENS, go.BARIATRIC_NUT_CAP_G)):
                if _nombra(tokens):
                    topes.append(float(cap))
                    break                                # el primero que casa, como `cap_bariatric_portions`
        if _has_condition(conds, DIABETES_CONDITION_TERMS) and not any(x in low for x in go._DM2_HIGH_GI_CAP_EXCLUDE):
            if _nombra(go._DM2_HIGH_GI_STARCH_TOKENS):
                topes.append(float(go.DM2_HIGH_GI_CAP_G))
            elif (go.DM2_SWEET_FRUIT_CAP_G and _nombra(go._DM2_SWEET_FRUIT_TOKENS)
                  and not any(x in low for x in go._DM2_SWEET_FRUIT_EXCLUDE)):
                topes.append(float(go.DM2_SWEET_FRUIT_CAP_G))
        if _is_pregnancy_or_lactation({"medicalConditions": conds}) and ep._PESCADO.search(str(linea)):
            topes.append(0.0)
        return min(topes) if topes else None
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-915] tope clínico no-op: {type(e).__name__}: {e}")
        return None


def tope_por_volumen(meal, linea, condiciones, db) -> Optional[float]:
    """En cirugía bariátrica, los gramos hasta los que `linea` puede subir sin que la comida pase su volumen (los
    sólidos que la base mide: 300 g, 200 g la merienda — `cap_bariatric_portions`, 2.ª pasada). `None` = sin tope."""
    try:
        if not condiciones or db is None or not isinstance(meal, dict) or not activo():
            return None
        from micronutrients import _has_condition
        from constants import BARIATRIC_CONDITION_TERMS
        import graph_orchestrator as go
        if not go.BARIATRIC_VOLUME_CAP_ENABLED or not _has_condition([str(c) for c in condiciones], BARIATRIC_CONDITION_TERMS):
            return None
        franja = go._norm_text(str(meal.get("meal") or meal.get("slot") or meal.get("name") or ""))
        tope = go.BARIATRIC_SNACK_VOLUME_G if ("merienda" in franja or "snack" in franja) else go.BARIATRIC_MEAL_VOLUME_G
        total, suya, vista = 0.0, 0.0, False
        for x in meal.get("ingredients") or []:
            if not isinstance(x, str):
                continue
            g = float((db.macros_from_ingredient_string(x) or {}).get("grams") or 0)
            total += g
            if not vista and x == linea:
                suya, vista = g, True
        return max(0.0, float(tope) - (total - suya))
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-915] tope de volumen no-op: {type(e).__name__}: {e}")
        return None


__all__ = ["activo", "condiciones_de", "es_renal", "tope_de_linea", "tope_por_volumen"]
