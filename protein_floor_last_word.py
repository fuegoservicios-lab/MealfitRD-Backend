# -*- coding: utf-8 -*-
"""[P1-PROTEIN-FLOOR-LAST-WORD · 2026-09-09] El piso de proteína, medido sobre lo que SE ENTREGA.

## Lo medido, no deducido

Plan real `cd1b2fd0` (dueño, `gain_muscle`, target 123 g/día, piso 0,90 → 110,7 g), generado el
09-sep a las 17:41 y entregado sin rechazo:

    proteína por día entregada .......... [107, 116, 112]
    el reviewer registró, día 1 ......... ratio 0.902  (≈ 111 g)  → PASA por dos milésimas
    lo que hay guardado, día 1 .......... ratio 0.870  (107 g)    → NO pasa

Las dos cifras son honestas y de momentos distintos. El cerrador de proteína sube el día, el
reviewer lo mide, y DESPUÉS corren los recortes: la comida entregada lleva `_egg_day_capped` y
`_portion_realism_capped`. Nadie vuelve a medir.

Comprobado ejecutando, no leyendo: repetir la cadena de calidad sobre ese mismo plan **no añade ni
una marca nueva** (`_protein_closed`, `_final_protein_close`, `_gainmuscle_kcal_floor` ya estaban) —
o sea que la cadena SÍ corrió— y aun así deja el día 1 en 113 g. Lo que falta no es un pase: es que
el último pase que toca cantidades no es el último que mide.

## Lo que este módulo NO hace

**No pelea con los caps.** `P1-CAPS-LAST-WORD` dejó escrito el trade-off y es correcto:

    «si el recorte reabre un hueco de proteína que el cerrador había cerrado, gana el cap.
     360 g de queso cottage en un desayuno no es servible, y un plan que no se puede comer no
     cumple el objetivo aunque el número cuadre.»

Ese acuerdo se respeta AQUÍ TAMBIÉN: la secuencia es **intentar → recortar → medir**, en ese orden
y una sola vez. El bump propone, el cap dispone. Sin bucle: dos guardas que se persiguen sobre la
misma condición OSCILAN, y este repo ya pagó esa lección.

## Lo que sí hace, y es lo que faltaba

Lo que nadie aceptó —porque nadie lo vio— es que el **expediente clínico afirme lo que no se
entregó**. Si tras los caps el día sigue bajo el piso, eso queda ESCRITO en el plan y en la
métrica, con su número real. Un plan puede quedar 4 g corto por una razón legítima; lo que no puede
es que el registro diga que no lo está.

Es el mismo defecto que este día cerró en otras dos formas —el gate de fidelidad puntuando 1.0 con
cero platos del catálogo, y el registry nombrando «ración» a un peso en crudo—: **un informe que
describe algo distinto de lo que se envió.**
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: Fracción del target diario por debajo de la cual el día se considera corto. Es el MISMO
#: `PROTEIN_FLOOR_HARD_PCT` que usa el gate de review: dos pisos distintos para la misma pregunta
#: es cómo nace un veredicto que contradice a otro.
_PISO_POR_DEFECTO = 0.90


def _habilitado() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PROTEIN_FLOOR_LAST_WORD", True)
    except Exception:
        return True


def _num(v) -> float:
    """Gramos de un macro declarado, tolerante a `42`, `"42g"`, `"P:42g"` o basura."""
    try:
        from graph_orchestrator import _meal_macro_num
        return float(_meal_macro_num(v) or 0.0)
    except Exception:
        import re
        m = re.search(r"(\d+(?:[.,]\d+)?)", str(v or ""))
        return float(m.group(1).replace(",", ".")) if m else 0.0


def _proteina_por_dia(days) -> list:
    fuera = []
    for i, d in enumerate(days or [], 1):
        if not isinstance(d, dict):
            continue
        ms = [m for m in (d.get("meals") or []) if isinstance(m, dict)]
        fuera.append((d.get("day", i), round(sum(_num(m.get("protein")) for m in ms), 1)))
    return fuera


def _target(plan_data: dict) -> float:
    return _num((plan_data.get("macros") or {}).get("protein"))


def medir(plan_data: dict, *, piso_pct: float = _PISO_POR_DEFECTO) -> dict:
    """Sólo mide. Sin efectos. Devuelve `{}` si no hay target con el que comparar.

    Separado de `reencuadra_y_mide` a propósito: un medidor que además corrige no puede usarse
    para auditar lo que ya se entregó, y auditar lo entregado es justo para lo que existe.
    """
    try:
        tgt = _target(plan_data)
        if tgt <= 0:
            return {}
        piso = round(tgt * piso_pct, 1)
        dias = _proteina_por_dia(plan_data.get("days"))
        cortos = [{"dia": n, "proteina_g": p, "piso_g": piso, "falta_g": round(piso - p, 1)}
                  for n, p in dias if p < piso]
        return {
            "target_g": tgt, "piso_pct": piso_pct, "piso_g": piso,
            "dias_medidos": len(dias), "cortos": cortos, "cumple": not cortos,
        }
    except Exception as e:
        logger.debug(f"[P1-PROTEIN-FLOOR-LAST-WORD] medición no-op: {type(e).__name__}: {e}")
        return {}


def reencuadra_y_mide(plan_data: dict, *, form_data: Optional[dict] = None,
                      surface: str = "chunk-T1", piso_pct: float = _PISO_POR_DEFECTO) -> dict:
    """Intenta recuperar el piso, deja que el cap tenga la última palabra, y MIDE lo que queda.

    Secuencia FIJA, una sola vuelta: bump → cap → medir. Escribe el resultado en
    `plan_data['_protein_floor_delivered']` y lo devuelve. Fail-safe: jamás levanta, jamás
    bloquea la entrega — un medidor que puede tumbar un plan es peor que no medir.
    """
    if not (_habilitado() and isinstance(plan_data, dict)):
        return {}
    antes = medir(plan_data, piso_pct=piso_pct)
    if not antes:
        return {}
    informe = dict(antes)
    informe["surface"] = str(surface or "")[:40]
    informe["recuperado"] = False

    if antes["cortos"]:
        try:
            import graph_orchestrator as go
            # 1. el bump PROPONE: re-escala porciones proteína-dominantes que YA existen
            #    (nunca añade ingredientes ⇒ cero riesgo de alérgeno; el shield no tiene form_data).
            subio = bool(go.reconcile_protein_band_post_finalize(plan_data))
            # 2. el cap DISPONE — pero SÓLO sobre lo que este pase acaba de inflar.
            #
            #    Dos condiciones, ambas aprendidas de un gate en rojo:
            #
            #    · `subio`: si el bump no tocó nada, este pase no tiene nada que recortar y
            #      llamar al cap sería correr un pase ajeno por la puerta de atrás. Un plan
            #      sintético con 0 g de proteína (el bump devuelve False) veía su línea de
            #      pepino recortada por MI llamada, no por la cadena.
            #    · `CAPS_AFTER_BAND_CLOSER`: es el knob de ROLLBACK de los caps en este punto
            #      de la cadena. Honrar sólo `PORTION_REALISM_CAP_ENABLED` dejaba a un operador
            #      que hizo rollback con los caps corriendo igual — desde aquí. Una defensa que
            #      ignora el interruptor de otra convierte su rollback en mentira.
            #
            #    Una pasada, no un bucle: dos guardas persiguiéndose OSCILAN. Si el recorte
            #    reabre el hueco, el hueco se REPORTA — no se vuelve a pelear.
            if subio:
                try:
                    if go.PORTION_REALISM_CAP_ENABLED and go.CAPS_AFTER_BAND_CLOSER:
                        go._cap_unrealistic_portions(plan_data.get("days"))
                except Exception as e_cap:
                    logger.debug(f"[P1-PROTEIN-FLOOR-LAST-WORD] cap post-bump no-op: "
                                 f"{type(e_cap).__name__}: {e_cap}")
                try:
                    go.refresh_delivered_macros(plan_data)
                except Exception:
                    pass
            informe["recuperado"] = subio
        except Exception as e:
            logger.debug(f"[P1-PROTEIN-FLOOR-LAST-WORD] re-encuadre no-op: {type(e).__name__}: {e}")

    # 3. la medición que manda es la de DESPUÉS de todo lo que toca cantidades.
    despues = medir(plan_data, piso_pct=piso_pct) or {}
    informe["cortos_antes"] = antes["cortos"]
    informe["cortos"] = despues.get("cortos", antes["cortos"])
    informe["cumple"] = despues.get("cumple", antes["cumple"])
    informe["dias_medidos"] = despues.get("dias_medidos", antes["dias_medidos"])

    try:
        plan_data["_protein_floor_delivered"] = informe
    except Exception:
        pass

    if informe["cortos"]:
        logger.warning(
            f"🥩 [P1-PROTEIN-FLOOR-LAST-WORD] {surface}: {len(informe['cortos'])} día(s) se "
            f"ENTREGAN bajo el piso de proteína ({informe['piso_g']} g) tras los recortes: "
            f"{[(c['dia'], c['proteina_g']) for c in informe['cortos']]}. El cap tuvo la última "
            f"palabra sobre las porciones; esto queda registrado para que el veredicto no mienta.")
    elif informe["recuperado"]:
        logger.info(f"🥩 [P1-PROTEIN-FLOOR-LAST-WORD] {surface}: piso de proteína recuperado tras "
                    f"los recortes ({antes['cortos']} → cumple).")
    return informe
