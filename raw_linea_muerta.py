# -*- coding: utf-8 -*-
"""[P1-RAW-LINEA-MUERTA · 2026-09-09] La línea de `ingredients_raw` que compra lo que la
receta ya no menciona.

## El defecto, medido en un plan real

Plan `c4098931`, cena del día 1 («Guiso criollo de Pavo»): `ingredients` traía **una** línea de
pavo («185 g de pechuga de pavo») e `ingredients_raw` traía **las dos** — la superada («135 g de
pechuga de pavo en lonjas/tiras») y la vigente. La lista de compras lee `ingredients_raw`
PRIMERO (`shopping_calculator` :6383 y :13611), así que compró `Pechuga de pavo` RD$285 **y**
`Jamón de pavo` RD$191,25: **RD$476 de pavo para el único plato con pavo del plan**, y un
alimento en la lista que no aparece en ninguna de las 12 comidas. Segundo caso el mismo día:
desayuno del día 3, «1 huevo» superada por «1 clara de huevo». **2 de 12 comidas.**

## Por qué ninguna defensa existente lo vio (comprobado EJECUTÁNDOLAS, no leyéndolas)

* `_reconcile_display_raw_lines` corrido sobre esa cena devuelve **0 filas y no cambia nada**:
  usa el resolvedor NUTRICIONAL, para el que las dos líneas son `Pechuga de pavo` — un solo
  alimento, presente en ambos lados, nada que reconciliar.
* `_reconcile_raw_missing_in_display` va en la dirección contraria y es **solo-append** por
  diseño: «el display solo GANA la línea que ya está en raw/compras (**la verdad**)». Una línea
  superada rompe exactamente esa premisa: raw deja de ser la verdad y pasa a ser la verdad
  **más su historia**.
* El guard de coherencia vio las 3 divergencias y las llamó `recipe_unquantified`, con
  `action_taken: not_applicable`. Tiene una hipótesis para «la receta no cuantifica» y ninguna
  para «sobra una línea».

## El fondo

**La misma cadena resuelve a dos alimentos distintos según quién pregunte:**
`macros_from_ingredient_string('…pechuga de pavo en lonjas/tiras')` → `Pechuga de pavo`;
`_parse_quantity(…)` → `Jamón de pavo`. El discriminante medido es el token «lonjas» — «en
tiras» a secas resuelve bien en los dos. Todo lo que reconcilia usa el nutricional, así que
**del lado que MIRA hay un solo pavo y del lado que GASTA hay dos**. Por eso este barrido
pregunta por la identidad de COMPRA a propósito: es la que paga.

*Dos resolvedores que no coinciden hacen el defecto invisible para todo el que use el otro.*

## Por qué vive en su propio módulo

`graph_orchestrator.py` es un god-file con techo duro, y estaba **exactamente** en él (53.100
líneas). El primer intento metió las ~130 líneas de aquí dentro y el gate rechazó el despliegue
con el mensaje correcto: «extraer, no subir el cap». El enganche allí cuesta **0 líneas netas**
— dos líneas sustituidas por dos líneas.
"""
import logging
from typing import Optional

from knobs import _env_bool, _env_int  # registro de knobs compartido (/health/version)

logger = logging.getLogger(__name__)

# Corre DESPUÉS de `_reconcile_raw_missing_in_display`, y ese orden es lo que impide que las dos
# guardas oscilen: la recíproca ya adoptó al display las líneas de raw legítimamente ausentes,
# así que lo que siga sin respaldo en el display está muerto de verdad.
RAW_DEAD_LINE_SWEEP_ENABLED = _env_bool("MEALFIT_RAW_DEAD_LINE_SWEEP", True)
# Tope por comida: si «muchas» líneas parecen muertas, lo que falló es el resolvedor y no la
# receta — se registra y no se toca nada. Un barrido sin tope convierte una regresión del
# resolvedor en una lista de compras vacía, y el usuario no tendría cómo enterarse.
RAW_DEAD_LINE_SWEEP_MAX_PER_MEAL = _env_int("MEALFIT_RAW_DEAD_LINE_SWEEP_MAX", 2,
                                            validator=lambda v: 1 <= v <= 10)


def _identidad_de_compra(linea: str) -> Optional[str]:
    """El nombre con el que la LISTA DE COMPRAS registraría esta línea, o None si no resuelve.

    Es deliberadamente el resolvedor de compras (`_parse_quantity`) y no el nutricional: son la
    misma pregunta para casi todo el catálogo y **discrepan justo en el caso que cuesta dinero**.
    """
    try:
        from shopping_calculator import _parse_quantity
        _q, _u, nombre = _parse_quantity(linea, apply_yield_multiplier=False,
                                         apply_legumbres_yield_only=True,
                                         apply_protein_yield=False)
        return nombre or None
    except Exception:                                                  # noqa: BLE001
        return None


def _barrer_lineas_muertas_de_raw(days) -> int:
    """Quita de `ingredients_raw` la línea que COMPRA un alimento que la receta ya no menciona.

    Conservador por tres lados: si alguna línea del display no resuelve, la comida entera se
    salta (sin el conjunto completo de «vivos» no hay veredicto — contra un conjunto incompleto
    TODO parece muerto); nunca vacía `raw`; y por encima del tope por comida no toca nada.
    Idempotente y fail-safe. tooltip-anchor: P1-RAW-LINEA-MUERTA
    """
    if not RAW_DEAD_LINE_SWEEP_ENABLED:
        return 0
    borradas = 0
    try:
        for dia in days or []:
            for meal in (dia.get("meals") or []) if isinstance(dia, dict) else []:
                if not isinstance(meal, dict):
                    continue
                ings = meal.get("ingredients")
                raw = meal.get("ingredients_raw")
                if not isinstance(ings, list) or not isinstance(raw, list) or not ings or not raw:
                    continue
                vivos, display_incompleto = set(), False
                for linea in ings:
                    if not isinstance(linea, str) or not linea.strip():
                        continue
                    ident = _identidad_de_compra(linea)
                    if ident is None:
                        display_incompleto = True
                        break
                    vivos.add(ident)
                if display_incompleto or not vivos:
                    continue
                muertas = []
                for idx, linea in enumerate(raw):
                    if not isinstance(linea, str) or not linea.strip():
                        continue
                    ident = _identidad_de_compra(linea)
                    if ident is not None and ident not in vivos:
                        muertas.append((idx, linea, ident))
                if not muertas:
                    continue
                if len(muertas) > RAW_DEAD_LINE_SWEEP_MAX_PER_MEAL:
                    logger.warning(
                        f"[P1-RAW-LINEA-MUERTA] «{str(meal.get('name'))[:40]}»: {len(muertas)} "
                        f"líneas parecen muertas (tope {RAW_DEAD_LINE_SWEEP_MAX_PER_MEAL}) — eso "
                        f"acusa al resolvedor, no a la receta: no se toca nada")
                    continue
                fuera = {idx for idx, _, _ in muertas}
                nuevo = [linea for idx, linea in enumerate(raw) if idx not in fuera]
                if not nuevo:
                    continue
                meal["ingredients_raw"] = nuevo
                borradas += len(muertas)
                for _, linea, ident in muertas:
                    logger.info(
                        f"🧹 [P1-RAW-LINEA-MUERTA] «{str(meal.get('name'))[:40]}»: {linea!r} "
                        f"compraba {ident!r} y la receta ya no lo menciona → fuera de la lista")
        return borradas
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-RAW-LINEA-MUERTA] no-op: {type(e).__name__}: {e}")
        return 0


__all__ = ["_barrer_lineas_muertas_de_raw", "_identidad_de_compra",
           "RAW_DEAD_LINE_SWEEP_ENABLED", "RAW_DEAD_LINE_SWEEP_MAX_PER_MEAL"]
