# backend/food_identity.py
"""[P1-ARQ27-F2-IDENTIDAD · 2026-09-06] La identidad de un alimento NO es su categoría de tienda.

`master_ingredients.category` tiene seis valores —Despensa, Proteínas, Vegetales, Frutas, Lácteos,
Víveres— y son **el pasillo del supermercado**, no una afirmación sobre lo que el alimento es. Cinco
filas de nombre vegetal viven en «Lácteos» porque ahí es donde se compran:

    Leche de almendras · Leche de avena · Leche de coco · Leche de soya · Yogur de coco

El motor ya lo hace bien —el guard de dieta resuelve por CONSTITUYENTES desde `ARQ27-P0-01`, y las
clases de alérgeno salen del nombre: «Leche de almendras» → `frutos secos`, «Leche de coco» → `[]`,
«Queso blanco» → `lacteos, lactosa`. Medido, 11 de 11 correctos—. El hueco estaba **fuera**: la
proyección pública de `/api/catalog` mandaba `category` y **nada más con lo que decidir**, así que
cualquier consumidor que quisiera saber «¿puede comerse esto un vegano?» tenía únicamente el campo
que dice `Lácteos` para la leche de coco.

Este módulo no decide nada nuevo: **compone las dos autoridades que ya existen** y las publica. Esa
es toda la gracia — una tercera tabla de «qué es vegano» derivaría de las otras dos en cuanto alguien
la editara, que es la lección que este repo ya pagó con `canonicalize_diet_type` y con
`pantry_names_match`.

## La confianza del dato nutricional

El criterio de cierre pide que «quesos, preparados y fortificados conserven incertidumbres de receta
o etiqueta». No hace falta inventar la señal: está en el catálogo.

- `referenced` — `nutrition_source` externa (usda/bedca/latinfoods) con un `fdc_id` propio.
- `proxy` — el `fdc_id` lo comparte con otras filas: la cifra describe OTRO alimento. **Hoy no hay
  ninguno** —medido 2026-09-06: 288 filas con id, las 288 distintas—, así que esta rama es un guard
  contra regresión, no un retrato del catálogo actual. Existe porque el 19-ago sí lo era:
  `P1-BEDCA-DEPROXY-ES` encontró 47 de 347 compartiendo id, uno de ellos sustituyendo a SIETE
  embutidos distintos (Sobrasada: 595 kcal, no 296). Nada re-validaba la procedencia; ahora, si
  vuelve a pasar, el dato lo dice en vez de esperar a otra auditoría.
- `curated` — `nutrition_source='manual'` (42 filas): una estimación del equipo, no una medición.
  Un «queso de hoja» o un «salami de pavo» varían por marca y por receta; decir que su fósforo es un
  número medido sería afirmar de más.

Ninguna de las tres bloquea nada por sí sola. Lo que hacen es dejar de presentar una estimación con
la misma cara que una medición.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: Fuentes con una referencia externa comprobable. `manual` NO está: es la marca de lo curado.
_FUENTES_REFERENCIADAS = frozenset({"usda", "bedca", "latinfoods", "fndds"})

CONFIANZA_REFERENCIADA = "referenced"
CONFIANZA_PROXY = "proxy"
CONFIANZA_CURADA = "curated"
CONFIANZA_DESCONOCIDA = "unknown"


def fdc_ids_compartidos(rows: Iterable[dict]) -> frozenset:
    """Los `fdc_id` que más de una fila reclama. Un id compartido es un PROXY: la cifra que trae
    describe un alimento distinto del que la enseña."""
    vistos, repetidos = set(), set()
    for r in rows or []:
        fid = (r or {}).get("fdc_id")
        if fid in (None, "", 0):
            continue
        fid = str(fid)
        (repetidos if fid in vistos else vistos).add(fid)
    return frozenset(repetidos)


def confianza_nutricional(row: dict, compartidos: Optional[frozenset] = None) -> str:
    """`referenced` / `proxy` / `curated` / `unknown` para UNA fila del catálogo.

    El orden importa: un `fdc_id` compartido gana sobre la fuente, porque una fila `usda` cuyo id
    apunta a otro alimento es exactamente el caso que `P1-BEDCA-DEPROXY-ES` encontró — auditar los
    ids duplicados no ve el único MAL apuntado, pero sí ve los compartidos."""
    fuente = str((row or {}).get("nutrition_source") or "").strip().lower()
    fid = (row or {}).get("fdc_id")
    if compartidos and fid not in (None, "", 0) and str(fid) in compartidos:
        return CONFIANZA_PROXY
    if fuente == "manual":
        return CONFIANZA_CURADA
    if fuente in _FUENTES_REFERENCIADAS:
        return CONFIANZA_REFERENCIADA
    return CONFIANZA_DESCONOCIDA


def _prohibido_para(nombre: str, dieta: str) -> Optional[bool]:
    """`_diet_pool_item_banned` es el ÚNICO SSOT de qué prohíbe una dieta (P1-DIET-CANON-SSOT). Se
    importa perezosamente: `graph_orchestrator` es enorme y este módulo lo usa un endpoint."""
    try:
        from graph_orchestrator import _diet_pool_item_banned
        return bool(_diet_pool_item_banned(nombre, dieta))
    except Exception as e:
        logger.debug(f"[P1-ARQ27-F2-IDENTIDAD] guard de dieta no disponible: {e!r}")
        return None


def _clases_alergeno(nombre: str) -> Optional[list]:
    try:
        from dish_registry import allergen_classes_for
        return list(allergen_classes_for([nombre]) or [])
    except Exception as e:
        logger.debug(f"[P1-ARQ27-F2-IDENTIDAD] clases de alérgeno no disponibles: {e!r}")
        return None


def identidad(row: dict, compartidos: Optional[frozenset] = None) -> dict:
    """La identidad DIETARIA de una fila del catálogo, derivada de los SSOT que ya deciden.

    `None` en un campo significa **no se pudo determinar**, y eso NO es lo mismo que `False` — la
    invariante que el roadmap 2.6 llama I20 y que este repo ya pagó dos veces (`int(x or -1)` con
    `attempts=0`, y el nutriente ausente tratado como cero). Un consumidor que reciba `null` debe
    preguntar, no asumir que se puede comer.
    """
    nombre = str((row or {}).get("name") or "").strip()
    if not nombre:
        return {}
    veg = _prohibido_para(nombre, "vegana")
    vgt = _prohibido_para(nombre, "vegetariana")
    clases = _clases_alergeno(nombre)
    return {
        # `None` = indeterminado. Nunca se colapsa a False.
        "vegan_ok": (None if veg is None else not veg),
        "vegetarian_ok": (None if vgt is None else not vgt),
        "allergen_classes": clases,
        "nutrition_confidence": confianza_nutricional(row, compartidos),
        # Dicho explícitamente para que nadie vuelva a leer `category` como verdad dietaria:
        # es el pasillo de la tienda, y por eso «Leche de coco» aparece en «Lácteos».
        "category_is_presentation": True,
    }


def anotar_catalogo(rows: list) -> list:
    """Añade `diet` a cada fila. Aditivo: no toca ni renombra nada de lo que ya iba.

    Los `fdc_id` compartidos se calculan UNA vez sobre el catálogo entero — fila a fila es imposible
    saber si un id está compartido, que es justo por lo que la señal faltaba."""
    try:
        compartidos = fdc_ids_compartidos(rows)
        for r in rows or []:
            if isinstance(r, dict):
                r["diet"] = identidad(r, compartidos)
    except Exception as e:
        logger.warning(f"[P1-ARQ27-F2-IDENTIDAD] no se pudo anotar el catálogo: {e!r}")
    return rows


__all__ = ["identidad", "anotar_catalogo", "confianza_nutricional", "fdc_ids_compartidos",
           "CONFIANZA_REFERENCIADA", "CONFIANZA_PROXY", "CONFIANZA_CURADA", "CONFIANZA_DESCONOCIDA"]
