# -*- coding: utf-8 -*-
"""[P1-CANDIDATO-CON-PRECIO · 2026-09-09] El precio entra en la ELECCIÓN del plato, no sólo en la factura.

## De dónde sale

Medido hoy sobre planes vivos: al presupuesto `low` el prompt le dice «evita cortes premium,
mariscos caros» —y el modelo eligió **Cangrejo 2 lb = RD$958**, el 16 % de la compra—. Es el mismo
resultado que ya había dado la otra instrucción del prompt («elige EXACTAMENTE uno del catálogo»:
0 de 12 platos, tres planes seguidos). **Pedir no es imponer.**

Y la política YA sabe el presupuesto: `effective["budget"]` trae `tier`, `floor_dop`, `mode` y
`status` desde la Fase 2. Lo que no lo sabía era **quien elige los platos** — `horizon.py` no
nombraba `budget` ni una vez. Este módulo es el puente, y por eso vive aparte de los dos:
`dish_registry` es la COCINA y el precio es del MERCADO (I16), igual que `catalog_capability`.

## Lo medido antes de escribir una línea (179 plantillas DO)

```
cobertura ...... 154/175 con precio para el 100 % de sus constituyentes (88 %)
dispersión ..... min 13,44 · mediana 58,99 · max 156,79 RD$/ración   (2,7× la mediana)
la mordida ..... día más barato posible RD$101 → 30 d = RD$ 3.043
                 día más caro   posible RD$585 → 30 d = RD$17.565
                 suelo 'low' del presupuesto mensual  = RD$13.650
```

Elegir barato o elegir caro son **5,8×**, y el rango cruza el suelo del presupuesto por los dos
lados. La elección del plato no es un factor más del coste: es EL factor.

## Dos reglas que no son de estilo

**1. Una plantilla con algún constituyente sin precio NO tiene coste: tiene `None`.** Sumar sólo lo
que sí tiene precio produce un total que SUBESTIMA, y un total subestimado asciende en el ranking
justo al plato que nadie sabe costear. El modo de fallo queda invertido: el filtro promovería lo
que no puede ver. `None` se trata como «no sé», y lo que no se sabe no se poda.

**2. La poda es RELATIVA y nunca vacía.** El coste que se calcula aquí es el de los GRAMOS CRUDOS a
`price_per_lb` —coste marginal del ingrediente, no lo que se paga en caja: el empaque redondea
hacia arriba, y encima van básicos, condimentos y el multiplicador del hogar—. Convertirlo en un
tope absoluto contra el presupuesto exigiría un factor de calibración que hoy nadie ha medido. Lo
que sí es sólido es el ORDEN entre candidatos, así que se recorta la cola cara de cada conjunto y
se conserva el orden por hash de `template_candidates` entre los supervivientes —la variedad de
`ARQ27-P1-04` intacta—. *Un filtro que descarta todo es ciego, no preciso*: si la poda dejara menos
de `_MINIMO_SUPERVIVIENTES`, se queda con los más baratos y no vacía nunca.

El coste NO viaja en el dict del candidato: los candidatos se fijan al run y entran en `slice_hash`
→ `input_hash`, así que meter ahí un precio ataría la huella de un plan al cron de inflación. Se
calcula, se poda y se descarta.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from knobs import _env_bool, _env_float, _env_int

logger = logging.getLogger(__name__)

G_POR_LB = 453.59237

#: Por debajo de esto la poda no se aplica: recortar un conjunto ya pequeño lo deja sin variedad
#: y empuja al selector a repetir plato. tooltip-anchor: _MINIMO_SUPERVIVIENTES (test_p1_candidato_con_precio.py)
_MINIMO_SUPERVIVIENTES = 3


# ── Knobs ────────────────────────────────────────────────────────────────────
def price_filter_enabled() -> bool:
    """Interruptor maestro. Default ON: la conducta previa está MEDIDA como dañina (2 de 4 planes
    vivos entre 16 % y 30 % por encima del suelo `low`). Rollback sin redeploy."""
    return _env_bool("MEALFIT_CANDIDATE_PRICE_FILTER", True)


def _keep_low() -> float:
    return _env_float("MEALFIT_CANDIDATE_PRICE_KEEP_LOW", 0.60, validator=lambda v: 0.1 <= v <= 1.0)


def _keep_medium() -> float:
    return _env_float("MEALFIT_CANDIDATE_PRICE_KEEP_MEDIUM", 0.85, validator=lambda v: 0.1 <= v <= 1.0)


def _minimo() -> int:
    return _env_int("MEALFIT_CANDIDATE_PRICE_MIN_SURVIVORS", _MINIMO_SUPERVIVIENTES,
                    validator=lambda v: 1 <= v <= 50)


# ── Precios ──────────────────────────────────────────────────────────────────
def precio_por_lb(fila: Any) -> Optional[float]:
    """RD$ por libra de una fila de `master_ingredients`, o `None` si no se puede afirmar.

    El peso del que cuelga un `price_per_unit` es el del envase si lo hay, y si no el de la unidad
    natural (`density_g_per_unit`): un aguacate se vende POR PIEZA. Olvidar la segunda rama tachaba
    de «sin precio» a 18 plantillas que sí se pueden costear —medido—, y como una plantilla sin
    precio no se poda, el efecto era desactivar el filtro justo donde más falta hace.
    """
    if not isinstance(fila, dict):
        return None
    try:
        ppl = fila.get("price_per_lb")
        if ppl is not None and float(ppl) > 0:
            return float(ppl)
        ppu = fila.get("price_per_unit")
        if ppu is None or float(ppu) <= 0:
            return None
        for k in ("container_weight_g", "density_g_per_unit"):
            g = fila.get(k)
            if g is not None and float(g) > 0:
                return float(ppu) / (float(g) / G_POR_LB)
    except (TypeError, ValueError):
        return None
    return None


def tabla_de_precios(catalogo: Any = None) -> dict:
    """`{clave normalizada → RD$/lb}` del catálogo del mercado. `catalogo` puede venir dado (el
    `deterministic_day` ya lo tiene cargado) o se pide a `get_master_ingredients()`, que está
    cacheado con TTL. Sin catálogo ⇒ `{}` ⇒ nada tiene precio ⇒ no se poda nada."""
    filas: Iterable = ()
    if isinstance(catalogo, dict):
        filas = list(catalogo.values())
    elif isinstance(catalogo, (list, tuple)):
        filas = catalogo
    else:
        try:
            from shopping_calculator import get_master_ingredients
            filas = get_master_ingredients() or ()
        except Exception as e:
            logger.debug(f"[P1-CANDIDATO-CON-PRECIO] catálogo no disponible ({e!r}): sin poda por precio")
            return {}
    out: dict = {}
    for f in filas:
        p = precio_por_lb(f)
        if p is None:
            continue
        for k in (f.get("name"), f.get("slug")):
            if k:
                out.setdefault(str(k).strip().lower(), p)
    return out


def costo_racion(template: Any, precios: dict) -> Optional[float]:
    """RD$ de los INGREDIENTES CRUDOS de una ración, o `None` si algún constituyente no tiene precio.

    `None` no es un fallo: es la respuesta correcta. Ver la regla 1 del docstring del módulo — un
    total parcial subestima, y lo subestimado sube en el ranking.
    """
    if not isinstance(template, dict) or not precios:
        return None
    cons = template.get("constituents") or []
    if not cons:
        return None
    total = 0.0
    for c in cons:
        if not isinstance(c, dict):
            return None
        p = None
        for k in (c.get("canonical"), c.get("name"), c.get("ingredient_id")):
            if k:
                p = precios.get(str(k).strip().lower())
                if p is not None:
                    break
        if p is None:
            return None
        try:
            total += float(p) * (float(c.get("grams") or 0.0) / G_POR_LB)
        except (TypeError, ValueError):
            return None
    return round(total, 2)


# ── Poda ─────────────────────────────────────────────────────────────────────
def fraccion_asequible(tier: Any) -> Optional[float]:
    """Qué fracción del conjunto sobrevive según el nivel de presupuesto. `None` = no podar.

    `high` y `unlimited` no se podan: quien no tiene la restricción no debe perder variedad por
    ella. `custom` va con `medium` — el monto exacto se compara contra el suelo en la Fase 2, y
    duplicar aquí esa aritmética sería escribir la segunda tabla que `P1-DIET-CANON-SSOT` prohíbe.
    """
    t = str(tier or "").strip().lower()
    if t == "low":
        return _keep_low()
    if t in ("medium", "custom"):
        return _keep_medium()
    return None


def poda_por_presupuesto(pares: list, tier: Any, *, minimo: Optional[int] = None) -> list:
    """`pares` = `[(item, costo|None), ...]` en el orden que trae el llamador. Devuelve los items
    supervivientes **en ese mismo orden** — el hash de `ARQ27-P1-04` manda, esto sólo quita.

    Función PURA: ni catálogo, ni DB, ni knobs de encendido. Así se puede probar la frontera (que
    es lo que decide si esto ayuda o estorba) sin montar nada.
    """
    frac = fraccion_asequible(tier)
    if frac is None or not pares:
        return [it for it, _ in pares]
    minimo = int(minimo if minimo is not None else _minimo())
    # Se trabaja con ÍNDICES, no con los objetos: dos candidatos pueden ser dicts iguales y
    # `in`/`id()` los confundiría. El índice es la identidad honesta dentro de esta lista.
    con_precio = [(i, c) for i, (_, c) in enumerate(pares) if c is not None]
    if len(con_precio) < minimo:
        return [it for it, _ in pares]          # tan pocos costeados que podar es opinar sobre ruido
    cupo = max(minimo, int(round(len(con_precio) * float(frac))))
    if cupo >= len(con_precio):
        return [it for it, _ in pares]
    baratos = {i for i, _ in sorted(con_precio, key=lambda p: p[1])[:cupo]}
    # Lo que no tiene precio sobrevive SIEMPRE (regla 1): no se poda lo que no se sabe.
    return [it for i, (it, c) in enumerate(pares) if c is None or i in baratos]


__all__ = [
    "G_POR_LB", "price_filter_enabled", "precio_por_lb", "tabla_de_precios", "costo_racion",
    "fraccion_asequible", "poda_por_presupuesto",
]
