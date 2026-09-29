"""[P1-PLAN-LOTE-852 · 2026-09-29] La lista de compras de un país beta no lleva productos del súper de RD.

LO QUE SE MIDIÓ (validación beta G24 del 29-sep, 6 planes reales con el código de P1-PLAN-LOTE-815):
entre 14 y 20 alimentos de cada lista beta (ES/US/MX/PR/CO) llevaban `brand_product_id` —un producto
de `supermarket_products`, el supermercado DOMINICANO artificial— con su envase: «1 funda (1 Lb) de
Sal», «1 funda (Selecto 1 Lb) de Arroz blanco», «1 paquete (12 Oz) de Quinoa», y en Colombia
«1 tarro (1 Lb (454 gr) · Sosua) de Queso ricotta» con la marca dominicana a la vista (el saneador de
marcas de P1-BETA-PRICE-LEAKS busca el separador dentro de UN paréntesis y este rótulo trae dos
anidados). Entre 17 y 22 llevaban además `market_pkg_price_rd`, el precio en pesos del envase elegido:
`beta_no_prices` anulaba `estimated_cost_rd`, no éste. Esos envases eran los que metían libras, onzas
y la palabra «funda» en las listas de ES, MX y CO.

LA CAUSA. `get_shopping_list_delta` pedía las marcas default del súper (`fetch_brand_default_packages`,
P1-BRAND-LIST-VISIBILITY) y las preferencias del usuario (`fetch_brand_pref_packages`,
P1-SUPERMARKET-COSTING) para TODA lista, y el agregador las ponía encima del `market_packages` del
catálogo. El país de la lista ya viajaba por contexto desde el sello del plan (`envase_pais`, lote
790), pero ese paso no lo miraba.

LO QUE CAMBIA. Con el país de la lista distinto de RD:
  1. no se consultan ni las marcas default ni las preferencias: el envase sale del catálogo
     (`market_container`/`container_weight_g`/`market_packages`, lotes 790/791), igual que el de
     cualquier alimento sin producto en el súper;
  2. al final del agregador se quitan de cada ítem `brand_product_id` y `market_pkg_price_rd`. El
     segundo también lo pone el `market_packages` PROPIO del catálogo (filas dominicanas con precio RD:
     Yogurt 1,96 kg, Nueces, Habichuelas negras en lata). El precio se sigue usando DENTRO del
     agregador para elegir el tamaño (cuántos envases), como hasta hoy; lo que se quita es el dato en
     la lista persistida, que ningún usuario beta puede leer como precio de su mercado.
  3. en una lista MÉTRICA (ES/MX/CO), la talla imperial del envase del catálogo elegido («35.2 oz»,
     «1 lb seco») se reescribe en g/kg/ml DESPUÉS de elegirlo (ver el bloque de abajo).
La lista dominicana (RD o plan sin sello) y las superficies sin plan (chat, swap, scripts: país
`None`) quedan idénticas.

Palancas: `MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS` (1 y 2) y `MEALFIT_BETA_METRIC_PACKAGE_LABELS` (3),
default on. Apagadas, la conducta anterior.
Doc: docs/envases_y_catalogo_por_pais.md (sección del lote 852). Test: tests/test_p1_plan_lote_852.py.
Vive fuera de `shopping_calculator.py` porque éste está en su tope de líneas.
tooltip-anchor: P1-PLAN-LOTE-852
"""
from __future__ import annotations

import logging
import re

from envase_pais import pais_de_la_lista
from knobs import _env_bool

logger = logging.getLogger(__name__)

#: Campos del ítem que sólo pueden venir del súper dominicano (producto elegido y su precio en pesos).
CAMPOS_DEL_SUPER_RD = ("brand_product_id", "market_pkg_price_rd")


def _activo() -> bool:
    return _env_bool("MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS", True)


def super_rd_en_la_lista() -> bool:
    """¿La lista que se está construyendo puede llevar productos del súper dominicano?

    Sí en RD y cuando nadie fijó el país (superficies sin plan): la conducta de siempre. No en un país
    beta con la palanca encendida. Nunca revienta: ante cualquier duda, la conducta de siempre."""
    try:
        pais = pais_de_la_lista()
        if pais is None or pais == "DO":
            return True
        return not _activo()
    except Exception:
        return True


def quitar_super_rd(items) -> int:
    """Quita `brand_product_id` y `market_pkg_price_rd` de los ítems de una lista beta. Devuelve
    cuántos ítems tocó. Corre al final del agregador, cuando el costeo interno ya terminó."""
    if super_rd_en_la_lista():
        return 0
    tocados = 0
    for it in items or []:
        try:
            if not isinstance(it, dict):
                continue
            quitado = False
            for campo in CAMPOS_DEL_SUPER_RD:
                if campo in it:
                    del it[campo]
                    quitado = True
            tocados += quitado
        except Exception:
            # Corre por cada ítem: una excepción aquí no puede tumbar la lista.
            continue
    if tocados:
        logger.debug("[P1-PLAN-LOTE-852] lista de %s: %d ítem(s) sin producto ni precio del súper de RD",
                     pais_de_la_lista(), tocados)
    return tocados


# ── La talla del envase del catálogo, en el sistema del país ─────────────────────────────────────
#
# Quitado el súper, el envase de la lista beta sale del `market_packages` del catálogo. Esos envases
# también son del súper dominicano (`price_source='nacional_tienda'`, Supermercado Nacional) y 109 de
# sus rótulos hablan en libras y onzas: el replay de G24 dejaba «1 paquete (35.2 oz) de Harina de maíz
# precocida» en Colombia, donde antes el súper decía «500 gr». En una lista MÉTRICA (ES/MX/CO, la
# misma pregunta que el rótulo derivado del lote 790: `envase_pais._lista_metrica`) se reescribe la
# TALLA: «1 lb seco» → «454 g seco», «32 oz» de un cartón → «946 ml», «Amarilla 3 Lb» → «Amarilla
# 1,4 kg». El resto del rótulo se queda (variedad, «seco», «lata»). US/PR hablan en libras y onzas: no
# se tocan.
#
# Se hace DESPUÉS de elegir el envase, sobre el ítem ya construido: la cuenta de envases, el costeo y
# la conversión a seco de las legumbres (`envase_legumbre`, que LEE el rótulo «1 lb seco») no ven
# nada distinto. Esto toca la decisión pendiente n.º 4 del lote 790 («las etiquetas lb/oz de productos
# reales siguen igual en ES/MX»): en un país beta ese producto no está en su estante. Palanca propia,
# `MEALFIT_BETA_METRIC_PACKAGE_LABELS` (default on), para que el dueño pueda volver a la regla de 790
# sin perder lo demás.

_G_POR_OZ = 28.3495
_ML_POR_OZ_FLUIDA = 29.5735
_G_POR_LB = 453.592
_ENVASES_LIQUIDOS = frozenset({"botella", "botellas", "carton", "cartón", "cartones", "brik", "briks", "tetra",
                               "galón", "galon", "galones"})
# Un número suelto con su unidad imperial. El lookbehind deja fuera «80/20 Lb» (proporción magro/grasa).
_TALLA_IMPERIAL_RX = re.compile(r"(?<![/\d.,])(\d+(?:[.,]\d+)?)\s*(onzas?|onz|oz|lbs?|libras?)\b", re.IGNORECASE)


def talla_en_metrico(rotulo, unidad=None) -> str:
    """Reescribe en g/kg (o ml/L si el envase es de líquido y la unidad es onza) cada talla imperial del
    rótulo, token a token. Lo que no es una talla imperial queda byte a byte."""
    if not isinstance(rotulo, str) or not rotulo:
        return rotulo
    liquido = str(unidad or "").strip().lower() in _ENVASES_LIQUIDOS

    def _uno(m):
        try:
            qty = float(m.group(1).replace(",", "."))
        except ValueError:
            return m.group(0)
        u = m.group(2).lower()
        from envase_pais import _peso_metrico, _volumen_metrico
        if u.startswith(("oz", "onz")):
            return _volumen_metrico(qty * _ML_POR_OZ_FLUIDA) if liquido else _peso_metrico(qty * _G_POR_OZ)
        return _peso_metrico(qty * _G_POR_LB)

    return _TALLA_IMPERIAL_RX.sub(_uno, rotulo)


def _talla_metrica_activa() -> bool:
    try:
        pais = pais_de_la_lista()
        if pais is None or pais == "DO" or not _env_bool("MEALFIT_BETA_METRIC_PACKAGE_LABELS", True):
            return False
        from envase_pais import _lista_metrica
        return _lista_metrica()
    except Exception:
        return False


def talla_del_catalogo_en_su_sistema(items) -> int:
    """En una lista métrica beta, la talla imperial del envase elegido (`sku_size_label`) pasa a g/kg/ml
    en el ítem y en su `display_qty`/`display_string` (el rótulo va entre paréntesis tras la cuenta:
    «(8 oz c/u)»). Devuelve cuántos ítems tocó. Nunca revienta."""
    if not _talla_metrica_activa():
        return 0
    tocados = 0
    for it in items or []:
        try:
            if not isinstance(it, dict):
                continue
            rotulo = it.get("sku_size_label")
            nuevo = talla_en_metrico(rotulo, it.get("market_unit"))
            if not isinstance(rotulo, str) or nuevo == rotulo:
                continue
            it["sku_size_label"] = nuevo
            for campo in ("display_qty", "display_string"):
                if isinstance(it.get(campo), str):
                    it[campo] = it[campo].replace(f"({rotulo}", f"({nuevo}", 1)
            tocados += 1
        except Exception:
            continue
    return tocados


def sanear_lista_beta(items) -> int:
    """Lo que corre al final del agregador: sin campos del súper de RD y con la talla en su sistema."""
    return quitar_super_rd(items) + talla_del_catalogo_en_su_sistema(items)
