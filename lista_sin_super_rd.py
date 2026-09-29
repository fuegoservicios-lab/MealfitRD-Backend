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

Palancas: `MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS` (1, 2 y 3) y `MEALFIT_BETA_METRIC_PACKAGE_LABELS`
(sólo 3), default on. Apagar la primera devuelve la lista anterior byte a byte, talla incluida;
apagar la segunda deja los rótulos del catálogo tal cual y mantiene 1 y 2.
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
# `MEALFIT_BETA_METRIC_PACKAGE_LABELS` (default on), SUBORDINADA a la del súper: sólo actúa con
# `MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS` encendida, así que apagar ésa devuelve la lista de antes
# byte a byte (ronda 1 de revisión: la de tallas seguía viva y reescribía «Selecto 1 Lb» → «Selecto
# 454 g», un rótulo de producto real, lo que P1-UNIT-SYSTEM-BY-COUNTRY prohíbe). Y jamás toca un ítem
# con `brand_product_id`: eso es un producto del súper, no un envase del catálogo.
#
# DE DÓNDE SALE EL NÚMERO (ronda 1). La onza del rótulo es ambigua (de peso: 28,35 g; fluida: 29,57 ml)
# y el tamaño de ese envase ya está medido en `package_grams`, que es lo que guarda la Nevera
# (`/restock`). Si una talla del rótulo describe esos gramos (±3 % con el factor de peso o con el de
# volumen; gana el que más cuadra) se pinta DESDE `package_grams` y el factor decide g o ml: «Mostaza
# 8 oz» (227 g) → «227 g», no «237 ml»; «Salsa de soya 10 oz» (295) → «295 ml», no «283 g»; «Ajo en
# polvo 3 Oz» (85 g) → «85 g», no ml. No se usa `envase_pais._etiqueta_metrica_envase` para decidir la
# clase porque decide por el ENVASE («botella» ⇒ ml) y ése era justo el error de la mostaza; sí se usan
# sus formateadores (`_peso_metrico`/`_volumen_metrico`), que son el SSOT del formato. Si no cuadra
# ninguno —la legumbre «1 lb seco» pesa 1135 g COCIDA; «48 oz» de aceite, 1300— se convierte el número
# del rótulo (líquido si el envase lo es). A ≤0,5 % de un kilo/litro entero se redondea: «35.2 oz» →
# «1 kg», no «998 g».
#
# UN ENVASE QUE ES UNA UNIDAD DE PESO (ronda 1). Chicharrón, Pernil, Tocineta y Gallina criolla se
# venden «por libra» (`unit='libra'`, rótulo «1 lb»): la talla repite la cantidad y, tras la proyección
# de P1-UNIT-SYSTEM-BY-COUNTRY, la línea decía «5 kg (454 g c/u)». Si la talla es UNA unidad de ese
# peso (±3 %), el paréntesis se quita del texto; `sku_size_label` conserva la talla.

_G_POR_OZ = 28.3495
_ML_POR_OZ_FLUIDA = 29.5735
_G_POR_LB = 453.592
_TOLERANCIA_TALLA = 0.03
_TOLERANCIA_KILO = 0.005
_ENVASES_LIQUIDOS = frozenset({"botella", "botellas", "carton", "cartón", "cartones", "brik", "briks", "tetra",
                               "galón", "galon", "galones"})
# Un envase que ES una unidad de peso: la talla repite la cantidad («5 kg (454 g c/u)»).
_ENVASES_DE_PESO = frozenset({"lb", "lbs", "libra", "libras", "oz", "onza", "onzas"})
# Un número suelto con su unidad imperial. El lookbehind deja fuera «80/20 Lb» (proporción magro/grasa).
_TALLA_IMPERIAL_RX = re.compile(r"(?<![/\d.,])(\d+(?:[.,]\d+)?)\s*(onzas?|onz|oz|lbs?|libras?)\b", re.IGNORECASE)


def _kilo_redondo(valor: float) -> float:
    """997,9 → 1000: a ≤0,5 % de un kilo (o litro) entero, el entero. El resto, tal cual."""
    enteros = round(valor / 1000.0)
    if enteros >= 1 and abs(valor - enteros * 1000.0) <= _TOLERANCIA_KILO * enteros * 1000.0:
        return enteros * 1000.0
    return valor


def _pintar(clase: str, valor: float) -> str:
    from envase_pais import _peso_metrico, _volumen_metrico
    valor = _kilo_redondo(valor)
    return _volumen_metrico(valor) if clase == "ml" else _peso_metrico(valor)


def _gramos_validos(gramos):
    try:
        g = float(gramos)
    except (TypeError, ValueError):
        return None
    return g if g > 0 else None


def talla_en_metrico(rotulo, unidad=None, gramos=None) -> str:
    """Reescribe en g/kg (o ml/L) cada talla imperial del rótulo, token a token. Con `gramos` (el
    `package_grams` del envase elegido) y una talla que los describe, pinta desde esos gramos con la
    clase del factor que cuadró; si no, convierte el número del rótulo. Lo que no es una talla
    imperial queda byte a byte."""
    if not isinstance(rotulo, str) or not rotulo:
        return rotulo
    liquido = str(unidad or "").strip().lower() in _ENVASES_LIQUIDOS
    pkg_g = _gramos_validos(gramos)

    def _uno(m):
        try:
            qty = float(m.group(1).replace(",", "."))
        except ValueError:
            return m.group(0)
        if m.group(2).lower().startswith(("oz", "onz")):
            candidatos = [("g", qty * _G_POR_OZ), ("ml", qty * _ML_POR_OZ_FLUIDA)]
            por_rotulo = candidatos[1] if liquido else candidatos[0]
        else:
            candidatos = [("g", qty * _G_POR_LB)]
            por_rotulo = candidatos[0]
        if pkg_g:
            error, clase = min((abs(valor - pkg_g) / pkg_g, clase) for clase, valor in candidatos)
            if error <= _TOLERANCIA_TALLA:
                return _pintar(clase, pkg_g)
        return _pintar(*por_rotulo)

    return _TALLA_IMPERIAL_RX.sub(_uno, rotulo)


def _talla_metrica_activa() -> bool:
    try:
        pais = pais_de_la_lista()
        if pais is None or pais == "DO" or not _activo():
            return False
        if not _env_bool("MEALFIT_BETA_METRIC_PACKAGE_LABELS", True):
            return False
        from envase_pais import _lista_metrica
        return _lista_metrica()
    except Exception:
        return False


def _es_una_unidad_de_peso(it, gramos) -> bool:
    unidad = str(it.get("market_unit") or "").strip().lower().rstrip(".")
    if unidad not in _ENVASES_DE_PESO or not gramos:
        return False
    una = _G_POR_OZ if unidad.startswith(("oz", "onza")) else _G_POR_LB
    return abs(gramos - una) / una <= _TOLERANCIA_TALLA


def _sin_talla_repetida(texto: str, talla: str) -> str:
    """«11 libras (454 g c/u) de X» → «11 libras de X»: el envase ES la unidad de peso."""
    for sufijo in (f" ({talla} c/u)", f" ({talla})"):
        if sufijo in texto:
            return texto.replace(sufijo, "", 1)
    return texto


def talla_del_catalogo_en_su_sistema(items) -> int:
    """En una lista métrica beta, la talla imperial del envase elegido (`sku_size_label`) pasa a g/kg/ml
    en el ítem y en su `display_qty`/`display_string` (el rótulo va entre paréntesis tras la cuenta:
    «(8 oz c/u)»). Nunca toca un producto del súper (`brand_product_id`). Devuelve cuántos ítems tocó.
    Nunca revienta."""
    if not _talla_metrica_activa():
        return 0
    tocados = 0
    for it in items or []:
        try:
            if not isinstance(it, dict) or it.get("brand_product_id"):
                continue
            rotulo = it.get("sku_size_label")
            gramos = _gramos_validos(it.get("package_grams"))
            nuevo = talla_en_metrico(rotulo, it.get("market_unit"), gramos)
            if not isinstance(rotulo, str) or nuevo == rotulo:
                continue
            it["sku_size_label"] = nuevo
            repetida = _es_una_unidad_de_peso(it, gramos)
            for campo in ("display_qty", "display_string"):
                if isinstance(it.get(campo), str):
                    texto = it[campo].replace(f"({rotulo}", f"({nuevo}", 1)
                    it[campo] = _sin_talla_repetida(texto, nuevo) if repetida else texto
            tocados += 1
        except Exception:
            continue
    return tocados


def sanear_lista_beta(items) -> int:
    """Lo que corre al final del agregador: la talla en su sistema (ANTES de quitar `brand_product_id`,
    para que la defensa «no tocar un producto del súper» vea el campo) y sin campos del súper de RD."""
    tallas = talla_del_catalogo_en_su_sistema(items)
    return quitar_super_rd(items) + tallas
