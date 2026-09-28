"""[P1-PLAN-LOTE-790 · 2026-09-28] El envase y el catálogo hablan el idioma del PAÍS de la lista.

Dos defectos de la lista de compras de los países beta, medidos con el agregador real sobre la copia
de `master_ingredients` (replay determinista, sin DB ni IA — auditoría G58/G13 del 28-sep):

(a) EL RÓTULO DEL ENVASE (G58, parte de código). `_sku_size_label` nació para el súper DOMINICANO:
    100 g → «¼ lb», 1 kg → «2.2 lbs», un pote de 454 g → «16 oz». Y terminaba en `int(size_g)`, así
    que el sobre de azafrán de 0,4 g salía «1 sobre (0g)». La proyección métrica de
    P1-UNIT-SYSTEM-BY-COUNTRY convierte la CANTIDAD que se pesa («1 lb» → «454 g») y deja el
    paréntesis a propósito, porque ahí también viven rótulos de productos reales. Resultado: aun con
    los envases curados (lote 791), un español leería «1 paquete (¼ lb)».

    Lo que cambia: el rótulo que ESTE código DERIVA del peso del envase (`container_weight_g` /
    `available_sizes_g`) sale en el sistema del país de la lista — ES/MX/CO en g/kg/ml/L, DO/US/PR
    exactamente como hoy — y bajo 1 g lleva decimales en todos («0,4 g» / «0.4g»). Lo que NO cambia:
    la etiqueta de un envase REAL (`market_packages[].label`, «5 Oz · Genérico», «Selecto 1 Lb ·
    Wala») — convertirla sería falsificar el rótulo de un producto del estante, la misma regla que
    fijó P1-UNIT-SYSTEM-BY-COUNTRY. La palanca es la MISMA (`MEALFIT_UNIT_SYSTEM_BY_COUNTRY`): una
    lista no puede quedar mitad en libras y mitad en gramos.

(b) LA FUGA DE CATÁLOGO ENTRE PAÍSES. El keep del agregador (P1-COUNTRY-SYSTEM-F2 T5) pregunta a
    `is_country_catalog_unpriced_item` SIN país, o sea a la UNIÓN de los seis: el plan español rd587
    llevó «Tortilla de maíz» (sólo MX) a su lista como si fuera comida de su mercado. El catálogo
    verificado del generador ya pregunta por país (P1-COUNTRY-CATALOG-BY-COUNTRY): la tortilla no se
    le OFRECIÓ al modelo, la escribió igual. La lista es la última red, no la primera.

    DECISIÓN sobre el ítem ajeno: se QUEDA en la lista y se SELLA. `catalogo_de_otro_pais=[<países>]`
    en el ítem (consultable por SQL sobre `plan_data`) + un WARN grep-able por lista. No se dropea:
      1. la receta lo sigue pidiendo; quitarlo deja una lista incompleta sin aviso al usuario — el
         miedo explícito del dueño (P1-VERIFIED-ONLY-OBSERVABILITY);
      2. el espejo del guard de coherencia (`_survives_shopping_list`) no sabe de países: dropear en
         un lado y no en el otro fabrica una divergencia `expected_only` en cada recálculo;
      3. el bloque de un país NO es la lista de lo que se vende allí, sólo las altas SIN PRECIO que se
         le hicieron. Medido en el replay: el mismo plan español rd587 lleva «Trucha», que sólo
         reclama el bloque de CO — y en España la trucha se vende en cualquier pescadería. Dropear
         por país le habría quitado el pescado de la semana. Por eso el sello se llama «catálogo de
         otro país» y no «no existe en tu mercado»: afirma lo que el dato sabe, nada más.
    DO conserva el predicado de siempre (la unión, sin sello) — mismo criterio que el catálogo del
    generador (`_vc_beta = _vc_country != "DO"`): byte-identidad de la lista dominicana. Knob
    `MEALFIT_COUNTRY_CATALOG_FOREIGN_FLAG` (default on).
    [Revisión ronda 1] Los dos falsos positivos medidos se cierran en el DATO y en el mercado: «trucha»
    pasa también al bloque de ES y «chile en polvo» al de MX (se venden allí), y PR↔US cuentan como el
    mismo mercado (`mercado_de`). El sello todavía no lo pinta el frontend: decisión del dueño.

EL PAÍS VIAJA POR CONTEXTO. `aggregate_and_deduct_shopping_list` tiene 26 call sites y ninguno sabe
de países; las dos puertas que reciben el PLAN (`get_shopping_list_delta`, `get_realtime_pantry`)
llevan `@con_pais_del_plan`, que saca el país del SELLO del plan con `constants.country_for_plan` (el
mismo SSOT que ya usa la proyección métrica) y lo deja en un `ContextVar` mientras dura la llamada.
Sin plan (chat, swap, scripts) el contexto es `None` y todo se comporta como hoy.

Vive fuera de `shopping_calculator.py` porque éste está en su tope de líneas (roadmap 2.5 §11).
Doc (qué hace, por qué, lo pendiente del dueño): docs/envases_y_catalogo_por_pais.md
tooltip-anchor: P1-PLAN-LOTE-790
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging

from knobs import _env_bool

logger = logging.getLogger(__name__)

_PAIS_DE_LA_LISTA: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "mealfit_pais_de_la_lista", default=None)


def pais_de_la_lista():
    """País canónico de la lista que se está construyendo, o `None` si nadie lo fijó."""
    return _PAIS_DE_LA_LISTA.get()


@contextlib.contextmanager
def lista_de_pais(pais):
    """Fija el país de la lista durante el bloque. Anidable y a prueba de excepciones."""
    token = _PAIS_DE_LA_LISTA.set(pais)
    try:
        yield pais
    finally:
        _PAIS_DE_LA_LISTA.reset(token)


def _pais_del_plan(plan_result):
    try:
        from constants import country_for_plan
        return country_for_plan(plan_result if isinstance(plan_result, dict) else {}, None)
    except Exception:
        return None


def con_pais_del_plan(fn):
    """Decorador para las funciones que reciben `plan_result`: fija el país de SU lista."""
    firma = inspect.signature(fn)

    @functools.wraps(fn)
    def envuelta(*args, **kwargs):
        try:
            plan = firma.bind_partial(*args, **kwargs).arguments.get("plan_result")
        except TypeError:
            plan = None
        with lista_de_pais(_pais_del_plan(plan)):
            return fn(*args, **kwargs)

    return envuelta


# ── (a) El rótulo del envase ─────────────────────────────────────────────────────────────────────

_G_POR_LB = 453.592
_ENVASES_DE_VOLUMEN = ("cartón", "carton", "botella", "ml", "l", "galón", "envase", "lata")
_ENVASES_LIQUIDOS = ("botella", "ml", "l", "galón")
# Tamaños de volumen conocidos (leche, jugos — se venden por ml, no por peso).
_VOLUMENES_IMPERIAL = {250: "250ml", 473: "473ml", 946: "946ml", 1000: "1L", 1892: "1/2 Galón"}


def _lista_metrica() -> bool:
    pais = pais_de_la_lista()
    if pais is None or not _env_bool("MEALFIT_UNIT_SYSTEM_BY_COUNTRY", True):
        return False
    try:
        from constants import unit_system_for_country
        return unit_system_for_country(pais) == "metric"
    except Exception:
        return False


def _bajo_la_unidad(x: float, coma: bool) -> str:
    """0.4 → '0.4' · 0.38 → '0.38' (dos cifras significativas: el azafrán se vende a 0,38 g)."""
    txt = f"{x:.2g}"
    if "e" in txt:
        txt = f"{x:.3f}".rstrip("0").rstrip(".") or "0"
    return txt.replace(".", ",") if coma else txt


def etiqueta_envase(size_g, unit_hint=None) -> str:
    """Gramos del envase → rótulo legible en el sistema de unidades del país de la lista.

    Sin país (o DO/US/PR): el rótulo dominicano de siempre, byte a byte, salvo bajo 1 g.
    453g → '1 lb', 908g → '2 lbs', 473g → '473ml', 946g → '946ml', 200g → '200g', 0.4g → '0.4g'.
    ES/MX/CO: 454 g, 1 kg, 2,3 kg, 946 ml, 1,9 L, 0,4 g."""
    if size_g is None:
        return ""
    size_g = float(size_g)
    if _lista_metrica():
        return _etiqueta_metrica_envase(size_g, unit_hint)
    return _etiqueta_imperial(size_g, unit_hint)


def _etiqueta_imperial(size_g: float, unit_hint=None) -> str:
    """El cuerpo de `_sku_size_label` anterior a este lote, intacto salvo la última línea."""
    if unit_hint and unit_hint.lower() in _ENVASES_DE_VOLUMEN:
        for vol_g, label in _VOLUMENES_IMPERIAL.items():
            if abs(size_g - vol_g) < 10:
                return label
        # [BOTELLA-ML-FALLBACK] Si el contenedor es una botella/lata pero el peso
        # no matchea ninguno de los tamaños canónicos (e.g. aceite de oliva 500g),
        # NO debemos caer al fallback genérico que produciría "500g". Los líquidos
        # de cocina (aceite, vinagre, salsas) tienen densidad ≈1 g/ml, así que
        # mostrar el mismo número como "ml" es correcto y mucho más legible que
        # "500g" en una botella de aceite (visto 2026-05-06).
        if unit_hint.lower() in _ENVASES_LIQUIDOS:
            if size_g >= 1000:
                # Convertir a litros con un decimal cuando es ≥1L (1500g → "1.5L")
                liters = size_g / 1000
                if abs(liters - round(liters)) < 0.05:
                    return f"{round(liters):d}L"
                return f"{liters:.1f}L"
            return f"{int(round(size_g))}ml"

    if unit_hint and unit_hint.lower() in ("pote", "frasco"):
        # Mapeos típicos de onzas para potes (yogurt, queso crema, aceitunas)
        if abs(size_g - 453.592) < 15: return "16 oz"
        if abs(size_g - 226.796) < 15: return "8 oz"
        if abs(size_g - 340.194) < 15: return "12 oz"

    lbs = size_g / _G_POR_LB
    # Libras enteras limpias — threshold estricto (±2%) para no confundir 473g con 1lb
    if abs(lbs - round(lbs)) < 0.05 and round(lbs) >= 1:
        return f"{round(lbs)} lb" if round(lbs) == 1 else f"{round(lbs)} lbs"
    # Media libra
    if abs(lbs - 0.5) < 0.05:
        return "½ lb"
    if abs(lbs - 0.25) < 0.05:
        return "¼ lb"
    # Mejorar la etiqueta para pesos de mega frutas o porciones grandes (ej. 800g -> ~1.8 lbs)
    if lbs > 1.2:
        return f"{round(lbs, 1):g} lbs"
    # Todo lo demás en gramos. [P1-PLAN-LOTE-790] Bajo 1 g con decimales: `int(0.4)` era «0g».
    if size_g < 1:
        return f"{_bajo_la_unidad(size_g, coma=False)}g"
    return f"{int(size_g)}g"


def _etiqueta_metrica_envase(size_g: float, unit_hint=None) -> str:
    """La MISMA clasificación que la imperial (qué envase es de volumen, qué tamaño es canónico),
    con otras unidades: así un cambio de país cambia el idioma del rótulo, nunca qué se rotula."""
    hint = str(unit_hint or "").lower()
    if hint in _ENVASES_DE_VOLUMEN:
        if hint in _ENVASES_LIQUIDOS or any(abs(size_g - v) < 10 for v in _VOLUMENES_IMPERIAL):
            return _volumen_metrico(size_g)
    return _peso_metrico(size_g)


def _volumen_metrico(ml: float) -> str:
    if ml >= 1000:
        litros = round(ml / 1000.0, 1)
        txt = f"{litros:.1f}".replace(".", ",")
        return f"{txt[:-2] if txt.endswith(',0') else txt} L"
    if ml < 1:
        return f"{_bajo_la_unidad(ml, coma=True)} ml"
    return f"{int(round(ml))} ml"


def _peso_metrico(g: float) -> str:
    if g < 1:
        return f"{_bajo_la_unidad(g, coma=True)} g"
    # El formato de peso métrico ya tiene SSOT (la proyección de P1-UNIT-SYSTEM-BY-COUNTRY): «454 g»,
    # «1,4 kg». Import tardío: `shopping_calculator` importa este módulo al cargar.
    from shopping_calculator import _etiqueta_metrica
    return _etiqueta_metrica(g)


# ── (b) El alimento de catálogo-país de OTRO país ────────────────────────────────────────────────

def _sello_catalogo_de_otro_pais_activo() -> bool:
    return _env_bool("MEALFIT_COUNTRY_CATALOG_FOREIGN_FLAG", True)


def paises_del_alimento(nombre) -> list:
    """Países beta cuyo catálogo sin precio reclama `nombre` (el SSOT de tokens por país)."""
    try:
        import shopping_calculator as sc
        return [cc for cc, toks in sc._COUNTRY_CATALOG_UNPRICED_BY_COUNTRY.items()
                if toks and sc.is_country_catalog_unpriced_item(nombre, country=cc)]
    except Exception:
        return []


# [P1-PLAN-LOTE-790 · 2026-09-28 · revisión ronda 1, defecto 2] Puerto Rico compra en el súper de Estados Unidos
# (la 791 ya usa US como sustituto DECLARADO de PR): lo que sólo reclama el bloque del otro no es
# «de otro país» para el sello. Sin esto, el Chile en polvo de una lista boricua salía sellado `['US']`.
_MISMO_MERCADO = {"PR": frozenset({"PR", "US"}), "US": frozenset({"US", "PR"})}


def mercado_de(pais) -> frozenset:
    """Los países cuyo catálogo sin precio cuenta como el mercado de `pais` para el sello."""
    return _MISMO_MERCADO.get(pais, frozenset({pais}))


def sellar_catalogo_de_otro_pais(items) -> int:
    """Sella `catalogo_de_otro_pais` en los ítems de catálogo-país sin precio que NO son del país de la
    lista (ni de su mismo mercado: PR↔US). Devuelve cuántos selló. Corre al final del agregador."""
    pais = pais_de_la_lista()
    if not pais or pais == "DO" or not _sello_catalogo_de_otro_pais_activo():
        return 0
    try:
        import shopping_calculator as sc
    except Exception:
        return 0
    mercado = mercado_de(pais)
    sellados = []
    for it in items or []:
        try:
            if not isinstance(it, dict):
                continue
            nombre = str(it.get("name") or "")
            if not nombre or not sc.is_country_catalog_unpriced_item(nombre):
                continue
            if any(sc.is_country_catalog_unpriced_item(nombre, country=cc) for cc in mercado):
                continue  # es de SU mercado
            if sc._is_verified_for_shopping(nombre):
                continue  # tiene precio: es comida verificada, no catálogo-país
            ajenos = [cc for cc in paises_del_alimento(nombre) if cc not in mercado]
            if not ajenos:
                continue
            it["catalogo_de_otro_pais"] = ajenos
            sellados.append(f"{nombre}→{'/'.join(ajenos)}")
        except Exception:
            # Corre por cada ítem: una excepción aquí no puede tumbar la lista.
            continue
    if sellados:
        _avisar_fuga(pais, sellados)
    return len(sellados)


# [P1-PLAN-LOTE-790 · 2026-09-28 · revisión ronda 1, defecto 8] Un recálculo llama al agregador 3-9 veces sobre la
# misma lista (semanal, quincenal, mensual, delta): el WARN salía otras tantas y enterraba el caso
# nuevo. Se avisa UNA vez por (país, alimentos sellados) cada hora y por proceso; lo repetido va a
# DEBUG. El SELLO no se deduplica: cada lista lo lleva siempre.
_SELLOS_AVISADOS: "dict[str, float]" = {}
_AVISO_CADA_S = 3600.0
_AVISOS_MAX = 512


def _avisar_fuga(pais, sellados) -> None:
    import time
    firma = f"{pais}|{'|'.join(sorted(sellados))}"
    ahora = time.monotonic()
    try:
        ultimo = _SELLOS_AVISADOS.get(firma)
        if ultimo is not None and ahora - ultimo < _AVISO_CADA_S:
            logger.debug("[P1-PLAN-LOTE-790] (repetido) lista de %s: %s", pais, ", ".join(sellados))
            return
        if len(_SELLOS_AVISADOS) >= _AVISOS_MAX:
            _SELLOS_AVISADOS.clear()
        _SELLOS_AVISADOS[firma] = ahora
    except Exception:
        pass
    logger.warning(
        "[P1-PLAN-LOTE-790] lista de %s con %d alimento(s) de catálogo de OTRO país (se quedan, "
        "sellados catalogo_de_otro_pais): %s. Fuga aguas arriba: el generador no se los ofreció.",
        pais, len(sellados), ", ".join(sellados))


# ── (c) En un PAQUETE de comida, el tope de condimentos no corta los gramos de la receta ──────────

# [P1-PLAN-LOTE-791 · 2026-09-28 · revisión ronda 1, defecto 1] El tope de condimentos
# (P1-SHOPLIST-SANITY-CAP) reconoce un condimento por su DATO: Despensa + envase ≤ 120 g. Con los
# envases del lote 791, los seis chiles secos mexicanos (paquete de 85 g) cumplían las dos cosas y
# el tope los capaba a 1-3 paquetes por ciclo: 4 × «60 g de Chile guajillo» en una semana (240 g)
# salían «1 paquete de Chile guajillo», 85 g de 240, sin nota de cobertura.
#
# Quitarles el tope a secas fue el primer arreglo, y el replay de 426 planes guardados lo tumbó: los
# planes reales escriben el chile por CONTEO («1 chile chipotle seco», «½ chile chipotle») y, sin peso
# por unidad en la fila, cada chile se convierte en UN paquete — la misma inflación que el tope corta
# en «1 orégano» × 30. Sin tope, dos chiles pedían «3 paquetes (85 g c/u)».
#
# La regla que sirve a los dos casos: en un envase de COMIDA (paquete, bolsa, funda — el especiero es
# sobre, frasco, pote, caja) el tope no baja de los envases que cubren la demanda que la receta dio EN
# GRAMOS (`base_unit == "g"`); la que llega por conteo se capa como siempre. El especiero no cambia: su
# premisa («el consumo de un condimento no escala con las recetas que lo mencionan») sigue valiendo
# aunque la demanda venga en gramos. Sale del DATO de la fila (`market_container`), no de una lista de
# nombres. Sin los diminutivos: la «fundita» de orégano sí es especiero. Knob
# `MEALFIT_CONDIMENT_CAP_FOOD_GRAMS_FLOOR` (default on).
#
# [P1-PLAN-LOTE-791 · 2026-09-28 · revisión ronda 2, defecto 1] Y SÓLO en filas SIN precio (el
# catálogo-país del lote). La regla valía para cualquier fila Despensa en paquete de ≤ 120 g, y en la
# tabla viva hay UNA fila dominicana así: «Nueces mixtas» (paquete de 100 g, RD$95). 7 × «30 g de
# Nueces mixtas» pasaba de «1 paquete» (RD$95) a «3 paquetes» (RD$285), y la mensual de RD$285 a
# RD$855: arreglaba una compra corta anterior al lote, pero cambiaba la lista DO sin declararlo — el
# mismo criterio que dejó fuera el pan («cambia listas DO: su propio lote»). Con precio, el tope de
# siempre; la compra corta de las nueces queda abierta (docs/envases_y_catalogo_por_pais.md).
_ENVASES_DE_COMIDA = frozenset({"paquete", "paquetes", "bolsa", "bolsas", "funda", "fundas"})


def fila_sin_precio(master_item) -> bool:
    """¿La fila no tiene NINGÚN precio (ni por libra, ni por unidad, ni un paquete con precio)? Es la
    forma del catálogo-país de los países beta. Ante la duda (dato ilegible), `False`: con precio."""
    try:
        if not isinstance(master_item, dict):
            return False
        for col in ("price_per_lb", "price_per_unit"):
            if float(master_item.get(col) or 0) > 0:
                return False
        for paquete in master_item.get("market_packages") or []:
            if isinstance(paquete, dict) and float(paquete.get("price") or 0) > 0:
                return False
        return True
    except Exception:
        return False


def envase_de_comida(master_item) -> bool:
    """¿La fila se vende en un envase de COMIDA (paquete, bolsa, funda) y no de especiero?"""
    try:
        envase = (master_item or {}).get("market_container") if isinstance(master_item, dict) else None
        return str(envase or "").strip().lower() in _ENVASES_DE_COMIDA
    except Exception:
        return False


def tope_de_comida(market_obj, master_item, tope) -> int:
    """El tope de envases de condimento, subido hasta los envases que cubren la demanda EN GRAMOS
    cuando la fila es un paquete de comida SIN precio. Cualquier otro caso devuelve `tope` tal cual.
    Nunca revienta: corre en el camino caliente del agregador."""
    try:
        if not _env_bool("MEALFIT_CONDIMENT_CAP_FOOD_GRAMS_FLOOR", True) or not envase_de_comida(master_item):
            return tope
        if not fila_sin_precio(master_item):
            return tope  # [revisión ronda 2, defecto 1] con precio (Nueces mixtas, DO): la lista de siempre
        if str((market_obj or {}).get("base_unit") or "").strip().lower() != "g":
            return tope  # conteo sin peso: «1 chile» no es «1 paquete»
        gramos = float((market_obj or {}).get("base_qty") or 0)
        envase_g = float((master_item or {}).get("container_weight_g") or 0)
        if gramos <= 0 or envase_g <= 0:
            return tope
        import math
        return max(int(tope), int(math.ceil(gramos / envase_g - 1e-6)))
    except Exception:
        return tope
