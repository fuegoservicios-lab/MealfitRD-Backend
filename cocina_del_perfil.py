# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-850 · 2026-09-29] La asignación previa y los ejemplos del prompt salen de la COCINA del perfil.

Batería G24 (29-sep, 6 planes reales con el 815 desplegado): lo que más pesaba en los países beta no era el prompt de
sistema sino la asignación DETERMINISTA que el planificador recibe como obligatoria:
  · el sembrador (`ai_helpers.get_deterministic_variety_prompt`) sortea los carbohidratos del pool de MERCADO más los
    básicos universales (`constants.UNIVERSAL_MARKET_STAPLES`, con batata, yuca, plátano, habichuelas y harina de maíz
    precocida): España recibió «Harina de trigo + Batata» los tres días y México «Yuca + Harina de maíz precocida»;
  · el selector de técnicas (`graph_orchestrator._select_techniques`) sortea de un catálogo con «Estilo Fusión Criolla»,
    «Desmenuzado (Ropa Vieja)», «Relleno (Ej. Canoas…)» y «Wrap o Burrito Dominicano» (en US, ropa vieja de huevos);
  · y los ejemplos del prompt («comidas dominicanas», «guineo, plátano, batata», «Mangú de plátano / ñame / batata»,
    habichuelas y gandules) seguían siendo los dominicanos.

Reglas (I16, cocina ≠ mercado):
  · Sólo cambia QUÉ se asigna y cómo se ETIQUETA; el catálogo (lo que el modelo puede usar) y los nombres de sus filas
    —identificadores del motor: `pantry_names_match`, guard de coherencia, backstop de alergias— no se tocan.
  · La cocina es `constants.cultural_country_for_form_data` (la principal); el mercado sigue decidiendo el pool.
  · Los datos viven en `cultural_profiles.PROFILE_KITCHEN` y en la biblioteca compilada del país (`dish_registry`).
  · DO (y la cocina DO en un mercado beta) devuelve la entrada INTACTA: mismo objeto, mismo orden, mismo sorteo.
  · Knob `MEALFIT_BETA_CULTURAL_ASSIGNMENT` (True), leído en cada llamada: apagado ⇒ la conducta del 815.
tooltip-anchor: P1-PLAN-LOTE-850
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Con menos bases que esto tras el filtro, el pool se queda como estaba: dos bases por día y tres o cuatro días por
# bloque; un pool más corto convierte la variedad en repetición (lección del gate de fruta, P1-FRUIT-SEEDER-GATE).
MIN_BASES = 4

_CONSTITUYENTES: dict = {}


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_BETA_CULTURAL_ASSIGNMENT", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _norm(s) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(s or "")).strip().lower()
    except Exception:                                                          # noqa: BLE001
        return str(s or "").strip().lower()


def _perfil(cocina):
    """(código canónico, datos de `PROFILE_KITCHEN`) de una cocina beta; (None, None) si no aplica: sin cocina, DO,
    país desconocido (el fail-safe de `canonicalize_country` cae a DO), knob apagado o perfil sin datos."""
    if cocina is None or not enabled():
        return None, None
    try:
        from constants import canonicalize_country
        from cultural_profiles import PROFILE_KITCHEN, profile_for_market
        cc = canonicalize_country(cocina)
        if cc == "DO":
            return None, None
        datos = PROFILE_KITCHEN.get(profile_for_market(cc))
        return (cc, datos) if datos else (None, None)
    except Exception as _e:                                                    # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-850] cocina {cocina!r} sin resolver ({type(_e).__name__}): conducta previa")
        return None, None


def _constituyentes(cc: str) -> frozenset:
    """Los alimentos (nombres canónicos del catálogo) que usan las plantillas de la biblioteca compilada del país."""
    if cc not in _CONSTITUYENTES:
        try:
            import dish_registry as dr
            snap = dr.load_registry(cc) or {}
            _CONSTITUYENTES[cc] = frozenset(
                _norm(c.get("canonical") or c.get("name")) for t in (snap.get("templates") or [])
                for c in (t.get("constituents") or []) if (c.get("canonical") or c.get("name")))
        except Exception as _e:                                                # noqa: BLE001
            logger.warning(f"[P1-PLAN-LOTE-850] biblioteca de {cc} sin cargar ({type(_e).__name__})")
            return frozenset()
    return _CONSTITUYENTES[cc]


# ─── (1) carbohidratos que el sembrador puede ASIGNAR ────────────────────────────────────────────────────────────────
def _solo_desayuno() -> tuple:
    try:
        from ai_helpers import _BREAKFAST_ONLY_BASES                      # SSOT: la lista vive en el sembrador
        return tuple(_BREAKFAST_ONLY_BASES)
    except Exception:                                                          # noqa: BLE001
        return ("avena", "granola", "cereal", "hojuelas", "corn flakes", "muesli")


def carbos_de_la_cocina(carbs, cocina) -> list:
    """Del pool de carbohidratos del mercado, los que la biblioteca de platos de la cocina usa, menos los que el perfil
    declara ajenos como base (`carb_bases_excluded`) y menos los cereales de desayuno. Orden del pool conservado. Sin
    datos, sin biblioteca o con menos de `MIN_BASES` supervivientes ⇒ la lista tal cual (misma lista).

    Los cereales de desayuno (avena…) salen porque lo que el sembrador asigna es la base OBLIGATORIA del almuerzo o la
    cena («El Almuerzo o Cena principal DEBE incluir…») y la avena nunca lo es (P1-OATS-NOT-A-DINNER; su sitio es la
    categoría B del desayuno). Con el pool más corto de la cocina, la avena caía entre las dos bases del bloque ~1 de
    cada 7 veces y dejaba el par en «Avena + otra base distinta del catálogo» (G24, PR: «Atún en agua + Avena»)."""
    cc, datos = _perfil(cocina)
    if not datos:
        return carbs
    lib = _constituyentes(cc)
    if not lib:
        return carbs
    fuera = {_norm(x) for x in (datos.get("carb_bases_excluded") or ())}
    desayuno = _solo_desayuno()
    quedan = [c for c in (carbs or []) if _norm(c) in lib and _norm(c) not in fuera
              and not any(t in _norm(c) for t in desayuno)]
    if len(quedan) < MIN_BASES:
        logger.info(f"[P1-PLAN-LOTE-850] {cc}: sólo {len(quedan)} bases de su cocina en el pool; se deja el pool entero")
        return carbs
    return quedan


# ─── (2) técnicas ─────────────────────────────────────────────────────────────────────────────────────────────────────
def tecnicas_de_la_cocina(cooking_time, cocina):
    """(técnicas, familia de cada una) para `graph_orchestrator._select_techniques`.

    DO/sin cocina ⇒ `(constants.techniques_for_cooking_time(...), constants.TECH_TO_FAMILY)`: la lista y el MISMO dict
    de siempre, así que el sorteo consume el mismo azar y elige lo mismo. Beta ⇒ el filtro por tiempo se aplica sobre la
    técnica ORIGINAL y después se cambia su etiqueta (o se quita, `None`); la familia es la de la original."""
    import constants as C
    base = C.techniques_for_cooking_time(cooking_time)
    _cc, datos = _perfil(cocina)
    if not datos:
        return base, C.TECH_TO_FAMILY
    etiquetas = datos.get("technique_labels") or {}
    out, fam = [], {}
    for t in base:
        local = etiquetas.get(t, t)
        if local and local not in fam:
            out.append(local)
            fam[local] = C.TECH_TO_FAMILY.get(t, "")
    return (out, fam) if len(out) >= 3 else (base, C.TECH_TO_FAMILY)


# ─── (3) ejemplos del prompt ──────────────────────────────────────────────────────────────────────────────────────────
_MICROS_DO = ("comidas dominicanas variadas y sabrosas", "legumbres (habichuelas)",
              "guineo, plátano, batata, aguacate, espinaca, legumbres y naranja")


def textos_micros(cocina) -> tuple:
    """(dónde integrar los micros, ejemplo de legumbre del magnesio, ejemplos del potasio) del bloque de
    micronutrientes del día. DO/sin cocina ⇒ los literales de siempre."""
    cc, datos = _perfil(cocina)
    if not datos:
        return _MICROS_DO
    try:
        from cultural_profiles import PROFILES, profile_for_market
        nombre = str((PROFILES.get(profile_for_market(cc)) or {}).get("name_es") or "").strip()
        voc = datos["vocab"]
        mesa = f"comidas variadas y sabrosas de la {nombre[:1].lower()}{nombre[1:]}" if nombre else \
            "comidas variadas y sabrosas de su cocina"
        return mesa, f"legumbres ({voc['legumbre']})", voc["potasio"]
    except Exception:                                                          # noqa: BLE001
        return _MICROS_DO


# Prompt de sistema del generador de días (render beta, DESPUÉS de la neutralización léxica; el sobreviviente
# P1-CASABE-NO-BOIL es una regla de técnica documentada y no se toca). Pares (literal DO, plantilla con {vocab}).
_SISTEMA = (
    ("(Mangú/tubérculos, Avena/cereales, Pan/tostadas, Batido/bowl, Revoltillo/tortilla). NO elijas mangú si el "
     "planificador asignó otra categoría.",
     "(desayuno típico local, Avena/cereales, Pan/tostadas, Batido/bowl, Revoltillo/tortilla). NO elijas otra categoría "
     "que la asignada."),
    ("Al menos 1 comida con leguminosas (habichuelas, gandules, lentejas).", "Al menos 1 comida con leguminosas ({legumbres})."),
    ("Guiso de leguminosas (lentejas, garbanzos, habichuelas)", "Guiso de leguminosas ({legumbres})"),
    ("habichuelas/lentejas/garbanzos", "{legumbre}/lentejas/garbanzos"),
    ("OTRA proteína (habichuelas, lentejas, garbanzos", "OTRA proteína ({legumbre}, lentejas, garbanzos"),
    ("(queso, yogur, habichuelas, lentejas, garbanzos)", "(queso, yogur, {legumbre}, lentejas, garbanzos)"),
    ("Garbanzos/habichuelas raw equivalente", "Garbanzos/{legumbre} raw equivalente"),
    ("maíz o habichuela de lata", "maíz o {legumbre_1} de lata"),
    ("½ tomate, ½ guineo)", "½ tomate{medio_banana})"),   # «½ plátano» ya está en la lista: en ES/MX no se repite
    ("guineo maduro)", "{banana_maduro})"),
    # tabla de macros (`graph_orchestrator._NUTRITION_LOOKUP_INSTRUCTION`): casabe y salami no se venden en beta
    ("(ej. gandules, yuca, guineo, yogurt griego, mango, ñame, casabe, salami, jamón, sardina)",
     "(ej. {legumbre}, yuca, {banana}, yogurt griego, mango, ñame, jamón, sardina)"),
    ("Legumbres secas cocidas (gandules, garbanzos, lentejas)", "Legumbres secas cocidas ({legumbre}, garbanzos, lentejas)"),
    ("  - Guineo (banana): ~89", "  - {Banana_tabla}: ~89"),
)

_PLANIFICADOR = (
    ("Ejemplo INCORRECTO: Día 1=Mangú de plátano (A), Día 2=Mangú de ñame (A), Día 3=Mangú de batata (A) ← MISMO "
     "CONCEPTO, PROHIBIDO.",
     "Ejemplo INCORRECTO: Día 1=Avena con fresas (B), Día 2=Avena con manzana (B), Día 3=Avena con mango (B) ← MISMO "
     "CONCEPTO, PROHIBIDO."),
)

# Prompt de variedad del sembrador: se aplica ANTES de `neutralize_do_lexicon` (literales crudos de la plantilla).
_VARIEDAD = (
    ("(arroz/víver/pasta)", "(arroz, tubérculo o pasta)"),
    ("Si la base es yuca/plátano/víver, además del hervido clásico puedes transformarla (bollitos de yuca, majado, "
     "arepitas de yuca, mangú).",
     "Si la base es un tubérculo o un plátano para cocinar (verde o maduro), además del hervido clásico puedes "
     "transformarla (majado, puré, tortitas al horno)."),
    ("(mangú solo, casabe solo,", "(puré de tubérculo solo, casabe solo,"),
)


def _aplica(texto, pares, voc):
    if not isinstance(texto, str) or not texto:
        return texto
    for viejo, nuevo in pares:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo.format(**voc) if voc else nuevo)
    return texto


# Memo de los renders localizados: los prompts de sistema llegan de cachés por (dieta, país) y el mismo texto debe
# salir como el MISMO objeto (prompt-cache del proveedor y el contrato «la segunda llamada reutiliza el render»). La
# clave incluye el texto de entrada, así que apagar el knob o cambiar el render no sirve nada viejo. Acotado.
_MEMO: dict = {}
_MEMO_MAX = 64


def _memo(tipo, cc, texto, calcular):
    if not isinstance(texto, str):
        return texto
    clave = (tipo, cc, texto)
    hit = _MEMO.get(clave)
    if hit is None:
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.clear()
        hit = _MEMO[clave] = calcular(texto)
    return hit


def _voc(datos) -> dict:
    v = dict(datos.get("vocab") or {})
    b = str(v.get("banana") or "")
    v["Banana_tabla"] = "Banana" if b.lower() == "banana" else f"{b[:1].upper()}{b[1:]} (banana)"
    v["medio_banana"] = "" if b.lower() == "plátano" else f", ½ {b}"
    return v


def localizar_sistema(texto, cocina):
    """Ejemplos del prompt de sistema del generador de días con el léxico de la cocina. DO ⇒ el mismo objeto."""
    cc, datos = _perfil(cocina)
    if not datos:
        return texto
    try:
        return _memo("sistema", cc, texto, lambda t: _aplica(t, _SISTEMA, _voc(datos)))
    except Exception as _e:                                                    # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-850] prompt de sistema sin localizar ({type(_e).__name__})")
        return texto


def localizar_planificador(texto, pais):
    """El ejemplo INCORRECTO de la regla de desayunos, sin mangú (el CORRECTO lo neutralizó F1). DO ⇒ el mismo objeto."""
    cc, datos = _perfil(pais)
    return _memo("planificador", cc, texto, lambda t: _aplica(t, _PLANIFICADOR, None)) if datos else texto


def localizar_variedad(texto, pais):
    """La regla de bases transformables y la de proteína en cada comida, sin víveres ni mangú. DO ⇒ el mismo objeto."""
    _cc, datos = _perfil(pais)
    return _aplica(texto, _VARIEDAD, None) if datos else texto
