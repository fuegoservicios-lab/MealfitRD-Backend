# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-181 · 2026-09-23] El último pulido de las líneas que el usuario LEE, al final de la cola del escudo.

Métrica sobre los planes REALES de la batería (rd9 + rd10, 817 líneas de ingredientes): 37 con el alimento en mayúscula
a media frase («⅓ taza de Yogurt natural entero», «25 g de Harina de trigo»: los cerradores y las sustituciones escriben
el nombre tal como está en el catálogo), 4 «0.5 pepino», 11 especias contadas sin unidad («1 orégano dominicano», «1
canela en polvo»), 2 «½ pizca», «0.06 g de semillas de chía» y «1 pechugas de pollo». El pulido de frontera
(`graph_orchestrator._polish_finalize_display`) ya sabe arreglar parte de eso, pero corre ANTES de los últimos pases de
la cola del escudo (identidad del plato, contrato de receta), que vuelven a escribir líneas con decimales. Aquí se
re-dispara después de ellos y se añaden las reglas que faltaban.

Las reglas nuevas tocan sólo el DISPLAY (`ingredients`): `ingredients_raw` es la fuente de macros y de la compra (el
re-disparo del pulido de frontera conserva su propio contrato). La longitud de las dos listas no cambia (hay
consumidores que las recorren juntas). Idempotente, nunca lanza.
Knob `MEALFIT_LINE_POLISH_TAIL` (True). tooltip-anchor: P1-PLAN-LOTE-181-PULIDO-DE-LINEAS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_LEAD = r"(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔⅛]?|[½¼¾⅓⅔⅛])"
_UNIDAD = (r"(?:g|gr|gramos|kg|ml|l|oz|lb|libras?|onzas?|tazas?|cdas?|cdtas?|cditas?|cucharadas?|cucharaditas?|"
           r"pizcas?|latas?|rebanadas?|porci[oó]n|porciones|pedazos?|lonjas?|filetes?|unidad(?:es)?|potes?|dientes?|"
           r"ramas?|hojas?|ramitas?|puñados?|vasos?|tallos?|rodajas?)")

# «⅓ taza de Yogurt natural entero» / «60 g de Guineo» / «1 cdta (4 g) de Canela» → minúscula. Sólo si es la ÚNICA
# mayúscula del resto de la línea: «1 taza de Corn Flakes» es una marca y se queda como está.
_MAYUS_TRAS_DE = re.compile(rf"^(\s*{_LEAD}\s*(?:{_UNIDAD}\.?\s+)?(?:\([^)]*\)\s+)?de\s+)([A-ZÁÉÍÓÚÑ])(?=[a-záéíóúñü])")
# «1 Lechosa mediana» → «1 lechosa mediana» (conteo sin «de»).
_MAYUS_TRAS_CONTEO = re.compile(rf"^(\s*{_LEAD}\s+)([A-ZÁÉÍÓÚÑ])(?=[a-záéíóúñü]{{2,}})")

# «½ pizca de canela» → «1 pizca de canela»: media pizca no es una medida.
_MEDIA_PIZCA = re.compile(r"^\s*(?:[½¼⅓⅛]|0[.,]\d+)\s*pizca\b", re.IGNORECASE)

# Especia seca contada sin unidad: «1 orégano dominicano», «1 canela en polvo» → «Orégano dominicano al gusto».
_ESPECIA = (r"(?:or[eé]gano|canela|comino|pimienta|nuez moscada|c[uú]rcuma|paprika|piment[oó]n|curry|tomillo|romero|"
            r"jengibre en polvo|ajo en polvo|cebolla en polvo|adobo|saz[oó]n|sal)")
_ESPECIA_SIN_UNIDAD = re.compile(rf"^\s*{_LEAD}\s+(?!de\b)({_ESPECIA}\b.*)$", re.IGNORECASE)

# Menos de 1 g de una semilla o un polvo: «0.06 g de semillas de chía» → «1 pizca de semillas de chía».
# [P1-PLAN-LOTE-219] también con fracción unicode: «⅔ g de semillas de girasol» (batería del 24-sep)
_MIGAJA = re.compile(r"^\s*(?:0(?:[.,]\d+)?|[½¼¾⅓⅔⅛])\s*(?:g|gr|gramos)\s+de\s+(.+)$", re.IGNORECASE)
# [P1-PLAN-LOTE-219] lo que se sirve a cucharadas: «1.53 g de yogurt natural» → «1 cdta de yogurt natural»
_CUCHARABLE = re.compile(r"^(?:yogur|yogurt|leche|crema|queso crema|miel|mantequilla)\b", re.IGNORECASE)
_PIZCABLE = re.compile(rf"\b(?:semillas?|ch[ií]a|linaza|ajonjol[ií]|s[eé]samo|cacao|sal|{_ESPECIA})\b", re.IGNORECASE)
# [P1-PLAN-LOTE-352] Menos de 1 g de un fruto seco que el nombre del plato promete (la traza se queda): «1 pizca de maní»,
# como adorno. Antes salía pizca por accidente — el «sal» de «sin sal» casaba con `_PIZCABLE`.
_FRUTO_SECO_352 = re.compile(r"\b(?:man[ií]|almendras?|nuez|nueces|pistachos?|mara[ñn][oó]n|avellanas?)\b", re.IGNORECASE)
# [P1-PLAN-LOTE-203 · 2026-09-24] Entre 1 y 2,5 g con decimales —la zona que el cuantizador deja sin tocar («ya es pesable
# en báscula de precisión»)—: «1.21 g de Ajo» (plan canario del dueño), «1.23 g de semillas de linaza». Nadie pesa 1,21 g
# de ajo: medio diente (≈1,5 g) o media cucharadita (semillas, especias, sal) es lo que se mide en una cocina. Sólo con
# decimales: «2 g de sal» ya es una medida.
_DECIMAL_1_A_2_5 = re.compile(r"^\s*(\d+[.,]\d+)\s*(?:g|gr|gramos)\s+de\s+(.+)$", re.IGNORECASE)
_AJO_FRESCO = re.compile(r"^ajos?\b(?!\s+en\s+polvo)", re.IGNORECASE)

# «1 pechugas de pollo (≈134 g)» → «1 pechuga …»: la concordancia con UNO (o una fracción sola) que el re-cuadre de
# conteos no cubre (su tabla es la del qty-sync de los pasos, y no se toca por esto).
_SINGULAR = {
    "pechugas": "pechuga", "filetes": "filete", "muslos": "muslo", "tortillas": "tortilla", "rodajas": "rodaja",
    "tajadas": "tajada", "chuletas": "chuleta", "arepas": "arepa", "galletas": "galleta", "tostadas": "tostada",
    "batatas": "batata", "yucas": "yuca", "zanahorias": "zanahoria", "cebollas": "cebolla", "tomates": "tomate",
    "limones": "limón", "pepinos": "pepino", "manzanas": "manzana", "peras": "pera", "ciruelas": "ciruela",
    "tallos": "tallo", "hojas": "hoja", "ramas": "rama", "dientes": "diente", "piezas": "pieza", "bolsitas": "bolsita",
}
_UNO_PLURAL = re.compile(r"^(\s*(?:1|[½¼¾⅓⅔⅛]))\s+(" + "|".join(_SINGULAR) + r")\b", re.IGNORECASE)


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_LINE_POLISH_TAIL", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _mayuscula(s: str) -> str:
    m = _MAYUS_TRAS_DE.match(s) or _MAYUS_TRAS_CONTEO.match(s)
    if not m:
        return s
    resto = s[m.end(2):]
    if re.search(r"\b[A-ZÁÉÍÓÚÑ]", re.sub(r"\([^)]*\)", "", resto)):
        return s                                   # otra mayúscula detrás: nombre propio o marca
    return s[:m.start(2)] + m.group(2).lower() + resto


# [P1-PLAN-LOTE-341 · 2026-09-25] Y el número concuerda cuando son VARIOS: «1½ ají cubanela», «4 ají cubanela», «2 nabo
# mediano» (26 líneas en el corpus de 308 planes) → «ajíes», «nabos medianos»; y el participio con su sustantivo en plural:
# «2½ tazas de espinacas picado» (38 líneas) → «picadas». Solo el display; la cabeza cerrada de sustantivos contables.
# tooltip-anchor: P1-PLAN-LOTE-341
_PLURALES = {"aji": "ajíes", "ají": "ajíes", "nabo": "nabos", "tomate": "tomates", "limon": "limones", "limón": "limones",
             "cebolla": "cebollas", "pepino": "pepinos", "zanahoria": "zanahorias", "diente": "dientes",
             "tortilla": "tortillas", "arepita": "arepitas", "berenjena": "berenjenas", "papa": "papas",
             "batata": "batatas", "pechuga": "pechugas"}
_ADJ_PLURALES = {"mediano": "medianos", "mediana": "medianas", "pequeño": "pequeños", "pequeña": "pequeñas",
                 "grande": "grandes", "verde": "verdes", "maduro": "maduros", "madura": "maduras", "rojo": "rojos",
                 "roja": "rojas", "entero": "enteros", "entera": "enteras", "fresco": "frescos", "fresca": "frescas"}
_VARIOS_SINGULAR = re.compile(
    r"^(\s*(?:\d+\s*[½¼¾⅓⅔]|\d+(?:[.,]\d+)?))\s+(" + "|".join(sorted(_PLURALES, key=len, reverse=True))
    + r")\b(\s+([a-záéíóúñ]+))?", re.IGNORECASE)
_PARTICIPIO_TRAS_PLURAL = re.compile(r"\b([a-záéíóúñ]+(?:as|os))\s+(picad|rallad|cocid|trocead|cortad|pelad|lavad)o\b",
                                     re.IGNORECASE)
_FRAC_VAL = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}


def _plural_con_varios(s: str) -> str:
    m = _VARIOS_SINGULAR.match(s)
    if not m:
        return s
    try:
        valor = sum(_FRAC_VAL[p] if p in _FRAC_VAL else float(p.replace(",", "."))
                    for p in re.findall(r"\d+(?:[.,]\d+)?|[½¼¾⅓⅔]", m.group(1)))
    except Exception:
        return s
    if valor <= 1:
        return s
    sust = _PLURALES[m.group(2).lower()]
    if m.group(2)[:1].isupper():
        sust = sust[:1].upper() + sust[1:]
    cola = m.group(3) or ""
    if m.group(4) and m.group(4).lower() in _ADJ_PLURALES:
        cola = cola[:len(cola) - len(m.group(4))] + _ADJ_PLURALES[m.group(4).lower()]
    return s[:m.start(2)] + sust + cola + s[m.end():]


def _concuerda_participio(mm) -> str:
    fem = mm.group(1).lower().endswith("as")
    return f"{mm.group(1)} {mm.group(2)}{'as' if fem else 'os'}"


def pulir_linea(s: str) -> str:
    """Una línea del display, pulida. Lo que no reconoce lo devuelve igual."""
    if not isinstance(s, str) or not s.strip():
        return s
    out = s
    m = _MIGAJA.match(out)
    if m and (_PIZCABLE.search(re.sub(r"\bsin\s+sal\b", "", m.group(1), flags=re.IGNORECASE))  # [P1-PLAN-LOTE-352]
              or _FRUTO_SECO_352.search(m.group(1))):
        out = f"1 pizca de {m.group(1).strip()}"
    m = _DECIMAL_1_A_2_5.match(out)                                          # [P1-PLAN-LOTE-203]
    if m and 1.0 <= float(m.group(1).replace(",", ".")) < 2.5:
        nombre = m.group(2).strip()
        if _AJO_FRESCO.match(nombre):
            out = f"½ diente de {nombre}"
        elif re.match(r"^sal\b", nombre, re.IGNORECASE):
            out = f"¼ cdta de {nombre}"                  # la sal pesa ~6 g la cucharadita
        elif _PIZCABLE.search(re.sub(r"\bsin\s+sal\b", "", nombre, flags=re.IGNORECASE)):   # «maní … sin sal» no es sal
            out = f"½ cdta de {nombre}"
        elif _CUCHARABLE.match(nombre):                                        # [P1-PLAN-LOTE-219]
            out = f"1 cdta de {nombre}"
    if _MEDIA_PIZCA.match(out):
        out = _MEDIA_PIZCA.sub("1 pizca", out, count=1)
    m = _ESPECIA_SIN_UNIDAD.match(out)
    if m:
        cuerpo = m.group(1).strip()
        cuerpo = cuerpo[:1].upper() + cuerpo[1:]
        out = cuerpo if re.search(r"\bal gusto\b", cuerpo, re.IGNORECASE) else f"{cuerpo} al gusto"
    m = _UNO_PLURAL.match(out)
    if m:
        sing = _SINGULAR[m.group(2).lower()]
        out = f"{m.group(1)} {sing}" + out[m.end(2):]
    out = _PARTICIPIO_TRAS_PLURAL.sub(_concuerda_participio, _plural_con_varios(out))  # [P1-PLAN-LOTE-341]
    return _mayuscula(out)


def pulir_plan(plan_data) -> int:
    """Re-dispara el pulido de frontera y aplica las reglas de arriba al display. Devuelve cuántas líneas cambió."""
    if not (enabled() and isinstance(plan_data, dict) and isinstance(plan_data.get("days"), list)):
        return 0
    try:
        import graph_orchestrator as go
        go.refire_display_polish_post_finalize(plan_data)
        go.fix_ingredient_count_agreement(plan_data)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-181] re-pulido de frontera no-op: {type(e).__name__}: {e}")
    n = 0
    for d in plan_data.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict) or not isinstance(m.get("ingredients"), list):
                continue
            try:
                nuevas = [pulir_linea(x) for x in m["ingredients"]]
            except Exception as e:                                             # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-181] pulido no-op en {str(m.get('name'))[:40]}: {e!r}")
                continue
            cambios = sum(1 for a, b in zip(m["ingredients"], nuevas) if a != b)
            if cambios:
                m["ingredients"] = nuevas
                m.pop("_display", None)
                n += cambios
                # [P1-PLAN-LOTE-358 · 2026-09-26] El contrato (lote 330) ya corrió: si esta pasada convirtió una migaja
                # de la lista en pizca («0.47 g de Sal» → «1 pizca de sal»), el paso («½ g de Sal») la sigue aquí.
                # tooltip-anchor: P1-PLAN-LOTE-358
                try:
                    __import__("pasos_cantidades").pizcas_de_la_lista(m)
                except Exception:                                              # noqa: BLE001
                    pass
    if n:
        logger.info(f"🪄 [P1-PLAN-LOTE-181] {n} línea(s) pulidas al final de la cola (mayúsculas, pizcas, especias, conteos)")
    return n
