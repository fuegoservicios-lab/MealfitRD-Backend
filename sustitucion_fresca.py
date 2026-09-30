# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-460/461 · 2026-09-27] La sustitución por frescura de la compra única, coherente en TODO el plato.

En la compra única de 30 días sin congelador, lo fresco que no aguanta hasta su día se cambia por su duradero
(`compra_unica.sustituir_linea`, lotes 214-216): «¾ pechuga de pollo» → «sardinas en lata». La LISTA cambiaba; el resto
del plato, no. En el plan vivo del dueño (6594aae1) el día 3 decía «Wok express de **pollo**… saltea el **pollo**… hasta
que el pollo esté completamente cocido» sobre una lista con sardinas, y el día 2 mandaba a cocinar «el sardinas en lata
blanco a la plancha, 3-4 min por lado o hasta que esté completamente cocido». Forzando la compra única sobre el corpus de
322 planes: de 1.248 sustituciones, 628 nombres, 1.035 descripciones y 1.009 recetas seguían nombrando la proteína vieja,
375 cocinaban lo enlatado «por lado» y 305 «hasta que esté cocido».

**460 · el texto.** El reemplazo era de UN token (`hit_tok`): «pechuga de pollo» casaba en la línea pero el paso decía
«el pollo», y el género quedaba del fresco («el sardinas»). Aquí cambia la FRASE entera del alimento —artículo,
cantidad, corte («filete de…», «cubos de…»), calificativos («blanco», «magra», «ya cocida») y el paréntesis de peso—
por el duradero con su artículo, y los adjetivos pegados concuerdan («el pollo salteado» → «las sardinas salteadas»),
también el pronombre del verbo que lo retoma («ásalo» → «caliéntalas»). Lo que ya viene cocido (atún, sardinas,
garbanzos, lentejas) no se cocina como crudo: fuera «por lado», las temperaturas de cocción segura (60-79 °C) y «hasta
que esté cocido/opaco»; «sella/dora/asa/cocina» pasan a «calienta» con 2-3 min; «corta/seca… en tiras» pasa a
«escurre»; y si se cocinaba JUNTO con vegetales, los vegetales conservan su verbo y su tiempo y el duradero entra al
final. Del nombre y la descripción salen los métodos de un crudo («a la plancha», «horneado», «bien cocido»). Los
nombres de la lista no se tocan aquí: son identificadores (compra, Nevera).

**461 · la lista.** `raw[idx] = nueva` era por ÍNDICE, y `ingredients_raw` no sigue el orden de la lista visible: en el
plan del dueño escribió las sardinas encima de otra línea y dejó «1¼ filetes de pescado» — la lista compraba el pescado
fresco que ningún plato usaba (546 de 1.248 en el corpus). Ahora la pareja se busca por ALIMENTO (clase de la tabla de
sustitutos, sin lo que ya es duradero ni lo que empareja con otra línea visible): la primera se reescribe, sus
duplicados salen y, si no hay ninguna, la línea nueva se añade para que la lista compre el duradero.
tooltip-anchor: P1-PLAN-LOTE-460-TEXTO · P1-PLAN-LOTE-461-RAW
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# identidad del duradero (`compra_unica`) → (nombre en el texto, género, número, ya viene listo para comer)
_DURADERO = {
    "atun en agua": ("atún", "m", "s", True),
    "sardinas en lata": ("sardinas", "f", "p", True),
    "garbanzos cocidos": ("garbanzos", "m", "p", True),
    "lentejas cocidas": ("lentejas", "f", "p", True),
    "repollo": ("repollo", "m", "s", False),
    "zanahoria": ("zanahoria", "f", "s", False),
    "oregano": ("orégano", "m", "s", False),
    "manzana": ("manzana", "f", "s", False),
    "leche UHT": ("leche UHT", "f", "s", False),
    "batata": ("batata", "f", "s", False),
    "casabe": ("casabe", "m", "s", False),
    "salsa de tomate": ("salsa de tomate", "f", "s", False),     # [P1-PLAN-LOTE-466] el tomate de un guiso
    "claras de huevo": ("claras", "f", "p", False),              # [P1-PLAN-LOTE-495] se cocinan: no vienen listas
    # [P1-PLAN-LOTE-936] la rueda de fruta duradera y las aceitunas del aguacate: sin fila aquí el plato seguía
    # nombrando el fresco («…con aguacate fresco» con aceitunas en la lista)
    "naranja": ("naranja", "f", "s", False),
    "pera": ("pera", "f", "s", False),
    "aceitunas": ("aceitunas", "f", "p", True),
}
#: [P1-PLAN-LOTE-466] duraderos que se MIDEN, no se cortan: «pica 2 tomates» → «mide 60 g de salsa de tomate»
# [P1-PLAN-LOTE-490 · 2026-09-27] el orégano SECO tampoco se pica: «pica 2 cdas de cilantro» → «mide 2 cdtas de orégano»,
# «termina con el cilantro picado» → «termina con el orégano». tooltip-anchor: P1-PLAN-LOTE-490-OREGANO-SE-MIDE
_LIQUIDOS = ("salsa de tomate", "orégano")
_ADJ_CORTE = re.compile(r"^(?:picad|cortad|trocead|rallad|rebanad|laminad|fresc)$", re.IGNORECASE)

# género de la palabra NÚCLEO de la frase del fresco (sin acentos)
_GENERO = {
    "pechuga": "f", "muslo": "m", "filete": "m", "chuleta": "f", "bistec": "m", "lomo": "m", "carne": "f", "cubo": "m",
    "trozo": "m", "tira": "f", "dado": "m", "pieza": "f", "pedazo": "m", "lasca": "f", "medallon": "m", "porcion": "f",
    "pollo": "m", "pavo": "m", "res": "f", "cerdo": "m", "chivo": "m", "conejo": "m", "higado": "m", "pescado": "m",
    "tilapia": "f", "salmon": "m", "mero": "m", "chillo": "m", "dorado": "m", "bacalao": "m", "merluza": "f",
    "camaron": "m", "marisco": "m", "calamar": "m", "pulpo": "m", "cangrejo": "m", "langosta": "f", "lambi": "m",
    "lechuga": "f", "berro": "m", "rucula": "f", "arugula": "f", "espinaca": "f", "acelga": "f", "kale": "m", "col": "f",
    "tomate": "m", "pepino": "m", "calabacin": "m", "zucchini": "m", "brocoli": "m", "coliflor": "f", "vainita": "f",
    "habichuela": "f", "esparrago": "m", "champinon": "m", "hongo": "m", "seta": "f", "cilantro": "m", "perejil": "m",
    "albahaca": "f", "menta": "f", "cebollin": "m", "cebollino": "m", "fresa": "f", "frambuesa": "f", "mora": "f",
    "arandano": "m", "uva": "f", "lechosa": "f", "papaya": "f", "mango": "m", "pina": "f", "melon": "m", "sandia": "f",
    "guineo": "m", "banana": "f", "durazno": "m", "melocoton": "m", "pera": "f", "kiwi": "m", "cereza": "f",
    "mamey": "m", "nispero": "m", "aguacate": "m", "leche": "f", "platano": "m", "rulo": "m", "yuca": "f", "pan": "m",
    "panecillo": "m", "bagel": "m", "edamame": "m",
}
_CABEZAS = ("pechuga", "muslo", "filete", "chuleta", "bistec", "lomo", "carne")
#: tokens de la tabla que en el texto suelen ser OTRA palabra: sólo cuentan si la línea vieja los nombra
_HOMONIMOS = ("dorado", "mora", "lomo", "pan")
# [P1-PLAN-LOTE-469 · 2026-09-27] «la carne de filete de pescado blanco cocida» (el raw de la IA): la «carne de» se va con
# el fresco —«escurre el atún», no «escurre la carne de atún»—. tooltip-anchor: P1-PLAN-LOTE-469-CARNE-DE
_CORTES = (r"(?:cubos|trozos|tiras|dados|lascas|piezas?|pedazos?|filetes?|pechugas?|muslos?|lomos?|chuletas?|medallones|"
           r"porci[oó]n(?:es)?|carne)")
_NOTA = ("⚠", "💡", "🤰", "⚕", "🧊", "🛒", "🍽", "❄", "⏱")
_VOCAL = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]"}
# [P1-PLAN-LOTE-469] «ten listas 1 rebanada de pan integral familiar» → la medida del pan se va con él
_UNIDADES = (r"(?:g|gr|gramos|kg|oz|onzas?|lb|libras?|tazas?|unidad(?:es)?|porci[oó]n(?:es)?|piezas?|cdas?|cdtas?|"
             r"cucharadas?|cucharaditas?|rebanadas?|"
             # [P1-PLAN-LOTE-497] «escurre 2 latas de filete de pescado blanco» (la IA lo pensó enlatado) → la medida de la
             # lista, no «escurre 2 latas de sardinas» con 80 g comprados
             r"latas?)")
_NUM = r"(?:\d+(?:[.,]\d+)?[½¼¾⅓⅔]?|[½¼¾⅓⅔])"

# adjetivos que concuerdan con el alimento (raíz + o/a/os/as)
_ADJ_RAIZ = (r"guisad|saltead|desmenuzad|sazonad|marinad|aliñad|adobad|especiad|ahumad|glasead|tiern|jugos|cítric|"
             r"citric|crioll|picad|cortad|trocead|rallad|terminad|servid|acompañad|preparad|bañad|cubiert|envuelt|"
             r"combinad|mezclad|aderezad|perfumad|fresc|frí|fri|liger|dominican|caribeñ|enter|enfriad|tibi|rellen|"
             r"asad|hornead|dorad|frit|grillad|sellad|empanizad|rebozad|cocid|hervid|blanc|magr|molid|deshuesad|"
             r"cocinad|escurrid|condimentad|estofad|gratinad|ahogad|reposad|reservad|sofrit|tapad")
#: métodos que sólo tienen sentido para un crudo: salen del nombre, la descripción y la cadena del paso
_METODO_RAIZ = re.compile(r"^(?:asad|hornead|dorad|frit|grillad|sellad|empanizad|rebozad|cocid|hervid|blanc|magr|molid|"
                          r"deshuesad|cocinad)$", re.IGNORECASE)
#: participios de cocción de un crudo que, en la descripción, se van con él («…, horneada junto a la berenjena»)
_METODO_DESC = re.compile(r"^(?:marcad|dorad|asad|sellad|hornead|grillad|cocid|cocinad)$", re.IGNORECASE)
_FRASE_METODO = (r"(?:a\s+la\s+plancha|a\s+la\s+parrilla|al\s+horno|a\s+la\s+brasa|al\s+vapor|al\s+carb[oó]n|al\s+punto|"
                 r"por\s+completo|por\s+dentro|a\s+fuego\s+lento|en\s+su\s+punto|sin\s+piel|sin\s+hueso|de\s+pollo|"
                 r"de\s+pavo|de\s+res|de\s+cerdo)")
_RX_ESLABON = re.compile(
    r"(?P<sep>\s*,?\s+)(?:(?P<conj>y|o|e)\s+)?(?:(?P<bien>bien|muy|ya)\s+)?"
    r"(?:(?P<adj>(?:" + _ADJ_RAIZ + r"))(?P<fin>os|as|o|a)\b|(?P<frase>" + _FRASE_METODO + r")\b)",
    re.IGNORECASE)
#: la cocción de un crudo, que a un duradero listo sólo le toca como «calienta»
_VERBOS_CALIENTA = re.compile(r"^(?:sella|dora|asa|grilla|fr[ií]e|cocina|cuece|hierve)$", re.IGNORECASE)
_VERBOS_ESCURRE = re.compile(r"^(?:corta|trocea|filetea|seca|limpia|lava|deshuesa|pela)$", re.IGNORECASE)
#: raíces (sin tilde) de los verbos que llevan el pronombre del crudo: «ásalo», «cocínala», «séllalos»
_COCCION_CLITICO = ("asa", "cocina", "sella", "dora", "frie", "grilla", "cuece", "hierve")
_CORTE_TRAS = re.compile(r"\s+en\s+(?:tiras|cubos|cubitos|dados|trozos|piezas|una\s+pieza|filetes|lascas|l[aá]minas|"
                         r"porciones|medallones|mitades|rodajas|rebanadas)(?:\s+(?:finas?|finos?|gruesas?|gruesos?|"
                         r"pequeñ[oa]s|medianos?|medianas?|grandes|parejas?|parejos|uniformes?|de\s+\d+\s*cm))*"
                         r"(?:\s+(?:de|para)\s+(?:la\s+)?(?:parrilla|plancha|guisar|asar|fre[ií]r|horno|sart[eé]n))?",
                         re.IGNORECASE)
#: lo que lleva artículo y NO es un alimento: no corta el alcance de un pronombre («…a fuego medio-alto y déjala»)
_NO_ALIMENTO = (r"(?:centro|parte|fuego|sart[eé]n|olla|horno|plato|bowl|temperatura|mitad|resto|punto|tapa|caldero|"
                r"bandeja|molde|taz[oó]n|recipiente|superficie|lado|borde|medio|minuto|airfryer|freidora|plancha|"
                r"parrilla|calor|vapor|wok|grill|cocci[oó]n|term[oó]metro)")
_FIN_AMBITO = re.compile(r"[;.]|\b(?:el|la|los|las|al|del)\s+(?!" + _NO_ALIMENTO + r"\b)[a-záéíóúñ]", re.IGNORECASE)
_OBJETO_SIGUE = re.compile(r"(?:,\s+|\s+y\s+)(?=(?:\d|[½¼¾⅓⅔]|(?:el|la|los|las|un|una|unos|unas)\s))", re.IGNORECASE)
_TIEMPO = re.compile(r"(?P<pre>(?:durante\s+|unos\s+|por\s+|aproximadamente\s+)?)(?P<a>\d+)(?:\s*(?:-|–|a)\s*(?P<b>\d+))?\s*"
                     r"(?P<u>min(?:utos?)?\b)(?:\s+por\s+(?:cada\s+)?lado)?", re.IGNORECASE)
_TEMP_SEGURA = re.compile(
    r"(?:,\s*|\s+)?(?:\(\s*)?(?:\b(?:o|y)\s+)?"
    r"(?:(?:verifica(?:ndo)?|comprueba|comprobando|aseg[uú]rate\s+de|asegur[aá]ndote\s+de|revisa(?:ndo)?|midiendo)\s+"
    r"(?:con\s+(?:un\s+)?term[oó]metro\s+)?que\s+"
    # [P1-PLAN-LOTE-495] también en plural: «sella los filetes… hasta que alcancen 74 °C» dejaba «hasta que alcancen»
    r"(?:[^,;.()]{0,60}?\s+)?(?:alcancen?|lleguen?\s+a|marquen?|est[eé]n?\s+a)\s+"
    r"|hasta\s+(?:(?:alcanzar|llegar\s+a)\s+|que\s+(?:[^,;.()]{0,60}?\s+)?(?:alcancen?|lleguen?\s+a|marquen?)\s+)?"
    r"|(?:la\s+parte\s+m[aá]s\s+gruesa|el\s+centro|el\s+interior)[^,;.()]{0,40}?\s+(?:alcancen?|lleguen?\s+a)\s+)?"
    r"(?:6[0-9]|7[0-9])\s*°\s*C"
    r"(?:\s+(?:en\s+(?:el\s+centro|la\s+parte\s+m[aá]s\s+gruesa|el\s+interior|su\s+interior)|internos?|"
    r"de\s+temperatura\s+interna))?(?:\s+si\s+no\s+estaba\s+previamente\s+cocid[oa]s?)?(?:\s*\))?",
    re.IGNORECASE)
_HASTA_COCIDO = re.compile(
    r"(?:,\s*|\s+)(?:(?:o|y)\s+)?hasta\s+que\s+(?:"
    r"(?:[^,;.()]{0,50}?\s+)?(?:est[eé]n?|queden?|se\s+vean?|luzcan?|tengan?)\s+"
    r"(?:bien\s+|completamente\s+|totalmente\s+|ligeramente\s+)?"
    r"(?:opac[oa]s?|cocid[oa]s?|dorad[oa]s?|firmes?|blanc[oa]s?|hech[oa]s?|sellad[oa]s?|cocinad[oa]s?)"
    # [P1-PLAN-LOTE-466] «opaco y firme», «dorado y completamente cocido»: la cadena entera se va
    r"(?:\s*(?:,|y)\s+(?:bien\s+|completamente\s+|totalmente\s+)?(?:opac[oa]s?|cocid[oa]s?|dorad[oa]s?|firmes?|"
    r"blanc[oa]s?|hech[oa]s?|tiern[oa]s?|crujientes?))*"
    r"(?:\s+(?:por\s+dentro|en\s+el\s+centro|por\s+completo|por\s+ambos\s+lados))?"
    r"(?:\s+y\s+(?:se\s+desmenucen?|se\s+separen?\s+en\s+lascas|suelten?\s+sus\s+jugos|"
    r"el\s+(?:centro|interior)\s+(?:a[uú]n\s+|siga\s+|quede\s+)?(?:suave|jugoso|tierno)))?"
    r"|se\s+desmenucen?)"
    r"(?:\s+(?:f[aá]cilmente|f[aá]cil|con\s+facilidad|con\s+un\s+tenedor|en\s+lascas))?",
    re.IGNORECASE)
_GROSOR = re.compile(r"\s+seg[uú]n\s+(?:el\s+|su\s+)?grosor", re.IGNORECASE)
# [P1-PLAN-LOTE-924 · 2026-09-30] La frase de punto del ave que escribe el 888 («hasta que el pollo alcance 74 °C por
# dentro y dore y esté cocido», con el «hasta que» del modelo fundido tras «y») se va ENTERA: `_TEMP_SEGURA` sólo quitaba
# la temperatura y el paso del enlatado decía «calienta las sardinas 2-3 minutos por dentro y dore y esté cocido» (batería
# real rdb932, día 8). La cadena «y dore / y esté… / y las sardinas esté humeante» es del crudo y se va; otro «hasta
# que» fundido («y el guiso espese») recupera su «hasta que»; una acción («y retira del fuego») se queda; lo que va tras
# una coma es otra cosa. Knob `MEALFIT_SUBST_POULTRY_DONENESS_TAIL` (True). tooltip-anchor: P1-PLAN-LOTE-924
_PUNTO_POR_DENTRO = re.compile(
    r"(?P<lead>,\s*|\s+)hasta\s+(?:que\s+(?:[^,;.()]{0,60}?\s+)?(?:alcancen?|lleguen?\s+a)\s+|(?:alcanzar|llegar\s+a)\s+)?"
    r"(?:6[0-9]|7[0-9])\s*°\s*C\s+por\s+dentro(?:\s+y\s+(?:(?:el|la|los|las)\s+[^\s,;.()]+\s+)?(?:no\s+)?(?:se\s+)?"
    r"(?:doren?|est[eé]n?|queden?|vean?|luzcan?|tengan?|suelten?|pierdan?)\b[^,;.()]*)*"
    r"(?:\s+y\s+(?P<otra>[^,;.()]+))?",
    re.IGNORECASE)


def _sin_punto_por_dentro(cl: str) -> str:
    def _f(m):
        otra = m.group("otra")
        if not otra:
            return ""
        if re.match(r"(?:el|la|los|las|un|una|unos|unas)\s", otra, re.IGNORECASE):
            return m.group("lead") + "hasta que " + otra                        # «y el guiso espese»
        return " y " + otra                                                    # «y retira del fuego»
    return _PUNTO_POR_DENTRO.sub(_f, cl)


def _punto_por_dentro_on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SUBST_POULTRY_DONENESS_TAIL", True)
    except Exception:                                                          # noqa: BLE001
        return True
_VEGETAL = re.compile(r"\b(?:cebollas?|tomates?|repollo|aj[ií]es?|pimientos?|morr[oó]n|vegetales|verduras|espinacas?|"
                      r"br[oó]coli|zanahorias?|berenjenas?|vainitas|molondrones|calabac[ií]n|tayota|auyama|puerros?|"
                      r"apio|coliflor|bok\s+choy|col|chayote|habichuelas|pepino)\b", re.IGNORECASE)
_PROTEINA_CRUDA = ("pollo", "pechuga", "pavo", "res", "cerdo", "pescado", "tilapia", "salmon", "mero", "camaron",
                   "huevo", "clara", "chivo", "chuleta", "bistec", "filete", "carne", "muslo", "merluza", "chillo")


def _sa(t) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(t or "")).lower()
    except Exception:
        return str(t or "").lower()


def _tolerante(palabra: str) -> str:
    return "".join(_VOCAL.get(ch, re.escape(ch)) for ch in palabra)


def _patron_token(tok: str) -> str:
    """«pechuga de pollo» → «pechugas? de pollo»; «camaron» → «camar[oó]n(?:es)?». El plural va en la PRIMERA palabra."""
    palabras = tok.split()
    p0 = palabras[0]
    plural = r"(?:es)?" if p0[-1] in "nlrzd" else r"s?"
    return r"\s+".join([_tolerante(p0) + plural] + [_tolerante(p) for p in palabras[1:]])


def _grupo_de(viejo_low: str):
    """(tokens del grupo de sustitutos al que pertenece la línea vieja, duradero) o (None, None)."""
    try:
        from compra_unica import SUSTITUTOS
    except Exception:
        return None, None
    for toks, rep in SUSTITUTOS:
        if any(re.search(r"\b" + re.escape(t) + r"s?\b", viejo_low) for t in toks):
            return toks, rep
    return None, None


def _genero_num(corte: str, nucleo: str):
    base = (corte or nucleo or "").strip()
    w = _sa(base).split()[0] if base else ""
    plural = (w.endswith("s") and w not in ("res",)) or w in ("mariscos",)
    singular = w
    for cand in (w, w[:-2] if w.endswith("es") else w, w[:-1] if w.endswith("s") else w):
        if cand in _GENERO:
            singular = cand
            break
    return _GENERO.get(singular, "m"), ("p" if plural else "s")


def _pron(g: str, n: str) -> str:
    return ("lo" if g == "m" else "la") + ("s" if n == "p" else "")


def _articulo(tipo: str, g: str, n: str) -> str:
    defi = {("m", "s"): "el", ("f", "s"): "la", ("m", "p"): "los", ("f", "p"): "las"}
    indef = {("m", "s"): "un", ("f", "s"): "una", ("m", "p"): "unos", ("f", "p"): "unas"}
    t = tipo.lower()
    if t in ("el", "la", "los", "las"):
        out = defi[(g, n)]
    elif t in ("un", "una", "unos", "unas"):
        out = indef[(g, n)]
    elif t == "del":
        out = "del" if (g, n) == ("m", "s") else "de " + defi[(g, n)]
    elif t == "al":
        out = "al" if (g, n) == ("m", "s") else "a " + defi[(g, n)]
    else:
        out = t
    return (out[:1].upper() + out[1:]) if tipo[:1].isupper() else out


def _inflexion(fin_viejo: str, g: str, n: str) -> str:
    fin = ("o" if g == "m" else "a") + ("s" if n == "p" else "")
    return fin.upper() if fin_viejo.isupper() else fin


def _regex_frase(tokens) -> re.Pattern:
    nucleos = "|".join(_patron_token(t) for t in sorted(set(tokens), key=len, reverse=True))
    return re.compile(
        r"(?P<art>\b(?:el|la|los|las|un|una|unos|unas|del|al)\s+)?"
        r"(?P<cant>" + _NUM + r"\s*(?:" + _UNIDADES + r"\.?\s+)?(?:de\s+)?)?"
        r"(?P<corte>" + _CORTES + r"(?:\s+(?:blanc|fresc|magr|enter)[oa]s?)?\s+de\s+)?"
        r"(?P<nucleo>\b(?:" + nucleos + r"))\b"
        r"(?P<cola>(?:\s+(?:blanc[oa]s?|fresc[oa]s?|magr[oa]s?|molid[oa]s?|deshuesad[oa]s?|limpi[oa]s?|"
        r"sin\s+piel|sin\s+hueso|de\s+pollo|de\s+pavo|de\s+res|de\s+cerdo|de\s+pescado(?:\s+blanco)?|"
        r"(?:ya\s+)?cocid[oa]s?|(?:ya\s+)?desmenuzad[oa]s?|enter[oa]s?|verdes?|madur[oa]s?|familiar(?:es)?|"
        # [P1-PLAN-LOTE-493] el tamaño es del fresco: «mide 1 tomate mediano» → «mide 60 g de salsa de tomate», sin «mediano»
        r"median[oa]s?|grandes?|pequeñ[oa]s?))*)"
        r"(?P<cant2>\s+de\s+" + _NUM + r"\s*(?:g|gr|gramos)\b)?"
        r"(?P<par>(?:\s*\((?:porci[oó]n|≈?\s*\d[^)]{0,20}|[a-záéíóúñ ]{2,20})\))*)",
        re.IGNORECASE)


def _cadena(texto: str, pos: int, g: str, n: str, listo: bool, liquido: bool = False):
    """Adjetivos y frases de método pegados al alimento desde `pos`: (texto nuevo de la cadena, fin)."""
    partes, j, primero = [], pos, True
    while True:
        m = _RX_ESLABON.match(texto, j)
        if not m:
            break
        adj, frase = m.group("adj"), m.group("frase")
        fuera = listo and (frase is not None or (adj is not None and _METODO_RAIZ.match(adj)))
        # [P1-PLAN-LOTE-466] una salsa no va «picada» ni «fresca»
        fuera = fuera or (liquido and adj is not None and bool(_ADJ_CORTE.match(adj)))
        if frase is not None and not fuera:
            break                       # «de pollo», «sin piel» que no se quitan: la cadena termina aquí
        if not fuera:
            conj = m.group("conj")
            sep = m.group("sep")
            if primero and conj:
                sep, conj = " ", None    # «dorada a la plancha y terminada» → «terminado»
            partes.append(sep + ((conj + " ") if conj else "") + ((m.group("bien") + " ") if m.group("bien") else "")
                          + adj + _inflexion(m.group("fin"), g, n))
            primero = False
        j = m.end()
    return "".join(partes), j


_RX_CLITICO = re.compile(r"\b([a-zñ]*[áéíóú][a-zñ]*?(?:[aeií]|ndo))(l[oa]s?)\b", re.IGNORECASE)


def _clitico(texto: str, desde: int, hasta: int, viejos: set, g: str, n: str) -> str:
    """«…y sazónalas», «mézclala»: el pronombre pegado al verbo que hablaba del fresco pasa al duradero."""
    nuevo = _pron(g, n)

    def sub(m):
        return m.group(1) + (nuevo if m.group(2).lower() in viejos else m.group(2))
    return texto[:desde] + _RX_CLITICO.sub(sub, texto[desde:hasta]) + texto[hasta:]


def _limpia_puntuacion(t: str) -> str:
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r",\s*([;.])", r"\1", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+(?:y|o|e|hasta)\s*([;.,])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([;.])", r"\1", t)
    t = re.sub(r";\s*;", ";", t)
    t = re.sub(r"[;,]\s*\.", ".", t)
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(r":\s*;\s*", ": ", t)
    return t.strip()


def _clausulas(texto: str):
    """Trozos (inicio, fin) separados por «;» o «. »."""
    out, i = [], 0
    for m in re.finditer(r"[;.](?=\s|$)", texto):
        out.append((i, m.end()))
        i = m.end()
    if i < len(texto):
        out.append((i, len(texto)))
    return out


def _es_split(resto: str) -> bool:
    """¿Lo que sigue a «con» son vegetales que se cocinan con su tiempo? (entonces el duradero entra al final)
    [P1-PLAN-LOTE-466] Sólo cuenta lo que va CON el duradero, hasta el siguiente verbo: en «saltea el pescado con el ajo
    en polvo 3 minutos, añade la cebolla…» los vegetales entran después y el ajo solo se quemaba 3 minutos."""
    cl = re.split(r"[;.]", resto, maxsplit=1)[0]
    cl = re.split(r",?\s+(?:y\s+)?(?:añade|agrega|incorpora|luego|después|despues|retira|sirve|reserva)\b", cl,
                  maxsplit=1, flags=re.IGNORECASE)[0]
    return bool(_VEGETAL.search(cl))


_CORTE_GENERO = (("tiras", "f"), ("lascas", "f"), ("piezas", "f"), ("laminas", "f"), ("porciones", "f"),
                 ("mitades", "f"), ("rodajas", "f"), ("rebanadas", "f"), ("cubos", "m"), ("cubitos", "m"),
                 ("dados", "m"), ("trozos", "m"), ("filetes", "m"), ("medallones", "m"))


def _sin_corte(out: str, fin: int, viejos: set):
    """((inicio, fin) del corte que sigue al alimento o vacío, pronombres del fresco + el del corte: «en tiras y
    sazónalas» hablaba de las tiras)."""
    mc = _CORTE_TRAS.match(out, fin)
    if not mc:
        return (fin, fin), viejos
    corte_txt = _sa(mc.group(0))
    viejos = set(viejos) | {("los" if gc == "m" else "las") for w, gc in _CORTE_GENERO if w in corte_txt}
    return (fin, mc.end()), viejos


def _fin_ambito(texto: str, desde: int) -> int:
    m = _FIN_AMBITO.search(texto, desde)
    return m.start() if m else len(texto)


def _es_producto(texto: str, m, productos, salsa_tomate: bool) -> bool:
    """¿La mención es un PRODUCTO hecho del alimento, no el fresco sustituido?
    [P1-PLAN-LOTE-466] «salsa de tomate», «caldo de pollo» no se reescriben.
    [P1-PLAN-LOTE-492 · 2026-09-27] …pero «el puré de plátano verde», «la salsa de pechuga de pavo» o «la salsa de
    espinacas» se HACEN en esta receta con el fresco: el nombre y el montaje seguían diciendo «plátano verde» con batata en
    la lista (replay forzado de los días 21+). Un caldo es despensa siempre; con la salsa de tomate de sustituto, toda
    preparación «de tomate» lo sigue siendo («salsa ligera de tomate», no «… de salsa de tomate»); lo demás, sólo si el
    producto es otra línea de la lista del plato. tooltip-anchor: P1-PLAN-LOTE-492-PRODUCTO-O-RECETA"""
    antes = texto[max(0, m.start() - 30):m.start()]
    if re.search(r"\b(?:caldo|consom[eé]|fondo|cubitos?)\s+(?:[a-záéíóúñ]+\s+)?de\s+$", antes, re.IGNORECASE):
        return True
    mp = re.search(r"\b(salsa|pasta|pur[eé]|jugo|sopa|crema)\s+(?:[a-záéíóúñ]+\s+)?de\s+$", antes, re.IGNORECASE)
    if not mp:
        return False
    if salsa_tomate:
        return True
    frase = _sa(mp.group(1)) + " de " + _sa(m.group("nucleo"))
    return any(frase in o for o in (productos or ()))


def _reescribe(texto: str, rx: re.Pattern, nueva: str, corto: str, g: str, n: str, listo: bool, *, paso: bool,
               productos=()):
    """Reescribe cada frase del fresco en `texto`. Devuelve (texto, hubo_cambio, pronombres del fresco)."""
    hits = [m for m in rx.finditer(texto) if not _es_producto(texto, m, productos, corto == "salsa de tomate")]
    if not hits:
        return texto, False, set()
    out = texto
    tocadas = []                                   # (inicio, fin) de cada reemplazo en `out`
    viejos_total = set()
    for m in reversed(hits):
        g0, n0 = _genero_num(m.group("corte"), m.group("nucleo"))
        art = m.group("art")
        con_cant = bool(m.group("cant") and m.group("cant").strip()) or bool(m.group("cant2"))
        frase = nueva if (con_cant and paso) else corto
        if art and frase is not nueva:
            frase = _articulo(art.strip(), g, n) + " " + frase
        elif (art and art.strip()[:1].isupper()) or m.group(0)[:1].isupper() or (m.start() == 0 and not paso):
            frase = frase[:1].upper() + frase[1:]
        liquido = corto in _LIQUIDOS
        cadena, fin = _cadena(out, m.end(), g, n, listo, liquido)
        ini = m.start()
        viejos = {_pron(g0, n0)}
        if paso and liquido:
            # [P1-PLAN-LOTE-466] «pica 2 tomates en cubos» → «mide 60 g de salsa de tomate»
            antes = out[:ini]
            mv = re.search(r"(\b[\wáéíóúñ]+)(\s+)$", antes)
            if mv and re.match(r"^(?:pica|corta|trocea|rebana|lava|lamina|ralla)$", mv.group(1), re.IGNORECASE):
                verbo = "Mide" if mv.group(1)[:1].isupper() else "mide"
                out = antes[:mv.start()] + verbo + mv.group(2) + out[ini:]
                delta = len(verbo) - len(mv.group(1))
                ini += delta
                fin += delta
            fin, viejos = _sin_corte(out, fin, viejos)
            out = out[:fin[0]] + out[fin[1]:]
            fin = fin[0]
        if paso and listo:
            antes = out[:ini]
            mv = re.search(r"(\b[\wáéíóúñ]+)(\s+)$", antes)
            if mv and _VERBOS_ESCURRE.match(mv.group(1)):
                verbo_orig = mv.group(1)
                verbo = "Escurre" if verbo_orig[:1].isupper() else "escurre"
                out = antes[:mv.start()] + verbo + mv.group(2) + out[ini:]
                delta = len(verbo) - len(verbo_orig)
                ini += delta
                fin += delta
                fin, viejos = _sin_corte(out, fin, viejos)
                out = out[:fin[0]] + out[fin[1]:]
                fin = fin[0]
                # el verbo repartía una lista («corta el pollo en cubos, la berenjena en dados…»): el resto conserva
                # su verbo y el duradero se escurre aparte
                mo = _OBJETO_SIGUE.match(out, fin)
                if mo:
                    out = out[:fin] + "; " + verbo_orig.lower() + " " + out[mo.end():]
            else:
                _lista466 = bool(_OBJETO_SIGUE.match(out, fin)) and _es_split(out[fin:])   # «la carne, el tomate y…»
                if mv and _VERBOS_CALIENTA.match(mv.group(1)) and not _lista466 and not (
                        re.match(r"\s+con\s+", out[fin:], re.IGNORECASE) and _es_split(out[fin:])):
                    verbo = "Calienta" if mv.group(1)[:1].isupper() else "calienta"
                    out = antes[:mv.start()] + verbo + mv.group(2) + out[ini:]
                    delta = len(verbo) - len(mv.group(1))
                    ini += delta
                    fin += delta
                # «sirve el pollo en rebanadas», «corta la carne en piezas de parrilla»: lo listo no se corta así
                fin, viejos = _sin_corte(out, fin, viejos)
                out = out[:fin[0]] + out[fin[1]:]
                fin = fin[0]
        out = out[:ini] + frase + cadena + out[fin:]
        tocadas.append((ini, ini + len(frase + cadena), viejos))
        viejos_total |= viejos
    if paso:
        # pronombres pegados al verbo en la misma cláusula, hasta que otro sustantivo con artículo tome el relevo
        # («…el tomate…; añade el atún y caliéntalo»: el «lo» ya es del atún)
        for ini, fin_r, viejos in tocadas:
            out = _clitico(out, fin_r, _fin_ambito(out, fin_r), viejos, g, n)
    else:
        # la descripción: «Pechuga desmenuzada en un guiso…, acompañada de» → el participio tras la coma concuerda
        for ini, fin_r, viejos in tocadas:
            fin_or = out.find(".", fin_r)
            fin_or = len(out) if fin_or < 0 else fin_or
            gv = {"lo": ("m", "s"), "la": ("f", "s"), "los": ("m", "p"), "las": ("f", "p")}
            olds = {gv[v] for v in viejos if v in gv}

            def _conc(mm):
                fin_v = mm.group(3).lower()
                gn = ("m" if fin_v.startswith("o") else "f", "p" if fin_v.endswith("s") else "s")
                if gn not in olds:
                    return mm.group(0)                     # habla de otro sustantivo («arepitas horneadas»)
                if listo and _METODO_DESC.match(mm.group(2)):
                    return ""                              # «…, horneada junto a», «y marcada a la plancha»: fuera
                if gn != (g, n):
                    return mm.group(1) + mm.group(2) + _inflexion(mm.group(3), g, n) + (mm.group(4) or "")
                return mm.group(0)
            trozo = re.sub(r"(,\s+|\s+y\s+)(acompañad|servid|terminad|cubiert|bañad|preparad|cocinad|aderezad|aliñad|"
                           r"combinad|mezclad|sazonad|marinad|estofad|guisad|marcad|dorad|asad|sellad|hornead|grillad|"
                           r"cocid)(os|as|o|a)\b((?:\s+(?:a\s+la\s+plancha|a\s+la\s+parrilla|al\s+horno|al\s+vapor|"
                           r"al\s+carb[oó]n|al\s+punto|a\s+fuego\s+lento|suavemente|lentamente))?)",
                           _conc, out[fin_r:fin_or], flags=re.IGNORECASE)
            out = out[:fin_r] + trozo + out[fin_or:]
    return out, True, viejos_total


def _cap_tiempo(cl: str, desde: int, rango: str = "2-3") -> str:
    """El primer tiempo tras `desde` baja a `rango` (lo listo sólo se calienta); los que ya caben se quedan."""
    tope = int(rango.split("-")[-1])

    def _t(mt):
        a_ = int(mt.group("a"))
        b_ = int(mt.group("b") or a_)
        if max(a_, b_) <= tope and "lado" not in mt.group(0).lower():
            return mt.group(0)
        return mt.group("pre") + rango + " " + ("minutos" if mt.group("u").lower().startswith("minut") else "min")
    return cl[:desde] + _TIEMPO.sub(_t, cl[desde:], count=1)


def _limpia_coccion(cl: str) -> str:
    if _punto_por_dentro_on():
        cl = _sin_punto_por_dentro(cl)                                        # [P1-PLAN-LOTE-924]
    cl = _TEMP_SEGURA.sub("", cl)
    cl = _HASTA_COCIDO.sub("", cl)
    cl = _GROSOR.sub("", cl)
    return re.sub(r"\b(?P<pre>(?:durante|unos|aproximadamente)\s+)?\d+(?:\s*(?:-|–|a)\s*\d+)?\s*(?P<u>min(?:utos?)?)\s+"
                  r"por\s+(?:cada\s+)?lado\b",
                  lambda m: (m.group("pre") or "") + "2-3 " + m.group("u"), cl, flags=re.IGNORECASE)


def _listo_en_paso(texto: str, corto: str, g: str, n: str, viejos: set, rx_otra) -> str:
    """Cláusulas del paso sobre el duradero ya listo (las que lo nombran y la que lo retoma con un pronombre): fuera
    temperatura segura, «hasta que esté cocido», «por lado»; «calienta» 2-3 min; un cocinado con vegetales conserva su
    verbo y su tiempo para ellos y el duradero entra al final."""
    rx_corto = re.compile(r"\b" + _tolerante(_sa(corto)) + r"\b", re.IGNORECASE)
    pron = _pron(g, n)
    art = {("m", "s"): "el", ("f", "s"): "la", ("m", "p"): "los", ("f", "p"): "las"}[(g, n)]
    partes, previa, bandeja = [], False, False
    for a, b in _clausulas(texto):
        cl = texto[a:b]
        nombra = bool(rx_corto.search(cl))
        retoma = None
        if not nombra and previa:
            for mc in _RX_CLITICO.finditer(cl):
                if mc.group(2).lower() in viejos:
                    # [P1-PLAN-LOTE-466] «Estira la masa en una tortilla fina y cocínala»: la «la» es de la tortilla
                    if not _FIN_AMBITO.search(cl[:mc.start()].replace(";", " ").replace(".", " ")):
                        retoma = mc
                    break
        if not nombra and retoma is None:
            # la temperatura segura de una cláusula sin otra proteína cruda también era del fresco sustituido
            if not (rx_otra and rx_otra.search(_sa(cl))) and _TEMP_SEGURA.search(cl):
                if _punto_por_dentro_on():
                    cl = _sin_punto_por_dentro(cl)                            # [P1-PLAN-LOTE-924]
                cl = _TEMP_SEGURA.sub("", cl)
            partes.append(cl)
            previa = False
            continue
        if retoma is not None:
            raiz = _sa(retoma.group(1))
            if raiz in _COCCION_CLITICO:
                verbo = "Caliénta" if retoma.group(1)[:1].isupper() else "caliénta"
                cl = cl[:retoma.start()] + verbo + pron + cl[retoma.end():]
                mp = re.match(r"(\s+)(tapad|cubiert)(os|as|o|a)\b", cl[retoma.start() + len(verbo + pron):],
                              re.IGNORECASE)
                if mp:
                    k = retoma.start() + len(verbo + pron)
                    cl = cl[:k] + mp.group(1) + mp.group(2) + _inflexion(mp.group(3), g, n) + cl[k + mp.end():]
                cl = _cap_tiempo(_limpia_coccion(cl), retoma.start())
            elif raiz in ("hornea", "gratina"):                                 # «hornéalo… 12-15 min» → 5-8
                cl = cl[:retoma.start()] + retoma.group(1) + pron + cl[retoma.end():]
                cl = _cap_tiempo(_limpia_coccion(cl), retoma.start(), "5-8")
            else:
                cl = cl[:retoma.start()] + retoma.group(1) + pron + cl[retoma.end():]
                cl = _limpia_coccion(cl)
            cl = _clitico(cl, retoma.start(), len(cl), viejos, g, n)     # «volteándolo» también
            partes.append(cl)
            previa = True
            continue
        cl = _limpia_coccion(cl)
        # [P1-PLAN-LOTE-469] «verifica que la carne de pescado esté cocida y lista para consumir» → «escurre las sardinas»:
        # lo que viene en lata o en su paquete ya está listo; no hay nada que verificar (y el participio no concordaba)
        mver = re.search(r"\b(?P<v>verifica|comprueba|confirma|aseg[uú]rate\s+de)\s+que\s+(?P<obj>(?:(?:el|la|los|las)\s+)?"
                         + _tolerante(_sa(corto)) + r")\s+est[eé]n?\s+(?:bien\s+|completamente\s+)?cocid[oa]s?"
                         r"(?:\s+y\s+list[oa]s?(?:\s+para\s+(?:consumir|comer))?)?", cl, re.IGNORECASE)
        if mver:
            verbo = "Escurre" if mver.group("v")[:1].isupper() else "escurre"
            cl = cl[:mver.start()] + verbo + " " + mver.group("obj") + cl[mver.end():]
        # sin la temperatura de por medio, el pronombre que la seguía ya está al alcance («…a fuego medio y déjala»)
        mn = rx_corto.search(cl)
        if mn:
            cl = _clitico(cl, mn.end(), _fin_ambito(cl, mn.end()), viejos, g, n)
        mv = re.search(r"\b(?P<v>saltea|sofr[ií]e|cocina|guisa|calienta|sella|dora|asa)\s+(?:(?:el|la|los|las)\s+)?"
                       + _tolerante(_sa(corto)) + r"\b[^,;.]*?\s+con\s+(?P<resto>[^;.]+)", cl, re.IGNORECASE)
        if not (mv and _es_split(mv.group("resto"))):
            # [P1-PLAN-LOTE-466] la misma cocción en LISTA: «cocina la carne, el tomate y la cebolla… 4-5 minutos»
            mv = re.search(r"\b(?P<v>saltea|sofr[ií]e|cocina|guisa|calienta|sella|dora|asa)\s+(?:(?:el|la|los|las)\s+)?"
                           + _tolerante(_sa(corto)) + r"\b(?:\s+en\s+lata|\s+en\s+agua|\s+cocid[oa]s)?\s*,\s+"
                           r"(?P<resto>[^;.]+)", cl, re.IGNORECASE)
        if mv and _es_split(mv.group("resto")):
            v = mv.group("v")
            if _sa(v) in ("sella", "dora", "asa"):
                v = "Saltea" if v[:1].isupper() else "saltea"
            resto = mv.group("resto").rstrip()
            cola = re.search(r"[;.]\s*$", cl)
            cl = (cl[:mv.start()] + v + " " + resto + f"; añade {art} {corto} al final y caliénta{pron} 1-2 min"
                  + (cola.group(0).strip() if cola else ""))
        else:
            # [P1-PLAN-LOTE-463] el tiempo baja sólo si el verbo es DEL duradero («calienta las sardinas 8-10 min»,
            # «hornéalas»); «Hornea 20-25 minutos» a una bandeja con vegetales es el tiempo de los vegetales
            obj = r"\s+(?:(?:el|la|los|las)\s+)?(?:\d[^;.,]{0,20}?\s+de\s+)?" + _tolerante(_sa(corto)) + r"\b"
            mcal = re.search(r"\b(?:calienta|saltea|sofr[ií]e)" + obj + r"|\bcali[eé]ntal(?:o|a|os|as)\b"
                             r"|\bsalt[eé]al(?:o|a|os|as)\b|\bsofr[ií]el(?:o|a|os|as)\b", cl,     # [P1-PLAN-LOTE-466]
                             re.IGNORECASE)
            mhor = re.search(r"\b(?:hornea|gratina)" + obj + r"|\bhorn[eé]al(?:o|a|os|as)\b|\bgrat[ií]nal(?:o|a|os|as)\b",
                             cl, re.IGNORECASE)
            if mcal:
                cl = _cap_tiempo(cl, mcal.end())
            elif mhor:
                cl = _cap_tiempo(cl, mhor.end(), "5-8")      # al horno, sólo hasta que se caliente
            # «Coloca las sardinas, el brócoli y la cebolla en una bandeja»: los vegetales hacen su horno completo y el
            # duradero entra al final (antes salía horneado 20-25 min, o la bandeja entera a 5-8)
            mb = re.search(r"(?:(?:el|la|los|las)\s+)?" + _tolerante(_sa(corto)) + r"\b(?:\s+en\s+lata|\s+en\s+agua|"
                           r"\s+cocid[oa]s)?\s*,\s+(?=[^;.]*\b(?:bandeja|fuente|molde|refractario)\b)", cl, re.IGNORECASE)
            if mb and _VEGETAL.search(cl[mb.end():]):
                cl = cl[:mb.start()] + cl[mb.end():]
                bandeja = len(partes)
        partes.append(cl)
        previa = True
    if bandeja is not False:
        frase_b = f" Añade {art} {corto} los últimos 5 minutos, solo para calentarl{pron[1:]}."
        # justo después del horno de la bandeja; si no se encuentra, al final del paso
        k = next((j for j in range(bandeja, len(partes)) if re.search(r"\bhorn(?:ea|éa|ea\w+)\b|\bhornea\b", partes[j],
                                                                          re.IGNORECASE)
                  and partes[j].rstrip().endswith(".")), None)
        if k is not None:
            partes[k] = partes[k].rstrip() + frase_b
        else:
            ult = "".join(partes).rstrip()
            partes = [(ult if ult.endswith(".") else ult + ".") + frase_b]
    return _limpia_puntuacion("".join(partes)) if partes else texto


_ESTADO = re.compile(r"\b(?P<v>est[eé]n?|queden?)(?P<adv>\s+(?:(?:bien|completamente|muy|ligeramente)\s+)?)"
                     r"(?P<raiz>tiern|cocid|bland|dorad|hech|list)(?P<fin>os|as|o|a)\b", re.IGNORECASE)


def _concuerda_estado(resto: str, g: str, n: str, viejos: set) -> str:
    """[P1-PLAN-LOTE-494 · 2026-09-27] «hierve la batata… hasta que esté tierno» (el participio era del plátano): el
    estado que la cláusula pide al alimento concuerda con el nuevo —«tierna»— y el verbo con su número («estén» →
    «esté»). Sólo si concordaba con el viejo. tooltip-anchor: P1-PLAN-LOTE-494-CONCORDANCIA"""
    gv = {"lo": ("m", "s"), "la": ("f", "s"), "los": ("m", "p"), "las": ("f", "p")}
    olds = {gv[v] for v in viejos if v in gv}
    if not olds or (g, n) in olds:
        return resto
    fin_cl = _fin_ambito(resto, 0)

    def _f(mm):
        fv = mm.group("fin").lower()
        if ("m" if fv.startswith("o") else "f", "p" if fv.endswith("s") else "s") not in olds:
            return mm.group(0)
        v = mm.group("v")
        tiene_n = v.lower().endswith("n")
        v2 = (v[:-1] if tiene_n else v) if n == "s" else (v if tiene_n else v + "n")
        return v2 + mm.group("adv") + mm.group("raiz") + _inflexion(mm.group("fin"), g, n)
    return _ESTADO.sub(_f, resto[:fin_cl]) + resto[fin_cl:]


def _sigue_pronombre(texto: str, corto: str, g: str, n: str, viejos: set) -> str:
    """[P1-PLAN-LOTE-466] Lo que no viene cocinado (batata, zanahoria…) no cambia su cocción, pero la cláusula que lo
    retoma con un pronombre sí cambia de género: «hierve la batata…; escúrrelo y májalo» → «escúrrela y májala»."""
    rx_corto = re.compile(r"\b" + _tolerante(_sa(corto)) + r"\b", re.IGNORECASE)
    partes, previa = [], False
    for a, b in _clausulas(texto):
        cl = texto[a:b]
        mc0 = rx_corto.search(cl)
        if mc0:
            partes.append(cl[:mc0.end()] + _concuerda_estado(cl[mc0.end():], g, n, viejos))
            previa = True
            continue
        if previa and any(mc.group(2).lower() in viejos for mc in _RX_CLITICO.finditer(cl)):
            cl = _clitico(cl, 0, len(cl), viejos, g, n)      # la cláusula entera retoma al alimento de la anterior
        partes.append(cl)
        previa = False
    return "".join(partes)


# [P1-PLAN-LOTE-495 · 2026-09-27] Las claras pasteurizadas (botella del súper) son la reserva de proteína de quien no come
# pescado en la compra única: se BATEN y se CUAJAN; no se cortan en filetes ni se sellan 6 minutos por lado hasta 74 °C.
_CLARAS_OBJ = re.compile(r"(?P<v>\b[a-záéíóúñ]+)(?P<sep>\s+)(?P<obj>(?:(?:las|unas)\s+)?(?:\d+\s+)?claras"
                         r"(?:\s+de\s+huevo)?)\b", re.IGNORECASE)
_V_CORTE_CLARAS = re.compile(r"^(?:corta|filetea|trocea|pica|limpia|lava|seca|deshuesa|aplana|porciona|prepara)$",
                             re.IGNORECASE)
_V_COCCION_CLARAS = re.compile(r"^(?:sella|dora|asa|grilla|hierve|fr[ií]e|cocina|saltea|sofr[ií]e|guisa)$", re.IGNORECASE)


def _claras_en_paso(texto: str) -> str:
    """«corta 90 g de pechuga de pollo en filetes finos» → «bate 3 claras de huevo»; «sella el pollo 6-7 minutos por lado
    hasta 74 °C» → «cocina las claras 2-3 minutos por lado, hasta que cuajen». tooltip-anchor: P1-PLAN-LOTE-495-CLARAS"""
    partes = []
    for a, b in _clausulas(texto):
        cl = texto[a:b]
        m = _CLARAS_OBJ.search(cl)
        if not m:
            partes.append(cl)
            continue
        v = m.group("v")
        if _V_CORTE_CLARAS.match(v):
            verbo = "Bate" if v[:1].isupper() else "bate"
            # el corte se va y el pronombre que lo retomaba («en filetes finos y sazónalos») pasa a las claras
            (c0, c1), viejos_c = _sin_corte(cl, m.end("obj"), set())
            if c1 > c0:
                cl = cl[:c0] + cl[c1:]
                cl = _clitico(cl, c0, _fin_ambito(cl, c0), viejos_c, "f", "p")
            # el verbo repartía una lista («corta el pollo, ½ ají morrón en tiras y ½ cebolla»): el resto conserva el suyo
            mo = _OBJETO_SIGUE.match(cl, c0)
            if mo:
                cl = cl[:c0] + "; " + v.lower() + " " + cl[mo.end():]
            cl = cl[:m.start("v")] + verbo + cl[m.end("v"):]
        elif _V_COCCION_CLARAS.match(v):
            verbo = "Cocina" if v[:1].isupper() else "cocina"
            cl = cl[:m.start("v")] + verbo + cl[m.end("v"):]
            cl = _limpia_coccion(cl)       # fuera la temperatura y «hasta que esté dorado/cocido»; «por lado» 2-3 min
            if "cuaj" not in _sa(cl):
                k = m.start("v") + len(verbo) + len(m.group("sep")) + len(m.group("obj"))
                # [P1-PLAN-LOTE-498] la plantilla del cerrador «Cocina X a la plancha o hervida y sírvela…»: unas claras
                # no se hierven sueltas — «Cocina claras, hasta que cuajen a la plancha o hervida» no se leía
                mp = re.match(r"\s+a\s+la\s+plancha(?:\s+o\s+hervid[oa]s?)?", cl[k:], re.IGNORECASE)
                if mp:
                    cl = cl[:k] + cl[k + mp.end():]
                mt = _TIEMPO.match(cl, k) or re.match(r"\s+" + _TIEMPO.pattern, cl[k:], re.IGNORECASE)
                if mt is not None and mt.re is _TIEMPO:
                    k = mt.end()
                elif mt is not None:
                    k = k + mt.end()
                cl = cl[:k] + ", hasta que cuajen" + cl[k:]
        partes.append(cl)
    return _limpia_puntuacion("".join(partes))


# [P1-PLAN-LOTE-497 · 2026-09-27] La IA escribió «extrae las semillas de 1 guineo… añade las semillas de guineo por
# encima» (batería real, perfil del dueño, día 13) y la sustitución lo volvió «añade las semillas de manzana»: servir las
# pepitas de una manzana. Lo que se extrae pasa a cortarse sin semillas y lo que se sirve es la manzana; descorazonarla
# («retira las semillas») sigue igual. tooltip-anchor: P1-PLAN-LOTE-497-SEMILLAS
_SEMILLAS_MANZANA = re.compile(
    r"\b(?:(?P<v>extrae|saca|separa|retira|quita|desecha|elimina)\s+)?(?:las\s+)?(?:semillas|pepitas|pulpa)\s+de\s+"
    r"(?P<obj>(?:\d+(?:[.,]\d+)?\s*g\s+de\s+|la\s+|una\s+|[\d½¼¾]+\s+)?manzanas?)\b", re.IGNORECASE)


def _manzana_sin_semillas(texto: str) -> str:
    def _f(m):
        v = (m.group("v") or "").lower()
        if v in ("retira", "quita", "desecha", "elimina"):
            return m.group(0)
        obj = m.group("obj")
        if v:
            return ("Corta" if m.group("v")[:1].isupper() else "corta") + f" {obj} en cubos, sin semillas"
        return obj if re.match(r"(?:\d|[½¼¾]|la\s|una\s)", obj, re.IGNORECASE) else "la " + obj
    return _SEMILLAS_MANZANA.sub(_f, texto)


def reescribir_plato(meal: dict, viejo: str, nueva: str, sub: str) -> int:
    """Nombre, descripción y pasos dejan de nombrar el fresco de `viejo` y de cocinar como crudo lo que ya viene
    listo. Devuelve cuántos textos cambió; 0 ante cualquier error (fail-open: la lista ya cambió)."""
    try:
        info = _DURADERO.get(sub)
        if not info or not isinstance(meal, dict):
            return 0
        corto, g, n, listo = info
        viejo_low = _sa(viejo)
        toks, _rep = _grupo_de(viejo_low)
        if not toks:
            return 0
        otras = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str) and _sa(x) != _sa(nueva)]
        productos = [_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]      # [P1-PLAN-LOTE-492]
        cand = [t for t in toks if re.search(r"\b" + re.escape(t) + r"s?\b", viejo_low)]
        proteina = listo or corto == "claras"                                               # [P1-PLAN-LOTE-495]
        if proteina:
            # la proteína: toda la clase que el plato nombra sin tener línea propia en la lista (menos los homónimos:
            # «dorado a la plancha» es un participio, no el pez — P1-REWRITE-DORADO-HOMONYM)
            cand += [t for t in toks if t not in _HOMONIMOS
                     and not any(re.search(r"\b" + re.escape(t) + r"s?\b", o) for o in otras)]
            cand += [h for h in _CABEZAS if re.search(r"\b" + h + r"s?\b", viejo_low)]
        # el sustituto que CONTIENE el token es el mismo alimento («leche» → «leche UHT»: nada que reescribir), salvo
        # un producto hecho de él («tomate» → «salsa de tomate»: el fresco sí se reescribe) [P1-PLAN-LOTE-466]
        cand = [t for t in dict.fromkeys(cand)
                if t and (corto in _LIQUIDOS or not re.search(r"\b" + re.escape(t) + r"s?\b", _sa(nueva)))]
        if not cand:
            return 0
        rx = _regex_frase(cand)
        otra = [p for p in _PROTEINA_CRUDA if any(re.search(r"\b" + p + r"(?:s|es)?\b", o) for o in otras)
                and not any(k in o for o in otras for k in ("en lata", "en agua", "cocid"))]
        rx_otra = re.compile(r"\b(?:" + "|".join(otra) + r")(?:s|es)?\b") if otra else None
        _cl524 = _CLAVE_524.get(sub)                                                         # [P1-PLAN-LOTE-524]
        _ls524 = [x for x in (meal.get("ingredients") or []) if isinstance(x, str) and _cl524 and re.search(_cl524, _sa(x))]
        linea524 = _ls524[0] if len(_ls524) == 1 else None
        cambios = 0
        for k in ("name", "desc", "description"):
            t = meal.get(k)
            if isinstance(t, str):
                q, hubo, _v = _reescribe(t, rx, nueva, corto, g, n, listo, paso=False, productos=productos)
                q = _limpia_puntuacion(_sin_repetir(q, corto, linea524))           # [P1-PLAN-LOTE-524]
                if hubo and q != t:
                    meal[k] = q
                    cambios += 1
        rec = meal.get("recipe")
        if isinstance(rec, list):
            # «⚠️ Seguridad alimentaria: cocina pechuga de pollo por completo…» sobre un atún en agua: sin crudo no hay
            # riesgo que advertir (las notas 🤰 generales del embarazo no nombran el plato y se quedan)
            if proteina:
                quedan = [p for p in rec if not (isinstance(p, str) and p.lstrip().startswith("⚠")
                                                 and "seguridad alimentaria" in _sa(p) and rx.search(p))]
                if len(quedan) != len(rec):
                    cambios += len(rec) - len(quedan)
                    rec[:] = quedan
            for i, p in enumerate(rec):
                if not isinstance(p, str) or p.lstrip().startswith(_NOTA):
                    continue
                q, hubo, viejos = _reescribe(p, rx, nueva, corto, g, n, listo, paso=True, productos=productos)
                if not hubo:
                    continue
                if listo:
                    q = _listo_en_paso(q, corto, g, n, viejos, rx_otra)
                else:
                    q = _sigue_pronombre(q, corto, g, n, viejos)                 # [P1-PLAN-LOTE-466]
                    if corto == "claras":
                        q = _claras_en_paso(q)                                   # [P1-PLAN-LOTE-495]
                    elif corto == "manzana":
                        q = _manzana_sin_semillas(q)                             # [P1-PLAN-LOTE-497]
                q = _limpia_puntuacion(_sin_repetir(q, corto, linea524))           # [P1-PLAN-LOTE-524]
                if q != p:
                    rec[i] = q
                    cambios += 1
        if cambios:
            meal.pop("_display", None)
        return cambios
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-460] reescritura no-op: {type(e).__name__}: {e}")
        return 0


_DURADERO_EN_TEXTO = ("en lata", "enlatad", "congelad", "seco", "secos", "en polvo", "deshidratad", "uht")


def parear_raw(meal: dict, viejo: str, nueva: str) -> str:
    """[P1-PLAN-LOTE-461] La línea de `ingredients_raw` del fresco sustituido pasa a `nueva`, buscada por ALIMENTO.
    «pareado» · «añadido» (no había pareja: la lista compra el duradero) · «sin_raw» · «error»."""
    try:
        raw = meal.get("ingredients_raw")
        if not isinstance(raw, list):
            return "sin_raw"
        toks, _rep = _grupo_de(_sa(viejo))
        if not toks:
            return "sin_raw"
        heads = [h for h in _CABEZAS if re.search(r"\b" + h + r"s?\b", _sa(viejo))]
        rx = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in list(toks) + heads) + r")s?\b")
        try:
            from compra_unica import _CLAVE_ROTACION
            claves = tuple(_CLAVE_ROTACION.values())
        except Exception:
            claves = ("atun", "sardina", "garbanzo", "lenteja")
        visibles = {_sa(x).strip() for x in (meal.get("ingredients") or []) if isinstance(x, str)}
        hits = []
        for j, r in enumerate(raw):
            if not isinstance(r, str):
                continue
            low = _sa(r)
            # [P1-PLAN-LOTE-463] «previamente congelado»/«descongelado» ya no es despensa (lote 289): su pareja sí sale
            duradero = [h for h in _DURADERO_EN_TEXTO if h in low
                        and not (h == "congelad" and re.search(r"previamente congelad|descongelad", low))]
            if not rx.search(low) or duradero or any(c in low for c in claves):
                continue
            if low.strip() in visibles and low.strip() != _sa(viejo).strip():
                continue                  # es la pareja exacta de OTRA línea visible
            hits.append(j)
        if not hits:
            raw.append(nueva)
            return "añadido"
        raw[hits[0]] = nueva
        for j in reversed(hits[1:]):
            del raw[j]                    # duplicados del mismo fresco: tampoco aguantan y la lista los compraba
        return "pareado"
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-461] raw no-op: {type(e).__name__}: {e}")
        return "error"


def _num(v) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(v)) or 0)
    except ValueError:
        return 0.0


def ajustar_macros(meal: dict, viejo: str, nueva: str, db) -> bool:
    """[P1-PLAN-LOTE-468 · 2026-09-27] Las macros del plato cambian con la línea sustituida: plato − vieja + nueva.
    El recálculo completo desde las líneas (`_truth_up_meal_macros_from_strings`) se NIEGA si una línea con nombre no
    trae cantidad sumable («Ajo, 1 diente picado», «Orégano dominicano, ½ cdta»), y el plato se quedaba con las macros
    del fresco: batería real del 27-sep (alérgico al pescado), «Bowl de garbanzos» con 286 g de garbanzos declaraba
    98 g de proteína — la de 1¾ pechugas. Si después el recálculo completo sí corre, manda él."""
    try:
        if db is None:
            return False
        a = db.macros_from_ingredient_string(str(viejo))
        b = db.macros_from_ingredient_string(str(nueva))
        if not a or not b:
            return False
        for k_plato, k_mc in (("protein", "protein"), ("carbs", "carbs"), ("fats", "fats"), ("cals", "kcal")):
            meal[k_plato] = max(0, round(_num(meal.get(k_plato)) - float(a.get(k_mc) or 0) + float(b.get(k_mc) or 0)))
        meal["macros"] = [f"P:{meal['protein']}g", f"C:{meal['carbs']}g", f"G:{meal['fats']}g"]
        return True
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-468] macros por diferencia no-op: {type(e).__name__}: {e}")
        return False


# [P1-PLAN-LOTE-496 · 2026-09-27] Dos líneas de proteína del MISMO plato («1 filete de pescado» + «75 g de camarones
# cocidos») recibían dos duraderos distintos —la rueda aparta lo que el día ya lleva— y el plato quedaba «Atún con
# zanahoria… Calienta atún y sírvela… Acompaña con atún y sardinas en lata» (8 de 1.240 comidas del replay forzado). Ahora
# la segunda recibe el MISMO duradero (`compra_unica.sustituir_linea(forzar=…)`) y su cantidad se suma a la línea que ya
# existe, en la lista y en `ingredients_raw`. tooltip-anchor: P1-PLAN-LOTE-496-UN-DURADERO-POR-PLATO
_CANT_LINEA = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(g|claras)\s+(?:de\s+)?(.+?)\s*$", re.IGNORECASE)


def _suma_lineas(a: str, b: str):
    """«150 g de atún en agua» + «75 g de atún en agua» → «225 g de atún en agua»; None si no son la misma medida."""
    ma, mb = _CANT_LINEA.match(str(a or "")), _CANT_LINEA.match(str(b or ""))
    if not (ma and mb) or ma.group(2).lower() != mb.group(2).lower() or _sa(ma.group(3)) != _sa(mb.group(3)):
        return None
    total = float(ma.group(1).replace(",", ".")) + float(mb.group(1).replace(",", "."))
    cifra = str(int(round(total))) if abs(total - round(total)) < 1e-6 or ma.group(2).lower() == "claras" \
        else f"{total:g}"
    return f"{cifra} {ma.group(2)} de {ma.group(3)}"


def _fusiona(lista, nueva: str) -> bool:
    """Si `nueva` está dos veces por medida en `lista` (la vieja duradera y la recién escrita), queda una con la suma."""
    if not isinstance(lista, list):
        return False
    pos = [j for j, x in enumerate(lista) if isinstance(x, str) and _suma_lineas(x, nueva) is not None]
    if len(pos) < 2:
        return False
    k = next((j for j in pos if str(lista[j]) == str(nueva)), pos[-1])
    j = next(j for j in pos if j != k)
    lista[j] = _suma_lineas(lista[j], lista[k])
    del lista[k]
    return True


# [P1-PLAN-LOTE-524 · 2026-09-27] El duradero que el plato YA traía. Replay forzado de los días 21+ (322 planes): en ~620
# de 3.519 comidas con sustitución el duradero quedaba DOS veces —«½ zanahoria» + «170 g de zanahoria» (pepino →
# zanahoria), «1 cdta de orégano» + «Orégano dominicano al gusto», «65 g de casabe» + «1½ tortas pequeñas de casabe»— y
# el paso decía «mezcla… el repollo, la zanahoria, la zanahoria» (217). Ahora la nueva se suma a la que había (misma
# medida, o gramos del catálogo); una «al gusto» cede su sitio a la que dice cuánto; si no se puede sumar, queda la que
# había. El texto no nombra dos veces el mismo duradero. tooltip-anchor: P1-PLAN-LOTE-524
_CLAVE_524 = {"zanahoria": r"zanahoria", "repollo": r"repollo", "manzana": r"manzana", "batata": r"batata",
              "casabe": r"casabe", "oregano": r"or[eé]gano", "salsa de tomate": r"salsa\s+de\s+tomate",
              "leche UHT": r"\bleche\b", "garbanzos cocidos": r"garbanzo", "lentejas cocidas": r"lenteja",
              "claras de huevo": r"\bclaras?\b", "atun en agua": r"at[uú]n", "sardinas en lata": r"sardina"}
_UNIDAD_524 = (r"(?:g|gr|gramos|kg|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|rebanadas?|lonjas?|piezas?"
               r"|porci[oó]n(?:es)?|unidad(?:es)?|latas?|pizcas?|tortas?(?:\s+peque[ñn]as?)?)")
# sólo una unidad conocida es unidad: «½ zanahoria rallada» se llama «zanahoria rallada», no «rallada»
_CANT_524 = re.compile(r"^\s*[\d.,/½¼¾⅓⅔]+\s*(?:" + _UNIDAD_524 + r"\.?\s+)?(?:de\s+)?", re.IGNORECASE)
_Q_524 = r"(?:\d+(?:[.,]\d+)?(?:\s*[½¼¾⅓⅔])?|[½¼¾⅓⅔])"
_ADJ_524 = (r"(?:\s+(?:en\s+(?:cubos|rodajas|tiras|juliana|trozos|láminas|laminas|bastones)|rallad[oa]s?|picad[oa]s?"
            r"|cortad[oa]s?|fresc[oa]s?|dominican[oa]s?|median[oa]s?|grandes?|peque[ñn][oa]s?|sec[oa]s?|molid[oa]s?"
            r"|fin[oa]s?|grues[oa]s?|delgad[oa]s?))*")
# el tamaño de la pieza no dice nada de una línea en gramos: «280 g de zanahoria», no «280 g de zanahoria mediana»
_TAMANO_524 = re.compile(r"\s+(?:median[oa]s?|grandes?|peque[ñn][oa]s?)\b", re.IGNORECASE)
_SINGULAR_524 = ("zanahoria", "manzana", "batata")


def _gramos_524(linea: str):
    try:
        import compra_unica as _cu
        return _cu._gramos_de_linea(linea)
    except Exception:
        return None


def _nombre_524(linea: str) -> str:
    return re.sub(r"\([^)]*\)", "", _CANT_524.sub("", str(linea or ""), count=1)).strip(" ,.")


def _nombre_gramos_524(linea: str) -> str:
    """«1 zanahoria mediana» → «zanahoria»; «3 zanahorias» → «zanahoria» (una línea en gramos no lleva tamaño ni plural)."""
    nom = re.sub(r"\s+", " ", _TAMANO_524.sub("", _nombre_524(linea))).strip(" ,.")
    cab, _, resto = nom.partition(" ")
    if cab.lower().endswith("s") and _sa(cab[:-1]) in _SINGULAR_524:
        nom = (cab[:-1] + " " + resto).strip()
    return nom


_FRAC_524 = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
_MEDIDA_524 = re.compile(r"^\s*(\d+(?:[.,]\d+)?)?\s*([½¼¾⅓⅔])?\s*(cdtas?|cdas?|cucharaditas?|cucharadas?|tazas?|rebanadas?"
                         r"|latas?|ml)\.?\s+(?:de\s+)?(.+?)\s*$", re.IGNORECASE)
_UNIDAD_CANON_524 = {"cucharadita": "cdta", "cucharada": "cda"}


def _fmt_524(x: float) -> str:
    ent = int(x + 1e-9)
    for s, v in _FRAC_524.items():
        if abs((x - ent) - v) < 0.02:
            return (str(ent) if ent else "") + s
    return str(ent) if abs(x - ent) < 0.02 else f"{x:.1f}".rstrip("0").rstrip(".")


def _suma_medida_524(vieja: str, nueva: str):
    """«½ cdta de orégano dominicano» + «1 cdta de orégano» → «1½ cdtas de orégano dominicano»; «1 cda» + «½ cdta» →
    «3½ cdtas» (1 cda = 3 cdtas; «cucharadita» es cdta); None si no es la misma medida de cuchara/taza/rebanada/lata/ml."""
    ma, mb = _MEDIDA_524.match(str(vieja or "")), _MEDIDA_524.match(str(nueva or ""))
    if not (ma and mb) or not (ma.group(1) or ma.group(2)) or not (mb.group(1) or mb.group(2)):
        return None
    uni = lambda m: _UNIDAD_CANON_524.get(m.group(3).lower().rstrip("s"), m.group(3).lower().rstrip("s"))
    q = lambda m: (float(m.group(1).replace(",", ".")) if m.group(1) else 0.0) + _FRAC_524.get(m.group(2) or "", 0.0)
    ua, ub = uni(ma), uni(mb)
    qa, qb = q(ma), q(mb)
    if {ua, ub} == {"cda", "cdta"}:
        qa, qb, ua, ub = (qa * 3 if ua == "cda" else qa), (qb * 3 if ub == "cda" else qb), "cdta", "cdta"
    if ua != ub:
        return None
    tot = qa + qb
    return f"{_fmt_524(tot)} {ua + ('s' if tot > 1 and ua != 'ml' else '')} de {ma.group(4)}"


def _nucleo_524(linea: str) -> str:
    """El alimento sin cantidad ni adjetivos: «½ zanahoria rallada» y «170 g de zanahoria» son «zanahoria»; «1 cda de
    vinagre de manzana» es «vinagre de manzana», no «manzana» (no se suman)."""
    t = _sa(_nombre_524(linea))
    t = re.sub(r"\bal\s+gusto\b", " ", t)
    t = re.sub(_ADJ_524[:-1] + r"+\b", " ", " " + t, flags=re.IGNORECASE)              # «+»: nunca un match vacío
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    return re.sub(r"(?<=[a-z])s\b", "", t)


def _fusiona_con_existente(meal: dict, nueva: str, sub: str, db=None) -> bool:
    ings = meal.get("ingredients")
    clave = _CLAVE_524.get(sub)
    if not isinstance(ings, list) or not clave:
        return False
    try:
        k = next(j for j, x in enumerate(ings) if isinstance(x, str) and x == nueva)
    except StopIteration:
        return False
    nucleo = _nucleo_524(nueva)
    otras = [j for j, x in enumerate(ings) if j != k and isinstance(x, str) and re.search(clave, _sa(x))
             and _nucleo_524(x) == nucleo]
    if not otras or not nucleo:
        return False
    j = otras[0]
    vieja = str(ings[j])
    suma = _suma_lineas(vieja, nueva) or _suma_medida_524(vieja, nueva)
    if suma is None and not re.match(r"^\s*[\d½¼¾⅓⅔]", vieja):
        suma = nueva                                  # «Orégano dominicano al gusto» cede a la que dice cuánto
    if suma is None:
        gv, gn = _gramos_524(vieja), _gramos_524(nueva)
        if gv and gn and gv > 0 and gn > 0:
            tot = gv + gn
            suma = f"{int(round(tot)) if tot < 20 else int(5 * round(tot / 5.0))} g de {_nombre_gramos_524(vieja)}"
    if suma is not None:
        ings[j] = suma
    else:
        _resta_524(meal, nueva, db)                               # la que sale ya no cuenta
    del ings[k]
    raw = meal.get("ingredients_raw")
    if isinstance(raw, list):
        rk = [i for i, x in enumerate(raw) if isinstance(x, str) and x == nueva]
        rj = [i for i, x in enumerate(raw) if isinstance(x, str) and x != nueva and re.search(clave, _sa(x))
              and _nucleo_524(x) == nucleo]
        if rk and rj:
            if suma is not None:
                raw[rj[0]] = suma
            del raw[rk[0]]
    return True


def _resta_524(meal: dict, linea: str, db) -> None:
    try:
        a = db.macros_from_ingredient_string(str(linea)) if db is not None else None
        if not a:
            return
        for k_plato, k_mc in (("protein", "protein"), ("carbs", "carbs"), ("fats", "fats"), ("cals", "kcal")):
            meal[k_plato] = max(0, round(_num(meal.get(k_plato)) - float(a.get(k_mc) or 0)))
        meal["macros"] = [f"P:{meal['protein']}g", f"C:{meal['carbs']}g", f"G:{meal['fats']}g"]
    except Exception:
        return


def _mencion_524(c: str, n: int) -> str:
    """Una mención del alimento `c` con cantidad («½ zanahoria», «170 g de zanahoria en cubos») o con artículo («la
    zanahoria»); el grupo `q{n}` sólo existe en la de cantidad y `adj{n}` guarda su preparación."""
    return (r"(?:(?<![\w½¼¾⅓⅔])(?P<q" + str(n) + r">" + _Q_524 + r")\s*(?:" + _UNIDAD_524 + r"\.?\s+)?(?:de\s+)?" + c
            + r"s?(?P<adj" + str(n) + r">" + _ADJ_524 + r")\b|\b(?:el|la|los|las)\s+" + c + r"s?(?:\s+dominican[oa])?\b)")


def _sin_repetir(texto: str, corto: str, linea=None) -> str:
    """«el repollo, la zanahoria y la zanahoria» → «el repollo y la zanahoria»; «el orégano y el orégano dominicano» →
    uno; «½ zanahoria, 170 g de zanahoria en cubos» → la cantidad de la línea que quedó en la lista («200 g de zanahoria
    en cubos»)."""
    c = _tolerante(_sa(corto))
    x = c + r"s?(?:\s+dominican[oa])?"
    rx = re.compile(r"(?P<pre>,\s+)?\b(?P<a>(?:el|la|los|las)\s+)?(?P<x>" + x + r")(?P<sep>,\s+con\s+|,\s+|\s+y\s+)"
                    r"(?:(?:el|la|los|las)\s+)?" + x + r"\b", re.IGNORECASE)

    def _art(m):
        pre = m.group("pre") or ""
        if pre and m.group("sep").strip() == "y":                  # «A, la X y la X» → «A y la X», no «A, la X»
            pre = " y "
        return pre + (m.group("a") or "") + m.group("x")

    texto = rx.sub(_art, texto)
    linea_txt = re.sub(r"\s*\([^)]*\)", "", str(linea or "")).strip(" ,.")
    if not re.match(r"^\s*[\d½¼¾⅓⅔]", linea_txt):
        linea_txt = ""
    rq = re.compile(r"(?P<m1>" + _mencion_524(c, 1) + r")(?:,\s+|\s+y\s+)(?P<m2>" + _mencion_524(c, 2) + r")",
                    re.IGNORECASE)

    def _cant(m):
        if not (m.group("q1") or m.group("q2")):
            return m.group(0)                                      # dos artículos: ya los colapsó `rx`
        if not linea_txt:
            return m.group("m1") if m.group("q1") else m.group("m2")
        lt = re.sub(r"\s+", " ", _TAMANO_524.sub("", linea_txt)).strip()
        if re.search(_ADJ_524[:-1] + r"+$", lt, flags=re.IGNORECASE):
            return lt                                              # la lista ya dice el corte: manda la lista
        return lt + _TAMANO_524.sub("", (m.group("adj2") or "") or (m.group("adj1") or ""))

    return rq.sub(_cant, texto)


def _menciones_524(meal: dict, corto: str) -> int:
    """Cuántas veces los pasos dan una CANTIDAD del alimento (o «la X restante»): con dos, el plato lo usa en dos
    preparaciones («ralla ½ zanahoria… corta 75 g de zanahoria») y una sola línea sumada contaría el total dos veces."""
    c = _tolerante(_sa(corto))
    rx = re.compile(r"(?<![\w½¼¾⅓⅔])" + _Q_524 + r"\s*(?:" + _UNIDAD_524 + r"\.?\s+)?(?:de\s+)?" + c + r"s?\b"
                    r"|\b(?:el|la|los|las)\s+" + c + r"s?\s+restantes?\b", re.IGNORECASE)
    return sum(len(rx.findall(p)) for p in (meal.get("recipe") or [])
               if isinstance(p, str) and not p.lstrip().startswith(_NOTA))


_FOTO_524 = ("ingredients", "ingredients_raw", "protein", "carbs", "fats", "cals", "macros")


class _DosPreparaciones524(Exception):
    """El plato usa el alimento en dos preparaciones: la suma se deshace (no es un error)."""


def sustituir_en_plato(meal: dict, idx: int, viejo: str, nueva: str, sub: str, db=None) -> None:
    """La línea visible `idx` pasa a `nueva`; su pareja en `ingredients_raw` también (por alimento); el plato deja de
    nombrar el fresco y sus macros cambian con la línea (lote 468). Marca `_fresh_substituted`."""
    ings = meal.get("ingredients")
    if isinstance(ings, list):
        # [P1-PLAN-LOTE-496] la fusión de abajo acorta la lista: el índice del bucle del llamador puede haber corrido
        if not (0 <= idx < len(ings)) or str(ings[idx]) != str(viejo):
            idx = next((k for k, x in enumerate(ings) if str(x) == str(viejo)), -1)
        if 0 <= idx < len(ings):
            ings[idx] = nueva
    ajustar_macros(meal, viejo, nueva, db)                                            # [P1-PLAN-LOTE-468]
    parear_raw(meal, viejo, nueva)                                                     # [P1-PLAN-LOTE-461]
    if _fusiona(ings, nueva):                                                          # [P1-PLAN-LOTE-496]
        _fusiona(meal.get("ingredients_raw"), nueva)
    else:
        import copy as _copy
        foto = {k: _copy.deepcopy(meal[k]) for k in _FOTO_524 if k in meal}
        try:
            if _fusiona_con_existente(meal, nueva, sub, db=db):                        # [P1-PLAN-LOTE-524]
                # el plato que usa el alimento en dos preparaciones conserva sus dos líneas: se ensaya la reescritura
                # sobre una copia y, si los pasos siguen dando dos cantidades, la suma se deshace
                prueba = _copy.deepcopy(meal)
                reescribir_plato(prueba, viejo, nueva, sub)
                corto = (_DURADERO.get(sub) or ("",))[0]
                if corto and _menciones_524(prueba, corto) >= 2:
                    raise _DosPreparaciones524()
        except Exception as e:                                                         # fail-open: sin sumar
            if not isinstance(e, _DosPreparaciones524):
                logger.debug(f"[P1-PLAN-LOTE-524] suma no-op: {type(e).__name__}: {e}")
            for k in _FOTO_524:
                if k not in foto:
                    meal.pop(k, None)
                elif isinstance(meal.get(k), list) and isinstance(foto[k], list):
                    meal[k][:] = foto[k]          # EN SITIO: el bucle del llamador guarda la referencia a la lista
                else:
                    meal[k] = foto[k]
    meal["_fresh_substituted"] = (meal.get("_fresh_substituted") or []) + [f"{str(viejo)[:40]} → {sub}"]
    reescribir_plato(meal, viejo, nueva, sub)                                          # [P1-PLAN-LOTE-460]
    if sub == "naranja":
        __import__("fruta_duradera").pulir_naranja(meal)                               # [P1-PLAN-LOTE-936] en gajos


__all__ = ["reescribir_plato", "parear_raw", "sustituir_en_plato"]
