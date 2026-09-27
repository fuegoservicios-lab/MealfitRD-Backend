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
}

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
    "panecillo": "m", "bagel": "m",
}
_CABEZAS = ("pechuga", "muslo", "filete", "chuleta", "bistec", "lomo")
#: tokens de la tabla que en el texto suelen ser OTRA palabra: sólo cuentan si la línea vieja los nombra
_HOMONIMOS = ("dorado", "mora", "lomo", "pan")
_CORTES = r"(?:cubos|trozos|tiras|dados|lascas|piezas?|pedazos?|filetes?|pechugas?|muslos?|lomos?|chuletas?|medallones|porci[oó]n(?:es)?)"
_NOTA = ("⚠", "💡", "🤰", "⚕", "🧊", "🛒", "🍽", "❄", "⏱")
_VOCAL = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]"}
_UNIDADES = (r"(?:g|gr|gramos|kg|oz|onzas?|lb|libras?|tazas?|unidad(?:es)?|porci[oó]n(?:es)?|piezas?|cdas?|cdtas?|"
             r"cucharadas?|cucharaditas?)")
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
                         r"porciones|medallones|mitades|rodajas)(?:\s+(?:finas?|finos?|gruesas?|gruesos?|pequeñ[oa]s|"
                         r"medianos?|medianas?|grandes|parejas?|parejos|uniformes?|de\s+\d+\s*cm))*", re.IGNORECASE)
_OBJETO_SIGUE = re.compile(r"(?:,\s+|\s+y\s+)(?=(?:\d|[½¼¾⅓⅔]|(?:el|la|los|las|un|una|unos|unas)\s))", re.IGNORECASE)
_TIEMPO = re.compile(r"(?P<pre>(?:durante\s+|unos\s+|por\s+|aproximadamente\s+)?)(?P<a>\d+)(?:\s*(?:-|–|a)\s*(?P<b>\d+))?\s*"
                     r"(?P<u>min(?:utos?)?\b)(?:\s+por\s+(?:cada\s+)?lado)?", re.IGNORECASE)
_TEMP_SEGURA = re.compile(
    r"(?:,\s*|\s+)?(?:\(\s*)?(?:\b(?:o|y)\s+)?"
    r"(?:(?:verifica(?:ndo)?|comprueba|aseg[uú]rate\s+de|revisa)\s+(?:con\s+(?:un\s+)?term[oó]metro\s+)?que\s+"
    r"(?:[^,;.()]{0,60}?\s+)?(?:alcance|llegue\s+a|marque|est[eé]\s+a)\s+"
    r"|hasta\s+(?:(?:alcanzar|llegar\s+a)\s+|que\s+(?:[^,;.()]{0,60}?\s+)?(?:alcance|llegue\s+a|marque)\s+)?"
    r"|(?:la\s+parte\s+m[aá]s\s+gruesa|el\s+centro|el\s+interior)[^,;.()]{0,40}?\s+(?:alcance|llegue\s+a)\s+)?"
    r"(?:6[0-9]|7[0-9])\s*°\s*C"
    r"(?:\s+(?:en\s+(?:el\s+centro|la\s+parte\s+m[aá]s\s+gruesa|el\s+interior|su\s+interior)|internos?|"
    r"de\s+temperatura\s+interna))?(?:\s+si\s+no\s+estaba\s+previamente\s+cocid[oa]s?)?(?:\s*\))?",
    re.IGNORECASE)
_HASTA_COCIDO = re.compile(
    r"(?:,\s*|\s+)(?:(?:o|y)\s+)?hasta\s+que\s+(?:"
    r"(?:[^,;.()]{0,50}?\s+)?(?:est[eé]n?|queden?|se\s+vean?|luzcan?|tengan?)\s+"
    r"(?:bien\s+|completamente\s+|totalmente\s+|ligeramente\s+)?"
    r"(?:opac|cocid|dorad|firme|blanc|hech|sellad|cocinad)[oa]s?"
    r"(?:\s+(?:por\s+dentro|en\s+el\s+centro|por\s+completo|por\s+ambos\s+lados))?"
    r"(?:\s+y\s+(?:se\s+desmenucen?|se\s+separen?\s+en\s+lascas|suelten?\s+sus\s+jugos))?"
    r"|se\s+desmenucen?)"
    r"(?:\s+(?:f[aá]cilmente|f[aá]cil|con\s+facilidad|con\s+un\s+tenedor|en\s+lascas))?",
    re.IGNORECASE)
_GROSOR = re.compile(r"\s+seg[uú]n\s+(?:el\s+|su\s+)?grosor", re.IGNORECASE)
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
        r"(?P<corte>" + _CORTES + r"\s+de\s+)?"
        r"(?P<nucleo>\b(?:" + nucleos + r"))\b"
        r"(?P<cola>(?:\s+(?:blanc[oa]s?|fresc[oa]s?|magr[oa]s?|molid[oa]s?|deshuesad[oa]s?|limpi[oa]s?|"
        r"sin\s+piel|sin\s+hueso|de\s+pollo|de\s+pavo|de\s+res|de\s+cerdo|de\s+pescado(?:\s+blanco)?|"
        r"(?:ya\s+)?cocid[oa]s?|(?:ya\s+)?desmenuzad[oa]s?|enter[oa]s?))*)"
        r"(?P<cant2>\s+de\s+" + _NUM + r"\s*(?:g|gr|gramos)\b)?"
        r"(?P<par>(?:\s*\((?:porci[oó]n|≈?\s*\d[^)]{0,20}|[a-záéíóúñ ]{2,20})\))*)",
        re.IGNORECASE)


def _cadena(texto: str, pos: int, g: str, n: str, listo: bool):
    """Adjetivos y frases de método pegados al alimento desde `pos`: (texto nuevo de la cadena, fin)."""
    partes, j, primero = [], pos, True
    while True:
        m = _RX_ESLABON.match(texto, j)
        if not m:
            break
        adj, frase = m.group("adj"), m.group("frase")
        fuera = listo and (frase is not None or (adj is not None and _METODO_RAIZ.match(adj)))
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
    """¿Lo que sigue a «con» son vegetales que se cocinan con su tiempo? (entonces el duradero entra al final)"""
    cl = re.split(r"[;.]", resto, maxsplit=1)[0]
    return bool(_VEGETAL.search(cl))


def _reescribe(texto: str, rx: re.Pattern, nueva: str, corto: str, g: str, n: str, listo: bool, *, paso: bool):
    """Reescribe cada frase del fresco en `texto`. Devuelve (texto, hubo_cambio, pronombres del fresco)."""
    hits = list(rx.finditer(texto))
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
        cadena, fin = _cadena(out, m.end(), g, n, listo)
        ini = m.start()
        viejos = {_pron(g0, n0)}
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
                mc = _CORTE_TRAS.match(out, fin)
                if mc:
                    corte_txt = _sa(mc.group(0))
                    for w, gc in (("tiras", "f"), ("lascas", "f"), ("piezas", "f"), ("laminas", "f"),
                                  ("porciones", "f"), ("mitades", "f"), ("rodajas", "f"), ("cubos", "m"),
                                  ("cubitos", "m"), ("dados", "m"), ("trozos", "m"), ("filetes", "m"),
                                  ("medallones", "m")):
                        if w in corte_txt:
                            viejos.add("los" if gc == "m" else "las")
                    out = out[:fin] + out[mc.end():]
                # el verbo repartía una lista («corta el pollo en cubos, la berenjena en dados…»): el resto conserva
                # su verbo y el duradero se escurre aparte
                mo = _OBJETO_SIGUE.match(out, fin)
                if mo:
                    out = out[:fin] + "; " + verbo_orig.lower() + " " + out[mo.end():]
            elif mv and _VERBOS_CALIENTA.match(mv.group(1)) and not (
                    re.match(r"\s+con\s+", out[fin:], re.IGNORECASE) and _es_split(out[fin:])):
                verbo = "Calienta" if mv.group(1)[:1].isupper() else "calienta"
                out = antes[:mv.start()] + verbo + mv.group(2) + out[ini:]
                delta = len(verbo) - len(mv.group(1))
                ini += delta
                fin += delta
        out = out[:ini] + frase + cadena + out[fin:]
        tocadas.append((ini, ini + len(frase + cadena), viejos))
        viejos_total |= viejos
    if paso:
        # pronombres pegados al verbo en la misma cláusula, hasta que otro sustantivo con artículo tome el relevo
        # («…el tomate…; añade el atún y caliéntalo»: el «lo» ya es del atún)
        for ini, fin_r, viejos in tocadas:
            corte = re.search(r"[;.]|\b(?:el|la|los|las|al|del)\s+[a-záéíóúñ]", out[fin_r:], re.IGNORECASE)
            fin_or = fin_r + corte.start() if corte else len(out)
            out = _clitico(out, fin_r, fin_or, viejos, g, n)
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
                if gn in olds and gn != (g, n):
                    return mm.group(1) + mm.group(2) + _inflexion(mm.group(3), g, n)
                return mm.group(0)
            trozo = re.sub(r"(,\s+)(acompañad|servid|terminad|cubiert|bañad|preparad|cocinad|aderezad|aliñad|"
                           r"combinad|mezclad)(os|as|o|a)\b", _conc, out[fin_r:fin_or], flags=re.IGNORECASE)
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
    partes, previa = [], False
    for a, b in _clausulas(texto):
        cl = texto[a:b]
        nombra = bool(rx_corto.search(cl))
        retoma = None
        if not nombra and previa:
            for mc in _RX_CLITICO.finditer(cl):
                if mc.group(2).lower() in viejos:
                    retoma = mc
                    break
        if not nombra and retoma is None:
            # la temperatura segura de una cláusula sin otra proteína cruda también era del fresco sustituido
            if not (rx_otra and rx_otra.search(_sa(cl))) and _TEMP_SEGURA.search(cl):
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
            else:
                cl = cl[:retoma.start()] + retoma.group(1) + pron + cl[retoma.end():]
                cl = _limpia_coccion(cl)
            cl = _clitico(cl, retoma.start(), len(cl), viejos, g, n)     # «volteándolo» también
            partes.append(cl)
            previa = True
            continue
        cl = _limpia_coccion(cl)
        mv = re.search(r"\b(?P<v>saltea|sofr[ií]e|cocina|guisa|calienta|sella|dora|asa)\s+(?:(?:el|la|los|las)\s+)?"
                       + _tolerante(_sa(corto)) + r"\b[^,;.]*?\s+con\s+(?P<resto>[^;.]+)", cl, re.IGNORECASE)
        if mv and _es_split(mv.group("resto")):
            v = mv.group("v")
            if _sa(v) in ("sella", "dora", "asa"):
                v = "Saltea" if v[:1].isupper() else "saltea"
            resto = mv.group("resto").rstrip()
            cola = re.search(r"[;.]\s*$", cl)
            cl = (cl[:mv.start()] + v + " " + resto + f"; añade {art} {corto} al final y caliénta{pron} 1-2 min"
                  + (cola.group(0).strip() if cola else ""))
        else:
            mcal = re.search(r"\b(?:calienta|saltea|sofr[ií]e)\b", cl, re.IGNORECASE)
            mhor = re.search(r"\b(?:hornea|horn[eé]al[oa]s?|gratina|gratínal[oa]s?)\b", cl, re.IGNORECASE)
            if mcal:
                cl = _cap_tiempo(cl, mcal.end())
            elif mhor:
                cl = _cap_tiempo(cl, mhor.end(), "5-8")      # al horno, sólo hasta que se caliente
        partes.append(cl)
        previa = True
    return _limpia_puntuacion("".join(partes)) if partes else texto


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
        cand = [t for t in toks if re.search(r"\b" + re.escape(t) + r"s?\b", viejo_low)]
        if listo:
            # la proteína: toda la clase que el plato nombra sin tener línea propia en la lista (menos los homónimos:
            # «dorado a la plancha» es un participio, no el pez — P1-REWRITE-DORADO-HOMONYM)
            cand += [t for t in toks if t not in _HOMONIMOS
                     and not any(re.search(r"\b" + re.escape(t) + r"s?\b", o) for o in otras)]
            cand += [h for h in _CABEZAS if re.search(r"\b" + h + r"s?\b", viejo_low)]
        cand = [t for t in dict.fromkeys(cand) if t and not re.search(r"\b" + re.escape(t) + r"s?\b", _sa(nueva))]
        if not cand:
            return 0
        rx = _regex_frase(cand)
        otra = [p for p in _PROTEINA_CRUDA if any(re.search(r"\b" + p + r"(?:s|es)?\b", o) for o in otras)
                and not any(k in o for o in otras for k in ("en lata", "en agua", "cocid"))]
        rx_otra = re.compile(r"\b(?:" + "|".join(otra) + r")(?:s|es)?\b") if otra else None
        cambios = 0
        for k in ("name", "desc", "description"):
            t = meal.get(k)
            if isinstance(t, str):
                q, hubo, _v = _reescribe(t, rx, nueva, corto, g, n, listo, paso=False)
                q = _limpia_puntuacion(q)
                if hubo and q != t:
                    meal[k] = q
                    cambios += 1
        rec = meal.get("recipe")
        if isinstance(rec, list):
            # «⚠️ Seguridad alimentaria: cocina pechuga de pollo por completo…» sobre un atún en agua: sin crudo no hay
            # riesgo que advertir (las notas 🤰 generales del embarazo no nombran el plato y se quedan)
            if listo:
                quedan = [p for p in rec if not (isinstance(p, str) and p.lstrip().startswith("⚠")
                                                 and "seguridad alimentaria" in _sa(p) and rx.search(p))]
                if len(quedan) != len(rec):
                    cambios += len(rec) - len(quedan)
                    rec[:] = quedan
            for i, p in enumerate(rec):
                if not isinstance(p, str) or p.lstrip().startswith(_NOTA):
                    continue
                q, hubo, viejos = _reescribe(p, rx, nueva, corto, g, n, listo, paso=True)
                if not hubo:
                    continue
                if listo:
                    q = _listo_en_paso(q, corto, g, n, viejos, rx_otra)
                q = _limpia_puntuacion(q)
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
            if not rx.search(low) or any(h in low for h in _DURADERO_EN_TEXTO) or any(c in low for c in claves):
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


def sustituir_en_plato(meal: dict, idx: int, viejo: str, nueva: str, sub: str) -> None:
    """La línea visible `idx` pasa a `nueva`; su pareja en `ingredients_raw` también (por alimento); el plato deja de
    nombrar el fresco. Marca `_fresh_substituted`."""
    ings = meal.get("ingredients")
    if isinstance(ings, list) and 0 <= idx < len(ings):
        ings[idx] = nueva
    parear_raw(meal, viejo, nueva)                                                     # [P1-PLAN-LOTE-461]
    meal["_fresh_substituted"] = (meal.get("_fresh_substituted") or []) + [f"{str(viejo)[:40]} → {sub}"]
    reescribir_plato(meal, viejo, nueva, sub)                                          # [P1-PLAN-LOTE-460]


__all__ = ["reescribir_plato", "parear_raw", "sustituir_en_plato"]
