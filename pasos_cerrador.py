# -*- coding: utf-8 -*-
"""Lo que escriben el cerrador y las sustituciones, contra el alimento que QUEDA (lotes 425-427, 2026-09-26).

La batería real sobre el 424 (dm2 con insulina + familia de 4), leída entera, salió en banda y con los medidores en 0, y
tenía tres familias de frases que ningún medidor leía:

  · 425 — «Cocina edamame en agua hasta que ablanden e incorpóralo al plato» con «215 g de edamame cocido» en la lista y
    «Acompaña con edamame» en el montaje: la plantilla del cerrador para una legumbre SECA, sobre un alimento que ya viene
    cocido, que a la vez se mete en el plato y se sirve al lado (214 comidas de 3.964; las 214 con el «Acompaña con»).
  · 426 — «Aparte, hierve pechuga de pollo 10-12 minutos, pélalos y desmenúzalos»: el tope de huevo cambió el huevo duro
    por pollo y la frase conservó el hervor y el pelado del huevo (y «hierve queso blanco… y pélalos»). La nota de
    seguridad del cambio decía «cocina pechuga de pollo por completo…; evita consumirlo crudo» (14).
  · 427 — «cocina el Yuca en agua con sal según el paquete hasta que quede suelto»: el arroz de noche pasó a yuca y la
    frase conservó el artículo, la mayúscula y la técnica del arroz.

Texto puro (no toca la lista ni los macros), fail-closed: ante la duda, la frase no se toca. Las notas (⚠ 💡 🌱 ⚕ 🤰)
sólo se leen, salvo la de seguridad del cambio de proteína (426). tooltip-anchor: P1-PLAN-LOTE-425-PASOS-CERRADOR
"""
from __future__ import annotations

import re
import unicodedata

_NOTAS = ("⚠", "💡", "🌱", "⚕", "🤰")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _es_nota(p) -> bool:
    return not isinstance(p, str) or any(e in p for e in _NOTAS)


def _pilar(p: str) -> str:
    t = _sa(p).lstrip()
    for k in ("mise en place", "el toque de fuego", "montaje"):
        if t.startswith(k):
            return k
    return ""


def _genero_numero(nombre: str) -> tuple:
    """(plural, femenino) del NÚCLEO («soya texturizada» → soya; «habichuelas rojas» → habichuelas)."""
    w = (_sa(nombre).split(" de ")[0].split() or [""])[0]
    masc = {"edamame", "arroz", "maiz", "brocoli", "calabacin", "casabe", "name", "pan", "tomate", "jitomate", "nabo",
            "puerro", "ajo", "platano", "guineo", "mapuey", "cebollin", "pimenton", "aji", "bulgur", "cuscus", "trigo"}
    plural = w.endswith("s") and w not in ("cuscus",)
    if w.rstrip("s") in masc or w in masc or re.sub(r"es$", "", w) in masc:
        return plural, False
    return plural, w.endswith("a") or w.endswith("as")


def _art(nombre: str) -> str:
    pl, fem = _genero_numero(nombre)
    return ("las" if fem else "los") if pl else ("la" if fem else "el")


def _pron(nombre: str) -> str:
    pl, fem = _genero_numero(nombre)
    return ("las" if fem else "los") if pl else ("la" if fem else "lo")


def _adj(nombre: str, raiz: str) -> str:
    pl, fem = _genero_numero(nombre)
    return raiz + (("as" if fem else "os") if pl else ("a" if fem else "o"))


def _este(nombre: str) -> str:
    return "estén" if _genero_numero(nombre)[0] else "esté"


# ── [P1-PLAN-LOTE-425 · 2026-09-26] La legumbre del cerrador: se prepara según lo que compra la lista, y una sola vez ─────
# `_closer_protein_step_text` escribe «Cocina {x} en agua hasta que ablanden e incorpóralo al plato» para una legumbre, y
# el cerrador añade además «Acompaña con {x}» al montaje. Con el edamame que la lista compra COCIDO (127 de las 214), con
# la soya texturizada (que se hidrata), con lo que una sustitución posterior puso en su lugar (brócoli, cebolla, quinoa
# seca: «Añade berenjena en agua hasta que ablanden…») o con la legumbre seca que un 💡 ya hierve, la frase dice algo que
# no se hace. Aquí la frase pasa a ser la preparación real del alimento que quedó: se calienta lo cocido, se hidrata la
# soya, se hierve el víver, se saltea lo del sofrito; sin «incorpóralo al plato» si el montaje ya lo sirve al lado; y se
# va si otro paso ya lo cocina. Lo mismo para «Incorpora lentejas secas al guiso y cocínalos… hasta que estén cocidos por
# dentro» cuando el 💡 ya las hirvió. tooltip-anchor: P1-PLAN-LOTE-425
_ABLANDEN_425_RE = re.compile(
    r"(?P<v>Cocina|Incorpora|Añade|Agrega)\s+(?P<obj>(?:(?!\b(?:[Cc]ocina|[Ii]ncorpora|[Aa]ñade|[Aa]grega)\b)[^.;:]){2,60}?)"
    r"\s+en\s+agua\s+hasta\s+que\s+ablanden\s+e\s+"
    r"incorp[oó]ral[oa]s?\s+al\s+plato(?:\s*\(~[^)]*\))?\.?", re.IGNORECASE)
_GUISO_LEG_425_RE = re.compile(
    r"(?P<v>Incorpora|Añade|Agrega)\s+(?P<obj>(?:lentejas|habichuelas|frijoles|garbanzos|gandules|arvejas|guisantes|habas)"
    r"[^.;:]{0,30}?)\s+al\s+guiso\s+y\s+coc[ií]nal[oa]s?\s+a\s+fuego\s+medio\s+12-15\s+minutos,\s+hasta\s+que\s+est[eé]n?\s+"
    r"cocid[oa]s?\s+por\s+dentro;\s*incorp[oó]ral[oa]s?\s+con\s+cuidado\s+para\s+no\s+deshacer\s+el\s+resto\.?",
    re.IGNORECASE)
_ACOMPANA_425_RE = re.compile(r"Acompaña con (?P<x>[^.;]+)\.")
_LEGUMBRES_425 = ("lenteja", "habichuela", "frijol", "garbanzo", "gandul", "haba", "arveja", "guisante", "chicharo",
                  "alubia", "judia", "poroto")
_GRANOS_425 = {"quinoa": ("12-15 minutos", "tierna"), "arroz": ("18-20 minutos", "tierno"), "pasta": None, "fideo": None,
               "espagueti": None, "macarron": None, "cebada": ("25-30 minutos", "tierna"), "trigo": ("15-20 minutos", "tierno"),
               "bulgur": ("12-15 minutos", "tierno"), "cuscus": None, "maiz": ("5-6 minutos", "tierno")}
_TUBERCULOS_425 = {"papa": "15-20", "batata": "12-15", "yuca": "20-25", "name": "20-25", "yautia": "20-25",
                   "malanga": "20-25", "platano": "18-20", "guineito": "15-18", "guineo": "15-18", "auyama": "10-12",
                   "mapuey": "20-25"}
_SOFRITO_425 = ("cebolla", "cebollin", "cebollita", "tomate", "jitomate", "aji", "pimiento", "pimenton", "ajo", "puerro",
                "apio", "chile")
_QUITAR_425_RE = re.compile(
    r",.*$|\b(?:en\s+crudo|previamente|y\s+fri[oa]s?|fri[oa]s?|crud[oa]s?|sec[oa]s?|picad[oa]s?|pelad[oa]s?\s+y\s+"
    r"cortad[oa]s?\s+en\s+\w+|en\s+(?:cubos|floretes|rodajas|tiras|granos|trozos)|median[oa]s?|grandes?|pequeñ[oa]s?)\b",
    re.IGNORECASE)
_COCIDO_425_RE = re.compile(r"\bcocid[oa]s?\b|\ben\s+lata\b|\bde\s+lata\b|\benlatad[oa]s?\b|\bprecocid[oa]s?\b")
_FUEGO_425_RE = re.compile(r"\b(?:calienta\w*|sofr[ie]\w*|salte\w*|sarten|olla|caldero|cazuela|wok)\b")


def _limpio_425(obj: str) -> str:
    x = re.sub(r"^\s*(?:el|la|los|las|un|una|unos|unas)\s+", "", str(obj), flags=re.IGNORECASE)
    x = _QUITAR_425_RE.sub(" ", x)
    x = re.sub(r"\s{2,}", " ", x).strip(" ,")
    return (x[:1].lower() + x[1:]) if x else ""


def _cabeza_425(nombre: str) -> str:
    return (_sa(nombre).split() or [""])[0]


def _clase_425(cab: str) -> str:
    if cab.startswith("edamame"):
        return "edamame"
    if cab.startswith(("soya", "soja")):
        return "soya"
    if any(cab.startswith(k) for k in _LEGUMBRES_425):
        return "legumbre"
    if any(cab.startswith(k) for k in _GRANOS_425):
        return "grano"
    if cab.startswith("casabe"):
        return "casabe"
    if any(cab.startswith(k) for k in _TUBERCULOS_425):
        return "tuberculo"
    if any(cab.startswith(k) for k in _SOFRITO_425):
        return "sofrito"
    return "verdura"


def _menciona(texto_sa: str, cab: str) -> bool:
    raiz = cab[:5] if len(cab) >= 5 else cab
    return bool(raiz) and re.search(r"\b" + re.escape(raiz), texto_sa) is not None


def _suf(nombre: str) -> str:
    """«lo» → «o», «la» → «a», «los» → «os», «las» → «as» (para «escúrrel…», «incorpóral…»)."""
    return _pron(nombre)[1:]


def _preparacion_425(nombre: str, clase: str, cocido: bool, lado: bool, con_fuego: bool) -> str:
    art, s = _art(nombre), _suf(nombre)
    tail_inc = "." if lado else f" e incorpóral{s} al plato."

    def escurre(base):
        return f"{base} y escúrrel{s}." if lado else f"{base}, escúrrel{s} e incorpóral{s} al plato."

    x = f"{art} {nombre}"
    xc = x if "cocid" in _sa(nombre) else f"{x} {_adj(nombre, 'cocid')}"
    if clase == "edamame":
        if cocido:
            return escurre(f"Calienta {xc} en agua hirviendo 2-3 minutos")
        return escurre(f"Cocina {x} en agua hirviendo 4-5 minutos")
    if clase == "soya":
        if cocido:
            return f"Calienta {xc} en la sartén 2-3 minutos{tail_inc}"
        return (f"Hidrata {x} en agua caliente 10 minutos y escúrrel{s} bien." if lado
                else f"Hidrata {x} en agua caliente 10 minutos, escúrrel{s} bien e incorpóral{s} al plato.")
    if cocido:
        return f"Calienta {xc} 2-3 minutos{tail_inc}"
    cab = _cabeza_425(nombre)
    if clase == "legumbre":
        if cab.startswith("lenteja"):
            return escurre(f"Cocina {x} en agua 20-25 minutos (no necesitan remojo), hasta que estén tiernas")
        return escurre(f"Remoja {x} 8-12 horas y hiérvel{s} 60-90 minutos (los primeros 10 a fuego fuerte), hasta que "
                       f"{_este(nombre)} {_adj(nombre, 'tiern')}")
    if clase == "grano":
        k = next((g for g in _GRANOS_425 if cab.startswith(g)), "")
        if _GRANOS_425.get(k) is None:
            return escurre(f"Cocina {x} en agua hirviendo con sal 8-10 minutos, hasta que {_este(nombre)} al dente")
        t, punto = _GRANOS_425[k]
        if k == "arroz" and "integral" in _sa(nombre):
            t = "25-30 minutos"
        return escurre(f"Cocina {x} en agua {t}, hasta que {_este(nombre)} {_adj(nombre, 'tiern')}")
    if clase == "casabe":
        return f"Tuesta {x} en una sartén seca 1-2 minutos por lado{tail_inc}"
    if clase == "tuberculo":
        k = next((t for t in _TUBERCULOS_425 if cab.startswith(t)), "papa")
        return escurre(f"Hierve {x} en agua con sal {_TUBERCULOS_425[k]} minutos, hasta que el cuchillo entre sin fuerza")
    if clase == "sofrito":
        return f"Añade {x} y cocina 3-4 minutos, hasta que se ablande{'n' if _genero_numero(nombre)[0] else ''}."
    if con_fuego:
        return f"Añade {x} y cocina 5-6 minutos, hasta que {_este(nombre)} {_adj(nombre, 'tiern')}."
    return f"Cocina {x} al vapor 4-5 minutos, hasta que {_este(nombre)} {_adj(nombre, 'tiern')}{',' if not lado else ''}" + (
        "." if lado else f" e incorpóral{s} al plato.")


def legumbre_del_cerrador(meal) -> int:
    """Nº de frases corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista = [str(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)]
        n = 0
        for i in range(len(rec)):
            p = rec[i]
            if not isinstance(p, str) or _es_nota(p):
                continue
            for rx in (_ABLANDEN_425_RE, _GUISO_LEG_425_RE):
                mm = rx.search(p)
                if not mm:
                    continue
                obj = mm.group("obj").strip()
                nombre = _limpio_425(obj)
                cab = _cabeza_425(nombre)
                if len(cab) < 3:
                    continue
                clase = _clase_425(cab)
                if rx is _GUISO_LEG_425_RE and clase != "legumbre":
                    continue
                lineas = [x for x in lista if _menciona(_sa(x), cab)]
                cocido = bool(_COCIDO_425_RE.search(_sa(obj))) or any(_COCIDO_425_RE.search(_sa(x)) for x in lineas)
                resto_sa = _sa(p[:mm.start()] + " " + p[mm.end():])
                otros_fuego = " ".join(_sa(q) for j, q in enumerate(rec)
                                       if j != i and isinstance(q, str) and not _es_nota(q) and _pilar(q) == "el toque de fuego")
                notas = " ".join(_sa(q) for q in rec if isinstance(q, str) and "💡" in q)
                montaje = " ".join(_sa(_ACOMPANA_425_RE.sub(" ", q)) for q in rec
                                   if isinstance(q, str) and not _es_nota(q) and _pilar(q) == "montaje")
                lado = any(_menciona(_sa(a.group("x")), cab) for q in rec if isinstance(q, str) and not _es_nota(q)
                           for a in _ACOMPANA_425_RE.finditer(q))
                en_fuego = _menciona(otros_fuego, cab) or (_pilar(p) == "el toque de fuego" and _menciona(resto_sa, cab))
                en_nota = _menciona(notas, cab)
                if rx is _GUISO_LEG_425_RE:
                    if not (en_nota or cocido):
                        continue
                    base = re.sub(r"\s+cocid[oa]s?\b", "", nombre)
                    nuevo = (f"Incorpora {_art(base)} {base} {_adj(base, 'cocid')} al guiso y cocínal{_suf(base)} a fuego "
                             f"medio 5 minutos para que tomen el sabor; remuéve{_pron(base)} con cuidado para no deshacer el resto.")
                elif en_fuego:
                    nuevo = ""
                elif en_nota and clase in ("legumbre", "grano", "edamame", "soya", "tuberculo"):
                    if lado or _menciona(montaje, cab):
                        nuevo = ""
                    else:
                        base = re.sub(r"\s+cocid[oa]s?\b", "", nombre)
                        nuevo = f"Incorpora {_art(base)} {base} {_adj(base, 'cocid')} al plato."
                else:
                    con_fuego = bool(_FUEGO_425_RE.search(_sa(p[:mm.start()])))
                    nuevo = _preparacion_425(nombre, clase, cocido, lado or _menciona(montaje, cab), con_fuego)
                s = p[:mm.start()] + nuevo + p[mm.end():]
                s = re.sub(r"\s{2,}", " ", s).strip()
                s = re.sub(r"\s+([.,;])", r"\1", s)
                if s != p:
                    rec[i] = p = s
                    n += 1
        n += _servido_una_vez_425(rec)
        n += _acompana_sin_preparar_425(rec, lista)
        if n:
            meal["recipe"] = [x for x in rec if not (isinstance(x, str) and re.fullmatch(r"\s*[^:.]{1,40}:\s*", x))]
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


#: las otras frases del cerrador que METEN el alimento en la preparación, con el «Acompaña con» del mismo alimento en el
#: montaje (corpus de 322 planes: 85 «Escurre e incorpora X (ya viene cocido) a la preparación», 46 «Agrega X a la
#: licuadora», 15 «Incorpora X a la preparación y mézclalo», 17 «Sirve X al lado para acompañar»): se sirve UNA vez.
_ESCURRE_425_RE = re.compile(
    r"(?:💪\s*)?Escurre e incorpora (?P<obj>[^.;()]{2,60}?) \((?P<ya>ya viene cocid[oa]s?)\) a la preparación antes de servir\.")
_MEZCLA_425_RE = re.compile(r"(?:💪\s*)?Incorpora (?P<obj>[^.;]{2,60}?) a la preparación y mézcl(?:alo|ala|alos|alas) antes de servir\.")
_SIRVE_LADO_425_RE = re.compile(r"(?:💪\s*)?Sirve (?P<obj>[^.;]{2,60}?) al lado para acompañar\.")
_LICUADORA_425_RE = re.compile(r"Agrega (?P<obj>[^.;]{2,60}?) a la licuadora y licúa hasta integrar\.")


def _acompana_items(rec: list) -> list:
    """[(índice del paso, match, [ítems])] de cada «Acompaña con A, B y C.» fuera de las notas."""
    out = []
    for j, q in enumerate(rec):
        if not isinstance(q, str) or _es_nota(q):
            continue
        for a in _ACOMPANA_425_RE.finditer(q):
            out.append((j, a, [x.strip() for x in re.split(r",\s*|\s+y\s+", a.group("x")) if x.strip()]))
    return out


def _servido_una_vez_425(rec: list) -> int:
    """La frase del cerrador y el «Acompaña con» del montaje no mandan el mismo alimento a dos sitios: lo que se escurre se
    escurre y se sirve al lado; lo que se mezcla «a la preparación» o se sirve «al lado» ya lo dice el montaje; lo que
    va a la licuadora no se sirve también al lado."""
    hechos = 0
    for rx in (_ESCURRE_425_RE, _MEZCLA_425_RE, _SIRVE_LADO_425_RE, _LICUADORA_425_RE):
        for i in range(len(rec)):
            p = rec[i]
            if not isinstance(p, str) or _es_nota(p):
                continue
            for mm in reversed(list(rx.finditer(p))):
                cab = _cabeza_425(_limpio_425(mm.group("obj")))
                if len(cab) < 3:
                    continue
                acs = [(j, a, items) for j, a, items in _acompana_items(rec)
                       if any(_menciona(_sa(x), cab) for x in items)]
                if not acs and rx in (_MEZCLA_425_RE, _SIRVE_LADO_425_RE):
                    montaje = " ".join(_sa(q) for q in rec if isinstance(q, str) and not _es_nota(q)
                                       and _pilar(q) == "montaje")
                    raiz = cab[:5] if len(cab) >= 5 else cab
                    if re.search(r"\bacompana(?:lo|la|los|las)?\s+con\b[^.;]{0,80}\b" + re.escape(raiz), montaje):
                        acs = [None]                         # «…y acompaña con el bol de yogurt»: ya lo sirve el montaje
                if not acs:
                    continue
                if rx is _LICUADORA_425_RE:
                    for j, a, items in reversed(acs):
                        quedan = [x for x in items if not _menciona(_sa(x), cab)]
                        q = rec[j]
                        nuevo = ("" if not quedan else "Acompaña con " + (", ".join(quedan[:-1]) + " y " + quedan[-1]
                                                                     if len(quedan) > 1 else quedan[0]) + ".")
                        rec[j] = re.sub(r"\s{2,}", " ", (q[:a.start()] + nuevo + q[a.end():])).strip()
                        hechos += 1
                    continue                           # [idempotencia] la siguiente frase de licuadora, en la misma pasada
                if rx is _ESCURRE_425_RE:
                    nombre = _limpio_425(mm.group("obj"))
                    pl = _genero_numero(nombre)[0]
                    nuevo = f"Escurre {_art(nombre)} {nombre} (ya viene{'n' if pl else ''} {_adj(nombre, 'cocid')})."
                else:
                    nuevo = ""
                s = re.sub(r"\s{2,}", " ", (p[:mm.start()] + nuevo + p[mm.end():])).strip()
                if s != p:
                    rec[i] = p = s
                    hechos += 1
    return hechos


def _acompana_sin_preparar_425(rec: list, lista: list) -> int:
    """«Acompaña con edamame» (crudo) o con soya texturizada (seca) que ningún paso ni 💡 prepara: la preparación va al
    final del último «El Toque de Fuego» (o en uno nuevo antes del montaje). El edamame COCIDO se come tal cual."""
    hechos = 0
    for q in list(rec):
        if not isinstance(q, str) or _es_nota(q) or _pilar(q) != "montaje":
            continue
        for a in _ACOMPANA_425_RE.finditer(q):
            nombre = _limpio_425(a.group("x"))
            cab = _cabeza_425(nombre)
            clase = _clase_425(cab)
            if clase not in ("edamame", "soya"):
                continue
            lineas = [x for x in lista if _menciona(_sa(x), cab)]
            if not lineas:
                continue
            cocido = any(_COCIDO_425_RE.search(_sa(x)) for x in lineas) or bool(_COCIDO_425_RE.search(_sa(a.group("x"))))
            if cocido:
                continue
            fuera = " ".join(_sa(r) for r in rec if isinstance(r, str) and _pilar(r) not in ("montaje", "mise en place"))
            if _menciona(fuera, cab):
                continue
            frase = _preparacion_425(nombre, clase, False, True, False)
            k = max((j for j, r in enumerate(rec) if isinstance(r, str) and not _es_nota(r)
                     and _pilar(r) == "el toque de fuego"), default=None)
            if k is not None:
                t = rec[k].rstrip()
                rec[k] = t + ("" if t.endswith((".", "!", "…")) else ".") + " " + frase
            else:
                m = next((j for j, r in enumerate(rec) if isinstance(r, str) and _pilar(r) == "montaje"), len(rec))
                rec.insert(m, "El Toque de Fuego: " + frase)
            hechos += 1
    return hechos


# ── [P1-PLAN-LOTE-427 · 2026-09-26] El víver que reemplaza al arroz de noche, con su artículo y su técnica ─────────────
# Batería REAL sobre el 424 (familia de 4, cena del día 2): el arroz de noche pasó a yuca (`_night_rice_autofix`) y el
# reemplazo de texto dejó «lava y mide el Yuca», «cocina el Yuca en agua con sal según el paquete hasta que quede suelto»
# e «Incorpora el Yuca cocido»: el artículo y el participio del arroz, la mayúscula del catálogo y la técnica de un grano
# (corpus: 10 comidas con «el Batata/el Yuca» o la mayúscula). `tecnica_del_sustituto` (lote 47) sólo sabía del casabe;
# ahora la batata, la yuca, la auyama y el ñame también quedan en minúscula, con su artículo, su participio, pelados en
# vez de enjuagados y hervidos hasta que el cuchillo entre, no «hasta que quede suelto». tooltip-anchor: P1-PLAN-LOTE-427
_TUB_427 = {"batata": ("f", "12-15"), "yuca": ("f", "20-25"), "auyama": ("f", "10-12"), "ñame": ("m", "20-25")}
_TUB_427_ALT = "batata|yuca|auyama|ñame"
_MAYUS_427_RE = re.compile(r"(?<=[a-záéíóúñü,;] )(Batata|Yuca|Auyama|Ñame)\b")
_ART_427_RE = re.compile(r"\b(el|del|al|un|El|Del|Al|Un)\s+(batata|yuca|auyama)\b")
_PART_427_RE = re.compile(r"\b(la|una|La|Una)\s+(batata|yuca|auyama)\s+(cocid|hervid|suelt|cocinad|tiern|lavad|medid|escurrid)o\b")
_GRANO_427_RE = re.compile(r"\b(?:suelt[oa]s?|absorba|absorbido|evapore|granos?|según (?:el paquete|las instrucciones del paquete))\b")
_ENJUAGA_427_RE = re.compile(r"\b(?:enjuaga|lava)(?:\s+y\s+mide)?\s+(?P<a>la|el)\s+(?P<t>" + _TUB_427_ALT + r")\b")
_CLAUSULA_GRANO_427_RE = re.compile(
    r"\s*(?:según (?:el paquete|las instrucciones del paquete)\s*)?(?:,\s*)?"
    r"(?:hasta que (?:quede|esté|estén|queden) suelt[oa]s?|hasta que (?:el agua )?se (?:absorba|evapore)(?: por completo)?"
    r"|hasta que el grano esté (?:tierno|suelto))")


def concordar_tuberculo(texto, tuberculo) -> str:
    """El texto de un paso tras cambiar el arroz por `tuberculo`: minúscula, artículo, participio y técnica del víver."""
    if not isinstance(texto, str) or not texto:
        return texto
    t0 = _sa(tuberculo).strip()
    t = "ñame" if t0 == "name" else t0
    if t not in _TUB_427:
        return texto
    s = _MAYUS_427_RE.sub(lambda m: m.group(1).lower(), texto)
    if _TUB_427[t][0] == "f":
        def _art_f(m):
            a = {"el": "la", "del": "de la", "al": "a la", "un": "una"}[m.group(1).lower()]
            if m.group(1)[:1].isupper():
                a = a[:1].upper() + a[1:]
            return f"{a} {m.group(2)}"
        s = _ART_427_RE.sub(_art_f, s)
        s = _PART_427_RE.sub(lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}a", s)
    s = _ENJUAGA_427_RE.sub(lambda m: f"pela y corta {m.group('a') if t == 'ñame' else 'la'} {m.group('t')} en trozos", s)
    frases = re.split(r"(?<=[.;])\s+", s)
    for k, f in enumerate(frases):
        fs = _sa(f)
        if t0 not in fs or "arroz" in fs or not _GRANO_427_RE.search(fs):
            continue
        g = _CLAUSULA_GRANO_427_RE.sub(f" {_TUB_427[t][1]} minutos, hasta que el cuchillo entre sin fuerza", f, count=1)
        g = re.sub(r"\s+según (?:el paquete|las instrucciones del paquete)", "", g)
        g = re.sub(r"\b([Cc])ocina (la|el) (" + _TUB_427_ALT + r") en agua",
                   lambda m: ("Hierve" if m.group(1) == "C" else "hierve") + f" {m.group(2)} {m.group(3)} en agua", g)
        frases[k] = re.sub(r"\s{2,}", " ", g)
    return " ".join(frases)


def tuberculo_del_arroz(meal) -> int:
    """Nº de pasos corregidos; 0 ante cualquier error. Sólo con un víver de la rotación del arroz de noche en la lista."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        lista = _sa(" ".join(str(x) for x in (meal.get("ingredients") or [])))
        tubs = [t for t in _TUB_427 if re.search(r"\b" + _sa(t) + r"\b", lista)]
        if not tubs:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _es_nota(p):
                continue
            s = p
            for t in tubs:
                s = concordar_tuberculo(s, t)
            if s != p:
                rec[i] = s
                n += 1
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-426 · 2026-09-26] Lo que el tope de huevo cambió por pollo, queso o yogur ya no se trata como huevo ────
# El tope de huevo (`_egg_cap_autofix`, `_replace_meal_egg_lines`) cambia la línea de huevo por «120 g de pechuga de
# pollo», «60 g de queso blanco» o «170 g de yogurt griego entero» y reescribe los pasos palabra por palabra. Las frases
# del huevo sobreviven con el alimento nuevo dentro (71 comidas de 3.964, batería real sobre el 424 incluida): «Aparte,
# hierve pechuga de pollo 10-12 minutos, pélalos y desmenúzalos», «casca los 2 queso blanco», «bate 1 pechuga de pollo
# entero con 1 pechuga de pollo», «escalfa el ½ pechuga de pollo… hasta que pechuga de pollo esté firme y pechuga de pollo
# aún suave», «corona con las mitades de pechuga de pollo duro», «hasta que el agua salga pechuga de pollo» (la «clara» del
# agua de enjuagar), y el nombre «Tostadas… con queso blanco fresco, aguacate, queso blanco cuajado y queso blanco». Aquí,
# sólo en platos con esa marca, cláusula por cláusula y EN SU SITIO: lo que casca o bate el sustituto pasa a cortarlo; lo
# que lo cuece como huevo pasa a su cocción (el pollo a la plancha hasta 74 °C, el queso dorado); lo que lo añade a la
# sartén lo añade en tiras y lo cocina hasta 74 °C; lo que lo pela, lo enfría o lo cuaja sale; el montaje lo sirve sin
# «duro» ni «mitades»; las notas del huevo se van o hablan del pollo; y el nombre pierde el adjetivo del huevo y el
# sustituto repetido. tooltip-anchor: P1-PLAN-LOTE-426
_SUST_426 = {"pollo": r"pechugas?\s+de\s+pollo",
             "queso": r"quesos?\s+blancos?(?:\s+(?:pasteurizados?|frescos?|bajos?\s+en\s+sodio))*",
             "yogurt": r"yogu?rt?\s+(?:griego|natural)(?:\s+enteros?)?(?:\s+pasteurizados?)?"}
_CANT_426 = r"(?:(?:el|la|los|las)\s+)?(?:(?:\d+(?:[.,]\d+)?\s*[½¼¾⅓⅔]?|[½¼¾⅓⅔])\s+)?"
_HUEVO_VOC_426 = re.compile(
    r"\b(?:bat(?:e|ir|id[oa]s?)|b[aá]tel[oa]s|casc\w*|romp\w*|cuaj\w*|revuelv\w*|revuelt[oa]s|escalf\w*|poch[ao]\w*|"
    r"estrellad\w*|p[eé]l(?:a|alo|ala|alos|alas)|pelad[oa]s|dur[oa]s?|firmes?|enter[oa]s|enfr[ií](?:a|alo|ala|alos|alas)|"
    r"yemas?|claras?|separa|mitades|hiérvel[oa]s|p[aá]sal[oa]s)\b|\b(?:en|por\s+la)\s+mitad\b|\ba\s+hervir\b|"
    r"\b(?:9|10)(?:\s*-\s*1[0-2])?\s*min", re.IGNORECASE)
_VERBO_426 = (r"(?:corta|pica|mide|casca|bate|separa|lava|pela|ralla|exprime|ten|prepara|pesa|reserva|precalienta|enjuaga|"
              r"machaca|rebana|desmenuza|sazona|mezcla|añade|agrega|incorpora|vierte|cocina|calienta|sofríe|saltea|hierve|"
              r"cuece|pon|escalfa|pocha|cuaja|revuelve|remueve|tapa|retira|dora|sirve|coloca|corona|acompaña|termina|"
              r"reparte|dobla|voltea|hornea|tuesta|enfría|deja|escurre|rompe|apaga|aparta|parte|lleva|sella|asa|"
              r"[a-záéíóúñ]*[áéíóú][a-záéíóúñ]*(?:lo|la|los|las))\b")
_CORTE_426_RE = re.compile(r"(,\s+(?:y\s+)?|\s+y\s+)(?=" + _VERBO_426 + ")", re.IGNORECASE)
_ANADE_426 = r"^\s*(?:añade|agrega|incorpora|vierte)\s+(?="
_CONECTOR_426_RE = re.compile(r"^\s*(?:aparte|en paralelo|mientras tanto|luego|después|por separado|al final)\s*$",
                              re.IGNORECASE)
_SERVIR_426 = {"pollo": "la pechuga de pollo en tiras", "queso": "el queso blanco en cubos"}
#: [contrato V1] el queso blanco es listo para comer: sólo se calienta si hay embarazo (74 °C), nunca se «dora»
_SERVIR_QUESO_EMBARAZO_426 = "el queso blanco caliente"
_MISE_426 = {"pollo": "corta la pechuga de pollo en tiras", "queso": "corta el queso blanco en cubos"}
_FUEGO_426 = {"pollo": ("cocina la pechuga de pollo a la plancha con unas gotas de aceite 6-7 minutos por lado, hasta que "
                        "alcance 74 °C en la parte más gruesa, y córtala en tiras"),
              "queso": "calienta el queso blanco en la sartén 1-2 minutos por lado, hasta que humee por dentro"}
_PUNTO_426 = {"pollo": "cocina 8-10 minutos, hasta que el pollo alcance 74 °C por dentro",
              "queso": "cocina 2-3 minutos más, hasta que el queso se ablande"}
_ANADIR_426 = {"pollo": "la pechuga de pollo en tiras", "queso": "el queso blanco en cubos"}
_PRON_426 = {"pollo": "la", "queso": "lo"}
#: un tiempo de huevo («a la plancha 3-4 min») no cocina una pechuga
_TIEMPO_HUEVO_426_RE = re.compile(r"\b[1-5](?:\s*-\s*[1-6])?\s*min", re.IGNORECASE)
#: lo que no es un alimento aunque lleve artículo («el centro», «la sartén»)
_NO_ALIMENTO_426_RE = re.compile(
    r"\b(?:el|la|los|las|un|una)\s+(?:centro|parte|fuego|sart[eé]n|olla|horno|plato|bowl|temperatura|mitad|resto|punto|"
    r"tapa|caldero|bandeja|molde|taz[oó]n|recipiente|superficie|lado|borde|medio|minuto|airfryer|freidora|plancha|"
    r"calor|vapor|agua)\b", re.IGNORECASE)
_GUISO_QUESO_426_RE = re.compile(
    r"\b(?:Añade|Incorpora|Agrega)\s+(?:el\s+)?queso\s+blanco[^.;]{0,25}?\s+al\s+guiso\s+y\s+cocínalo\s+a\s+fuego\s+medio\s+"
    r"12-15\s+minutos,\s+hasta\s+que\s+esté\s+cocido\s+por\s+dentro;\s*incorpóralo\s+con\s+cuidado\s+para\s+no\s+deshacer\s+"
    r"el\s+resto\.?", re.IGNORECASE)
_ADJ_HUEVO_426 = (r"(?:\s+(?:dur[oa]s?|bien\s+cocid[oa]s?|cocid[oa]s?|escalfad[oa]s?|pochad[oa]s?|revuelt[oa]s?|batid[oa]s?|"
                  r"cuajad[oa]s?(?:\s+al\s+airfryer)?|enter[oa]s))+(?:\s+cortad[oa]s?\s+(?:por\s+la\s+mitad|en\s+"
                  r"(?:mitades|rodajas|cuartos)))?(?:\s+en\s+(?:mitades|rodajas|cuartos))?")
_NOTA_HUEVO_426_RE = re.compile(r"cuaj|firme|yema|clara|escalfad|pochad|sin puntos liquidos")


def _label_426(meal) -> str:
    pa = str(meal.get("_protein_autofix_applied") or "") if isinstance(meal, dict) else ""
    lab = pa.split("->", 1)[1].strip().lower() if pa.startswith("huevo->") else ""
    return lab if lab in _SUST_426 else ""


def _clausulas(frase: str) -> list:
    """[(separador_previo, cláusula)] de una frase partida por «, <verbo>» y « y <verbo>»."""
    out, pos, sep = [], 0, ""
    for m in _CORTE_426_RE.finditer(frase):
        out.append((sep, frase[pos:m.start()]))
        sep, pos = m.group(1), m.end()
    out.append((sep, frase[pos:]))
    return out


_ADJ_SUELTO_426_RE = re.compile(r"^(?:\s*(?:dur[oa]s?|bien\s+cocid[oa]s?|cocid[oa]s?|escalfad[oa]s?|pochad[oa]s?|"
                                r"revuelt[oa]s?|batid[oa]s?|cuajad[oa]s?|enter[oa]s))+", re.IGNORECASE)


_OTRO_426_RE = re.compile(r"\b(?:el|la|los|las|un|una|unos|unas|\d+(?:[.,]\d+)?|[½¼¾⅓⅔])\s*"
                          r"(?!minutos?\b|min\b|segundos?\b|horas?\b|°)[a-záéíóúñ]{3,}|\bde\s+(?!la\b|el\b|los\b|las\b)"
                          r"[a-záéíóúñ]{4,}", re.IGNORECASE)
_CIERRE_426 = {"pollo": "Cocina la pechuga de pollo a la plancha o hervida y sírvela como proteína del plato",
               "queso": "Sirve el queso blanco en cubos como proteína del plato"}
_MEZCLA_426_RE = re.compile(r"^\s*(?:m[eé]zcl\w*|combin\w*|integr\w*|une)\b", re.IGNORECASE)


def _quitar_de_la_lista_426(c: str, lab: str, s_rx) -> str:
    """«…con el queso blanco, pechuga de pollo batidas, el ajo y la sal» → «…con el queso blanco, el ajo y la sal»."""
    item = "(?:" + s_rx.pattern + ")" + "(?:" + _ADJ_HUEVO_426 + ")?"
    t = re.sub(r"\bcon\s+" + item + r"\s*,\s*", "con ", c, flags=re.IGNORECASE)
    t = re.sub(r",\s*" + item + r"(?=\s*,)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+y\s+" + item + r"\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r",\s*" + item + r"(?=\s+y\s)", "", t, flags=re.IGNORECASE)
    t = re.sub(r",[^,;.]*\b(?:cuaj|firme|revuelv|revolv)\w*[^,;.]*$", "", t)
    if t != c and " y " not in t and "," in t:
        t = re.sub(r",\s*([^,]*)$", r" y \1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _reescribir_paso_426(paso: str, lab: str, s_rx, estado: dict) -> tuple:
    """(paso reescrito, nº de cláusulas tocadas). `estado` lleva, entre pasos, si ya se cortó (mise) y si ya se cocinó."""
    m = re.match(r"^(\s*(?:mise en place|el toque de fuego|montaje)[^:]{0,24}:\s*)", paso, re.IGNORECASE)
    pre, cuerpo = (m.group(1), paso[m.end():]) if m else ("", paso)
    es_mise = _pilar(paso) == "mise en place"
    fuego = _FUEGO_426.get(lab, "")
    if lab == "queso" and not estado.get("embarazo"):
        fuego = ""                                              # fuera del embarazo el queso no se cocina
    tocadas = 0
    if lab in _CIERRE_426:
        rx_c = re.compile(r"\b(?:Cocina|Incorpora|Añade|Agrega)\s+(?:el\s+|la\s+)?(?:" + _SUST_426[lab] + r")\s+a la plancha "
                          r"o hervid[oa]s?\s+y\s+sírvel[oa]s?\s+como proteína del plato", re.IGNORECASE)
        c2 = rx_c.sub(_CIERRE_426[lab], cuerpo)
        if lab == "queso":
            c2 = _GUISO_QUESO_426_RE.sub("Incorpora el queso blanco al guiso en el último minuto, solo para que se ablande.", c2)
        if c2 != cuerpo:
            tocadas += c2 != cuerpo and _CIERRE_426[lab] not in cuerpo
            cuerpo = c2
            estado["cocinado"] = True
    frases_out, arrastre = [], False
    for frase in re.split(r"(?<=[.;])\s+", cuerpo):
        fin = frase[-1] if frase and frase[-1] in ".;" else ""
        base = frase[:-1] if fin else frase
        if es_mise or (_OTRO_426_RE.search(s_rx.sub(" ", base)) and not s_rx.search(base)):
            arrastre = False
        if (not s_rx.search(base) and not arrastre) or (lab in _CIERRE_426 and _CIERRE_426[lab] in base
                                                         and not _HUEVO_VOC_426.search(base)):
            frases_out.append(frase)
            continue
        quedan, falta_punto, desmenuza, k_fuego = [], False, False, None
        for sep, c in _clausulas(base):
            tiene_s = bool(s_rx.search(c))
            sin_s = s_rx.sub(" ", c)
            voc = bool(_HUEVO_VOC_426.search(sin_s)
                       or (lab == "pollo" and tiene_s and _TIEMPO_HUEVO_426_RE.search(sin_s))
                       or (lab == "queso" and re.search(r"\ben\s+hilos\b", sin_s)))
            if "74 °C" in c or "se ablande" in c or "humee por dentro" in c:
                voc = False                                     # [idempotencia] la cláusula ya es la del sustituto
            otro = bool(_OTRO_426_RE.search(_NO_ALIMENTO_426_RE.sub(" ", sin_s)))
            anade = re.match(_ANADE_426 + s_rx.pattern + ")", c, re.IGNORECASE)
            if anade and not voc and re.search(s_rx.pattern + r"\s+en\s+(?:tiras|cubos)\b", c, re.IGNORECASE):
                anade = None                                    # [idempotencia] ya dice «la pechuga de pollo en tiras»
            directo = re.search(r"\b" + _VERBO_426 + r"\s+" + s_rx.pattern, c, re.IGNORECASE)
            if tiene_s and lab == "pollo" and not anade and _MEZCLA_426_RE.match(c) and otro:
                c2 = _quitar_de_la_lista_426(c, lab, s_rx)                 # el pollo crudo no va a la masa
                tocadas += c2 != c
                quedan.append((sep, c2))
                continue
            if tiene_s and (voc or (anade and lab == "pollo")):
                arrastre = True
                if lab == "yogurt":
                    tocadas += 1
                    continue
                if es_mise:
                    tocadas += 1
                    if not estado.get("mise"):
                        quedan.append((sep, _MISE_426[lab]))
                        estado["mise"] = True
                    continue
                if anade:
                    verbo = anade.group(0).strip()
                    if lab == "queso" and verbo.lower() == "vierte":
                        verbo = "Reparte" if verbo[:1].isupper() else "reparte"
                    resto = _ADJ_SUELTO_426_RE.sub("", s_rx.sub(" ", c[anade.end():]))
                    resto = re.sub(r"^\s*dorad[oa]s?\b", "", resto)
                    resto = re.sub(r"\ben\s+hilos\s*", "", resto)
                    resto = re.sub(r",[^,;.]*\b(?:cuaj|firme|revuelv|revolv)\w*[^,;.]*$", "", resto)
                    resto = re.sub(r"^\s*con\s*(?=$|,)", "", resto.strip())
                    resto = re.sub(r"\s+(?:con|y)\s*$", "", resto).strip()
                    nueva = f"{verbo} {_ANADIR_426[lab]}" + (f" {resto}" if resto else "")
                    tocadas += nueva != c.strip()
                    quedan.append((sep, nueva))
                    falta_punto = lab in _PUNTO_426
                    estado["cocinado"] = True
                    continue
                if otro and not directo:
                    c2 = _quitar_de_la_lista_426(c, lab, s_rx)
                    tocadas += c2 != c
                    quedan.append((sep, c2))
                    continue
                tocadas += 1
                if fuego and not estado.get("cocinado"):
                    k_fuego = len(quedan)
                    quedan.append((sep, fuego))
                    estado["cocinado"] = True
                continue
            if arrastre and not tiene_s and not otro:
                tocadas += 1
                desmenuza = desmenuza or "desmenuz" in _sa(c)
                if falta_punto and voc:
                    quedan.append((sep, _PUNTO_426[lab]))
                    falta_punto = False
                continue
            if arrastre and not tiene_s and lab in _PRON_426:            # «…y báñalos» del huevo: «báñalo»
                c = re.sub(r"\b([a-záéíóúñ]*[áéíóú][a-záéíóúñ]*?)l(?:os|as)\b", r"\1" + _PRON_426[lab], c)
            quedan.append((sep, c))
        if falta_punto:
            quedan.append((" y ", _PUNTO_426[lab]))
        if desmenuza and k_fuego is not None and lab == "pollo":
            sp, c = quedan[k_fuego]
            quedan[k_fuego] = (sp, c.replace("y córtala en tiras", "y desmenúzala"))
        while quedan and _CONECTOR_426_RE.match(quedan[-1][1]):
            quedan.pop()
        if not quedan:
            continue
        txt = quedan[0][1].strip()
        for sep, c in quedan[1:]:
            txt += (sep if sep.strip() else " ") + c.strip()
        txt = re.sub(r"^(?:y|,)\s+", "", txt.strip())
        if txt:
            frases_out.append(txt + fin)
        if fin == "." or es_mise:
            arrastre = False
    out = []
    for f in frases_out:
        if out and out[-1].endswith("."):
            f = f[:1].upper() + f[1:]
        elif out and out[-1].endswith(";"):
            f = f[:1].lower() + f[1:]
        out.append(f)
    cuerpo2 = " ".join(out).strip()
    if cuerpo2.endswith(";"):
        cuerpo2 = cuerpo2[:-1] + "."
    return (pre + cuerpo2 if cuerpo2 else ""), tocadas


def _nombre_sin_huevo_426(nombre: str, lab: str) -> str:
    s_rx = re.compile(_SUST_426[lab], re.IGNORECASE)
    n = re.sub("(" + _SUST_426[lab] + ")" + _ADJ_HUEVO_426, r"\1", nombre, flags=re.IGNORECASE)
    trozos = re.split(r"(,\s+|\s+y\s+|\s+con\s+|\s+de\s+)", n)
    vistos, out = [], []
    for k in range(0, len(trozos), 2):
        item = trozos[k]
        sep = trozos[k - 1] if k else ""
        clave = _sa(item).strip()
        if s_rx.fullmatch(item.strip() or "x") and any(clave == v or v.endswith(" " + clave) for v in vistos):
            if sep.strip() == "y" and out and out[-1][0].strip() == ",":
                out[-1] = (" y ", out[-1][1])                  # «…, aguacate, queso blanco y queso blanco» → «… y queso»
            continue
        vistos.append(clave)
        out.append((sep, item))
    res = "".join(sep + item for sep, item in out)
    return re.sub(r"\s{2,}", " ", res).strip()


def _nota_426(q: str, lab: str, s_rx) -> tuple:
    """(nota corregida o None si se va, cambió). Sólo las notas que heredó el huevo."""
    qs = _sa(q)
    if "🌱" in q and "usa solo" in qs and s_rx.search(q):
        return None, True                                           # «NO botes pechuga de pollo»: no sobran yemas
    if s_rx.search(q) and _NOTA_HUEVO_426_RE.search(qs):
        cabeza, _, resto = q.partition(":")
        clausulas = [c for c in re.split(r";\s*", resto) if c.strip()]
        quedan = [c for c in clausulas if not (s_rx.search(c) and _NOTA_HUEVO_426_RE.search(_sa(c)))]
        if lab == "pollo" and len(quedan) < len(clausulas) and "74" not in _sa(" ".join(quedan)):
            quedan.insert(0, "la pechuga de pollo debe alcanzar 74 °C en la parte más gruesa")
        if not quedan:
            return None, True
        t = "; ".join(c.strip().rstrip(".") for c in quedan)
        if lab == "pollo":
            t = t.replace("hasta que estén bien cocidos", "hasta que esté bien cocida")
        return f"{cabeza}: {t}.", True
    if re.search(r"cocina " + _SUST_426[lab] + r" por completo antes de servir; evita consumirl", q, re.IGNORECASE):
        if lab != "pollo":
            return None, True                                       # queso/yogur: nada que advertir
        return (re.sub(r"cocina pechugas? de pollo por completo antes de servir; evita consumirl[oa]s? crud[oa]s? o poco "
                       r"cocid[oa]s?", "cocina la pechuga de pollo por completo antes de servir; evita consumirla cruda o "
                                       "poco cocida", q), True)
    return q, False


def huevo_sustituido(meal) -> int:
    """Nº de cambios; 0 sin la marca «huevo->pollo/queso/yogurt» o ante cualquier error."""
    try:
        lab = _label_426(meal)
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not lab or not isinstance(rec, list) or not rec:
            return 0
        # [P1-PLAN-LOTE-460] la marca «huevo->pollo» es de CUANDO se cambió el huevo; si después la compra única cambió
        # esa pechuga por atún, la lista ya no la trae y este reparador la devolvía al plato («Cocina la pechuga… 74 °C»
        # y «Acompaña con la pechuga de pollo en tiras» sobre una lista con atún). Sin el sustituto en la lista, nada.
        if not re.search(_SUST_426[lab], " ".join(_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str)),
                         re.IGNORECASE):
            return 0
        s_rx = re.compile(_CANT_426 + "(?:" + _SUST_426[lab] + ")", re.IGNORECASE)
        estado = {"embarazo": any("embarazo" in _sa(x) for x in rec if isinstance(x, str))}
        servir = (_SERVIR_QUESO_EMBARAZO_426 if lab == "queso" and estado["embarazo"]
                  else _SERVIR_426.get(lab, ""))
        for x in rec:                                              # el sustituto ya cocinado de verdad (no con tiempo de huevo)
            if isinstance(x, str) and not _es_nota(x) and _pilar(x) == "el toque de fuego":
                for f in re.split(r"(?<=[.;])\s+", x):
                    if (re.search(r"\b(?:cocin|dor|asa|sell|horne|hierv|salte|guis)\w*\s+(?:\S+\s+){0,3}?" + s_rx.pattern, f,
                                  re.IGNORECASE)
                            and not _HUEVO_VOC_426.search(s_rx.sub(" ", f))
                            and not (lab == "pollo" and _TIEMPO_HUEVO_426_RE.search(f))):
                        estado["cocinado"] = True
        n = 0
        nuevos = []
        for p in rec:
            if not isinstance(p, str):
                nuevos.append(p)
                continue
            q = re.sub(r"\b(salga|sale|salgan|quede|queda|corra|corre)\s+(?:" + _SUST_426[lab] + ")", r"\1 clara", p,
                       flags=re.IGNORECASE)                                        # «hasta que el agua salga clara»
            n += q != p
            if _es_nota(q):
                q2, cambio = _nota_426(q, lab, s_rx)
                n += cambio
                if q2 is not None:
                    nuevos.append(q2)
                continue
            if _pilar(q) == "montaje":
                if lab in _SERVIR_426:
                    q2 = re.sub(r"(?:(?:las|los)\s+(?:mitades|rodajas|cuartos)\s+de\s+|(?:los|las|el|la)\s+)?(?:\d+\s+)?(?:"
                                + _SUST_426[lab] + ")" + _ADJ_HUEVO_426, servir, q, flags=re.IGNORECASE)
                    q2 = re.sub(r"\bde el\b", "del", re.sub(r"\ba el\b", "al", q2))
                    n += q2 != q
                    q = q2
                nuevos.append(q)
                continue
            q2, k = _reescribir_paso_426(q, lab, s_rx, estado)
            n += k
            if q2:
                nuevos.append(q2)
        if not n:
            return 0
        if lab == "pollo" and not estado.get("cocinado"):
            prep = _FUEGO_426["pollo"][:1].upper() + _FUEGO_426["pollo"][1:] + "."
            k = max((j for j, x in enumerate(nuevos) if isinstance(x, str) and _pilar(x) == "el toque de fuego"),
                    default=None)
            if k is not None:
                t = nuevos[k].rstrip()
                nuevos[k] = t + ("" if t.endswith(".") else ".") + " " + prep
            else:
                m_i = next((j for j, x in enumerate(nuevos) if isinstance(x, str) and _pilar(x) == "montaje"), len(nuevos))
                nuevos.insert(m_i, "El Toque de Fuego: " + prep)
            estado["servir"] = True
        if lab == "pollo":
            nuevos = [re.sub(r"\b(pechugas? de pollo) (desmenuzad|cocid|picad|cortad|trocead|deshebrad|dorad|sellad|asad|"
                             r"hervid)o(s?)\b", r"\1 \2a\3", x) if isinstance(x, str) and not _es_nota(x) else x
                      for x in nuevos]
        todo = _sa(" ".join(x for x in nuevos if isinstance(x, str) and not _es_nota(x)))
        montaje = _sa(" ".join(x for x in nuevos if isinstance(x, str) and _pilar(x) == "montaje"))
        if lab in _SERVIR_426 and (not re.search(_SUST_426[lab], todo)
                                   or (estado.get("servir") and not re.search(_SUST_426[lab], montaje))):
            m_i = next((j for j, x in enumerate(nuevos) if isinstance(x, str) and _pilar(x) == "montaje"), None)
            if m_i is not None:
                t = nuevos[m_i].rstrip()
                nuevos[m_i] = t + ("" if t.endswith(".") else ".") + f" Acompaña con {servir}."
        meal["recipe"] = nuevos
        nombre = str(meal.get("name") or "")
        if nombre:
            nn = _nombre_sin_huevo_426(nombre, lab)
            if nn and nn != nombre:
                meal["name"] = nn
        meal.pop("_display", None)
        return n
    except Exception:
        return 0



# ── [P1-PLAN-LOTE-443 · 2026-09-26] Las notas que hablan de otro plato ─────────────────────────────────────────────────
# Replay de la cola sobre 322 planes: «⚠️ … NO licúes yuca, víveres ni leguminosas CRUDOS en un batido… Elimina este
# batido» en un «Bowl tropical de yautía con huevo duro», unos «Tacos integrales» y unas «Brochetas de pollo con yuca» (3):
# la nota nació con una versión del plato que sí licuaba y el plato cambió; y la de sodio «enjuaga los enlatados (yogurt
# griego sin azúcar, granos)» (4, planes de antes del lote 345): el yogur no se enjuaga. La del batido se va si ningún paso
# licúa; de la de sodio salen los lácteos del paréntesis. tooltip-anchor: P1-PLAN-LOTE-443
_NOTA_BATIDO_443 = "NO licúes yuca, víveres ni leguminosas CRUDOS en un batido"
_LICUA_443_RE = re.compile(r"\b(?:licu\w*|licú\w*|batido|smoothie|licuadora)\b")
_ENLATADOS_443_RE = re.compile(r"(enjuaga los enlatados \()([^)]*)(\))")
_LACTEO_443_RE = re.compile(r"\b(?:yogu?rt?|queso|leche|cottage|ricotta|kefir|crema)\b")


def notas_de_otro_plato(meal) -> int:
    """Nº de notas quitadas o corregidas; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list) or not rec:
            return 0
        pasos = _sa(" ".join(str(p) for p in rec if isinstance(p, str) and not _es_nota(p)) + " " +
                    str(meal.get("name") or ""))
        n, nuevos = 0, []
        for p in rec:
            if isinstance(p, str) and _NOTA_BATIDO_443 in p and not _LICUA_443_RE.search(pasos):
                n += 1
                continue
            if isinstance(p, str) and "enjuaga los enlatados (" in p:
                def _sin_lacteos(mm):
                    items = [x.strip() for x in mm.group(2).split(",") if x.strip()]
                    quedan = [x for x in items if not _LACTEO_443_RE.search(_sa(x))] or ["atún"]
                    return mm.group(1) + ", ".join(quedan) + mm.group(3)
                q = _ENLATADOS_443_RE.sub(_sin_lacteos, p)
                if q != p:
                    n += 1
                    p = q
            nuevos.append(p)
        if n:
            meal["recipe"] = nuevos
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


# ── [P1-PLAN-LOTE-449 · 2026-09-27] Lo que el cerrador ya sirve no se acompaña otra vez; los «Acompaña…» van en una frase ──
# Replay de la cola sobre 322 planes: (1) 119 comidas con «Cocina filete de pescado blanco a la plancha o hervido y sírvelo
# como proteína del plato.» Y, en el Montaje, «Acompaña con filete de pescado blanco.»: el mismo alimento servido dos veces
# (el 425 cubría «ablanden e incorpóralo», «escurre e incorpora», la licuadora y «sírvelo al lado», no esta frase). (2) 176
# montajes con dos o tres «Acompaña…» seguidos («Acompaña la cena con agua. Acompaña con edamame.»): una sola frase
# («Acompaña la cena con edamame y agua.»). tooltip-anchor: P1-PLAN-LOTE-449
_SIRVE_PROT_449_RE = re.compile(r"Cocina (?P<x>[^.;]+?) a la plancha o hervid[oa]s? y sírvel[oa]s? como proteína del plato\."
                                r"|Escurre e incorpora (?P<y>[^.;(]+?) \(ya viene[n]? cocid[oa]s?\) al guiso")
#: «sardinas en lata (ya viene cocido)» → «(ya vienen cocidas)»: el paréntesis concuerda con el alimento (89 frases)
_YA_VIENE_449_RE = re.compile(r"(?P<x>\b[a-záéíóúñ]+)(?P<resto>(?: (?:en|de) [a-záéíóúñ]+)?) \(ya viene cocido\)")
_ACOMPANA_449_RE = re.compile(r"(?P<cab>Acompaña(?: (?:la cena|el almuerzo|el desayuno|la merienda|el plato))?) con "
                              r"(?P<obj>[^.]+?)\.(?=\s|$)")


def proteina_servida_una_vez(meal) -> int:
    """Nº de «Acompaña con X.» quitados porque un paso ya sirve X como proteína del plato; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        servidas = {_sa(m.group("x") or m.group("y")).strip() for p in rec if isinstance(p, str) and not _es_nota(p)
                    for m in _SIRVE_PROT_449_RE.finditer(p)}
        if not servidas:
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _pilar(p) != "montaje":
                continue
            q = p
            for m in list(_ACOMPANA_449_RE.finditer(p)):
                if m.group("cab") == "Acompaña" and _sa(m.group("obj")).strip() in servidas:
                    q = q.replace(m.group(0), "", 1)
                    n += 1
            if q != p and re.sub(r"^Montaje:\s*", "", q).strip():      # un Montaje nunca se queda vacío
                rec[i] = re.sub(r"\s{2,}", " ", q).rstrip()
            elif q != p:
                n -= 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


def ya_viene_concordado(meal) -> int:
    """«sardinas en lata (ya viene cocido)» → «(ya vienen cocidas)». Nº de pasos corregidos; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        n = 0

        def _conc(m):
            pl, fem = _genero_numero(m.group("x"))
            if not pl and not fem:
                return m.group(0)
            return f"{m.group('x')}{m.group('resto')} (ya viene{'n' if pl else ''} cocid{'a' if fem else 'o'}{'s' if pl else ''})"

        for i, p in enumerate(rec):
            if isinstance(p, str) and "(ya viene cocido)" in p:
                q = _YA_VIENE_449_RE.sub(_conc, p)
                if q != p:
                    rec[i] = q
                    n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


def _y_449(ultimo: str) -> str:
    t = _sa(ultimo)
    return "e" if re.match(r"(?:i|hi)(?!e)", t) else "y"


def acompanamientos_en_una_frase(meal) -> int:
    """Nº de montajes cuyos «Acompaña…» seguidos pasaron a una frase; 0 ante cualquier error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or _pilar(p) != "montaje":
                continue
            ms = list(_ACOMPANA_449_RE.finditer(p))
            if len(ms) < 2:
                continue
            # sólo frases SEGUIDAS (entre una y otra, nada más que espacio)
            grupo = [ms[0]]
            for a, b in zip(ms, ms[1:]):
                if p[a.end():b.start()].strip():
                    break
                grupo.append(b)
            if len(grupo) < 2:
                continue
            cab = next((m.group("cab") for m in grupo if m.group("cab") != "Acompaña"), "Acompaña")
            agua = [m.group("obj") for m in grupo if re.search(r"\bagua\b", _sa(m.group("obj")))]
            resto = [m.group("obj") for m in grupo if m.group("obj") not in agua]
            objs = resto + agua                            # el agua, al final: «con edamame y agua»
            lista = objs[0] if len(objs) == 1 else ", ".join(objs[:-1]) + f" {_y_449(objs[-1])} " + objs[-1]
            rec[i] = p[:grupo[0].start()] + f"{cab} con {lista}." + p[grupo[-1].end():]
            n += 1
        if n:
            meal.pop("_display", None)
        return n
    except Exception:
        return 0
