# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-612 · 2026-09-27] Tras una sustitución, la concordancia sigue al alimento NUEVO.

`_rewrite_recipe_steps_after_subs` (13 llamadas: presupuesto, clínica, fruta en plato salado, embarazo, DM2…) cambia el
sustantivo y deja el género y el número del alimento viejo. Corpus de baterías (421 planes, 57 comidas): «pesa ½ taza de
arroz integral (80 g) y enjuágala» (era quinoa), «hasta que esté tierna; escúrrela muy bien y tuéstala», «saltea
espinacas 2-3 min hasta que se ablande» y «sirve junto al espinacas salteado» (era kale), «hasta que vainitas estén
tiernos» (espárragos), «Añade filete de pescado blanco al guiso y cocínalos… incorpóralos» (camarones), «tuesta maní
picado… hasta que doren y suelten aroma; retíralas» (almendras).

Aquí, en el paso ya reescrito, desde cada mención del alimento nuevo hasta que el texto pasa a otro sustantivo (un
determinante que no sea de un utensilio, u otro alimento de la comida): el adjetivo pegado al alimento, el artículo
contraído de delante («al espinacas»), el pronombre del verbo («escúrrela»), el predicativo («esté tierna») y el número
del verbo («se ablande», «que doren») pasan al género y número del alimento nuevo, SOLO si concordaban con el viejo. El
pronombre y el predicativo no se tocan si en la misma frase, antes del alimento, hay otro sustantivo del género viejo
que pueda ser su antecedente («reparte el pollo entre la tortilla, agrega maní y ciérralas»), ni con los verbos que
actúan sobre el plato armado (cerrar, enrollar, doblar, rellenar, armar, envolver). Sin el género de uno de los dos,
nada. Con maní, el corte del fruto seco sale también de los pasos (lote 287: «maní laminado» → «maní picado»). La lista
de ingredientes no se toca (identificador), salvo la línea que el pase de presupuesto crea («½ taza de Arroz blanco
seca»): el lector de cantidades no distingue género (`sec[oa]s?`). tooltip-anchor: P1-PLAN-LOTE-612
"""
from __future__ import annotations

import re
import unicodedata

#: género de la cabeza del nombre (singular, sin acentos). Número: por la forma escrita.
_GENERO = {
    "f": ("quinoa quinua almendra nuez chia linaza granada fresa frambuesa mora haba ricotta avena batata lechosa pina "
          "espinaca vainita habichuela carne leche pechuga papa yuca auyama yautia berenjena zanahoria cebolla lenteja "
          "sardina tilapia merluza res manzana pera naranja toronja sandia uva ciruela calabaza lechuga col coliflor arepa "
          "tortilla galleta harina pasta salsa miel mantequilla stevia sal cebada cereza guayaba chinola parcha "
          "mandarina acelga remolacha alcachofa aceituna costilla chuleta longaniza salchicha mortadela clara yema "
          "semilla hojuela almeja langosta"),
    "m": ("arroz mani guineo pollo pavo cerdo pescado filete salmon camaron kale brocoli aguacate tomate queso yogurt "
          "yogur casabe pan platano mango melon huevo edamame tofu atun bacalao mero chivo esparrago arandano pistacho "
          "garbanzo frijol gandul champinon ajonjoli sesamo bulgur cuscus maiz coco limon pepino repollo cangrejo pulpo "
          "calamar jamon chorizo salami higado requeson mascarpone azucar muslo lomo cordero conejo bistec churrasco "
          "solomillo tempeh seitan hongo pimiento aji ajo puerro apio berro cilantro oregano jengibre name mapuey "
          "rabano nabo calabacin molondron durazno kiwi higo datil anacardo maranon cajuil"),
}
_LEX = {w: g for g, ws in _GENERO.items() for w in ws.split()}
_INVARIANTE_N = {"res", "cuscus", "anis", "maiz", "arroz", "tempeh"}

#: raíces de adjetivo/participio que concuerdan con el alimento («tiern-a», «cocid-os»)
_ADJ = (r"fresc|median|madur|pequeñ|cortad|picad|rallad|laminad|pelad|cocid|asad|tostad|enter|triturad|machacad|majad|"
        r"hervid|trocead|rebanad|dorad|tibi|fri|frí|crud|sec|tiern|suelt|escurrid|saltead|reservad|opac|cremos|esponjos|"
        r"jugos|list|bland|cocinad|hornead|guisad|frit|desmenuzad|molid|derretid|remojad|enjuagad|limpi|lavad|"
        r"sazonad|marinad|sellad|glasead|caramelizad|blanquead|cubiert|tapad|abiert|cerrad")
#: adjetivos sin género: sólo cambian de número («tiernas pero verdes»)
_INV = r"verde|suave|caliente|crujiente|firme|brillante|fragante|dulce"
_ADJ_RX = re.compile(r"\b(?P<r>" + _ADJ + r")(?P<g>[aoAO])(?P<n>[sS]?)\b")
_INV_RX = re.compile(r"\b(?P<r>" + _INV + r")(?P<n>s?)\b", re.IGNORECASE)
_CADENA = re.compile(r"(?:\s+(?:" + _ADJ + r")[aoAO][sS]?\b)+")
_PRED = re.compile(r"\b(?P<v>est[eé]n?|est[aá]n?|queden?|quedan?|se\s+vean?|luzcan?|resulten?)\s+(?P<m>(?:bien|muy|"
                   r"ligeramente|apenas|casi|completamente|totalmente)\s+)?(?P<adj>(?:" + _ADJ + r")[aoAO][sS]?)"
                   r"(?P<mas>(?:\s*(?:,|y|pero)\s*(?:(?:" + _ADJ + r")[aoAO][sS]?|(?:" + _INV + r")s?)\b)*)",
                   re.IGNORECASE)
#: pronombre pegado a un imperativo («tuéstala»), gerundio («removiéndola») o infinitivo («dorarla»); «película» y
#: «espátula» no (la raíz acaba en «u»)
_CLITICO = re.compile(r"\b(?P<v>[a-záéíóúñü]*[áéíóú][a-záéíóúñü]*?(?:[aeáé]|nd[oó])|[a-zñü]+?(?:ar|er|ir))"
                      r"(?P<c>la|las|lo|los)\b", re.IGNORECASE)
#: verbos que actúan sobre el plato armado, no sobre un ingrediente: «ciérralas» son las tortillas
_DEL_PLATO = ("cierr", "enroll", "dobl", "rellen", "arm", "envuelv", "emplat")
_VERBO_N = re.compile(r"\b(?P<pre>(?:que|y|e)\s+(?:no\s+)?(?:se\s+)?|se\s+)(?P<v>ablande|dore|cocine|suavice|marchite|"
                      r"reduzca|abra|suelte|cueza|reviente|hinche|infle|caramelice|derrita|cuaje|queme|pegue|rompa|"
                      r"deshaga|seque|enfr[ií]e|absorba)(?P<n>n?)\b", re.IGNORECASE)
_DET = re.compile(r"\b(?P<d>el|la|los|las|un|una|unos|unas|del|al|su|sus)\s+(?P<s>[a-záéíóúñü]+)", re.IGNORECASE)
_DET_G = {"el": "m", "los": "m", "un": "m", "unos": "m", "del": "m", "al": "m", "la": "f", "las": "f", "una": "f",
          "unas": "f"}
_UTENSILIO = {"sarten", "olla", "agua", "fuego", "horno", "plato", "bowl", "tazon", "caldero", "guiso", "caldo", "sofrito",
              "salsa", "mezcla", "fondo", "microondas", "plancha", "parrilla", "freidora", "licuadora", "bandeja", "vapor",
              "recipiente", "envase", "paquete", "centro", "mitad", "resto", "punto", "final", "borde", "lado", "tabla",
              "colador", "cuchillo", "tenedor", "cuchara", "papel", "nevera", "refrigerador", "aceite", "jugo", "zumo",
              "liquido", "grano", "interior", "exterior", "superficie", "base", "vaso", "taza", "hielo", "tapa", "misma",
              "mismo", "toque", "espatula", "rejilla", "cazuela", "wok"}
_MIXTAS = re.compile(r"\s+mixt[oa]s?\b", re.IGNORECASE)


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def genero_numero(nombre):
    """(«f»|«m», plural) de la cabeza del nombre; None si no se sabe."""
    t = re.sub(r"^[\s\d½¼¾⅓⅔⅛.,/-]+(?:(?:g|gr|kg|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?|unidad(?:es)?|"
               r"porci[oó]n(?:es)?|piezas?|rebanadas?|lonjas?|filetes?\b(?=\s+de))\s+)?(?:de\s+)?", "", _sa(nombre))
    cab = (re.findall(r"[a-zñ]+", t) or [""])[0]
    if not cab:
        return None
    if cab == "nueces":
        return "f", True
    if cab in _LEX:
        return _LEX[cab], False
    if cab not in _INVARIANTE_N:
        if cab.endswith("es") and cab[:-2] in _LEX:
            return _LEX[cab[:-2]], True
        if cab.endswith("s") and cab[:-1] in _LEX:
            return _LEX[cab[:-1]], True
    return None


def _sufijo(g, pl, mayus=False) -> str:
    s = ("o" if g == "m" else "a") + ("s" if pl else "")
    return s.upper() if mayus else s


def _clitico(g, pl) -> str:
    return ("lo" if g == "m" else "la") + ("s" if pl else "")


def _adjetivos(txt, viejo, nuevo) -> str:
    def _uno(m):
        if (m.group("g").lower() == ("o" if viejo[0] == "m" else "a")) and (bool(m.group("n")) == viejo[1]):
            return m.group("r") + _sufijo(nuevo[0], nuevo[1], m.group("g").isupper())
        return m.group(0)

    def _inv(m):
        if viejo[1] != nuevo[1] and bool(m.group("n")) == viejo[1]:
            return m.group("r") + ("s" if nuevo[1] else "")
        return m.group(0)
    return _INV_RX.sub(_inv, _ADJ_RX.sub(_uno, txt))


def _predicado(p, viejo, nuevo) -> str:
    adj = _adjetivos(p.group("adj"), viejo, nuevo)
    mas = _adjetivos(p.group("mas") or "", viejo, nuevo)
    if adj == p.group("adj") and mas == (p.group("mas") or ""):
        return p.group(0)
    v = p.group("v")
    if viejo[1] != nuevo[1]:
        v = (v[:-1] if v.endswith("n") else v) + ("n" if nuevo[1] else "")
    return v + " " + (p.group("m") or "") + adj + mas


def _alcance(texto, desde, otros) -> int:
    """Fin del tramo que habla del alimento: otro sustantivo con determinante (no un utensilio) u otro alimento."""
    fin = len(texto)
    for d in _DET.finditer(texto, desde):
        if _sa(d.group("s")) not in _UTENSILIO:
            fin = d.start()
            break
    if otros:
        rx = re.compile(r"\b(?:" + "|".join(re.escape(o) for o in otros) + r")", re.IGNORECASE)
        o = rx.search(_sa(texto), desde)
        if o:
            fin = min(fin, o.start())
    return fin


def _ambiguo(texto, a, g_viejo) -> bool:
    """¿Hay, en la frase del alimento y antes de él, otro sustantivo del género viejo que pueda ser el antecedente?"""
    ini = max(texto.rfind(ch, 0, a) for ch in ".;:") + 1
    for d in _DET.finditer(texto[ini:a]):
        if _DET_G.get(d.group("d").lower()) == g_viejo and _sa(d.group("s")) not in _UTENSILIO:
            return True
    return False


def concordar(texto, nuevo_nombre, viejo_nombre, otros=()) -> str:
    """El paso `texto` (ya con `nuevo_nombre`) con la concordancia del alimento nuevo."""
    try:
        viejo, nuevo = genero_numero(viejo_nombre), genero_numero(nuevo_nombre)
        if not viejo or not nuevo or viejo == nuevo or not texto:
            return texto
        nd = re.sub(r"^[\s\d½¼¾⅓⅔⅛.,/-]+", "", str(nuevo_nombre or "")).strip()
        rx = re.compile(r"\b" + re.escape(nd).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
        otros = [o for o in (_sa(x) for x in otros) if o and o not in _sa(nd) and _sa(nd) not in o]
        s, pos = str(texto), 0
        while True:
            m = rx.search(s, pos)
            if not m:
                break
            a, b = m.start(), m.end()
            # el artículo contraído de delante: «junto al espinacas» → «junto a las espinacas»
            pre = re.search(r"\b(al|del)\s+$", s[:a], re.IGNORECASE)
            if pre and nuevo != ("m", False):
                art = {("f", False): "la", ("f", True): "las", ("m", True): "los"}[nuevo]
                rep = ("a " if pre.group(1).lower() == "al" else "de ") + art + " "
                s = s[:pre.start()] + rep + s[pre.end():]
                delta = len(rep) - (pre.end() - pre.start())
                a, b = a + delta, b + delta
            mx = _MIXTAS.match(s, b)                       # «maní mixtas» → «maní»: la mezcla era del premium
            if mx:
                s = s[:b] + s[mx.end():]
            c = _CADENA.match(s, b)
            if c:
                nuevo_c = _adjetivos(c.group(0), viejo, nuevo)
                s = s[:b] + nuevo_c + s[c.end():]
                b += len(nuevo_c)
            fin = _alcance(s, b, otros)
            tramo = s[b:fin]
            if not _ambiguo(s, a, viejo[0]):
                tramo = _PRED.sub(lambda p: _predicado(p, viejo, nuevo), tramo)
                tramo = _CLITICO.sub(
                    lambda k: k.group("v") + (_clitico(*nuevo) if (k.group("c").lower() == _clitico(*viejo)
                                                                   and not _sa(k.group("v")).startswith(_DEL_PLATO))
                                              else k.group("c")), tramo)
                if viejo[1] != nuevo[1]:
                    tramo = _VERBO_N.sub(lambda v: v.group("pre") + v.group("v") + ("n" if nuevo[1] else ""), tramo)
            s = s[:b] + tramo + s[fin:]
            pos = max(b + len(tramo), a + 1)
        if _sa(nd).startswith("mani"):
            try:
                import presupuesto_texto as pt
                s = pt._CORTE_MANI_RX.sub(lambda mm: f"{mm.group(1)} picado", s)
            except Exception:                                                  # noqa: BLE001
                pass
        return s
    except Exception:                                                          # noqa: BLE001
        return texto


def tras_reescritura(texto, token_subs, meal=None) -> str:
    """El paso que `_rewrite_recipe_steps_after_subs` acaba de reescribir, con la concordancia de cada alimento nuevo."""
    try:
        import graph_orchestrator as go
        otros = nombres_de_la_comida(meal)
        for tokens, nuevo in token_subs or []:
            viejo = next((str(t) for t in (tokens or []) if t), "")
            if viejo:
                texto = concordar(texto, go._strip_qty_prefix_for_step(nuevo), viejo, otros)
        return texto
    except Exception:                                                          # noqa: BLE001
        return texto


def nombres_de_la_comida(meal) -> list:
    """Cabezas de los otros alimentos de la comida (para cerrar el tramo)."""
    out = []
    try:
        for x in (meal.get("ingredients") or []) if isinstance(meal, dict) else []:
            t = re.sub(r"^[\s\d½¼¾⅓⅔⅛.,/()≈-]+(?:[a-zñ]+\s+de\s+)?", "", _sa(x))
            cab = (re.findall(r"[a-zñ]{4,}", t) or [""])[0]
            if cab and cab not in _UTENSILIO:
                out.append(cab)
    except Exception:                                                          # noqa: BLE001
        pass
    return out


__all__ = ["concordar", "genero_numero", "nombres_de_la_comida", "tras_reescritura"]
