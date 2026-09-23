# backend/dish_naming.py
"""[P1-CLOSER-TITLE-CASE · 2026-09-06] Cómo se ESCRIBE el nombre de un plato y de sus ingredientes.

Morfología del español y convención tipográfica: concordancia de género y número del participio,
y la caja de un alimento que se añade a un título. Nada de esto es orquestación — vivía dentro de
`graph_orchestrator.py` porque allí es donde se usaba, y el techo de líneas de ese fichero (roadmap
2.5 §11) pedía exactamente esto: extraer un módulo, no subir el tope.

Las tres funciones son puras y fail-safe: ante cualquier duda devuelven la entrada sin tocar. Se
reexportan desde `graph_orchestrator` para no romper los call sites existentes.
"""
from __future__ import annotations

# [P2-DISH-COHERENCE-NAMEFIX · 2026-06-25] Stopwords es-DO para el reflejo del nombre: no se
# capitalizan ('Carne de Res', no 'Carne De Res') ni cuentan como token significativo de la proteína.
_NAME_STOPWORDS = {"de", "del", "la", "el", "los", "las", "con", "y", "a", "en", "sin", "al"}

# [P1-NAME-GENDER-POLISH · 2026-07-26] Alimentos FEMENINOS frecuentes en los nombres de plato +
# adjetivos con forma femenina. Deliberadamente CORTO: solo se corrige lo que se puede afirmar.
_NAME_FEM_FOODS = {
    "lechosa", "manzana", "pera", "pina", "naranja", "auyama", "batata", "yuca",
    "avena", "guayaba", "toronja", "mandarina", "chinola", "ciruela", "uva",
    "sandia", "papaya", "coliflor", "zanahoria", "berenjena", "remolacha",
    "pechuga", "carne", "tilapia", "tortilla", "ensalada", "sopa", "crema",
    # [P1-CLOSER-LINE-SPANISH] terminan en -e y la morfología no las ve como femeninas.
    "leche", "clara", "claras", "legumbre", "legumbres",
}
_NAME_ADJ_FEM = {
    "fresco": "fresca", "asado": "asada", "salteado": "salteada", "molido": "molida",
    "tostado": "tostada", "horneado": "horneada", "guisado": "guisada",
    "cocido": "cocida", "rallado": "rallada", "picado": "picada", "crudo": "cruda",
}


def participio_concordado(nm, participio: str = "cocido") -> str:
    """[P1-CLOSER-LINE-SPANISH · 2026-09-06] Concuerda el participio con el NÚCLEO del nombre del
    alimento: «soya texturizada» → *cocida*, «lentejas» → *cocidas*, «garbanzos» → *cocidos*.

    Julio descartó la concordancia de género general «por falsos positivos: hace falta el núcleo del
    sintagma» (ver `_SHELLFISH_HINT`). Aquí el núcleo NO hay que adivinarlo: el cerrador conoce el
    nombre del alimento que acaba de elegir, y su primera palabra ES el núcleo. Por eso este caso sí
    se puede cerrar y aquel no.

    Reusa `_NAME_ADJ_FEM` (masculino → femenino) y `_NAME_FEM_FOODS`, que ya existían para el título.
    Fail-safe: ante cualquier duda devuelve el participio tal cual llega."""
    try:
        from constants import strip_accents as _sa_pc
        cabeza = _sa_pc(str(nm or "").strip().lower()).split()
        if not cabeza:
            return participio
        cabeza = cabeza[0].strip(",.;:")
        plural = cabeza.endswith("s") and len(cabeza) > 3
        # Femenino por morfología (-a / -as) o por la tabla, que cubre los que la morfología no ve
        # («carne», «leche»). El singular se prueba también sin la -s para que «lentejas» case.
        femenino = (cabeza.endswith("a") or cabeza.endswith("as")
                    or cabeza in _NAME_FEM_FOODS
                    or cabeza[:-1] in _NAME_FEM_FOODS
                    or cabeza[:-2] in _NAME_FEM_FOODS)
        base = _NAME_ADJ_FEM.get(participio, participio) if femenino else participio
        return base + ("s" if plural else "")
    except Exception:
        return participio


def _titulo_en_title_case(name: str) -> bool:
    """[P1-CLOSER-TITLE-CASE · 2026-09-06] ¿El título anfitrión capitaliza sus palabras significativas?

    Medido sobre los 84 planes vivos: el 80,4 % de los títulos que escribe el modelo van en frase
    normal («Batata dominicana con huevo pochado y melón fresco») y solo el 19,6 % en Title Case
    («Carne de Res y Arroz»). El cerrador añadía SIEMPRE en Title Case, así que la costura se leía a
    simple vista en 129 títulos de 57 planes: «Arepita de maíz rellena de ricotta, tomate y espinacas
    con Soya Texturizada».

    Ante la duda —un título demasiado corto para tener palabra significativa tras la primera— devuelve
    True, que es la conducta previa: el cambio actúa solo cuando hay evidencia de frase normal.
    """
    _ws = str(name or "").split()
    _sig = [w for w in _ws[1:]
            if len(w) >= 3 and w.lower() not in _NAME_STOPWORDS and w[:1].isalpha()]
    return all(w[:1].isupper() for w in _sig)


def _food_display_for_title(pname: str, host: str) -> str:
    """[P1-CLOSER-TITLE-CASE · 2026-09-06] El alimento añadido, en la caja del título que lo acoge.

    Cuando el anfitrión NO va en Title Case se respeta la caja que el catálogo escribió palabra por
    palabra y solo se minusculiza la PRIMERA. Los 5 nombres propios del catálogo —«Nuez de Castilla»,
    «Coles de Bruselas», «Galletas Graham», «Flor de Jamaica», «Harina de Negrito»— conservan así su
    mayúscula interna. `capitalize()` la destruía en las dos direcciones: «Nuez De Castilla» dentro de
    un título en Title Case y «Soya Texturizada» dentro de una frase en minúsculas.
    """
    _ws = str(pname or "").split()
    if not _ws:
        return ""
    if _titulo_en_title_case(host):
        return " ".join(w if w.lower() in _NAME_STOPWORDS else w.capitalize() for w in _ws)
    return " ".join([_ws[0][:1].lower() + _ws[0][1:]] + _ws[1:])


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-175 · 2026-09-23] La costura de los swaps en el NOMBRE y en la DESCRIPCIÓN.
#
# Batería real del generador (lactancia, RD): «Tostadas con huevo y Aguacate fresca», y la descripción seguía diciendo
# «…con huevo bien cocido y guayaba fresca». El autofix fruta-dulce + base salada (`_fruit_savory_autofix`) cambia la
# fruta por aguacate en el nombre, la lista y los pasos, pero escribía el sustituto con la mayúscula del catálogo en
# mitad de una frase, no concordaba el adjetivo y no tocaba la descripción. Los swaps de presupuesto y de fruta
# repetida dejan la misma costura en la caja: «Yogurt con Guineo y semillas», «ricotta, Espinacas al limón y edamame»,
# «pera, Linaza y queso cottage» (13 títulos en 8 perfiles de la batería). tooltip-anchor: P1-PLAN-LOTE-175-NOMBRE

# Masculinos frecuentes en los nombres (espejo de `_NAME_FEM_FOODS`), y el adjetivo femenino → su masculino.
_NAME_MASC_FOODS = {
    "aguacate", "mango", "guineo", "melon", "huevo", "pollo", "queso", "pescado", "arroz", "platano", "casabe",
    "pan", "tomate", "mani", "yogurt", "yogur", "pavo", "cerdo", "chivo", "atun", "salmon", "limon", "coco",
    "brocoli", "repollo", "pepino", "maiz", "tofu", "edamame", "nispero",
}
_ADJ_MASC = {fem: masc for masc, fem in _NAME_ADJ_FEM.items()}

# Alimentos comunes que un swap escribe con la caja del catálogo. Lista POSITIVA a propósito: los nombres propios
# del catálogo («Harina de Negrito», «Coles de Bruselas», «Flor de Jamaica») y cualquier palabra que no esté aquí
# conservan su mayúscula.
_ALIMENTO_COMUN = _NAME_FEM_FOODS | _NAME_MASC_FOODS | {
    "fresa", "fresas", "uvas", "linaza", "espinaca", "espinacas", "almendras", "nueces", "ricotta", "cottage", "kale",
    "zanahorias", "habichuelas", "lentejas", "garbanzos", "papa", "papas", "huevos", "sardinas", "camarones",
    "ajonjoli", "chia", "vegetales", "berro", "berros", "acelga", "acelgas", "puerro", "vainitas", "tayota", "yautia",
    "mapuey", "platanos", "guineos", "mangos", "aguacates", "semillas", "quinoa", "cebada", "molondrones",
}
_PROSA_ADJ = ("fresc", "madur", "jugos", "cremos", "picad", "tostad", "asad", "hornead", "cortad", "rallad", "cocid")


def _sa(s: str) -> str:
    from constants import strip_accents
    return strip_accents(str(s or "")).lower()


def _es_femenino(nombre: str) -> bool:
    cab = _sa(nombre).split()
    cab = cab[0] if cab else ""
    return cab in _NAME_FEM_FOODS or (cab.endswith("a") and cab not in _NAME_MASC_FOODS)


def pulir_nombre(name):
    """Concordancia de los alimentos MASCULINOS («Aguacate fresca» → «Aguacate fresco») y caja de los alimentos
    comunes dentro de un título en frase normal («…tomate con Aguacate» → «…tomate con aguacate»). Devuelve el nombre
    nuevo o `None` si no hay nada que tocar.

    Mismo criterio de NÚCLEO que la regla femenina (`_fix_name_gender_agreement`): al principio, o tras «y», «con» o
    «e» — no tras «de»: en «Ensalada de aguacate fresca» el núcleo puede ser la ensalada. Y la caja sólo se toca cuando
    la MAYORÍA de las palabras significativas del título va en minúscula: en un título en Title Case («Pollo Asado con
    Vegetales») la mayúscula es el estilo, no la costura."""
    try:
        if not isinstance(name, str) or not name.strip():
            return None
        toks = name.split()
        if len(toks) < 2:
            return None
        out = list(toks)
        for i in range(len(out) - 1):
            adj_raw = out[i + 1]
            adj = _sa(adj_raw).strip(",.;:")
            if _sa(out[i]).strip(",.;:") not in _NAME_MASC_FOODS or adj not in _ADJ_MASC:
                continue
            if i > 0 and _sa(out[i - 1]).strip(",.;:") not in ("y", "con", "e"):
                continue
            nuevo = _ADJ_MASC[adj]
            nuevo = nuevo.capitalize() if adj_raw[:1].isupper() else nuevo
            if adj_raw.endswith((",", ".", ";", ":")):
                nuevo += adj_raw[-1]
            out[i + 1] = nuevo
        sig = [w for w in out[1:] if len(w) >= 3 and w.lower() not in _NAME_STOPWORDS and w[:1].isalpha()]
        if sig and sum(1 for w in sig if w[:1].islower()) * 2 > len(sig):
            for i in range(1, len(out)):
                core = out[i].strip(",.;:")
                if (len(core) >= 3 and core[:1].isupper() and core[1:].islower()
                        and _sa(core) in _ALIMENTO_COMUN):
                    out[i] = out[i].replace(core, core[:1].lower() + core[1:], 1)
        res = " ".join(out)
        return res if res != name else None
    except Exception:
        return None


def fix_name_gender_agreement(name):
    """[P1-NAME-GENDER-POLISH · 2026-07-26] Concuerda el adjetivo cuando el NÚCLEO del sintagma
    es un alimento femenino. Devuelve el nombre corregido, o `None` si no hay nada que tocar.

    Medido en 60 planes (196 nombres): 2 casos reales —«Maní y **Lechosa Fresco**…» y «**Lechosa
    Fresco** con Almendras…»— sobre lechosa, que es femenina.

    ⚠️ La regla exige que el sustantivo femenino sea el NÚCLEO, es decir que vaya al principio o
    justo tras `y`/`con`/`de`/`e`. Sin eso, «Queso **Crema Batido**» se "corregiría" a «Crema
    Batida» — y ahí el núcleo es *queso* (masculino), así que "batido" ya concuerda bien. Mi
    primer detector cometió exactamente ese error: 2 de sus 4 hallazgos de género eran falsos.
    Por la misma razón NO se toca la redundancia de palabras («…pescado **blanco**… Arroz
    **Blanco**» es correcto: son dos alimentos distintos). Con 5 defectos cosméticos en 196
    nombres, un reescritor amplio corrompe más de lo que arregla.

    [P1-PLAN-LOTE-175 · 2026-09-23] Vivía en `graph_orchestrator` (que se reexporta con el mismo nombre); aquí, junto a
    la regla MASCULINA y la caja de `pulir_nombre`, que corren al final sobre su resultado.
    """
    try:
        if not isinstance(name, str) or not name.strip():
            return None
        from constants import strip_accents as _sa_ng
        _toks = name.split()
        if len(_toks) < 2:
            return None
        _cambios = 0
        for _i in range(len(_toks) - 1):
            _sust = _sa_ng(_toks[_i].lower()).strip(",.;:")
            _adj = _sa_ng(_toks[_i + 1].lower()).strip(",.;:")
            if _sust not in _NAME_FEM_FOODS or _adj not in _NAME_ADJ_FEM:
                continue
            # el sustantivo debe ser NÚCLEO: inicio del nombre o tras y/con/de/e
            if _i > 0:
                _prev = _sa_ng(_toks[_i - 1].lower()).strip(",.;:")
                if _prev not in ("y", "con", "de", "e"):
                    continue
            _fem = _NAME_ADJ_FEM[_adj]
            _orig = _toks[_i + 1]
            _nuevo = _fem.capitalize() if _orig[:1].isupper() else _fem
            if _orig.endswith((",", ".", ";", ":")):
                _nuevo += _orig[-1]
            _toks[_i + 1] = _nuevo
            _cambios += 1
        _base = " ".join(_toks)
        return pulir_nombre(_base) or (_base if _cambios else None)
    except Exception:
        return None


def _articulo(pre: str, fem: bool, plural: bool) -> str:
    """El artículo o la contracción que va pegado al alimento, en el género del sustituto."""
    import re
    pares = ((r"\bde\s+la\s+$", "del "), (r"\ba\s+la\s+$", "al "), (r"\bla\s+$", "el "), (r"\buna\s+$", "un "),
             (r"\blas\s+$", "los "), (r"\bunas\s+$", "unos "))
    if fem:
        pares = ((r"\bdel\s+$", "de la "), (r"\bal\s+$", "a la "), (r"\bel\s+$", "la "), (r"\bun\s+$", "una "),
                 (r"\blos\s+$", "las "), (r"\bunos\s+$", "unas "))
    for rx, rep in pares:
        m = re.search(rx, pre, re.IGNORECASE)
        if m:
            return pre[:m.start()] + (rep[:1].upper() + rep[1:] if pre[m.start():m.start() + 1].isupper() else rep)
    return pre


def _en_prosa(texto: str, pat, nuevo: str) -> str:
    """Cambia el alimento de una DESCRIPCIÓN con su artículo y el adjetivo que lo sigue en el género del sustituto."""
    import re
    fem = _es_femenino(nuevo)
    trozos, pos = [], 0
    for m in pat.finditer(texto):
        plural = m.group(0).lower().endswith("s")
        lbl = nuevo + ("s" if plural and not nuevo.endswith("s") else "")
        antes = texto[:m.start()].rstrip()
        if m.start() == 0 or antes.endswith((".", "!", "?")):
            lbl = lbl[:1].upper() + lbl[1:]
        trozos.append(_articulo(texto[pos:m.start()], fem, plural) + lbl)
        pos = m.end()
        mm = re.match(r"(\s+)([A-Za-zÁÉÍÓÚÑáéíóúñ]+)", texto[pos:])
        if mm and _sa(mm.group(2)).startswith(_PROSA_ADJ):
            adj = re.sub(r"([oa])(s?)$", lambda x: ("a" if fem else "o") + x.group(2), mm.group(2))
            trozos.append(mm.group(1) + adj)
            pos += mm.end()
    trozos.append(texto[pos:])
    return "".join(trozos)


def sustituir_alimento(meal: dict, pat, repl: str) -> str:
    """El alimento de un swap, en el NOMBRE con la caja del título (mayúscula sólo si abre el nombre) y en la
    DESCRIPCIÓN con artículo y adjetivo concordados. Devuelve el nombre nuevo; la descripción se escribe en `meal`.
    `pat` es el patrón del alimento viejo, con frontera de palabra. Fail-safe: la descripción queda como estaba."""
    nombre = str(meal.get("name") or "")
    bajo = repl[:1].lower() + repl[1:]

    def _en_nombre(m):
        lbl = repl if m.start() == 0 else bajo
        return lbl + ("s" if m.group(0).lower().endswith("s") and not lbl.endswith("s") else "")
    nuevo = pat.sub(_en_nombre, nombre)
    try:
        desc = meal.get("desc")
        if isinstance(desc, str) and desc.strip():
            meal["desc"] = _en_prosa(desc, pat, bajo)
    except Exception:
        pass
    return pulir_nombre(nuevo) or nuevo


__all__ = ["_NAME_STOPWORDS", "_NAME_FEM_FOODS", "_NAME_ADJ_FEM",
           "participio_concordado", "_titulo_en_title_case", "_food_display_for_title",
           "pulir_nombre", "sustituir_alimento", "fix_name_gender_agreement"]
