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


__all__ = ["_NAME_STOPWORDS", "_NAME_FEM_FOODS", "_NAME_ADJ_FEM",
           "participio_concordado", "_titulo_en_title_case", "_food_display_for_title"]
