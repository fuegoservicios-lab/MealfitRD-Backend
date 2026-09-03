"""[P2-CHICHARO-CHICHARRON · 2026-08-21] «Chícharo» resolvía a «Chicharrón»: cerdo para quien pidió
guisantes.

`normalize_name('chicharo')` → **'Chicharrón'**. El fuzzy de difflib da 0,889 y el umbral es 0,87,
así que pasa; y como resuelve a una fila real del catálogo, **sobrevive al filtro de
verified-only**: no se cae de la lista, se compra. Un mexicano que pide chícharos recibe corteza de
cerdo, y si además es vegetariano, musulmán o judío, el plato es inaceptable por razones que no son
de nutrición.

Es la 18ª colisión de subcadena/fuzzy documentada en este proyecto (sal⊂salsa, pollo⊂repollo,
res⊂fresco, piñones⊂champiñones…).

MEDIDO ANTES DE ELEGIR LA FORMA DEL ARREGLO, porque «guard puntual» y «regla general» no cuestan lo
mismo: se barrieron 57 términos regionales de ES/MX/CO/PR y **43 resolvieron**. De esos, 9 cayeron
en Proteínas y **8 eran correctos** (gamba→Gambas, atún→Atún en agua, res→Carne de res…). El único
falso positivo era éste. O sea: una anécdota, no una clase — un guard general sobre «cruce de
categoría» habría sido una cuarta tabla a mano (la lección de P1-DIET-CANON-SSOT) para atrapar un
caso.

PERO LA ANÉCDOTA SE DEFIENDE COMO CLASE. El guard cubre el caso conocido; el barrido de este mismo
fichero es la defensa de verdad: si un alta futura del catálogo crea otra colisión de esta forma,
falla aquí antes de llegar a un plato.

LO QUE ESTO NO ARREGLA, dicho explícitamente: el catálogo **no tiene fila de guisante fresco**. La
única de esa familia es `Guisantes secos` (341 kcal, otro alimento). Así que tras el guard,
«chícharo» no resuelve a nada — que es peor de lo ideal y mejor que cerdo. Dar de alta la fila
correcta es curación de datos con procedencia verificable, y inventar aquí un `fdc_id` de memoria
es exactamente lo que costó la auditoría de procedencia del catálogo.

Cubre:
  A. El caso concreto.
  B. La forma del cruce, como barrido sobre el catálogo vivo.
  C. Lo que el guard NO puede romper: los fuzzy legítimos.
  D. Byte-identidad de «chicharrón» de verdad.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    _sc.get_master_ingredients()
    return _sc


@pytest.fixture(scope="module")
def catalogo(sc):
    filas = sc.get_master_ingredients() or []
    if not filas:
        pytest.skip("catálogo no disponible (sin DB)")
    return filas


# ── A. El caso concreto ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("termino", ["chicharo", "chícharo", "chicharos", "chícharos",
                                     "Chícharos", "chicharo verde", "chícharos congelados"])
def test_chicharo_jamas_resuelve_a_cerdo(sc, catalogo, termino):
    """El daño no es sólo nutricional: a un vegetariano, a un musulmán o a un judío el plato le
    resulta inaceptable por razones que la nutrición no cubre."""
    r = str(sc.normalize_name(termino) or "")
    assert "chicharr" not in r.lower(), f"{termino!r} → {r!r}: sigue resolviendo a cerdo"


# ── B. La forma del cruce, como barrido ─────────────────────────────────────────────────────────

# Términos regionales que un plan de ES/MX/CO/PR puede emitir. El barrido no busca ESTE bug: busca
# su FORMA — un término sin palabra de carne que acaba en una fila de Proteínas.
_TERMINOS_REGIONALES = [
    "chicharo", "chicharos", "guisante", "guisantes", "arveja", "arvejas", "ejote", "ejotes",
    "judias verdes", "alubias", "garbanzo", "calabacin", "pimiento", "aguacate", "palta",
    "platano", "melocoton", "albaricoque", "damasco", "frutilla", "zumo", "nata", "elote",
    "choclo", "maiz", "pimenton", "paprika", "cilantro", "perejil", "papa", "patata", "boniato",
    "camote", "batata", "yuca", "mandioca", "calabaza", "zapallo", "auyama", "champinon",
    "seta", "hongo", "yogur", "requeson", "cuajada", "tahini", "lenteja", "lentejas",
]

# Palabras que SÍ autorizan a caer en Proteínas: si el término las lleva, el destino es correcto.
_PALABRAS_DE_CARNE = (
    "carne", "res", "cerdo", "puerco", "chancho", "ternera", "pollo", "pavo", "chivo", "cordero",
    "pescado", "atun", "salmon", "merluza", "bacalao", "gamba", "camaron", "langostino",
    "marisco", "almeja", "jamon", "chorizo", "salami", "longaniza", "tocineta", "bacon",
    "salchicha", "chicharron", "huevo", "trucha", "anchoa", "boqueron", "pernil", "cecina",
)


def test_ningun_termino_vegetal_cae_en_proteinas(sc, catalogo):
    """La defensa de clase. El guard de arriba cubre el caso conocido; esto atrapa el SIGUIENTE:
    si un alta del catálogo crea otra colisión de esta forma, falla aquí y no en un plato."""
    cat = {r["name"]: (r.get("category") or "") for r in catalogo}
    culpables = []
    for t in _TERMINOS_REGIONALES:
        if any(p in t for p in _PALABRAS_DE_CARNE):
            continue
        r = str(sc.normalize_name(t) or "")
        if cat.get(r) == "Proteínas":
            culpables.append((t, r))
    assert not culpables, (
        f"términos sin palabra de carne que resuelven a Proteínas: {culpables}"
    )


# ── C. Lo que el guard NO puede romper ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("termino,esperado_contiene", [
    ("gamba", "gamba"),
    ("camaron", "camaron"),
    ("platanno", "platano"),      # el typo que el fuzzy existe para atrapar
    ("yogur griego", "yogurt"),
])
def test_los_fuzzy_legitimos_siguen_funcionando(sc, catalogo, termino, esperado_contiene):
    """El error opuesto sería subir el umbral global o desactivar el fuzzy: 8 de los 9 destinos a
    Proteínas del barrido eran CORRECTOS, y el fuzzy existe para los typos («platanno»→«plátano»).
    Un arreglo que los rompa cambia un bug por otro más caro."""
    from constants import strip_accents
    r = strip_accents(str(sc.normalize_name(termino) or "").lower())
    assert esperado_contiene in r, f"{termino!r} → {r!r}: se rompió un fuzzy legítimo"


# ── D. Byte-identidad ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("termino", ["chicharron", "chicharrón", "chicharrones",
                                     "chicharron de cerdo"])
def test_el_chicharron_de_verdad_sigue_resolviendo(sc, catalogo, termino):
    """El guard rechaza que un GUISANTE llegue al cerdo, no que el cerdo llegue al cerdo. Romper
    esto le quitaría el chicharrón a la cocina dominicana, que es donde el alimento vive."""
    r = str(sc.normalize_name(termino) or "")
    assert "hicharr" in r.lower(), f"{termino!r} → {r!r}: se perdió el chicharrón real"
