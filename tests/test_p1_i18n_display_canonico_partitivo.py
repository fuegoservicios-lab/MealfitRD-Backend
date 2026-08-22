"""[P1-I18N-DISPLAY-CANONICO-PARTITIVO · 2026-08-22] El extractor metia el PARTITIVO y la
fraccion vulgar dentro del «nombre canonico», asi que rechazaba glosses CORRECTOS y esas
lineas se quedaban en espanol.

    '2 dientes de ajo'          -> canonico 'dientes de ajo'   (deberia ser 'ajo')
    '½ pedazo de ñame (≈150 g)' -> canonico 'pedazo de ñame'   (deberia ser 'ñame')
    '⅓ taza de yogurt griego'   -> la fraccion no se consumia siquiera

MEDIDO con una corrida DIRIGIDA contra el modelo real (deepseek-v4-flash) sobre dias reales
de tres planes de produccion: fr-FR tenia 8 de 186 lineas de ingrediente caidas al espanol
(4,3 %), y SEIS de esas ocho son de estas dos clases. Tres de las cuatro caidas del primer
dia traian un gloss PERFECTAMENTE correcto -- «2 gousses d'ail hachees (Ajo picado)» para
'2 dientes de ajo picados'-- y el check las tiraba.

POR QUE DUELE MAS DE LO QUE EL 4,3 % SUGIERE: el ajo aparece en casi todos los platos
salados, asi que no son ocho lineas al azar, son SIEMPRE LAS MISMAS. El usuario frances lee
su receta en frances con una de cada ~20 lineas en espanol, y siempre la del ajo. Es
exactamente el «mitad frances mitad espanol en la misma pantalla» que el fallback per-linea
existe para EVITAR, producido por un defecto del extractor en vez de por una mala traduccion.

EL ARREGLO NO INVENTA UNA CATEGORIA. Los partitivos son lo mismo que 'lonjas', 'rebanadas',
'rodajas' y 'lascas', que YA estaban en la lista: una porcion contable de un alimento, que no
es el alimento. Se anaden sus hermanos ('dientes', 'pedazos', 'ramitas', 'hojas', 'tallos',
'filetes', 'puñados', 'gajos', 'cabezas', 'latas', 'paquetes', 'sobres', 'potes', 'fundas',
'manojos') y la clase completa de fracciones vulgares mas los guiones de rango.

LA MITAD DE ESTE FICHERO SON NO-REGRESIONES, y no es relleno: ensanchar un prefijo que se
consume es exactamente como se rompe un extractor. Si 'Pechuga de pollo' o 'semillas de
linaza' perdieran su «de» interno, o si 'judias pintas' perdiera 'pintas', el canonico
dejaria de casar el catalogo y el efecto seria el MISMO defecto, al reves.

tooltip-anchor: P1-I18N-DISPLAY-CANONICO-PARTITIVO
"""
from __future__ import annotations

import pytest

from plan_display_i18n import _extract_canonical_name

_MARKER = "P1-I18N-DISPLAY-CANONICO-PARTITIVO"

# (linea de ingrediente, canonico esperado)
_PARTITIVOS = [
    ("2 dientes de ajo", "ajo"),
    ("1-2 dientes de ajo picados", "ajo"),
    ("½ pedazo de ñame (≈150 g)", "ñame"),
    ("⅓ taza de yogurt griego", "yogurt griego"),
    ("1 rama de canela", "canela"),
    ("2 ramitas de cilantro", "cilantro"),
    ("1 hoja de laurel", "laurel"),
    ("2 tallos de apio", "apio"),
    ("1 puñado de espinaca", "espinaca"),
    ("1 punado de espinaca", "espinaca"),  # sin la tilde de la ñ
    ("1 lata de atún", "atún"),
    ("1 cabeza de ajo", "ajo"),
    ("1 filete de pechuga de pollo", "pechuga de pollo"),
    ("1 gajo de mandarina", "mandarina"),
    ("1 sobre de levadura", "levadura"),
]

# Lo que YA funcionaba y no puede moverse.
_NO_REGRESION = [
    ("180 g de Pechuga de pollo", "Pechuga de pollo"),
    ("100 g de Arroz blanco", "Arroz blanco"),
    ("30 g Habichuelas rojas", "Habichuelas rojas"),
    ("2 lonjas de queso", "queso"),
    ("1 unidad Cebolla", "Cebolla"),
    ("2 cucharadas de aceite de oliva", "aceite de oliva"),
    # Nombres compuestos: el «de» INTERNO se conserva porque solo se quita el PRIMER match.
    ("semillas de linaza", "semillas de linaza"),
    # Modificadores que SI cambian la identidad del alimento — se conservan a proposito.
    ("Oregano dominicano", "Oregano dominicano"),
    ("judías pintas", "judías pintas"),
    ("arroz blanco", "arroz blanco"),
]


@pytest.mark.parametrize("linea,esperado", _PARTITIVOS,
                         ids=[l for l, _ in _PARTITIVOS])
def test_el_partitivo_no_entra_en_el_nombre_canonico(linea, esperado) -> None:
    got = _extract_canonical_name(linea)
    assert got.lower() == esperado.lower(), (
        f"«{linea}» -> canonico {got!r}, se esperaba {esperado!r}. Con el partitivo dentro, "
        f"el check rechaza un gloss correcto y esa linea se queda en espanol dentro de una "
        f"receta traducida. [{_MARKER}]"
    )


@pytest.mark.parametrize("linea,esperado", _NO_REGRESION,
                         ids=[l for l, _ in _NO_REGRESION])
def test_lo_que_ya_funcionaba_no_se_mueve(linea, esperado) -> None:
    """Ensanchar el prefijo es exactamente como se rompe un extractor."""
    got = _extract_canonical_name(linea)
    assert got == esperado, (
        f"REGRESION: «{linea}» -> {got!r}, antes {esperado!r}. Si el canonico deja de casar "
        f"el catalogo, el efecto es el MISMO defecto al reves. [{_MARKER}]"
    )


def test_una_unidad_corta_no_se_come_el_principio_de_una_palabra_larga() -> None:
    """La leccion de alternacion-no-longest-match que el propio regex documenta: la
    alternativa 'l' consumia la primera letra de 'lonjas' y dejaba «onjas de queso».
    Los partitivos nuevos comparten prefijo entre si ('pedazo'/'pedazos'), asi que el
    `\\b` sigue siendo load-bearing."""
    for linea, esperado in (("2 lonjas de queso", "queso"),
                            ("1 pedazo de pan", "pan"),
                            ("2 pedazos de pan", "pan"),
                            ("1 lata de coco", "coco")):
        assert _extract_canonical_name(linea).lower() == esperado, (
            f"«{linea}» -> {_extract_canonical_name(linea)!r} [{_MARKER}]"
        )
