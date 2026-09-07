# -*- coding: utf-8 -*-
"""[P1-CULINARY-ALIAS-INDEX · 2026-09-07] El índice culinario carga los ALIAS del catálogo.

Hasta hoy `build_culinary_index` leía sólo `row["name"]` — mientras su propio docstring
decía «regex del ALIAS» y `find_catalog_foods` decía «alias más largo gana». El código
hablaba de alias por todas partes y no cargaba ninguno: el vocabulario de TODA la capa
culinaria eran los 348 nombres canónicos, así que `Clara de huevo`, `Yema de huevo`,
`Yogurt griego entero` y `Queso blanco` eran INVISIBLES cuando la receta los nombraba
como el catálogo mismo dice que se les llama. Medido sobre la flota el día del arreglo:
**597 de 1.194 comidas (50 %)** mencionaban al menos un alimento que ninguna capa podía
ver. Un detector no puede acusar lo que no sabe nombrar.

Ampliar vocabulario es la vía clásica al falso positivo (19 colisiones por subcadena
documentadas en este repo), así que el índice trae CUATRO reglas de seguridad y este
fichero las ancla una a una. Las reglas 3 y 4 no son teóricas: cada una nació de una
captura real que se perdía al meter los alias sin ella, encontrada midiendo contra las
140 comidas que el dueño etiquetó a mano.

Marcador contra esas 140 (80 del golden set + 60 retenidas), antes → después:
precisión 93,5 % → 94,5 %, un falso positivo menos, +4 capturas verdaderas, y CINCO
acusaciones falsas retiradas que el marcador contaba como aciertos porque la etiqueta es
por COMIDA, no por acusación — una capa puede acertar la comida por la razón equivocada.
"""
import re

import pytest

import culinary_coherence as cc


# ─────────────────────────────────────────────────────────────────────────────────────────
# Catálogo sintético: mismas formas que las filas reales que motivaron cada regla.
# ─────────────────────────────────────────────────────────────────────────────────────────
def _fila(name, aliases=None, prep=None, rte=False):
    return {"name": name, "aliases": list(aliases or []),
            "prep_methods": prep, "ready_to_eat": rte}


CATALOGO = [
    _fila("Clara de huevo", ["claras", "claras de huevo", "egg white"]),
    _fila("Queso mozzarella", ["mozzarella", "mozzarella light"]),
    _fila("Harina de trigo", ["harina", "harina blanca", "harina de todo uso"]),
    _fila("Harina de maíz", ["harina de maiz precocida"]),
    _fila("Pasta integral", ["pasta", "pasta integral de trigo"]),
    _fila("Plátano maduro", ["platano maduro"]),
    _fila("Plátano verde", ["platano", "platano verde"]),
    _fila("Salsa de soya", ["soya", "salsa de soja"]),
    _fila("Soya texturizada", ["tvp", "proteina de soya"]),
    _fila("Pulpo", ["mariscos"]),
    _fila("Calamar", ["mariscos"]),
    _fila("Nueces mixtas", ["nueces"]),
    _fila("Almendras fileteadas", ["almendras", "nueces"]),
    _fila("Repollo", ["repollo morado"]),
    _fila("Repollo morado", []),
    _fila("Ciruela", ["ciruelas"]),
    _fila("Mandarina", ["mandarinas"]),
    _fila("Carne de res", ["res"]),
    _fila("Queso fresco", []),
]


@pytest.fixture(scope="module")
def idx():
    return cc.build_culinary_index(CATALOGO)


# ─────────────────────────────────────────────────────────────────────────────────────────
# El contrato central: los alias entran.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_el_indice_carga_los_alias_no_solo_los_nombres(idx):
    """Sin esto la mitad de la flota era invisible para la capa culinaria."""
    assert cc.find_catalog_foods("2 claras de huevo", idx) == ["Clara de huevo"]
    assert cc.find_catalog_foods("ralla la mozzarella light", idx) == ["Queso mozzarella"]
    assert cc.find_catalog_foods("almendras fileteadas al final", idx) == ["Almendras fileteadas"]


def test_el_indice_sigue_conociendo_todos_los_nombres_canonicos(idx):
    """Los alias AÑADEN vocabulario; jamás pueden quitar un nombre canónico del índice."""
    for fila in CATALOGO:
        assert cc._norm(fila["name"]) in idx, f"{fila['name']} desapareció del índice"


# ─────────────────────────────────────────────────────────────────────────────────────────
# Regla 1 — el nombre canónico SIEMPRE gana.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_regla1_un_alias_ajeno_no_desplaza_a_un_nombre_canonico(idx):
    """`repollo morado` es alias de `Repollo` Y nombre de `Repollo morado`: manda el nombre.

    Si ganara el alias, una receta con repollo morado quedaría archivada como repollo a
    secas y las capas razonarían sobre el alimento equivocado."""
    assert idx[cc._norm("repollo morado")]["name"] == "Repollo morado"
    assert cc.find_catalog_foods("media taza de repollo morado", idx) == ["Repollo morado"]


# ─────────────────────────────────────────────────────────────────────────────────────────
# Regla 2 — un alias ambiguo se DESCARTA, no se reparte.
# ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("alias", ["mariscos", "nueces"])
def test_regla2_alias_reclamado_por_dos_alimentos_no_entra(alias, idx):
    """Elegir uno por orden de fila sería inventar una identidad que el dato no tiene.

    `mariscos` lo reclaman Pulpo y Calamar; `nueces`, Nueces mixtas y Almendras fileteadas.
    Ninguno identifica un alimento, así que la respuesta honesta es «no sé»."""
    assert cc._norm(alias) not in idx
    assert cc.find_catalog_foods(f"añade los {alias} al sofrito", idx) == []


def test_regla2_no_castiga_al_alias_inequivoco_del_mismo_alimento(idx):
    """`almendras` sólo lo reclama Almendras fileteadas: entra, aunque su vecino `nueces` no."""
    assert cc.find_catalog_foods("30 g de almendras", idx) == ["Almendras fileteadas"]


# ─────────────────────────────────────────────────────────────────────────────────────────
# Regla 3 — token ambiguo entre nombres canónicos.
# ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("palabra", ["platano", "soya"])
def test_regla3_una_palabra_que_es_token_de_dos_canonicos_no_identifica(palabra, idx):
    """`platano` es token de `Plátano maduro` y de `Plátano verde`.

    Nació de una captura perdida: en una receta de plátano MADURO, el paso «maja el
    plátano» añadía un span espurio de `Plátano verde` que desplazaba al objeto real del
    verbo, y con él se perdía la acusación de V1 sobre unas lentejas horneadas. La
    ambigüedad se mide contra los nombres canónicos, no sólo entre alias."""
    assert cc._norm(palabra) not in idx


def test_regla3_el_platano_maduro_se_resuelve_por_su_nombre_completo(idx):
    assert cc.find_catalog_foods("medio platano maduro", idx) == ["Plátano maduro"]
    assert cc.find_catalog_foods("2 platanos verdes", idx) == ["Plátano verde"]


# ─────────────────────────────────────────────────────────────────────────────────────────
# Regla 4 — un alias de UNA palabra que nombra una FORMA no identifica un alimento.
# ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("forma", ["harina", "pasta"])
def test_regla4_alias_de_forma_generica_no_entra(forma, idx):
    """LA MISMA ceguera que `_V3_FORMA_GENERICA` cerró en `_mencionado_por_prefijo`, entrando
    por otra puerta: aquel guard filtra PREFIJOS del nombre canónico y un alias llega como
    clave entera, así que jamás pasaba por él.

    Medido: sin esta regla V3 perdía sus tres mejores capturas — las tres decían literalmente
    «la harina de trigo queda sin utilizar»."""
    assert cc._norm(forma) not in idx


def test_regla4_la_harina_de_otro_cereal_no_es_harina_de_trigo(idx):
    """«harina de avena» resolviendo a `Harina de trigo` es una identidad FALSA, no un acierto."""
    assert "Harina de trigo" not in cc.find_catalog_foods("muele la avena hasta harina fina", idx)
    assert cc.find_catalog_foods("½ taza de harina de maiz precocida", idx) == ["Harina de maíz"]


def test_regla4_el_nombre_completo_sigue_entrando(idx):
    """La regla descarta el alias `harina`, nunca el alimento: `Harina de trigo` sigue vivo."""
    assert cc.find_catalog_foods("120 g de harina de trigo", idx) == ["Harina de trigo"]


# ─────────────────────────────────────────────────────────────────────────────────────────
# Los `\b` que ya protegían al nombre protegen igual al alias.
# ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("texto,ausente", [
    ("100 g de queso fresco", "Carne de res"),      # 'res' ⊂ que-so f-res-co (documentada)
    ("una ensalada templada", "Carne de res"),
])
def test_los_alias_cortos_no_casan_dentro_de_otra_palabra(texto, ausente, idx):
    """`res` (alias de `Carne de res`) es de tres letras y NO casa dentro de «fresco».

    No es una defensa nueva: es la misma `\\b` + `_sing_plural_pattern` que ya usaba el
    nombre canónico. Por eso ampliar a los alias no reabre la clase de bug por subcadena
    que este repo lleva 19 veces documentada."""
    assert ausente not in cc.find_catalog_foods(texto, idx)


# ─────────────────────────────────────────────────────────────────────────────────────────
# V7a: la medida DETRÁS del alimento describe el corte, y un rango no es un conteo.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_v7a_la_medida_tras_el_alimento_describe_el_corte_no_el_conteo(idx):
    """«corta 2 ciruelas medianas EN GAJOS» son dos ciruelas partidas, no «2 gajos».

    El guard descartaba la pieza entera al ver `gajos` en la cola, y por eso V7a no veía su
    propio caso de manual (lista «3 ciruelas», paso «corta 2 ciruelas»). Se compara la
    POSICIÓN de la medida contra la del alimento, no su presencia."""
    piezas = cc._v7_piezas("corta 2 ciruelas medianas en gajos", idx)
    assert piezas == {"Ciruela": 2.0}


def test_v7a_la_medida_delante_del_alimento_sigue_invalidando_el_conteo(idx):
    """«2 cucharadas de pasta integral» son cucharadas, no piezas — eso lo mide V6.

    Es la razón de ser del guard original y no puede caerse con el arreglo del corte."""
    assert cc._v7_piezas("2 cucharadas de pasta integral", idx) == {}


def test_v7a_el_caso_completo_de_las_ciruelas_dispara(idx):
    """Lista «3 ciruelas», pasos «corta 2 ciruelas»: la comida compra una de más."""
    meal = {
        "meal": "Merienda",
        "name": "Vasito de ciruela",
        "ingredients": ["3 ciruelas medianas", "15 g de avena"],
        "recipe": ["Mise en place: corta 2 ciruelas medianas en gajos.",
                   "Montaje: coloca las ciruelas encima."],
    }
    viols = cc._v7a_lista_compra_de_mas({}, meal, idx)
    assert [v["food"] for v in viols] == ["Ciruela"]


@pytest.mark.parametrize("rango", ["1-2 mandarinas medianas", "1–2 mandarinas medianas",
                                   "1 a 2 mandarinas medianas"])
def test_v7a_el_techo_de_un_rango_no_fija_una_cantidad(rango, idx):
    """«1–2 mandarinas» con un paso que dice «pela la mandarina» NO se contradice: el límite
    inferior autoriza el singular. Era el único falso positivo del detector sobre las 60
    comidas retenidas. Un rango no es un conteo."""
    meal = {
        "meal": "Merienda",
        "name": "Bowl de yogurt",
        "ingredients": [rango, "1 taza de yogurt"],
        "recipe": ["Mise en place: pela la mandarina y separa los gajos.",
                   "Montaje: añade los gajos de mandarina."],
    }
    assert cc._v7a_lista_compra_de_mas({}, meal, idx) == []


def test_v7a_sin_rango_el_plural_contra_singular_sigue_disparando(idx):
    """El guard del rango no puede desactivar la señal del número gramatical en el caso llano."""
    meal = {
        "meal": "Merienda",
        "name": "Bowl de yogurt",
        "ingredients": ["2 mandarinas medianas", "1 taza de yogurt"],
        "recipe": ["Mise en place: pela la mandarina y separa los gajos.",
                   "Montaje: añade los gajos de mandarina."],
    }
    assert [v["food"] for v in cc._v7a_lista_compra_de_mas({}, meal, idx)] == ["Mandarina"]


# ─────────────────────────────────────────────────────────────────────────────────────────
# Anclas de código: que un renombre falle aquí antes que en producción.
# ─────────────────────────────────────────────────────────────────────────────────────────
def test_el_indice_lee_la_columna_aliases_en_el_codigo():
    """Ancla textual: si alguien vuelve a construir el índice sólo con `name`, este test cae
    antes de que la mitad de la flota se vuelva invisible otra vez, en silencio."""
    import inspect

    src = inspect.getsource(cc.build_culinary_index)
    assert 'row.get("aliases")' in src, "build_culinary_index dejó de leer la columna aliases"
    assert "_V3_FORMA_GENERICA" in src, "se perdió la regla de la forma genérica"
    assert "tokens_ambiguos" in src, "se perdió la regla del token ambiguo entre canónicos"


def test_v7_piezas_compara_posiciones_no_presencia():
    """Ancla textual del arreglo del corte: `_V7_MEDIDA_RE` no puede volver a descartar por
    el mero hecho de aparecer en la cola."""
    import inspect

    src = inspect.getsource(cc._v7_piezas)
    assert "_catalog_food_spans" in src, "V7a dejó de comparar posiciones medida↔alimento"
    assert "_V7_RANGO_RE" in src, "V7a dejó de reconocer el techo de un rango"


def test_la_regex_del_rango_no_casa_un_numero_suelto():
    """`_V7_RANGO_RE` sólo debe casar «N-», «N–» o «N a » al final del texto previo."""
    assert cc._V7_RANGO_RE.search("mide 1-")
    assert cc._V7_RANGO_RE.search("mide 1 a ")
    assert not cc._V7_RANGO_RE.search("mide 1 ")
    assert not cc._V7_RANGO_RE.search("corta ")
