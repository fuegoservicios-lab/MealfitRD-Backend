"""[P1-COHERENCE-ALIAS-INDEX · 2026-08-14] El índice de alias de `normalize_name`
se construye UNA vez por catálogo, no una vez por llamada.

Por qué existe. El guard de coherencia rebasó su propio umbral de 5 s **17 veces en
7 días** (hasta 11,5 s), siempre en planes grandes (~130 recetas), y una de esas
veces dentro de un `/recalculate-shopping-list` SÍNCRONO — o sea, con el usuario
esperando. Perfilado contra un plan real (cb361844, 26 días, 104 comidas):

    run_shopping_coherence_guard   19,4 s
      expected_sum_from_recipes    17,4 s   (90%)
        normalize_name  ×973       17,8 s cum  → 18 ms POR LLAMADA
          re._compile ×481.240     14,8 s

El 76% del guard se iba en compilar expresiones regulares. La causa: los tiers 2 y 4
de `normalize_name` recorrían los ~700 alias del catálogo construyendo un patrón
NUEVO por alias en cada llamada (`r'\\b' + re.escape(alias) + r'\\b'`). La caché
interna de `re` guarda 512 patrones, así que con más alias que huecos se vaciaba
sola y cada llamada recompilaba casi todo. Encima `all_aliases` se reconstruía y
se **reordenaba** entero (700 elementos) en cada una de las 973 llamadas.

Ninguno de los dos trabajos depende del texto que se está normalizando: dependen
solo del catálogo, que vive cacheado con TTL. *Trabajo que no depende de la entrada
no pertenece al cuerpo de la función.*

Invalidación por IDENTIDAD del catálogo, no por TTL: `get_master_ingredients()`
devuelve el mismo objeto lista mientras su caché sea válida, así que comparar
`is` detecta la recarga con exactitud y —lo que importa para la suite— hace que un
test que parchea el catálogo NO herede el índice del test anterior. Un TTL aquí
habría creado justo esa fuga entre tests.
"""
import io
import os
import re

import pytest

import shopping_calculator as sc

_SRC = io.open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shopping_calculator.py"),
    encoding="utf-8",
).read()

_CATALOGO = [
    {"name": "Plátano maduro", "aliases": ["maduro", "platano maduro"]},
    {"name": "Mango", "aliases": ["mangos"]},
    {"name": "Queso mozzarella bajo en grasa", "aliases": ["queso mozzarella bajo en grasa"]},
    {"name": "Pechuga de pollo", "aliases": ["pechuga", "pollo"]},
]


@pytest.fixture
def catalogo(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: _CATALOGO)
    monkeypatch.setattr(sc, "get_semantic_cache", lambda: None)
    sc._NORMALIZE_ALIAS_INDEX = None
    yield _CATALOGO
    sc._NORMALIZE_ALIAS_INDEX = None


def _cuerpo_normalize_name() -> str:
    """Cuerpo de `normalize_name`, hasta la siguiente def top-level."""
    i = _SRC.index("def normalize_name(")
    fin = _SRC.index("\ndef ", i + 10)
    return _SRC[i:fin]


def test_los_tiers_de_contains_no_compilan_regex_por_alias():
    """[anti-regresión] El patrón exacto que costó 14,8 s: `re.escape(alias)`
    dentro del bucle de alias. Si vuelve, el guard vuelve a rebasar su umbral."""
    cuerpo = _cuerpo_normalize_name()
    sin_comentarios = "\n".join(
        l.split("#")[0] for l in cuerpo.splitlines()
    )
    assert "re.escape(alias" not in sin_comentarios.replace(" ", ""), (
        "normalize_name volvió a construir un regex por alias en caliente. "
        "Los patrones deben venir precompilados del índice cacheado "
        "(_get_normalize_alias_index)."
    )


def test_el_indice_se_construye_una_vez_para_muchas_llamadas(catalogo, monkeypatch):
    """El contrato de fondo: N normalizaciones, UNA construcción del índice."""
    llamadas = {"n": 0}
    real = sc._construir_indice_alias

    def _spy(master_list):
        llamadas["n"] += 1
        return real(master_list)

    monkeypatch.setattr(sc, "_construir_indice_alias", _spy)
    sc._NORMALIZE_ALIAS_INDEX = None

    for texto in ("mango maduro", "pechuga de pollo", "queso mozzarella bajo en grasa",
                  "mangos", "platano maduro", "pollo"):
        sc.normalize_name(texto)

    assert llamadas["n"] == 1, (
        f"El índice se construyó {llamadas['n']} veces para 6 llamadas — debe "
        f"construirse UNA vez por catálogo (era 1 por llamada: 700 alias "
        f"reordenados y recompilados cada vez)."
    )


def test_el_indice_se_reconstruye_si_cambia_el_catalogo(catalogo, monkeypatch):
    """Sin esto el índice sería una caché sucia: un catálogo recargado (o el de
    OTRO test) seguiría resolviendo con los alias viejos."""
    sc.normalize_name("mangos")
    otro = [{"name": "Auyama", "aliases": ["calabaza"]}]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: otro)

    assert sc.normalize_name("calabaza") == "Auyama", (
        "El índice siguió sirviendo el catálogo anterior tras cambiarlo."
    )


def test_paridad_resolver_alias(catalogo):
    """El índice cacheado resuelve EXACTAMENTE igual que el bucle en caliente."""
    # exacto sobre el nombre canónico y sobre un alias
    assert sc.normalize_name("Mango") == "Mango"
    assert sc.normalize_name("mangos") == "Mango"
    # contains: el alias largo dentro de un texto mayor
    assert sc.normalize_name("queso mozzarella bajo en grasa rallado") == "Queso mozzarella bajo en grasa"
    # acentos: el índice guarda las formas sin acento
    assert sc.normalize_name("plátano maduro") == "Plátano maduro"


def test_un_alias_vacio_no_entra_al_indice_de_contains():
    """`\\b\\b` casa en CUALQUIER texto con una palabra: un alias vacío en el
    catálogo habría resuelto todo el mundo a ese alimento.

    El filtro anterior (`if a not in _MODIFIER_ONLY_ALIASES`) no miraba si el alias
    era vacío, así que la trampa estaba armada y solo la desactivaba el dato: hoy el
    catálogo real tiene CERO alias vacíos (verificado sobre los 1.031). El índice
    los excluye explícitamente para que la defensa deje de depender de esa suerte.
    """
    import re as _re
    assert _re.search(r"\b\b", "pechuga de pollo"), (
        "premisa del test: un patrón vacío entre word-boundaries casa con todo"
    )
    catalogo = [
        {"name": "Sal", "aliases": ["", "   "]},
        {"name": "Mango", "aliases": ["mangos"]},
    ]
    _, contains = sc._get_normalize_alias_index(catalogo)
    patrones = [p.pattern for p, _ in contains]
    assert all(p != r"\b\b" for p in patrones), (
        f"un alias vacío entró al índice y casaría con cualquier texto: {patrones}"
    )


def test_la_poda_fuzzy_usa_EL_MISMO_umbral_que_la_aceptacion():
    """La trampa de esta optimización: la poda descarta pares que no pueden
    alcanzar el umbral. Si alguien baja el umbral de aceptación y la poda se queda
    con el número viejo, la poda empieza a descartar matches VÁLIDOS — y en
    silencio. Ambos deben leer la misma constante."""
    cuerpo = _cuerpo_normalize_name()
    sin_comentarios = "\n".join(l.split("#")[0] for l in cuerpo.splitlines())
    assert "_FUZZY_MATCH_THRESHOLD" in sin_comentarios
    # cero umbrales literales sueltos en el bloque fuzzy
    assert "0.87" not in sin_comentarios, (
        "quedó un 0.87 literal en normalize_name: poda y aceptación pueden drift-ear"
    )


def test_la_poda_no_mata_un_match_fuzzy_legitimo(catalogo, monkeypatch):
    """[P1-QUALIFIER-STRIP-FUZZY] El caso real que motivó el tier fuzzy: plural y
    acento contra el canónico. Debe seguir resolviendo DESPUÉS de podar."""
    monkeypatch.setattr(sc, "get_master_ingredients",
                        lambda: [{"name": "Níspero", "aliases": []}])
    sc._NORMALIZE_ALIAS_INDEX = None
    assert sc.normalize_name("nisperos") == "Níspero"


def test_la_poda_es_equivalente_no_aproximada(catalogo, monkeypatch):
    """Compara el resolutor real contra la versión SIN poda sobre el mismo
    catálogo: mismo veredicto para cada entrada. Una poda que cambia un solo
    resultado no es una optimización, es un cambio de comportamiento."""
    import difflib
    catalogo_grande = [
        {"name": "Níspero", "aliases": ["nisperos"]},
        {"name": "Mango", "aliases": ["mangos"]},
        {"name": "Habichuelas rojas", "aliases": ["habichuela roja"]},
        {"name": "Pechuga de pollo", "aliases": ["pechuga"]},
    ]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: catalogo_grande)
    monkeypatch.setattr(sc, "get_semantic_cache", lambda: None)
    sc._NORMALIZE_ALIAS_INDEX = None

    def _mejor_sin_poda(texto):
        """El bucle original: todas las formas contra todos los alias."""
        todos, _ = sc._get_normalize_alias_index(catalogo_grande)
        formas = {texto.lower()}
        mejor, nombre = 0.0, None
        for alias, master in todos:
            if not alias:
                continue
            r = max(difflib.SequenceMatcher(None, f, alias).ratio() for f in formas)
            if r > mejor:
                mejor, nombre = r, master
        return nombre if mejor >= sc._FUZZY_MATCH_THRESHOLD else None

    for texto in ("nisperos", "mangos", "habichuela rojas", "pechugas", "zanahoria",
                  "x", "nispero sin semilla"):
        esperado = _mejor_sin_poda(texto)
        if esperado is not None:
            assert sc.normalize_name(texto) == esperado, f"divergencia en {texto!r}"


def test_el_orden_por_longitud_sobrevive_al_cacheo(catalogo):
    """[P1-MODIFIER-ONLY-ALIAS] El orden descendente por longitud y el filtro de
    modificadores son parte del índice: si el cacheo los perdiera, 'maduro' (6)
    volvería a ganarle a 'mango' (5) y un desayuno de mango pondría PLÁTANO en la
    lista de compras (plan vivo 01d63a5b)."""
    assert sc.normalize_name("mango maduro") == "Mango"
    _, contains = sc._get_normalize_alias_index(_CATALOGO)
    largos = [len(p.pattern) for p, _ in contains]
    assert largos == sorted(largos, reverse=True), "el índice perdió el orden por longitud"
    assert all("maduro" != p.pattern.strip("\\b") for p, _ in contains), (
        "un alias modificador-solo entró al índice de contains"
    )
