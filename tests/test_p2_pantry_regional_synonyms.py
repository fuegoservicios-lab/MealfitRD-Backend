"""[P2-PANTRY-REGIONAL-SYNONYMS · 2026-08-21] La Nevera no reconocía NI UN sinónimo regional.

Medido: **0 de 22** pares ES/MX/CO que el propio catálogo declara como alias de la misma fila.
`pimiento`↔`Ají morrón`, `palta`↔`Aguacate`, `ejotes`↔`Vainitas`, `papaya`↔`Lechosa`,
`chayote`↔`Tayota`, `patata`↔`Papa`, `gambas`↔`Camarones`… ninguno.

`pantry_names_match` es puramente léxico (mayúsculas, acentos, cantidad, plural, token a token) y
no sabe que el catálogo tiene una columna `aliases` donde esos pares están escritos. Es la mitad
que le faltaba a `P1-PANTRY-NAME-RESOLUTION`: aquella cerró «"2 huevos" contra la fila Huevo», que
es una diferencia de FORMA; ésta es una diferencia de PALABRA.

CONSECUENCIA, y es la del fallo silencioso: un español marca la compra, la Nevera guarda «Ají
morrón», y cuando se come el plato que pide «pimiento» la deducción **no encuentra la fila**. El
descuento no ocurre, no hay fila en `failed_inventory_deductions` y no hay alerta — exactamente el
desenlace que P1-PANTRY-NAME-RESOLUTION documentó como el peor.

POR QUÉ NO SE PUEDE USAR EL MAPA DE SINÓNIMOS QUE YA HAY. `_CATALOG_ALIAS_INDEX` sale de
`PROTEIN_SYNONYMS`/`CARB_SYNONYMS`/…, que colapsan `pechuga`→`pollo` **a propósito**. CLAUDE.md lo
prohíbe por escrito para esta pregunta: comerte una pechuga descontaría del muslo. La fuente
correcta es la columna `aliases` del catálogo, que es «el mismo alimento con otro nombre», no una
categoría.

Y EL RIESGO SE MIDIÓ ANTES DE ABRIR LA PUERTA. Emparejar por alias a ciegas podría descontar del
alimento equivocado si dos filas comparten una clave. Barrido del catálogo vivo: **5 claves de
1487** las reclama más de una fila, y son genuinamente ambiguas — `nueces` la reclaman
`Almendras fileteadas` **y** `Nueces mixtas`; `mariscos`, tres filas. Así que la regla no es «hay
alias» sino «hay alias **inequívoco**»: las 5 ambiguas quedan fuera y conservan la conducta de hoy.
La lista de exclusión se CALCULA, no se escribe a mano.

Cubre:
  A. Los 22 pares.
  B. Las claves ambiguas NO emparejan.
  C. Lo que el léxico ya distinguía sigue distinguido.
  D. Knob y fail-open.
"""
from __future__ import annotations

import pytest

from constants import pantry_names_match


@pytest.fixture(scope="module", autouse=True)
def catalogo_vivo():
    """Estos pares viven en la columna `aliases` del catálogo: sin catálogo no hay nada que probar."""
    import db_core
    from dotenv import load_dotenv
    import os
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    try:
        if db_core.connection_pool is not None:
            db_core.connection_pool.open()
        import shopping_calculator as sc
        if not (sc.get_master_ingredients() or []):
            pytest.skip("catálogo no disponible (sin DB)")
    except Exception:
        pytest.skip("catálogo no disponible (sin DB)")
    import constants
    constants._reset_pantry_alias_index_cache()
    yield
    constants._reset_pantry_alias_index_cache()


# ── A. Los 22 pares ─────────────────────────────────────────────────────────────────────────────

_PARES = [
    ("Ají morrón", "pimiento"), ("Aguacate", "palta"), ("Vainitas", "ejotes"),
    ("Vainitas", "judías verdes"), ("Auyama", "calabaza"), ("Lechosa", "papaya"),
    ("Chinola", "maracuyá"), ("Tayota", "chayote"), ("Molondrones", "okra"),
    ("Guineo", "banana"), ("Yuca", "mandioca"), ("Batata", "boniato"),
    ("Batata", "camote"), ("Yautía", "malanga"), ("Habichuelas rojas", "frijoles rojos"),
    ("Habichuelas negras", "frijoles negros"), ("Duraznos", "durazno fresco"),
    ("Níspero", "sapodilla"), ("Ajonjolí", "sesamo"), ("Papa", "patata"),
]

# Dos pares que la auditoría contaba entre los 22 y que NO deben emparejar. Los descubrió este
# mismo test al fallar, y en los dos casos el que estaba equivocado era el test:
_PARES_QUE_NO_SON_SINONIMOS = [
    # `Gambas` es su PROPIA fila del catálogo (alta de España), no un alias de `Camarones`. Son dos
    # compras distintas con precio y presentación distintos: emparejarlas descontaría de la fila
    # que no es.
    ("Camarones", "gambas"),
    # `platano` es alias de `Plátano verde`, pero es SUBCONJUNTO de su nombre canónico: una
    # etiqueta genérica, no un sinónimo. Y el caso es doblemente peligroso justo en el país al que
    # sirve este P-fix: para un español «plátano» es el GUINEO, no el plátano verde.
    ("Plátano verde", "plátano"),
]


@pytest.mark.parametrize("fila,sinonimo", _PARES)
def test_la_nevera_reconoce_el_sinonimo_regional(fila, sinonimo):
    assert pantry_names_match(fila, sinonimo), (
        f"«{sinonimo}» no encuentra la fila «{fila}»: la deducción fallaría en silencio"
    )


@pytest.mark.parametrize("fila,termino", _PARES_QUE_NO_SON_SINONIMOS)
def test_lo_que_parecia_sinonimo_y_no_lo_es(fila, termino):
    """Ver el comentario de `_PARES_QUE_NO_SON_SINONIMOS`: la auditoría los contaba entre los 22 y
    los dos habrían descontado del alimento equivocado."""
    assert pantry_names_match(fila, termino) is False


@pytest.mark.parametrize("fila,sinonimo", _PARES[:6])
def test_el_reconocimiento_es_simetrico(fila, sinonimo):
    """La Nevera se consulta en los dos sentidos (fila↔ingrediente de receta)."""
    assert pantry_names_match(sinonimo, fila) is pantry_names_match(fila, sinonimo) is True


@pytest.mark.parametrize("texto", ["2 pimientos", "1 palta", "3 ejotes"])
def test_el_sinonimo_sobrevive_a_la_cantidad(texto):
    """P1-PANTRY-NAME-RESOLUTION cerró la cantidad; esto no puede reabrirla por el otro lado."""
    fila = {"2 pimientos": "Ají morrón", "1 palta": "Aguacate", "3 ejotes": "Vainitas"}[texto]
    assert pantry_names_match(fila, texto)


# ── B. Las claves ambiguas no emparejan ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fila,clave_ambigua", [
    ("Almendras fileteadas", "nueces"),          # 'nueces' la reclaman también las Nueces mixtas
    ("Calamar", "mariscos"),                     # 'mariscos': Calamar, Mejillones y Pulpo
    ("Mejillones", "mariscos"),
    ("Filete de pescado blanco", "tilapia"),     # 'tilapia': dos filas
    ("Filete de pescado blanco", "mero"),        # 'mero': dos filas
])
def test_una_clave_ambigua_no_empareja(fila, clave_ambigua):
    """El error caro del lado contrario: con la fila «Almendras fileteadas» en la Nevera y una
    receta que pide «nueces», emparejar descontaría de las almendras.

    [reescrito tras la mutación] La primera versión comparaba dos nombres CANÓNICOS
    (`Almendras fileteadas` vs `Nueces mixtas`) y por eso pasaba con el filtro de ambigüedad
    quitado: dos canónicos distintos nunca chocan entre sí. El riesgo vive en consultar la clave
    ambigua DESNUDA contra una fila, que es como llega de verdad desde una receta. La mutación
    sobrevivió y así se descubrió que el test miraba donde no era."""
    assert pantry_names_match(fila, clave_ambigua) is False


# ── C. Lo que el léxico ya distinguía ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("Leche de coco", "Leche"),
    ("Arroz integral", "Arroz"),
    ("Pechuga de pollo", "Muslo de pollo"),
    ("Aceite de oliva", "Aceite de coco"),
    ("Habichuelas rojas", "Habichuelas negras"),
])
def test_los_alimentos_distintos_siguen_siendo_distintos(a, b):
    """El contrato original: «un no-match degrada a "no está en tu nevera", que es un desenlace
    seguro y visible». Abrir la puerta a los alias no puede colapsar compras distintas."""
    assert pantry_names_match(a, b) is False


def test_la_pechuga_no_descuenta_del_muslo():
    """El caso que CLAUDE.md nombra explícitamente al prohibir `GLOBAL_REVERSE_MAP` aquí. Se prueba
    aparte porque es la razón por la que este arreglo NO reusa ese mapa."""
    assert pantry_names_match("Pechuga de pollo", "pollo") is False


# ── D. Knob y fail-open ─────────────────────────────────────────────────────────────────────────

def test_el_knob_permite_revertir(monkeypatch):
    """Cambia el comportamiento de la DEDUCCIÓN, o sea de lo que el usuario ve descontarse de su
    Nevera: knob propio, según la convención del repo."""
    import constants
    monkeypatch.setenv("MEALFIT_PANTRY_ALIAS_MATCH", "false")
    constants._reset_pantry_alias_index_cache()
    assert pantry_names_match("Ají morrón", "pimiento") is False
    monkeypatch.delenv("MEALFIT_PANTRY_ALIAS_MATCH", raising=False)
    constants._reset_pantry_alias_index_cache()
    assert pantry_names_match("Ají morrón", "pimiento") is True


def test_sin_catalogo_conserva_la_conducta_lexica(monkeypatch):
    """Corre en el camino de la deducción: si el catálogo no está disponible, la Nevera tiene que
    seguir funcionando con lo léxico, no reventar."""
    import constants
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("DB caída")))
    constants._reset_pantry_alias_index_cache()
    assert pantry_names_match("Ají morrón", "pimiento") is False
    assert pantry_names_match("Ají morrón", "aji morron") is True
    constants._reset_pantry_alias_index_cache()
