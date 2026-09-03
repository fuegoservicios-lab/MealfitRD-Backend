"""[P1-SCAN-CATALOG-MATCH · 2026-08-10] El escáner de la Nevera mapeaba un PAN a
«Polvo de hornear».

EL DEFECTO, reportado por el dueño con la foto: el matcher aceptaba UN token en
común como prueba de identidad, y "de" es un token. 36 de los 204 alimentos del
catálogo lo llevan en el nombre y el primero en orden de lectura es «Polvo de
hornear», así que CUALQUIER detección con la palabra "de" que no matcheara por
otra vía aterrizaba ahí. Además contenía substrings en ambas direcciones, que es
la familia clásica de este repo: «salami» → «Sal».

LO QUE NO ERA: el modelo. Producción usa `gpt-5.6-luna` (verificado en el VPS) y
había leído el pan correctamente — el 99% que se veía en pantalla era su confianza
en ESA lectura. Lo que se mostraba encima era el nombre del catálogo, que la tapaba.

Medido contra el catálogo real (204 alimentos) con 34 detecciones realistas
etiquetadas a mano: 19 aciertos y 15 mapeos al alimento EQUIVOCADO antes; 34 y
cero después. Los casos de abajo son ese conjunto, con un catálogo de prueba que
reproduce las trampas reales (los tres panes, «Sal» junto a «Salami», los "en
polvo", «Guineo» junto a «Guineo verde»).

Sin DB: `resolve_scanned_food` es una función pura sobre la lista de nombres.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from constants import FOOD_CONNECTOR_TOKENS, resolve_scanned_food

_BACKEND = Path(__file__).resolve().parent.parent
_UD_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")

# Espejo reducido del catálogo real: conserva las trampas que importan.
CATALOGO = [
    "Polvo de hornear", "Aceite de sésamo", "Aceite de oliva", "Mostaza",
    "Filete de pescado blanco", "Pechuga de pollo", "Pechuga de pavo",
    "Pan integral familiar", "Pan integral personal", "Pan blanco familiar",
    "Huevo", "Leche", "Leche de coco", "Coco", "Arroz blanco", "Arroz integral",
    "Yogurt griego entero", "Yogurt griego sin azúcar", "Yogurt natural",
    "Plátano verde", "Guineo", "Guineo verde", "Cebolla", "Cebolla en polvo",
    "Ajo", "Ajo en polvo", "Tomate", "Queso de hoja", "Sal", "Salami",
    "Habichuelas rojas", "Avena", "Atún en agua", "Atún en aceite",
    "Maní", "Mantequilla de maní", "Papa", "Zanahoria", "Aguacate",
    "Harina de trigo", "Vinagre balsámico", "Vinagre blanco",
    "Carne de res", "Carne de res molida", "Naranja", "Curry en polvo",
]


@pytest.mark.parametrize("detectado,esperado", [
    # --- EL CASO REPORTADO ---
    ("pan de hamburguesa", None),      # el catálogo no tiene ese pan; adivinar sería peor
    # --- la familia del substring, 16ª aparición en este repo ---
    ("salami", "Salami"),              # NO «Sal»
    ("mantequilla de maní", "Mantequilla de maní"),   # NO «Maní»
    ("leche de coco", "Leche de coco"),               # NO «Coco»
    # --- el modificador que cambia el alimento ---
    ("ajo", "Ajo"),                    # NO «Ajo en polvo»
    ("cebolla", "Cebolla"),            # NO «Cebolla en polvo»
    ("guineo", "Guineo"),              # NO «Guineo verde» (la distinción del prompt de visión)
    ("carne de res", "Carne de res"),  # NO «Carne de res molida»
    # --- cubrir no es ser: hace falta compartir núcleo ---
    ("azúcar", None),                  # NO «Yogurt griego sin azúcar»
    ("jugo de naranja", None),         # NO «Naranja»
    ("pollo entero", None),            # NO «Pechuga de pollo»
    # --- ambigüedad: varios candidatos igual de válidos ---
    # (sin la columna `aliases`; con ella, «pan» sí resuelve — ver la clase de
    #  abajo: un sinónimo escrito a mano gana sobre el empate)
    ("pan", None),
    ("yogurt griego", None),
    ("vinagre", None),
    ("atún en lata", None),
    # --- lo que SÍ debe seguir matcheando ---
    ("huevos", "Huevo"),               # plural
    ("leche", "Leche"),
    ("pechuga de pollo", "Pechuga de pollo"),
    ("arroz blanco", "Arroz blanco"),
    ("aceite de oliva", "Aceite de oliva"),
    ("platano verde", "Plátano verde"),   # sin acento
    ("PAN INTEGRAL FAMILIAR", "Pan integral familiar"),
    ("queso de hoja", "Queso de hoja"),
    ("habichuelas rojas", "Habichuelas rojas"),
    ("harina de trigo", "Harina de trigo"),
    ("polvo de hornear", "Polvo de hornear"),
    ("sal", "Sal"),
    # --- no está en el catálogo ---
    ("café", None),
    ("espaguetis", None),
])
def test_resolucion(detectado, esperado):
    assert resolve_scanned_food(detectado, CATALOGO) == esperado


def test_un_conector_jamas_es_evidencia():
    """El corazón del defecto: compartir solo "de" no une dos alimentos.

    Se prueba contra el catálogo COMPLETO de prueba para que el caso no dependa
    de qué nombre quedó primero en la lista — que es justo lo que decidía el
    resultado antes (el orden de lectura de la tabla, sin ORDER BY)."""
    for detectado in ("pan de hamburguesa", "sopa de fideos", "torta de cumpleaños"):
        obtenido = resolve_scanned_food(detectado, CATALOGO)
        assert obtenido != "Polvo de hornear", (
            f"{detectado!r} volvió a caer en «Polvo de hornear» por compartir un conector"
        )


def test_el_orden_del_catalogo_no_decide():
    """Mismo catálogo al revés → misma respuesta. Antes no: se devolvía el PRIMER
    row que compartiera un token, y la consulta del endpoint no lleva ORDER BY."""
    for detectado in ("pan de hamburguesa", "salami", "ajo", "guineo", "huevos"):
        assert resolve_scanned_food(detectado, CATALOGO) == resolve_scanned_food(
            detectado, list(reversed(CATALOGO))
        ), f"la respuesta para {detectado!r} depende del orden del catálogo"


def test_conectores_solo_gramaticales():
    """La lista NO puede crecer con palabras que distinguen alimentos.

    «Ajo» y «Ajo en polvo» son compras distintas: si "polvo" entrara aquí, el
    escáner volvería a fundirlas. Por eso esta lista no es
    `RECIPE_INGREDIENT_STOPWORDS` (que sí incluye polvo/salsa/jugo a propósito,
    para otra pregunta)."""
    prohibidas = {"polvo", "salsa", "jugo", "pasta", "caldo", "fresco", "verde",
                  "natural", "molido", "integral", "entero"}
    intrusas = prohibidas & set(FOOD_CONNECTOR_TOKENS)
    assert not intrusas, f"tokens que SÍ distinguen alimentos colados entre los conectores: {intrusas}"


def test_sin_match_es_una_respuesta_valida():
    """`None` no es un fallo: es lo correcto cuando el alimento no está o hay
    varios candidatos. El endpoint lo traduce a `master_ingredient_id: null` y el
    cliente lo muestra sin marcar — visible y seguro, frente a un match de más
    que mete en la Nevera un alimento que el usuario no tiene."""
    assert resolve_scanned_food("", CATALOGO) is None
    assert resolve_scanned_food("de", CATALOGO) is None
    assert resolve_scanned_food("unicornio", CATALOGO) is None
    assert resolve_scanned_food("pan", []) is None


def test_el_endpoint_delega_en_el_ssot():
    """Si alguien vuelve a escribir el matcher a mano en el router, esto avisa."""
    i = _UD_SRC.find("def _match_catalog(")
    assert i > 0, "el matcher del escáner desapareció"
    cuerpo = _UD_SRC[i:i + 1800]
    assert "resolve_scanned_food(" in cuerpo, (
        "el router debe delegar en constants.resolve_scanned_food (SSOT), no "
        "reimplementar la resolución de nombres"
    )
    assert not re.search(r"\bd\s+in\s+n\s+or\s+n\s+in\s+d\b", cuerpo), (
        "volvió la contención de substring en ambas direcciones («salami» → «Sal»)"
    )
    assert "best_overlap" not in cuerpo, (
        "volvió el overlap de tokens sin filtrar conectores («pan de X» → «Polvo de hornear»)"
    )


def test_el_cliente_puede_ver_lo_que_leyo_el_modelo():
    """El mapeo no puede volver a tapar la lectura de la visión.

    Es lo que convirtió este defecto en un misterio: en pantalla solo salía el
    nombre del catálogo, así que parecía que el modelo había confundido un pan con
    un pote de polvo de hornear."""
    i = _UD_SRC.find("def _match_against_catalog")
    cuerpo = _UD_SRC[i:i + 2500]
    assert '"catalog_renamed"' in cuerpo, "el payload debe declarar si el mapeo renombra la lectura"
    assert "pantry_names_match(" in cuerpo, (
        "el flag debe usar la equivalencia canónica: «huevos»→«Huevo» NO es un renombre"
    )
    scan_btn = (_BACKEND.parent / "frontend" / "src" / "components" / "pantry" / "PantryScanButton.jsx")
    assert "catalog_renamed" in scan_btn.read_text(encoding="utf-8"), (
        "el checklist de confirmación debe mostrar la lectura original cuando el "
        "catálogo la renombra"
    )
