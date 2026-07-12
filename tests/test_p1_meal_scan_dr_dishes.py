"""[P1-MEAL-SCAN-DR-DISHES · 2026-07-12] Prompt criollo del meal-scan + modal sin scroll.

Vivo (owner, primer scan real): foto de los tres golpes (mangu + huevo frito +
salami) → gemma nombro el mangu "arroz" y estimo 450 kcal (subestimacion de un
plato de ~750). Ademas la fase de revision del modal exigia scroll para llegar
a "Registrar comida" (preview 4:3 de ~345px + campos apilados + botones en
columna).

Fix backend: desambiguacion explicita en _MEAL_VISION_PROMPT (mangu = pure
compacto con cebolla encurtida vs arroz = granos sueltos; platos criollos
comunes) + regla "estima cada componente POR SEPARADO y SUMA".
Fix frontend (P1-MEAL-SCAN-POLISH): preview banner 120px en revision, Tipo +
Porcion en una fila, acciones en una fila (Registrar protagonista).
tooltip-anchor: P1-MEAL-SCAN-DR-DISHES
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)

# OJO (lección de este mismo test, primer intento): las aserciones van sobre el
# STRING RUNTIME, no sobre el source — un frase partida entre dos literales
# concatenados ("POR "\n"SEPARADO") no aparece contigua en el texto del archivo.
from vision_agent import _MEAL_VISION_PROMPT as _PROMPT


def test_prompt_disambiguates_mangu_from_rice():
    assert "MANGU" in _PROMPT and "pure COMPACTO" in _PROMPT, \
        "la desambiguacion mangu (masa lisa) vs arroz (granos sueltos) desaparecio"
    assert "granos sueltos" in _PROMPT
    assert "cebolla roja encurtida" in _PROMPT, "la senal visual mas distintiva del mangu"
    assert "tres golpes" in _PROMPT, "el desayuno criollo canonico debe estar nombrado"


def test_prompt_requires_per_component_sum():
    assert "POR SEPARADO" in _PROMPT and "SUMA" in _PROMPT, \
        "sin suma por componente gemma subestima el total a ojo (450 vs ~750 kcal vivo)"


def test_prompt_v3_full_inventory_no_omissions():
    """[v3] Segundo scan vivo del MISMO plato: gemma ya no confundio el mangu
    con arroz pero lo OMITIO por completo (nombre + macros). El prompt exige
    inventario completo y amarra la suma a ese inventario."""
    assert "INVENTARIO" in _PROMPT, "el inventario de componentes desaparecio"
    assert "NUNCA lo omitas" in _PROMPT, \
        "la regla anti-omision del mangu (masa + cebollita ⇒ lleva mangu) desaparecio"
    assert "TODOS los componentes" in _PROMPT
    assert "guarniciones" in _PROMPT, "cebolla encurtida/aguacate se perdian del listado"


def test_prompt_v3_named_dishes_and_base_first():
    """[v3] Nombre: plato con nombre propio (los tres golpes, la bandera) o
    componentes principales EMPEZANDO por la base de carbohidrato — el cap de
    palabras recortaba justo el mangu del nombre."""
    assert "nombre propio" in _PROMPT
    assert "Los tres golpes" in _PROMPT and "La bandera" in _PROMPT
    assert "EMPEZANDO por la base" in _PROMPT
    assert "max 8 palabras" in _PROMPT, "el cap subio de 6 a 8 para que quepa la base"


def test_prompt_ascii_only():
    """El prompt viaja crudo a Ollama — se mantiene sin acentos ni em-dashes
    (convencion del prompt del escaner de Nevera; evita sorpresas de encoding
    en el transporte)."""
    assert _PROMPT.isascii(), \
        f"chars no-ASCII en el prompt: {[c for c in _PROMPT if not c.isascii()][:5]}"


def test_modal_review_fits_without_scroll():
    """[P1-MEAL-SCAN-POLISH] Los 3 recortes verticales del modal de revision."""
    with open(os.path.join(_ROOT, "frontend", "src", "components", "dashboard",
                           "ScanMealModal.module.css"), encoding="utf-8") as f:
        css = f.read()
    assert "previewCompact" in css, "la preview banner de revision desaparecio"
    assert "fieldRow" in css, "Tipo + Porcion deben compartir fila"
    i = css.find(".actions {")
    assert "flex-direction: row" in css[i:i + 300], \
        "las acciones vuelven a columna → el boton Registrar cae bajo el fold"

    with open(os.path.join(_ROOT, "frontend", "src", "components", "dashboard",
                           "ScanMealModal.jsx"), encoding="utf-8") as f:
        jsx = f.read()
    assert "styles.previewCompact" in jsx and "styles.fieldRow" in jsx
