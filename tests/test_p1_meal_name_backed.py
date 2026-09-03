"""[P1-MEAL-NAME-BACKED · 2026-08-10] El rótulo del plato no puede contradecir su
propio inventario.

EL CASO REAL (owner, scan del 2026-08-10 20:44, `corr=6c0a12e0` en los logs de
producción). El modelo describió bien lo que veía:

    'Arroz blanco y carne molida guisada con vegetales y salsa.
     (Estimación: Calorías: 560, Proteína: 25g, Carbohidratos: 65g, Grasas: 8g)'

...y las macros de la tarjeta (560/25/65/8) correspondían a ESE inventario. Pero
`meal_name` decía «Arroz blanco con lazaña». La lasaña no estaba en el plato, ni en
la descripción, ni en las macros. Lo único que mentía era el rótulo — que es
exactamente lo que el usuario ve en su diario y lo que queda registrado.

Tercera vez en el día que el modelo LEE bien y una capa posterior lo desmiente
(antes: el mapeo al catálogo renombró un pan a «Polvo de hornear», y el checklist
mostraba solo el resultado del mapeo). Aquí la capa que miente es un segundo campo
del mismo modelo, y hay con qué contrastarla: el prompt YA obliga a que
`description` sea el inventario completo (P1-MEAL-SCAN-DR-DISHES v3, escrito tras
dos fallos previos de este mismo campo).

Van tres versiones del prompt para `meal_name`. Un aviso no es un guard.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from constants import (
    DR_DISH_PROPER_NAMES,
    derive_meal_name_from_description,
    meal_name_backed_by_description,
)

_BACKEND = Path(__file__).resolve().parent.parent
_VA_SRC = (_BACKEND / "vision_agent.py").read_text(encoding="utf-8")

# El texto exacto que devolvió el modelo en el scan reportado.
DESC_REAL = "Arroz blanco y carne molida guisada con vegetales y salsa. (Estimación: Calorías: 560, Proteína: 25g, Carbohidratos: 65g, Grasas Saludables: 8g)"


def test_el_caso_reportado():
    assert meal_name_backed_by_description("Arroz blanco con lazaña", DESC_REAL) is False


def test_el_rotulo_correcto_del_mismo_plato_pasa():
    assert meal_name_backed_by_description("Arroz blanco con carne molida", DESC_REAL) is True


@pytest.mark.parametrize("nombre,desc,respaldado", [
    # --- respaldados ---
    ("Mangú con huevo, salami y queso",
     "Mangú de plátano verde, huevo frito, salami y queso frito.", True),
    ("Pollo guisado con arroz", "Arroz blanco, pollo guisado y ensalada verde.", True),
    ("Huevos revueltos", "Dos huevos revueltos con cebolla.", True),   # plural/singular
    ("Platano maduro", "Plátano maduro frito.", True),                 # sin acento
    # --- NO respaldados: el rótulo mete algo que no está en el inventario ---
    ("Arroz con pollo", "Arroz blanco, habichuelas rojas y ensalada.", False),
    ("Pizza de peperoni", "Pan tostado con queso y tomate.", False),
    ("Espaguetis con albóndigas", "Espaguetis con salsa de tomate.", False),
])
def test_respaldo(nombre, desc, respaldado):
    assert meal_name_backed_by_description(nombre, desc) is respaldado


@pytest.mark.parametrize("nombre,desc", [
    # Los clásicos con nombre propio NO describen sus componentes: es la razón de
    # que exista la exención, y el prompt los pide a propósito.
    ("La bandera dominicana", "Arroz blanco, habichuelas rojas guisadas y carne de res."),
    ("Los tres golpes", "Mangú, huevo frito, salami frito y queso frito."),
    ("Sancocho", "Caldo espeso con víveres, carne de res y pollo."),
    ("Mofongo", "Plátano verde majado con chicharrón."),
])
def test_nombres_propios_exentos(nombre, desc):
    assert meal_name_backed_by_description(nombre, desc) is True, (
        "un clásico con nombre propio no comparte palabras con sus componentes — "
        "si esta exención se cae, el guard renombra platos correctos"
    )


def test_no_castiga_lo_que_no_puede_juzgar():
    """Sin rótulo o sin inventario no hay contradicción que detectar. Un guard que
    dispara cuando le falta el dato con el que comparar es ruido, no vigilancia."""
    assert meal_name_backed_by_description("", "Arroz con pollo.") is True
    assert meal_name_backed_by_description("Arroz con pollo", "") is True
    assert meal_name_backed_by_description("", "") is True


@pytest.mark.parametrize("desc,esperado", [
    (DESC_REAL, "Arroz blanco y carne molida guisada con vegetales y salsa"),
    ("Mangú, huevo frito, salami y queso frito.", "Mangú, huevo frito, salami y queso frito"),
    ("", ""),
])
def test_rotulo_derivado(desc, esperado):
    """El reemplazo sale del inventario verificado y corta ANTES de la estimación
    de macros que el modelo concatena — si no, el nombre del plato acabaría siendo
    «... (Estimación: Calorías: 560 ...)»."""
    obtenido = derive_meal_name_from_description(desc, max_words=12)
    assert obtenido == esperado
    assert "Estimación" not in obtenido and "Calorías" not in obtenido


def test_el_derivado_del_caso_real_es_corto_y_honesto():
    d = derive_meal_name_from_description(DESC_REAL)
    assert d == "Arroz blanco y carne molida guisada con vegetales"
    assert "lazaña" not in d.lower()
    assert meal_name_backed_by_description(d, DESC_REAL) is True, (
        "el rótulo de reemplazo tiene que pasar su propio guard, o el siguiente "
        "scan entraría en bucle de sustitución"
    )


def test_la_lista_de_nombres_propios_es_corta_y_criolla():
    """Cada entrada es un PERMISO para que el rótulo no cuadre con el inventario.
    Si crece sin control, el guard deja de guardar."""
    assert len(DR_DISH_PROPER_NAMES) <= 40
    assert "lazaña" not in DR_DISH_PROPER_NAMES and "lasagna" not in DR_DISH_PROPER_NAMES
    assert {"bandera", "mangu", "mofongo", "sancocho"} <= DR_DISH_PROPER_NAMES


def test_el_agente_de_vision_aplica_el_backstop():
    i = _VA_SRC.find("meal_name = str(data.get(\"meal_name\")")
    assert i > 0, "el parseo de meal_name desapareció"
    cuerpo = _VA_SRC[i:i + 2200]
    assert "meal_name_backed_by_description(" in cuerpo, (
        "el rótulo debe contrastarse contra el inventario antes de devolverlo"
    )
    assert "derive_meal_name_from_description(" in cuerpo, (
        "un rótulo no respaldado se SUSTITUYE por uno derivado del inventario, "
        "no se deja pasar ni se vacía"
    )
    assert re.search(r"logger\.warning\(\s*\n?\s*\"\[P1-MEAL-NAME-BACKED\]", cuerpo), (
        "la sustitución tiene que dejar rastro: sin log no se puede medir cuántas "
        "veces el modelo inventa el rótulo"
    )


def test_el_prompt_tambien_lo_pide():
    """Prompt Y guard: el prompt evita el caso fácil, el guard cubre cuando no
    obedece — que es lo que pasó tres veces con este mismo campo."""
    i = _VA_SRC.find("_MEAL_VISION_PROMPT = (")
    prompt = _VA_SRC[i:i + 5000]
    assert "SOLO puede nombrar componentes" in prompt
