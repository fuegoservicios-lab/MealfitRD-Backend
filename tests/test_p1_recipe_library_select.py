# -*- coding: utf-8 -*-
"""[P1-RECIPE-LIBRARY-SELECT · 2026-09-08] El enganche receta↔plato, vivo y apagado.

Con `MEALFIT_RECIPE_LIBRARY_SELECT=False` (default) el sistema es byte-idéntico al de hoy: el LLM
sigue escribiendo la receta de cada plato. Con el knob encendido, un plato que sea una plantilla del
registry recibe la receta ya escrita y revisada, y el prompt deja de invitar a salirse del catálogo.

## Lo que se midió antes de escribir esto

La cadena política → blueprint → rebanada → CandidateSet, montada a mano sobre un formulario
sintético: propone **61 platos del registry para 7 días, y los 61 tienen receta**. La maquinaria
funciona; lo que faltaba eran tres interruptores, y dos de ellos son este módulo y esta frase.

El tercero —`MEALFIT_PLAN_POLICY_MODE`, que nace en `off`— es configuración del VPS y no se tocó:
94 de 95 planes vivos no llevan sello de política, o sea que la cadena nunca ha corrido en
producción.

## La frase que habría arruinado la medición

El prompt decía «elige uno de estos **o una variante equivalente**». Con esa puerta abierta, encender
la política y ver platos que no casan con ninguna plantilla llevaría a concluir que la arquitectura
falla — cuando lo que falla es una frase. Es un trade-off real (variedad contra determinismo), por
eso va atado al MISMO knob y no se cambia por libre.
"""
import os

import pytest

import recipe_library as rl


@pytest.fixture(autouse=True)
def _limpia_cache():
    rl._library.cache_clear()
    rl._name_index.cache_clear()
    yield
    rl._library.cache_clear()
    rl._name_index.cache_clear()


def test_apagado_por_defecto_y_no_devuelve_nada(monkeypatch):
    monkeypatch.delenv("MEALFIT_RECIPE_LIBRARY_SELECT", raising=False)
    assert rl.library_select_enabled() is False
    assert rl.recipe_for_dish_name("Mangú de plátano verde con cebolla encurtida y aguacate") is None


def test_encendido_sirve_la_receta_escrita(monkeypatch):
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    cov = rl.coverage("DO")
    if not cov["recetas"]:
        pytest.skip("la biblioteca no está en el árbol")
    # un plato que el CandidateSet propone de verdad (medido el 08-sep en el bloque del día 1)
    pasos = rl.recipe_for_dish_name("Tostada integral con mantequilla de maní y guineo")
    assert pasos and len(pasos) >= 3, "el plato está en el catálogo y debería traer su receta"
    assert all(isinstance(p, str) and p.strip() for p in pasos)


def test_la_coincidencia_es_EXACTA_no_parecida(monkeypatch):
    """Servir la receta de otro plato es peor que no servir ninguna.

    Es la misma doctrina que la lista de compras aplica desde `_remove_one_raw_line_by_food`:
    recortar la línea equivocada deja al usuario comprando otra cosa.
    """
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    if not rl.coverage("DO")["recetas"]:
        pytest.skip("la biblioteca no está en el árbol")
    assert rl.recipe_for_dish_name("Tostada integral con mantequilla de maní") is None
    assert rl.recipe_for_dish_name("Tostada con maní") is None
    assert rl.recipe_for_dish_name("") is None
    assert rl.recipe_for_dish_name(None) is None


def test_normaliza_acentos_y_mayusculas_pero_no_afloja_el_nombre(monkeypatch):
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    if not rl.coverage("DO")["recetas"]:
        pytest.skip("la biblioteca no está en el árbol")
    a = rl.recipe_for_dish_name("Tostada integral con mantequilla de maní y guineo")
    b = rl.recipe_for_dish_name("TOSTADA INTEGRAL CON MANTEQUILLA DE MANI Y GUINEO")
    assert a and a == b, "el mismo plato escrito con acentos o mayúsculas es el mismo plato"


def test_pais_sin_biblioteca_no_revienta(monkeypatch):
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    assert rl.recipe_for_dish_name("Cualquier cosa", country="XX") is None
    assert rl.coverage("XX") == {"con_receta": 0, "recetas": 0}


def test_el_prompt_cierra_la_puerta_SOLO_con_el_knob(monkeypatch):
    """Apagado: el texto es el de siempre. Encendido: prohíbe la variante.

    Si alguien quita el condicional y deja una sola de las dos redacciones, rompe uno de los dos
    objetivos —variedad o determinismo— sin que nadie lo decida.
    """
    import inspect

    import horizon

    src = inspect.getsource(horizon.registry_prompt_lines)
    assert "o una variante equivalente" in src, "desapareció la redacción de variedad (knob OFF)"
    assert "sin variantes" in src, "desapareció la redacción determinista (knob ON)"
    assert "library_select_enabled" in src, "la frase dejó de colgar del knob de la biblioteca"


def test_cobertura_del_registry_DO():
    cov = rl.coverage("DO")
    if not cov["recetas"]:
        pytest.skip("la biblioteca no está en el árbol")
    assert cov["con_receta"] == cov["recetas"] == 140, (
        f"cobertura inesperada: {cov}. Si el registry creció, regenerar con "
        "`scratchpad/genera_biblioteca_rd.py` antes de encender el knob.")
