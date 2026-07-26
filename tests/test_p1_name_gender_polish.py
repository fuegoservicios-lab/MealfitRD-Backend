"""[P1-NAME-GENDER-POLISH · 2026-07-26] "Lechosa Fresco" — y por qué NO se arregla el resto.

Medido sobre 60 planes (196 nombres de plato):

    palabra repetida ....... 5  (2.6%)
    desacuerdo de genero ... 4  (2.0%)

Pero **mi propio detector tuvo falsos positivos en ambas categorias**, y eso decidio el
alcance del fix:

  · «Croquetas de Filete de pescado **blanco** … con Arroz **Blanco**» repite la palabra, pero
    son DOS alimentos distintos. Correcto.
  · «Queso **Crema Batido**» no es error de genero: el nucleo es *queso* (masculino) y "batido"
    concuerda bien. Mi regla miro *crema* y lo marco mal.

Descontando eso quedan **3 redundancias y 2 errores de genero reales en 196 nombres**.

## La decision

Se corrige SOLO el genero, y solo cuando el sustantivo femenino es el NUCLEO del sintagma
(inicio del nombre, o justo tras `y`/`con`/`de`/`e`). Es la regla que se puede afirmar.

**NO se construye un reescritor de redundancias.** Con un 40-50% de falsos positivos en mi
deteccion cuidadosa, un reescritor amplio corrompe nombres correctos para arreglar 5 cadenas
cosmeticas de 196. El coste esperado es mayor que el beneficio.

Display-only: no toca macros, ni ingredientes, ni la lista de compras.

tooltip-anchor: P1-NAME-GENDER-POLISH
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g


def _fix(name):
    """Resuelve la funcion en tiempo de LLAMADA, no de import.

    Enlazarla a nivel de modulo (`_fix = g._fix_name_gender_agreement`) hacia que este archivo
    diera `AttributeError` al colectarse junto a otros del mismo glob — algo ahi recarga
    `graph_orchestrator` y el binding queda apuntando al objeto viejo. Dos archivos vecinos
    (`test_p1_coherence_finalize`, `test_p1_finalize_tail_parity`) tienen el mismo error EN LA
    LINEA BASE, asi que es un patron preexistente del repo, no de este fix. Resolver por getattr
    en cada llamada lo esquiva.
    """
    import graph_orchestrator as _g
    return _g._fix_name_gender_agreement(name)


# ───────────── 1. los dos casos reales ─────────────

def test_lechosa_fresco_tras_conjuncion():
    assert _fix("Maní y Lechosa Fresco con Queso Crema Batido y Yogurt") == \
        "Maní y Lechosa Fresca con Queso Crema Batido y Yogurt"


def test_lechosa_fresco_al_inicio():
    assert _fix("Lechosa Fresco con Almendras y Tostada Integral") == \
        "Lechosa Fresca con Almendras y Tostada Integral"


# ───────────── 2. el falso positivo que definio el alcance ─────────────

def test_queso_crema_batido_NO_se_toca():
    """El nucleo es *queso* (masculino): "batido" ya concuerda. Si esto se 'corrige', el fix
    esta rompiendo nombres correctos — que es justo lo que se quiso evitar."""
    assert _fix("Tostadas de Pan Integral con Crema de Queso Crema Batido y Mango") is None


def test_la_redundancia_de_palabras_NO_se_toca():
    """Decision explicita: «pescado blanco … Arroz Blanco» son dos alimentos distintos."""
    assert _fix("Croquetas de Filete de pescado blanco al Horno con Arroz Blanco") is None
    assert _fix("Salteado de Berro Salteado con Queso Mozzarella y Limón") is None


# ───────────── 3. no toca lo que ya esta bien ─────────────

@pytest.mark.parametrize("nombre", [
    "Uvas Frescas con Mantequilla de Maní",
    "Pollo Asado con Vegetales",
    "Arroz Blanco con Habichuelas",
    "Revoltillo Dominicano con Casabe Crujiente",
    "Pavo Molido a la Plancha con Cítricos",
    "Filete de Res Salteado al Wok",
])
def test_nombres_correctos_intactos(nombre):
    assert _fix(nombre) is None, nombre


def test_masculino_con_adjetivo_masculino_intacto():
    """'Pescado Fresco' es correcto: pescado es masculino."""
    assert _fix("Pescado Fresco con Ensalada") is None


# ───────────── 4. mas femeninos, mas adjetivos ─────────────

@pytest.mark.parametrize("malo,bueno", [
    ("Manzana Asado con Canela", "Manzana Asada con Canela"),
    ("Auyama Horneado con Especias", "Auyama Horneada con Especias"),
    ("Pechuga Molido en Salsa", "Pechuga Molida en Salsa"),
    ("Bowl con Batata Asado", "Bowl con Batata Asada"),
])
def test_otros_pares(malo, bueno):
    assert _fix(malo) == bueno


def test_preserva_mayuscula_y_minuscula():
    assert _fix("Lechosa fresco con nueces") == "Lechosa fresca con nueces"


# ───────────── 5. bordes y fail-safe ─────────────

@pytest.mark.parametrize("v", [None, "", "   ", "Lechosa", 12345, ["Lechosa Fresco"]])
def test_entradas_raras(v):
    assert _fix(v) is None


def test_idempotente():
    once = _fix("Lechosa Fresco con Almendras")
    assert once is not None
    assert _fix(once) is None, "una segunda pasada no debe cambiar nada"


def test_es_display_only_y_esta_cableado():
    """El helper puede ser correcto y no llamarse nunca. Se ancla el callsite dentro del pase
    de display, y que NO toque ingredientes."""
    from pathlib import Path
    src = Path(g.__file__).resolve().read_text(encoding="utf-8")
    assert src.count("_fix_name_gender_agreement(meal.get(\"name\"))") == 1
    i = src.index("_fix_name_gender_agreement(meal.get(\"name\"))")
    bloque = src[i:i + 320]
    assert 'meal["name"] = _nm_fix' in bloque
    assert "ingredients" not in bloque.split("for _key")[0], \
        "el fix del nombre no debe tocar ingredientes"
