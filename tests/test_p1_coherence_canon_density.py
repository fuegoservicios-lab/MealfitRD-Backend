"""[P1-COHERENCE-CANON-DENSITY · 2026-07-26] El guard se inventaba fantasmas por un lookup fallido.

## El caso

Plan vivo `fbe53a5b`, el único con `base_qty` en 48/48 items de la lista:

    Plátano   esperado=0.0    lista=1400.0   unit_mismatch  ->  unknown
    Yogur     esperado=0.0    lista=907.2    unit_mismatch  ->  unknown

Las recetas SÍ pedían plátano ("2 plátanos maduro medianos", "½ plátano verde") y yogurt
("⅔ taza de yogurt griego"). No era un fantasma del pipeline: era un fantasma que fabricaba
el propio guard.

## Por qué

`_normalize_food_dict_to_grams` corre DESPUÉS de `_canonicalize_food_dict_for_coherence`, así
que recibe **etiquetas canónicas** — y una etiqueta canónica muchas veces no es una fila del
catálogo:

    Plátano verde  ─┐
    Plátano maduro ─┴─>  "Plátano"   (no existe en master_ingredients)
    Yogurt, Yogurt griego entero, … ->  "Yogur"   (tampoco)

El lookup era `master_map[nombre.lower()]` a secas → miss → `convert_amount(2.5, 'unidad',
'g', {})` sin densidad → `None` (modo strict, correcto) → la fila se quedaba en `unidad`
mientras la lista hablaba en `g` → sin unidad común, `expected_qty = 0.0`.

El síntoma estaba a la vista en los logs y no lo miré: `convert_amount(0.44 unidad→g,
item='<unknown>')`. **El `<unknown>` era el bug**, no un dato faltante del catálogo — ambas
filas de plátano traen `density_g_per_unit=280` y Yogurt trae `density_g_per_cup=245`.

## El efecto, medido

19 planes vivos, guard REAL en modo block. Solo `fbe53a5b` tiene la lista con `base_qty`
completo (los otros 18 se persistieron antes de P1-COHERENCE-BASE-QTY y no se reescribe
historia), así que es el único donde el fix puede notarse:

    antes:  10 divergencias  (Plátano y Yogur entre ellas, ambas fantasma)
    despues: 8 divergencias  (las dos fantasma desaparecen, ya emparejan)

Y el lado esperado gana precisión donde ya emparejaba: Pescado esp 349.7 -> 574.7 g.

tooltip-anchor: P1-COHERENCE-CANON-DENSITY
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] Este módulo necesita el catálogo/la base de datos o el
# .env local (pasa en el checkout del dueño; en el CI sin NEON_DATABASE_URL se salta con motivo).
pytestmark = pytest.mark.needs_local_data


# ───────────── 1. el efecto: la etiqueta canónica encuentra su densidad ─────────────

def test_platano_canonico_convierte_a_gramos():
    """"Plátano" no es fila del catálogo — es el canónico de Plátano verde/maduro, que traen
    density_g_per_unit=280. Antes del fix esto salía {'unidad': 2.5} y no emparejaba con nada."""
    out = sc._normalize_food_dict_to_grams({"Plátano": {"unidad": 2.5}})
    g = out.get("Plátano", {}).get("g")
    assert g, f"'Plátano' debe resolver densidad via su grupo canónico: {out}"
    assert g == pytest.approx(2.5 * 280, rel=0.02)


def test_yogur_canonico_convierte_a_gramos():
    """"Yogur" es el canónico de Yogurt / Yogurt griego … (density_g_per_cup=245)."""
    out = sc._normalize_food_dict_to_grams({"Yogur": {"taza": 2.0}})
    assert out.get("Yogur", {}).get("g"), out


def test_el_nombre_exacto_sigue_funcionando():
    """El fix es ADITIVO: indexar por alias y canónico no debe estorbar al nombre exacto."""
    out = sc._normalize_food_dict_to_grams({"Plátano verde": {"unidad": 1.0}})
    assert out.get("Plátano verde", {}).get("g") == pytest.approx(280, rel=0.02)


def test_por_alias_tambien():
    """Indexar los alias cierra la otra mitad de los `item='<unknown>'` del log."""
    out = sc._normalize_food_dict_to_grams({"huevos": {"unidad": 2.0}})
    assert out.get("huevos", {}).get("g") == pytest.approx(100, rel=0.05), out


# ───────────── 2. no se degradó lo que ya era correcto ─────────────

def test_gramos_pasan_intactos():
    assert sc._normalize_food_dict_to_grams({"Pollo": {"g": 250.0}}) == {"Pollo": {"g": 250.0}}


def test_lo_genuinamente_inconvertible_conserva_su_unidad():
    """La honestidad del modo strict se mantiene: sin densidad NO se inventa un número.
    "Sal al gusto" entra como pizca y debe SEGUIR sin pareja — es el comportamiento correcto,
    no un fallo (ver la clase de divergencias de condimento en el docstring)."""
    out = sc._normalize_food_dict_to_grams({"Sal": {"pizca": 3.0}})
    assert out.get("Sal") == {"pizca": 3.0}


@pytest.mark.parametrize("entrada", [{}, None, "no-dict", {"X": None}, {"X": {"g": "abc"}},
                                     {"X": {"unidad": -1}}])
def test_fail_safe(entrada):
    assert isinstance(sc._normalize_food_dict_to_grams(entrada), dict)


def test_suma_unidades_mixtas_del_mismo_alimento():
    out = sc._normalize_food_dict_to_grams({"Plátano": {"unidad": 1.0, "g": 100.0}})
    g = out.get("Plátano", {}).get("g")
    assert g and g > 100.0, f"debe sumar los convertidos a los que ya venían en g: {out}"


# ───────────── 3. ancla de la clase ─────────────

def test_el_lookup_no_es_solo_por_nombre_exacto():
    """Ancla: si alguien vuelve al `{name.lower(): m}` de una línea, los canónicos que NO son
    fila del catálogo pierden su densidad y los fantasmas vuelven — sin que ningún test de
    unidades falle, porque el síntoma solo aparece con nombres canónicos colapsados."""
    import inspect
    src = inspect.getsource(sc._normalize_food_dict_to_grams)
    assert "_canonicalize_for_coherence" in src, (
        "el master_map debe indexarse también por forma canónica"
    )
    assert 'm.get("aliases")' in src, "y por alias"


def test_la_precedencia_es_nombre_exacto_primero():
    """`setdefault` en los tres niveles: un alias de otra fila no debe poder pisar un nombre."""
    import inspect
    src = inspect.getsource(sc._normalize_food_dict_to_grams)
    i = src.index("master_map = {}")
    bloque = src[i:src.index("out = {}", i)]
    assert "master_map[" not in bloque.replace("master_map[nm.lower()]", ""), \
        "solo setdefault: una asignación directa permitiría que un alias pise un nombre"
    assert bloque.count("setdefault") >= 3
