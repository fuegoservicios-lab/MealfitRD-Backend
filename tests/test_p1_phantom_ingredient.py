"""[P1-PHANTOM-INGREDIENT · 2026-07-24] Dirección INVERSA del validador de coherencia.

Hallazgo de la auditoría de 28 agentes al plan vivo `732588f8` (D1 Desayuno):

    name        "Avena Proteica con Guanábana y Maní"
    desc        "…cubierta con pulpa fresca de guanábana…"
    recipe[0]   "…30 g de pulpa de guanábana (despepitada y cortada en trozos)…"
    recipe[1]   "…extender los trozos de guanábana por encima…"
    ingredients ['¼ taza de avena', '395 ml de leche descremada', '30 g de maní mixtas',
                 '1 cda de miel', '½ cdta de canela en polvo']     ← CERO guanábana
    listas      4 listas de compras (7/15/30 días + híbrida)       ← CERO guanábana

Cuatro menciones, cantidad exacta declarada en un paso, y el alimento no existe en ninguna parte
comprable. La receta es físicamente incocinable e incomprable. La guanábana SÍ está en
`master_ingredients` con densidades buenas — la línea simplemente nunca se escribió.

`_recipe_coherence_errors` solo validaba "listado pero no usado" (reverse) y, desde
P1-AUTO-PATCH-FORWARD, "la receta nombra una PROTEÍNA sin ingrediente equivalente". Una fruta
declarada con gramos caía por el hueco entre ambas.

La regla nueva es sintáctica: **toda cantidad declarada en los pasos debe existir como línea de
ingrediente**. Se eligió sobre "escanear sustantivos del nombre" porque un paso que dice "30 g de
X" es una declaración, no una mención casual — 'Ensalada Verde' no flagea 'verde'.
"""
import pytest

import graph_orchestrator as go


# Índice de catálogo inyectado: el test no depende de la DB (el pool no está abierto en pytest).
FAKE_INDEX = {
    "guanabana": "Guanábana", "avena": "Avena", "mani": "Maní", "miel": "Miel",
    "leche descremada": "Leche descremada", "leche": "Leche", "canela": "Canela",
    "fresa": "Fresas", "fresas": "Fresas", "pina": "Piña", "aguacate": "Aguacate",
}


@pytest.fixture(autouse=True)
def _inject_catalog(monkeypatch):
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", dict(FAKE_INDEX), raising=False)
    yield
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", None, raising=False)


def _guanabana_meal():
    """Copia literal del meal vivo (texto real, no sintético)."""
    return {
        "name": "Avena Proteica con Guanábana y Maní",
        "desc": "Cremosa avena cocida con leche, endulzada con un toque de miel y cubierta con "
                "pulpa fresca de guanábana y crujientes nueces picadas.",
        "ingredients": ["¼ taza de avena", "395 ml de leche descremada", "30 g de maní mixtas",
                        "1 cda de miel", "½ cdta de canela en polvo"],
        "ingredients_raw": ["¼ taza de avena", "395 ml de leche descremada", "30 g de maní mixtas",
                            "1 cda de miel", "½ cdta de canela en polvo"],
        "recipe": [
            "Mise en place: medir ¼ taza de avena (65 g), 395 ml de leche descremada, 30 g de "
            "pulpa de guanábana (despepitada y cortada en trozos), 30 g de maní mixtas (picadas "
            "gruesas), 1 cda de miel y canela en polvo.",
            "Montaje: servir la avena en un bowl, extender los trozos de guanábana por encima, "
            "espolvorear con maní picado y un toque de canela en polvo. Disfrutar tibia.",
        ],
    }


# ───────────── 1. el caso vivo ─────────────

def test_guanabana_fantasma_se_reinserta_con_su_cantidad():
    meal = _guanabana_meal()
    fixed = go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}])

    assert len(fixed) == 1, f"exactamente un fantasma en este meal, no {fixed}"
    assert fixed[0]["food"] == "Guanábana"
    assert any("guanábana" in str(i).lower() for i in meal["ingredients"]), (
        "la guanábana tiene que quedar en `ingredients` o no se puede comprar ni cocinar"
    )
    # La cantidad sale del propio texto — cero invención.
    assert "30 g" in fixed[0]["line"], f"la cantidad la declara el paso: {fixed[0]['line']!r}"


def test_nombre_canonico_del_catalogo_no_la_frase_cruda():
    """La línea se escribe con el nombre del catálogo: es el vocabulario que la lista entiende.
    Si se escribiera 'pulpa de guanábana', el shopping calculator no resolvería la fila."""
    meal = _guanabana_meal()
    go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}])
    assert meal["ingredients"][-1] == "30 g de Guanábana"


def test_ingredients_raw_queda_alineado():
    """`ingredients_raw` viaja índice-a-índice con `ingredients` (P1-RAW-MISALIGN-TRACE):
    apendear a una sola lista desfasa el display una posición."""
    meal = _guanabana_meal()
    go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}])
    assert len(meal["ingredients"]) == len(meal["ingredients_raw"])
    assert meal["ingredients"][-1] == meal["ingredients_raw"][-1]


def test_idempotente():
    meal = _guanabana_meal()
    days = [{"day": 1, "meals": [meal]}]
    assert len(go._repair_declared_but_unlisted_ingredients(days)) == 1
    assert go._repair_declared_but_unlisted_ingredients(days) == [], (
        "segunda pasada: el alimento ya está listado → nada que hacer"
    )
    assert len(meal["ingredients"]) == 6, "no puede duplicar la línea"


# ───────────── 2. lo que NO debe tocar ─────────────

def test_ingrediente_ya_listado_no_se_duplica():
    meal = {"name": "Avena con Maní", "ingredients": ["30 g de maní mixtas"],
            "recipe": ["Mise en place: 30 g de maní mixtas picadas."]}
    assert go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients"]) == 1


@pytest.mark.parametrize("staple", ["agua", "sal", "aceite", "hielo", "azúcar"])
def test_staples_nunca_se_insertan(staple):
    """El motor omite estos a propósito (condimentos consolidados / no comprables).
    Insertarlos metería agua en la lista de compras."""
    meal = {"name": "Sopa", "ingredients": ["100 g de pollo"],
            "recipe": [f"Hervir 500 ml de {staple} en una olla."]}
    assert go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients"]) == 1


def test_el_skip_se_evalua_tambien_sobre_el_nucleo(monkeypatch):
    """'aceite' estaba excluido pero 'aceite de oliva' esquivaba la lista por ser otra cadena.
    Insertar grasa automáticamente es lo que más mueve la banda (2 cdas ≈ 240 kcal) y el motor
    consolida condimentos por su cuenta."""
    monkeypatch.setitem(go._PHANTOM_CATALOG_INDEX_CACHE, "aceite de oliva", "Aceite de oliva")
    monkeypatch.setitem(go._PHANTOM_CATALOG_INDEX_CACHE, "aceite", "Aceite de oliva")
    meal = {"name": "Salteado", "ingredients": ["120 g de pollo"],
            "recipe": ["Calienta 2 cdas de aceite de oliva en un sartén."]}
    assert go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}]) == []
    assert go._phantom_resolve_food("aceite de oliva") is None


def test_alimento_fuera_del_catalogo_se_deja_pasar():
    """Sin fila en `master_ingredients` no hay densidad ni precio: insertarlo ensuciaría la lista.
    Fail-open a propósito — el fallo caro es comprar lo que la receta no pide."""
    meal = {"name": "Plato exótico", "ingredients": ["100 g de pollo"],
            "recipe": ["Añadir 40 g de rambután pelado."]}
    assert go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}]) == []


def test_notas_deterministas_exentas():
    """Las notas ⚠/💡 y las de sustitución citan el alimento ORIGINAL a propósito
    (P2-AUTOFIX-NOTE-EXEMPT). Nunca declaran ingredientes del plato."""
    meal = {"name": "Ceviche de Pescado", "ingredients": ["150 g de pescado"],
            "recipe": ["Mise en place: cortar el pescado.",
                       "💡 Nota del Nutricionista: se reemplazó 100 g de aguacate por pescado "
                       "para ajustar las grasas."]}
    assert go._repair_declared_but_unlisted_ingredients([{"day": 1, "meals": [meal]}]) == [], (
        "el aguacate de la nota es el alimento que se QUITÓ — reinsertarlo revierte la sustitución"
    )


def test_resolucion_no_usa_modificadores():
    """'maní mixtas' resuelve por su NÚCLEO ('maní'), nunca por el modificador ('mixtas').
    Probar todos los spans invitaría a comprar el alimento equivocado."""
    assert go._phantom_resolve_food("maní mixtas") == ("mani", "Maní")
    assert go._phantom_resolve_food("pulpa de guanábana") == ("guanabana", "Guanábana")
    assert go._phantom_resolve_food("mezcla anterior") is None
    assert go._phantom_resolve_food("agua") is None


# ───────────── 3. orden de ejecución (lo que hace que el fix sirva) ─────────────

def test_corre_antes_de_construir_la_lista_de_compras():
    """Si corriera después, la línea insertada no llegaría a la lista y el plan seguiría
    siendo incomprable — que es exactamente el defecto que cierra."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_repair = src.index("_repair_declared_but_unlisted_ingredients(result.get(\"days\")")
    i_list = src.index("# Calcular shopping lists")
    assert i_repair < i_list, "el repair DEBE preceder a la construcción de la lista de compras"


def test_knob_de_rollback():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'PHANTOM_INGREDIENT_REPAIR = _env_bool("MEALFIT_PHANTOM_INGREDIENT_REPAIR", True)' in src
    assert "if PHANTOM_INGREDIENT_REPAIR:" in src
