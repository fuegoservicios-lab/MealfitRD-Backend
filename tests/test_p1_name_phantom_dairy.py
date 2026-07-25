"""[P1-NAME-PHANTOM-DAIRY · 2026-07-25] El nombre promete un queso que el plato no lleva.

Reporte del owner sobre el plan vivo `2dbc836c` — dos platos con el chip *"El nombre puede no
reflejar la proteína real"*:

    "Tostadas … con Queso Mozzarella y Níspero y Huevo"  → pan, níspero, miel, huevo, sal
    "Arepitas … Rellenas de Queso Mozzarella"            → harina, agua, lechuga, tomate, aguacate

El segundo se queda **sin ninguna proteína** en una cena de 611 kcal.

El chip decía la verdad. `P1-PHANTOM-PROTEIN-NAMEFIX` renombra cuando hay otra carne real con la
que sustituir; cuando no la hay marca `_name_honesty_degraded` en vez de mentir (*"no se 'asa' un
yogur"*). Decisión correcta, pero deja el plato pobre: renombrar lo vuelve honesto y **sigue sin
proteína**.

`P1-PHANTOM-INGREDIENT` no lo cubre porque sólo actúa cuando los PASOS declaran una cantidad.
Aquí el queso vive únicamente en el nombre, así que hay que elegir una porción — y por eso el
alcance es deliberadamente estrecho:

  · **Sólo lácteos.** Frutas y vegetales fantasma en el nombre siguen siendo sólo aviso: una
    guarnición ausente no arruina la comida y no justifica inventar gramos.
  · Porción **fija y conservadora** (30 g ≈ una lonja), no derivada de un hueco de macros.
  · El alimento debe resolver en `master_ingredients` (sin fila no hay densidad ni precio).
  · Los caps corren después como última palabra (P1-CAPS-LAST-WORD) → no puede disparar la grasa.
"""
import pytest

import graph_orchestrator as go


CATALOGO = {
    "queso mozzarella": "Queso mozzarella", "mozzarella": "Queso mozzarella",
    "queso blanco": "Queso blanco", "queso": "Queso blanco",
    "queso cottage": "Queso cottage", "cottage": "Queso cottage",
    "huevo": "Huevo", "lechuga": "Lechuga",
}


@pytest.fixture(autouse=True)
def _catalogo(monkeypatch):
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", dict(CATALOGO), raising=False)
    yield
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", None, raising=False)


def _arepitas():
    return {"name": "Arepitas de Harina de Negrito Rellenas de Queso Mozzarella con Ensalada Fresca",
            "ingredients": ["70 g de harina de Negrito", "2 tazas de lechuga", "½ tomate", "Sal al gusto"],
            "ingredients_raw": ["70 g de harina de Negrito", "2 tazas de lechuga", "Sal al gusto", "0.5 tomate"],
            "_name_honesty_degraded": True}


# ───────────── 1. el caso reportado ─────────────

def test_anade_el_queso_que_el_nombre_promete():
    meal = _arepitas()
    out = go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}])
    assert len(out) == 1 and out[0]["food"] == "Queso mozzarella"
    assert any("mozzarella" in str(i).lower() for i in meal["ingredients"])


def test_llega_tambien_a_ingredients_raw():
    """La lista de compras lee raw primero (P1-PHANTOM-RAW-PARITY): sin esto sería visible
    pero no comprable."""
    meal = _arepitas()
    go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}])
    assert any("mozzarella" in str(i).lower() for i in meal["ingredients_raw"])


def test_usa_el_queso_ESPECIFICO_del_nombre():
    """Si el nombre dice mozzarella, insertar un 'queso' genérico sería otra media verdad."""
    meal = _arepitas()
    go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}])
    linea = next(i for i in meal["ingredients"] if "queso" in str(i).lower())
    assert "mozzarella" in linea.lower(), linea


def test_retira_el_chip_de_honestidad():
    """El nombre pasó a ser verdad: mantener el aviso sería mentir en la otra dirección."""
    meal = _arepitas()
    go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}])
    assert "_name_honesty_degraded" not in meal


def test_porcion_conservadora():
    meal = _arepitas()
    go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}])
    linea = next(i for i in meal["ingredients"] if "mozzarella" in str(i).lower())
    assert linea.startswith(f"{go.NAME_PHANTOM_DAIRY_G} g"), linea
    assert 10 <= go.NAME_PHANTOM_DAIRY_G <= 80


def test_idempotente():
    meal = _arepitas()
    days = [{"day": 1, "meals": [meal]}]
    assert go._repair_name_phantom_dairy(days)
    assert go._repair_name_phantom_dairy(days) == []
    assert sum(1 for i in meal["ingredients"] if "mozzarella" in str(i).lower()) == 1


# ───────────── 2. lo que NO debe tocar ─────────────

def test_no_duplica_si_el_queso_YA_esta():
    meal = {"name": "Tostadas con Queso Mozzarella y Huevo",
            "ingredients": ["2 rebanadas de pan", "45 g de queso mozzarella", "1 huevo"],
            "ingredients_raw": ["2 rebanadas de pan", "45 g de queso mozzarella", "1 huevo"]}
    assert go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients"]) == 3


def test_nombre_sin_lacteo_no_se_toca():
    meal = {"name": "Ensalada Fresca de Lechuga", "ingredients": ["2 tazas de lechuga"],
            "ingredients_raw": ["2 tazas de lechuga"]}
    assert go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}]) == []


def test_frutas_y_vegetales_fantasma_siguen_siendo_SOLO_aviso():
    """Alcance deliberado: una guarnición ausente no justifica inventar gramos. Si algún día se
    amplía, que sea con evidencia como la que motivó el caso del queso."""
    meal = {"name": "Bowl Tropical con Piña y Mango", "ingredients": ["150 g de pollo"],
            "ingredients_raw": ["150 g de pollo"]}
    assert go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}]) == []


def test_sin_catalogo_no_inventa(monkeypatch):
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", {}, raising=False)
    meal = _arepitas()
    assert go._repair_name_phantom_dairy([{"day": 1, "meals": [meal]}]) == []


# ───────────── 3. cableado ─────────────

def test_corre_antes_de_la_lista_y_con_los_caps_detras():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_npd = src.index('_repair_name_phantom_dairy(result.get("days")')
    i_list = src.index("# Calcular shopping lists")
    assert i_npd < i_list, "sin esto el queso añadido no se compraría"
    assert "P1-CAPS-LAST-WORD" in src, "los caps quedan como última palabra tras insertar grasa"


def test_knobs():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'NAME_PHANTOM_DAIRY_REPAIR = _env_bool("MEALFIT_NAME_PHANTOM_DAIRY_REPAIR", True)' in src
    assert '_env_int("MEALFIT_NAME_PHANTOM_DAIRY_G", 30' in src
