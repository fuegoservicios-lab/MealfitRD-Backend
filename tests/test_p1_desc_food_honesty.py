"""[P1-DESC-FOOD-HONESTY · 2026-07-27] La descripción vendía alimentos que el plato no lleva.

## Lo que veía el owner

    "…y la dulzura natural del mango. ¡Rápido y perfecto para un lunes!"   (no hay mango)
    "…un guiso suave de mero con vegetales…"                               (el plato lleva pollo)

Medido sobre 164 comidas de 14 planes vivos: **20 (12%)** mencionan un alimento ausente de los
ingredientes. El NOMBRE está sano (0/164: PHANTOM-PROTEIN-NAMEFIX funciona); el hueco era que el
autofix solo reescribe `desc` cuando hay huérfanos en los PASOS.

## Por qué la escalera es corta

La 1ª versión (swap laxo + retirada agresiva) se midió sobre los planes vivos ANTES de desplegar y
**mutiló 6 de 22 salidas**: "Salmón horneado"→"queso horneado", "arroz integral"→"yuca integral",
"coronado jí" (comió media palabra), "lechosa jugoso" (género), "lechosa asadas" (número), y listas
españolas descabezadas ("ensalada de manzana, pera y…"→", pera y…"). Cada una de esas es hoy un
caso de este archivo. La versión final: swap SOLO dentro del subgrupo, regla de token siguiente,
re-concordancia de género Y número, retirada solo a cierre de cláusula SIN contar la coma.

⚠️ El primer conteo dio 43 y era el doble de lo real: "sabrosa y SIN arroz" contaba como mención.
⚠️ Y en el camino, el heredoc volvió a convertir `\\b` en 0x08 (8ª vez) — más un `\\1` en 0x01.

tooltip-anchor: P1-DESC-FOOD-HONESTY
"""
from __future__ import annotations

import copy

import pytest

import graph_orchestrator as g


def _run(desc, ingredients):
    meal = {"name": "Plato", "desc": desc, "ingredients": list(ingredients)}
    n = g._desc_food_honesty_pass([{"day": 1, "meals": [meal]}])
    return n, meal["desc"]


# ───────────── 1. los casos reales reparados ─────────────

def test_swap_mero_por_pollo():
    n, d = _run("Unas papas al horno rellenas de un guiso suave de mero con vegetales.",
                ["2 papas grandes", "40 g de filete de pechuga de pollo"])
    assert n == 1 and "guiso suave de pollo" in d and "mero" not in d


def test_retirada_del_mango_a_cierre_de_clausula():
    n, d = _run("Huevos revueltos esponjosos con toque criollo y la dulzura natural del mango. "
                "¡Rápido y perfecto para un lunes!",
                ["1 huevo", "½ cebolla"])
    assert n == 1 and "mango" not in d
    assert "¡Rápido y perfecto para un lunes!" in d, "la prosa posterior no se puede perder"


def test_swap_con_genero_y_articulo():
    n, d = _run("Un bowl refrescante de mango dulce y pistachos.", ["1 lechosa mediana (198g)"])
    assert n == 1 and "de lechosa dulce" in d


def test_swap_reconcuerda_el_adjetivo():
    """'mango jugoso' → 'lechosa jugosA' — la 1ª versión dejó 'lechosa jugoso'."""
    n, d = _run("Merienda práctica: mango jugoso y almendras crujientes.",
                ["100 g de lechosa en cubos"])
    assert n == 1 and "lechosa jugosa" in d and "jugoso" not in d


def test_swap_respeta_el_numero():
    """'ciruelas asadas' → 'lechosaS asadas' — el singular dejaba 'lechosa asadas que aportan'."""
    n, d = _run("Bollitos servidos con ciruelas asadas al horno que aportan un toque dulce.",
                ["1 lechosa mediana", "42 g de harina de trigo"])
    assert n == 1 and "lechosas asadas" in d


# ───────────── 2. las mutilaciones de la 1ª versión, hoy imposibles ─────────────

def test_salmon_no_se_convierte_en_queso_horneado():
    """Subgrupos: carne↔carne, lácteo↔lácteo. Sin esto: 'QUESO horneado con glaseado de miel'."""
    n, d = _run("Salmón horneado con un glaseado de limón y miel.",
                ["40 g de queso mozzarella", "1 batata"])
    assert "queso horneado" not in d.lower()
    assert d.startswith("Salmón"), "sin arreglo seguro la desc queda INTACTA (mentir < mutilar)"


def test_arroz_integral_no_se_vuelve_yuca_integral():
    """Regla de token siguiente: 'integral' no está en la whitelist de adjetivos swapeables."""
    n, d = _run("Servido sobre arroz integral esponjoso.", ["1 yuca mediana"])
    assert "yuca integral" not in d.lower()


def test_no_corta_a_mitad_de_palabra():
    """'coronado con atún en agua escurrido y ají' — la 1ª versión dejó 'coronado jí'."""
    n, d = _run("Puré de batata coronado con atún en agua escurrido y ají, acompañado de vainitas.",
                ["1 batata", "½ ají cubanela", "100 g de vainitas"])
    assert "jí," not in d.replace("ají", "")
    assert ("atún" in d) or ("ají" in d), d


def test_la_coma_no_cuenta_como_cierre():
    """Con la coma como cierre, la mención pegada a una coma dispara la retirada y arranca medio
    sintagma por la izquierda (caso real del dry-run: se llevó 'mezcla cremosa de queso cheddar').
    La mutación que reactiva la coma DEBE caer aquí."""
    n, d = _run("Papas horneadas, rellenas con una mezcla cremosa de queso cheddar y claras "
                "de huevo, gratinadas al horno.",
                ["2 papas grandes", "30 g de queso cheddar"])
    assert "mezcla cremosa" in d, "la coma no es cierre de cláusula: no se retira nada"


def test_lista_espanola_intacta():
    """'ensalada de manzana, pera y cebolla' — sin conector antes de la mención no hay retirada."""
    n, d = _run("Acompañado de una ensalada fresca de manzana, pera y cebolla morada.",
                ["1 pera", "1 cebolla morada"])
    assert "ensalada" in d, "no se puede arrancar la cabeza del sintagma"


def test_mencion_negada_es_honesta():
    n, d = _run("Una cena sustanciosa, sabrosa y sin arroz, perfecta para cerrar el martes.",
                ["2 papas grandes"])
    assert n == 0 and "sin arroz" in d


def test_mencion_presente_no_se_toca():
    n, d = _run("Un batido cremoso de lechosa y chinola.", ["1 lechosa mediana"])
    assert n == 0 and "lechosa" in d


# ───────────── 3. contratos ─────────────

def test_idempotente():
    meal = {"name": "P", "desc": "Un bowl de mango dulce.", "ingredients": ["1 lechosa"]}
    g._desc_food_honesty_pass([{"day": 1, "meals": [meal]}])
    una = meal["desc"]
    g._desc_food_honesty_pass([{"day": 1, "meals": [meal]}])
    assert meal["desc"] == una


@pytest.mark.parametrize("dias", [None, [], [{}], [{"meals": None}], [{"meals": [{}]}],
                                  [{"meals": [{"desc": 123}]}]])
def test_fail_safe(dias):
    g._desc_food_honesty_pass(copy.deepcopy(dias))


def test_knob_de_rollback():
    import inspect
    src = inspect.getsource(g)
    assert '_env_bool("MEALFIT_DESC_FOOD_HONESTY", True)' in src


def test_conectado_al_finalize():
    """Ancla 'código presente, efecto ausente': el pase debe correr en `finalize_plan_data_coherence`
    junto al resto del pulido — es el choke point que cubre TODOS los caminos de generación."""
    import inspect
    src = inspect.getsource(g.finalize_plan_data_coherence)
    assert "_desc_food_honesty_pass(" in src


def test_sin_caracteres_de_control_en_el_modulo():
    """8ª mordida del heredoc: `\\b`→0x08 y `\\1`→0x01 llegaron al archivo y el regex moría en
    silencio (el except por comida se lo tragaba). Este ancla hace imposible repetirlo."""
    import pathlib
    raw = pathlib.Path(g.__file__).with_suffix(".py").read_text(encoding="utf-8")
    malos = sorted({ord(c) for c in raw if ord(c) < 32 and c not in "\n\r\t"})
    assert not malos, f"caracteres de control en graph_orchestrator.py: {malos}"
