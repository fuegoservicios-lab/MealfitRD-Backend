"""[P1-SLOT-MERIENDA-CRUDITES · 2026-07-26] El vegetal crudo como vehículo de un dip o una
crema es merienda de dieta AMERICANA, no dominicana.

## De dónde salió

El dueño señaló la merienda **"Apio Relleno con Mantequilla de Maní y Queso Cottage"**. La
parte del "y Queso Cottage" se cerró aparte (P1-CLOSER-SWEET-DAIRY-FIT: el cerrador pegaba
85 g de cottage a todo plato dulce corto de proteína, por una ventaja de densidad del 2%).
Lo que quedaba era el plato BASE — apio con mantequilla de maní — que no lo pone el cerrador
sino el generador de días.

En el corpus vivo (60 planes) aparece además **"Brócoli al Vapor con Dip de Yogurt y
Hierbas" 3 veces**. Misma clase.

## La causa: una fuga de generalización, no una violación

La regla 15c del prompt lista como categoría de merienda válida:

    • Fruta + mantequilla de maní/almendras (manzana con pb, guineo con pb)

El apio no es fruta. El modelo generalizó de *fruta* a *vegetal* y produjo el snack
americano. Ninguna regla del gate lo atrapaba: las de merienda son de TÉCNICA (locrio,
guisado, salteado) y de JUNK (pizza, hamburguesa), y ni "relleno" ni "al vapor" estaban.

Se cierra por los dos lados: el prompt ahora dice explícitamente "SOLO FRUTA, no vegetales",
y esta regla determinista es la red debajo — porque el prompt solo es un lever blando.

## Por qué tokens COMPUESTOS

`"apio"` suelto NO sirve: en RD el **apio criollo (arracacha) es un víver legítimo** de
sancocho. Lo que delata la merienda americana es la PREPARACIÓN — relleno, tallos, bastones,
dip. Mismo criterio que la regla de guisados del desayuno, que usa "pollo guisado" y no
"guisado" porque los huevos guisados sí son desayuno RD.

⚠️ Es un blocklist: cubre las clases con evidencia en el corpus, no toda combinación foránea
imaginable. Una alien nueva se colará hasta que alguien la añada. Eso es la naturaleza del
diseño existente, no una promesa de completitud.

tooltip-anchor: P1-SLOT-MERIENDA-CRUDITES
"""
from __future__ import annotations

import pytest

from constants import (
    SLOT_INAPPROPRIATE_FOODS,
    canonical_slot_key,
    slot_violations_for_meal_name,
)

_LABEL = "vegetal crudo como vehículo de dip/crema (merienda americana, no dominicana)"


def _viola(nombre, slot="merienda"):
    return [v["label"] for v in slot_violations_for_meal_name(nombre, slot)]


# ───────────── 1. los casos VIVOS ─────────────

def test_el_apio_relleno_del_dueno():
    """El plato exacto de la captura."""
    assert _LABEL in _viola("Apio Relleno con Mantequilla de Maní y Queso Cottage")


def test_el_brocoli_con_dip_del_corpus():
    """Apareció 3 veces en 60 planes."""
    assert _LABEL in _viola("Brócoli al Vapor con Dip de Yogurt y Hierbas")


@pytest.mark.parametrize("nombre", [
    "Bastones de Zanahoria con Crema de Maní",
    "Palitos de Apio con Mantequilla de Maní",
    "Tallos de Apio Rellenos",
    "Pepino con Dip de Queso",
    "Crudités con Hummus",
])
def test_otras_variantes_de_la_misma_clase(nombre):
    assert _LABEL in _viola(nombre)


# ───────────── 2. lo que NO debe tocar (falsos positivos) ─────────────

def test_la_fruta_con_mani_sigue_permitida():
    """El prompt la lista como categoría VÁLIDA y es una merienda real. Si el fix la bloquea,
    le quita al generador una de sus pocas meriendas legítimas con grasa saludable."""
    for n in ("Manzana con Mantequilla de Maní",
              "Guineo con Mantequilla de Maní",
              "Yogurt Griego con Mantequilla de Maní y Guineo"):
        assert _viola(n) == [], n


def test_el_apio_criollo_como_viver_no_se_flagea():
    """En RD 'apio' es también la arracacha, un víver de sancocho. Un token suelto habría
    convertido este fix en un falso positivo sobre comida dominicana de verdad."""
    assert _viola("Sancocho de Apio y Yuca", slot="almuerzo") == []
    assert _viola("Puré de Apio Criollo", slot="cena") == []


def test_vegetales_en_otros_slots_no_se_tocan():
    """La regla es de MERIENDA. Brócoli al vapor como guarnición del almuerzo es correcto."""
    assert _viola("Pollo a la Plancha con Brócoli al Vapor", slot="almuerzo") == []
    assert _viola("Pescado al Horno con Brócoli al Vapor", slot="cena") == []


def test_meriendas_dominicanas_legitimas_pasan():
    for n in ("Casabe con Queso Blanco",
              "Yogurt Griego con Lechosa y Granola",
              "Batido de Mamey",
              "Arroz con Leche",
              "Huevo Duro con Aguacate"):
        assert _viola(n) == [], n


# ───────────── 3. contrato de la regla ─────────────

def test_es_soft_nunca_deja_al_usuario_sin_plan():
    """Toda regla de merienda degrada a advisory en el intento final. Un `hard` aquí podría
    dejar un plan sin generar por una cuestión de estilo culinario."""
    v = slot_violations_for_meal_name("Apio Relleno con Mantequilla de Maní", "merienda")
    assert v and all(x["hard"] is False for x in v)


def test_la_regla_vive_en_el_SSOT_del_slot():
    labels = [r["label"] for r in SLOT_INAPPROPRIATE_FOODS["merienda"]]
    assert _LABEL in labels, "debe vivir en SLOT_INAPPROPRIATE_FOODS, que es lo que leen todas las surfaces"


def test_tokens_compuestos_no_sueltos():
    """Un token de una sola palabra ('apio', 'zanahoria', 'brocoli') reabriría el falso
    positivo sobre víveres y guarniciones."""
    regla = next(r for r in SLOT_INAPPROPRIATE_FOODS["merienda"] if r["label"] == _LABEL)
    sueltos = [t for t in regla["tokens"] if " " not in t]
    assert sueltos == ["crudites"], f"tokens de una palabra inesperados: {sueltos}"


def test_el_prompt_tambien_lo_dice():
    """Defensa en profundidad: el gate es la red, el prompt es el lever upstream. Sin el
    prompt el modelo sigue proponiéndolo y se gasta un reintento cada vez."""
    from pathlib import Path
    import prompts.day_generator as dg
    src = Path(dg.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("mantequilla de maní/almendras")
    bloque = src[i:i + 500]
    assert "SOLO FRUTA" in bloque
    assert "P1-SLOT-MERIENDA-CRUDITES" in bloque


def test_canonicalizacion_del_slot_sigue_funcionando():
    """Las meriendas llegan como 'Merienda', 'Merienda AM', 'Snack'…"""
    for etiqueta in ("Merienda", "merienda", "Merienda AM"):
        assert canonical_slot_key(etiqueta) == "merienda", etiqueta
