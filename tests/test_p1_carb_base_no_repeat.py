"""[P1-CARB-BASE-NO-REPEAT · 2026-07-27] La asignación de bases no llegaba a quien escribe la comida.

## Lo que pasó

Por la mañana se repartieron DOS bases de carbohidrato por día (P1-CARB-SEEDER-PAIRS) para cerrar
los "824 g de papa en un día" que reportó el owner. Al revisar el primer plan vivo generado con el
cambio (08114452), el día 2 seguía llevando papa en **almuerzo Y cena** (225 g + 74 g).

## Por qué el arreglo de la mañana no bastaba

Los placeholders `{carb_0b}/{carb_1b}/{carb_2b}` existen **solo** en `prompts/preferences.py`, que
alimenta el prompt del ESQUELETO. Quien escribe los ingredientes reales es `day_generator`, y ahí
el `carb_pool` del día se listaba pelado:

    • Carbohidratos Asignados: Batata, Papas

Sin ninguna regla. Es "contar los caminos" otra vez — [[project_pantry_gate_ssot_loop]]: el dato se
calcula bien, se imprime bien en los logs, y no llega al destinatario que decide.

⚠️ Casi diagnostico esto mal: medí "la batata aparece 0 veces" contra la siembra del intento 1,
pero el plan fue RECHAZADO y el sembrador **volvió a correr con bases distintas en cada reintento**
(3 siembras para 1 plan). Comparar contra una asignación descartada no prueba nada.

## Por qué la regla solo sale con ≥2 bases

Pedir "no repitas" con una sola base asignada es una restricción **insatisfacible**, y ya se pagó
esa factura: el gate de fruta pedía 1 por comida mientras el sembrador daba 1 por día → 67% de
planes reintentando (P1-FRUIT-SEEDER-GATE-CONTRACT). El esquema pide 2 (`carb_pool`, ejemplo
`['Arroz integral', 'Batata']`), pero si llega 1 la regla se calla en vez de forzar reintentos.

tooltip-anchor: P1-CARB-BASE-NO-REPEAT
"""
from __future__ import annotations

import pytest

from prompts.day_generator import build_day_assignment_context


def _esqueleto(carbs):
    return {"brief_concept": "Día variado", "assigned_technique": "Salteado tipo Wok",
            "protein_pool": ["Pollo"], "carb_pool": list(carbs), "fruit_pool": ["Mango"],
            "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"]}


# ───────────── 1. con dos bases, la regla sale y las nombra ─────────────

def test_con_dos_bases_emite_la_regla():
    ctx = build_day_assignment_context(_esqueleto(["Batata", "Papas"]), 1)
    assert "NO REPITAS LA BASE" in ctx


def test_nombra_las_dos_bases_concretas():
    """Una regla genérica ('varía las bases') es mucho más fácil de ignorar que uno con nombres."""
    ctx = build_day_assignment_context(_esqueleto(["Batata", "Papas"]), 1)
    assert "'Batata'" in ctx and "'Papas'" in ctx


def test_la_regla_va_pegada_a_la_linea_de_carbohidratos():
    """Separada del dato que califica, el modelo la lee como una consigna suelta más."""
    ctx = build_day_assignment_context(_esqueleto(["Batata", "Papas"]), 1)
    i = ctx.index("Carbohidratos Asignados")
    j = ctx.index("NO REPITAS LA BASE")
    assert 0 < j - i < 200, "la regla debe ir junto al pool que califica"


def test_menciona_las_dos_comidas_fuertes():
    ctx = build_day_assignment_context(_esqueleto(["Yuca", "Arroz"]), 2)
    assert "Almuerzo" in ctx and "Cena" in ctx


# ───────────── 2. nunca pide lo imposible ─────────────

@pytest.mark.parametrize("carbs", [[], ["Papas"], ["   "], [""]])
def test_sin_dos_bases_la_regla_se_calla(carbs):
    """Ancla de la CLASE. Una restricción insatisfacible no produce mejores planes: produce
    reintentos. El gate de fruta costó 67% de planes reintentando por exactamente esto."""
    ctx = build_day_assignment_context(_esqueleto(carbs), 1)
    assert "NO REPITAS LA BASE" not in ctx


def test_entradas_basura_no_revientan_el_prompt():
    """Si esto lanza, NO se genera ningún día — el prompt es camino crítico."""
    for carbs in ([None, "Papas"], [123, 456], ["Batata", None]):
        ctx = build_day_assignment_context(_esqueleto(carbs), 1)
        assert "ASIGNACIÓN DEL PLANIFICADOR" in ctx


# ───────────── 3. el canal correcto ─────────────

def test_la_regla_vive_en_el_generador_de_dias_no_solo_en_el_esqueleto():
    """Ancla del fallo original: la asignación estaba SOLO en el prompt del esqueleto y el
    generador de días —el que escribe los ingredientes— nunca la veía."""
    import inspect

    from prompts import day_generator, preferences

    gen = inspect.getsource(day_generator)
    assert "carb_no_repeat_block" in gen, (
        "el generador de días debe llevar la regla; el prompt del esqueleto no escribe ingredientes"
    )
    # y la fuente del dato sigue siendo el pool del día, no una lista hardcodeada
    assert "skeleton_day.get('carb_pool'" in gen

    # el reparto de la mañana sigue vivo en el prompt del esqueleto (no se sustituye, se suma)
    assert "{carb_0b}" in preferences.DETERMINISTIC_VARIETY_PROMPT
