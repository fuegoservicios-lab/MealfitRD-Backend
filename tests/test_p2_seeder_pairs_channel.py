"""[P2-SEEDER-PAIRS-CHANNEL · 2026-07-31] (audit solver+seeder v6 · P2 / F23) El reparto de
carbohidratos y frutas del seeder moría en la prosa del prompt.

`get_deterministic_variety_prompt` calcula un reparto DETERMINISTA por día para las tres
categorías (`_carb_slots`, `_fruit_slots`, `_veg_slots`), pero solo los vegetales viajaban al
esqueleto como DATO (`out_assignment['veggie_pairs']` → `plan_skeleton_node`). Carbos y frutas
solo iban como texto dentro del prompt del planner, así que llegaban al day-generator únicamente
si el LLM se molestaba en copiarlos a `carb_pool`/`fruit_pool`. Cuando emitía un pool de un solo
ítem, la regla anti-repetición se perdía y el gate de variedad rechazaba el día: un retry de
90-210 s por una asignación que el seeder ya había resuelto bien.

⚠️ LA PARTE QUE NO SE PUEDE COPIAR DEL BLOQUE DE VEGETALES

`harden_day_pools` (A1-HARDEN-POOLS) poda `protein_pool|carb_pool|fruit_pool` por CONDICIÓN
MÉDICA — toronja fuera con CYP3A4, arroz blanco por índice glucémico. NO toca `veggie_pool`.
El seeder filtra por alergias/dislikes/dieta pero no sabe nada de condiciones clínicas.

Por eso el bloque de vegetales puede vivir DESPUÉS del enforcer y el de carbos/frutas NO:
estampar después reinyectaría justo lo que el enforcer acaba de quitar. Y como
`HARDEN_POOLS_ENABLED` es False por default, el daño quedaría LATENTE hasta que alguien encienda
el knob — una mina, no un bug visible. El orden es la parte del fix que más importa y por eso
tiene test propio.

Anchor de producción: P2-SEEDER-PAIRS-CHANNEL.
"""
import re
from pathlib import Path

import pytest

GO = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


# --------------------------------------------------------------- el canal existe

def test_el_seeder_publica_los_tres_repartos():
    """`out_assignment` debe transportar las tres categorías, no solo los vegetales."""
    import ai_helpers as ah

    salida = {}
    ah.get_deterministic_variety_prompt("", {"mainGoal": "maintenance"},
                                        user_id=None, out_assignment=salida)
    faltan = [k for k in ("veggie_pairs", "carb_pairs", "fruit_pairs") if k not in salida]
    assert not faltan, f"el seeder no publica {faltan} (mueren en la prosa del prompt)"
    for k in ("carb_pairs", "fruit_pairs"):
        assert salida[k], f"{k} vacío"
        for par in salida[k]:
            assert len(par) == 2 and par[0] != par[1], f"{k} trae un par degenerado: {par}"


# --------------------------------------------------------------- el extractor

@pytest.mark.parametrize("clave", ["veggie_pairs", "carb_pairs", "fruit_pairs"])
def test_el_extractor_sirve_para_las_tres_claves(clave):
    import graph_orchestrator as go

    ctx = {"seeder_assignment": {clave: [("A", "B"), ("B", "C")]}}
    assert go._extract_seeder_pairs(ctx, clave) == [("A", "B"), ("B", "C")]
    assert go._extract_seeder_pairs({}, clave) == []
    assert go._extract_seeder_pairs(None, clave) == []
    assert go._extract_seeder_pairs({"seeder_assignment": {clave: [("A", "A")]}}, clave) == [], \
        "un par degenerado (mismo ítem dos veces) no puede propagarse"


def test_el_alias_viejo_sigue_vivo():
    """`_extract_seeder_veggie_pairs` tiene tests que lo llaman por nombre."""
    import graph_orchestrator as go
    ctx = {"seeder_assignment": {"veggie_pairs": [("A", "B")]}}
    assert go._extract_seeder_veggie_pairs(ctx) == [("A", "B")]


# --------------------------------------------------------------- LA invariante de orden

def test_carbos_y_frutas_se_estampan_antes_del_enforcer_clinico():
    """tooltip-anchor de producción: P2-SEEDER-PAIRS-CHANNEL

    `harden_day_pools` poda carb_pool y fruit_pool por condición médica. Si el estampado del
    seeder corre DESPUÉS, reinyecta lo que el enforcer quitó — y con el knob OFF por default eso
    no se vería hasta que alguien lo encienda.
    """
    src = GO.read_text(encoding="utf-8", errors="ignore")
    i_enforcer = src.index("_harden_counts = harden_day_pools(")
    # Ancla al bucle que hace el estampado (la línea que DECIDE), no a un literal de asignación:
    # la clave del pool es una variable, así que no aparece escrita en el sitio de la escritura.
    i_estampado = src.index('for _pk, _sk in (("carb_pairs", "carb_pool")')

    assert i_estampado < i_enforcer, (
        "el estampado de carb_pool/fruit_pool corre DESPUÉS de harden_day_pools: reinyectaría los "
        "alimentos que el enforcer quita por condición médica (toronja↔CYP3A4, arroz blanco por "
        "índice glucémico). Con HARDEN_POOLS_ENABLED=False el daño no se vería hasta encender el knob."
    )


def test_el_bloque_de_vegetales_sigue_donde_estaba():
    """Control: veggie_pool NO lo toca el enforcer, así que su bloque puede seguir después."""
    src = GO.read_text(encoding="utf-8", errors="ignore")
    assert src.index("_harden_counts = harden_day_pools(") < src.index('"veggie_pool"] = '), (
        "el bloque de vegetales se movió; si fue a propósito, revisar que harden_day_pools "
        "sigue sin tocar veggie_pool"
    )


# --------------------------------------------------------------- no pisar al planner

def test_el_estampado_respeta_lo_que_el_planner_emitio():
    """Solo COMPLETA pools cortos; jamás sustituye la elección temática del planner."""
    src = GO.read_text(encoding="utf-8", errors="ignore")
    i = src.index("P2-SEEDER-PAIRS-CHANNEL")
    bloque = src[i: i + 2000]
    assert "len(_cur) >= 2" in bloque or "len(_cur) < 2" in bloque, (
        "debe haber un guard de aridad: con 2+ ítems del planner no se toca nada"
    )
