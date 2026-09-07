# -*- coding: utf-8 -*-
"""[P1-POLICY-STAMP-ENFORCED · 2026-09-07] El sello de la política dice si estaba EN VIGOR.

Medido en producción sobre el plan `b9e9671a` del usuario del canary, la primera generación real
después de desplegar `P1-PORTION-HONORED`:

    chunks wk=1..8 -> `_policy_enforced = True`   (el canary funcionaba)
    sello          -> `enforced = False`          (y el techo del persist lee el SELLO)

La causa no era el canary sino DE QUÉ DICCIONARIO lee el sello. `stamp_plan_policy` resuelve el
flag con `form_data.get("_policy_enforced")`, y en el router ese diccionario es el formulario
**crudo del cliente**: la clave la inyecta el servidor, y solo la llevaban los snapshots de los
chunks 2..N (`_enqueue_remaining_chunks`) y el worker al ejecutar. El único call site del sello
—el que corre en la entrega— se quedaba fuera de los tres.

Consecuencia exacta: `build_count_caps_override` devuelve `None` sin `enforced`, así que la ración
pedida («desayuno 10 claras») volvía a recortarse a 6 al persistir. La cadena entera de
P1-ANCHOR-PORTION + P1-PORTION-HONORED quedaba **inerte en la vía que genera los planes**,
con todos sus tests en verde: ninguno miraba el valor sellado en un plan real.

> Un flag que se calcula en tres sitios y se LEE en un cuarto no está enhebrado: está adivinado.

## Dos huecos vecinos, cerrados MIDIENDO en vez de implementando (2026-09-07)

Al enumerar los productores aparecieron dos sitios más donde el flag podría faltar. Ninguno se
tocó, porque los dos resultaron vacíos en producción:

  · **La vía SÍNCRONA** (entrega sin cola de bloques) no inyecta el flag en ningún punto, así que
    el grafo usaría el techo por defecto. Medido: **0 de 72** planes de los últimos 30 días son
    síncronos — todos pasan por `plan_chunk_queue`, donde el worker sí lo resuelve al ejecutar.
  · **El queso no lonjeable** («4¾ lonjas de queso cottage»), el sub-patrón que
    [P1-JUDGE-REVISION-STAMP] dejó anotado como agujero abierto. Remedido cubriendo ahora también
    los PASOS de la receta, no solo los ingredientes: **0 de 25** líneas de queso con unidad vaga
    en 95 planes vivos. La conversión a gramos ya reescribe la línea —y el `ingredients_raw` en
    lockstep— antes de entregar.

Se dejan escritos aquí y no en un informe suelto para que el día que alguien mida distinto tenga
contra qué comparar. Un cero medido no es lo mismo que un cero supuesto.
"""
import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROUTER = BACKEND / "routers" / "plans.py"
POLICY = BACKEND / "plan_policy.py"
HORIZON = BACKEND / "horizon.py"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- el arreglo


def test_el_router_resuelve_el_flag_antes_de_sellar():
    """El call site del sello pregunta por la persona, no confía en el formulario crudo."""
    src = _src(ROUTER)
    assert "from horizon import policy_enforced as _pol_enf_stamp" in src
    assert 'data["_policy_enforced"] = _pol_enf_stamp(actual_user_id)' in src


def test_la_resolucion_va_ANTES_de_la_llamada_al_sello():
    """El orden es el arreglo: resolverlo después dejaría el sello exactamente como estaba."""
    src = _src(ROUTER)
    i_flag = src.index('data["_policy_enforced"] = _pol_enf_stamp(actual_user_id)')
    i_sello = src.index("_compiled_policy = _stamp_policy(result, data")
    assert i_flag < i_sello, "el flag se inyecta después de sellar: el sello no lo vería"


def test_el_fallo_al_resolver_no_tumba_la_entrega():
    """Fail-open: si `horizon` no responde se sella `False` (la conducta de antes), nunca un 500."""
    src = _src(ROUTER)
    bloque = src[src.index("from horizon import policy_enforced as _pol_enf_stamp") - 400:
                 src.index("_compiled_policy = _stamp_policy(result, data")]
    assert "try:" in bloque and "except Exception" in bloque
    assert "P1-POLICY-STAMP-ENFORCED" in bloque


def test_el_sello_sigue_leyendo_del_formulario():
    """No se cambia la firma de `stamp_plan_policy`: el arreglo es enhebrar el dato, no otra vía.

    Cambiar la lectura a un `user_id` propio habría dado al sello una SEGUNDA fuente de verdad
    sobre el modo, que es justo la clase de bug que P1-DIET-CANON-SSOT documenta.
    """
    assert 'compiled["enforced"] = bool((form_data or {}).get("_policy_enforced"))' in _src(POLICY)


# ------------------------------------------------- por qué importaba (blast radius)


def test_sin_enforced_el_techo_por_conteo_no_se_eleva():
    """La razón de que esto fuera P1 y no cosmético: el override exige el sello."""
    import sys

    sys.path.insert(0, str(BACKEND))
    from plan_policy import build_count_caps_override, ingredient_id_for

    base = {"clara": 6.0, "huevo": 4.0}
    # El ancla se identifica por `ingredient_id`, no por el nombre suelto: es lo que el compilador
    # sella y lo que `portion_cap_for` compara. Se calcula aquí en vez de fijarlo a mano para que
    # un cambio en el resolvedor rompa este test antes que la producción.
    iid = ingredient_id_for("Clara de huevo")
    assert iid, "sin ingredient_id el ancla no es identificable — el compilador debe sellarlo"
    anc = {"name": "Clara de huevo", "ingredient_id": iid, "portion": {"qty": 10, "unit": "unidad"}}
    eff = {"food_anchors": [anc]}

    assert build_count_caps_override({"effective": eff, "enforced": False}, base) is None
    assert build_count_caps_override({"effective": eff}, base) is None  # ausente == no en vigor

    elevado = build_count_caps_override({"effective": eff, "enforced": True}, base)
    assert elevado is not None and elevado["clara"] == 10.0
    assert elevado["huevo"] == 4.0, "el huevo entero no se toca: «Clara de huevo» no es «huevo»"
    assert base["clara"] == 6.0, "el diccionario global no se muta"

    # Y sin `ingredient_id` el ancla no eleva nada: la identidad no se adivina por el nombre.
    sin_id = {"food_anchors": [{k: v for k, v in anc.items() if k != "ingredient_id"}]}
    assert build_count_caps_override({"effective": sin_id, "enforced": True}, base) is None


# ---------------------------------------------------- los cuatro sitios, enumerados


def test_los_cuatro_call_sites_resuelven_el_flag_por_la_misma_puerta():
    """`horizon.policy_enforced(user_id)` es la ÚNICA vía: cuatro sitios, cero tablas paralelas."""
    sitios = {
        "generation_lifecycle.py": 'form_data["_policy_enforced"] = _policy_enforced_f3(user_id)',
        "cron_tasks.py": 'form_data["_policy_enforced"] = _policy_enforced_f3(user_id)',
    }
    for fichero, linea in sitios.items():
        assert linea in _src(BACKEND / fichero), f"{fichero} dejó de resolver el flag"

    router = _src(ROUTER)
    assert "_enforced_f3 = _pol_enf_f3(actual_user_id)" in router, "chunks 2..N"
    assert 'data["_policy_enforced"] = _pol_enf_stamp(actual_user_id)' in router, "el sello"


def test_policy_enforced_sigue_siendo_la_puerta_del_canary():
    """Si alguien colapsa `policy_enforced` a un knob global, el canary deja de existir."""
    src = _src(HORIZON)
    assert "def policy_enforced(" in src
    assert "policy_mode_for_user(user_id) == \"enforce\"" in src
    assert "MEALFIT_PLAN_POLICY_ENFORCE_USERS" in src


# ------------------------------------------------------------------ la clave es interna


def test_el_flag_es_una_clave_que_el_servidor_inyecta_no_el_cliente():
    """`_policy_enforced` está en la lista de internas: un cliente que la mande cae en el strip.

    Sin esto el arreglo sería un agujero: cualquiera pediría 100 claras mandando el flag a mano.
    """
    go = _src(BACKEND / "graph_orchestrator.py")
    assert '"_policy_enforced",' in go
    # y la lista es la de claves INYECTADAS POR EL SERVIDOR, no una whitelist para el cliente
    ventana = go[go.index('"_blueprint_slice",') - 700:go.index('"_policy_day_index",')]
    assert "strip estricto" in ventana or "Las inyecta el servidor" in ventana


def test_el_marcador_vive_en_el_router():
    assert re.search(r"P1-POLICY-STAMP-ENFORCED\s*·\s*2026-09-07", _src(ROUTER))


def test_el_bloque_compila():
    """Guard barato contra un try/except mal cerrado en un fichero de 50k líneas."""
    ast.parse(_src(ROUTER))
