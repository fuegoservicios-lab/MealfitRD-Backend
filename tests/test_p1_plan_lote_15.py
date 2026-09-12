# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-15 · 2026-09-12] Decimoquinto lote del plan de pendientes: E4 (ARQ27-P1-06), medido antes que construido.

La batería de entrega (`delivery_battery.py`) declaraba abiertas dos cosas: «no mide latencia ni coste por plan
entregado — eso es el canary y necesita generaciones reales» y «no ejercita swap ni último chunk end-to-end». Las
generaciones reales YA existen: son los planes de producción. Medido el 09-12:

  · El coste LLM no se podía atribuir a NINGÚN plan de la cola: 0 de 116 filas de `day_generator` con `plan_id` y 0 con
    `corr`; `swap_meal` 0 de 117 con `user_id`. El «canje corr → plan_id» de P1-COST-ATTRIBUTION nació para el SSE,
    donde el id llegaba DESPUÉS de generar; en la cola el placeholder ya tiene id ANTES y nadie lo estampaba.
    → `plan_id_var` + `set_llm_attribution` en el worker de chunks (con reset en el finally: el thread se reutiliza),
    /swap-meal (sólo si el plan es SUYO) y /regenerate-day (tras verificar propiedad); lo lee el emisor.
  · `scripts/canary_plan_delivery.py`: fallos, reintentos, latencia del bloque 1, coste y swaps por plan, con
    denominador por cohorte; «sin atribuir» no es US$ 0.
  · `scripts/verify_swap_last_chunk.py`: recetas↔lista con el MISMO guard de producción (emit de métricas anulado
    en el proceso), recetas completas, horizonte y proyección, sobre los planes persistidos — 6/6 coherentes el 09-12,
    incluido el plan con 48 revisiones de swaps. Sin comidas vivas el veredicto es «no concluyente», nunca «pasa».

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import asyncio
import re
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _script(name: str):
    """Carga `scripts/<name>.py` por RUTA, sin tocar sys.path (ratchet de LOTE-13)."""
    import importlib.util
    p = _BACKEND / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_lote15_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resultado:
    """Resultado LangChain mínimo: lo único que el emisor necesita es `usage_metadata`."""
    usage_metadata = {"input_tokens": 10, "output_tokens": 5}


# ─────────────────────────── atribución del coste LLM por plan ───────────────────────────

def test_plan_id_var_existe_y_por_defecto_no_atribuye():
    import graph_orchestrator as go
    assert go.plan_id_var.get() is None
    assert go.set_llm_attribution(None, None) == []
    go.reset_llm_attribution(None)  # best-effort: nunca revienta
    go.reset_llm_attribution([])


def test_el_emisor_atribuye_plan_y_usuario_y_el_reset_lo_deshace(monkeypatch):
    import graph_orchestrator as go
    import db_profiles
    capturado: list = []
    monkeypatch.setattr(db_profiles, "log_llm_usage_event", lambda **kw: capturado.append(kw))
    llm = types.SimpleNamespace(model="glm-5.3-flash")
    toks = go.set_llm_attribution("u-1", "p-1")
    try:
        go._emit_llm_usage_event_best_effort(llm=llm, result=_Resultado(), duration_s=0.5, node="day_generator")
    finally:
        go.reset_llm_attribution(toks)
    go._emit_llm_usage_event_best_effort(llm=llm, result=_Resultado(), duration_s=0.5, node="day_generator")
    assert capturado[0]["plan_id"] == "p-1" and capturado[0]["user_id"] == "u-1" and capturado[0]["node"] == "day_generator"
    assert capturado[1]["plan_id"] is None and capturado[1]["user_id"] is None, "tras el reset la fila nace sin atribuir, como antes"


def test_la_atribucion_se_propaga_a_tareas_e_hilos_de_asyncio():
    """El pipeline corre en tareas hijas y en `to_thread`; ambas heredan el contexto (por eso un ContextVar y no un global)."""
    import graph_orchestrator as go

    async def leer():
        return go.plan_id_var.get()

    async def main():
        toks = go.set_llm_attribution("u", "plan-x")
        try:
            return await asyncio.gather(asyncio.create_task(leer()), asyncio.to_thread(go.plan_id_var.get))
        finally:
            go.reset_llm_attribution(toks)

    assert asyncio.run(main()) == ["plan-x", "plan-x"]


def test_el_worker_de_chunks_fija_la_atribucion_antes_de_generar_y_la_deshace_en_el_finally():
    src = _src("cron_tasks.py")
    i = src.index("def _chunk_worker(task):")
    cabeza = src[i:i + 5000]
    assert "_llm_attr_toks = _set_llm_attr(user_id, meal_plan_id)" in cabeza
    assert "tooltip-anchor: chunk_worker_llm_attribution" in cabeza
    j = src.index("_reset_llm_attr(_llm_attr_toks)")
    assert j > i, "el reset pertenece al worker"
    assert "_CHUNK_WORKER_CTX.pickup_attempts = None" in src[j - 800:j], "el reset vive junto a la limpieza del contexto thread-local, en el finally"


def test_swap_y_regenerar_dia_atribuyen_solo_al_plan_propio():
    src = _src("routers/plans.py")
    swap = src[src.index('@router.post("/swap-meal")'):src.index('@router.post("/{plan_id}/swap-meal/persist")')]
    assert "tooltip-anchor: swap_llm_attribution" in swap
    assert '_owned_plan_id_for_attribution(verified_user_id, data.get("plan_id"))' in swap, "el body no es prueba de propiedad"
    regen = src[src.index('@router.post("/{plan_id}/regenerate-day")'):]
    regen = regen[:regen.index("@router.post", 10)]
    assert "tooltip-anchor: regenerate_day_llm_attribution" in regen
    k = regen.index("_set_llm_attr_regen(user_id, plan_id)")
    assert 'raise HTTPException(status_code=404, detail="Plan no encontrado")' in regen[:k], "la atribución va DESPUÉS de verificar la propiedad"


def test_owned_plan_id_for_attribution_niega_lo_ajeno_y_lo_invalido(monkeypatch):
    import routers.plans as rp
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: None)
    assert rp._owned_plan_id_for_attribution("u1", "p1") is None
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: {"ok": 1})
    assert rp._owned_plan_id_for_attribution("u1", "p1") == "p1"
    assert rp._owned_plan_id_for_attribution("guest", "p1") is None
    assert rp._owned_plan_id_for_attribution("u1", None) is None
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: (_ for _ in ()).throw(ValueError("uuid inválido")))
    assert rp._owned_plan_id_for_attribution("u1", "no-es-uuid") is None, "un id inválido no revienta el swap: la fila queda sin plan"


def test_el_emisor_lee_el_plan_del_contextvar_y_no_de_otra_tabla():
    src = _src("graph_orchestrator.py")
    i = src.index("def _emit_llm_usage_event_best_effort(")
    body = src[i:i + 9000]
    assert "_attr_pid = plan_id_var.get()" in body and "plan_id=_attr_pid," in body
    # El ContextVar vive en su módulo: graph_orchestrator.py está CONGELADO por tamaño (extraer, no subir el tope).
    assert "from llm_attribution import plan_id_var, set_llm_attribution, reset_llm_attribution" in src
    assert "tooltip-anchor: plan_id_var" in _src("llm_attribution.py")
    import graph_orchestrator as go
    import llm_attribution as la
    assert go.plan_id_var is la.plan_id_var and go.set_llm_attribution is la.set_llm_attribution


# ─────────────────────────── los dos scripts: sólo lectura ───────────────────────────

@pytest.mark.parametrize("name", ["canary_plan_delivery", "verify_swap_last_chunk"])
def test_los_scripts_son_solo_lectura_y_repetibles(name):
    src = _src(f"scripts/{name}.py")
    sin_comentarios = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert re.search(r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|TRUNCATE|ALTER\s+TABLE|DROP\s+TABLE)\b",
                     sin_comentarios, re.I) is None, "sólo SELECTs"
    assert re.search(r"sys\.path\.insert\(\s*0", src) is None, "ratchet LOTE-13: scripts/ nunca en cabeza de sys.path"
    assert "P2-LOGGER-EXEMPT" in src and '"--json"' in src


def test_el_canario_abre_en_read_only_y_el_verificador_anula_el_emit_del_guard():
    assert "conn.read_only = True" in _src("scripts/canary_plan_delivery.py")
    v = _src("scripts/verify_swap_last_chunk.py")
    assert "sc._emit_coherence_guard_metric = lambda **kw: None" in v, "el guard de producción no puede escribir métricas desde el script"
    assert "prod_profile.perfil_aplicado()" in v, "medido con los knobs de producción, no con la suite apagada"
    assert "sc.run_shopping_coherence_guard(pd, mode_override=\"warn\")" in v, "el MISMO guard, no una reimplementación"


def test_canario_clasifica_y_resume_con_denominador():
    m = _script("canary_plan_delivery")
    base = {"days_archivados": 0, "quality_degraded": False, "review_failed_delivered": False, "dead": 0,
            "done_con_reintento": 0, "llm_llamadas": None, "usd": None, "llm_s": None, "swaps": 0, "alertas": 0}
    filas = [
        dict(base, id="a" * 36, country="DO", diet="balanced", conditions=False, gs="complete", days_vivos=4, days_archivados=3,
             bloque1_s=200.0, llm_llamadas=8, usd=0.5, llm_s=120.0, swaps=1, proyeccion="ready"),
        dict(base, id="b" * 36, country="DO", diet="balanced", conditions=True, gs="complete", days_vivos=4,
             quality_degraded=True, done_con_reintento=2, bloque1_s=300.0, alertas=1, proyeccion="stale"),
        dict(base, id="c" * 36, country="ES", diet="vegana", conditions=False, gs="generating", days_vivos=0, dead=1,
             bloque1_s=None, proyeccion="none"),
    ]
    assert m.cohorte_de(filas[1]) == "DO·balanced·clinica" and m.cohorte_de(filas[2]) == "ES·vegana"
    c1 = m.clasificar(filas[1])
    assert c1["entregado"] and not c1["valido"] and "quality_degraded" in c1["motivos"]
    c2 = m.clasificar(filas[2])
    assert not c2["entregado"] and "dead=1" in c2["motivos"]
    r = m.resumen(filas)
    assert r["TODAS"]["n"] == 3 and r["TODAS"]["entregados"] == 2 and r["TODAS"]["validos"] == 1
    assert r["TODAS"]["coste_atribuidos"] == 1 and r["TODAS"]["coste_sin_atribuir"] == 1, "«sin atribuir» no es US$ 0"
    assert r["TODAS"]["usd_total"] == 0.5 and r["TODAS"]["usd_p50"] == 0.5
    assert r["TODAS"]["bloque1_p50_s"] == 250.0 and r["TODAS"]["dead"] == 1 and r["TODAS"]["con_reintento"] == 1
    assert r["DO·balanced"]["n"] == 1 and r["ES·vegana"]["no_entregados"][0]["plan"] == "c" * 8
    assert list(r)[-1] == "TODAS", "la fila global va al final: primero las cohortes"
    assert m.percentil([], 0.5) is None and m.percentil([1, 3], 0.5) == 2.0 and m.percentil([0.0011], 0.5, 3) == 0.001


def test_verificador_helpers_puros_y_no_concluyente_sin_comidas():
    m = _script("verify_swap_last_chunk")
    divs = [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "delta_pct": float("inf")},
            {"food": "Ajo", "hypothesis": "recipe_unquantified", "delta_pct": 10}]
    r = m.resumir_divergencias(divs)
    assert r["total"] == 2 and r["severa"] is True and r["en_receta_no_en_lista"] == ["Pollo"]
    assert m.resumir_divergencias([{"food": "Ajo", "hypothesis": "recipe_unquantified", "delta_pct": 10}])["severa"] is False
    assert m.resumir_divergencias(divs, severa_fn=lambda d: False)["severa"] is False, "si se pasa el criterio del guard, manda él"
    pd = {"days": [{"meals": [{"name": "Mangú", "ingredients": ["x"], "recipe": "..."},
                              {"name": "Sin receta", "ingredients": ["y"], "recipe": ""},
                              {"name": "Suplemento X", "ingredients": []}]}],
          "total_days_requested": 7, "_archived_days": [1, 2]}
    rec = m.recetas_incompletas(pd)
    assert rec["comidas"] == 2 and rec["sin_receta"] == ["d0m1:Sin receta"] and rec["sin_ingredientes"] == []
    h = m.horizonte(pd, {"done": 2, "abiertos": 0})
    assert h["generados"] == 3 and h["completo"] is False and h["ultimo_chunk_cerrado"] is True
    ok = m.veredicto({"severa": False}, {"comidas": 2, "sin_ingredientes": [], "sin_receta": []})
    assert ok["ok"] is True
    assert m.veredicto({"severa": True}, {"comidas": 2, "sin_ingredientes": [], "sin_receta": []})["ok"] is False
    nc = m.veredicto({"severa": False}, {"comidas": 0, "sin_ingredientes": [], "sin_receta": []})
    assert nc["ok"] is None and nc["recetas_ok"] is None, "sin comidas vivas no hay nada que aprobar"


# ─────────────────────────── docs y marker ───────────────────────────

def test_los_docs_cuentan_el_lote():
    plan = _src("docs/plan_pendientes_2026_09_11.md")
    assert re.search(r"^\| E4 \| ✅ 2026-09-12", plan, re.M), "E4 cerrado en el Estado del plan"
    doc = _src("docs/canary_entrega_e4.md")
    for frag in ("canary_plan_delivery.py", "verify_swap_last_chunk.py", "plan_id_var", "sin atribuir", "0 de 116"):
        assert frag in doc, frag
    bat = _src("scripts/delivery_battery.py")
    assert "canary_plan_delivery.py" in bat and "verify_swap_last_chunk.py" in bat, "la batería apunta a lo que la cierra"
    estado = (_ROOT / "docs" / "audits" / "2026-09-06-generacion" / "ESTADO-IMPLEMENTACION.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-15" in estado


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-15 · 2026-09-12]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
