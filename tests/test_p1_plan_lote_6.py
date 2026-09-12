"""[P1-PLAN-LOTE-6 · 2026-09-11] Sexto lote del plan de pendientes: F5 (guards inertes), segunda tanda.

  · Los 68 manejadores `except Exception: pass|continue|return <permisivo>` de las funciones que el informe señala
    (§4.1-4.3 de `docs/audits/f5_guards_inertes_2026_09_11.md`) registran el paso que se tragan; en la cadena de
    `finalize_plan_data_coherence` el fallo además viaja en `parts` (`<paso>=ERR`). Antes, un pulido que reventaba
    dejaba el plan sin esa reparación y el log no decía nada.
  · Los veredictos «advisory en el intento final» tenían escritor y ningún lector: ahora llegan a la alerta SRE
    `plan_quality_degraded` (`final_attempt_advisories`). El banner del usuario NO cambia (sus 14 motivos son contrato).
  · Tres knobs sin lector o sin rama, fuera; dos `if True:` vestigiales, fuera; 20 knobs default-off documentados.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

# Las funciones del informe. Si añades un `except Exception: pass` a una de ellas, este test te lo dirá.
_FUNCIONES_DEL_INFORME = {
    "graph_orchestrator.py": [
        "finalize_plan_data_coherence", "_run_assembly_validations", "_apply_food_safety_fixes", "_day_sodium_autofix",
        "finalize_single_meal_recipe_coherence", "_repair_recipe_contract", "_generation_sanity_autofix",
        "_apply_coherence_history_cap", "_coherence_block_history_cap", "refresh_clinical_band_score_post_finalize",
        "semantic_cache_check_node", "_variety_repeat_gate_issues", "_recipe_step_contract_issues",
        "slot_coherence_backstop_for_meal", "_clamp_recipe_time_temp_outliers", "_maybe_mark_clinical_layer_incomplete_degraded",
    ],
    "cron_tasks.py": ["_refresh_chunk_pantry_inner", "_persist_fresh_pantry_to_chunks", "_recover_pantry_paused_chunks",
                      "_alert_coherence_watchdog_silent", "_shopping_coherence_alert_job", "_clinical_band_drift_alert_job",
                      "_alert_chunk_pantry_snapshots_stale"],
    "deterministic_day.py": ["verifica_comida"],
    "db_inventory.py": ["find_pantry_rows_for_name"],
    "agent.py": ["_swap_real_pantry_ledger_lines"],
    "condition_rules.py": ["collect_allergen_substitutions"],
    "culinary_coherence.py": ["culinary_contract_scan"],
    "ai_helpers.py": ["_n_gate_fruits"],
}

_KNOBS_DOCUMENTADOS = (
    "MEALFIT_HARDEN_MAIN_ARITY", "MEALFIT_MICRONUTRIENT_SOFT_REJECT", "MEALFIT_FAT_LEAN_SWAP",
    "MEALFIT_VARIETY_GATE_BASE_DISH_REPEAT", "MEALFIT_CARB_TARGET_TRIM", "MEALFIT_CORRECTOR_NONE_DIAGNOSTIC",
    "MEALFIT_EVALUATOR_USE_PRO", "MEALFIT_DAYGEN_LITE_FOR_EASY", "MEALFIT_INITIAL_CHUNK_PANTRY_GUARD",
    "MEALFIT_RENEWAL_PANTRY_AWARE_ENABLED", "MEALFIT_PANTRY_COMPLETION_LIST_ENABLED", "MEALFIT_PANTRY_SUFFICIENCY_MICROS_GATE",
    "MEALFIT_REQUIRE_ATOMIC_POOL", "MEALFIT_INVENTORY_RPC_STRICT", "MEALFIT_LEAK_DB_ERRORS", "MEALFIT_READY_REQUIRE_DB",
    "MEALFIT_LIGHT_PROTEIN_SEED", "MEALFIT_GROCERY_CYCLE_LOCK", "MEALFIT_ANEMIA_CONDITION_TARGET", "MEALFIT_DISABLE_SEMANTIC_CACHE",
)


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _cuerpo_trivial(body) -> bool:
    if len(body) != 1:
        return False
    st = body[0]
    if isinstance(st, (ast.Pass, ast.Continue)):
        return True
    if isinstance(st, ast.Return):
        v = st.value
        return v is None or isinstance(v, ast.Constant) or (
            isinstance(v, (ast.List, ast.Dict, ast.Tuple)) and not getattr(v, "elts", getattr(v, "keys", None)))
    return False


def _manejadores_mudos(path: str, fn: str) -> list:
    tree = ast.parse(_src(path))
    node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn), None)
    assert node is not None, f"{path}: {fn} no existe (si la renombraste, actualiza la lista del informe y este test)"
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.ExceptHandler):
            ty = n.type
            generico = ty is None or (isinstance(ty, ast.Name) and ty.id in ("Exception", "BaseException"))
            if generico and n.name is None and _cuerpo_trivial(n.body):
                out.append(n.lineno)
    return out


# ─────────────────────────── A · el paso tragado deja rastro ───────────────────────────

def test_f5_las_funciones_del_informe_ya_no_se_tragan_el_paso_en_silencio():
    mudos = {f"{p}:{fn}": _manejadores_mudos(p, fn) for p, fns in _FUNCIONES_DEL_INFORME.items() for fn in fns}
    assert not any(mudos.values()), {k: v for k, v in mudos.items() if v}


def test_f5_finalize_registra_el_pulido_que_revienta_y_lo_lleva_en_parts(monkeypatch, caplog):
    import graph_orchestrator as go

    def _boom(days):
        raise RuntimeError("pulido roto")

    monkeypatch.setattr(go, "_polish_finalize_display", _boom)
    days = [{"day": 1, "meals": [{"name": "Pollo guisado", "ingredients": ["200 g de Pollo"], "recipe": ["Guisar el pollo."]}]}]
    with caplog.at_level(logging.WARNING):
        total, parts = go.finalize_plan_data_coherence(days)
    assert "_polish_finalize_display=ERR" in parts, parts
    assert any("[P1-PLAN-LOTE-6]" in r.getMessage() and "_polish_finalize_display" in r.getMessage() for r in caplog.records)


def test_f5_dentro_de_un_bucle_el_rastro_es_info_para_no_inundar():
    src = _src("graph_orchestrator.py")
    i = src.find("def _run_assembly_validations(")
    body = src[i:src.find("\ndef ", i + 10)]
    assert 'logger.info(f"[P1-PLAN-LOTE-6] _run_assembly_validations:' in body
    assert "except Exception:\n" not in body.replace("except Exception as", "")


def test_f5_los_modulos_sin_logger_ganaron_uno():
    for mod in ("condition_rules.py", "culinary_coherence.py"):
        src = _src(mod)
        assert re.search(r"^logger = logging\.getLogger\(__name__\)", src, re.M), mod
        assert re.search(r"^import logging", src, re.M), mod


# ─────────────────────────── D · los advisories del intento final tienen lector ───────────────────────────

def test_f5_final_attempt_advisories_lee_los_cinco_y_el_reviewer_solo_acompana():
    import graph_orchestrator as go
    assert go._final_attempt_advisories(None) == [] and go._final_attempt_advisories({}) == []
    plan = {"_slot_incoherence_advisory_final": True, "_staple_repeat_advisory_final": False,
            "variety_report": {"_repeat_gate_advisory_final_attempt": True}, "_reviewer_advisories": ["verificar sodio"]}
    assert go._final_attempt_advisories(plan) == [
        "_slot_incoherence_advisory_final", "_repeat_gate_advisory_final_attempt", "_reviewer_advisories"]
    assert go._final_attempt_advisories(plan, include_reviewer=False) == [
        "_slot_incoherence_advisory_final", "_repeat_gate_advisory_final_attempt"]
    assert go._final_attempt_advisories({"_reviewer_advisories": ["x"]}, include_reviewer=False) == []
    assert go._final_attempt_advisories({"_dish_quality_advisory_final": True}) == ["_dish_quality_advisory_final"]


def test_f5_la_alerta_sre_lleva_los_advisories(monkeypatch):
    import db_core
    import graph_orchestrator as go
    capturado = {}
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: {"key": "lock"})

    def _write(sql, params=None, *a, **k):
        capturado["params"] = params
        return 1

    monkeypatch.setattr(db_core, "execute_sql_write", _write)
    state = {"form_data": {"user_id": "u-1"}, "attempt": 3, "review_passed": True,
             "plan_result": {"id": "p-1", "_dish_quality_advisory_final": True, "_reviewer_advisories": ["ver"]}}
    go._emit_plan_quality_degraded_alert(state, "approved_with_residual", severity="minor")
    metas = []
    for p in capturado.get("params") or ():
        if isinstance(p, str) and p.startswith("{"):
            try:
                metas.append(json.loads(p))
            except ValueError:
                pass
    meta = next((m for m in metas if "final_attempt_advisories" in m), None)
    assert meta is not None, capturado
    assert meta["final_attempt_advisories"] == ["_dish_quality_advisory_final", "_reviewer_advisories"]


def test_f5_should_retry_emite_la_alerta_cuando_un_gate_se_degrado_a_advisory():
    src = _src("graph_orchestrator.py")
    j = src.find("def should_retry(")
    k = src.find('logger.info("✅ [ORQUESTADOR] Revisión aprobada → Enviando al usuario.")')
    assert 0 < j < k
    tramo = src[k - 1500:k]
    assert '_final_attempt_advisories(state.get("plan_result"), include_reviewer=False)' in tramo
    assert 'if APPROVED_RESIDUAL_ALERT_ENABLED:' in tramo
    assert '_emit_plan_quality_degraded_alert(state, "approved_with_residual", severity="minor")' in tramo
    assert "_mark_plan_result_quality_degraded" not in tramo.split("_final_attempt_advisories")[-1], (
        "el banner del usuario no cambia en este lote: sus 14 motivos son contrato con el frontend")


# ─────────────────────────── B/C/E · muertos fuera, documentación ───────────────────────────

def test_f5_knobs_sin_lector_o_sin_rama_fuera():
    import constants
    import graph_orchestrator as go
    for n in ("COUNTRY_SYSTEM_ENABLED", "CHUNK_STALE_FINAL_LIVE_TIMEOUT_SECONDS"):
        assert not hasattr(constants, n), n
    assert not hasattr(go, "SLOT_AWARE_DAY_REPAIR")
    assert 'if not _env_bool("MEALFIT_COUNTRY_SYSTEM", False)' in _src("constants.py"), "el knob maestro se lee POR LLAMADA"


def test_f5_if_true_vestigiales_fuera():
    assert not re.search(r"^\s*if True:\s*$", _src("graph_orchestrator.py"), re.M)


def test_f5_knobs_default_off_documentados_con_criterio():
    doc = _src("docs/knobs_reference.md")
    assert "inventario F5" in doc
    faltan = [k for k in _KNOBS_DOCUMENTADOS if f"`{k}`" not in doc]
    assert not faltan, faltan


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-6 · 2026-09-11]" in _src("app.py")
    # [P1-PLAN-LOTE-13 · 2026-09-12] «no anterior a este lote», no «igual a hoy»: el pin de la fecha y del prefijo
    # `P1-PLAN-` rompía 12 tests el primer día en que otro P-fix bumpeaba el marker.
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-11", app._LAST_KNOWN_PFIX
