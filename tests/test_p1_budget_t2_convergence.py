"""[P1-BUDGET-T2-CONVERGENCE · 2026-07-29] En el flujo deep-search el costeo de assemble
sale VACÍO (la lista aún no trae precios) → `budget_reconciliation` y la convergencia de
assemble mueren de hambre; el único momento con costo REAL es el seam T2 del chunk worker,
que MEDÍA `excedido` y no actuaba. Plan vivo 73db1e79: RD$21,673 vs RD$15,324 (ratio 1.414),
banner rojo, CERO líneas de CHEAPEN/DRIVER-AWARE en toda la generación (verificado en logs
corr=8469264b). Cierre: helper síncrono `apply_budget_convergence_for_days` (misma secuencia
del bloque de assemble) + gate en T2: excedido → sustituir → rebuild con los MISMOS
snapshots → re-costear → re-reconciliar. Una pasada, jamás loop, fail-open total.

tooltip-anchor: P1-BUDGET-T2-CONVERGENCE
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def _plan():
    return {"days": [{"meals": [{"name": "Mero al Horno", "ingredients": ["200 g de mero"],
                                 "recipe": ["Hornea."], "protein": 40, "carbs": 10,
                                 "fats": 8, "cals": 280}]}],
            "aggregated_shopping_list_weekly": [
                {"name": "Mero", "price_estimate_rd": 671, "total_grams": 975}]}


def test_helper_orchestrates_and_marks(monkeypatch):
    calls = {}
    # [P2-BUDGET-CONVERGENCE-FUTURE-ONLY · 2026-08-03] `**kw` por la MISMA razón que el doble
    # del motor de macros documentada abajo: los dos pases recibieron `inventory_names=` (skip
    # de lo ya comprado) y con firma cerrada el TypeError lo tragaba el fail-open del helper →
    # 0 sustituciones y este test caía por "no orquestó" cuando sí orquestaba.
    monkeypatch.setattr(go, "_apply_budget_driver_aware_pass",
                        lambda days, fd, weekly, **kw: calls.setdefault("driver", (len(days), len(weekly))) and 2 or 2)
    monkeypatch.setattr(go, "_apply_budget_cheapen_pass",
                        lambda days, fd, force=False, **kw: calls.setdefault("cheapen_force", force) or 1)
    monkeypatch.setattr(go, "_protein_repeat_autofix", lambda days, fd: 0)
    # [P1-UPDATE-RECAP-ALL-SURFACES · 2026-07-30] `**kw` a propósito: el doble del motor tenía
    # firma CERRADA (`pd, surface, db`), así que al empezar a pasarle `form_data=` el stub lanzaba
    # TypeError → lo tragaba el `except Exception` del bloque T2 → `engine_surface` nunca se
    # seteaba y el test caía por "no se invocó el motor" cuando el motor SÍ se invocaba.
    # Un doble con firma cerrada convierte un argumento nuevo en un falso negativo.
    monkeypatch.setattr(go, "apply_update_macro_engine",
                        lambda pd, surface=None, db=None, **kw: (
                            calls.setdefault("engine_surface", surface),
                            calls.setdefault("engine_form_data", kw.get("form_data")))[0])
    monkeypatch.setattr(go, "recompute_micronutrient_report_for_plan",
                        lambda pd, fd, db=None: calls.setdefault("micros", True))
    plan = _plan()
    n = go.apply_budget_convergence_for_days(plan, {"budget": "low"})
    assert n == 3
    assert plan.get("_budget_adjusted") is True
    assert calls.get("cheapen_force") is True, "el cheapen corre con force=True (salta el gate de economía)"
    assert calls.get("engine_surface") == "budget_convergence_t2"
    # [P1-UPDATE-RECAP-ALL-SURFACES · 2026-07-30] T2 (chunk worker, semanas 2+) es la superficie
    # más silenciosa del sistema y llamaba al motor SIN form_data ⇒ el re-cap clínico DM2/
    # bariátrico se omitía sobre porciones que el rebalance/refine acababa de re-inflar.
    assert calls.get("engine_form_data") is not None, (
        "el motor debe recibir form_data para poder re-aplicar los caps clínicos de porción")


def test_helper_zero_subs_is_pure_noop(monkeypatch):
    # [FINAL-REVIEW-P2 · 2026-08-03] 3ª vez de la clase: un stub de firma CERRADA
    # (`lambda *a`) no acepta el kwarg `inventory_names=` que el caller pasa
    # incondicionalmente (P2-BUDGET-CONVERGENCE-FUTURE-ONLY) → TypeError tragado por
    # el fail-open `except Exception` del helper → `apply_budget_convergence_for_days`
    # retorna 0 ANTES de invocar el segundo stub, y el test pasaba sin haber ejecutado
    # nada (verificado: con el stub viejo, `reached` quedaba `[]`). Firma abierta
    # (`**k`) en AMBOS stubs + contador que prueba que de verdad se alcanzaron.
    reached = []
    monkeypatch.setattr(go, "_apply_budget_driver_aware_pass",
                        lambda *a, **k: reached.append("driver") or 0)
    monkeypatch.setattr(go, "_apply_budget_cheapen_pass",
                        lambda *a, **k: reached.append("cheapen") or 0)
    plan = _plan()
    assert go.apply_budget_convergence_for_days(plan, {}) == 0
    assert "_budget_adjusted" not in plan
    assert reached == ["driver", "cheapen"], (
        "los dos stubs deben ser alcanzados de verdad — si uno solo aparece (o ninguno), "
        "un TypeError por firma cerrada se está tragando silenciosamente el fail-open")


def test_t2_seam_wired_with_rebuild_and_rereconcile():
    i = _CRON.index("P1-BUDGET-T2-CONVERGENCE")
    # [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] 4200 -> 5000: el ventaneo anadio
    # `window_days=_trip_win` a las 3 llamadas + su comentario dentro de ESTE bloque, y
    # `_rbr_t2(full_plan_data)` quedaba fuera de la ventana fija. El tamano del slice es
    # detalle del test (no un contrato): las aserciones de abajo son las mismas.
    # [P2-BUDGET-CONVERGENCE-FUTURE-ONLY · 2026-08-03] 5000 -> 7000: el comentario que
    # justifica pasarle `inventory_names=_inv_s` al helper volvió a empujar
    # `_rbr_t2(full_plan_data)` fuera de la ventana fija. Segunda vez que pasa — el slice
    # sigue siendo detalle del test, no un contrato.
    blk = _CRON[i:i + 7000]
    assert 'str(_bc_rec_t2.get("status") or "") == "excedido"' in blk, "gate por status excedido"
    assert "apply_budget_convergence_for_days" in blk
    assert blk.count("get_shopping_list_delta(") == 3, "rebuild de las 3 multiplicidades"
    assert "inventory_override=_inv_s" in blk, "MISMOS snapshots de la 1ª pasada (no re-fetch)"
    assert "_rbr_t2(full_plan_data)" in blk, "re-reconciliación honesta post-rebuild"
    assert "_build_hybrid_shopping_list" in blk, "las listas re-construidas pasan por el híbrido"


def test_helper_fail_open():
    assert go.apply_budget_convergence_for_days(None, None) == 0
    assert go.apply_budget_convergence_for_days({}, {}) == 0
