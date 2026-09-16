# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-66 · 2026-09-16] El invitado conserva su lista de compras después del swap.

Encontrado generando 4 planes REALES con el pipeline (bench `--real`, perfiles del landing con `user_id: guest`, $0,043):
los cuatro nacieron con lista sana (48 ítems distintos, coste calculado, 2 divergencias) y los cuatro acabaron con
`aggregated_shopping_list*` VACÍA y con 45-51 divergencias «críticas» (`cap_swallowed_modifier`) escaladas al banner
`_swap_coherence_warnings`. La causa no estaba en el generador: `assemble_plan_node` construye la lista del invitado a
propósito (`P1-GUEST-SHOPPING`) y `_recompute_aggregates_after_swap` —swap entre intentos y auto-patch del review— la
vaciaba, y su propia re-validación comparaba las recetas contra cero.

(a) el invitado sale de la re-agregación CON lista (antes: vacía);
(b) las dos ramas de invitado son la misma: mismo builder, overrides vacíos, mismo fallback;
(c) el usuario con `user_id` real no cambia: sigue descontando su inventario;
(d) si el builder revienta, la lista queda vacía y se dice en el log (fail-open, sin romper la entrega);
(e) docs y marker ≥ 66.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_GO_PY = _BACKEND / "graph_orchestrator.py"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@pytest.fixture
def estado_invitado():
    return {
        "plan_result": {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "ingredients": ["150 g de Pollo"]}]}],
                        "calc_household_multiplier": 1.0},
        "form_data": {"user_id": "guest", "groceryDuration": "weekly"},
    }


@pytest.fixture
def constructor(monkeypatch):
    """El builder devuelve lista NO vacía; el guard, sin divergencias (aquí no se mide la telemetría)."""
    import shopping_calculator as sc
    llamadas = []

    def _delta(uid, plan, *a, **kw):
        llamadas.append({"uid": uid, "inventory_override": kw.get("inventory_override"),
                         "consumed_override": kw.get("consumed_override")})
        return [{"name": "Pollo", "quantity": 1, "unit": "lb"}]

    monkeypatch.setattr(sc, "get_shopping_list_delta", _delta)
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", lambda *a, **kw: ([], []))
    monkeypatch.setattr(sc, "_build_hybrid_shopping_list", lambda a, b: a or b or [])
    monkeypatch.setattr(sc, "run_shopping_coherence_guard", lambda *a, **kw: [])
    return llamadas


# ─────────────────────────────── (a) el invitado sale con lista

def test_el_invitado_conserva_su_lista_tras_el_swap(estado_invitado, constructor):
    from graph_orchestrator import _recompute_aggregates_after_swap
    asyncio.run(_recompute_aggregates_after_swap(estado_invitado))
    plan = estado_invitado["plan_result"]
    for clave in ("aggregated_shopping_list", "aggregated_shopping_list_weekly",
                  "aggregated_shopping_list_biweekly", "aggregated_shopping_list_monthly"):
        assert plan.get(clave), f"{clave} vacía: el invitado se queda sin lista tras el swap"
    assert constructor and all(l["uid"] is None for l in constructor), "el invitado no tiene uid que descontar"
    assert all(l["inventory_override"] == [] and l["consumed_override"] == [] for l in constructor), \
        "sin inventario que descontar, los overrides van vacíos (igual que en assemble)"


# ─────────────────────────────── (b) las dos ramas de invitado son la misma

def test_las_dos_ramas_de_invitado_usan_el_mismo_builder():
    src = _src(_GO_PY)
    assert src.count("inventory_override=[], consumed_override=[]") >= 4, \
        "las dos ramas de invitado (assemble y re-agregación) construyen con overrides vacíos"
    i = src.index("async def _recompute_aggregates_after_swap")
    cuerpo = src[i:i + 9000]
    assert "P1-GUEST-SHOPPING" in cuerpo, "la re-agregación debe decir de dónde viene la regla del invitado"
    assert "aggr_list_7, aggr_list_15, aggr_list_30 = [], [], []" in cuerpo, "el fallback sigue existiendo"
    assert re.search(r"except Exception as _e_guest_reagg", cuerpo), "el fallback es best-effort, como en assemble"


def test_solo_el_pipeline_llama_a_la_reagregacion():
    """`/swap-meal/persist` vacía las 4 `aggregated_shopping_list*` A PROPÓSITO (el frontend las recalcula).
    Si alguien enruta ese endpoint por aquí, esta rama se las volvería a llenar y las dos decisiones chocarían."""
    llamadas = [n for n, l in enumerate(_src(_GO_PY).splitlines(), 1)
                if "await _recompute_aggregates_after_swap(" in l]
    assert len(llamadas) == 2, f"call sites en GO: {llamadas} (eran 2, ambos del pipeline tras el swap)"
    assert "_recompute_aggregates_after_swap" not in _src(_BACKEND / "routers" / "plans.py")


# ─────────────────────────────── (c) el usuario real no cambia

def test_el_usuario_con_cuenta_sigue_descontando_su_inventario(constructor):
    estado = {"plan_result": {"days": [{"day": 1, "meals": []}], "calc_household_multiplier": 1.0},
              "form_data": {"user_id": "u-real", "groceryDuration": "weekly"}}
    from graph_orchestrator import _recompute_aggregates_after_swap
    asyncio.run(_recompute_aggregates_after_swap(estado))
    assert constructor and all(l["uid"] == "u-real" for l in constructor)
    assert all(l["inventory_override"] == [] for l in constructor), "el stub devuelve inventario vacío, pero LO PASA"


# ─────────────────────────────── (d) si el builder revienta, no se rompe la entrega

def test_si_el_builder_revienta_la_lista_queda_vacia_y_se_dice(estado_invitado, monkeypatch, caplog):
    import shopping_calculator as sc

    def _boom(*a, **kw):
        raise RuntimeError("builder caído")

    monkeypatch.setattr(sc, "get_shopping_list_delta", _boom)
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", lambda *a, **kw: ([], []))
    monkeypatch.setattr(sc, "_build_hybrid_shopping_list", lambda a, b: a or b or [])
    monkeypatch.setattr(sc, "run_shopping_coherence_guard", lambda *a, **kw: [])
    from graph_orchestrator import _recompute_aggregates_after_swap
    with caplog.at_level("WARNING"):
        asyncio.run(_recompute_aggregates_after_swap(estado_invitado))
    assert estado_invitado["plan_result"].get("aggregated_shopping_list") == []
    assert any("P1-GUEST-SHOPPING" in r.message for r in caplog.records), "un fallo silencioso es el bug de origen"


# ─────────────────────────────── (e) docs y marker

def test_docs_y_marker():
    doc = _src(_BACKEND / "docs" / "coherence_surfaces_table.md")
    assert "P1-PLAN-LOTE-66" in doc and "invitado" in doc.lower()
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src(_BACKEND / "app.py"), re.M)
    assert m and int(m.group(1)) >= 66 and m.group(2) >= "2026-09-16"
