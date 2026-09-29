# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-817 · 2026-09-29] Dos pendientes de la renovación con política (lote 811).

(A) La doc del día del ciclo decía dos cosas falsas: que el defecto «es raro» (exige que mueran todos los bloques) y que
    en la renovación semanal «el índice es exacto». El defecto ya existe en TODO el sistema (el rebase de offsets cuenta
    desde el ancla móvil); llevar la política lo extiende a las renovaciones y rellenos de 15/30 días de compra única.
(B) La renovación lleva la política CONGELADA de la creación (anclas), mientras el perfil puede haber ganado una alergia,
    un rechazo o un cambio de dieta. Un ancla que el guard rechaza es una regla insatisfacible: gasta reintentos. El
    snapshot de la renovación retira esas anclas con la MISMA puerta del guard y anota la retirada.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

_EFF = {
    "policy_hash": "hash-del-plan-817",
    "recurrence": {"global_mode": "routine"},
    "food_anchors": [
        {"ingredient_id": "camarones", "name": "Camarones", "min_per_7d": 2, "max_per_7d": 3, "slots": ["lunch"]},
        {"ingredient_id": "huevo", "name": "Huevo", "min_per_7d": 5, "max_per_7d": 7, "slots": ["breakfast"]},
        {"ingredient_id": "pollo", "name": "Pollo", "min_per_7d": 2, "max_per_7d": 4, "slots": []},
    ],
    "diet": {"type": "balanced", "allergies": [], "exclusions": []},
    "budget": {"mode": "hard", "tier": "low", "status": "ok"},
}
_HP = {"age": 30, "gender": "female", "mainGoal": "lose_fat", "tzOffset": 0, "allergies": ["Ninguna"],
       "dislikes": [], "dietType": "balanced"}


@pytest.fixture(autouse=True)
def _politica_activa(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_POLICY_MODE", "enforce")
    for k in ("MEALFIT_REFILL_POLICY_ANCHORS_RECHECK", "MEALFIT_REFILL_CARRIES_POLICY",
              "MEALFIT_7D_ORPHAN_GAP_HTTP_REFILL"):
        monkeypatch.delenv(k, raising=False)


def _plan(total=7, n_dias=3, ancla=None):
    return {
        "grocery_start_date": ancla or (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat(),
        "generation_status": "complete",
        "total_days_requested": total,
        "days": [{"day": i, "meals": [{"name": f"Plato {i}"}]} for i in range(1, n_dias + 1)],
        "_plan_policy": {"requested": {}, "effective": json.loads(json.dumps(_EFF)), "relaxations": []},
    }


def _snap(hp, plan_data=None):
    import relleno_rolling as rr
    return rr.snapshot_relleno(hp=hp, user_id="user-817", chunk_count=4, ancla_iso="2026-09-29",
                               plan_data=plan_data if plan_data is not None else _plan(), previous_meals=["A"],
                               triggered_by="t", semanal=True)


def _nombres(fd):
    return sorted(a["name"] for a in fd["_plan_policy_effective"]["food_anchors"])


# ─────────────── (B) 1. la alergia nueva retira el ancla y la anota ───────────────
def test_ancla_camarones_con_alergia_nueva_a_mariscos_no_viaja_y_se_anota():
    plan = _plan()
    fd = _snap(dict(_HP, allergies=["Mariscos"]), plan)["form_data"]
    eff = fd["_plan_policy_effective"]
    assert _nombres(fd) == ["Huevo", "Pollo"], "antes: Camarones viajaba y el guard lo rechazaba en cada reintento"
    rel = eff["renewal_relaxations"]
    assert len(rel) == 1
    r = rel[0]
    assert r["reason_code"] == "anchor_conflicts_allergy" and r["rank"] == 1 and r["requested"] == "Camarones"
    assert r["applied"] is None and r["field"] == "food_anchors" and r["action"] == "applied"
    assert r["evidence"]["allergy"] == "Mariscos"
    assert set(r) >= {"field", "requested", "applied", "reason_code", "evidence", "rank", "action"}, "forma de F2"
    # la política cambió: el hash lo dice, y el padre queda para la trazabilidad
    assert eff["renewal_parent_policy_hash"] == "hash-del-plan-817"
    assert eff["policy_hash"] != "hash-del-plan-817"
    # el plan persistido NO se toca (el snapshot trabaja sobre una copia)
    assert [a["name"] for a in plan["_plan_policy"]["effective"]["food_anchors"]] == ["Camarones", "Huevo", "Pollo"]
    assert "renewal_relaxations" not in plan["_plan_policy"]["effective"]


def test_la_alergia_tecleada_en_otra_tambien_cuenta():
    """El generador une «Otra…» (`otherAllergies`) a la lista; el perfil guardado no: se usa la MISMA unión (con su
    regla P0-FORM-1: el centinela «Ninguna» descarta el texto libre, igual que en la generación)."""
    fd = _snap(dict(_HP, allergies=[], otherAllergies="mariscos"))["form_data"]
    assert "Camarones" not in _nombres(fd)
    fd = _snap(dict(_HP, allergies=["Ninguna"], otherAllergies="mariscos"))["form_data"]
    assert "Camarones" in _nombres(fd), "misma unión que el generador: «Ninguna» manda sobre el texto libre"


def test_rechazo_nuevo_retira_el_ancla_como_exclusion():
    fd = _snap(dict(_HP, dislikes=["Huevo"]))["form_data"]
    assert _nombres(fd) == ["Camarones", "Pollo"]
    r = fd["_plan_policy_effective"]["renewal_relaxations"][0]
    assert r["reason_code"] == "anchor_conflicts_exclusion" and r["rank"] == 4 and r["requested"] == "Huevo"
    assert r["evidence"]["exclusion"]


def test_dieta_nueva_vegetariana_retira_la_carne_y_el_marisco():
    fd = _snap(dict(_HP, dietType="vegetariana"))["form_data"]
    assert _nombres(fd) == ["Huevo"]
    rel = fd["_plan_policy_effective"]["renewal_relaxations"]
    assert sorted(r["requested"] for r in rel) == ["Camarones", "Pollo"]
    assert {r["reason_code"] for r in rel} == {"anchor_conflicts_diet"}
    assert {r["evidence"]["diet"] for r in rel} == {"vegetarian"}


# ─────────────── (B) 2. sin conflicto → idéntico; knob off → como el 811 ───────────────
def test_sin_conflicto_el_snapshot_es_identico_al_del_811(monkeypatch):
    hp = dict(_HP, allergies=["Maní"], dislikes=["Cilantro"])
    con = _snap(hp)
    monkeypatch.setenv("MEALFIT_REFILL_POLICY_ANCHORS_RECHECK", "false")
    sin = _snap(hp)
    assert json.dumps(con, sort_keys=True, ensure_ascii=False) == json.dumps(sin, sort_keys=True, ensure_ascii=False)
    eff = con["form_data"]["_plan_policy_effective"]
    assert eff["policy_hash"] == "hash-del-plan-817" and "renewal_relaxations" not in eff


def test_con_el_knob_apagado_el_ancla_viaja_como_en_el_811(monkeypatch):
    monkeypatch.setenv("MEALFIT_REFILL_POLICY_ANCHORS_RECHECK", "false")
    fd = _snap(dict(_HP, allergies=["Mariscos"]))["form_data"]
    eff = fd["_plan_policy_effective"]
    assert _nombres(fd) == ["Camarones", "Huevo", "Pollo"]
    assert eff["policy_hash"] == "hash-del-plan-817" and "renewal_relaxations" not in eff


def test_sin_politica_no_hay_nada_que_revisar(monkeypatch):
    monkeypatch.setenv("MEALFIT_REFILL_CARRIES_POLICY", "false")
    fd = _snap(dict(_HP, allergies=["Mariscos"]))["form_data"]
    assert "_plan_policy_effective" not in fd


def test_si_la_puerta_no_se_puede_cargar_queda_la_politica_del_811(monkeypatch, caplog):
    """Fail-open al snapshot del 811 (el guard de la generación sigue en pie); nunca se inventa otra tabla."""
    monkeypatch.setitem(sys.modules, "rechazos", None)
    with caplog.at_level(logging.WARNING):
        fd = _snap(dict(_HP, allergies=["Mariscos"]))["form_data"]
    assert _nombres(fd) == ["Camarones", "Huevo", "Pollo"]
    assert any("P1-PLAN-LOTE-817" in r.getMessage() for r in caplog.records)


def test_la_retirada_se_registra_en_el_log(caplog):
    with caplog.at_level(logging.WARNING):
        _snap(dict(_HP, allergies=["Mariscos"]))
    msgs = [r.getMessage() for r in caplog.records if "P1-PLAN-LOTE-817" in r.getMessage()]
    assert msgs and "Camarones" in msgs[0]


# ─────────────── (B) 3. la puerta es la del guard, no otra tabla ───────────────
def test_la_puerta_es_la_del_guard_y_no_hay_tabla_propia():
    src = (_BACKEND / "anclas_vigentes.py").read_text(encoding="utf-8")
    for puerta in ("_allergen_pool_item_banned", "_diet_pool_item_banned", "_scan_dislike_violations",
                   "profile_with_free_text", "_has_real_medical_flags"):
        assert puerta in src, puerta
    # ninguna tabla de alérgenos/dietas propia (lección P1-DIET-CANON-SSOT)
    assert "_ALLERGEN_CLASS_TOKENS" not in src and "_ALLERGEN_SYNONYMS" not in src
    assert not re.search(r"^\s*_?[A-Z_]{4,}\s*=\s*[\{\(]", src, re.M), "sin tablas a nivel de módulo"


def test_snapshot_relleno_llama_a_la_puerta():
    src = (_BACKEND / "relleno_rolling.py").read_text(encoding="utf-8")
    i = src.index("def snapshot_relleno(")
    cuerpo = src[i:]
    assert "anclas_vigentes" in cuerpo and "tooltip-anchor: P1-PLAN-LOTE-817" in src


# ─────────────── (B) 4. de punta a punta: la renovación del cron ───────────────
def _cursor(plan_data, hp):
    ultimo = {"q": ""}

    def _exec(sql, *a, **k):
        ultimo["q"] = " ".join(str(sql).split())

    def _uno():
        q = ultimo["q"]
        if "plan_mode" in q:
            return {"plan_mode": "plan", "plan_mode_changed_at": None}
        if "SELECT id FROM meal_plans" in q:
            return {"id": "plan-817"}
        if "health_profile" in q:
            return {"health_profile": dict(hp)}
        if "plan_data" in q:
            return {"plan_data": plan_data}
        if "en_vuelo" in q:
            return {"en_vuelo": 0}
        if "COUNT(*) AS cnt" in q:
            return {"cnt": 0}
        if "max_week" in q:
            return {"max_week": 2}
        if "chunk_kind" in q:
            return None
        return {}

    cur = MagicMock()
    cur.execute.side_effect = _exec
    cur.fetchone.side_effect = _uno
    cur.fetchall.return_value = []
    pool = MagicMock()
    conn = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    conn.transaction.return_value.__enter__.return_value = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return pool


def test_cron_renovacion_semanal_con_alergia_nueva_no_lleva_el_ancla():
    import cron_tasks
    from constants import CHUNK_MIN_FRESH_PANTRY_ITEMS
    ancla = (datetime.now(timezone.utc) - timedelta(days=15)).date().isoformat()
    plan = _plan(total=15, n_dias=2, ancla=ancla)
    inv = [f"Alimento {i}" for i in range(CHUNK_MIN_FRESH_PANTRY_ITEMS + 2)]
    with patch("cron_tasks._enqueue_plan_chunk") as enq, \
            patch("db_core.connection_pool", _cursor(plan, dict(_HP, allergies=["Mariscos"]))), \
            patch("db_inventory.get_user_inventory_net", return_value=inv):
        r = cron_tasks._background_shift_plan_for_user("user-817", 0)
    assert r is True and enq.call_count >= 1
    for c in enq.call_args_list:
        fd = c.args[5]["form_data"]
        assert "Camarones" not in _nombres(fd)
        assert fd["_plan_policy_effective"]["renewal_relaxations"][0]["reason_code"] == "anchor_conflicts_allergy"


# ─────────────── (A) la doc del día del ciclo dice lo que el sistema hace ───────────────
def _bloque_dia_del_ciclo() -> str:
    src = (_BACKEND / "relleno_rolling.py").read_text(encoding="utf-8")
    i = src.index("Aproximación conocida")
    j = src.index("tooltip-anchor: P1-PLAN-LOTE-811-DIA-DEL-CICLO")
    return src[i:j]


def _fila_f3() -> str:
    doc = (_BACKEND / "docs" / "plan_policy_f3.md").read_text(encoding="utf-8")
    return [ln for ln in doc.splitlines() if "relleno_rolling.snapshot_relleno" in ln][0]


@pytest.mark.parametrize("texto", [_bloque_dia_del_ciclo, _fila_f3], ids=["relleno_rolling", "plan_policy_f3"])
def test_la_doc_no_dice_que_la_renovacion_sea_exacta_ni_que_sea_raro(texto):
    """El revisor final del 811: la renovación NO tiene el índice exacto (el rebase de offsets lo desplaza como a todo
    bloque pendiente) y el defecto no es raro. Que nadie lo vuelva a escribir."""
    t = texto()
    assert not re.search(r"exact[oa]s?\b", t, re.I), "la renovación no es «exacta»"
    assert "Es raro" not in t and "es raro" not in t


@pytest.mark.parametrize("texto", [_bloque_dia_del_ciclo, _fila_f3], ids=["relleno_rolling", "plan_policy_f3"])
def test_la_doc_dice_de_donde_viene_el_defecto_y_donde_se_arregla(texto):
    t = texto()
    assert "rebase" in t, "el defecto ya existe en todo el sistema: el rebase de offsets"
    assert "816" in t, "el arreglo va en el lote 816"
    assert "_blueprint_slice" in t and "archivados" in t
    assert "durabilidad" in t and "Nevera real" in t, "durabilidad siempre; Nevera virtual solo con la real vacía/apagada"
    assert "creación" in t, "el default True se sostiene: la renovación queda igual que la creación"


# ─────────────── knob, marker, docs ───────────────
def test_knob_registrado_y_documentado():
    import anclas_vigentes as av
    from knobs import get_knobs_registry_snapshot
    av.recheck_activo()
    snap = get_knobs_registry_snapshot()
    nombres = set(snap) if isinstance(snap, dict) else {k.get("name") for k in snap}
    assert "MEALFIT_REFILL_POLICY_ANCHORS_RECHECK" in nombres
    assert "MEALFIT_REFILL_POLICY_ANCHORS_RECHECK" in (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    f3 = (_BACKEND / "docs" / "plan_policy_f3.md").read_text(encoding="utf-8")
    assert "MEALFIT_REFILL_POLICY_ANCHORS_RECHECK" in f3 and "anchor_conflicts_exclusion" in f3
    assert "[P1-PLAN-LOTE-817 · 2026-09-29]" in (_BACKEND / "anclas_vigentes.py").read_text(encoding="utf-8")
