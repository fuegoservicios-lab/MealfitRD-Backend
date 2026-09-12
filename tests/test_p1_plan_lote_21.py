# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-21 · 2026-09-12] El allocator mínimo del horizonte (E6 · ARQ30-P1-03) se enciende por defecto.

El lote 20 lo midió con el knob `MEALFIT_HORIZON_VIABLE_FAMILY` apagado (4.311 franjas del horizonte sin candidato,
4.211 rescatables por otra familia) y encendido (4.311 → 100, todas huecos de biblioteca). El dueño delegó la
decisión («enciende el knob por mí»): el default pasa a ON en el código, sin tocar el `.env` del VPS.

Tres cosas que este fichero ancla:

  · el default vivo es ON; `MEALFIT_HORIZON_VIABLE_FAMILY=0` lo apaga sin redeploy y devuelve el blueprint
    byte-idéntico al anterior (la promesa del lote 20 sigue valiendo, sólo que ahora hay que pedirla);
  · no hace falta cohorte por usuario: los runs en vuelo conservan su blueprint (`_run_blueprint_for_plan`), así
    que sólo los runs NUEVOS ven la reasignación;
  · el medidor `scripts/measure_horizon_slots.py` fija el knob EXPLÍCITAMENTE en las dos direcciones. Antes medía el
    round-robin puro «sin tocar el knob» — es decir, con el default — y al cambiar el default habría medido el
    allocator creyendo medir el diagnóstico. *Un medidor que hereda el default mide cosas distintas cuando el
    default cambia.*
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402

_EFF = {"diet": {"type": "balanced", "allergies": [], "exclusions": []}, "market_country": "DO",
        "recurrence": {"global_mode": "balanced"}, "shopping": {"main_cycle_days": 7}, "policy_hash": "p"}


def _registry_falso(monkeypatch, sin_plato):
    import dish_registry as dr

    def _tc(country, slot, family=None, *, k=6, **kw):
        if family is not None and (slot, family) in sin_plato:
            return []
        return [{"template_id": f"tpl_{slot}_{family or 'any'}_{i}", "name": f"{slot} {family or 'any'} {i}"}
                for i in range(min(k, 3))]

    monkeypatch.setattr(dr, "template_candidates", _tc)
    monkeypatch.setattr(dr, "registry_hash", lambda c: "abcd1234")
    monkeypatch.setattr(dr, "registry_snapshot_version", lambda: "v-test")


def _bp(monkeypatch, sin_plato=(), dias=3):
    _registry_falso(monkeypatch, set(sin_plato))
    return horizon.build_blueprint(dict(_EFF), total_days=dias, meals_per_day=3)


# ────────────────────────────────────────────────────────────── el default vivo

def test_el_knob_es_ON_por_defecto_y_se_apaga_por_entorno(monkeypatch):
    monkeypatch.delenv("MEALFIT_HORIZON_VIABLE_FAMILY", raising=False)
    assert horizon.viable_family_enabled() is True
    for apagado in ("0", "false", "off", "no"):
        monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", apagado)
        assert horizon.viable_family_enabled() is False, apagado
    monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", "1")
    assert horizon.viable_family_enabled() is True


def test_sin_entorno_el_blueprint_reasigna_la_familia_del_dia_con_hueco(monkeypatch):
    """Lo que el lote 20 exigía pedir con `viable="true"`, ahora pasa solo."""
    monkeypatch.delenv("MEALFIT_HORIZON_VIABLE_FAMILY", raising=False)
    bp0 = _bp(monkeypatch)
    fam0 = bp0["days"][0]["protein"]
    bp = _bp(monkeypatch, sin_plato={("lunch", fam0)})
    reg = bp["registry"]
    assert reg.get("viable_family") is True
    assert reg["family_reassignments"] and reg["family_reassignments"][0]["from"] == fam0
    assert bp["days"][0]["protein"] != fam0
    assert "0:lunch" in reg["candidates"], "la franja rescatada tiene candidatos"
    assert "empty_slots" not in reg, "con el allocator no queda franja rescatable sin rescatar"


def test_apagado_por_entorno_vuelve_al_blueprint_byte_identico_del_lote_20(monkeypatch):
    """La promesa del lote 20 («apagado = byte-idéntico salvo el diagnóstico») sigue en pie: sólo cambió quién
    tiene que pedirla."""
    monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", "0")
    bp0 = _bp(monkeypatch)
    fam0 = bp0["days"][0]["protein"]
    bp = _bp(monkeypatch, sin_plato={("lunch", fam0)})
    reg = bp["registry"]
    assert "viable_family" not in reg and "family_reassignments" not in reg
    assert bp["days"][0]["protein"] == fam0
    assert reg["empty_slots"] == [{"day_index": 0, "slot": "lunch", "family": fam0, "culture_country": "DO",
                                   "rescuable_by_family": True}]


def test_sin_huecos_el_default_ON_no_mueve_el_hash_salvo_la_marca(monkeypatch):
    """Un blueprint sin huecos sólo gana `viable_family: True`; ningún día cambia de familia."""
    monkeypatch.setenv("MEALFIT_HORIZON_VIABLE_FAMILY", "0")
    off = _bp(monkeypatch)
    monkeypatch.delenv("MEALFIT_HORIZON_VIABLE_FAMILY", raising=False)
    on = _bp(monkeypatch)
    assert [d["protein"] for d in on["days"]] == [d["protein"] for d in off["days"]]
    assert "family_reassignments" not in on["registry"]
    assert {k: v for k, v in on["registry"].items() if k != "viable_family"} == off["registry"]


# ────────────────────────────────────────────────────────────── por qué no hace falta cohorte

def test_los_runs_en_vuelo_conservan_su_blueprint():
    """`blueprint_for_plan` devuelve el blueprint GUARDADO del run antes de reconstruir nada: encender el knob no
    cambia la rebanada de un chunk cuyo run ya nació."""
    src = inspect.getsource(horizon.blueprint_for_plan)
    assert "_run_blueprint_for_plan(plan_id)" in src
    assert src.index("_run_blueprint_for_plan(plan_id)") < src.index("build_blueprint(")
    doc = horizon.viable_family_enabled.__doc__ or ""
    assert "P1-PLAN-LOTE-21" in doc and "_run_blueprint_for_plan" in doc


# ────────────────────────────────────────────────────────────── el medidor no hereda el default

def test_el_medidor_fija_el_knob_en_las_dos_direcciones():
    src = (_BACKEND / "scripts" / "measure_horizon_slots.py").read_text(encoding="utf-8")
    assert 'os.environ["MEALFIT_HORIZON_VIABLE_FAMILY"] = "true" if args.viable else "false"' in src
    assert "P1-PLAN-LOTE-21-MEDIR-EXPLICITO" in src
    assert 'if args.viable:\n        os.environ["MEALFIT_HORIZON_VIABLE_FAMILY"] = "true"' not in src


# ────────────────────────────────────────────────────────────── docs, plan, marker

def test_docs_plan_y_marker():
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_HORIZON_VIABLE_FAMILY` | `True` (nació `False` en el lote 20; ON desde `P1-PLAN-LOTE-21`" in knobs
    arq = (_BACKEND / "docs" / "arq30_e5_e7_diseno_canario.md").read_text(encoding="utf-8")
    assert "ON por defecto desde `P1-PLAN-LOTE-21`" in arq and "**Decidido (2026-09-12, `P1-PLAN-LOTE-21`)**" in arq
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "knob ON desde `P1-PLAN-LOTE-21`" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8", errors="ignore")
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "([^"]+)"', app)
    assert m and m.group(1).split("·")[-1].strip() >= "2026-09-12"
    assert "P1-PLAN-LOTE-21" in (_BACKEND / "horizon.py").read_text(encoding="utf-8")
