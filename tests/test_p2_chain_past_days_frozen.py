"""[P2-CHAIN-PAST-DAYS-FROZEN · 2026-08-04] (audit solver+seeder v7 · P3) Los días que el
usuario YA COCINÓ son registro histórico: ningún pase global del finalize-chain los reescribe.

Colisión detectada en el review final de la tanda P2: `P2-BUDGET-CONVERGENCE-FUTURE-ONLY`
ventaneó la convergencia de presupuesto a los días futuros… pero la convergencia INVOCA el
chain de calidad (`apply_plan_quality_finalize_chain`) justo detrás, y ese chain sigue barriendo
el plan ENTERO. Los pases que mueven gramos —el band-closer de proteína, el band-closer all-4
(rebalance ×[0.3, 2.5] por pase, `passes=3`), el re-cap de realismo iterado, el reconcile
display↔raw, el polish y el cap de condimentos— no distinguen «día 3, cocinado el martes» de
«día 12, todavía por comprar».

Consecuencia real (no teórica): `consultar_dia_del_plan` (la tool con la que el coach responde
"¿qué comí el martes?") y el índice de calidad leen un historial que el sistema reescribe
DESPUÉS de cocinado. El usuario cocinó 150 g de arroz y el plan persistido dice 210 g porque un
rebalance de tres días más tarde necesitaba carbos en otra parte.

Diseño (un solo punto, no N ventanas que drifteen):
  · snapshot profundo de los días PASADOS ANTES del primer pase que mueve cantidades;
  · restauración byte-a-byte DESPUÉS del último;
  · los recomputes de SOLO-LECTURA que estampan métricas (band score + panel de micros, que ya
    cierran el chain) quedan detrás de la restauración → miden el estado final REAL.

Fecha de un día: se reusa la derivación de `P2-BUDGET-CONVERGENCE-FUTURE-ONLY`
(`_budget_future_days_window`) — un segundo derivador habría driftado del primero.

Fail-open explícito, en dos capas:
  · sin NINGUNA `date` estampada en los días ⇒ cero días congelados (todo cuenta como futuro);
  · el día de HOY cuenta como FUTURO (todavía se puede cocinar).

La primera capa NO es cosmética: el merge T1 del chunk worker pasa por el chain una VISTA
PARCIAL del plan (`P0-CHUNK-CHAIN-SCOPED`) que lleva el `grocery_start_date` del plan completo
pero solo los días NUEVOS de la semana N. Derivar sus fechas por `grocery_start_date + índice`
los fecharía como los días 1..k del plan (pasado) y congelaría el chunk entero: el chain se
volvería un no-op silencioso para ~7/8 de los días de un plan mensual. Por eso el freeze exige
un ancla ESTAMPADA y no acepta la del tier `grocery_start_date`.

Rollback sin redeploy: MEALFIT_CHAIN_PAST_DAYS_FROZEN=false ⇒ chain byte-idéntico al previo.

tooltip-anchor: P2-CHAIN-PAST-DAYS-FROZEN
"""
from __future__ import annotations

import copy
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_DBP = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")


def _fn_body(src: str, name: str) -> str:
    """Cuerpo textual de una función top-level (hasta el siguiente `def `/`class ` a col 0)."""
    i = src.index(f"def {name}(")
    j = len(src)
    for _tok in ("\ndef ", "\nclass "):
        k = src.find(_tok, i + 1)
        if k != -1:
            j = min(j, k)
    return src[i:j]


def _hoy_rd():
    """Mismo 'hoy' que el helper de producción: UTC-4 (convención `rd_today`)."""
    return (datetime.now(timezone.utc) - timedelta(hours=4)).date()


# ═════════════════════ 1 · Parser: contrato estructural ═════════════════════

def test_parser_marker_inline_en_ambos_archivos():
    assert "[P2-CHAIN-PAST-DAYS-FROZEN · 2026-08-04]" in _DBP, (
        "falta el marker inline en db_plans.py (donde vive el freeze)"
    )
    assert "P2-CHAIN-PAST-DAYS-FROZEN" in _GO, (
        "falta el marker en graph_orchestrator.py (donde viven el knob y el oráculo de fechas)"
    )


def test_parser_knob_registrado_via_env_bool_default_true():
    """Es una corrección de INTEGRIDAD DEL HISTORIAL, no un experimento: nace encendido.
    El knob existe por si sorprende en producción, no para pilotarlo."""
    assert '_env_bool("MEALFIT_CHAIN_PAST_DAYS_FROZEN", True)' in _GO, (
        "el knob debe declararse con `_env_bool` (auto-registro en _KNOBS_REGISTRY), default True"
    )
    import graph_orchestrator as g
    assert isinstance(g.CHAIN_PAST_DAYS_FROZEN, bool)
    from graph_orchestrator import get_knobs_registry_snapshot
    assert "MEALFIT_CHAIN_PAST_DAYS_FROZEN" in get_knobs_registry_snapshot(), (
        "el knob no llegó al registro (¿se leyó con os.environ crudo?)"
    )


def test_parser_reusa_la_derivacion_de_fechas_de_la_convergencia():
    """Un segundo derivador de fechas driftaría del primero. El oráculo del freeze delega en
    `_budget_future_days_window` (SSOT de P2-BUDGET-CONVERGENCE-FUTURE-ONLY)."""
    blk = _fn_body(_GO, "frozen_past_day_indices")
    assert "_budget_future_days_window(" in blk, (
        "el oráculo del freeze debe REUSAR `_budget_future_days_window`, no duplicar la "
        "derivación de fechas (dos derivadores drifean)"
    )


def test_parser_el_freeze_exige_ancla_estampada():
    """Guard anti-regresión del chunk worker: la vista parcial del merge T1 lleva el
    `grocery_start_date` del plan completo. Sin este guard, sus días se fecharían como los
    días 1..k y el chain se apagaría para las semanas 2+."""
    blk = _fn_body(_GO, "frozen_past_day_indices")
    assert '"date"' in blk or "'date'" in blk, (
        "el oráculo debe exigir al menos una `date` ESTAMPADA antes de congelar nada "
        "(el tier `grocery_start_date` es inseguro sobre una vista parcial del plan)"
    )


def test_parser_snapshot_antes_del_primer_pase_que_mueve_gramos():
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_snap = body.find("P2-CHAIN-PAST-DAYS-FROZEN")
    assert i_snap != -1, "el freeze desapareció del shield pre-INSERT"
    i_fpc = body.index("_n, _summ = _fpc(")
    assert i_snap < i_fpc, (
        "el snapshot debe tomarse ANTES del primer pase que mueve cantidades "
        "(`finalize_plan_data_coherence`): si se toma después, ya se perdió el original"
    )


def test_parser_restauracion_despues_del_ultimo_pase_que_muta_dias():
    """El último pase que TOCA el contenido de un día es la concordancia número-sustantivo
    (`_fica`). Todo lo que viene detrás (detectores warn-only, stale-clear, band score, panel
    de micros) es lectura o metadata plan-level."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_fica = body.index("_fica(_pd)")
    i_restore = body.find("P2-CHAIN-PAST-DAYS-FROZEN", body.index("_n, _summ = _fpc("))
    assert i_restore != -1, "falta el bloque de restauración"
    assert i_fica < i_restore, (
        "la restauración debe ir DESPUÉS del último pase que muta días; si va antes, el pase "
        "siguiente vuelve a reescribir el historial"
    )


def test_parser_las_metricas_se_estampan_despues_de_la_restauracion():
    """La lección del repo: dos mediciones honestas que discrepan = bug. El band score y el
    panel de micros PERSISTIDOS tienen que medir el estado final (con pasados restaurados),
    no el intermedio."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_restore = body.index("P2-CHAIN-PAST-DAYS-FROZEN", body.index("_n, _summ = _fpc("))
    for _metrica in ("_rbs(_pd", "_rmr(_pd"):
        assert body.index(_metrica) > i_restore, (
            f"{_metrica} se estampa ANTES de restaurar los días pasados: la métrica "
            f"persistida mediría un estado intermedio que nunca existió"
        )


def test_parser_last_known_pfix_bumpeado():
    """Contrato de formato + floor de fecha (sin re-anclar el slug: el siguiente P-fix lo
    sobreescribe legítimamente — misma lección de `test_p2_caps_after_band_closer.py`)."""
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    _fecha = re.search(r"(\d{4}-\d{2}-\d{2})\s*$", m.group(1))
    assert _fecha, f"Marker sin fecha ISO al final: {m.group(1)!r}"
    assert _fecha.group(1) >= "2026-08-04", (
        f"Marker sospechosamente viejo: {m.group(1)!r} "
        f"(floor P2-CHAIN-PAST-DAYS-FROZEN · 2026-08-04)"
    )


# ═════════════════════ 2 · Funcional: el chain completo, offline ═════════════════════

class _StubDB:
    """DB offline. Espejo del dummy de `test_p2_caps_after_band_closer.py`: cero red, cero pool."""

    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


@pytest.fixture
def chain_offline(monkeypatch):
    """Chain real con la DB stub inyectada y la emisión de telemetría APAGADA (el refresh de
    banda escribe una fila en `pipeline_metrics`; este test no toca DB)."""
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_METRIC_FINAL_EMIT", False)
    import db_plans
    return db_plans


def _dia(n: int, fecha_iso, *, linea="150 g de arroz blanco"):
    return {
        "day": n,
        "day_name": f"Día {n}",
        "date": fecha_iso,
        "meals": [{
            "meal": "Almuerzo",
            "name": "Pollo guisado con arroz",
            "protein": 35, "carbs": 60, "fats": 12, "cals": 488,
            "ingredients": [linea, "120 g de pechuga de pollo"],
            "ingredients_raw": [linea, "120 g de pechuga de pollo"],
            "recipe": ["MISE EN PLACE: Pica el pollo.",
                       "EL TOQUE DE FUEGO: Guisa 20 min y hierve el arroz.",
                       "MONTAJE: Sirve caliente."],
        }],
    }


def _plan(n_pasados: int, n_futuros: int, *, con_fechas: bool = True) -> dict:
    hoy = _hoy_rd()
    days = []
    for k in range(n_pasados + n_futuros):
        f = (hoy + timedelta(days=k - n_pasados)).isoformat() if con_fechas else None
        d = _dia(k + 1, f)
        if not con_fechas:
            d.pop("date")
        days.append(d)
    return {
        "days": days,
        "macros": {"protein": "100g", "carbs": "200g", "fats": "60g"},
        "calories": "2000 kcal",
        "grocery_start_date": (hoy - timedelta(days=n_pasados)).isoformat(),
    }


def _lead_g(linea: str) -> float:
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", str(linea).lower())
    return float(m.group(1).replace(",", ".")) if m else -1.0


# 205 g y no un número redondo enorme a propósito: por encima de `LINE_GRAM_HARD_CAP` (600 g)
# el re-cap de realismo del propio chain (P2-CAPS-AFTER-BAND-CLOSER) recortaría la marca y el
# test mediría el cap, no el freeze.
_MARCA_CLOSER = "205 g de arroz blanco"


def _fake_band_closer(marca=_MARCA_CLOSER):
    """Fake del band-closer all-4: reproduce lo que hace el rebalance ×[0.3, 2.5] — reescribe
    la primera línea de CADA día (pasados incluidos, que es justo el defecto) y desplaza los
    macros del meal para que el band score intermedio sea distinguible del final."""
    def _fake(plan_data, form_data=None, db=None):
        n = 0
        for d in plan_data.get("days") or []:
            for m in d.get("meals") or []:
                for lista in ("ingredients", "ingredients_raw"):
                    ings = m.get(lista)
                    if isinstance(ings, list) and ings:
                        ings[0] = marca
                m["carbs"] = 400
                m["cals"] = 2000
                n += 1
        return n
    return _fake


def _sin_freeze(monkeypatch):
    """Neutraliza el freeze desde el oráculo (equivale a que el código no existiera)."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "frozen_past_day_indices", lambda *a, **k: [])


# ---- el caso central -------------------------------------------------------

def test_los_dias_pasados_salen_byte_identicos_y_los_futuros_procesados(
        chain_offline, monkeypatch):
    """6 días pasados (con `date`) + 9 futuros. El band-closer infla TODOS los días. Tras el
    chain completo, los 6 pasados son BYTE-IDÉNTICOS a la entrada y los 9 futuros sí se
    procesaron."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())

    plan = _plan(6, 9)
    entrada = copy.deepcopy(plan)
    chain_offline.apply_plan_quality_finalize_chain(plan)

    for i in range(6):
        assert plan["days"][i] == entrada["days"][i], (
            f"el día pasado #{i + 1} ({entrada['days'][i]['date']}) fue reescrito por el chain: "
            f"{plan['days'][i]!r}"
        )
    for i in range(6, 15):
        assert plan["days"][i] != entrada["days"][i], (
            f"el día futuro #{i + 1} NO se procesó: el freeze se pasó de ancho"
        )
        assert _lead_g(plan["days"][i]["meals"][0]["ingredients"][0]) != 150.0, (
            f"el día futuro #{i + 1} conserva los 150 g de la entrada: el band-closer no lo tocó"
        )
    assert len(plan["days"]) == 15


def test_control_sin_el_freeze_el_dia_pasado_si_se_reescribe(chain_offline, monkeypatch):
    """Control del propio test: si esto pasara con el freeze puesto, el fixture no mediría nada."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())
    _sin_freeze(monkeypatch)

    plan = _plan(6, 9)
    entrada = copy.deepcopy(plan)
    chain_offline.apply_plan_quality_finalize_chain(plan)
    assert plan["days"][0] != entrada["days"][0], (
        "sin freeze el defecto tiene que reaparecer intacto (día pasado reescrito)"
    )
    assert _lead_g(plan["days"][0]["meals"][0]["ingredients"][0]) != 150.0


def test_knob_off_es_byte_identico_al_chain_sin_freeze(chain_offline, monkeypatch):
    """Contrato de rollback: `MEALFIT_CHAIN_PAST_DAYS_FROZEN=false` ⇒ el plan resultante es
    IDÉNTICO al que produce un chain donde el freeze no existe."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())

    base = _plan(6, 9)

    plan_off = copy.deepcopy(base)
    monkeypatch.setattr(g, "CHAIN_PAST_DAYS_FROZEN", False)
    chain_offline.apply_plan_quality_finalize_chain(plan_off)

    plan_ref = copy.deepcopy(base)
    _sin_freeze(monkeypatch)
    chain_offline.apply_plan_quality_finalize_chain(plan_ref)

    assert plan_off == plan_ref, (
        "con el knob apagado el chain debe quedar byte-idéntico al comportamiento previo"
    )


def test_plan_sin_fechas_se_procesa_entero(chain_offline, monkeypatch):
    """Fail-open capa 1: sin `date` estampada en NINGÚN día no hay ancla → todo es futuro."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())

    plan = _plan(6, 9, con_fechas=False)
    entrada = copy.deepcopy(plan)
    chain_offline.apply_plan_quality_finalize_chain(plan)
    for i, d in enumerate(plan["days"]):
        assert d != entrada["days"][i], (
            f"día #{i + 1} congelado sin ancla de fechas: el fail-open se rompió"
        )


def test_insert_de_plan_nuevo_es_no_op_exacto(chain_offline, monkeypatch):
    """En un plan recién generado TODOS los días son futuros ⇒ el freeze no toca nada."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())

    base = _plan(0, 7)

    plan_con = copy.deepcopy(base)
    chain_offline.apply_plan_quality_finalize_chain(plan_con)

    plan_ref = copy.deepcopy(base)
    _sin_freeze(monkeypatch)
    chain_offline.apply_plan_quality_finalize_chain(plan_ref)

    assert plan_con == plan_ref, "el freeze debe ser no-op EXACTO en el INSERT de un plan nuevo"


def test_el_dia_de_hoy_cuenta_como_futuro(chain_offline, monkeypatch):
    """Hoy todavía se puede cocinar: el día de hoy NO se congela."""
    import graph_orchestrator as g
    plan = _plan(3, 4)  # days[3] es hoy
    idx = g.frozen_past_day_indices(plan, plan["days"])
    assert idx == [0, 1, 2], f"esperados los 3 días anteriores a hoy, se obtuvo {idx}"


def test_vista_parcial_del_chunk_worker_no_se_congela():
    """Guard anti-regresión P0-CHUNK-CHAIN-SCOPED: la vista del merge T1 lleva el
    `grocery_start_date` del plan completo y solo los días NUEVOS (sin `date` si el plan es
    viejo). Fecharlos por `grocery_start_date + índice` los pondría en el pasado y apagaría el
    chain para ~7/8 de los días de un plan mensual."""
    import graph_orchestrator as g
    hoy = _hoy_rd()
    vista = {
        "days": [{"day": n, "meals": []} for n in (8, 9, 10, 11, 12, 13)],
        "macros": {"protein": "100g", "carbs": "200g", "fats": "60g"},
        "calories": "2000 kcal",
        "grocery_start_date": (hoy - timedelta(days=20)).isoformat(),
    }
    assert g.frozen_past_day_indices(vista, vista["days"]) == [], (
        "la vista parcial del chunk worker no puede congelarse: sus días son futuros aunque "
        "el `grocery_start_date` del plan sea de hace 20 días"
    )


def test_el_band_score_persistido_mide_el_estado_final(chain_offline, monkeypatch):
    """Dos mediciones honestas que discrepan = bug. El score y los macros entregados que se
    PERSISTEN tienen que coincidir con recomputarlos sobre el plan tal como queda."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())

    plan = _plan(6, 9)
    chain_offline.apply_plan_quality_finalize_chain(plan)

    persistido = plan.get("clinical_band_score")
    assert isinstance(persistido, dict) and persistido.get("score") is not None, (
        "el chain debe seguir persistiendo `clinical_band_score`"
    )
    recomputado = g.compute_clinical_band_score(plan, {})
    assert abs(float(persistido["score"]) - float(recomputado["score"])) < 1e-9, (
        f"el band score persistido ({persistido['score']}) midió un estado intermedio: "
        f"recomputado sobre el plan final da {recomputado['score']}"
    )

    entregados = copy.deepcopy(plan.get("delivered_macros"))
    g.refresh_delivered_macros(plan)
    assert plan.get("delivered_macros") == entregados, (
        "`delivered_macros` persistido mide el estado intermedio (pre-restauración)"
    )


def test_la_lista_de_dias_conserva_identidad_y_tamano(chain_offline, monkeypatch):
    """El adapter promete mutación in-place (`plan['days'] is days_ref`, anclado en
    test_p0_band_pre_review). La restauración reemplaza ELEMENTOS, nunca la lista."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _fake_band_closer())

    plan = _plan(6, 9)
    ref = plan["days"]
    chain_offline.apply_plan_quality_finalize_chain(plan)
    assert plan["days"] is ref and len(ref) == 15


def test_oraculo_fail_safe_ante_entradas_absurdas():
    import graph_orchestrator as g
    assert g.frozen_past_day_indices(None, None) == []
    assert g.frozen_past_day_indices({}, []) == []
    assert g.frozen_past_day_indices({}, [{"day": 1}]) == []
    assert g.frozen_past_day_indices({}, ["no soy un dict"]) == []
