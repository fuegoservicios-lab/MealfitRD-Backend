"""[P2-T2-PAST-DAYS-FROZEN · 2026-08-04] (audit solver+seeder v7 · P3) Los días que el usuario YA
COCINÓ son registro histórico: el **seam T2 del chunk worker** no los reescribe.

`P2-BUDGET-CONVERGENCE-FUTURE-ONLY` ventaneó las SUSTITUCIONES de la convergencia de presupuesto.
Pero todo lo que corre detrás de ellas en el mismo seam —el truth-up, `apply_update_macro_engine`
(rebalance ×[0.3, 2.5] + refine de 5 g) y el finalize chain entero— seguía barriendo `plan_data`
completo. Medido: 150 g de arroz en un día pasado salían del seam en 650 g. `consultar_dia_del_plan`
(la tool con la que el coach responde "¿qué comí el martes?") y el índice de calidad leen ese
historial reescrito después de cocinado.

═══ POR QUÉ EL FREEZE ES OPT-IN Y NO UN DEFAULT DEL CHAIN ═══

La primera versión de este fix (commit 023ddff) congelaba por fecha DENTRO de
`_finalize_plan_data_for_insert`, o sea en sus 6 superficies. El review lo tumbó por Critical, y
tenía razón: en una superficie de GENERACIÓN, "fecha pasada" NO significa "el usuario ya lo comió".
La `date` de un día se estampa desde el `_plan_start_date` ORIGINAL del snapshot de la cola, y toda
desviación del gate temporal es unidireccional hacia el pasado. Tres rutas alcanzables dejan días
RECIÉN GENERADOS con `date` pasada al llegar a `assemble-tail`:

  · deferral learning-ready 12h ×2 (rutinario en usuarios que no registran) ⇒ −1 día;
  · `/retry-chunk` sin guard de ventana ⇒ el chunk entero;
  · resume de un freeze ⇒ hasta −30 días.

Con el freeze global, esos días se persistían EXACTAMENTE como los emitió la LLM: sin coherence
stack, sin `LINE_GRAM_HARD_CAP` (900 g de arroz sobreviven), sin band-closer, sin caps clínicos — y
el log presumiendo de "día ya cocinado restaurado" sobre días generados segundos antes. Un freeze
mal puesto no es una mejora tibia: apaga la calidad y encima miente en el log.

De ahí el contrato actual:
  1. `freeze_past_days` es un kwarg EXPLÍCITO keyword-only, **default False**;
  2. el ÚNICO caller que lo enciende es el seam T2 — única superficie que corre DÍAS después de la
     generación sobre días ya finalizados en su momento;
  3. el knob `MEALFIT_CHAIN_PAST_DAYS_FROZEN` (default True) es el KILL-SWITCH de ese opt-in, no
     su interruptor: con `false`, el kwarg en True se ignora.

El snapshot del seam se toma ANTES de `apply_update_macro_engine` (si se tomara después, el freeze
conservaría fielmente lo que el caller acababa de reescribir — el hallazgo I2 del review), y el
rebuild de listas del caller corre DESPUÉS del restore, así que lista lo que el usuario tiene
apuntado. La restauración muta el dict del día in-place porque reasignar el elemento de la lista
es INERTE sobre una vista (hallazgo I1).

tooltip-anchor: P2-T2-PAST-DAYS-FROZEN
"""
from __future__ import annotations

import ast
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
_CT = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")


_TREES = {}


def _fn_body(src: str, name: str) -> str:
    """Fuente EXACTA de una función top-level, vía AST.

    No se usa el clásico "hasta el siguiente `\\ndef `": `apply_budget_convergence_for_days` va
    seguida de la tabla module-level `_BUDGET_DRIVER_FAMILIES`, así que ese recorte se tragaba
    ~3 KB de otra cosa y los tests de ORDEN comparaban índices de funciones distintas (lo cazó
    el propio test al fallar con un `apply_update_macro_engine` que no era el suyo)."""
    tree = _TREES.get(id(src))
    if tree is None:
        tree = _TREES[id(src)] = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"función top-level {name!r} no encontrada")


def _fn_node(src: str, name: str):
    tree = _TREES.get(id(src))
    if tree is None:
        tree = _TREES[id(src)] = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"función top-level {name!r} no encontrada")


def _call_lineno(node, fname: str) -> int:
    """Línea de la ÚNICA llamada a `fname` dentro de `node`.

    Sobre el AST y no sobre el texto: los comentarios y docstrings de este fix explican
    literalmente cuándo NO usar el freeze, así que un `str.index` encontraba la PROSA antes
    que el código y los tests de orden comparaban un comentario contra una llamada."""
    hits = [n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == fname]
    assert len(hits) == 1, f"se esperaba 1 llamada a {fname}(), hay {len(hits)}"
    return hits[0]


def _freeze_kwarg_callsites(src: str, fichero: str) -> list:
    """Callsites REALES que pasan `freeze_past_days` (con el valor que sea), vía AST.

    Dos decisiones:
      · AST y no `re.finditer`: la regex contaba también las menciones en PROSA (el bloque del
        knob y los dos docstrings explican precisamente cuándo NO usarlo) — un comentario que
        documenta algo no es una violación de ese algo, lección ya pagada en este repo.
      · se cuenta el kwarg PRESENTE, no `=True`: el seam lo pasa como `_future_only` (una
        variable, para que el rollback del barrido completo revierta el seam entero), así que
        anclar el literal habría dejado de ver el único callsite legítimo… y de vigilar a un
        futuro callsite de generación que lo pasara con una variable."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "freeze_past_days":
                continue
            # `freeze_past_days=freeze_past_days` es la PROPAGACIÓN del adapter al shield, no
            # una superficie pidiendo freeze. Contarla haría imposible cumplir el invariante.
            if isinstance(kw.value, ast.Name) and kw.value.id == "freeze_past_days":
                continue
            out.append((fichero, node.lineno))
    return out


def _hoy_rd():
    """Mismo 'hoy' que el helper de producción: UTC-4 (convención `rd_today`)."""
    return (datetime.now(timezone.utc) - timedelta(hours=4)).date()


# ═══════════ 1 · Parser: el freeze es opt-in y solo el seam T2 lo enciende ═══════════

def test_parser_marker_inline_en_ambos_archivos():
    assert "[P2-T2-PAST-DAYS-FROZEN · 2026-08-04]" in _DBP, "falta el marker en db_plans.py"
    assert "[P2-T2-PAST-DAYS-FROZEN · 2026-08-04]" in _GO, "falta el marker en graph_orchestrator.py"


@pytest.mark.parametrize("fn", ["_finalize_plan_data_for_insert", "apply_plan_quality_finalize_chain"])
def test_parser_kwarg_opt_in_default_false(fn):
    """El default en False es lo que impide el Critical: una superficie que no pide freeze no
    congela. Y keyword-only, para que nadie lo pase por error copiando otra llamada."""
    i = _DBP.index(f"def {fn}(")
    sig = _DBP[i:_DBP.index("\n    \"\"\"", i)]
    assert "freeze_past_days: bool = False" in sig, (
        f"{fn} debe declarar `freeze_past_days: bool = False`. Sin el default en False, las "
        f"superficies de generación vuelven a congelar días recién generados."
    )
    assert sig.index("*,") < sig.index("freeze_past_days"), f"{fn}: debe ser keyword-only"


def test_parser_el_adapter_propaga_el_kwarg_al_shield():
    blk = _fn_body(_DBP, "apply_plan_quality_finalize_chain")
    assert "freeze_past_days=freeze_past_days" in blk, (
        "el adapter debe propagar el kwarg al shield (si no, el seam T2 pide freeze y no lo obtiene)"
    )


def test_parser_solo_el_seam_t2_pide_el_freeze():
    """ANTI-C1 estructural: UN único callsite en todo el backend productivo pasa
    `freeze_past_days`, y vive en el seam T2. Si aparece un segundo, el review vuelve a empezar."""
    encendidos = []
    for _nombre, _src in (("graph_orchestrator.py", _GO), ("db_plans.py", _DBP),
                          ("cron_tasks.py", _CT), ("app.py", _APP)):
        encendidos += _freeze_kwarg_callsites(_src, _nombre)
    assert len(encendidos) == 1 and encendidos[0][0] == "graph_orchestrator.py", (
        f"se esperaba EXACTAMENTE 1 callsite pasando `freeze_past_days` (el seam T2); "
        f"encontrados {len(encendidos)}: {encendidos}"
    )
    seam = _fn_node(_GO, "apply_budget_convergence_for_days")
    assert encendidos[0][1] in {n.lineno for n in ast.walk(seam) if hasattr(n, "lineno")}, (
        "el único callsite con freeze debe vivir en `apply_budget_convergence_for_days`"
    )


def test_el_freeze_del_seam_cuelga_tambien_del_rollback_del_barrido_completo():
    """Dos knobs sobre la MISMA pregunta en el MISMO seam no pueden contradecirse.

    `MEALFIT_BUDGET_CONVERGENCE_FUTURE_ONLY=false` significa "barre el plan entero". Si el
    freeze siguiera puesto, las sustituciones caerían sobre los días pasados y el restore las
    borraría acto seguido: un knob de rollback que ya no revierte lo que dice revertir (lo cazó
    `test_p2_budget_convergence_future_only::test_knob_off_restaura_el_barrido_completo`)."""
    seam = _fn_body(_GO, "apply_budget_convergence_for_days")
    assert "if _future_only else nullcontext()" in seam, (
        "el guard de historial del seam debe apagarse cuando el operador pide barrido completo"
    )
    assert "freeze_past_days=_future_only" in seam, (
        "y el chain debe recibir el mismo valor, no un True literal"
    )


@pytest.mark.parametrize("superficie", [
    "assemble-tail", "assemble-budget-convergence", "chunk-T1", "post-review-patch",
])
def test_parser_ninguna_superficie_de_generacion_pide_freeze(superficie):
    """Se comprueba sobre la LLAMADA real (ventana alrededor del literal de `surface=`), no
    sobre el fichero entero."""
    for _src in (_GO, _CT):
        for m in re.finditer(re.escape(superficie), _src):
            ventana = _src[max(0, m.start() - 400): m.start() + 400]
            if "_apqfc" not in ventana and "apply_plan_quality_finalize_chain" not in ventana:
                continue
            assert "freeze_past_days" not in ventana, (
                f"la superficie de GENERACIÓN {superficie!r} pide freeze. Sus días pueden llevar "
                f"`date` pasada recién generados (deferral 12h ×2 / retry-chunk / resume de "
                f"freeze) y se persistirían crudos, sin coherence stack ni caps."
            )


def test_parser_knob_es_kill_switch_no_interruptor():
    """El knob se lee en `snapshot_past_days` (que decide si se ACTÚA), no en el oráculo (que
    solo responde una pregunta de calendario)."""
    assert '_env_bool("MEALFIT_CHAIN_PAST_DAYS_FROZEN", True)' in _GO, (
        "el knob debe declararse con `_env_bool` (auto-registro en _KNOBS_REGISTRY), default True"
    )
    import graph_orchestrator as g
    assert isinstance(g.CHAIN_PAST_DAYS_FROZEN, bool)
    from graph_orchestrator import get_knobs_registry_snapshot
    assert "MEALFIT_CHAIN_PAST_DAYS_FROZEN" in get_knobs_registry_snapshot()

    assert "CHAIN_PAST_DAYS_FROZEN" not in _fn_body(_GO, "frozen_past_day_indices"), (
        "el oráculo debe ser PURO: el gate del kill-switch vive en `snapshot_past_days`"
    )
    assert "CHAIN_PAST_DAYS_FROZEN" in _fn_body(_GO, "snapshot_past_days")


def test_parser_reusa_la_derivacion_de_fechas_de_la_convergencia():
    assert "_budget_future_days_window(" in _fn_body(_GO, "frozen_past_day_indices"), (
        "el oráculo debe REUSAR `_budget_future_days_window`, no duplicar la derivación de fechas"
    )


def test_parser_el_freeze_exige_ancla_estampada():
    blk = _fn_body(_GO, "frozen_past_day_indices")
    assert '"date"' in blk or "'date'" in blk, (
        "el oráculo debe exigir al menos una `date` ESTAMPADA (el tier `grocery_start_date` es "
        "inseguro sobre cualquier vista parcial del plan)"
    )


def test_parser_m2_documenta_la_asimetria_de_parseo_de_fechas():
    """`date` se lee date-only (`_parse_date`) y el ancla con `_to_local_date`. Si algún día se
    persiste `date` como timestamptz, el corte se movería 24 h en silencio."""
    blk = _fn_body(_GO, "frozen_past_day_indices")
    assert "_parse_date" in blk and "_to_local_date" in blk, (
        "el docstring del oráculo debe dejar constancia de la asimetría date-only vs local-date"
    )


def test_parser_m1_nota_sobre_los_autofixes_fuera_del_chain():
    blk = _fn_body(_DBP, "apply_plan_quality_finalize_chain")
    assert "_protein_repeat_autofix" in blk and "_fruit_savory_autofix" in blk, (
        "falta la nota M1: esos dos autofixes corren FUERA del chain en la cola de assemble; "
        "quien los mueva dentro tiene que saber que no es parte de un cambio de freeze"
    )


def test_parser_restore_in_place_no_reasigna_el_elemento():
    """I1 del review: sobre una VISTA (lista nueva con los mismos dicts) una reasignación
    `days[i] = snapshot` escribe en la vista y jamás alcanza el plan — restore INERTE."""
    blk = _fn_body(_GO, "restore_past_days")
    assert ".clear()" in blk and ".update(" in blk, (
        "la restauración debe mutar el dict del día in-place (clear+update); reasignar el "
        "elemento de la lista es inerte sobre cualquier vista"
    )


# ═══════════ 2 · Parser: orden dentro del shield y dentro del seam ═══════════

def test_parser_snapshot_antes_del_primer_pase_que_mueve_gramos():
    shield = _fn_node(_DBP, "_finalize_plan_data_for_insert")
    assert _call_lineno(shield, "_spd") < _call_lineno(shield, "_fpc"), (
        "el snapshot del chain debe tomarse ANTES de `finalize_plan_data_coherence`"
    )


def test_parser_restauracion_despues_del_ultimo_pase_y_antes_de_las_metricas():
    """El último pase que toca el contenido de un día es `_fica`. Detrás solo hay lectura y
    metadata plan-level — y ahí es donde tienen que caer las métricas."""
    shield = _fn_node(_DBP, "_finalize_plan_data_for_insert")
    i_restore = _call_lineno(shield, "_rpd_frz")
    assert _call_lineno(shield, "_fica") < i_restore, (
        "restaurar antes del último pase que muta días dejaría que el siguiente reescriba"
    )
    for _metrica in ("_rbs", "_rmr"):
        assert _call_lineno(shield, _metrica) > i_restore, (
            f"{_metrica}() se estampa ANTES de restaurar: la métrica persistida mediría un "
            f"estado intermedio que nunca existió"
        )


def test_parser_el_seam_congela_antes_del_motor_de_macros():
    """I2 del review: `apply_update_macro_engine` corre sobre el plan ENTERO sin ventana de
    fechas. Si el snapshot se tomara después, el freeze conservaría fielmente lo que el caller
    acababa de reescribir."""
    seam = _fn_node(_GO, "apply_budget_convergence_for_days")
    i_guard = _call_lineno(seam, "frozen_past_days")
    withs = [n for n in ast.walk(seam) if isinstance(n, ast.With)]
    assert len(withs) == 1, "el seam debe abrir exactamente un bloque `with` (el del historial)"
    dentro = {n.lineno for n in ast.walk(withs[0]) if hasattr(n, "lineno")}
    assert i_guard < withs[0].lineno, (
        "el guard se construye ANTES de abrir el bloque (se envuelve en `nullcontext` cuando el "
        "operador pide barrido completo)"
    )

    assert _call_lineno(seam, "apply_update_macro_engine") in dentro, (
        "el motor de macros —que NO tiene ventana de fechas y es el que medía 150 g → 650 g "
        "sobre un día ya comido— debe correr DENTRO del freeze. Si el snapshot se toma después, "
        "el freeze conserva fielmente lo que el caller acababa de reescribir (I2)."
    )
    assert _call_lineno(seam, "_apply_budget_cheapen_pass") in dentro, (
        "las sustituciones también van dentro (ya vienen ventaneadas: defensa en profundidad)"
    )
    chain = [ln for _f, ln in _freeze_kwarg_callsites(_GO, "graph_orchestrator.py")]
    assert len(chain) == 1 and chain[0] not in dentro, (
        "el chain (con su PROPIO freeze) debe correr FUERA del `with`, o sea sobre la línea base "
        "ya restaurada: dos ventanas consecutivas con un solo estado de referencia"
    )
    assert chain[0] > max(dentro), "y después, no antes"


def test_parser_el_rebuild_de_listas_corre_tras_el_restore():
    """El restore ocurre al salir de `apply_budget_convergence_for_days`; el rebuild vive en el
    caller, después de la llamada ⇒ lista el plan restaurado."""
    i = _CT.index("P1-BUDGET-T2-CONVERGENCE")
    blk = _CT[i:i + 7000]
    i_seam = blk.index("_abc_t2(full_plan_data")
    assert i_seam < blk.index("aggr_7 = get_shopping_list_delta(", i_seam), (
        "el rebuild de listas debe correr DESPUÉS de que el seam retorne (y por tanto tras el "
        "restore): listar un intermedio pondría en la lista comida que el usuario no tiene"
    )
    assert "get_shopping_list_delta" not in _fn_body(_GO, "apply_budget_convergence_for_days"), (
        "el rebuild NO puede migrar dentro del seam: quedaría dentro del freeze, antes del restore"
    )


def test_parser_last_known_pfix_bumpeado():
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    _fecha = re.search(r"(\d{4}-\d{2}-\d{2})\s*$", m.group(1))
    assert _fecha, f"Marker sin fecha ISO al final: {m.group(1)!r}"
    assert _fecha.group(1) >= "2026-08-04", (
        f"Marker sospechosamente viejo: {m.group(1)!r} (floor P2-T2-PAST-DAYS-FROZEN · 2026-08-04)"
    )


# ═══════════ 3 · Funcional offline ═══════════

class _StubDB:
    """DB offline. Espejo del dummy de `test_p2_caps_after_band_closer.py`: cero red, cero pool."""

    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


@pytest.fixture
def offline(monkeypatch):
    """Chain y seam reales con la DB stub y la emisión de telemetría APAGADA (el refresh de
    banda escribe una fila en `pipeline_metrics`; este test no toca DB)."""
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_METRIC_FINAL_EMIT", False)
    return g


def _lead_g(linea: str) -> float:
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", str(linea).lower())
    return float(m.group(1).replace(",", ".")) if m else -1.0


# 205 g y no un número enorme a propósito: por encima de `LINE_GRAM_HARD_CAP` (600 g) el re-cap
# de realismo del propio chain recortaría la marca y el test mediría el cap, no el freeze.
_MARCA = "205 g de arroz blanco"


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


def _motor_que_infla(marca=_MARCA):
    """Fake de `apply_update_macro_engine`: reproduce el rebalance ×[0.3, 2.5] reescribiendo la
    primera línea de CADA día (pasados incluidos — no tiene ventana de fechas, que es el defecto)
    y desplazando los macros para que el band score intermedio difiera del final."""
    def _fake(plan_data, *a, **k):
        n = 0
        for d in (plan_data or {}).get("days") or []:
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


# ---- ANTI-C1: las superficies de generación NO congelan --------------------

def test_anti_c1_assemble_tail_con_fechas_pasadas_procesa_todos_los_dias(offline, monkeypatch):
    """El caso reproducido por el review: un chunk diferido/reintentado llega a `assemble-tail`
    con días RECIÉN GENERADOS cuya `date` ya pasó. El chain tiene que procesarlos TODOS."""
    import db_plans
    monkeypatch.setattr(offline, "reconcile_all_macros_band_post_finalize", _motor_que_infla())

    plan = _plan(6, 0)  # los 6 días llevan fecha pasada y son recién generados
    entrada = copy.deepcopy(plan)
    db_plans.apply_plan_quality_finalize_chain(plan, surface="assemble-tail")

    for i in range(6):
        assert plan["days"][i] != entrada["days"][i], (
            f"día #{i + 1} CONGELADO en una superficie de generación: se persistiría tal cual lo "
            f"emitió la LLM, sin coherence stack ni caps clínicos"
        )


def test_anti_c1_una_linea_absurda_sigue_capada_en_assemble_tail(offline):
    """La consecuencia concreta del Critical: con el freeze global, 900 g de arroz en un día de
    fecha pasada sobrevivían a `LINE_GRAM_HARD_CAP`. Sin freeze en generación, el cap manda."""
    import db_plans
    plan = _plan(3, 0)
    for d in plan["days"]:
        d["meals"][0]["ingredients"][0] = "900 g de arroz blanco"
        d["meals"][0]["ingredients_raw"][0] = "900 g de arroz blanco"
    db_plans.apply_plan_quality_finalize_chain(plan, surface="assemble-tail")

    for i, d in enumerate(plan["days"]):
        assert _lead_g(d["meals"][0]["ingredients"][0]) <= float(offline.LINE_GRAM_HARD_CAP), (
            f"día #{i + 1}: {d['meals'][0]['ingredients'][0]!r} superó el techo duro "
            f"({offline.LINE_GRAM_HARD_CAP} g) — el freeze se coló en una superficie de generación"
        )


@pytest.mark.parametrize("surface", ["pre-INSERT", "chunk-T1 semana 3", "post-review-patch"])
def test_anti_c1_las_otras_superficies_de_generacion_tampoco_congelan(offline, monkeypatch, surface):
    import db_plans
    monkeypatch.setattr(offline, "reconcile_all_macros_band_post_finalize", _motor_que_infla())
    plan = _plan(4, 2)
    entrada = copy.deepcopy(plan)
    db_plans.apply_plan_quality_finalize_chain(plan, surface=surface)
    assert plan["days"][0] != entrada["days"][0], f"{surface} congeló días pasados"


# ---- el seam T2: el caso central -------------------------------------------

def _seam_offline(monkeypatch, g, *, subs=2):
    """Neutraliza el IO del seam (catálogo/precios) dejando el resto REAL: las sustituciones se
    declaran hechas, y el motor de macros es el fake que infla TODOS los días."""
    monkeypatch.setattr(g, "_apply_budget_driver_aware_pass", lambda *a, **k: 0)
    monkeypatch.setattr(g, "_apply_budget_cheapen_pass", lambda *a, **k: subs)
    monkeypatch.setattr(g, "_protein_repeat_autofix", lambda *a, **k: 0)
    monkeypatch.setattr(g, "apply_update_macro_engine", _motor_que_infla())
    monkeypatch.setattr(g, "MICRO_POSTENGINE_RECOMPUTE_ENABLED", False)


def test_seam_t2_los_dias_pasados_salen_byte_identicos_a_como_entraron(offline, monkeypatch):
    """E2E del seam: convergencia + motor de macros + chain. Los 6 días pasados salen idénticos
    a como ENTRARON AL SEAM (no a como quedaron tras el motor), y los 9 futuros se procesaron."""
    _seam_offline(monkeypatch, offline)

    plan = _plan(6, 9)
    entrada = copy.deepcopy(plan)
    n = offline.apply_budget_convergence_for_days(plan, {"budget": "low"})
    assert n == 2, "el seam debe haber corrido entero (si no, el test no mide nada)"

    for i in range(6):
        assert plan["days"][i] == entrada["days"][i], (
            f"día pasado #{i + 1} ({entrada['days'][i]['date']}) reescrito por el seam: "
            f"{plan['days'][i]['meals'][0]['ingredients'][0]!r}"
        )
    for i in range(6, 15):
        assert plan["days"][i] != entrada["days"][i], (
            f"día futuro #{i + 1} NO procesado: el freeze se pasó de ancho"
        )
        assert _lead_g(plan["days"][i]["meals"][0]["ingredients"][0]) != 150.0


def test_seam_t2_control_i2_sin_freeze_el_motor_reescribe_el_historial(offline, monkeypatch):
    """Control del propio test y reproducción del hallazgo I2: sin el freeze, el motor de macros
    —que corre ANTES del chain y no tiene ventana de fechas— reescribe los días pasados."""
    _seam_offline(monkeypatch, offline)
    monkeypatch.setattr(offline, "CHAIN_PAST_DAYS_FROZEN", False)

    plan = _plan(6, 9)
    entrada = copy.deepcopy(plan)
    offline.apply_budget_convergence_for_days(plan, {"budget": "low"})

    assert plan["days"][0] != entrada["days"][0], (
        "con el kill-switch en false el defecto tiene que reaparecer intacto"
    )
    assert _lead_g(plan["days"][0]["meals"][0]["ingredients"][0]) != 150.0


def test_seam_t2_kill_switch_es_byte_identico_al_prefix(offline, monkeypatch):
    """Contrato de rollback: `MEALFIT_CHAIN_PAST_DAYS_FROZEN=false` ⇒ plan idéntico al que
    produce un seam donde el freeze no existe (snapshot que nunca encuentra nada)."""
    _seam_offline(monkeypatch, offline)
    base = _plan(6, 9)

    plan_off = copy.deepcopy(base)
    monkeypatch.setattr(offline, "CHAIN_PAST_DAYS_FROZEN", False)
    offline.apply_budget_convergence_for_days(plan_off, {"budget": "low"})

    plan_ref = copy.deepcopy(base)
    monkeypatch.setattr(offline, "snapshot_past_days", lambda *a, **k: None)
    offline.apply_budget_convergence_for_days(plan_ref, {"budget": "low"})

    assert plan_off == plan_ref, (
        "con el kill-switch apagado el seam debe quedar byte-idéntico al comportamiento previo"
    )


def test_seam_t2_las_metricas_persistidas_miden_el_estado_final(offline, monkeypatch):
    """Dos mediciones honestas que discrepan = bug. El score y los macros entregados que se
    PERSISTEN tienen que coincidir con recomputarlos sobre el plan tal como queda."""
    _seam_offline(monkeypatch, offline)

    plan = _plan(6, 9)
    offline.apply_budget_convergence_for_days(plan, {"budget": "low"})

    persistido = plan.get("clinical_band_score")
    assert isinstance(persistido, dict) and persistido.get("score") is not None
    recomputado = offline.compute_clinical_band_score(plan, {})
    assert abs(float(persistido["score"]) - float(recomputado["score"])) < 1e-9, (
        f"band score persistido ({persistido['score']}) midió un estado intermedio; "
        f"recomputado sobre el plan final da {recomputado['score']}"
    )
    entregados = copy.deepcopy(plan.get("delivered_macros"))
    offline.refresh_delivered_macros(plan)
    assert plan.get("delivered_macros") == entregados, (
        "`delivered_macros` persistido mide el estado intermedio (pre-restauración)"
    )


def test_seam_t2_plan_sin_fechas_se_procesa_entero(offline, monkeypatch):
    """Fail-open: sin `date` estampada en ningún día no hay ancla → nada se congela."""
    _seam_offline(monkeypatch, offline)
    plan = _plan(6, 9, con_fechas=False)
    entrada = copy.deepcopy(plan)
    offline.apply_budget_convergence_for_days(plan, {"budget": "low"})
    for i, d in enumerate(plan["days"]):
        assert d != entrada["days"][i], f"día #{i + 1} congelado sin ancla: fail-open roto"


def test_seam_t2_conserva_identidad_de_la_lista_y_de_los_dias(offline, monkeypatch):
    """La restauración muta los dicts in-place: ni la lista ni los objetos de día se reemplazan
    (I1: sobre una vista, reasignar el elemento sería inerte)."""
    _seam_offline(monkeypatch, offline)
    plan = _plan(6, 9)
    ref_lista, ref_dia0 = plan["days"], plan["days"][0]
    offline.apply_budget_convergence_for_days(plan, {"budget": "low"})
    assert plan["days"] is ref_lista and len(ref_lista) == 15
    assert plan["days"][0] is ref_dia0, (
        "el día restaurado debe ser el MISMO objeto (clear+update), no una copia reasignada"
    )


def test_restore_alcanza_el_plan_a_traves_de_una_vista(offline):
    """I1 explícito: una vista es una lista NUEVA con los mismos dicts. El restore hecho sobre la
    vista tiene que llegar al plan."""
    plan = _plan(2, 2)
    vista = {"days": list(plan["days"]), "grocery_start_date": plan["grocery_start_date"]}
    assert vista["days"] is not plan["days"]

    token = offline.snapshot_past_days(vista, surface="test-vista")
    assert token is not None
    vista["days"][0]["meals"][0]["ingredients"][0] = "999 g de arroz blanco"
    assert offline.restore_past_days(vista, token) == 2
    assert plan["days"][0]["meals"][0]["ingredients"][0] == "150 g de arroz blanco", (
        "el restore sobre una vista no alcanzó el plan: es exactamente el restore INERTE que el "
        "review encontró (I1)"
    )


# ---- oráculo y primitivas ---------------------------------------------------

def test_el_dia_de_hoy_cuenta_como_futuro(offline):
    plan = _plan(3, 4)  # days[3] es hoy
    assert offline.frozen_past_day_indices(plan, plan["days"]) == [0, 1, 2]


def test_vista_parcial_sin_fechas_no_se_congela(offline):
    """Guard P0-CHUNK-CHAIN-SCOPED: la vista del merge T1 lleva el `grocery_start_date` del plan
    completo y solo los días nuevos. Fecharlos por índice los pondría en el pasado."""
    hoy = _hoy_rd()
    vista = {
        "days": [{"day": n, "meals": []} for n in (8, 9, 10, 11, 12, 13)],
        "grocery_start_date": (hoy - timedelta(days=20)).isoformat(),
    }
    assert offline.frozen_past_day_indices(vista, vista["days"]) == []


def test_el_oraculo_es_puro_e_ignora_el_kill_switch(offline, monkeypatch):
    """El oráculo responde calendario; quien obedece al knob es `snapshot_past_days`."""
    monkeypatch.setattr(offline, "CHAIN_PAST_DAYS_FROZEN", False)
    plan = _plan(3, 4)
    assert offline.frozen_past_day_indices(plan, plan["days"]) == [0, 1, 2]
    assert offline.snapshot_past_days(plan, surface="t") is None, (
        "con el kill-switch apagado, `snapshot_past_days` no debe congelar nada"
    )


def test_primitivas_fail_safe_ante_entradas_absurdas(offline):
    assert offline.frozen_past_day_indices(None, None) == []
    assert offline.frozen_past_day_indices({}, []) == []
    assert offline.frozen_past_day_indices({}, [{"day": 1}]) == []
    assert offline.frozen_past_day_indices({}, ["no soy un dict"]) == []
    assert offline.snapshot_past_days(None) is None
    assert offline.snapshot_past_days({}) is None
    assert offline.restore_past_days({}, None) == 0
    assert offline.restore_past_days(None, ((0,), [{}], 1, "t")) == 0


def test_el_context_manager_restaura_aunque_el_bloque_reviente(offline):
    plan = _plan(3, 3)
    entrada = copy.deepcopy(plan)
    with pytest.raises(RuntimeError):
        with offline.frozen_past_days(plan, surface="test"):
            plan["days"][0]["meals"][0]["ingredients"][0] = "999 g de arroz blanco"
            raise RuntimeError("boom a media escritura")
    assert plan["days"][0] == entrada["days"][0], (
        "el restore vive en un `finally`: si el bloque revienta a media escritura, el historial "
        "vuelve igual"
    )
