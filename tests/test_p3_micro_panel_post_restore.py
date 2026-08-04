"""[P3-MICRO-PANEL-POST-RESTORE · 2026-08-04] Re-review de P2-T2-PAST-DAYS-FROZEN
(higiene barata detectada en el re-review de D1): `recompute_micronutrient_report_for_plan`
vivía DENTRO del `with frozen_past_days(...)` del seam T2
(`apply_budget_convergence_for_days`, graph_orchestrator.py), es decir ANTES del restore
de los días congelados — estampaba el panel de micronutrientes sobre el estado
INTERMEDIO (post-`apply_update_macro_engine`, que reescribe TODO `plan_data` sin ventana
de fechas, días pasados incluidos — el freeze existe justo para revertir eso).

## Por qué es un bug real y no solo estilo

El propio contrato documentado en `restore_past_days` (graph_orchestrator.py) nombra
EXPLÍCITAMENTE "panel de micros" entre las métricas derivadas que DEBEN quedar detrás del
restore: "todo estampado de MÉTRICAS derivadas del plan (band score, `delivered_macros`,
panel de micros) tiene que quedar DETRÁS de esta llamada [restore_past_days]. Si se mide
antes, la métrica persistida describe un estado intermedio que nunca existió". El call
site del seam T2 violaba su propio contrato.

## Por qué hoy el síntoma queda tapado (y por qué eso NO lo hace inofensivo)

`apply_plan_quality_finalize_chain` (llamado justo después, con su PROPIO freeze) vuelve
a recomputar el panel en `db_plans._finalize_plan_data_for_insert`, DESPUÉS de su propio
restore — pero gateado por `if _clin_ctx or not _pd.get('micronutrient_report')`.
`_clin_ctx` típicamente es truthy (el seam T2 SÍ pasa `form_data`), así que en el camino
común el recompute correcto de db_plans sobre-escribe el panel incorrecto de
graph_orchestrator y nadie lo nota. Pero si `form_data` llegara vacío y no hay `user_id`
resoluble (el seam T2 no pasa `user_id` a propósito, ver `apply_plan_quality_finalize_chain`),
`_clin_ctx` cae a `{}` y, como el panel YA EXISTE (estampado por el call site roto), el
guard de db_plans NO recomputa — el panel INTERMEDIO mal medido queda persistido sin que
nadie lo corrija. Depender de un guard ajeno para que un bug local no se note es la misma
clase de fragilidad que "dos mediciones honestas que discrepan" cierra en otros lados.

## El fix

Se movió el recompute del panel de micros a JUSTO DESPUÉS del `with` (con los días ya
restaurados), como segunda capa de seguridad independiente del guard de db_plans — mide
el estado FINAL sin depender de qué haga la superficie downstream.

tooltip-anchor: P3-MICRO-PANEL-POST-RESTORE
"""
from __future__ import annotations

import ast
import copy
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")


def _fn_node(src: str, name: str):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"función top-level {name!r} no encontrada")


def _call_linenos(node, fname: str) -> list:
    return [n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == fname]


# ---------------------------------------------------------------------------
# 0. Marker
# ---------------------------------------------------------------------------
def test_marker_present():
    assert "P3-MICRO-PANEL-POST-RESTORE" in _GO_SRC


# ---------------------------------------------------------------------------
# 1. Parser: el recompute del panel de micros vive FUERA del `with` del seam
# ---------------------------------------------------------------------------
def test_parser_recompute_micro_fuera_del_with():
    seam = _fn_node(_GO_SRC, "apply_budget_convergence_for_days")
    withs = [n for n in ast.walk(seam) if isinstance(n, ast.With)]
    assert len(withs) == 1, "el seam debe seguir abriendo exactamente un `with` (el del historial)"
    dentro = {n.lineno for n in ast.walk(withs[0]) if hasattr(n, "lineno")}

    hits = _call_linenos(seam, "recompute_micronutrient_report_for_plan")
    assert len(hits) == 1, (
        f"se esperaba 1 llamada a recompute_micronutrient_report_for_plan en el seam, hay {len(hits)}"
    )
    assert hits[0] not in dentro, (
        "recompute_micronutrient_report_for_plan sigue DENTRO del `with frozen_past_days(...)` — "
        "mide el estado INTERMEDIO (antes del restore), violando el contrato documentado en "
        "`restore_past_days` (panel de micros debe estamparse DESPUÉS)."
    )
    assert hits[0] > max(dentro), "y después del bloque, no antes"


def test_parser_recompute_micro_antes_de_la_variable_apagada_del_chain():
    """El chain (`_apqfc_t2`, con su propio freeze) debe seguir corriendo DESPUÉS del
    recompute del seam — dos capas de seguridad en orden, ambas post-restore."""
    seam = _fn_node(_GO_SRC, "apply_budget_convergence_for_days")
    recompute_hits = _call_linenos(seam, "recompute_micronutrient_report_for_plan")
    apqfc_hits = _call_linenos(seam, "_apqfc_t2")
    assert recompute_hits and apqfc_hits
    assert recompute_hits[0] < apqfc_hits[0], (
        "el recompute del seam debe preceder al chain (ambos post-restore, en el mismo orden "
        "que ya tenían antes del fix)"
    )


def test_parser_recompute_micro_reusa_db2_ya_indexado():
    """[M-2 · review final] El call site debe pasar `db=` reusando `_db2` — sin él,
    `recompute_micronutrient_report_for_plan` reconstruye Y reindexa un `IngredientNutritionDB`
    nuevo en cada corrida del seam, en vez de reusar el que el truth-up ya creó dos líneas
    arriba."""
    seam = _fn_node(_GO_SRC, "apply_budget_convergence_for_days")
    call = next(n for n in ast.walk(seam)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "recompute_micronutrient_report_for_plan")
    kw_names = {kw.arg for kw in call.keywords}
    assert "db" in kw_names, (
        "el call site debe pasar `db=` — sin él, cada corrida del seam reconstruye y reindexa "
        "un IngredientNutritionDB nuevo (M-2 del review final)"
    )
    # El valor no puede ser un `None` literal (eso sería no-op disfrazado de fix): debe venir de
    # `locals().get("_db2")` — fail-safe si `_db2` nunca llegó a asignarse.
    valor = next(kw.value for kw in call.keywords if kw.arg == "db")
    assert not (isinstance(valor, ast.Constant) and valor.value is None), (
        "`db=None` literal no reusa nada — debe ser `locals().get(\"_db2\")` o equivalente"
    )


# ---------------------------------------------------------------------------
# 1b. Funcional: la db se REUSA (cero reindexado extra) y es fail-safe si `_db2` no se asignó
# ---------------------------------------------------------------------------
def test_funcional_reusa_la_misma_instancia_no_reindexa_dos_veces(monkeypatch):
    """[M-2 · review final] Antes del fix, `recompute_micronutrient_report_for_plan(db=None)`
    reconstruía y reindexaba un `IngredientNutritionDB` SEGUNDO, distinto del `_db2` ya creado
    unas líneas arriba para el truth-up. Con `db=locals().get("_db2")` la PRIMERA invocación (la
    del propio seam — el parser ya ancla que precede al chain) debe reusar la MISMA instancia.

    Solo se mide la PRIMERA llamada capturada: `_apqfc_t2`/`apply_plan_quality_finalize_chain`
    (que corre DESPUÉS, NO mockeado aquí — igual que en el resto de tests de este archivo) tiene
    su PROPIO call site preexistente a `recompute_micronutrient_report_for_plan` (db_plans.py,
    `_finalize_plan_data_for_insert`), ajeno a este fix — mezclar sus instancias en el conteo
    mediría ruido de otra superficie, no el bug de M-2."""
    import nutrition_db
    instancias = []

    class _TrackingDB(_StubDB):
        def __init__(self):
            instancias.append(self)

    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _TrackingDB)
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_METRIC_FINAL_EMIT", False)
    _seam_offline(monkeypatch, g)

    llamadas = []
    _real_recompute = g.recompute_micronutrient_report_for_plan

    def _spy(plan_data, form_data, db=None):
        llamadas.append((db, len(instancias)))
        return _real_recompute(plan_data, form_data, db=db)

    monkeypatch.setattr(g, "recompute_micronutrient_report_for_plan", _spy)

    plan = _plan(2, 2)
    n = g.apply_budget_convergence_for_days(plan, {"budget": "low"})

    assert n == 2, "el seam debe haber corrido entero (si no, el test no mide nada)"
    assert llamadas, "el spy no fue invocado — recompute_micronutrient_report_for_plan no corrió"
    primera_db, primera_n_instancias = llamadas[0]
    assert primera_n_instancias == 1, (
        f"al momento de la PRIMERA llamada (la del propio seam) ya se habían creado "
        f"{primera_n_instancias} instancia(s) de IngredientNutritionDB; se esperaba "
        f"exactamente 1 (`_db2`, creada para el truth-up)"
    )
    assert primera_db is instancias[0], (
        "el `db` recibido en la PRIMERA llamada a recompute_micronutrient_report_for_plan no es "
        "la MISMA instancia que `_db2` — el kwarg del seam no está reusando, está dejando que "
        "se reconstruya otra"
    )


def test_funcional_db2_no_asignada_no_rompe_el_recompute(monkeypatch):
    """[M-2 · review final] Si el bloque `try` que crea `_db2` (dos líneas antes) revienta ANTES
    de asignarla (falla el import o el constructor), `locals().get("_db2")` debe devolver
    `None` en vez de propagar `NameError` — y `recompute_micronutrient_report_for_plan` cae a
    su propio fallback (construye su propia instancia), igual que ANTES de este fix."""
    import nutrition_db

    class _BoomOnce(_StubDB):
        _raised = False

        def __init__(self):
            if not _BoomOnce._raised:
                _BoomOnce._raised = True
                raise RuntimeError("boom: el constructor de _db2 falla la PRIMERA vez")

    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _BoomOnce)
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_METRIC_FINAL_EMIT", False)
    _seam_offline(monkeypatch, g)

    plan = _plan(2, 2)
    n = g.apply_budget_convergence_for_days(plan, {"budget": "low"})  # no debe lanzar NameError

    assert n == 2, "el seam debe correr entero pese al fallo del primer constructor"
    assert isinstance(plan.get("micronutrient_report"), dict), (
        "el recompute debió caer al fallback interno (su propia instancia) y completar, "
        "en vez de quedar silenciosamente saltado por un NameError sobre `_db2`"
    )


# ---------------------------------------------------------------------------
# 2. Funcional offline: el panel persistido mide el estado FINAL, no el intermedio
# ---------------------------------------------------------------------------
class _StubDB:
    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None

    def micros_from_ingredient_string(self, s):
        # [M-2 · review final] `compute_plan_micronutrient_totals` (micronutrients.py) llama
        # este método — sin él, `build_micronutrient_report` revienta con AttributeError y
        # `recompute_micronutrient_report_for_plan` lo traga silenciosamente (best-effort),
        # dejando `micronutrient_report` sin estampar y los tests de reuso de `db=` sin poder
        # verificar que el recompute REALMENTE completó. `None` es una respuesta válida del
        # contrato real (nutrition_db.py): "no resuelve" → el acumulador hace `if not m: continue`.
        return None


@pytest.fixture
def offline(monkeypatch):
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)
    import graph_orchestrator as g
    monkeypatch.setattr(g, "BAND_METRIC_FINAL_EMIT", False)
    return g


def _hoy_rd():
    return (datetime.now(timezone.utc) - timedelta(hours=4)).date()


def _dia(n: int, fecha_iso, *, linea="150 g de arroz blanco"):
    return {
        "day": n, "day_name": f"Día {n}", "date": fecha_iso,
        "meals": [{
            "meal": "Almuerzo", "name": "Pollo guisado con arroz",
            "protein": 35, "carbs": 60, "fats": 12, "cals": 488,
            "ingredients": [linea, "120 g de pechuga de pollo"],
            "ingredients_raw": [linea, "120 g de pechuga de pollo"],
            "recipe": ["MISE EN PLACE: Pica el pollo.",
                       "EL TOQUE DE FUEGO: Guisa 20 min y hierve el arroz.",
                       "MONTAJE: Sirve caliente."],
        }],
    }


def _plan(n_pasados: int, n_futuros: int) -> dict:
    hoy = _hoy_rd()
    days = []
    for k in range(n_pasados + n_futuros):
        f = (hoy + timedelta(days=k - n_pasados)).isoformat()
        days.append(_dia(k + 1, f))
    return {
        "days": days,
        "macros": {"protein": "100g", "carbs": "200g", "fats": "60g"},
        "calories": "2000 kcal",
        "grocery_start_date": (hoy - timedelta(days=n_pasados)).isoformat(),
    }


def _motor_que_infla(marca="205 g de arroz blanco"):
    """Fake de `apply_update_macro_engine`: reescribe la primera línea de CADA día
    (pasados incluidos) — el defecto real que el freeze protege."""
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


def _seam_offline(monkeypatch, g, *, subs=2):
    monkeypatch.setattr(g, "_apply_budget_driver_aware_pass", lambda *a, **k: 0)
    monkeypatch.setattr(g, "_apply_budget_cheapen_pass", lambda *a, **k: subs)
    monkeypatch.setattr(g, "_protein_repeat_autofix", lambda *a, **k: 0)
    monkeypatch.setattr(g, "apply_update_macro_engine", _motor_que_infla())


def test_recompute_ve_el_plan_ya_restaurado_no_el_intermedio(offline, monkeypatch):
    """El caso central: al momento EXACTO en que corre
    `recompute_micronutrient_report_for_plan`, los días pasados ya deben estar
    restaurados a su contenido ORIGINAL (pre-`apply_update_macro_engine`), no al
    contenido inflado que el motor de macros acaba de escribir."""
    _seam_offline(monkeypatch, offline)

    plan = _plan(6, 9)
    entrada = copy.deepcopy(plan)

    capturas = []
    _real_recompute = offline.recompute_micronutrient_report_for_plan

    def _spy(plan_data, form_data, db=None):
        # Snapshot de los días PASADOS tal como los ve el recompute en este instante.
        capturas.append(copy.deepcopy(plan_data.get("days", [])[:6]))
        return _real_recompute(plan_data, form_data, db=db)

    monkeypatch.setattr(offline, "recompute_micronutrient_report_for_plan", _spy)

    offline.apply_budget_convergence_for_days(plan, {"budget": "low"})

    assert capturas, "el spy no fue invocado — recompute_micronutrient_report_for_plan no corrió"
    vistos = capturas[0]
    for i in range(6):
        assert vistos[i] == entrada["days"][i], (
            f"día pasado #{i + 1} ({entrada['days'][i]['date']}): el recompute del panel de "
            f"micros vio el estado INTERMEDIO (post-motor, pre-restore) en vez del original. "
            f"visto={vistos[i]['meals'][0]['ingredients'][0]!r} "
            f"original={entrada['days'][i]['meals'][0]['ingredients'][0]!r}"
        )


def test_los_dias_pasados_siguen_intactos_al_final_del_seam(offline, monkeypatch):
    """Control de regresión: el fix no debe romper la garantía YA anclada por
    test_p2_t2_past_days_frozen.py — los días pasados siguen saliendo byte-idénticos."""
    _seam_offline(monkeypatch, offline)
    plan = _plan(6, 9)
    entrada = copy.deepcopy(plan)
    n = offline.apply_budget_convergence_for_days(plan, {"budget": "low"})
    assert n == 2
    for i in range(6):
        assert plan["days"][i] == entrada["days"][i], f"día pasado #{i + 1} reescrito por el seam"


# ---------------------------------------------------------------------------
# 3. Marker bump — patrón fecha-floor
# ---------------------------------------------------------------------------
def test_last_known_pfix_bumpeado():
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    marker = m.group(1)
    from datetime import date, datetime as _dt
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha, f"Marker sin fecha ISO: {marker!r}"
    marker_date = _dt.strptime(fecha.group(1), "%Y-%m-%d").date()
    floor = date(2026, 8, 4)
    assert marker_date >= floor, (
        f"_LAST_KNOWN_PFIX={marker!r} (fecha={marker_date}) anterior al floor {floor} "
        f"de cierre de P3-MICRO-PANEL-POST-RESTORE."
    )
