"""[P1-CHAIN-CLINICAL-CTX · 2026-08-02] El adapter del finalize-chain no transportaba contexto clínico.

`P1-PREINSERT-CLINICAL-CTX` (2026-07-30) arregló el shield del INSERT: el contexto clínico se
deriva del PERFIL por `user_id` porque `plan_data['form_data']` está vacío en el 100% de los
planes. Pero el mismo shield corre en OTRAS 5 superficies a través del adapter público
`apply_plan_quality_finalize_chain`, y ese adapter construía literalmente `{"plan_data": plan_data}`
— sin `user_id` NI `form_data`. O sea: en esas 5 superficies `_clin_ctx` resolvía a `{}` incluso
después de aquel fix, porque no había NADA de donde derivar el perfil.

Dos consecuencias, ambas silenciosas:

  (a) **El band-closer final re-infla lo que la clínica acababa de recortar.** El chain termina en
      `reconcile_all_macros_band_post_finalize` → `apply_update_macro_engine`, cuyo re-cap clínico
      (`cap_dm2_high_gi_portions` / `cap_bariatric_portions`) se gatea por `form_data`. Sin él, la
      rama `elif _hit and ... and not form_data` cuenta el día como "re-cap omitido" y avisa que
      "nadie re-capea aguas abajo". En `chunk-T1` eso es terminal: las semanas 2+ persisten vía
      UPDATE crudo y NO hay ningún punto posterior.

  (b) **El panel de micros condicionado se pisa con uno genérico.** `_rmr(_pd, _clin_ctx)` con
      `{}` recomputa con `sex='female'` y `conditions=[]` — sin techo de sodio para HTA ni de
      potasio para ERC — encima de un panel que SÍ los tenía. El gate de sodio del review consume
      ese panel: para un hipertenso el gate quedaba MÁS laxo que antes de recomputar.

El fix es de corrección pura (pasar datos que ya están en scope), sin knob: un knob aquí sería una
vía documentada para reintroducir el bug.

La asimetría del guard del panel es intencional y es el corazón de este test: `{}` significa
"no sé", NO "sin condiciones" (el mismo criterio que `build_clinical_form_from_profile` documenta
en su docstring). Por eso el panel se computa cuando NO existe uno previo — aunque el contexto sea
vacío —, pero JAMÁS se sobrescribe uno existente con un recompute a ciegas.
"""
from __future__ import annotations

import inspect
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

MARKER = "P1-CHAIN-CLINICAL-CTX"


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, *rel.split("/")), encoding="utf-8") as f:
        return f.read()


_DBP = _read("db_plans.py")
_GO = _read("graph_orchestrator.py")
_CT = _read("cron_tasks.py")


# ═══════════════════════ 1. la firma del adapter ═══════════════════════

def test_el_adapter_acepta_contexto_clinico():
    """Sin estos dos parámetros el adapter no tiene NADA que poner en el dict del shield, y
    `_clin_ctx` resuelve a `{}` por construcción en las 5 superficies."""
    import db_plans
    sig = inspect.signature(db_plans.apply_plan_quality_finalize_chain)
    for _p in ("form_data", "user_id"):
        assert _p in sig.parameters, (
            f"el adapter sigue sin `{_p}`: el shield no puede derivar el contexto clínico y el "
            f"band-closer re-infla las porciones capadas sin re-capearlas")
        assert sig.parameters[_p].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"`{_p}` debe ser keyword-only (los callers posicionales existentes no se rompen)")
        assert sig.parameters[_p].default is None, (
            f"`{_p}` debe ser opcional: el shield del INSERT lo llama sin él")


def test_el_adapter_puebla_el_dict_del_shield(monkeypatch):
    """El contrato real: lo que el adapter recibe tiene que llegar al dict `data` que
    `_finalize_plan_data_for_insert` inspecciona (`data.get("form_data")` / `data.get("user_id")`).
    Verificar solo la firma dejaría pasar un parámetro aceptado y descartado."""
    import db_plans
    capturado = {}
    monkeypatch.setattr(db_plans, "_finalize_plan_data_for_insert",
                        lambda data, **kw: capturado.update(data))
    _fd = {"gender": "male", "medicalConditions": ["Hipertensión"]}
    db_plans.apply_plan_quality_finalize_chain({"days": []}, surface="assemble-tail",
                                               form_data=_fd, user_id="u-1")
    assert capturado.get("form_data") == _fd, "el form_data del caller no llega al shield"
    assert capturado.get("user_id") == "u-1", "el user_id del caller no llega al shield"
    assert capturado.get("plan_data") is not None, "el plan_data debe seguir viajando igual"


# ═══════════════════════ 2. el guard del panel de micros ═══════════════════════

def _chain_con_espias(monkeypatch):
    """Neutraliza el coherence stack (CPU + DB) y espía los dos consumidores del contexto
    clínico. Devuelve (llamadas_rmr, llamadas_ramb) — listas que se pueblan al invocar."""
    import graph_orchestrator as go
    rmr, ramb = [], []
    monkeypatch.setattr(go, "finalize_plan_data_coherence",
                        lambda days, **kw: (0, ""), raising=False)
    monkeypatch.setattr(go, "recompute_micronutrient_report_for_plan",
                        lambda plan, form_data, db=None: rmr.append(form_data) or True)
    monkeypatch.setattr(go, "reconcile_all_macros_band_post_finalize",
                        lambda plan_data, db=None, form_data=None: ramb.append(form_data) or 0)
    return rmr, ramb


def test_un_panel_condicionado_no_se_pisa_con_contexto_vacio(monkeypatch):
    """El caso que motivó el guard: ctx vacío + panel previo = NO tocar.

    Con `{}` el recompute usa `sex='female'` y `conditions=[]`. Sobrescribir con eso un panel que
    trae los techos de HTA/ERC no es "refrescar": es DEGRADAR, y en la dirección peligrosa (el
    propio `micronutrients.py` avisa de que la sub-estimación de un techo es la que hace daño)."""
    import db_plans
    rmr, _ = _chain_con_espias(monkeypatch)
    plan = {"days": [], "micronutrient_report": {"_ctx": "male+HTA", "sodio": {"techo": 1500}}}
    db_plans.apply_plan_quality_finalize_chain(plan, surface="assemble-tail")  # sin form_data
    assert rmr == [], (
        "el panel condicionado se recomputó con contexto vacío: un panel previo NUNCA debe ser "
        "pisado por uno computado sin condiciones ({} significa 'no sé', no 'sin condiciones')")
    assert plan["micronutrient_report"].get("sodio", {}).get("techo") == 1500


def test_sin_panel_previo_el_recompute_corre_aunque_el_contexto_sea_vacio(monkeypatch):
    """La otra mitad de la asimetría. Si no hay panel, un panel genérico es mejor que ninguno —
    el guard es "no pisar", no "no computar". Convertirlo en `if _clin_ctx:` a secas dejaría sin
    panel a todo plan cuyo perfil no se pudo leer."""
    import db_plans
    rmr, _ = _chain_con_espias(monkeypatch)
    db_plans.apply_plan_quality_finalize_chain({"days": []}, surface="assemble-tail")
    assert rmr == [{}], f"el panel debe computarse cuando no existe uno previo (rmr={rmr!r})"


def test_con_contexto_clinico_real_el_panel_si_se_refresca(monkeypatch):
    """Y con contexto, el refresco vuelve a ser obligatorio — es el motivo original de
    P1-MICRO-PANEL-POST-FINALIZE (9 de 12 planes con el panel desfasado)."""
    import db_plans
    rmr, _ = _chain_con_espias(monkeypatch)
    _fd = {"gender": "male", "medicalConditions": ["Hipertensión"]}
    plan = {"days": [], "micronutrient_report": {"_ctx": "generico"}}
    db_plans.apply_plan_quality_finalize_chain(plan, surface="assemble-tail", form_data=_fd)
    assert rmr == [_fd], (
        f"con condiciones declaradas el panel DEBE recomputarse con ellas (rmr={rmr!r})")


def test_el_band_closer_recibe_el_contexto_clinico(monkeypatch):
    """La consecuencia (a): sin `form_data` el motor de macros no re-aplica cap_dm2/cap_bariatric
    y el rebalance/refine devuelve gramos a la porción que la clínica recortó."""
    import db_plans
    _, ramb = _chain_con_espias(monkeypatch)
    _fd = {"gender": "male", "medicalConditions": ["Diabetes Tipo 2"]}
    db_plans.apply_plan_quality_finalize_chain({"days": []}, surface="chunk-T1 semana 2",
                                               form_data=_fd)
    assert ramb == [_fd], (
        f"el band-closer final sigue recibiendo el contexto vacío (ramb={ramb!r}): en chunk-T1 "
        f"esto es terminal — las semanas 2+ persisten vía UPDATE y nadie re-capea después")


def test_el_user_id_hidrata_el_perfil_cuando_no_hay_form_data(monkeypatch):
    """Fallback: si una superficie futura solo tiene `user_id`, el shield debe poder derivar el
    contexto del perfil (misma ruta que el INSERT). Sin esto, `user_id` sería decorativo."""
    import db_plans
    rmr, _ = _chain_con_espias(monkeypatch)
    _perfil = {"gender": "male", "medicalConditions": ["ERC"], "allergies": []}
    monkeypatch.setattr(db_plans, "_build_clinical_form", lambda uid: _perfil if uid else {})
    plan = {"days": [], "micronutrient_report": {"_ctx": "generico"}}
    db_plans.apply_plan_quality_finalize_chain(plan, surface="assemble-tail", user_id="u-1")
    assert rmr == [_perfil], f"el user_id no hidrató el contexto clínico (rmr={rmr!r})"


# ═══════ 2-bis. el contexto llega HASTA el re-cap clínico (no es cosmético) ═══════

def _plan_fuera_de_banda() -> dict:
    """Día con los carbos muy por encima del target → `_out_of_band` en el motor de macros."""
    return {
        "days": [{"day": 1, "meals": [{
            "meal": "Almuerzo", "name": "Batata al horno con pollo",
            "protein": 10, "carbs": 200, "fats": 5, "cals": 885,
            "ingredients": ["300g de batata", "100g de pechuga de pollo"],
            "ingredients_raw": ["300g de batata", "100g de pechuga de pollo"],
            "recipe": ["MISE EN PLACE: Pela la batata.",
                       "EL TOQUE DE FUEGO: Hornea 30 min.",
                       "MONTAJE: Sirve caliente."],
        }]}],
        "macros": {"protein": "100g", "carbs": "150g", "fats": "60g"},
        "calories": "2000 kcal",
        "micronutrient_report": {"_ctx": "previo"},
    }


def _espia_los_caps_clinicos(monkeypatch):
    """Ejecuta el chain REAL hasta el motor de macros (solo se neutraliza el coherence stack, que
    es CPU pesado y podría dejar el día en banda) y espía los dos caps clínicos."""
    import graph_orchestrator as go
    caps = []
    monkeypatch.setattr(go, "finalize_plan_data_coherence", lambda days, **kw: (0, ""))
    # el rebalance decide `_hit`, que es la puerta COMPARTIDA por la rama de re-cap y la de
    # warning: forzarlo a True aísla la única diferencia real entre ambas — `form_data`.
    monkeypatch.setattr(go, "_rebalance_day_macros_to_target",
                        lambda meals, cg, fg, db, target_protein=None: True)
    monkeypatch.setattr(go, "cap_dm2_high_gi_portions",
                        lambda days, fd, db=None: caps.append(("dm2", fd)) or 0)
    monkeypatch.setattr(go, "cap_bariatric_portions",
                        lambda days, fd, db=None: caps.append(("bariatrico", fd)) or 0)
    return caps


def test_el_recap_clinico_del_motor_se_ejecuta_de_verdad_con_form_data(monkeypatch):
    """El riesgo que había que descartar: que el threading fuera cosmético porque otra condición
    (el gate `_hit`) hiciera la rama inalcanzable en esta superficie.

    No lo es. `_hit` es la MISMA puerta para la rama de re-cap y para la de warning
    (`elif _hit and ... and not form_data`), así que threadear `form_data` es exactamente lo que
    decide cuál de las dos corre en los días que el motor re-dimensiona."""
    import db_plans
    caps = _espia_los_caps_clinicos(monkeypatch)
    _fd = {"gender": "male", "medicalConditions": ["Diabetes Tipo 2"]}
    db_plans.apply_plan_quality_finalize_chain(_plan_fuera_de_banda(),
                                               surface="chunk-T1 semana 2", form_data=_fd)
    assert [n for n, _ in caps] == ["dm2", "bariatrico"], (
        f"el re-cap clínico NO llegó a ejecutarse pese a pasar form_data (caps={caps!r}): el "
        f"threading sería cosmético y la batata del DM2 seguiría volviendo de 150 g a 300+")
    assert all(fd == _fd for _, fd in caps), "los caps deben recibir el contexto del caller"


def test_sin_form_data_el_recap_clinico_no_corre(monkeypatch):
    """El contrafactual — sin él, el test de arriba no probaría nada (podría estar corriendo
    siempre). Este es el estado ANTERIOR al fix en las 5 superficies del chain."""
    import db_plans
    caps = _espia_los_caps_clinicos(monkeypatch)
    db_plans.apply_plan_quality_finalize_chain(_plan_fuera_de_banda(),
                                               surface="chunk-T1 semana 2")
    assert caps == [], f"sin contexto clínico no hay nada que re-capear (caps={caps!r})"


# ═══════════════════════ 3. las 5 superficies ═══════════════════════

# (fuente, aguja del surface, por qué duele si queda ctx-less)
_CALLSITES = (
    ("graph_orchestrator.py", 'surface="assemble-tail"',
     "cola de assemble: último pase que mueve cantidades antes del review y del SSE"),
    ("graph_orchestrator.py", 'surface="assemble-budget-convergence"',
     "re-fire post-convergencia: las sustituciones económicas re-dimensionan porciones"),
    ("graph_orchestrator.py", 'surface="t2-budget-convergence"',
     "seam T2 del chunk worker: sin usuario mirando, persiste directo"),
    ("graph_orchestrator.py", 'surface="post-review-patch"',
     "re-cierre de banda tras el auto-patch de huérfanos del review"),
    ("cron_tasks.py", 'surface=f"chunk-T1',
     "merge T1 semanas 2+: NO hay ningún punto posterior que re-capee"),
)

_FUENTES = {"graph_orchestrator.py": _GO, "cron_tasks.py": _CT}


def _expresion_de_llamada(src: str, aguja: str) -> str:
    """Devuelve la expresión COMPLETA de la llamada al chain que contiene `aguja`.

    Nada de ventanas de bytes fijas: en este repo ya caducaron varias veces (una ventana de ±400
    alrededor de `surface="assemble-budget-convergence"` roza el `form_data=form_data` de
    `apply_update_macro_engine`, que está 15 líneas más arriba y NO es este callsite). Se recorta
    por sintaxis: desde la línea que nombra el alias `_apqfc*` hasta su paréntesis de cierre.
    """
    i = src.index(aguja)
    j = src.rindex("\n", 0, i)
    while "_apqfc" not in src[j + 1:src.index("\n", j + 1)]:
        j = src.rindex("\n", 0, j)
    inicio = j + 1
    profundidad = 0
    for k in range(inicio, len(src)):
        if src[k] == "(":
            profundidad += 1
        elif src[k] == ")":
            profundidad -= 1
            if profundidad == 0:
                return src[inicio:k + 1]
    raise AssertionError(f"llamada sin cerrar para {aguja!r}")


def test_las_cinco_superficies_threadean_el_contexto_clinico():
    """tooltip-anchor: P1-CHAIN-CLINICAL-CTX (marker inline en cada callsite)."""
    for archivo, aguja, porque in _CALLSITES:
        expr = _expresion_de_llamada(_FUENTES[archivo], aguja)
        assert "form_data" in expr, (
            f"{archivo} {aguja} sigue ctx-less ({porque}). Llamada: {expr!r}")


def test_el_chain_del_chunk_pasa_el_contexto_como_argumento_no_como_clave_del_view():
    """`_chain_view_ck` es un view REDUCIDO cuyos writes plan-level se descartan a propósito
    (medir banda sobre 4 días y copiarla a un plan de 22+ sería deshonesto). El contexto clínico
    no es un write: va como argumento de la llamada. Meterlo en el view lo convertiría en una
    clave de `plan_data` que nadie limpia y que el resto del chain podría confundir con datos
    persistibles."""
    i = _CT.index("_chain_view_ck = {")
    view = _CT[i:_CT.index("}", i)]
    assert "form_data" not in view and "user_id" not in view, (
        f"el contexto clínico se coló en el view del chunk: {view!r}")


def test_el_marker_esta_en_las_tres_fuentes():
    """Los tests que parsean prod necesitan ancla: un renombre debe romper el test ANTES de
    cambiar producción."""
    for archivo, src in (("db_plans.py", _DBP), ("graph_orchestrator.py", _GO),
                         ("cron_tasks.py", _CT)):
        assert MARKER in src, f"falta el marker {MARKER} en {archivo}"


def test_el_guard_del_panel_es_explicito_en_el_fuente():
    """La condición exacta importa: `if _clin_ctx:` a secas dejaría sin panel a los planes cuyo
    perfil no se pudo leer, y `if not _pd.get(...)` a secas nunca refrescaría el panel de un
    plan con condiciones. La conjunción con `or` es la que codifica la asimetría."""
    i = _DBP.index("recompute_micronutrient_report_for_plan as _rmr")
    blq = _DBP[i:_DBP.index("_rmr(", i)]
    assert "if _clin_ctx or not _pd.get(\"micronutrient_report\")" in blq, (
        f"el guard del panel no está o cambió de forma: {blq[-300:]!r}")


def test_no_hay_knob_para_este_fix():
    """Decisión explícita: esto es una corrección (pasar datos que ya existen), no un cambio de
    política. Un knob aquí solo daría una vía soportada para reintroducir el bug."""
    for src in (_DBP, _GO, _CT):
        assert "MEALFIT_CHAIN_CLINICAL_CTX" not in src, (
            "apareció un knob para P1-CHAIN-CLINICAL-CTX: el fix no es reversible por diseño")
