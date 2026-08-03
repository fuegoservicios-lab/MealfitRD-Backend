"""[P2-CAPS-AFTER-BAND-CLOSER · 2026-08-03] (audit solver+seeder v7) Los caps de REALISMO
vuelven a ser la ULTIMA palabra del chain pre-INSERT.

El marker `P1-CAPS-LAST-WORD` (graph_orchestrator.py) documenta la doctrina: "correr los caps
al final los convierte en invariante en vez de en un pase mas de la cadena". Esa ultima
invocacion vive DENTRO de `finalize_plan_data_coherence` (`_fpc`), y en
`db_plans._finalize_plan_data_for_insert` DESPUES de `_fpc` corren dos band-closers que MUEVEN
cantidades sin consultar ningun cap:

    _fpc  (contiene la ultima _cap_unrealistic_portions bajo CAPS_LAST_WORD)
    _rpb  (reconcile_protein_band_post_finalize — bump de proteina hasta x1.2)
    _ramb (reconcile_all_macros_band_post_finalize — rebalance bidireccional x[0.3, 2.5])
    reconcile display<->raw / polish / condimentos   <- NINGUNO capea

`_rebalance_day_macros_to_target` escala TODAS las lineas del grupo macro-dominante por un
factor comun clampeado a [0.3, 2.5]. Un "250 g de pepino" ya capado es carbo-dominante => entra
SIEMPRE al set movible del pase de carbs => puede salir a 350-600 g en el ULTIMO pase antes de
persistir. Es la clase exacta del caso vivo 943c604b (360 g de queso cottage, el doble del
techo por comida) que P1-CAPS-LAST-WORD cerro un nivel mas arriba.

El fix es aditivo: re-invocar los DOS caps de realismo (`_cap_unrealistic_portions` +
`_cap_cheese_dumps_final`) inmediatamente despues de `_ramb` y ANTES del reconcile display<->raw
de P2-RECONCILE-AFTER-BAND-CLOSER, para que la reconciliacion, el polish y el refresh de banda
midan el estado YA recortado. Ambos son idempotentes y SOLO-BAJAN (mismo argumento del marker
original: re-ejecutarlos no puede inventar comida).

Trade-off asumido, identico al de P1-CAPS-LAST-WORD: si el recorte reabre el hueco de macro que
`_ramb` acababa de cerrar, GANA EL CAP. No se anade un segundo rebalance detras — dos guardas
sobre el mismo campo oscilan. Una banda ligeramente fuera es preferible a persistir una porcion
no servible.

Knob de rollback sin redeploy: MEALFIT_CAPS_AFTER_BAND_CLOSER=false => chain identico al previo.
"""
import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_DBP = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_APP = (_BACKEND / "app.py").read_text(encoding="utf-8")


def _fn_body(src: str, name: str) -> str:
    """Cuerpo textual de una funcion top-level (hasta el siguiente `def `/`class ` a col 0)."""
    i = src.index(f"def {name}(")
    j = len(src)
    for _tok in ("\ndef ", "\nclass "):
        k = src.find(_tok, i + 1)
        if k != -1:
            j = min(j, k)
    return src[i:j]


# ═════════════════════ 1 · Parser: el ORDEN dentro del chain ═════════════════════

def test_parser_recap_corre_entre_el_band_closer_y_el_reconcile():
    """tooltip-anchor: P2-CAPS-AFTER-BAND-CLOSER

    El orden es la ESENCIA del fix (el cap aislado ya funcionaba en 943c604b: "no falla el cap,
    falla el ORDEN"). Se ancla que el re-cap va DESPUES de la invocacion de `_ramb` y ANTES del
    reconcile display<->raw, para que la reconciliacion vea el estado recortado.
    """
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_ramb = body.index("_ramb(_pd,")
    i_recap = body.find("_cap_unrealistic_portions", i_ramb)
    i_reconcile = body.find("RECONCILE_AFTER_BAND_CLOSER", i_ramb)
    assert i_recap != -1, (
        "P2-CAPS-AFTER-BAND-CLOSER: no hay re-cap de realismo tras `_ramb` en el chain "
        "pre-INSERT — el rebalance x2.5 corre despues del ultimo cap y un pepino de 250 g "
        "puede salir de 600 g."
    )
    assert i_reconcile != -1, "el reconcile display<->raw (P2-RECONCILE-AFTER-BAND-CLOSER) desaparecio"
    assert i_ramb < i_recap < i_reconcile, (
        f"orden roto: _ramb={i_ramb} re-cap={i_recap} reconcile={i_reconcile}. El re-cap DEBE ir "
        f"entre ambos: despues del band-closer (que re-infla) y antes del reconciliador "
        f"display<->raw (que debe ver el estado ya recortado)."
    )


def test_parser_el_recap_incluye_tambien_el_cap_de_queso():
    """`_cap_cheese_dumps_final` es el otro cap de realismo del persist boundary (sweet-aware,
    techo por comida/merienda) y el caso vivo 943c604b era JUSTO queso cottage. Re-capear solo
    `_cap_unrealistic_portions` dejaria fuera al cap que nombra la evidencia."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_ramb = body.index("_ramb(_pd,")
    i_reconcile = body.index("RECONCILE_AFTER_BAND_CLOSER", i_ramb)
    bloque = body[i_ramb:i_reconcile]
    assert "_cap_cheese_dumps_final" in bloque, (
        "P2-CAPS-AFTER-BAND-CLOSER: falta el cap de queso en el re-cap post-band-closer "
        "(el caso vivo 943c604b eran 360 g de queso cottage)."
    )


def test_parser_marker_inline_presente():
    assert "[P2-CAPS-AFTER-BAND-CLOSER · 2026-08-03]" in _DBP, (
        "falta el marker inline en db_plans.py"
    )


def test_parser_knob_registrado_via_env_bool():
    """El knob nace en graph_orchestrator (donde vive `_env_bool` y su `_KNOBS_REGISTRY`),
    igual que `RECONCILE_AFTER_BAND_CLOSER` y `CAPS_LAST_WORD` — nunca `os.environ` crudo."""
    assert 'CAPS_AFTER_BAND_CLOSER = _env_bool("MEALFIT_CAPS_AFTER_BAND_CLOSER", True)' in _GO, (
        "el knob debe declararse con `_env_bool` (auto-registro en _KNOBS_REGISTRY), default True"
    )
    import graph_orchestrator as g
    assert isinstance(g.CAPS_AFTER_BAND_CLOSER, bool)
    from graph_orchestrator import get_knobs_registry_snapshot
    assert "MEALFIT_CAPS_AFTER_BAND_CLOSER" in get_knobs_registry_snapshot(), (
        "el knob no llego al registro de knobs (se leyo con os.environ crudo?)"
    )
    # No se puede confundir con el knob homonimo del reconciliador.
    assert "MEALFIT_CAPS_AFTER_BAND_CLOSER" != "MEALFIT_RECONCILE_AFTER_BAND_CLOSER"


def test_parser_el_recap_esta_gateado_por_el_knob():
    """Con el knob en False el bloque no debe correr: es el contrato de rollback sin redeploy."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_ramb = body.index("_ramb(_pd,")
    i_reconcile = body.index("RECONCILE_AFTER_BAND_CLOSER", i_ramb)
    bloque = body[i_ramb:i_reconcile]
    assert "CAPS_AFTER_BAND_CLOSER" in bloque, (
        "el re-cap debe leer el knob `CAPS_AFTER_BAND_CLOSER` (rollback sin redeploy)"
    )


def test_parser_no_se_anadio_un_segundo_rebalance_detras_del_recap():
    """Anti-oscilacion: la leccion repetida del repo es que dos guardas sobre el mismo campo
    OSCILAN. El re-cap recorta lo que `_ramb` subio y ahi termina — si alguien anade otro
    band-closer detras, el par entra en ciclo (cap baja, closer sube, cap baja...)."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_recap = body.index("_cap_unrealistic_portions")
    cola = body[i_recap:]
    for _prohibido in ("_ramb(", "reconcile_all_macros_band_post_finalize",
                       "reconcile_protein_band_post_finalize"):
        assert _prohibido not in cola, (
            f"{_prohibido} aparece DESPUES del re-cap: el par cap<->closer oscila. "
            f"El trade-off documentado es que gana el cap."
        )


def test_parser_last_known_pfix_bumpeado():
    assert '"P2-CAPS-AFTER-BAND-CLOSER · 2026-08-03"' in _APP, (
        "_LAST_KNOWN_PFIX sin bumpear en app.py"
    )


# ═════════════════════ 2 · Funcional: el chain completo ═════════════════════

class _StubDB:
    """DB offline. Espejo del dummy de `test_p2_veg_volume_tokens_2.py`: solo responde a
    `macros_from_ingredient_string`, suficiente para las ramas de gramos de los caps de
    realismo. Cero red, cero pool."""

    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


def _plan_con(linea_display, *, nombre="Ensalada fria de pepino", comida="Cena"):
    return {
        "days": [{
            "day": 1,
            "meals": [{
                "meal": comida,
                "name": nombre,
                "protein": 8, "carbs": 30, "fats": 4, "cals": 188,
                "ingredients": [linea_display, "100 g de pechuga de pollo"],
                "ingredients_raw": [linea_display, "100 g de pechuga de pollo"],
                "recipe": ["MISE EN PLACE: Pica todo.",
                           "EL TOQUE DE FUEGO: Saltea 5 min.",
                           "MONTAJE: Sirve frio."],
            }],
        }],
        "macros": {"protein": "100g", "carbs": "200g", "fats": "60g"},
        "calories": "2000 kcal",
    }


def _lead_g(linea: str) -> float:
    import re
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", str(linea).lower())
    return float(m.group(1).replace(",", ".")) if m else -1.0


@pytest.fixture
def chain_offline(monkeypatch):
    """Chain real con la DB stub inyectada (`db_plans` importa `IngredientNutritionDB` de forma
    LAZY dentro de la funcion, asi que parchear el modulo basta)."""
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)
    import db_plans
    return db_plans


def _inflar(objetivo_g, indice=0):
    """Fake de `_ramb`: reproduce lo que hace `_rebalance_day_macros_to_target` con un factor
    >1 sobre el grupo carbo-dominante — re-escribe la linea a `objetivo_g` gramos. No inventa
    nada nuevo: es exactamente la mutacion que el rebalance x[0.3, 2.5] produce."""
    import re

    def _fake(plan_data, form_data=None, db=None):
        for d in plan_data.get("days") or []:
            for m in d.get("meals") or []:
                for lista in ("ingredients", "ingredients_raw"):
                    ings = m.get(lista)
                    if isinstance(ings, list) and len(ings) > indice:
                        ings[indice] = re.sub(r"^\s*\d+(?:[.,]\d+)?", str(objetivo_g),
                                              str(ings[indice]))
        return 1
    return _fake


def test_funcional_el_band_closer_reinfla_el_pepino_y_el_recap_lo_devuelve_al_techo(
        chain_offline, monkeypatch):
    """El caso de la clase 943c604b, con vegetal acuoso: una linea ya al techo de realismo
    (250 g) que el rebalance de carbos infla a 600 g en el ULTIMO pase antes de persistir.
    Tras el fix, lo que se persiste esta en el techo de la CLASE (250 g).

    600 g esta elegido a proposito: es el valor mas alto para el que UNA pasada alcanza el
    techo de vegetal acuoso (ver `test_funcional_por_encima_del_hard_cap_una_pasada_solo_llega_al_hard_cap`
    para lo que pasa por encima). El test de al lado cubre ese otro regimen, para que este no
    se lea como una promesa universal."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(600))

    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    linea = plan["days"][0]["meals"][0]["ingredients"][0]
    assert _lead_g(linea) <= float(g.REALISM_VEG_VOLUME_CAP_G), (
        f"el band-closer dejo {linea!r} sobre el techo de realismo "
        f"({g.REALISM_VEG_VOLUME_CAP_G} g) en el estado PERSISTIDO"
    )


def test_funcional_por_encima_del_hard_cap_una_pasada_solo_llega_al_hard_cap(
        chain_offline, monkeypatch):
    """ALCANCE HONESTO del fix, anclado para que nadie lo lea de mas.

    Los topes por gramos de `_cap_unrealistic_portions` son una cascada `if/elif`: el techo
    DURO generico `LINE_GRAM_HARD_CAP` (600 g) casa primero y deja sin evaluar el techo de
    vegetal acuoso (250 g). Medido: 601/770/1645 g -> 600 g en la 1a pasada, 250 g en la 2a
    (punto fijo estable). O sea, una inflacion por encima de 600 g queda ACOTADA a 600 g en
    este callsite, no recortada al techo de su clase.

    Es semantica PREEXISTENTE, identica en el callsite hermano (CAPS_LAST_WORD dentro de
    `_fpc`), y se deja igual a proposito para no crear asimetria entre los dos. Si algun dia
    se hace converger la cascada, este test debe ACTUALIZARSE (no borrarse): pasaria a
    esperar `<= REALISM_VEG_VOLUME_CAP_G`.
    """
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(1645))

    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    linea = plan["days"][0]["meals"][0]["ingredients"][0]
    obtenido = _lead_g(linea)
    assert obtenido <= float(g.LINE_GRAM_HARD_CAP), (
        f"ni siquiera el techo DURO generico se aplico tras el band-closer: {linea!r}"
    )
    assert obtenido > float(g.REALISM_VEG_VOLUME_CAP_G), (
        f"la cascada converge en una sola pasada ({linea!r}): el fix quedo mejor de lo "
        f"documentado — actualiza este test y el comentario de db_plans.py en vez de borrarlos"
    )


def test_funcional_con_el_knob_off_la_linea_inflada_se_persiste(chain_offline, monkeypatch):
    """Control del knob Y control del propio test: con el rollback puesto, el defecto
    reaparece intacto. Si este test pasara con el knob en True, el fixture no estaria
    midiendo nada."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(600))
    monkeypatch.setattr(g, "CAPS_AFTER_BAND_CLOSER", False)

    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    linea = plan["days"][0]["meals"][0]["ingredients"][0]
    assert _lead_g(linea) > float(g.REALISM_VEG_VOLUME_CAP_G), (
        f"con MEALFIT_CAPS_AFTER_BAND_CLOSER=false el chain debe quedar como antes "
        f"(defecto incluido); se obtuvo {linea!r}"
    )


def test_funcional_el_queso_reinflado_vuelve_al_techo_por_comida(chain_offline, monkeypatch):
    """El caso vivo LITERAL: 360 g de queso cottage en una comida (techo 180 g)."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(360))

    plan = _plan_con("180 g de queso cottage", nombre="Tostadas de Yuca con Queso",
                     comida="Almuerzo")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    linea = plan["days"][0]["meals"][0]["ingredients"][0]
    assert _lead_g(linea) <= float(g.MEAL_CHEESE_CAP_G), (
        f"queso sobre el techo por comida ({g.MEAL_CHEESE_CAP_G} g) tras el band-closer: {linea!r}"
    )


def test_funcional_el_raw_tambien_queda_recortado(chain_offline, monkeypatch):
    """`ingredients_raw` es lo que COMPRA la lista (leccion de P1-CAP-RAW-BY-FOOD). Un re-cap
    que solo arregla el display deja la receta diciendo una cosa y la lista comprando otra."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(600))

    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    raw = plan["days"][0]["meals"][0]["ingredients_raw"][0]
    assert _lead_g(raw) <= float(g.REALISM_VEG_VOLUME_CAP_G), (
        f"raw sin recortar: la lista compraria {raw!r}"
    )


def test_funcional_una_porcion_realista_no_se_toca(chain_offline, monkeypatch):
    """Solo-bajar: el re-cap no puede inventar comida ni mover una linea que ya cumple.
    Se neutraliza el band-closer para aislar el efecto del re-cap."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", lambda *a, **k: 0)

    plan = _plan_con("120 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    assert _lead_g(plan["days"][0]["meals"][0]["ingredients"][0]) <= 120.0, (
        "el re-cap SUBIO una porcion que ya cumplia el techo"
    )


def test_funcional_idempotente_re_ejecutar_el_chain_no_recorta_mas(chain_offline, monkeypatch):
    """Idempotencia: el segundo pase sobre el estado ya recortado es no-op. Sin esto, cada
    superficie que corre el chain (6) encogeria el plato un poco mas."""
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(600))

    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)
    primera = _lead_g(plan["days"][0]["meals"][0]["ingredients"][0])

    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", lambda *a, **k: 0)
    chain_offline.apply_plan_quality_finalize_chain(plan)
    segunda = _lead_g(plan["days"][0]["meals"][0]["ingredients"][0])

    assert segunda == primera, f"chain no idempotente: {primera} -> {segunda}"


def test_funcional_el_chain_sigue_siendo_fail_safe_si_el_recap_revienta(
        chain_offline, monkeypatch):
    """El shield pre-INSERT NUNCA debe bloquear el INSERT. Un cap que lanza no puede perder
    un plan de 13 min de pipeline (la leccion de P0-PERSIST-TXN-IDLE)."""
    import graph_orchestrator as g

    def _boom(*a, **k):
        raise RuntimeError("cap roto a proposito")

    monkeypatch.setattr(g, "_cap_unrealistic_portions", _boom)
    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)  # no debe propagar
    assert plan.get("grocery_start_date"), "el chain se abortó: los pases posteriores no corrieron"


# ═════════════════════ 3 · El gemelo del assemble (hallazgo, no fix) ═════════════════════

def test_parser_assemble_cierra_los_caps_de_realismo_delegando_en_el_chain():
    """`assemble_plan_node` tiene su propio rebalance post-quantize
    (`_rebalance_day_macros_to_target`) y su recap CLINICO (P1-ASSEMBLE-CLINICAL-RECAP, que
    cubre DM2/bariatrico, NO realismo). Los caps de realismo de assemble corren antes de ese
    rebalance... pero la cola de assemble delega en `apply_plan_quality_finalize_chain`, que
    ejecuta `_fpc` (caps de realismo) y ahora tambien este re-cap. O sea: el agujero del
    gemelo lo cierra este mismo fix, siempre que la cola siga delegando en el chain.

    Este test ancla esa DEPENDENCIA: si alguien saca el chain de la cola de assemble, el
    rebalance post-quantize se queda otra vez sin cap detras."""
    i_reb = _GO.index("_rebalance_day_macros_to_target(")
    i_reb = _GO.index("if _drift and _rebalance_day_macros_to_target(")
    i_chain = _GO.find("apply_plan_quality_finalize_chain", i_reb)
    assert i_chain != -1, (
        "la cola de assemble ya no delega en apply_plan_quality_finalize_chain: el rebalance "
        "post-quantize se queda sin caps de realismo detras (mismo agujero, otra superficie)"
    )


def test_parser_el_recap_clinico_no_cubre_los_caps_de_realismo():
    """Distincion load-bearing: `reapply_clinical_portion_caps` (P1-ASSEMBLE-CLINICAL-RECAP /
    P1-UPDATE-RECAP-ALL-SURFACES) re-aplica los caps CLINICOS (DM2 alto-IG, bariatrico) y no
    sabe nada de los techos de realismo. Confundirlos llevaria a creer que este fix ya existia."""
    cuerpo = _fn_body(_GO, "reapply_clinical_portion_caps")
    assert "_cap_unrealistic_portions" not in cuerpo and "_cap_cheese_dumps_final" not in cuerpo, (
        "el recap clinico ahora tambien capea realismo — revisar si este fix quedo duplicado"
    )
