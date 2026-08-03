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
import re
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
    """[P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] `_LAST_KNOWN_PFIX` es un marker GLOBAL de
    UN SOLO P-fix a la vez — el siguiente cierre (esta misma tanda P2) lo sobreescribe
    legítimamente. Re-anclar el valor EXACTO de este P-fix rompía con el próximo bump, la
    MISMA clase que ya se corrigió en `test_p2_help_chatbot.py::test_marker_bumped` (ahí el
    bug era comparar el marker completo lexicográficamente; acá es peor: comparar el string
    literal completo, que deja de existir en cuanto CUALQUIER P-fix futuro bumpea). El
    contrato real (formato + floor de fecha, sin re-anclar el slug) ya vive en
    `test_p3_1_last_known_pfix_freshness.py` — este test se reduce a verificar que el floor
    de fecha no retrocedió por debajo de ESTE P-fix, sin asumir que sigue siendo el vigente."""
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    marker = m.group(1)
    _fecha = re.search(r"(\d{4}-\d{2}-\d{2})\s*$", marker)
    assert _fecha, f"Marker sin fecha ISO al final (formato `Pn-X · YYYY-MM-DD`): {marker!r}"
    assert _fecha.group(1) >= "2026-08-03", (
        f"Marker sospechosamente viejo: {marker!r} (floor P2-CAPS-AFTER-BAND-CLOSER · 2026-08-03)"
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


@pytest.mark.parametrize("inflado", [601, 770, 1645])
def test_funcional_el_recap_converge_al_techo_de_clase_por_encima_del_hard_cap(
        chain_offline, monkeypatch, inflado):
    """Por encima de `LINE_GRAM_HARD_CAP` UNA pasada no basta, y el re-cap itera.

    Los topes por gramos de `_cap_unrealistic_portions` son una cascada `if/elif`: el techo
    DURO generico (600 g) casa primero y deja sin evaluar el techo de vegetal acuoso (250 g).
    Medido: 601/770/1645 g -> 600 g en la 1a pasada, 250 g en la 2a (punto fijo estable).

    Que el caso TIPICO caiga en este regimen no es teorico: el clamp x2.5 de
    `_rebalance_day_macros_to_target` es POR PASE y corre con `passes=3` (hasta x15.6
    compuesto), asi que 250 -> 770 es lo normal, no un borde. Con una sola pasada el fix
    persistia 600 g de pepino (2.4x el techo de clase) en el caso tipico y el titular "los
    techos vuelven a ser la ultima palabra" era falso.
    """
    import graph_orchestrator as g
    monkeypatch.setattr(g, "reconcile_all_macros_band_post_finalize", _inflar(inflado))

    plan = _plan_con("250 g de pepino")
    chain_offline.apply_plan_quality_finalize_chain(plan)

    linea = plan["days"][0]["meals"][0]["ingredients"][0]
    assert _lead_g(linea) <= float(g.REALISM_VEG_VOLUME_CAP_G), (
        f"inflado a {inflado} g, el chain persistio {linea!r}: el re-cap no convergio al techo "
        f"de clase ({g.REALISM_VEG_VOLUME_CAP_G} g). Si solo llego a LINE_GRAM_HARD_CAP "
        f"({g.LINE_GRAM_HARD_CAP} g), el bucle a punto fijo se rompio."
    )


def test_parser_el_recap_itera_hasta_punto_fijo():
    """Ancla estructural del bucle: sin el, la cascada `if/elif` deja el caso tipico en
    `LINE_GRAM_HARD_CAP` y el fix incumple su propio titular."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    i_ramb = body.index("_ramb(_pd,")
    i_reconcile = body.index("RECONCILE_AFTER_BAND_CLOSER", i_ramb)
    bloque = body[i_ramb:i_reconcile]
    assert "for _ in range(" in bloque, (
        "el re-cap debe iterar: una sola pasada solo alcanza LINE_GRAM_HARD_CAP"
    )
    assert "break" in bloque, (
        "el bucle debe cortar en la pasada que no cambia nada (punto fijo, no N vueltas fijas)"
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


# ═══════════ 3 · La PREMISA, con el motor real (no con un fake de `_ramb`) ═══════════
#
# Los funcionales de arriba monkeypatchean `_ramb` para aislar el ORDEN del chain. Eso deja sin
# probar la premisa de la que cuelga todo el fix: que el rebalance REAL re-infla una linea que el
# cap acaba de recortar. Vivia solo en prosa y en un harness desechable — o sea, si la premisa
# dejara de ser cierta, ningun test del repo se enteraria.
#
# Aqui corre el motor de verdad (`_rebalance_day_macros_to_target` + los dos caps) sobre una
# `IngredientNutritionDB(rows=...)` con per-100g PUBLICADOS (USDA). Offline: `rows=` evita el
# pool. No es un benchmark sintetico de calidad — lo que se mide es el MECANISMO, no la nutricion.

_ROWS_REALES = [
    {"name": "Pepino", "aliases": ["pepinos"], "kcal_per_100g": 15,
     "protein_g_per_100g": 0.65, "carbs_g_per_100g": 3.63, "fats_g_per_100g": 0.11,
     "category": "vegetal", "density_g_per_unit": 300, "density_g_per_cup": 120},
    {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 130,
     "protein_g_per_100g": 2.69, "carbs_g_per_100g": 28.17, "fats_g_per_100g": 0.28,
     "category": "cereal", "density_g_per_cup": 158},
    {"name": "Batata", "aliases": ["batatas"], "kcal_per_100g": 86,
     "protein_g_per_100g": 1.57, "carbs_g_per_100g": 20.12, "fats_g_per_100g": 0.05,
     "category": "viveres", "density_g_per_unit": 130},
    {"name": "Pechuga de pollo", "aliases": ["pollo"], "kcal_per_100g": 165,
     "protein_g_per_100g": 31.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 3.6,
     "category": "proteina"},
    {"name": "Aceite de oliva", "aliases": ["aceite"], "kcal_per_100g": 884,
     "protein_g_per_100g": 0.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 100.0,
     "category": "grasa", "density_g_per_cup": 216},
]

_TARGET_CARBS, _TARGET_FATS, _TARGET_PROT = 220.0, 60.0, 140.0


def _db_real():
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=_ROWS_REALES)


def _dia_tipico(db):
    """Dia BAJO en carbos (ratio inicial ~0.32) con el pepino como 1 de 3 lineas
    carbo-dominantes — el reparto realista, no el borde."""
    import graph_orchestrator as g

    def _meal(nombre, comida, lineas):
        m = {"name": nombre, "meal": comida, "ingredients": list(lineas),
             "ingredients_raw": list(lineas)}
        p = c = f = 0.0
        for s in lineas:
            mc = db.macros_from_ingredient_string(s) or {}
            p += mc.get("protein") or 0
            c += mc.get("carbs") or 0
            f += mc.get("fats") or 0
        m["protein"], m["carbs"], m["fats"] = round(p), round(c), round(f)
        m["cals"] = round(4 * m["protein"] + 4 * m["carbs"] + 9 * m["fats"])
        return m

    return [_meal("Almuerzo criollo", "Almuerzo",
                  ["150 g de arroz blanco", "120 g de pechuga de pollo",
                   "10 g de aceite de oliva"]),
            _meal("Ensalada fria de pepino", "Cena",
                  ["250 g de pepino", "100 g de batata", "100 g de pechuga de pollo"])]


def _pepino(meals):
    for m in meals:
        for s in m["ingredients"]:
            if "pepino" in s.lower():
                return s
    return ""


def _carbs(meals):
    import graph_orchestrator as g
    return sum(g._meal_macro_num(m.get("carbs")) for m in meals)


def test_premisa_el_rebalance_REAL_reinfla_una_linea_recien_capada():
    """LA PREMISA DEL FIX. Sin esto, todo lo demas es teoria.

    El clamp x2.5 de `_rebalance_day_macros_to_target` es POR PASE y la funcion corre con
    `passes=3` (hasta x15.6 compuesto), asi que la inflacion tipica supera de largo el techo
    DURO de 600 g — por eso una sola pasada del cap no basta."""
    import graph_orchestrator as g
    db = _db_real()
    meals = _dia_tipico(db)

    assert _lead_g(_pepino(meals)) == 250.0, "fixture: la linea entra YA capada al techo"

    g._rebalance_day_macros_to_target(meals, _TARGET_CARBS, _TARGET_FATS, db,
                                      target_protein=_TARGET_PROT)

    inflado = _lead_g(_pepino(meals))
    assert inflado > float(g.REALISM_VEG_VOLUME_CAP_G), (
        f"el rebalance REAL no re-inflo la linea capada ({inflado} g): si esto deja de pasar, "
        f"P2-CAPS-AFTER-BAND-CLOSER ya no tiene motivo de existir"
    )
    assert inflado > float(g.LINE_GRAM_HARD_CAP), (
        f"la inflacion tipica ({inflado} g, medido 770) deberia superar el techo DURO de "
        f"{g.LINE_GRAM_HARD_CAP} g — es lo que hace insuficiente UNA pasada del cap"
    )


def test_premisa_el_recap_iterado_devuelve_la_linea_al_techo_y_el_dia_sigue_en_banda():
    """El caso TIPICO completo, con el motor real: el re-cap a punto fijo devuelve el pepino a
    250 g y el dia se queda DENTRO de la banda del gate (medido 0.914, piso 0.90).

    El margen es estrecho a proposito en la documentacion: 8.6 pp de drift sobre 220 g de
    target. No se anade un segundo rebalance para recuperarlo — dos guardas sobre el mismo
    campo oscilan, y el trade-off aceptado es que gana el cap."""
    import graph_orchestrator as g
    db = _db_real()
    meals = _dia_tipico(db)

    g._rebalance_day_macros_to_target(meals, _TARGET_CARBS, _TARGET_FATS, db,
                                      target_protein=_TARGET_PROT)
    carbs_tras_ramb = _carbs(meals)

    for _ in range(3):  # mismo bucle a punto fijo que el chain
        n = g._cap_unrealistic_portions([{"meals": meals}], db=db)
        n += g._cap_cheese_dumps_final([{"meals": meals}], db=db)
        if not n:
            break

    assert _lead_g(_pepino(meals)) <= float(g.REALISM_VEG_VOLUME_CAP_G), _pepino(meals)

    ratio = _carbs(meals) / _TARGET_CARBS
    assert g.BAND_SCORE_LOWER <= ratio <= g.BAND_SCORE_UPPER, (
        f"el caso TIPICO sale de banda tras el re-cap (ratio {ratio:.3f}, banda "
        f"[{g.BAND_SCORE_LOWER}, {g.BAND_SCORE_UPPER}]); carbs {carbs_tras_ramb}->{_carbs(meals)}. "
        f"Si esto empieza a fallar, el trade-off del fix cambio y hay que re-medirlo."
    )


def test_premisa_un_dia_ya_en_banda_es_no_op():
    """Control negativo con el motor real: sin inflacion no hay recorte."""
    import graph_orchestrator as g
    db = _db_real()
    meals = _dia_tipico(db)
    # dia ya en banda: no se corre el rebalance, solo los caps
    antes = [list(m["ingredients"]) for m in meals]
    for _ in range(3):
        n = g._cap_unrealistic_portions([{"meals": meals}], db=db)
        n += g._cap_cheese_dumps_final([{"meals": meals}], db=db)
        if not n:
            break
    assert [m["ingredients"] for m in meals] == antes, "el re-cap toco un dia que ya cumplia"


# ═════════════════════ 4 · El gemelo del assemble (hallazgo, no fix) ═════════════════════

def test_parser_assemble_cierra_los_caps_de_realismo_delegando_en_el_chain():
    """`assemble_plan_node` tiene su propio rebalance post-quantize
    (`_rebalance_day_macros_to_target`) y su recap CLINICO (P1-ASSEMBLE-CLINICAL-RECAP, que
    cubre DM2/bariatrico, NO realismo). Los caps de realismo de assemble corren antes de ese
    rebalance... pero la cola de assemble delega en `apply_plan_quality_finalize_chain`, que
    ejecuta `_fpc` (caps de realismo) y ahora tambien este re-cap. O sea: el agujero del
    gemelo lo cierra este mismo fix, siempre que la cola siga delegando en el chain.

    Este test ancla esa DEPENDENCIA: si alguien saca el chain de la cola de assemble, el
    rebalance post-quantize se queda otra vez sin cap detras.

    Se ancla el CALLSITE EJECUTABLE (`_adb(_apqfc, ...)`), no la mencion del nombre. La
    primera version buscaba `apply_plan_quality_finalize_chain` a secas y casaba con el
    COMENTARIO que hay encima ("SSOT `db_plans.apply_plan_quality_finalize_chain`"): borrar la
    llamada dejando el comentario habria mantenido el test VERDE. Es la trampa "comentario que
    documenta codigo borrado", que este repo ya ha pagado varias veces."""
    i_reb = _GO.index("if _drift and _rebalance_day_macros_to_target(")
    i_chain = _GO.find("_adb(_apqfc,", i_reb)
    assert i_chain != -1, (
        "la cola de assemble ya no INVOCA el chain (`_adb(_apqfc, ...)`): el rebalance "
        "post-quantize se queda sin caps de realismo detras (mismo agujero, otra superficie). "
        "Ojo: que el nombre siga apareciendo en un comentario no es que se ejecute."
    )
    assert "await " in _GO[max(0, i_chain - 20):i_chain], (
        "el callsite del chain en assemble debe seguir siendo el await real, no una mencion"
    )


def test_parser_el_recap_clinico_no_cubre_los_caps_de_realismo():
    """Distincion load-bearing: `reapply_clinical_portion_caps` (P1-ASSEMBLE-CLINICAL-RECAP /
    P1-UPDATE-RECAP-ALL-SURFACES) re-aplica los caps CLINICOS (DM2 alto-IG, bariatrico) y no
    sabe nada de los techos de realismo. Confundirlos llevaria a creer que este fix ya existia."""
    cuerpo = _fn_body(_GO, "reapply_clinical_portion_caps")
    assert "_cap_unrealistic_portions" not in cuerpo and "_cap_cheese_dumps_final" not in cuerpo, (
        "el recap clinico ahora tambien capea realismo — revisar si este fix quedo duplicado"
    )
