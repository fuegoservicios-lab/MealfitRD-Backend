# -*- coding: utf-8 -*-
"""[P1-JUDGE-REVISION-STAMP · 2026-09-06] Atar cada juicio del juez a la versión que juzgó.

`_culinary_judge_history` guardaba `{ts, model, violations, action_taken}` — **nada que atara una
entrada a una versión del plan**. Con eso, «el juez se quejó y lo arreglamos» y «se quejó y lo
entregamos» son indistinguibles, y su tasa se lee como si fuera de defectos ENTREGADOS.

No era una sospecha. Medido el 2026-09-06 sobre 96 planes vivos:

- de las **37** quejas juzgables del tipo «X no aparece en la lista», **6 nombraban algo que SÍ
  está** en el plan entregado (almendras, pistachos, guineítos verdes, queso cottage);
- el sub-patrón más citado —«los ingredientes dicen 4¾ lonjas/pedazos de queso» cuando el plato
  lleva cottage— aparece **0 veces de 23** líneas de queso con lonja en planes vivos. Existe el
  agujero que lo produce (`_VAGUE_SLICE_FOOD_RE` es `(queso|jamon|…)\\b.*`, y sí casa «queso
  cottage»), pero la reparación lo convierte antes de entregar: **el usuario nunca lo ve**.

## Las tres decisiones que este test ancla

1. **La huella cubre lo que el juez MIRA** — nombre, ingredientes y pasos. Reutilizar
   `services.compute_plan_hash` habría sido tentador («fuente única de verdad para detectar si un
   plan cambió»), pero hashea ingredientes y suplementos: el bucket más grande del juez
   (`paso_incoherente`) es de PASOS, y un paso reparado dejaría ese hash quieto justo en los casos
   que más importan. Una huella que no cubre lo que se juzgó reintroduce la misma ambigüedad,
   sólo que más difícil de ver.
2. **Tres estados, no dos.** Una entrada sin sello no dice ni que sí ni que no. Colapsar ese
   `None` hacia cualquier lado fabrica una cifra — que es exactamente el error que el P-fix cierra.
3. **La advertencia nace del código, no del JSON.** `--congelar` reescribe el fichero entero: una
   advertencia enmendada a mano en `culinary_baseline.json` se habría borrado en silencio en el
   siguiente congelado.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402

_PLAN = {"days": [{"meals": [{"meal": "almuerzo", "name": "Pollo guisado",
                              "ingredients": ["150 g de pollo", "1 cebolla"],
                              "recipe": ["Sofríe la cebolla", "Añade el pollo"]}]}]}


def _con(**cambios):
    import copy
    p = copy.deepcopy(_PLAN)
    p["days"][0]["meals"][0].update(cambios)
    return p


# ── la huella ─────────────────────────────────────────────────────────────────────────────────
def test_la_huella_es_estable_para_el_mismo_plan():
    assert cc.judged_fingerprint(_PLAN) == cc.judged_fingerprint(_con())


def test_cambiar_un_PASO_mueve_la_huella():
    """El caso que decide el diseño: `paso_incoherente` es el bucket más grande del juez, así que
    una huella ciega a los pasos diría «es el mismo plan» justo donde más importa."""
    otro = _con(recipe=["Sofríe la cebolla", "Añade el pollo y el comino"])
    assert cc.judged_fingerprint(otro) != cc.judged_fingerprint(_PLAN)


def test_cambiar_el_NOMBRE_mueve_la_huella():
    """`nombre_no_corresponde` son 53 de las quejas: si el nombre no entra, esas nunca serían
    decidibles."""
    assert cc.judged_fingerprint(_con(name="Pollo al horno")) != cc.judged_fingerprint(_PLAN)


def test_cambiar_un_INGREDIENTE_mueve_la_huella():
    otro = _con(ingredients=["150 g de pollo", "1 cebolla", "15 g de almendras"])
    assert cc.judged_fingerprint(otro) != cc.judged_fingerprint(_PLAN)


def test_NO_es_compute_plan_hash():
    """`compute_plan_hash` se declara «fuente única de verdad para detectar si un plan cambió» y
    aun así no sirve aquí: no mira pasos ni nombres. Que ambas existan es deliberado; fusionarlas
    devolvería la ambigüedad."""
    from services import compute_plan_hash
    solo_paso = _con(recipe=["Sofríe la cebolla", "Añade el pollo y el comino"])
    assert compute_plan_hash(solo_paso) == compute_plan_hash(_PLAN), (
        "si esto cambia, compute_plan_hash empezo a mirar pasos y hay que reconsiderar el diseño")
    assert cc.judged_fingerprint(solo_paso) != cc.judged_fingerprint(_PLAN)


def test_fail_open_y_plan_vacio():
    """Un sello que revienta bloquearía la entrega de un plan por un problema de OBSERVACIÓN."""
    assert cc.judged_fingerprint({}) is None
    assert cc.judged_fingerprint({"days": [{"meals": ["no soy un dict"]}]}) is None
    assert cc.judged_fingerprint(None) is None


# ── los tres estados ──────────────────────────────────────────────────────────────────────────
def test_sin_sello_la_respuesta_es_DESCONOCIDO_no_falso():
    """Las entradas anteriores a este P-fix no llevan sello. Devolver `False` convertiría «no lo
    sé» en «se reparó», que es la confusión que el P-fix cierra."""
    assert cc.judgment_covers_delivered({"ts": "x", "model": "y"}, _PLAN) is None
    assert cc.judgment_covers_delivered({}, _PLAN) is None


def test_con_sello_igual_dice_que_SI_juzgo_lo_entregado():
    e = {"judged_fingerprint": cc.judged_fingerprint(_PLAN)}
    assert cc.judgment_covers_delivered(e, _PLAN) is True


def test_con_sello_distinto_dice_que_NO():
    """El plan cambió después del juicio: la queja describe algo que ya no se entregó."""
    e = {"judged_fingerprint": cc.judged_fingerprint(_PLAN)}
    reparado = _con(ingredients=["150 g de pollo", "1 cebolla", "15 g de almendras"])
    assert cc.judgment_covers_delivered(e, reparado) is False


def test_un_plan_ilegible_tambien_es_desconocido():
    e = {"judged_fingerprint": "abc123"}
    assert cc.judgment_covers_delivered(e, {}) is None


# ── el sello se estampa de verdad ─────────────────────────────────────────────────────────────
def test_el_orquestador_estampa_la_huella_en_cada_entrada():
    """Un sello que nadie escribe es indistinguible de no tenerlo — la lección de P1-G y del
    catálogo INERTE de F2."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    i = src.index('_cj_hist.append({')
    bloque = src[i:i + 500]
    assert '"judged_fingerprint": _cj_fingerprint(plan)' in bloque, bloque[:300]
    assert "from culinary_coherence import judged_fingerprint as _cj_fingerprint" in src


# ── la advertencia enmendada ──────────────────────────────────────────────────────────────────
def test_la_advertencia_vive_en_el_CODIGO_no_solo_en_el_json():
    """`--congelar` reescribe el JSON entero: una enmienda hecha solo a mano en el fichero se
    borraría en silencio en el siguiente congelado."""
    src = (_BACKEND / "scripts" / "culinary_baseline.py").read_text(encoding="utf-8")
    assert "ADVERTENCIA = (" in src
    assert '"advertencia": ADVERTENCIA' in src
    assert "REPARO antes de entregar" in src


def test_el_json_congelado_lleva_la_segunda_razon_con_su_cifra():
    """Sin la cifra, «parte de sus quejas» es una impresión; con ella es una medición que otro
    puede rebatir."""
    d = json.loads((_BACKEND / "docs" / "culinary_baseline.json").read_text(encoding="utf-8"))
    adv = d["advertencia"].lower()
    assert "overfitting" in adv and "sin verdad de referencia" in adv
    assert "reparo antes de entregar" in adv
    m = d["juez_sobre_lo_entregado"]["medicion_2026_09_06"]
    assert m["quejas_falta_en_la_lista_juzgables"] == 37
    assert m["nombraban_algo_que_hoy_SI_esta"] == 6
    assert m["queso_no_lonjeable_en_lonjas_en_planes_vivos"] == 0


def test_la_foto_congelada_declara_que_NO_es_decidible():
    """Se tomó antes de que existiera el sello. Decirlo en el propio fichero evita que alguien
    la lea dentro de un año como si sus entradas fueran juicios sobre lo entregado."""
    d = json.loads((_BACKEND / "docs" / "culinary_baseline.json").read_text(encoding="utf-8"))
    assert d["juez_sobre_lo_entregado"]["decidible"] is False
    assert "judged_fingerprint" in d["juez_sobre_lo_entregado"]["motivo"]


def test_el_medidor_cuenta_los_tres_estados_por_separado():
    src = (_BACKEND / "scripts" / "culinary_baseline.py").read_text(encoding="utf-8")
    assert 'cobertura["si" if cubre else ("no" if cubre is False else "desconocido")]' in src
    assert '"juez_sobre_lo_entregado": dict(cobertura)' in src
