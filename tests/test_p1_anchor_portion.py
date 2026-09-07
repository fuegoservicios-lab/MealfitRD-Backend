# -*- coding: utf-8 -*-
"""[P1-ANCHOR-PORTION · 2026-09-07] La ración de un ancla: el «cuánto» que faltaba.

`stapleAnchors` sabía decir **cuándo** (franjas) y **cuán a menudo** (min/max por 7 días), pero no
**cuánto**. Sin ese campo, «desayuno 10 claras» no tenía dónde entrar en el sistema: no es una
alergia, ni un disgusto, ni un básico más — es una CANTIDAD, y el formulario no tenía casilla.

## Lo que se midió antes de escribir una línea

Sobre 96 planes vivos, las claras entregadas son **1, 2, 3, 4, 5 y 6**, y ahí se cortan en seco.
Una distribución que muere justo en el tope es la firma de `MAX_EGG_WHITES_PER_MEAL = 6`: nadie ha
recibido nunca más, pidiera lo que pidiera. Y el catálogo YA distingue las tres formas del huevo
—`Clara de huevo` (33 g/ud, 0,17 g de grasa), `Huevo` (50 g, 9,51) y `Yema de huevo` (17 g,
26,5)— con `fdc_id` distintos y 33+17=50: la identidad nutricional no era el problema.

## Lo que este P-fix hace, y lo que deliberadamente NO

**Transporta la intención y hace ruidoso el recorte.** Los topes siguen truncando exactamente
igual — pero el truncado deja de ser mudo, porque queda como relajación «pediste N, aplicamos M»
en el panel «solicitaste / aplicamos / por qué».

**No sube ningún tope.** Honrar la petición toca cinco capas (prompt, dos caps, solver y gate de
variedad) y mezclarlo aquí significaría subir un límite global: darle 10 claras a quien no las
pidió, que es justo lo que el encargo prohíbe.

## La unidad es obligatoria

Sin unidad, `150` de pollo se leería como 150 **unidades**. Es la lección de
`P1-UNKNOWN-UNIT-NOT-WHOLE`: una unidad desconocida no es una unidad entera. Una ración sin unidad
se descarta **diciéndolo**, nunca adivinando.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import plan_policy as pp  # noqa: E402


def _form(portion=None, name="Clara de huevo"):
    anchor = {"name": name, "slots": ["desayuno"], "min_per_7d": 7, "max_per_7d": 7}
    if portion is not None:
        anchor["portion"] = portion
    return {"stapleFoods": [name], "stapleAnchors": [anchor]}


def _anchor(form):
    return (pp.policy_from_form(form).get("food_anchors") or [{}])[0]


# ── el campo viaja ────────────────────────────────────────────────────────────────────────────
def test_la_racion_pedida_llega_a_la_politica():
    a = _anchor(_form({"qty": 10, "unit": "unidad"}))
    assert a["portion"] == {"qty": 10.0, "unit": "unidad"}


def test_doce_claras_se_conservan_como_doce():
    """El caso del encargo. Aquí solo se exige que la CIFRA sobreviva a la política: que el plato
    final las lleve es el paso siguiente, y depende de cinco capas que este P-fix no toca."""
    a = _anchor(_form({"qty": 12, "unit": "unidad"}))
    assert a["portion"]["qty"] == 12.0


def test_un_ancla_sin_racion_sigue_igual_que_siempre():
    """Quien no pide cantidad no puede notar este cambio: `None`, no un default inventado."""
    assert _anchor(_form())["portion"] is None


# ── la unidad obligatoria ─────────────────────────────────────────────────────────────────────
def test_sin_unidad_la_racion_se_descarta():
    """`150` de pollo sin unidad se leería como 150 unidades. Una unidad desconocida no es una
    unidad entera (P1-UNKNOWN-UNIT-NOT-WHOLE)."""
    assert _anchor(_form({"qty": 150}))["portion"] is None
    assert _anchor(_form({"qty": 150, "unit": "  "}))["portion"] is None


def test_una_cantidad_absurda_o_negativa_no_pasa():
    assert _anchor(_form({"qty": 0, "unit": "unidad"}))["portion"] is None
    assert _anchor(_form({"qty": -3, "unit": "unidad"}))["portion"] is None
    assert _anchor(_form({"qty": "diez", "unit": "unidad"}))["portion"] is None


def test_la_unidad_se_normaliza_pero_no_se_inventa():
    a = _anchor(_form({"qty": 8, "unit": "  Unidad  "}))
    assert a["portion"] == {"qty": 8.0, "unit": "unidad"}


# ── el recorte deja de ser mudo ───────────────────────────────────────────────────────────────
def _compilar(portion):
    req = _form(portion)
    pol = pp.compile_from_form(req)
    return pol.get("effective") or {}, pol.get("relaxations") or []


def test_una_racion_desmesurada_se_ajusta_Y_SE_DICE():
    eff, rels = _compilar({"qty": 5000, "unit": "unidad"})
    r = [x for x in rels if x.get("reason_code") == "portion_out_of_range"]
    assert r, rels
    assert r[0]["requested"]["qty"] == 5000.0
    assert r[0]["applied"]["qty"] == pp._PORTION_MAX


def test_la_cota_es_anti_basura_y_no_un_limite_clinico():
    """El encargo excluye explícitamente inventar un límite diario. 12 claras —y bastante más—
    tienen que pasar sin relajación: la cota solo frena entradas absurdas."""
    for q in (7, 8, 10, 12, 20, 30):
        eff, rels = _compilar({"qty": q, "unit": "unidad"})
        assert not [x for x in rels if str(x.get("reason_code")).startswith("portion")], (q, rels)
        a = (eff.get("food_anchors") or [{}])[0]
        assert a.get("portion", {}).get("qty") == float(q)


def test_cada_motivo_nuevo_tiene_copy_para_el_usuario():
    """Sin copy, la relajación vive en el jsonb y NO llega al panel «solicitaste / aplicamos /
    por qué» — que es justo la mitad del arreglo."""
    for code in ("portion_invalid", "portion_out_of_range"):
        assert code in pp._REASON_COPY, code
    textos = pp.explain_relaxations([
        {"reason_code": "portion_out_of_range", "requested": {"qty": 5000, "unit": "unidad"},
         "applied": {"qty": 100.0, "unit": "unidad"}, "evidence": {}}])
    assert textos and "cantidad" in textos[0].lower()


# ── el alcance declarado ──────────────────────────────────────────────────────────────────────
def test_este_pfix_NO_toca_los_topes_del_motor():
    """Si alguien sube el cap aquí, se lo da a TODO el mundo — incluido quien no pidió nada. El
    tope se mueve en su propio P-fix, leyendo esta política."""
    # Se mira el USO, no la mención: con `in src` este guard se ponía rojo por citar el cap en un
    # comentario que explica precisamente por qué NO se toca aquí. Un guard que prohíbe hablar de
    # algo empuja a borrar la explicación, que es lo contrario de lo que se quiere.
    import ast
    arbol = ast.parse((_BACKEND / "plan_policy.py").read_text(encoding="utf-8"))
    usados = {n.id for n in ast.walk(arbol) if isinstance(n, ast.Name)}
    usados |= {n.attr for n in ast.walk(arbol) if isinstance(n, ast.Attribute)}
    for prohibido in ("MAX_EGG_WHITES_PER_MEAL", "MAX_EGG_WHITES_PER_DAY", "_REALISM_COUNT_CAPS"):
        assert prohibido not in usados, (
            f"plan_policy USA {prohibido}: subir el tope aquí se lo daría a todo el mundo, "
            f"incluido quien no pidió nada")
