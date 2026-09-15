# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-63 · 2026-09-15] (lote 40 del plan 38-44 · C5/C6) El juez por código contra la verdad humana.

Sobre la columna refrescada del lote 38 (`maquina_juez_2026-09-15`) y la anotación del dueño: 28 hallazgos seguros,
precisión ESTRICTA 0 % (0/28) y «por sustancia» 82,1 % (23/28). El juez ve el mismo defecto que el dueño, sobre el mismo
alimento, y lo nombra con uno de sus 5 códigos donde la rúbrica pone una clase del determinista. Aquí:

(a) el post-proceso de `run_culinary_judge` marca `dudosa` los códigos del knob `MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES`
    y ninguno más, sin quitar, añadir ni reescribir hallazgos; el knob se parsea, se registra y sólo acepta los 5 del schema;
(b) el mapa por sustancia vive en el marcador (`SUSTANCIA_JUEZ`), no en `culinary_coherence` ni en `graph_orchestrator`, y
    la rúbrica estricta no cambia;
(c) la tabla por código: estricta y por sustancia, y `--observacion` cuenta esos códigos como `[dudosa]`;
(d) el default del knob es el que sale de la medición con el criterio del plan (estricta < 25 %, n ≥ 4);
(e) docs, knob documentado y marker ≥ 63.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import typing
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"
_ANOT = _BACKEND / "docs" / "culinary_golden_anotaciones_angelo.json"
_COL = "maquina_juez_2026-09-15"
_OBS = frozenset({"paso_incoherente", "tecnica_impropia", "nombre_no_corresponde"})


def _mod(rel: str, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, _BACKEND / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gsc():
    return _mod("scripts/culinary_golden_score.py", "culinary_golden_score_l63")


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator
    return graph_orchestrator


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────────── (a) el post-proceso y el knob

def test_el_post_proceso_marca_dudosa_solo_los_codigos_en_observacion(go):
    viol = [go.CulinaryViolation(day=1, meal="Cena", tipo=t, detalle=f"queja {t}", severidad="high")
            for t in go._CULINARY_JUDGE_TIPOS]
    viol.append(go.CulinaryViolation(day=2, meal="Almuerzo", tipo="combo_absurdo", detalle="la dudosa del propio juez",
                                     severidad="minor", certeza="dudosa"))
    antes = [(v.tipo, v.detalle, v.severidad, v.day, v.meal) for v in viol]
    out = go._culinary_judge_observacion(go.CulinaryJudgeReport(violations=viol),
                                         frozenset({"paso_incoherente", "tecnica_impropia"}))
    assert [(v.tipo, v.detalle, v.severidad, v.day, v.meal) for v in out.violations] == antes, "los mismos hallazgos"
    assert [(v.tipo, v.certeza) for v in out.violations] == [
        (t, "dudosa" if t in ("paso_incoherente", "tecnica_impropia") else "segura") for t in go._CULINARY_JUDGE_TIPOS
    ] + [("combo_absurdo", "dudosa")]
    assert go._culinary_judge_observacion(None) is None
    rep = go.CulinaryJudgeReport(violations=[go.CulinaryViolation(day=1, meal="Cena", tipo="paso_incoherente",
                                                                  detalle="x", severidad="high")])
    assert go._culinary_judge_observacion(rep, frozenset()).violations[0].certeza == "segura"


def test_el_knob_se_parsea_se_registra_y_solo_acepta_los_del_schema(go):
    from knobs import get_knobs_registry_snapshot
    assert go._culinary_judge_observacion_codes(" Paso_Incoherente, tecnica_impropia ,inventado,,") == \
        frozenset({"paso_incoherente", "tecnica_impropia"})
    assert go._culinary_judge_observacion_codes("") == frozenset()
    assert set(go._CULINARY_JUDGE_TIPOS) == set(typing.get_args(go.CulinaryViolation.model_fields["tipo"].annotation))
    assert "MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES" in get_knobs_registry_snapshot()
    assert '"MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES", "paso_incoherente,tecnica_impropia,nombre_no_corresponde"' in \
        _src("graph_orchestrator.py")
    if os.environ.get("MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES") is None:
        assert go.CULINARY_JUDGE_OBSERVACION_CODES == _OBS


def test_run_culinary_judge_pasa_por_el_post_proceso():
    src = _src("graph_orchestrator.py")
    cuerpo = re.search(r"async def run_culinary_judge\(.*?(?=\n(?:def |async def |class |#))", src, re.S).group(0)
    assert "return _culinary_judge_observacion(_rep)" in cuerpo
    assert "tooltip-anchor: P1-PLAN-LOTE-63-OBSERVACION" in src


# ─────────────────────────────── (b) el mapa por sustancia, sólo en el marcador

def test_el_mapa_por_sustancia_vive_en_el_marcador(gsc):
    codigos_juez = set().union(*gsc.RUBRICA.values()) - gsc._CODIGOS_DET
    assert set(gsc.SUSTANCIA_JUEZ) == codigos_juez
    assert set().union(*gsc.SUSTANCIA_JUEZ.values()) <= set(gsc.RUBRICA)
    for rel in ("culinary_coherence.py", "graph_orchestrator.py"):
        assert "SUSTANCIA_JUEZ" not in _src(rel), rel
    assert gsc.rubrica_por_sustancia()["cantidad_inconsistente"] >= {"V4", "paso_incoherente"}
    assert gsc.RUBRICA["cantidad_inconsistente"] == {"V4", "V6", "V7e"}, "la rúbrica estricta no cambia"


# ─────────────────────────────── (c) la tabla por código

def _caso(id_, ingredientes, juez, defectos, veredicto="defecto"):
    return {"id": id_, "plan": f"plan-{id_}", "estrato": "ambas", "ingredientes": ingredientes,
            "maquina_determinista": [], "maquina_juez": [], _COL: juez,
            "anotaciones": [{"anotador": "A", "veredicto": veredicto, "defectos": defectos}]}


def test_la_tabla_por_codigo_estricta_y_por_sustancia(gsc):
    d = {"casos": [
        _caso("a", ["1 tostada de casabe", "40 g de queso fresco"],
              ["paso_incoherente: la lista dice 1 tostada de casabe y el paso 3 tuesta 2 (componente: casabe)"],
              [{"clase": "cantidad_inconsistente", "severidad": "minor", "evidencia": "1 y 2", "alimento": "Casabe"}]),
        _caso("b", ["40 g de arroz blanco crudo", "2 huevos"],
              ["tecnica_impropia: el arroz blanco crudo se incorpora sin cocer (componente: arroz blanco crudo)",
               "combo_absurdo [dudosa]: huevo con arroz en el desayuno (componente: huevo)"],
              [{"clase": "seco_sin_coccion", "severidad": "high", "evidencia": "crudo", "alimento": "Arroz blanco"}]),
    ]}
    t = gsc.tabla_juez_por_codigo(d, {}, _COL)
    pi, ti, ca = t["codigos"]["paso_incoherente"], t["codigos"]["tecnica_impropia"], t["codigos"]["combo_absurdo"]
    assert (pi["n"], pi["tp"], pi["fp"], pi["sustancia"]) == (1, 0, 1, 1), pi
    assert pi["precision"] == 0 and pi["precision_sustancia"] == 100
    assert (ti["n"], ti["tp"], ti["sustancia"]) == (1, 0, 1), ti
    assert (ca["n"], ca["dudosas"]) == (0, 1), ca
    assert gsc.codigos_en_observacion(t, n_min=1) == ["paso_incoherente", "tecnica_impropia"]
    obs = gsc.tabla_juez_por_codigo(d, {}, _COL, frozenset({"paso_incoherente"}))
    assert (obs["codigos"]["paso_incoherente"]["n"], obs["codigos"]["paso_incoherente"]["dudosas"]) == (0, 1)


def test_el_marcador_cuenta_la_observacion_como_dudosa(gsc, monkeypatch):
    c = _caso("c", ["1 tostada de casabe"], ["paso_incoherente: x (componente: casabe)", "combo_absurdo: y"], [])
    assert [h["codigo"] for h in gsc._hallazgos_maquina(c, _COL)] == ["paso_incoherente", "combo_absurdo"]
    monkeypatch.setattr(gsc, "OBSERVACION", frozenset({"paso_incoherente"}))
    assert [h["codigo"] for h in gsc._hallazgos_maquina(c, _COL)] == ["combo_absurdo"]
    assert gsc.contar_dudosas({"casos": [c]}, _COL) == 1


# ─────────────────────────────── (d) el default del knob sale de la medición

def test_el_default_del_knob_sale_de_la_medicion(gsc):
    d = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    t = gsc.tabla_juez_por_codigo(d, gsc.cargar_anotaciones([str(_ANOT)]), _COL)
    assert set(gsc.codigos_en_observacion(t)) == _OBS
    assert (t["total"]["n"], t["total"]["tp"], t["total"]["sustancia"]) == (28, 0, 23)


# ─────────────────────────────── (e) docs, knob y marker

def test_docs_knob_y_marker():
    doc = _src("docs/culinary_coherence.md")
    assert "P1-PLAN-LOTE-63" in doc and "Por sustancia" in doc and "82,1" in doc
    assert re.search(r"Recomendación para C6[^\n]*2026-09-15", doc)
    knobs = _src("docs/knobs_reference.md")
    assert "| `MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES` | `paso_incoherente,tecnica_impropia,nombre_no_corresponde` |" in knobs
    assert "P1-PLAN-LOTE-63" in _src("docs/plan_pendientes_2026_09_11.md")
    assert re.search(r"^\| 40 \| C5 / C6 \(parte medible\) \| ✅ HECHO \(`P1-PLAN-LOTE-63`",
                     _src("docs/plan_agente_lotes_38_43_2026_09_14.md"), re.M)
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"), re.M)
    assert m and int(m.group(1)) >= 63 and m.group(2) >= "2026-09-15"
    score = _src("scripts/culinary_golden_score.py")
    assert "tooltip-anchor: P1-PLAN-LOTE-63-SUSTANCIA" in score and "--juez-por-codigo" in score
