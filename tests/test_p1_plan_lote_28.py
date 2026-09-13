# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-28 · 2026-09-12] C5 (segunda parte, b) del plan de pendientes: CUL-P1-06 (un juez que admite la creatividad
válida: componente, intención y estado INCIERTO) y CUL-P1-07 (benchmark culinario de todas las superficies, con dobles
offline, artefacto, informe pareado y modo real que exige presupuesto).

Lo que se prueba:
  · el esquema del juez lleva `certeza` / `componente` / `intencion` con defaults que no rompen a los jueces viejos; la
    rúbrica los pide; en review una `dudosa` se observa y no bloquea; el esquema de hallazgo sube de versión;
  · el marcador estricto excluye las `[dudosa]` salvo `--con-dudosas`, y saca los casos de desarrollo del holdout;
  · el sampler marca las dudosas; el benchmark corre offline sobre un plan, escribe un artefacto reconstruible, compara
    pareado, se niega a gastar sin presupuesto y no escribe en la base.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import culinary_coherence as cc  # noqa: E402
import graph_orchestrator as go  # noqa: E402

_SRC_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _mod(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND / rel)   # por ruta: `scripts/` no va a sys.path (lote 13)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ─────────────── el juez: componente, intención, certeza ───────────────

def test_el_esquema_del_juez_gana_certeza_componente_e_intencion_con_defaults_compatibles():
    v = go.CulinaryViolation(day=1, meal="Cena", tipo="combo_absurdo", detalle="x", severidad="minor")
    assert v.certeza == "segura" and v.componente is None and v.intencion is None
    d = go.CulinaryViolation(day=1, meal="Cena", tipo="combo_absurdo", detalle="x", severidad="minor", certeza="dudosa",
                             componente="salsa de mango", intencion="fusion").model_dump()
    assert d["certeza"] == "dudosa" and d["componente"] == "salsa de mango" and d["intencion"] == "fusion"
    with pytest.raises(Exception):
        go.CulinaryViolation(day=1, meal="Cena", tipo="combo_absurdo", detalle="x", severidad="minor", certeza="quizas")
    assert cc.FINDING_SCHEMA_VERSION == "2026-09-12.certeza"
    assert cc.judge_context(country="DO")["schema"] == "2026-09-12.certeza"


def test_la_rubrica_pide_los_tres_campos_y_dice_que_la_dudosa_se_observa():
    r = go._culinary_judge_rubric_for_country("DO")
    for frag in ("componente (el ingrediente o la parte del plato", "intencion (qué pretende el plato", "certeza ('segura'",
                 "una violación dudosa se OBSERVA y no", "bloquea)"):
        assert frag in r, frag
    beta = go._culinary_judge_rubric_for_country("ES")
    assert "certeza ('segura'" in beta and "dominican" not in beta.split("TIPOS CANÓNICOS")[0].split("REGLA DE HORARIO")[1].lower()


def test_en_review_una_dudosa_se_observa_y_no_bloquea():
    i = _SRC_GO.index("_cj_seguras = [v for v in _cj_viol if str(v.get(\"certeza\") or \"segura\") != \"dudosa\"]")
    j = _SRC_GO.index("_cj_viol = _cj_resolve(plan, _cj_viol)")
    assert j < i, "la certeza se lee después de atar cada queja a su comida"
    bloque = _SRC_GO[i:i + 3000]
    assert '"action_taken": ("blocked" if (_cj_seguras and CULINARY_JUDGE_GUARD == "block")' in bloque
    assert 'if _cj_seguras and CULINARY_JUDGE_GUARD == "block":' in bloque
    assert '"violations": _cj_viol,' in bloque, "la historia guarda TODAS las quejas, dudosas incluidas: se observan"


# ─────────────── el marcador estricto ───────────────

def test_el_marcador_excluye_las_dudosas_salvo_que_se_pidan_y_saca_los_casos_de_desarrollo():
    gsc = _mod("scripts/culinary_golden_score.py", "culinary_golden_score_l28")
    caso = {"id": "c1", "maquina_juez": ["combo_absurdo: salami con avena", "tecnica_impropia [dudosa]: yogur a la plancha"], "maquina_determinista": []}
    assert [h["codigo"] for h in gsc._hallazgos_maquina(caso, "maquina_juez")] == ["combo_absurdo"]
    gsc.INCLUIR_DUDOSAS = True
    try:
        assert [h["codigo"] for h in gsc._hallazgos_maquina(caso, "maquina_juez")] == ["combo_absurdo", "tecnica_impropia"]
    finally:
        gsc.INCLUIR_DUDOSAS = False
    d = {"casos": [caso, {"id": "dev-1", "maquina_juez": ["x [dudosa]: y"], "maquina_determinista": []}]}
    assert gsc.contar_dudosas(d) == 2
    d2 = gsc.excluir_casos(d, ["dev-1"])
    assert [c["id"] for c in d2["casos"]] == ["c1"] and d2["excluidos_dev"] == 1 and gsc.excluir_casos(d, []) is d
    src = (_BACKEND / "scripts" / "culinary_golden_score.py").read_text(encoding="utf-8")
    assert "--con-dudosas" in src and "--excluir-dev" in src and 'r["dudosas_excluidas"]' in src
    sampler = (_BACKEND / "scripts" / "culinary_golden_sample.py").read_text(encoding="utf-8")
    assert '" [dudosa]" if str(v.get("certeza") or "") == "dudosa"' in sampler


# ─────────────── el benchmark de superficies ───────────────

_CAT = [
    {"name": "Avena", "aliases": ["avena en hojuelas"], "category": "Granos", "prep_methods": ["cocido"]},
    {"name": "Leche descremada", "aliases": ["leche"], "category": "Lácteos", "prep_methods": ["ninguno"]},
    {"name": "Pechuga de pollo", "aliases": ["pollo"], "category": "Proteínas", "prep_methods": ["plancha"]},
]


def _plan():
    return {"plan_id": "p1", "plan_data": {"days": [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Avena con leche", "ingredients": ["40 g de avena", "200 ml de leche descremada"], "ingredients_raw": ["40 g de avena", "200 ml de leche descremada"],
         "recipe": ["Mise en place: mide 90 g de avena y 200 ml de leche descremada.", "Cocina 10 min."], "protein": "8g", "carbs": "30g", "fats": "4g", "calories": 190},
        {"meal": "Almuerzo", "name": "Pollo a la plancha", "ingredients": ["150 g de pechuga de pollo"], "ingredients_raw": ["150 g de pechuga de pollo"],
         "recipe": ["Asa el pollo a la plancha 8 min por lado."], "protein": "35g", "carbs": "0g", "fats": "4g", "calories": 180}]}]}}


def test_el_benchmark_corre_offline_escribe_un_artefacto_reconstruible_y_compara_pareado(tmp_path, capsys):
    bench = _mod("scripts/bench_superficies_culinarias.py", "bench_superficies_l28")
    art = bench.correr([_plan()], _CAT, ("closers", "expand"), modo="offline")
    assert art["schema_version"] == 1 and art["n_planes"] == 1 and set(art["superficies"]) == {"closers", "expand"}
    assert art["superficies"]["expand"]["planes"][0]["estado"] == "sin_cadena_determinista"
    assert art["superficies"]["closers"]["planes"][0]["estado"] in ("medido", "no_medido")
    assert art["reglas_huella"] == cc.rules_fingerprint()
    art["modo_catalogo"] = "nombres"
    p = tmp_path / "a.json"
    p.write_text(json.dumps(art, ensure_ascii=False, default=str), encoding="utf-8")
    assert bench.main(["--informe", str(p)]) == 0
    out = capsys.readouterr().out
    assert "bench superficies" in out and "catálogo SIN nutrición" in out and "closers" in out
    b = dict(art, git_sha="otro")
    b["superficies"] = json.loads(json.dumps(art["superficies"]))
    b["superficies"]["closers"]["n_nuevos"] = 3
    q = tmp_path / "b.json"
    q.write_text(json.dumps(b, ensure_ascii=False, default=str), encoding="utf-8")
    assert bench.main(["--comparar", str(p), str(q)]) == 0
    out2 = capsys.readouterr().out
    assert "pareado" in out2 and re.search(r"closers\s+0\s+→\s+3", out2)


def test_las_superficies_prometidas_existen_y_swap_y_chunk_comparten_el_contrato_final():
    bench = _mod("scripts/bench_superficies_culinarias.py", "bench_superficies_l28b")
    assert set(bench.SUPERFICIES) == {"insert", "quality", "chunk_t2", "swap", "modify", "closers", "degradado", "expand"}
    assert set(bench.ADAPTADORES) == set(bench.SUPERFICIES) and bench.ADAPTADORES["modify"] is bench.ADAPTADORES["swap"]
    src = (_BACKEND / "scripts" / "bench_superficies_culinarias.py").read_text(encoding="utf-8")
    assert "apply_final_contract(p.get(\"days\") or [], db)" in src, "el swap termina en el contrato final"
    dbp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert "_rfc_tail = apply_final_contract" in dbp or "from recipe_contract import apply_final_contract as _rfc_tail" in dbp, "la cola del INSERT/chunk también"
    assert not re.search(r"\b(INSERT INTO|UPDATE meal_plans|DELETE FROM)\b", src), "el benchmark no escribe planes de usuarios"
    assert "medir_cadena" in src and "_instalar_dobles" in src and "P1-PLAN-LOTE-28-BENCH-SUPERFICIES" in src


def test_el_modo_real_se_niega_sin_presupuesto(capsys):
    bench = _mod("scripts/bench_superficies_culinarias.py", "bench_superficies_l28c")
    assert bench.main(["--real", "--sin-guardar"]) == 2
    assert "presupuesto" in capsys.readouterr().out
    assert bench.main(["--real", "--perfil", "x.json", "--presupuesto-usd", "0", "--sin-guardar"]) == 2


# ─────────────── docs, plan, marcador ───────────────

def test_docs_plan_y_marcador():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-28" in doc and "certeza" in doc and "bench_superficies_culinarias" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert re.search(r"^\| C5 \| ✅ 2026-09-12", plan, re.M) and "P1-PLAN-LOTE-28" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 28 and m.group(2) >= "2026-09-12"
    assert len(_SRC_GO.splitlines()) <= 53100
