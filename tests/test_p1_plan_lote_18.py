# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-18 · 2026-09-12] Decimoctavo lote del plan de pendientes: C0, la línea base sobre un corpus FIJO.

La revisión del 7-sep (`docs/audits/2026-09-07-coherencia-culinaria/REVISION-DE-LOS-GAPS.md`) dejó dos prerrequisitos
para todo el bloque culinario: una línea base con huella sobre un corpus FIJO —la del 6-sep dejó de ser reproducible en
14 h porque `plan_data` es vivo y el shift encoge los días— y las 80 etiquetas humanas (del dueño). Aquí se cierra la
primera mitad:

  · `culinary_corpus.py` congela en un fichero EXACTAMENTE lo que las capas leen (`days`, `_culinary_judge_history`,
    estado) más el catálogo del índice culinario, con huella de contenido; el orden de lectura y el orden de claves
    no entran en la huella; un paso cambiado o un alias nuevo del catálogo sí.
  · `culinary_baseline.py --corpus F` mide ese fichero; `--congelar` escribe `docs/culinary_baseline_<fecha>.json`
    y `--verificar` re-mide y compara: 0 reproduce, 3 mismas reglas y cifras distintas (el medidor no es
    determinista), 4 sin línea base comparable. `--congelar` sin `--corpus` se niega: la ventana viva no se puede
    volver a medir mañana.
  · Congelado el 2026-09-12 y verificado dos veces: 5 planes, 64 comidas en `days`, catálogo 349 filas, huella
    `087cfc31d3105f79`. La flota es pequeña tras la purga del 09-11 (los 96 planes del 6-sep ya no existen); el
    instrumento vale igual y se re-congela cuando haya flota. `docs/culinary_baseline.json` (la foto viva del
    6/7-sep) se conserva como historia, no comparable.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_corpus as cc  # noqa: E402

_CORPUS = _BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json"
_BASE_FIJA = _BACKEND / "docs" / "culinary_baseline_2026_09_12.json"
_BASE_VIVA = _BACKEND / "docs" / "culinary_baseline.json"


def _baseline_mod():
    sys.path.append(str(_BACKEND / "scripts"))  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym
    import culinary_baseline
    return culinary_baseline


# ────────────────────────────────────────────────────────────── un corpus sintético

_CAT = [
    {"name": "Pollo", "aliases": ["pechuga de pollo"], "category": "proteina", "ready_to_eat": False,
     "prep_methods": ["hervido", "plancha", "horneado"]},
    {"name": "Arroz blanco", "aliases": [], "category": "cereal", "ready_to_eat": False, "prep_methods": ["hervido"]},
    {"name": "Tomate", "aliases": ["tomates"], "category": "vegetal", "ready_to_eat": True, "prep_methods": ["crudo"]},
    {"name": "Ajo", "aliases": ["dientes de ajo"], "category": "aromatico", "ready_to_eat": False, "prep_methods": ["sofrito"]},
]


def _plan(pid: str, paso_extra: str | None = None) -> dict:
    recipe = ["Hierve el arroz blanco 15 minutos.", "Cocina el pollo a la plancha y sirve con el tomate."]
    if paso_extra:
        recipe.append(paso_extra)
    return {
        "id": pid, "created_at": "2026-09-10 10:00:00+00:00", "updated_at": "2026-09-11 10:00:00+00:00", "revision": 3,
        "plan_data": {
            "name": "Plan de prueba", "generation_status": "complete",
            "aggregated_shopping_list": [{"name": "Pollo", "qty": 300}],   # NO se congela: la medición no lo lee
            "_plan_policy": {"requested": {}},                            # tampoco
            "days": [{"day": "Lunes", "meals": [{
                "meal": "Almuerzo", "name": "Pollo a la plancha con arroz y tomate",
                "ingredients": ["150 g de pollo", "60 g de arroz blanco crudo", "1 tomate", "1 diente de ajo"],
                "ingredients_raw": ["150 g de pollo", "60 g de arroz blanco", "1 tomate", "1 diente de ajo"],
                "recipe": recipe,
            }]}],
            "_culinary_judge_history": [{"ts": "2026-09-10", "model": "x", "violations": [], "action_taken": "warn_only"}],
        },
    }


def _corpus(planes=None, cat=None, **kw) -> dict:
    return cc.congelar(planes or [_plan("a" * 36), _plan("b" * 36)], cat or _CAT, motivo="test",
                       git_sha="deadbeef", **kw)


# ────────────────────────────────────────────────────────────── la huella

def test_la_huella_del_plan_no_depende_del_orden_de_claves_ni_de_un_viaje_por_json():
    a = _plan("a" * 36)["plan_data"]
    b = json.loads(json.dumps(a, ensure_ascii=False))
    c = {k: a[k] for k in reversed(list(a))}
    assert cc.huella_plan(a) == cc.huella_plan(b) == cc.huella_plan(c)
    assert re.fullmatch(r"[0-9a-f]{16}", cc.huella_plan(a))


def test_un_paso_cambiado_cambia_la_huella_del_plan_y_la_del_corpus():
    c1 = _corpus()
    c2 = _corpus(planes=[_plan("a" * 36, paso_extra="Fríe el pollo en aceite abundante."), _plan("b" * 36)])
    assert c1["planes"][0]["huella"] != c2["planes"][0]["huella"]
    assert c1["planes"][1]["huella"] == c2["planes"][1]["huella"]
    assert c1["huella"] != c2["huella"]


def test_el_catalogo_entra_en_la_huella_del_corpus():
    """El vocabulario del detector cambia con el catálogo: mismo corpus de planes + un alias nuevo ⇒ otra huella."""
    cat2 = copy.deepcopy(_CAT)
    cat2[0]["aliases"].append("pollo entero")
    assert _corpus(cat=_CAT)["huella"] != _corpus(cat=cat2)["huella"]
    assert cc.huella_catalogo(_CAT) != cc.huella_catalogo(cat2)


def test_el_orden_de_lectura_no_entra_en_la_huella():
    c1 = _corpus(planes=[_plan("a" * 36), _plan("b" * 36)])
    c2 = _corpus(planes=[_plan("b" * 36), _plan("a" * 36)], cat=list(reversed(_CAT)))
    assert c1["huella"] == c2["huella"]
    assert [p["plan_id"] for p in c2["planes"]] == sorted(p["plan_id"] for p in c2["planes"])


def test_solo_se_congela_lo_que_las_capas_leen():
    r = cc.rebanada_plan(_plan("a" * 36))
    assert set(r["plan_data"]) == {"days", "_culinary_judge_history", "generation_status", "name"}
    assert "aggregated_shopping_list" not in r["plan_data"] and "_plan_policy" not in r["plan_data"]
    assert r["revision"] == 3 and r["dias"] == 1 and r["comidas"] == 1
    c = _corpus()
    assert set(c["catalogo_filas"][0]) == set(cc.COLUMNAS_CATALOGO)
    assert c["catalogo"]["filas"] == len(_CAT) and c["comidas"] == 2 and c["planes_n"] == 2


def test_un_corpus_editado_a_mano_no_se_mide(tmp_path):
    c = _corpus()
    assert cc.verificar_integridad(c) == []
    c["planes"][0]["plan_data"]["days"][0]["meals"][0]["recipe"].append("Añade sal.")
    fallos = cc.verificar_integridad(c)
    assert any(f.startswith("plan aaaaaaaa") for f in fallos) and any(f.startswith("corpus") for f in fallos)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="no íntegro"):
        cc.cargar(p)


# ────────────────────────────────────────────────────────────── medir el corpus fijo

def test_medir_el_mismo_corpus_dos_veces_da_las_mismas_cifras_en_cualquier_orden():
    cb = _baseline_mod()
    c1 = _corpus(planes=[_plan("a" * 36), _plan("b" * 36, paso_extra="Pica 3 dientes de ajo y sofríelos.")])
    c2 = _corpus(planes=[_plan("b" * 36, paso_extra="Pica 3 dientes de ajo y sofríelos."), _plan("a" * 36)])
    r1, r2, r3 = cb.medir_corpus(c1), cb.medir_corpus(c1), cb.medir_corpus(c2)
    for k in cb.CIFRAS:
        assert r1[k] == r2[k] == r3[k], k
    assert r1["comidas"] == 2 and r1["planes"] == 2


def test_medir_corpus_publica_la_huella_del_fichero_y_declara_fijo():
    cb = _baseline_mod()
    c = _corpus()
    r = cb.medir_corpus(c, fichero="x/corpus.json")
    assert r["corpus"]["fijo"] is True and r["corpus"]["huella"] == c["huella"]
    assert r["corpus"]["fichero"] == "x/corpus.json" and r["corpus"]["huella_ventana_viva"] != c["huella"]
    assert r["computation"]["catalogo_huella"] == c["catalogo"]["huella"]
    assert r["computation"]["reglas_huella"] == cc.huella_reglas()
    assert "corpus FIJO x/corpus.json" in cb.render(r)
    # la ventana viva sigue declarándose como tal
    assert cb._medir_filas(cc.filas_para_medir(c), _CAT)["corpus"]["fijo"] is False


def test_congelar_o_verificar_la_ventana_viva_esta_prohibido(monkeypatch):
    """Antes de tocar la base: la única respuesta a `--congelar` sin corpus es negarse (exit 2)."""
    cb = _baseline_mod()
    monkeypatch.setattr(cb, "medir", lambda *a, **k: pytest.fail("no debe medir la ventana viva para congelar"))
    monkeypatch.setattr(sys, "argv", ["culinary_baseline.py", "--congelar"])
    assert cb.main() == 2
    monkeypatch.setattr(sys, "argv", ["culinary_baseline.py", "--verificar"])
    assert cb.main() == 2


def test_verificar_tiene_tres_salidas_y_ninguna_ambigua(tmp_path):
    cb = _baseline_mod()
    c = _corpus()
    r = cb.medir_corpus(c)
    destino = tmp_path / "base.json"
    assert cb.verificar(r, None, destino) == 4                                   # sin línea base
    otra = copy.deepcopy(r)
    otra["corpus"]["huella"] = "0000000000000000"
    assert cb.verificar(r, otra, destino) == 4                                   # otro corpus
    assert cb.verificar(r, copy.deepcopy(r), destino) == 0                       # reproduce
    distinta = copy.deepcopy(r)
    distinta["determinista"]["comidas"] += 1
    assert cb.verificar(r, distinta, destino) == 3                               # mismas reglas, cifras distintas
    distinta["computation"]["reglas_huella"] = "otra"
    assert cb.verificar(r, distinta, destino) == 0                               # el delta es del código, dicho


def test_la_linea_base_de_cada_corpus_lleva_su_fecha_y_la_foto_viva_se_conserva():
    cb = _baseline_mod()
    assert cb.baseline_path_for({"congelado_at": "2026-09-12T18:32:13+00:00"}).name == "culinary_baseline_2026_09_12.json"
    viva = json.loads(_BASE_VIVA.read_text(encoding="utf-8"))
    assert viva["planes"] == 96 and "fijo" not in viva["corpus"], "la foto del 6/7-sep es historia: no se reescribe"


# ────────────────────────────────────────────────────────────── el corpus congelado del 2026-09-12

def test_el_corpus_del_09_12_esta_integro_y_su_linea_base_lo_reproduce():
    corpus = cc.cargar(_CORPUS)
    base = json.loads(_BASE_FIJA.read_text(encoding="utf-8"))
    assert corpus["huella"] == base["corpus"]["huella"] == "087cfc31d3105f79"
    assert base["corpus"]["fijo"] is True and base["planes"] == corpus["planes_n"] and base["comidas"] == corpus["comidas"]
    assert base["computation"]["catalogo_huella"] == corpus["catalogo"]["huella"]
    if cc.huella_reglas() != base["computation"]["reglas_huella"]:
        pytest.skip("las reglas de culinary_coherence.py cambiaron desde la línea base: "
                    "`python scripts/culinary_baseline.py --corpus scripts/data/culinary_corpus_2026_09_12.json "
                    "--verificar` dice el delta; re-congela con --congelar si es el esperado")
    cb = _baseline_mod()
    r = cb.medir_corpus(corpus, fichero=_CORPUS.as_posix())
    for k in cb.CIFRAS:
        assert r[k] == base[k], k


def test_el_congelador_es_solo_lectura():
    src = (_BACKEND / "scripts" / "congela_corpus_culinario.py").read_text(encoding="utf-8")
    assert "conn.read_only = True" in src
    sql = " ".join(re.findall(r'"([^"]*)"', src))
    assert not re.search(r"\b(UPDATE|INSERT|DELETE|TRUNCATE|ALTER|DROP)\b", sql), sql


# ────────────────────────────────────────────────────────────── docs y marker

def test_docs_y_plan():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "### El corpus fijo" in doc and "087cfc31d3105f79" in doc and "--verificar" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| C0 | ✅ 2026-09-12 · mitad técnica" in plan


def test_marker_bumpeado():
    """«No anterior a este lote», no «igual a hoy»: el lote siguiente vuelve a bumpear el marker (lección de LOTE-13)."""
    import app

    assert "[P1-PLAN-LOTE-18 · 2026-09-12]" in (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-12", app._LAST_KNOWN_PFIX
