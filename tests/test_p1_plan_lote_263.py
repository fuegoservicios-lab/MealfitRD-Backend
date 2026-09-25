# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-263 · 2026-09-25] Los suplementos elegidos llegan a CADA día del plan: ni más, ni menos.

Batería final (whey + creatina + omega-3, con estatina): el plan no traía ningún suplemento. La autocrítica reemplaza
el día con lo que devuelve su corrector, cuyo modelo no tiene el campo `supplements`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import suplementos_dia as sd  # noqa: E402
from schemas import SingleDayCorrectionModel  # noqa: E402

_WHEY = {"name": "Proteína Whey Isolate", "dose": "1 scoop de 30 g", "timing": "Post-entreno",
         "reason": "Para llegar a tus 180 g de proteína"}
_D3 = {"name": "Vitamina D3", "dose": "2000 UI", "timing": "Desayuno", "reason": "extra que nadie pidió"}
_FORM = {"includeSupplements": True, "selectedSupplements": ["whey_protein", "creatine", "omega3"]}


def _meal():
    return {"meal": "Desayuno", "name": "Mangú con huevo", "desc": "x", "prep_time": "10 min", "difficulty": "Fácil",
            "cals": 500, "protein": 30, "carbs": 50, "fats": 15, "ingredients": ["2 huevos"],
            "recipe": ["Mise en place: x", "El Toque de Fuego: x", "Montaje: x"]}


def test_la_causa_el_corrector_no_tiene_campo_de_suplementos():
    dia = {"day": 1, "meals": [_meal()], "supplements": [_WHEY]}
    corregido = SingleDayCorrectionModel(**dia).model_dump()
    assert "supplements" not in corregido                  # por eso el día corregido los perdía
    assert sd.conservar(corregido, dia)["supplements"] == [_WHEY]


def test_conservar_no_pisa_lo_que_el_corrector_si_trajo():
    otro = [{"name": "Creatina", "dose": "5 g", "timing": "x", "reason": "y"}]
    assert sd.conservar({"supplements": otro}, {"supplements": [_WHEY]})["supplements"] == otro


def test_completar_los_elegidos_en_cada_dia_ni_mas_ni_menos():
    plan = {"days": [{"day": 1, "meals": [], "supplements": [_WHEY, _D3]},
                     {"day": 2, "meals": []},
                     {"day": 3, "meals": [], "supplements": []}]}
    assert sd.completar(plan, _FORM) > 0
    for d in plan["days"]:
        claves = [sd.clave_de(s["name"]) for s in d["supplements"]]
        assert sorted(claves) == ["creatine", "omega3", "whey_protein"], (d["day"], claves)
    # la entrada que escribió el modelo se reutiliza en los días que la perdieron
    assert plan["days"][1]["supplements"][0] == _WHEY
    assert all("Vitamina D3" not in s["name"] for d in plan["days"] for s in d["supplements"])


def test_el_veto_clinico_manda():
    form = {"includeSupplements": True, "selectedSupplements": ["creatine", "omega3"],
            "medicalConditions": ["Enfermedad Renal"]}
    plan = {"days": [{"day": 1, "meals": []}]}
    sd.completar(plan, form)
    assert [sd.clave_de(s["name"]) for s in plan["days"][0]["supplements"]] == ["omega3"]
    warfarina = {"includeSupplements": True, "selectedSupplements": ["omega3"], "medications": ["Warfarina"]}
    plan = {"days": [{"day": 1, "meals": [], "supplements": []}]}
    assert sd.completar(plan, warfarina) == 0 and plan["days"][0]["supplements"] == []


def test_recomendacion_libre_solo_copia_lo_del_modelo():
    form = {"includeSupplements": True, "selectedSupplements": []}
    plan = {"days": [{"day": 1, "meals": [], "supplements": [_D3]}, {"day": 2, "meals": []}]}
    sd.completar(plan, form)
    assert plan["days"][1]["supplements"] == [_D3]
    vacio = {"days": [{"day": 1, "meals": []}]}
    assert sd.completar(vacio, form) == 0 and not vacio["days"][0].get("supplements")


def test_sin_includesupplements_no_toca_nada():
    plan = {"days": [{"day": 1, "meals": []}]}
    assert sd.completar(plan, {"includeSupplements": False, "selectedSupplements": ["creatine"]}) == 0
    assert "supplements" not in plan["days"][0]


def test_reconoce_los_nombres_libres_de_los_12():
    casos = {"Electrolitos (Sodio + Potasio + Magnesio)": "electrolytes", "Multivitamínico Completo": "multivitamin",
             "Proteína vegana de guisante": "vegan_protein", "Proteína de suero aislada": "whey_protein",
             "Colágeno hidrolizado": "collagen", "Creatina Monohidrato": "creatine", "Aminoácidos BCAA": "bcaa",
             "Pre-Entreno": "pre_workout", "Quemador termogénico": "fat_burner",
             "Omega-3 (Aceite de Pescado)": "omega3", "Probióticos": "probiotics", "Magnesio glicinato": "magnesium"}
    for nombre, clave in casos.items():
        assert sd.clave_de(nombre) == clave, nombre
    from constants import SUPPLEMENT_NAMES
    assert set(sd._POR_DEFECTO) == set(SUPPLEMENT_NAMES) == {k for k, _ in sd._CLAVES}


def test_cableado():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count('__import__("suplementos_dia").conservar(corrected_result.model_dump(), target_day)') == 2
    i = src.index("async def assemble_plan_node(")
    j = src.index("\nasync def ", i + 10)
    cuerpo = src[i:j]
    assert '__import__("suplementos_dia").completar(result, form_data)' in cuerpo
    assert cuerpo.index("P1-SUPPLEMENT-CLINICAL-GATE") < cuerpo.index('suplementos_dia").completar')
    assert "P1-PLAN-LOTE-263-SUPLEMENTOS" in (_BACKEND / "suplementos_dia.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 263
