# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-34 · 2026-09-13] F6: el barrido descripción-USDA ↔ nombre sobre las 288 filas del catálogo con fdc_id.

Un fdc_id es una afirmación: «esta fila ES ese alimento de USDA». La auditoría del 19-ago cerró los ids compartidos y dejó
escrito lo que no veía — un id ÚNICO mal apuntado. `scripts/catalog_fdc_sweep.py` compara, fila a fila, la identidad (los
tokens del nombre frente a la descripción de USDA) y los valores (P/C/G/kcal con tolerancia absoluta). Medido: 19 filas
«mal apuntadas» y 54 «a revisar» de 288; revisadas a mano una a una, 22 ids apuntaban a OTRO alimento con los valores del
catálogo intactos (Tamarindo → verdolaga, Hígado de res → T-bone, Leche de almendras → un experimento con tomates…), 11
proxies no declarados, 7 filas con valores sin la fuente que decían tener, 17 kcal por Atwater general y 2 glosas
inglesas de otra fruta/hierba. La migración `p1_plan_lote_34_catalogo_fdc_2026_09_13.sql` escribe SOLO procedencia (ningún
valor nutricional) y cada corrección se verificó contra USDA antes de escribirse.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_MIG = "p1_plan_lote_34_catalogo_fdc_2026_09_13.sql"


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location("catalog_fdc_sweep", _BACKEND / "scripts" / "catalog_fdc_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def artefacto():
    return json.loads((_BACKEND / "scripts" / "data" / "catalog_fdc_sweep_2026_09_13.json").read_text(encoding="utf-8"))


# ─────────────── el instrumento ───────────────

def test_el_plural_ingles_no_come_letras(sweep):
    """La primera versión leía «limes» como «lim» y acusaba a Limón de no ser lima (identidad 0)."""
    assert sweep._singular("limes") == "lime" and sweep._singular("tomatoes") == "tomato"
    assert sweep._singular("berries") == "berry" and sweep._singular("peaches") == "peach"
    assert sweep.identidad(["Lime"], "Limes, raw") == 1.0
    assert sweep.identidad(["Tamarind"], "Purslane, raw") == 0.0


def test_la_convencion_de_energia_no_es_otro_alimento(sweep):
    """34 frente a 28 kcal en una verdura es Atwater general frente a específico, no un id mal apuntado."""
    dev, _, _ = sweep.desvio({"kcal": 34.0, "protein": 2.8, "fats": 0.4, "carbs": 6.6},
                             {"kcal": 28.0, "protein": 2.8, "fats": 0.4, "carbs": 6.6})
    assert dev == 0.0
    dev, peor, _ = sweep.desvio({"kcal": 239.0, "protein": 2.8, "fats": 0.6, "carbs": 62.5},
                                {"kcal": 20.0, "protein": 2.0, "fats": 0.4, "carbs": 3.4})
    assert dev > 1000 and peor == "carbs"


def test_el_veredicto_necesita_las_dos_senales_para_mal_apuntado(sweep):
    assert sweep.veredicto(1.0, 5.0) == "coincide"
    assert sweep.veredicto(0.0, 1743.7) == "mal_apuntado"
    assert sweep.veredicto(1.0, 52.7) == "revisar"      # Salmón: el nombre casa, los valores son de otro salmón
    assert sweep.veredicto(0.0, 0.0) == "revisar"       # Gambas / shrimp: el nombre en español no dice nada


def test_solo_lectura_y_la_clave_por_cabecera(sweep):
    src = inspect.getsource(sweep)
    assert "conn.read_only = True" in src
    assert '"X-Api-Key"' in src and "api_key=" not in src, "la clave de USDA nunca va en la URL"
    assert not re.search(r"\b(INSERT INTO|UPDATE master_ingredients|DELETE FROM)\b", src)


# ─────────────── lo medido y lo revisado ───────────────

def test_el_artefacto_cubre_todas_las_filas_con_fdc_id(artefacto):
    assert artefacto["n_filas"] == 288 and len(artefacto["filas"]) == 288
    assert sum(artefacto["cuenta"].values()) == 288
    assert artefacto["umbrales"]["identidad_min"] == 0.4 and "suelo_denominador" in artefacto


def test_cada_fila_marcada_tiene_veredicto_humano(artefacto):
    permitidos = {"id_corregido", "proxy_declarado", "valores_propios", "kcal_atwater_anotada",
                  "id_correcto_gloss_corregido", "correcto"}
    rev = artefacto["revision"]
    for f in artefacto["filas"]:
        if f["veredicto"] in ("mal_apuntado", "sin_respuesta"):
            assert f["name"] in rev, f"{f['name']}: marcada {f['veredicto']} y sin revisar"
    for name, r in rev.items():
        assert r["veredicto_final"] in permitidos, name
    corregidos = [r for r in rev.values() if r["veredicto_final"] == "id_corregido"]
    assert len(corregidos) == 22
    assert len({r["despues"] for r in corregidos}) == 22, "dos filas reclamando el mismo id"
    assert all(r["antes"] != r["despues"] for r in corregidos)


def test_las_correcciones_emblematicas(artefacto):
    rev = artefacto["revision"]
    assert rev["Tamarindo"]["despues"] == 167763 and "Purslane" in rev["Tamarindo"]["descripcion_antes"]
    assert rev["Hígado de res"]["despues"] == 169451
    assert rev["Salmón"]["despues"] == 173686, "el catálogo pesa el salmón SALVAJE (142 kcal), no el de granja"
    assert rev["Cangrejo"]["despues"] == 174204 and rev["Percebes"]["veredicto_final"] == "proxy_declarado"
    assert rev["Frijoles pintos"]["veredicto_final"] == "valores_propios", "sinónimo de Judías pintas, que tiene el id"


# ─────────────── la migración ───────────────

def test_la_migracion_vive_en_los_dos_directorios_y_es_identica():
    a = (_BACKEND / "migrations" / _MIG)
    assert a.exists()
    raiz = _BACKEND.parent / "migrations" / _MIG
    if raiz.parent.exists():
        assert raiz.read_bytes() == a.read_bytes(), "espejo SSOT (P3-MIGRATIONS-SSOT)"


def test_la_migracion_solo_escribe_procedencia():
    sql = (_BACKEND / "migrations" / _MIG).read_text(encoding="utf-8")
    sets = re.findall(r"UPDATE public\.master_ingredients SET (.*?) WHERE", sql, re.S)
    assert len(sets) >= 40
    for s in sets:
        sin_literales = re.sub(r"'(?:[^']|'')*'", "''", s)      # el texto de procedencia lleva comas y «=»
        cols = {c.split("=")[0].strip() for c in sin_literales.split(",")}
        assert cols <= {"fdc_id", "nutrition_source", "nutrition_source_ref", "name_en"}, cols
    sin_literales = re.sub(r"'(?:[^']|'')*'", "''", sql)       # los «;» del texto de procedencia no cierran nada
    updates = re.findall(r"UPDATE public\.master_ingredients .*?;", sin_literales, re.S)
    assert len(updates) == len(sets) and all("WHERE name = " in u for u in updates)
    assert sql.count("DO $$") == 3 and "fdc_id compartidos" in sql


def test_docs_plan_marker():
    doc = (_BACKEND / "docs" / "catalog_provenance_audit.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-34" in doc and "catalog_fdc_sweep.py" in doc and "Tamarindo" in doc
    assert "P1-PLAN-LOTE-34" in (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 34
