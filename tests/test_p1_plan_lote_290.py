# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-290 · 2026-09-25] Suplementos en la Alacena: datos, aislamiento del generador y Nevera que se
enciende sola. Spec: docs/superpowers/specs/2026-09-25-suplementos-alacena-design.md. Tooltip-anchor: P1-PLAN-LOTE-290"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


WHEY = {"gramos_porcion": 31, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5}


# ── Task 1: datos y SSOT ────────────────────────────────────────────────────────────────────────────────────────────

def test_macros_de_porciones_multiplica_la_etiqueta():
    import suplementos as s
    assert s.macros_de_porciones(WHEY, 2) == {"kcal": 240.0, "protein_g": 48.0, "carbs_g": 6.0, "fats_g": 3.0}
    assert s.macros_de_porciones(None, 2) is None
    assert s.macros_de_porciones(WHEY, 0) is None


def test_etiqueta_absurda_se_rechaza():
    import suplementos as s
    assert s.etiqueta_valida(WHEY) == WHEY
    assert s.etiqueta_valida({**WHEY, "kcal": 1200}) is None          # 1200 kcal en 31 g: foto mal leída
    assert s.etiqueta_valida({**WHEY, "protein_g": 40}) is None       # más macros que gramos de porción
    assert s.etiqueta_valida({"kcal": 0, "gramos_porcion": 5}) == {"gramos_porcion": 5, "kcal": 0, "protein_g": 0,
                                                              "carbs_g": 0, "fats_g": 0}   # creatina
    assert s.etiqueta_valida("no") is None


def test_estimados_cubren_las_claves_del_formulario():
    import suplementos as s
    from constants import SUPPLEMENT_NAMES
    assert set(s.ESTIMADOS) == set(SUPPLEMENT_NAMES)
    assert s.ESTIMADOS["creatine"]["kcal"] == 0
    assert s.ESTIMADOS["whey_protein"]["protein_g"] == 24


def test_migracion_idempotente_y_en_los_dos_directorios():
    rel = "p1_plan_lote_290_suplementos_alacena.sql"
    a = (_BACKEND / "migrations" / rel).read_text(encoding="utf-8")
    raiz = _BACKEND.parent / "migrations" / rel
    if raiz.parent.exists():
        assert raiz.exists() and raiz.read_text(encoding="utf-8") == a
    for frag in ("ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'food'",
                 "ADD COLUMN IF NOT EXISTS serving_label JSONB",
                 "ADD COLUMN IF NOT EXISTS serving_unit TEXT",
                 "ADD COLUMN IF NOT EXISTS label_source TEXT",
                 "DROP CONSTRAINT IF EXISTS user_inventory_kind_check",
                 "RAISE EXCEPTION"):
        assert frag in a, frag


# ── Task 2: aislamiento — los lectores de INGREDIENTES filtran kind = 'food' ─────────────────────────────────────────

_EXCLUIR = {"tests", "scripts", "migrations", "__pycache__", "docs"}


def _lecturas_de_inventario():
    """(fichero, línea, sql, 6 líneas previas) de cada literal SQL con `FROM user_inventory` en código de prod."""
    out = []
    for py in _BACKEND.rglob("*.py"):
        partes = set(py.relative_to(_BACKEND).parts)
        if _EXCLUIR & partes:
            continue
        texto = py.read_text(encoding="utf-8")
        if "user_inventory" not in texto:
            continue
        lineas = texto.splitlines()
        for nodo in ast.walk(ast.parse(texto)):
            valor = None
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
                valor = nodo.value
            elif isinstance(nodo, ast.JoinedStr):
                valor = "".join(v.value for v in nodo.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            if valor and re.search(r"\bFROM\s+(public\.)?user_inventory\b", valor, re.I):
                previas = "\n".join(lineas[max(0, nodo.lineno - 7):nodo.lineno])
                out.append((py.relative_to(_BACKEND).as_posix(), nodo.lineno, valor, previas))
    return out


def test_toda_lectura_de_inventario_filtra_los_suplementos():
    lecturas = _lecturas_de_inventario()
    assert len(lecturas) >= 20, "el escáner dejó de encontrar las lecturas: revisa el guard"
    faltan = [f"{f}:{n}" for f, n, sql, prev in lecturas
              if not re.search(r"kind\s*=\s*'food'", sql) and "[SUPLEMENTOS-OK:" not in prev]
    assert not faltan, "lecturas de user_inventory sin `kind = 'food'` ni marcador: " + ", ".join(faltan)


def test_el_apagado_de_la_nevera_cuenta_los_suplementos():
    import nevera_opcional
    # [P1-PLAN-LOTE-292 · revisión C1] un pote cuenta aunque sus porciones sean desconocidas (0)
    assert "i.kind = 'supplement'" in nevera_opcional._SQL_APAGAR and "kind = 'food'" not in nevera_opcional._SQL_APAGAR
    assert "[SUPLEMENTOS-OK:" in _src("nevera_opcional.py")


def test_la_pantalla_de_la_nevera_recibe_el_tipo_y_la_etiqueta():
    from routers import user_data
    for col in ("kind", "serving_label", "serving_unit", "label_source"):
        assert col in user_data._INVENTORY_SELECT, col


# ── Task 3: la Nevera se enciende sola cuando hace falta ─────────────────────────────────────────────────────────────

@pytest.fixture
def perfil(monkeypatch):
    import nevera_opcional as n
    estado = {"plan_mode": "tracking", "nevera_enabled": False, "nevera_auto_off_at": "2026-09-24T10:00:00Z"}
    escrituras = []
    monkeypatch.setattr(n, "_perfil_nevera", lambda uid: dict(estado))
    monkeypatch.setattr(n, "execute_sql_write", lambda sql, params, **k: escrituras.append((sql, params)))
    monkeypatch.setattr(n, "fijar_nevera", lambda uid, en: escrituras.append(("fijar", en)) or True)
    return n, estado, escrituras


def test_apagada_por_el_sistema_se_enciende_en_automatico(perfil):
    n, estado, escr = perfil
    assert n.encender_por_uso("u") == "encendida"
    sql = escr[0][0]
    assert "nevera_enabled = NULL" in sql and "nevera_auto_off_at = NULL" in sql and "nevera_reloj_desde = now()" in sql
    assert "WHERE id = %s" in sql


def test_apagada_a_mano_pregunta_y_con_forzar_enciende_definitiva(perfil):
    n, estado, escr = perfil
    estado["nevera_auto_off_at"] = None
    assert n.encender_por_uso("u") == "preguntar" and escr == []
    assert n.encender_por_uso("u", forzar=True) == "encendida" and escr == [("fijar", True)]


def test_activa_no_toca_nada(perfil):
    n, estado, escr = perfil
    estado["nevera_enabled"] = None
    assert n.encender_por_uso("u") == "activa" and escr == []


# ── Task 4: el coach guarda aunque la Nevera esté apagada (la regla decide) ─────────────────────────────────────────

def test_el_coach_enciende_o_pregunta_al_anadir(monkeypatch):
    import tools
    import nevera_opcional as n
    monkeypatch.setattr(tools, "nevera_activa", lambda uid: False)
    llamadas = []
    monkeypatch.setattr(n, "encender_por_uso", lambda uid, forzar=False: llamadas.append(forzar) or "preguntar")
    out = tools.modify_pantry_inventory.func("u", items_to_add=["1 pote de whey"])
    assert out == n.MENSAJE_NEVERA_PREGUNTAR and llamadas == [False]
    out = tools.modify_pantry_inventory.func("u", items_to_add=["1 pote de whey"], encender_nevera=True)
    assert llamadas[-1] is True
    llamadas.clear()
    assert tools.modify_pantry_inventory.func("u", items_to_remove=["arroz"]) == n.MENSAJE_NEVERA_APAGADA
    assert llamadas == []    # quitar o consultar no enciende


def test_el_prompt_con_la_nevera_apagada_deja_guardar_si_lo_pide():
    import nevera_opcional as n
    b = n.BLOQUE_PROMPT_NEVERA_APAGADA
    assert "Si te PIDE guardar" in b and "modify_pantry_inventory" in b and "guardar_suplemento" in b


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 290 and m.group(2) >= "2026-09-25"


def test_frontend_grupo_de_suplementos():
    front = _BACKEND.parent / "frontend" / "src"
    if not (front / "__tests__" / "lote290.test.jsx").exists():
        pytest.skip("el frontend de este checkout no trae el lote 290")
    assert "<GrupoSuplementos " in (front / "pages" / "Pantry.jsx").read_text(encoding="utf-8")
