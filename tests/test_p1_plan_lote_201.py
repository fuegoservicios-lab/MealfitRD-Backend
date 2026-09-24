# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-201 · 2026-09-24] Las anotaciones de la Nevera que lee la IA dejan de mentir.

Nevera real (41 alimentos del 4-sep): 33 «URGENTE: Caducado — IA: Prioriza su uso» (canela, sal, pasta, habichuelas
secas… por el `shelf_life_days` de relleno = 14) y «Huevo: se agotará en ~0 días» con un cartón de 20.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import anotacion_nevera as an  # noqa: E402

_HACE_20 = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%dT10:00:00")
_FILAS = [  # (nombre, cantidad, unidad, categoría del catálogo, shelf_life_days del catálogo)
    ("Canela en polvo", 14.2, "g", "Despensa", 14),
    ("Pasta integral", 500, "g", "Despensa", 14),
    ("Habichuelas blancas", 800, "g", "Despensa", 14),
    ("Soya texturizada", 200, "g", "Proteínas", 14),
    ("Nabo", 2, "unidad", "Vegetales", 14),
    ("Huevo", 1, "cartón (20 uds.)", "Proteínas", 35),
    ("Tilapia", 5, "unidad", "Proteínas", 14),
    ("Queso blanco", 0.25, "lb", "Lácteos", 14),
    ("Mango", 4, "unidad", "Frutas", 14),
]


def _lineas(monkeypatch):
    import db_inventory as dbi
    monkeypatch.setattr(dbi, "get_raw_user_inventory", lambda uid: [
        {"ingredient_name": n, "quantity": q, "unit": u, "created_at": _HACE_20} for n, q, u, _c, _s in _FILAS])
    monkeypatch.setattr(dbi, "get_master_ingredients", lambda: [
        {"name": n, "category": c, "shelf_life_days": s} for n, _q, _u, c, s in _FILAS])
    monkeypatch.setattr(dbi, "_compute_dynamic_consumption_rates", lambda *a, **k: {})
    lineas = dbi.get_user_inventory_net("u", household_size=1)
    return {re.sub(r"^[\d.]+\s+(?:\S+\s+(?:\(\d+ uds\.\)\s+)?de\s+)?", "", l.split(" [")[0]): l for l in lineas}


def test_los_secos_y_duraderos_no_salen_caducados(monkeypatch):
    por = _lineas(monkeypatch)
    for n in ("Canela en polvo", "Pasta integral", "Habichuelas blancas", "Soya texturizada", "Nabo", "Huevo"):
        assert "Caducado" not in por[n], por[n]


def test_lo_perecedero_sigue_avisando(monkeypatch):
    por = _lineas(monkeypatch)
    assert "Caducado" in por["Tilapia"] and "Caducado" in por["Queso blanco"], (por["Tilapia"], por["Queso blanco"])


def test_sin_gramos_no_se_predice_el_agotamiento(monkeypatch):
    por = _lineas(monkeypatch)
    assert "PREDICCIÓN" not in por["Huevo"], por["Huevo"]
    assert "PREDICCIÓN" not in por["Nabo"], por["Nabo"]


def test_helpers():
    assert an.plazo("Canela en polvo", "Despensa", 14) == 180
    assert an.plazo("Tilapia", "Proteínas", 14) == 14, "lo congelable no se toca"
    assert an.plazo("Lechuga", "Vegetales", 14) == 14, "lo fresco no se acorta"
    assert an.masa_conocida("lbs", "Vegetales") and an.masa_conocida("g", None)
    assert not an.masa_conocida("unidad", "Vegetales") and not an.masa_conocida("cartón (20 uds.)", "Proteínas")
    assert an.masa_conocida("unidad", "Frutas"), "la fruta ya se consume por unidades"
    assert not an.masa_conocida("unidad", "Frutas", 12.5), "pero la tasa dinámica del plan va en gramos/día"


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_PANTRY_ANNOTATIONS_SSOT", "false")
    assert an.plazo("Canela en polvo", "Despensa", 14) == 14 and an.masa_conocida("unidad", "Vegetales")


def test_los_dos_sitios_usan_los_ganchos():
    src = (_BACKEND / "db_inventory.py").read_text(encoding="utf-8")
    assert src.count('__import__("anotacion_nevera").plazo(name, master_item.get("category"), shelf_life)') == 2
    assert src.count('__import__("anotacion_nevera").masa_conocida(unit, category, _dynamic_rate)') == 2


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 201 and m.group(2) >= "2026-09-24"
