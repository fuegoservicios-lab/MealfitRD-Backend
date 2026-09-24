# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-209 · 2026-09-24] `master_ingredients.shelf_life_days` deja de ser relleno (permiso del dueño).

326 de 349 filas valían 14 días. La migración pone el plazo en fresco del SSOT de durabilidad
(`pantry_durability.classify`) para las clases pantry / cold / fresh y deja freezable / frozen (dependen del congelador)
y las filas curadas a mano. Aplicada en producción el 24-sep con el libro de migraciones.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_NOMBRE = "p1_plan_lote_209_shelf_life_ssot_2026_09_24.sql"


def _sql() -> str:
    return (_BACKEND / "migrations" / _NOMBRE).read_text(encoding="utf-8")


def _pares() -> dict:
    return {n.replace("''", "'"): int(d) for n, d in re.findall(r"^\s+\('((?:[^']|'')+)',\s*(\d+)\)", _sql(), re.M)}


def test_esta_en_los_dos_directorios_y_es_identica():
    raiz = _BACKEND.parent / "migrations" / _NOMBRE
    if raiz.exists():   # el árbol del backend puede vivir sin el workspace-root (VPS, worktrees)
        assert raiz.read_text(encoding="utf-8") == _sql()


def test_idempotente_y_con_sanity():
    sql = _sql()
    assert "AND m.shelf_life_days = 14;" in sql, "sólo reemplaza el relleno: una fila curada después no se pisa"
    assert "RAISE EXCEPTION" in sql and "DO $$" in sql


def test_los_valores_salen_del_ssot_de_durabilidad():
    from pantry_durability import classify
    pares = _pares()
    assert len(pares) == 256, len(pares)
    for nombre, cat in (("Aceite de oliva", "Despensa"), ("Sal", "Despensa"), ("Zanahoria", "Vegetales"),
                        ("Aguacate", "Frutas"), ("Ajo", "Vegetales"), ("Leche", "Lácteos")):
        c = classify(nombre, cat)
        assert pares[nombre] == c["days_fresh"], (nombre, pares.get(nombre), c)


def test_no_toca_lo_que_depende_del_congelador():
    pares = _pares()
    for nombre in ("Pechuga de pollo", "Camarones", "Edamame", "Chorizo mexicano"):
        assert nombre not in pares, nombre


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 209 and m.group(2) >= "2026-09-24"
