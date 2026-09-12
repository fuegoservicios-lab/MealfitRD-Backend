"""[P1-PLAN-LOTE-4 · 2026-09-11] Cuarto lote del plan de pendientes (`docs/plan_pendientes_2026_09_11.md`).

  D5  El gate de reservas de la Nevera contaba TODA línea con cantidad > 0 y exigía reservar la mitad: «7 g de
      arroz» o «1 cdta de sal» pesaban igual que «200 g de pollo». `constants.reservation_line_is_material`
      decide con la misma vara en las DOS orillas (lo esperado en el worker y lo reservado en `db_inventory`):
      por debajo de `MEALFIT_PANTRY_RESERVE_MIN_G` (15 g/ml) o condimento ⇒ no cuenta; unidad contable ⇒ cuenta.
      Y `malla` pesa 5 lb: «2 mallas de Papa» son 4,5 kg, no 2 unidades.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_d5_una_pizca_o_un_condimento_no_cuentan_y_un_alimento_real_si():
    from constants import reservation_line_is_material as m
    assert m(200, "g", "Pollo") is True
    assert m(7, "g", "Arroz blanco") is False, "7 g de arroz no sostienen un bloqueo"
    assert m(15, "g", "Arroz blanco") is True, "el umbral es inclusivo"
    assert m(1, "cdta", "Sal") is False and m(2, "cda", "Aceite") is False
    assert m(2, "cda", "Aceite de oliva") is True, "el set ignora términos EXACTOS (como el contador de la Nevera): el aceite de oliva es una fila real"
    assert m(1, "taza", "Arroz") is True, "236 ml de arroz son alimento"
    assert m(2, "unidad", "Huevo") is True and m(1, "malla", "Papa") is True
    assert m(0, "g", "Pollo") is False and m(100, "g", "") is False


def test_d5_el_umbral_es_knob_y_se_acota(monkeypatch):
    from constants import reservation_line_is_material as m, pantry_reserve_min_g
    monkeypatch.setenv("MEALFIT_PANTRY_RESERVE_MIN_G", "50")
    assert pantry_reserve_min_g() == 50.0
    assert m(40, "g", "Pollo") is False and m(60, "g", "Pollo") is True
    monkeypatch.setenv("MEALFIT_PANTRY_RESERVE_MIN_G", "900")
    assert pantry_reserve_min_g() == 100.0, "tope: 100 g; más sería aflojar la seguridad de los alimentos reales"


def test_d5_la_malla_pesa_cinco_libras():
    from constants import _to_base_unit
    q, u = _to_base_unit(2, "mallas")
    assert u == "g" and q == pytest.approx(2 * 2267.96)
    assert _to_base_unit(1, "malla") == (pytest.approx(2267.96), "g")


def test_d5_las_dos_orillas_miden_con_la_misma_vara():
    ct = _src("cron_tasks.py")
    i = ct.find("_expected_ingredients = 0")
    assert "from constants import reservation_line_is_material as _rlm" in ct[i - 400:i + 50]
    assert "if _rn and _rq > 0 and _rlm(_rq, _ru, _rn):" in ct[i:i + 900]
    assert "ignored_terms = PANTRY_IGNORED_TERMS" in ct, "el contador de la Nevera comparte el set"
    dbi = _src("db_inventory.py")
    assert "if name and qty > 0 and _reservation_line_is_material(qty, unit, name):" in dbi


def test_d5_reserve_plan_ingredients_salta_lo_inmaterial(monkeypatch):
    import db_inventory as dbi
    monkeypatch.setattr(dbi, "_db_available", lambda: True)
    monkeypatch.setattr(dbi, "execute_sql_query", lambda *a, **k: [])
    reservadas = []
    monkeypatch.setattr(dbi, "_apply_reservation_delta", lambda uid, name, qty, unit, key, **kw: reservadas.append(name) or True)
    days = [{"meals": [{"name": "Pollo guisado",
                        "ingredients": ["200 g de Pollo", "7 g de Arroz", "1 cdta de Sal", "2 huevos", "150 g de Yuca"]}]}]
    n = dbi.reserve_plan_ingredients("u1", "chunk1", days)
    assert n == 3, reservadas
    assert sorted(reservadas) == sorted(["Pollo", "huevos", "Yuca"]) or len(reservadas) == 3


def test_d5_el_contador_de_la_nevera_sigue_ignorando_condimentos():
    import cron_tasks as ct
    assert ct._count_meaningful_pantry_items(["Sal", "Aceite", "Pollo", "Arroz", "agua"]) == 2


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-4 · 2026-09-11]" in _src("app.py")
    # [P1-PLAN-LOTE-13 · 2026-09-12] «no anterior a este lote», no «igual a hoy»: el pin de la fecha y del prefijo
    # `P1-PLAN-` rompía 12 tests el primer día en que otro P-fix bumpeaba el marker.
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-11", app._LAST_KNOWN_PFIX
