"""[P1-INVENTORY-USABLE-PLAN · 2026-09-02] El lector de «plan activo» del inventario ignora el
placeholder vacío de la cola.

Medido en prod: con la cola global (Fase 1) cada generación deja ~7 min un `meal_plans`
más reciente con `generation_status='generating'` y `days=[]`. `get_dynamic_consumption_rates`
lo tomaba por plan activo, no encontraba `calc_household_multiplier` y caía al fallback
hardcodeado con 12 WARNING por tick del cron de refill (+10 por request). Todos los planes
COMPLETOS sí llevan el campo. Fix: el mismo `USABLE_PLAN_SQL_FILTER` que ya usan coach y tools.

Tooltip-anchor: P1-INVENTORY-USABLE-PLAN | USABLE_PLAN_SQL_FILTER en db_inventory
"""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "db_inventory.py").read_text(encoding="utf-8")


def test_active_plan_query_uses_usable_filter():
    i = SRC.index("P1-INVENTORY-USABLE-PLAN")
    blk = SRC[i:i + 900]
    assert "from db_plans import USABLE_PLAN_SQL_FILTER" in blk
    assert re.search(r'"SELECT plan_data FROM meal_plans WHERE user_id = %s AND " \+ _usable', blk)


def test_filter_excludes_only_empty_generating_placeholder():
    from db_plans import USABLE_PLAN_SQL_FILTER as f
    assert "generation_status" in f and "'generating'" in f and "jsonb_array_length" in f
    # un plan `generating` CON días (bloque 1 ya entregado) sigue siendo usable: solo cae el vacío
    assert "= 0" in f


def test_no_raw_newest_plan_reader_left_in_db_inventory():
    raw = re.findall(r'FROM meal_plans WHERE user_id = %s "\s*"ORDER BY created_at DESC LIMIT 1', SRC)
    assert raw == [], "quedó un lector del plan más reciente sin el filtro usable"
