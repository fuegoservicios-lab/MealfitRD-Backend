"""[P1-PLAN-LOTE-720 · 2026-09-28] La ficha de un plato registrado — la mitad del servidor.

Spec: la del 28-sep «ficha-plato-registrado» (en el workspace; no se nombra su ruta: el conftest salta los módulos
que la citan cuando el workspace no está). El dueño quería ver «más detalles de la info de los platos que hemos
agregado del contador de calorías». Lo que estos casos anclan:

  1. `ficha_comida`: el vocabulario del origen (lo desconocido es None, nunca un error), las coordenadas del plan
     (solo ellas) y el desglose por renglón — kcal SOLO si todos resuelven y suman ±25 % de la comida.
  2. `log_consumed_meal` guarda `source`/`plan_ref`, y si la base aún no tiene las columnas registra IGUAL sin ellas:
     una etiqueta jamás puede costar una comida.
  3. Cada camino dice su origen: foto, componedor (o estimado), «Registrar otra vez», «Me lo comí» (con sus
     coordenadas y ahora su `meal_id`) y el coach.
  4. `GET /api/diary/meal/{meal_id}`: filtrado por dueño, 404 igual para «no existe» y «es de otro», limitador propio
     y SIN `verify_api_quota` (lectura de cero LLM), y sale aunque falte la migración.
  5. La lista del día manda `created_at` (el cajón lo leía y nunca llegaba: «Lo anotaste el…» no salió jamás).
  6. Un doble toque ya no devuelve `meal_id: "deduped"`.
  7. La migración: en los dos directorios, idempotente y SIN CHECK.

Tooltip-anchor: P1-PLAN-LOTE-720
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ficha_comida as fc

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent
_UID = "11111111-1111-1111-1111-111111111111"
_MEAL = "22222222-2222-2222-2222-222222222222"
_MIG = "p1_plan_lote_720_consumed_meals_origen_2026_09_28.sql"


# ─────────────────────────── 1 · ficha_comida ───────────────────────────

@pytest.mark.parametrize("entrada,esperado", [
    ("photo", "photo"), ("PHOTO ", "photo"), ("plan_meal", "plan_meal"), ("estimate", "estimate"),
    ("repeat", "repeat"), ("chat", "chat"), ("manual", "manual"),
    ("foto", None), ("", None), (None, None), (42, None), ("'; drop table x;--", None),
])
def test_el_origen_solo_admite_su_vocabulario(entrada, esperado):
    assert fc.origen_de_comida(entrada) == esperado


def test_plan_ref_guarda_solo_las_tres_coordenadas():
    ref = fc.plan_ref_limpio({"plan_id": "abc", "day_index": "3", "meal_index": 1, "otra": "cosa"})
    assert ref == {"plan_id": "abc", "day_index": 3, "meal_index": 1}


@pytest.mark.parametrize("ref", [
    None, "abc", {}, {"plan_id": "", "day_index": 0, "meal_index": 0},
    {"plan_id": "x", "day_index": -1, "meal_index": 0}, {"plan_id": "x", "day_index": 0},
    {"plan_id": "x" * 65, "day_index": 0, "meal_index": 0}, {"plan_id": "x", "day_index": "a", "meal_index": 0},
])
def test_plan_ref_invalido_es_none(ref):
    assert fc.plan_ref_limpio(ref) is None


def test_columna_inexistente_por_sqlstate_o_por_clase():
    class UndefinedColumn(Exception):
        pass

    e = Exception("x")
    e.sqlstate = "42703"
    assert fc.es_columna_inexistente(e)
    assert fc.es_columna_inexistente(UndefinedColumn("col"))
    assert not fc.es_columna_inexistente(ValueError("otra cosa"))


class _DB:
    """Catálogo mínimo: kcal por renglón según una tabla fija; None = no resuelve."""

    def __init__(self, tabla):
        self.tabla = tabla

    def macros_from_ingredient_string(self, s):
        v = self.tabla.get(s)
        return None if v is None else {"name": s, "grams": v[0], "kcal": v[1]}


def test_desglose_con_kcal_cuando_todo_resuelve_y_cuadra():
    db = _DB({"150 g de Pechuga de pollo": (150, 161.1), "185 g de Arroz blanco": (185, 238.9)})
    out = fc.desglose_de_ingredientes(["150 g de Pechuga de pollo", "185 g de Arroz blanco"], 420, db)
    assert out["con_kcal"] is True
    assert out["lineas"] == [
        {"texto": "150 g de Pechuga de pollo", "kcal": 161, "gramos": 150},
        {"texto": "185 g de Arroz blanco", "kcal": 239, "gramos": 185},
    ]


def test_desglose_sin_kcal_si_un_renglon_no_resuelve():
    """Dos bases distintas: si falta un renglón, las cifras que sí salen no suman la comida."""
    db = _DB({"150 g de Pechuga de pollo": (150, 161.1)})
    out = fc.desglose_de_ingredientes(["150 g de Pechuga de pollo", "1 porción de salsa de la casa"], 420, db)
    assert out["con_kcal"] is False
    assert out["lineas"] == [{"texto": "150 g de Pechuga de pollo"}, {"texto": "1 porción de salsa de la casa"}]


def test_desglose_sin_kcal_si_la_suma_no_cuadra_con_la_comida():
    """El caso real del lote 103: «8 rodajas de plátano» resolvía a 8 plátanos (~3.000 kcal) en un almuerzo de 695."""
    db = _DB({"8 rodajas de plátano maduro hervido": (2240, 2990.0)})
    out = fc.desglose_de_ingredientes(["8 rodajas de plátano maduro hervido"], 695, db)
    assert out["con_kcal"] is False
    assert "kcal" not in out["lineas"][0]


def test_desglose_en_el_borde_de_la_tolerancia():
    db = _DB({"a": (100, 125.0)})
    assert fc.desglose_de_ingredientes(["a"], 100, db)["con_kcal"] is True     # +25 %: dentro
    db2 = _DB({"a": (100, 126.0)})
    assert fc.desglose_de_ingredientes(["a"], 100, db2)["con_kcal"] is False   # +26 %: fuera


@pytest.mark.parametrize("ingredientes", [None, [], "150 g de pollo", [None, "  ", 3]])
def test_desglose_sin_ingredientes_es_vacio(ingredientes):
    assert fc.desglose_de_ingredientes(ingredientes, 400, _DB({})) == {"lineas": [], "con_kcal": False}


def test_desglose_sin_catalogo_da_los_textos():
    out = fc.desglose_de_ingredientes(["2 huevos"], 150, None)
    assert out == {"lineas": [{"texto": "2 huevos"}], "con_kcal": False}


def test_la_ficha_tiene_la_forma_de_la_respuesta():
    from datetime import datetime, timezone
    fila = {
        "id": _MEAL, "meal_name": "Mangú", "meal_type": "desayuno", "calories": 500, "protein": 20,
        "carbs": 60, "healthy_fats": 18, "consumed_at": datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
        "created_at": datetime(2026, 9, 28, 12, 1, tzinfo=timezone.utc), "source": "plan_meal",
        "plan_ref": {"plan_id": "p1", "day_index": 2, "meal_index": 0}, "ingredients": ["2 huevos"],
    }
    f = fc.ficha_de_comida(fila, None)
    assert f["id"] == _MEAL and f["source"] == "plan_meal"
    assert f["plan_ref"] == {"plan_id": "p1", "day_index": 2, "meal_index": 0}
    assert f["consumed_at"].startswith("2026-09-28T12:00")
    assert f["created_at"].startswith("2026-09-28T12:01")
    assert f["ingredientes"]["lineas"] == [{"texto": "2 huevos"}]
    # una fila vieja (sin columnas del lote) no rompe
    viejo = fc.ficha_de_comida({"id": _MEAL, "meal_name": "X", "calories": 0}, None)
    assert viejo["source"] is None and viejo["plan_ref"] is None and viejo["ingredientes"]["lineas"] == []


# ─────────────────────────── 2 · log_consumed_meal ───────────────────────────

def _log(monkeypatch, fake_write, **kw):
    import db_facts
    monkeypatch.setattr(db_facts, "connection_pool", MagicMock())
    monkeypatch.setattr(db_facts, "execute_sql_query", lambda *a, **k: None)   # dedup: sin duplicado
    monkeypatch.setattr(db_facts, "execute_sql_write", fake_write)
    return db_facts.log_consumed_meal(_UID, "Mangú", 500, 20, 60, 18, ["2 huevos"], meal_type="desayuno", **kw)


def test_el_insert_lleva_origen_y_coordenadas(monkeypatch):
    vistos = []

    def fake(sql, params, returning=False):
        vistos.append((sql, params))
        return [{"id": _MEAL}]

    out = _log(monkeypatch, fake, source="plan_meal", plan_ref={"plan_id": "p1", "day_index": 1, "meal_index": 2})
    assert out == _MEAL
    sql, params = vistos[0]
    assert "source, plan_ref" in sql
    assert params[-2] == "plan_meal"
    assert params[-1].obj == {"plan_id": "p1", "day_index": 1, "meal_index": 2}


def test_sin_origen_el_insert_es_el_de_siempre(monkeypatch):
    vistos = []

    def fake(sql, params, returning=False):
        vistos.append(sql)
        return [{"id": _MEAL}]

    _log(monkeypatch, fake, source="inventado")
    assert "source" not in vistos[0]


def test_sin_la_migracion_registra_igual_sin_origen(monkeypatch):
    """La etiqueta se pierde; la comida NO."""
    vistos = []

    def fake(sql, params, returning=False):
        vistos.append(sql)
        if "source" in sql:
            e = Exception('column "source" of relation "consumed_meals" does not exist')
            e.sqlstate = "42703"
            raise e
        return [{"id": _MEAL}]

    assert _log(monkeypatch, fake, source="photo") == _MEAL
    assert len(vistos) == 2 and "source" not in vistos[1]


def test_otro_error_del_insert_no_se_disfraza(monkeypatch):
    """Solo «la columna no existe» reintenta; un error de verdad sigue siendo un fallo (None)."""
    calls = []

    def fake(sql, params, returning=False):
        calls.append(sql)
        raise RuntimeError("se cayó la base")

    assert _log(monkeypatch, fake, source="photo") is None
    assert len(calls) == 1


# ─────────────────────────── 3 · cada camino dice su origen ───────────────────────────

def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_persist_pasa_el_origen_a_la_fila():
    diary = _src("routers/diary.py")
    assert "source=(origen or source)," in diary
    assert 'origen=("estimate" if payload.origin == "estimate" else "manual"),' in diary
    assert 'source="manual", deduct=False, origen="repeat",' in diary
    assert 'source="photo",' in diary   # el escáner (`/consumed`) sigue diciendo foto


def test_me_lo_comi_guarda_sus_coordenadas_y_devuelve_el_id():
    import routers.diary as diary
    from routers.diary import ConsumedFromPlanRequest

    plan = {"days": [{"meals": [{"name": "Mangú", "meal": "Desayuno", "cals": 500, "protein": 20,
                                  "carbs": 60, "fats": 18, "ingredients": ["2 huevos"]}]}]}
    payload = ConsumedFromPlanRequest(plan_id="33333333-3333-3333-3333-333333333333", day_index=0, meal_index=0)
    import db_inventory
    with patch.object(diary, "execute_sql_query", return_value={"plan_data": plan}), \
         patch.object(diary, "log_consumed_meal", return_value=_MEAL) as log_mock, \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory", return_value={}), \
         patch.object(diary, "nevera_activa", return_value=True, create=True), \
         patch.object(diary, "trigger_incremental_learning"):
        out = diary.api_log_consumed_meal_from_plan(payload, verified_user_id=_UID)
    assert out["meal_id"] == _MEAL
    kw = log_mock.call_args.kwargs
    assert kw["source"] == "plan_meal"
    assert kw["plan_ref"] == {"plan_id": "33333333-3333-3333-3333-333333333333", "day_index": 0, "meal_index": 0}


def test_el_coach_registra_con_origen_chat():
    tools = _src("tools.py")
    i = tools.index("result = db_log_consumed_meal(")
    assert 'source="chat",' in tools[i:i + 600]


def test_el_componedor_acepta_el_origen_estimado():
    from routers.diary import ManualMealRequest
    m = ManualMealRequest(lines=[{"ref": "custom", "name": "Mangú", "macros": {"kcal": 300}}], origin="estimate")
    assert m.origin == "estimate"
    assert ManualMealRequest(lines=[{"ref": "food:1"}]).origin is None


# ─────────────────────────── 4 · GET /api/diary/meal/{meal_id} ───────────────────────────

def _detalle(query_impl):
    import routers.diary as diary
    with patch.object(diary, "execute_sql_query", side_effect=query_impl):
        return diary.api_get_consumed_meal_detail(_MEAL, verified_user_id=_UID)


def test_la_ficha_filtra_por_dueno():
    visto = {}

    def q(sql, params, fetch_one=False, fetch_all=False):
        visto["sql"], visto["params"] = sql, params
        return {"id": _MEAL, "meal_name": "Mangú", "calories": 500, "source": "photo", "ingredients": []}

    out = _detalle(q)
    assert out["success"] is True and out["meal"]["source"] == "photo"
    assert "WHERE id = %s AND user_id = %s" in visto["sql"]
    assert visto["params"] == (_MEAL, _UID)


def test_la_ficha_de_otro_es_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        _detalle(lambda *a, **k: None)
    assert ei.value.status_code == 404


def test_la_ficha_sale_sin_la_migracion():
    llamadas = []

    def q(sql, params, fetch_one=False, fetch_all=False):
        llamadas.append(sql)
        if "source" in sql:
            e = Exception('column "source" does not exist')
            e.sqlstate = "42703"
            raise e
        return {"id": _MEAL, "meal_name": "Mangú", "calories": 500}

    out = _detalle(q)
    assert out["meal"]["source"] is None
    assert len(llamadas) == 2


def test_la_ficha_rechaza_un_id_que_no_es_uuid():
    from fastapi import HTTPException
    import routers.diary as diary
    with pytest.raises(HTTPException) as ei:
        diary.api_get_consumed_meal_detail("no-es-uuid", verified_user_id=_UID)
    assert ei.value.status_code == 400


def test_la_ficha_usa_su_limitador_y_no_la_cuota():
    diary = _src("routers/diary.py")
    i = diary.index('@router.get("/meal/{meal_id}")')
    firma = diary[i:diary.index('"""', i)]
    assert "Depends(_MEAL_DETAIL_LIMITER)" in firma
    assert "verify_api_quota" not in firma
    assert "_MEAL_DETAIL_LIMITER = RateLimiter(max_calls=30, period_seconds=60)" in diary


# ─────────────────────────── 5 · 6 · la lista del día y el doble toque ───────────────────────────

def test_la_lista_del_dia_manda_created_at():
    dbf = _src("db_facts.py")
    i = dbf.index("def get_consumed_meals_today(")
    cuerpo = dbf[i:dbf.index("\ndef ", i + 10)]
    cols = re.search(r"_COLUMNS = \((.*?)\)", cuerpo, re.S).group(1)
    assert "created_at" in cols and "consumed_at" in cols and "ingredients" in cols


def test_un_doble_toque_no_devuelve_deduped_como_id():
    import routers.diary as diary
    bg = MagicMock()
    with patch.object(diary, "log_consumed_meal", return_value="deduped"), \
         patch.object(diary, "nevera_activa", return_value=False):
        out = diary._persist_consumed_meal(
            user_id=_UID, meal_name="X", meal_type="extra", calories=1, protein=0, carbs=0, healthy_fats=0,
            ingredients=None, days_ago=0, background_tasks=bg, source="photo",
        )
    assert out["already_logged"] is True
    assert out["meal_id"] is None


# ─────────────────────────── 7 · la migración ───────────────────────────

def test_la_migracion_en_los_dos_directorios_idempotente_y_sin_check():
    back = (_BACKEND / "migrations" / _MIG)
    assert back.exists()
    raiz = _ROOT / "migrations" / _MIG
    if raiz.exists():   # el árbol del backend puede vivir suelto (CI del repo backend)
        assert raiz.read_text(encoding="utf-8") == back.read_text(encoding="utf-8")
    sql = back.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS source TEXT" in sql
    assert "ADD COLUMN IF NOT EXISTS plan_ref JSONB" in sql
    assert "AND cm.source IS NULL" in sql            # el relleno no pisa lo ya escrito
    assert "RAISE EXCEPTION" in sql                   # sanity del repo
    assert not re.search(r"ADD\s+CONSTRAINT|CHECK\s*\(", sql, re.I), "una etiqueta no puede tumbar un registro"
