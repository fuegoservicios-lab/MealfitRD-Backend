"""[P1-CHAT-TOOLS-AUDIT · 2026-09-14] Auditoría de las tools del coach (tools.py).

Diez hallazgos verificados, cada uno anclado aquí por CONDUCTA (la tool real con la base como
doble; ningún test toca Neon ni llama al LLM):

  1. `mark_shopping_list_purchased` no marcaba el plan (`is_restocked`/`restocked_items` solo los
     escribía `/restock`) ⇒ pausa `awaiting_first_purchase` a quien ya compró, sin dedupe por
     ciclo, y strings en vez del payload estructurado (se perdía `package_grams`).
  2. `items_to_deplete` emparejaba por subcadena y luego DELETE: «se acabó la sal» borraba «Salami».
  3. `items_to_remove` pasaba por el descuento de CONSUMO (una porción, ledger) y contaba lo ausente.
  4. `update_form_field`: «ganar peso» → lose_fat, dieta con tabla propia, sin enum ni rangos, sin
     try, «¡Éxito!» aunque no se escribiera nada.
  5. `correct_consumed_meal` se saltaba el guard de comida principal y reescribía la hora.
  6. `log_consumed_meal` sin rangos y con `breakfast`/`almoço`/`dîner` cayendo a snack en silencio.
  7. `suggest_foods_for_nutrient` presentaba un perfil ilegible como «sin alergias».
  8. `items_to_add` descartaba sin cantidad en silencio y contaba los fallos como éxitos.
  9. Ocho `str(e)` devueltos al modelo.
 10. El doc citaba `hydration_log`; la tabla es `water_intake_log`.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_UID = "11111111-1111-1111-1111-111111111111"
_MARKER = "P1-CHAT-TOOLS-AUDIT"


def _fn(t):
    return getattr(t, "func", t)  # langchain @tool → función cruda


@pytest.fixture(autouse=True)
def _sin_alias_de_catalogo(monkeypatch):
    # El peldaño de alias de `pantry_names_match` lee el catálogo de la base: fuera.
    monkeypatch.setenv("MEALFIT_PANTRY_ALIAS_MATCH", "false")


# ══════════════════════════════ 1. mark_shopping_list_purchased ══════════════════════════════

@pytest.fixture
def compra(monkeypatch):
    import db
    import db_inventory
    import shopping_calculator
    import tools

    st = {
        "plan": {"days": [], "restocked_items": {}},
        "inv": [],
        "restocks": [],
        "marcas": [],
        "delta": [{"name": "Arroz", "market_qty_numeric": 2, "market_unit": "paquete",
                   "package_grams": 907, "display_string": "2 paquetes de Arroz (907 g)"}],
    }
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan_with_id",
                        lambda uid: {"id": "plan-1", "plan_data": st["plan"]})
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda uid: st["plan"])
    monkeypatch.setattr(shopping_calculator, "get_shopping_list_delta",
                        lambda uid, plan, structured=True, **kw: [dict(x) for x in st["delta"]])

    def _restock(uid, items):
        st["restocks"].append(list(items))
        return (True, [i["name"] if isinstance(i, dict) else str(i) for i in items])

    monkeypatch.setattr(db_inventory, "restock_inventory", _restock)

    def _q(sql, params=None, fetch_one=False, fetch_all=False, **kw):
        if "FROM user_inventory" in sql:
            return [dict(r) for r in st["inv"]]
        return [] if fetch_all else None

    monkeypatch.setattr(db, "execute_sql_query", _q)

    def _atomic(plan_id, mutator, lock_timeout_ms=None, *, user_id=None):
        st["marcas"].append((plan_id, user_id))
        fresco = dict(st["plan"])
        r = mutator(fresco)
        st["plan"] = r if isinstance(r, dict) else fresco
        return st["plan"]

    monkeypatch.setattr(db, "update_plan_data_atomic", _atomic)
    monkeypatch.setattr(db, "bulk_delete_depleted_items", lambda *a, **k: 0)
    try:
        import restock_cycle
        monkeypatch.setattr(restock_cycle, "after_purchase_side_effects",
                            lambda uid, pid, names: {"plan_unfrozen": False})
    except ImportError:  # código previo al fix: el test debe fallar por conducta, no por import
        pass
    return st


def _comprar(**kw):
    import tools
    return _fn(tools.mark_shopping_list_purchased)(_UID, **kw)


def test_la_compra_por_chat_marca_el_plan_como_comprado(compra):
    out = _comprar()
    assert out.startswith("¡Felicidades!"), out
    assert compra["marcas"] == [("plan-1", _UID)], (
        f"el plan no se marcó (o sin user_id ⇒ sin I2): la pausa awaiting_first_purchase pausaría a "
        f"quien ya compró [{_MARKER}]")
    assert compra["plan"].get("is_restocked") is True
    assert "arroz" in (compra["plan"].get("restocked_items") or {})


def test_la_compra_por_chat_manda_el_payload_estructurado(compra):
    _comprar()
    item = compra["restocks"][0][0]
    assert isinstance(item, dict), f"sigue mandando strings (ruta legacy, pierde package_grams): {item!r}"
    assert item["name"] == "Arroz" and item["quantity"] == 2 and item["unit"] == "paquete"
    assert item["package_grams"] == 907


def test_decirlo_dos_veces_no_suma_la_compra_dos_veces(compra):
    _comprar()
    compra["inv"] = [{"ingredient_name": "Arroz", "quantity": 1814.0}]
    out2 = _comprar()
    assert len(compra["restocks"]) == 1, "la re-emisión volvió a SUMAR la compra entera"
    assert "No se añadió nada" in out2 and not out2.startswith("¡Felicidades!"), out2


def test_si_la_nevera_se_vacio_la_recompra_es_legitima(compra):
    _comprar()
    compra["inv"] = []  # se lo comió todo / lo borró
    _comprar()
    assert len(compra["restocks"]) == 2, "el dedupe bloqueó una recompra de algo que ya no está"


def test_el_dedupe_reconoce_el_plural_de_la_fila(compra):
    compra["plan"]["restocked_items"] = {"huevos": datetime.now(timezone.utc).isoformat()}
    compra["delta"] = [{"name": "Huevos", "market_qty_numeric": 12, "market_unit": "unidad",
                        "display_string": "12 Huevos"}]
    compra["inv"] = [{"ingredient_name": "Huevo", "quantity": 6.0}]
    _comprar()
    assert compra["restocks"] == [], "«Huevos» contra la fila «Huevo» es el mismo alimento (pantry_names_match)"


def test_invitado_no_registra_compra(compra):
    import tools
    out = _fn(tools.mark_shopping_list_purchased)("guest")
    assert compra["restocks"] == [] and not out.startswith("¡Felicidades!"), out


# ══════════════════════════════ 2, 3, 8. modify_pantry_inventory ══════════════════════════════

@pytest.fixture
def nevera(monkeypatch):
    import db
    import db_inventory

    st = {"rows": [], "deletes": [], "adds": [], "consumos": [], "agotados": [], "add_ok": True}
    monkeypatch.setattr(db_inventory, "get_raw_user_inventory", lambda uid: [dict(r) for r in st["rows"]])

    def _w(sql, params=None, returning=False, **kw):
        if sql.strip().upper().startswith("DELETE"):
            rid = params[0]
            st["deletes"].append(rid)
            antes = len(st["rows"])
            st["rows"] = [r for r in st["rows"] if r["id"] != rid]
            return [{"id": rid}] if len(st["rows"]) < antes else []
        return []

    monkeypatch.setattr(db, "execute_sql_write", _w)

    def _add(uid, name, qty, unit, mutation_type="manual", source="manual", brand=None):
        st["adds"].append((name, qty, unit, mutation_type))
        if not st["add_ok"]:
            return False
        for r in st["rows"]:
            if r["ingredient_name"] == name and r["unit"] == unit:
                r["quantity"] = max(0.0, r["quantity"] + qty)
                return True
        if qty > 0:
            st["rows"].append({"id": f"n{len(st['rows'])}", "ingredient_name": name, "quantity": qty, "unit": unit})
        return True  # como el real: unidad incompatible + delta negativo ⇒ True sin mover nada

    monkeypatch.setattr(db_inventory, "add_or_update_inventory_item", _add)
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory",
                        lambda *a, **k: st["consumos"].append(a) or {})
    monkeypatch.setattr(db_inventory, "add_depleted_item", lambda uid, **k: st["agotados"].append(k) or True)
    return st


def _nevera(**kw):
    import tools
    return _fn(tools.modify_pantry_inventory)(_UID, **kw)


def test_se_acabo_la_sal_no_borra_el_salami(nevera):
    nevera["rows"] = [{"id": "s1", "ingredient_name": "Salami", "quantity": 200.0, "unit": "g"}]
    out = _nevera(items_to_deplete=["sal"])
    assert nevera["deletes"] == [], "subcadena otra vez: «sal» ⊂ «Salami» y se borró la fila"
    assert nevera["agotados"] == []
    assert "NO encontré" in out, out


def test_agotar_resuelve_el_plural_por_el_ssot(nevera):
    nevera["rows"] = [{"id": "h1", "ingredient_name": "Huevo", "quantity": 6.0, "unit": "unidad"}]
    out = _nevera(items_to_deplete=["huevos"])
    assert nevera["deletes"] == ["h1"]
    assert nevera["agotados"] and nevera["agotados"][0]["ingredient_name"] == "Huevo"
    assert "PANTRY_DEPLETED_JSON" in out


def test_botar_borra_la_fila_entera_y_no_es_consumo(nevera):
    nevera["rows"] = [{"id": "a1", "ingredient_name": "Arroz", "quantity": 2.0, "unit": "lb"}]
    out = _nevera(items_to_remove=["arroz"])
    assert nevera["consumos"] == [], "botar comida NO es consumirla: no va al ledger de consumo"
    assert nevera["deletes"] == ["a1"] and nevera["rows"] == []
    assert "se eliminaron 1" in out, out


def test_botar_con_cantidad_descuenta_solo_esa_cantidad_sin_ledger(nevera):
    nevera["rows"] = [{"id": "a1", "ingredient_name": "Arroz", "quantity": 1000.0, "unit": "g"}]
    out = _nevera(items_to_remove=["200 g de Arroz"])
    assert nevera["consumos"] == []
    assert ("Arroz", -200.0, "g", "discard") in nevera["adds"], nevera["adds"]
    assert nevera["rows"][0]["quantity"] == 800.0
    assert "se eliminaron 1" in out, out


def test_botar_encuentra_la_fila_aunque_el_catalogo_canonice_el_nombre(nevera, monkeypatch):
    """Con el catálogo real cargado `_parse_quantity("arroz")` devuelve «Arroz blanco»: la tool
    buscaba ESE nombre y no encontraba la fila «Arroz» (rojo del gate, verde sin catálogo). Se
    simula la canonización para que el caso falle en CUALQUIER entorno."""
    import shopping_calculator
    real = shopping_calculator._parse_quantity

    def _canoniza(item):
        q, u, n = real(item)
        return q, u, ("Arroz blanco" if str(n).strip().lower() == "arroz" else n)

    monkeypatch.setattr(shopping_calculator, "_parse_quantity", _canoniza)
    nevera["rows"] = [{"id": "a1", "ingredient_name": "Arroz", "quantity": 2.0, "unit": "lb"}]
    out = _nevera(items_to_remove=["arroz"])
    assert nevera["deletes"] == ["a1"], out
    nevera["rows"] = [{"id": "a2", "ingredient_name": "Arroz", "quantity": 1000.0, "unit": "g"}]
    _nevera(items_to_remove=["200 g de arroz"])
    assert ("Arroz", -200.0, "g", "discard") in nevera["adds"], nevera["adds"]


def test_botar_algo_que_no_esta_no_cuenta_como_eliminado(nevera):
    out = _nevera(items_to_remove=["caviar"])
    assert "se eliminaron" not in out, f"contó como eliminado algo ausente: {out!r}"
    assert "NO encontré" in out


def test_botar_con_unidad_incompatible_se_reporta(nevera):
    nevera["rows"] = [{"id": "p1", "ingredient_name": "Pollo", "quantity": 2.0, "unit": "unidad"}]
    out = _nevera(items_to_remove=["300 g de Pollo"])
    assert "se eliminaron" not in out, out
    assert "no es compatible" in out, out


def test_agregar_sin_cantidad_se_le_dice_al_modelo(nevera):
    out = _nevera(items_to_add=["leche"])
    assert nevera["adds"] == []
    assert "NO agregué" in out and "leche" in out, out


def test_agregar_fallido_no_cuenta_como_agregado(nevera):
    nevera["add_ok"] = False
    out = _nevera(items_to_add=["2 lb de Pollo"])
    assert "se agregaron" not in out, f"contó como éxito una escritura que devolvió False: {out!r}"
    assert "NO se pudo aplicar" in out


# ══════════════════════════════ 4. update_form_field ══════════════════════════════

@pytest.fixture
def perfil(monkeypatch):
    import tools

    st = {"hp": {"weightUnit": "lb", "allergies": ["Mariscos"]}, "llamadas": 0, "ret_none": False, "boom": False}

    def _atomic(uid, mutator):
        st["llamadas"] += 1
        if st["boom"]:
            raise RuntimeError("SELECT secreto FROM health_profile")
        if st["ret_none"]:
            return None
        hp = dict(st["hp"])
        r = mutator(hp)
        if r is False:
            return st["hp"]
        st["hp"] = hp
        return hp

    monkeypatch.setattr(tools, "update_user_health_profile_atomic", _atomic)
    monkeypatch.setattr(tools, "delete_user_facts_by_metadata", lambda *a, **k: 0)
    return st


def _form(field, value, uid=_UID):
    import tools
    return _fn(tools.update_form_field)(user_id=uid, field=field, new_value=value)


@pytest.mark.parametrize("texto,esperado", [
    ("ganar peso", "gain_muscle"),
    ("quiero ganar masa muscular", "gain_muscle"),
    ("mantener mi peso", "maintenance"),
    ("perder peso", "lose_fat"),
    ("bajar de peso", "lose_fat"),
    ("lose_fat", "lose_fat"),
    ("rendimiento deportivo", "performance"),
])
def test_objetivo_se_mapea_sin_invertir_el_signo(perfil, texto, esperado):
    out = _form("mainGoal", texto)
    assert out.startswith("¡Éxito!"), out
    assert perfil["hp"]["mainGoal"] == esperado, f"«{texto}» → {perfil['hp']['mainGoal']!r}"


def test_objetivo_ambiguo_se_rechaza(perfil):
    out = _form("mainGoal", "peso")
    assert not out.startswith("¡Éxito!") and perfil["llamadas"] == 0, out


@pytest.mark.parametrize("texto,esperado", [
    ("vegana", "vegan"), ("soy vegetariana", "vegetarian"), ("balanceada", "balanced"), ("vegan", "vegan"),
])
def test_dieta_por_el_ssot(perfil, texto, esperado):
    assert _form("dietType", texto).startswith("¡Éxito!")
    assert perfil["hp"]["dietType"] == esperado


@pytest.mark.parametrize("field,valor", [
    ("dietType", "keto"), ("budget", "carísimo"), ("age", "300"), ("weight", "5000"),
    ("height", "5'9"), ("gender", "robot"), ("cookingTime", "siempre"),
])
def test_valores_fuera_del_formulario_se_rechazan_sin_escribir(perfil, field, valor):
    out = _form(field, valor)
    assert not out.startswith("¡Éxito!"), f"{field}={valor!r} se aceptó: {out!r}"
    assert perfil["llamadas"] == 0, "se escribió en la base un valor que el formulario no admite"


@pytest.mark.parametrize("field,valor,esperado", [
    ("budget", "alto", "high"), ("budget", "Económico", "low"), ("activityLevel", "Sedentario", "sedentary"),
    ("cookingTime", "30 minutos", "30min"), ("gender", "mujer", "female"), ("age", "30 años", "30"),
    ("height", "1.80", "180"),
])
def test_etiquetas_del_formulario_se_canonicalizan(perfil, field, valor, esperado):
    assert _form(field, valor).startswith("¡Éxito!")
    assert perfil["hp"][field] == esperado


def test_peso_en_kg_sobre_un_perfil_en_libras_se_convierte(perfil):
    out = _form("weight", "80 kg")
    assert out.startswith("¡Éxito!"), out
    assert perfil["hp"]["weight"] == "176.4" and perfil["hp"]["weightUnit"] == "lb"


def test_sin_perfil_no_hay_exito(perfil):
    perfil["ret_none"] = True
    out = _form("budget", "low")
    assert not out.startswith("¡Éxito!"), f"contrato con agent.py roto: «¡Éxito!» sin escritura: {out!r}"


def test_una_excepcion_de_la_base_no_tumba_el_turno_ni_filtra_detalle(perfil):
    perfil["boom"] = True
    out = _form("budget", "low")
    assert not out.startswith("¡Éxito!") and "SELECT" not in out, out


def test_la_fusion_de_alergias_sigue_intacta(perfil):
    out = _form("allergies", "Lacteos")
    assert out.startswith("¡Éxito!")
    assert perfil["hp"]["allergies"] == ["Mariscos", "Lacteos"]
    assert "Mariscos" in out, "el éxito debe reportar el valor FINAL (fusionado), no solo lo pedido"


def test_invitado_valida_igual_pero_no_escribe(perfil):
    assert not _form("dietType", "keto", uid="guest").startswith("¡Éxito!")
    assert _form("dietType", "vegana", uid="guest").startswith("¡Éxito!")
    assert perfil["llamadas"] == 0


# ══════════════════════════════ 5, 6. diario ══════════════════════════════

@pytest.fixture
def diario(monkeypatch):
    import db
    import db_inventory
    import tools

    st = {"logs": [], "updates": [], "fila": None, "dup": None, "eventos": 0, "reverts": [], "descuentos": []}
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: st["logs"].append((a, k)) or "row-new")
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, kcal, d: mt)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 240)

    def _q(sql, params=None, fetch_one=False, fetch_all=False, **kw):
        if "inventory_consumption_events" in sql:
            return {"c": st["eventos"]}
        if "SELECT meal_type, consumed_at" in sql:
            return st["fila"]
        if "FROM consumed_meals" in sql:
            return st["dup"]
        return None

    monkeypatch.setattr(db, "execute_sql_query", _q)
    monkeypatch.setattr(tools, "db_update_consumed_meal",
                        lambda uid, mid, **k: st["updates"].append(k) or mid)
    monkeypatch.setattr(db_inventory, "revert_consumption_events",
                        lambda uid, mid: st["reverts"].append(mid) or {"reverted": ["1 lb de Pollo"]})
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory",
                        lambda uid, ings, **k: st["descuentos"].append((ings, k)) or {"not_in_pantry": []})
    return st


def _log(**kw):
    import tools
    base = {"meal_name": "Mangú", "calories": 500, "protein": 20}
    base.update(kw)
    return _fn(tools.log_consumed_meal)(_UID, **base)


def _corregir(**kw):
    import tools
    return _fn(tools.correct_consumed_meal)(_UID, "m1", **kw)


@pytest.mark.parametrize("kw", [{"calories": -300}, {"calories": 50000}, {"protein": 5000}, {"healthy_fats": -1}])
def test_log_rechaza_macros_imposibles(diario, kw):
    out = _log(**kw)
    assert diario["logs"] == [], f"se registró {kw}: {out!r}"
    assert not out.startswith("¡Éxito!")


@pytest.mark.parametrize("tipo,esperado", [
    ("breakfast", "desayuno"), ("lunch", "almuerzo"), ("dinner", "cena"),
    ("almoço", "almuerzo"), ("jantar", "cena"), ("petit-déjeuner", "desayuno"),
    ("dîner", "cena"), ("pranzo", "almuerzo"), ("colazione", "desayuno"), ("merenda", "merienda"),
])
def test_log_entiende_el_tipo_de_comida_en_los_5_idiomas(diario, tipo, esperado):
    assert _log(meal_type=tipo).startswith("¡Éxito!")
    assert diario["logs"][0][1]["meal_type"] == esperado


def test_log_tipo_irreconocible_ya_no_es_silencioso(diario):
    out = _log(meal_type="brunch")
    assert diario["logs"][0][1]["meal_type"] == "snack"
    assert "no reconocí el tipo de comida" in out


def _hace(dias: int, hh: int, mm: int) -> datetime:
    """Instante UTC de las hh:mm LOCALES (UTC-4) de hace `dias` días."""
    off = timedelta(minutes=240)
    dia = (datetime.now(timezone.utc) - off).date() - timedelta(days=dias)
    return datetime.combine(dia, datetime.min.time().replace(hour=hh, minute=mm)).replace(tzinfo=timezone.utc) + off


def test_corregir_el_tipo_respeta_el_guard_de_comida_principal(diario):
    diario["fila"] = {"meal_type": "snack", "consumed_at": _hace(0, 8, 15), "inventory_synced_at": None}
    diario["dup"] = {"meal_name": "Mangú con huevo", "calories": 600}
    out = _corregir(meal_type="desayuno")
    assert diario["updates"] == [], "la corrección dejó dos desayunos el mismo día"
    assert "NO CORREGIDO" in out and "force=true" in out
    _corregir(meal_type="desayuno", force=True)
    assert len(diario["updates"]) == 1


def test_mover_de_dia_conserva_la_hora(diario):
    diario["fila"] = {"meal_type": "desayuno", "consumed_at": _hace(3, 8, 15), "inventory_synced_at": None}
    _corregir(days_ago=1)
    nuevo = datetime.fromisoformat(diario["updates"][0]["consumed_at_override"])
    assert abs((nuevo - _hace(1, 8, 15)).total_seconds()) < 1, f"hora reescrita: {nuevo.isoformat()}"


def test_corregir_con_tipo_irreconocible_no_escribe(diario):
    out = _corregir(meal_type="brunch")
    assert out.startswith("ERROR") and diario["updates"] == []


def test_corregir_con_macros_imposibles_no_escribe(diario):
    out = _corregir(calories=-50)
    assert out.startswith("ERROR") and diario["updates"] == []


def test_corregir_ingredientes_ajusta_la_nevera_cuando_hay_rastro(diario):
    diario["fila"] = {"meal_type": "almuerzo", "consumed_at": _hace(0, 13, 0),
                      "inventory_synced_at": datetime.now(timezone.utc)}
    diario["eventos"] = 3
    out = _corregir(ingredients=["200 g de Pollo"])
    assert diario["reverts"] == ["m1"]
    assert diario["descuentos"] and diario["descuentos"][0][1]["consumed_meal_id"] == "m1"
    assert "Nevera ajustada" in out


def test_corregir_ingredientes_sin_rastro_no_toca_la_nevera_y_lo_dice(diario):
    diario["fila"] = {"meal_type": "almuerzo", "consumed_at": _hace(0, 13, 0),
                      "inventory_synced_at": datetime.now(timezone.utc)}
    diario["eventos"] = 0
    out = _corregir(ingredients=["200 g de Pollo"])
    assert diario["reverts"] == [] and diario["descuentos"] == []
    assert "NO se ajustó" in out


# ══════════════════════════════ 7. suggest_foods_for_nutrient ══════════════════════════════

def test_perfil_ilegible_no_se_presenta_como_sin_alergias(monkeypatch):
    import shopping_calculator
    import tools
    monkeypatch.setattr(tools, "get_user_profile", lambda uid: None)
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [
        {"name": "Lentejas", "fiber_g_per_100g": 8.0},
        {"name": "Avena", "fiber_g_per_100g": 10.0},
    ])
    out = _fn(tools.suggest_foods_for_nutrient)(_UID, "fibra")
    assert "no declara alergias" not in out, f"un perfil ilegible sonó a «sin alergias»: {out!r}"
    assert "NO pude leer el perfil" in out


# ══════════════════════════════ 9. errores sin detalle interno ══════════════════════════════

_TOOLS_SIN_FUGA = (
    "check_shopping_list", "check_current_pantry", "modify_pantry_inventory", "check_hydration_today",
    "log_water_glass", "mark_shopping_list_purchased", "suggest_foods_for_nutrient", "check_clinical_profile",
)


def _cuerpo(src: str, nombre: str) -> str:
    i = src.index(f"def {nombre}(")
    j = src.find("\n@tool", i)
    return src[i:j if j > 0 else len(src)]


@pytest.mark.parametrize("nombre", _TOOLS_SIN_FUGA)
def test_ningun_error_le_devuelve_str_e_al_modelo(nombre):
    cuerpo = _cuerpo((_BACKEND / "tools.py").read_text(encoding="utf-8"), nombre)
    assert not re.search(r"return[^\n]*\{str\(e\)\}", cuerpo), f"{nombre} devuelve str(e) al LLM"


def test_la_despensa_no_filtra_la_excepcion(monkeypatch):
    import db_inventory
    import tools

    def _boom(uid):
        raise RuntimeError("SELECT ingredient_name FROM user_inventory WHERE user_id = 'x'")

    monkeypatch.setattr(db_inventory, "get_user_inventory", _boom)
    out = _fn(tools.check_current_pantry)(_UID)
    assert "SELECT" not in out and "user_inventory" not in out, out


# ══════════════════════════════ 10. doc + paridad ══════════════════════════════

def test_el_doc_nombra_la_tabla_real_de_hidratacion():
    doc = (_BACKEND / "docs" / "agent_tools_user_id_table.md").read_text(encoding="utf-8")
    filas = [ln for ln in doc.splitlines() if re.match(r"^\|\s*[89]\s*\|", ln)]
    assert len(filas) == 2
    for fila in filas:
        assert "water_intake_log" in fila and "hydration_log" not in fila, fila


def test_los_rangos_del_chat_son_los_del_formulario():
    import tools
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    bloque = src[src.index("_BIO_RANGES = {"):]
    bloque = bloque[:bloque.index("\n}")]
    for clave in ("age", "weight_kg", "height_cm"):
        m = re.search(rf'"{clave}":\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', bloque)
        assert m, clave
        assert (float(m.group(1)), float(m.group(2))) == tuple(float(x) for x in tools._CHAT_BIO_RANGES[clave]), (
            f"_CHAT_BIO_RANGES['{clave}'] divergió de routers/plans.py::_BIO_RANGES")
