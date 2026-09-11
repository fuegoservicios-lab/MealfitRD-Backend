# -*- coding: utf-8 -*-
"""[P1-DO-DESPENSA-DE-SU-MERCADO · 2026-09-10] Seis ingredientes sin precio, resueltos con su mercado.

Tras pasar el «queso de papa» a gouda quedaban 12 platos dominicanos sin precio por 6 productos, y
ningún sinónimo en `supermarket_products`. El dueño contestó con capturas de su supermercado y con lo
que sabe de su cocina:

- **Azúcar morena** en RD se llama *azúcar crema* (Wala); **pan rallado** se dice *pan molido*
  (Buenhorno); el **Sazón Goya culantro y achiote** y el **hummus** sí se venden. ⇒ precio
  `owner_verified` + el nombre dominicano en `gloss_es` (el canónico no se toca: es la identidad del motor).
- **El sofrito se hace, no se compra** ⇒ se desglosa en ají cubanela, cilantro, cebolla y ajo.
- **Sémola de maíz**: no la conoce, la Maizena no la sustituye (almidón puro) y el maíz partido no
  está en su súper ⇒ los 3 platos (2 chenchén y el chacá) salen de la biblioteca dominicana.
- **Queso de hoja**: se vende en funditas, sin etiqueta ⇒ se DOCUMENTA la procedencia de sus valores
  (familia pasta hilada) en vez de inventar una referencia o poner un `fdc_id` que no case.
"""
import json
import pathlib
import re

import pytest

import dish_registry as dr

BACK = pathlib.Path(__file__).resolve().parents[1]
REG = pathlib.Path(dr.REGISTRY_DIR)
MIG = "p1_do_despensa_de_su_mercado_2026_09_10.sql"


def _json(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reg_do():
    return _json(REG / "dish_registry_do_v1.json")


@pytest.fixture(scope="module")
def sql():
    return (BACK / "migrations" / MIG).read_text(encoding="utf-8")


# ── sofrito y sémola: fuera de la cocina dominicana como PRODUCTO ────────────
@pytest.mark.parametrize("fila", ["Sofrito", "Sémola de maíz"])
def test_ningun_plato_dominicano_los_usa_como_producto(reg_do, fila):
    usan = [t["name"] for t in reg_do["templates"] if any(c["canonical"] == fila for c in t["constituents"])]
    assert not usan, f"platos dominicanos con «{fila}» como ingrediente comprado: {usan}"


@pytest.mark.parametrize("nombre,componentes", [
    ("Chuleta de soya guisada con arroz y habichuelas", {"Ají cubanela", "Cilantro", "Cebolla", "Ajo"}),
    ("Sancocho vegano de víveres con soya texturizada", {"Ají cubanela", "Cebolla", "Ajo", "Cilantro"}),
    ("Berenjena guisada con soya texturizada y batata", {"Ají cubanela", "Ajo", "Cilantro", "Cebolla"}),
])
def test_el_sofrito_esta_hecho_con_sus_ingredientes(reg_do, nombre, componentes):
    t = next((x for x in reg_do["templates"] if x["name"] == nombre), None)
    assert t is not None, f"desapareció «{nombre}»"
    tiene = {c["canonical"] for c in t["constituents"]}
    assert componentes <= tiene, f"«{nombre}» no lleva lo que un sofrito lleva: falta {componentes - tiene}"
    assert t["status"] == "ok"


def test_los_pasos_explican_como_se_hace_sin_venderlo_como_producto():
    rec = _json(REG / "recipe_library_do_v1.json")["por_id"]
    for tid in ("tpl_03a0f3927930", "tpl_3b2ae5748a82", "tpl_dbae3c086fa2"):
        texto = " ".join(rec[tid]["pasos"]).lower()
        assert "sofrito" not in texto, f"{tid}: sigue nombrando el sofrito como si se comprara"
        assert "picad" in texto or "pica " in texto, f"{tid}: no dice cómo se prepara"


def test_las_erratas_de_la_chuleta():
    paso = _json(REG / "recipe_library_do_v1.json")["por_id"]["tpl_03a0f3927930"]["pasos"][0]
    assert "escurría" not in paso and "lavalas" not in paso


@pytest.mark.parametrize("tid", ["tpl_16326aceee3b", "tpl_bbc63aafb357", "tpl_f8a2140ab081"])
def test_los_platos_de_semola_salieron_de_los_tres_sitios(reg_do, tid):
    assert tid not in {t["template_id"] for t in reg_do["templates"]}
    assert tid not in _json(REG / "recipe_library_do_v1.json")["por_id"]


def test_ninguna_receta_quedo_huerfana(reg_do):
    vivos = {t["template_id"] for t in reg_do["templates"]}
    huerfanas = [k for k in _json(REG / "recipe_library_do_v1.json")["por_id"] if k not in vivos]
    assert not huerfanas, f"recetas que apuntan a plantillas que ya no existen: {huerfanas}"


# ── la migración: precios del dueño, nombres dominicanos, procedencia ───────
def test_la_migracion_vive_en_los_dos_directorios(sql):
    assert (BACK.parent / "migrations" / MIG).read_text(encoding="utf-8") == sql


@pytest.mark.parametrize("fila,precio", [("Azúcar morena", "35.50"), ("Pan rallado", "73.00"),
                                         ("Sazón con culantro y achiote", "1122.64"), ("Hummus", "478.40")])
def test_los_precios_vienen_del_dueno(sql, fila, precio):
    bloque = sql.split(f"WHERE name = '{fila}'")[0].rsplit("UPDATE public.master_ingredients SET", 1)[1]
    assert f"price_per_lb = {precio}" in bloque and "'owner_verified'" in bloque


@pytest.mark.parametrize("fila,glosa", [("Azúcar morena", "azúcar crema"), ("Pan rallado", "pan molido")])
def test_el_nombre_dominicano_va_en_la_glosa(sql, fila, glosa):
    bloque = sql.split(f"WHERE name = '{fila}'")[0].rsplit("UPDATE public.master_ingredients SET", 1)[1]
    assert f"gloss_es = '{glosa}'" in bloque


def test_el_queso_de_hoja_documenta_su_procedencia_sin_inventar_un_puntero(sql):
    bloque = sql.split("WHERE name = 'Queso de hoja'")[0].rsplit("UPDATE public.master_ingredients SET", 1)[1]
    assert "nutrition_source_ref" in bloque and "funditas" in bloque
    assert "fdc_id" not in bloque, "un fdc_id con valores distintos es el puntero que miente"
    assert "_per_100g" not in bloque, "sin etiqueta no hay evidencia para cambiar sus valores"


def test_no_renombra_filas_y_verifica_lo_que_escribe(sql):
    assert not re.search(r"SET\s+name\s*=|,\s*name\s*=", sql)
    assert "RAISE EXCEPTION" in sql


# ── el sazón: con precio, ningún token de otro país puede reclamarlo ─────────
@pytest.mark.parametrize("nombre", ["Sazón con culantro y achiote", "sazon con achiote",
                                    "1 sobre de sazón con culantro y achiote"])
def test_el_token_achiote_de_mexico_ya_no_reclama_el_sazon(nombre):
    """Al salir su token de PR, el sazón seguía reclamado por el token SUELTO `achiote` de México
    (la semilla), que casa por palabra dentro de «Sazón con culantro y ACHIOTE». Mientras no tenía
    precio daba igual; con precio es el bug que `test_i2_registry_collision_sweep_...` caza."""
    import shopping_calculator as sc
    assert sc.is_country_catalog_unpriced_item(nombre) is False


@pytest.mark.parametrize("nombre", ["Achiote", "Aceite de achiote"])
def test_el_achiote_de_verdad_conserva_su_rescate(nombre):
    """La exclusión es del SAZÓN, no del achiote: la semilla (MX) y el aceite (PR) siguen sin precio
    RD y el agregador tiene que seguir conservándolos en la lista."""
    import shopping_calculator as sc
    assert sc.is_country_catalog_unpriced_item(nombre) is True


# ── envases: lo que se compra es lo de sus capturas, no «¼ lb» ───────────────
MIG_ENV = "p1_do_despensa_de_su_mercado_envases_2026_09_10.sql"


@pytest.fixture(scope="module")
def sql_env():
    return (BACK / "migrations" / MIG_ENV).read_text(encoding="utf-8")


def _bloque(sql_txt, fila):
    return sql_txt.split(f"WHERE name = '{fila}'")[0].rsplit("UPDATE public.master_ingredients SET", 1)[1]


def test_la_migracion_de_envases_vive_en_los_dos_directorios(sql_env):
    assert (BACK.parent / "migrations" / MIG_ENV).read_text(encoding="utf-8") == sql_env


@pytest.mark.parametrize("fila,envase,gramos,precio", [
    ("Sazón con culantro y achiote", "caja", 40, 99),   # 8 sobres · 1,41 oz
    ("Azúcar morena", "funda", 907, 71),                 # Wala 2 lb (y 5 lb a RD$157)
    ("Pan rallado", "paquete", 454, 73),                 # Buenhorno 1 lb
    ("Hummus", "pote", 283, 299),                        # Dietz & Watson 10 oz
])
def test_cada_producto_trae_el_envase_de_su_captura(sql_env, fila, envase, gramos, precio):
    b = _bloque(sql_env, fila)
    assert re.search(rf"market_container\s*=\s*'{envase}'", b)
    assert re.search(rf"container_weight_g\s*=\s*{gramos}\b", b)
    assert re.search(rf'"grams":\s*{gramos},[^}}]*"price":\s*{precio}\b', b)


def test_la_funda_grande_del_azucar_tambien_viaja(sql_env):
    """El agregador elige el envase por COSTE total: 5 tazas salen en dos fundas de 2 lb (RD$142),
    no en una de 5 lb (RD$157). Sin la grande no hay elección que hacer."""
    assert re.search(r'"grams":\s*2268,[^}]*"price":\s*157\b', _bloque(sql_env, "Azúcar morena"))


def test_los_envases_no_tocan_el_precio_por_libra(sql_env):
    """El coste por ración del plato lee `price_per_lb` por gramo y ya era correcto."""
    codigo = "\n".join(l for l in sql_env.splitlines() if not l.lstrip().startswith("--"))
    assert "price_per_lb" not in codigo


def test_el_sobre_del_sazon_pesa_5_g_y_hay_densidad_de_taza(sql_env):
    """Sin peso de sobre, «1 sobre» caía al peso por defecto de Despensa (450 g); sin densidad de
    taza, «1 taza de pan rallado» tomaba el peso de UNA UNIDAD de pan (30 g) y salía en «5 Uds.»."""
    assert re.search(r"density_g_per_unit\s*=\s*5\b", _bloque(sql_env, "Sazón con culantro y achiote"))
    assert re.search(r"density_g_per_cup\s*=\s*220\b", _bloque(sql_env, "Azúcar morena"))
    assert re.search(r"density_g_per_cup\s*=\s*108\b", _bloque(sql_env, "Pan rallado"))


@pytest.mark.parametrize("linea,fila,envase", [
    ("1 sobre de sazón con culantro y achiote", "Sazón con culantro y achiote", "caja"),
    ("1 taza de azúcar morena", "Azúcar morena", "funda"),
    ("1 taza de pan rallado", "Pan rallado", "paquete"),
    ("1 pote de hummus", "Hummus", "pote"),
])
def test_la_lista_real_compra_el_envase_y_lo_cobra_a_su_precio(monkeypatch, linea, fila, envase):
    """Contra el agregador REAL y el catálogo vivo. Antes: «¼ lb de sazón» a RD$280,66, el pan
    molido en «5 Uds.» sin coste, el hummus en «1 lb» a RD$478,40. El precio esperado se lee del
    propio catálogo (el envase más chico): el test fija el MECANISMO, no un número que el dueño
    puede mover mañana."""
    import shopping_calculator as sc
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    catalogo = sc.get_master_ingredients() or []
    if not catalogo:
        pytest.skip("sin catálogo vivo")
    row = next((r for r in catalogo if r.get("name") == fila), None)
    assert row is not None, f"desapareció la fila {fila!r}"
    assert row.get("market_packages"), f"{fila}: la migración de envases no está aplicada"
    res = sc.aggregate_and_deduct_shopping_list([linea], structured=True)
    items = res.get("items") if isinstance(res, dict) else res
    it = next((i for i in items or [] if i.get("name") == fila), None)
    assert it is not None, f"{fila} desapareció de la lista"
    assert it.get("market_unit") == envase, f"{fila}: sale como {it.get('display_qty')!r}"
    precio = min(float(p["price"]) for p in row["market_packages"])
    assert float(it.get("estimated_cost_rd") or 0) == pytest.approx(precio)
