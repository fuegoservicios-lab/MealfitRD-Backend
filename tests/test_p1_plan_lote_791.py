"""[P1-PLAN-LOTE-791 · 2026-09-28] G58 (datos): las filas de catálogo-país envasadas nacen con envase.

LO QUE SE MIDIÓ. Las 130 filas sin precio del registro de catálogo-país no tenían NINGÚN dato de
envase; las 64 que se venden en envase perdían el envase en la lista («1 sobre de Azafrán» → «1 lb de
Azafrán»): replay del agregador sobre una copia de la tabla, 0 de 64. Con esta migración aplicada a la
copia, 64 de 64 (más Dátiles y Cúrcuma, el mismo hueco en DO) conservan su envase con la etiqueta de su
país (lote 790): «1 sobre (0,4 g) de Azafrán», «1 lata (50 g) de Anchoas», «1 frasco (8 oz) de Adobo».

QUÉ ANCLA ESTE TEST (parser-based, sin DB — el ancla del DATO es el CHECK de la migración):
  A. idempotencia y forma (P3-MIGRATION-IDEMPOTENCE-DOC);
  B. cubre EXACTAMENTE las 64 filas beta envasadas de la foto de producción, y el bloque DO son
     exactamente las dos filas DO con unidad de envase sin peso;
  C. jamás escribe una columna de precio ni `market_packages`, y el bloque beta exige precio 0;
  D. la procedencia (ODbL) va fila por fila; PR declara US;
  E. la lista de unidades de envase del CHECK es la del agregador (paridad) y es NULL-segura;
  F. aplicada a la foto, deja 0 filas que violen el CHECK (hoy, 65).

REVISIÓN RONDA 1 (verificar_envases.md): rótulos de las 66 filas contra una tabla LITERAL (el replay
era circular), los chiles secos en paquete fuera del tope de condimentos (compra corta: 85 g de 240),
una sola regla de tamaño (Adobo 227 g, Sofrito 340 g en frasco), la cabecera que ya no promete lo que el
bloque DO no cumple, y la licencia ODbL + doc del lote.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_MIG_NAME = "p1_plan_lote_791_envases_beta_2026_09_28.sql"
_MIG = _BACKEND / "migrations" / _MIG_NAME
_FIX = _BACKEND / "tests" / "fixtures" / "envases_catalogo_2026_09_28.json"
_PK = {"paquete", "botella", "lata", "frasco", "funda", "sobre", "envase", "pote", "litro"}
_FILA = re.compile(
    r"\(\s*'((?:[^']|'')*)'\s*,\s*'((?:[^']|'')*)'\s*,\s*([\d.]+)\s*,\s*'([A-Z]{2})'\s*,\s*'((?:[^']|'')*)'\s*\)")


@pytest.fixture(scope="module")
def sql():
    assert _MIG.exists(), f"falta la migración {_MIG_NAME}"
    return _MIG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def foto():
    return json.loads(_FIX.read_text(encoding="utf-8"))["filas"]


def _bloques(sql):
    cuerpos = re.findall(r"FROM \(VALUES\n(.*?)\n\) AS v\(name, envase, gramos, pais, fuente\)", sql, re.S)
    assert len(cuerpos) == 2, "esperaba dos bloques VALUES (beta y DO)"
    out = []
    for c in cuerpos:
        filas = [(n.replace("''", "'"), e, float(g), p, f.replace("''", "'")) for n, e, g, p, f in _FILA.findall(c)]
        assert len(filas) == len([ln for ln in c.splitlines() if ln.strip()]), "una fila del VALUES no se pudo leer"
        out.append(filas)
    return out


def _sin_precio(r):
    return not r.get("con_precio")


# ── A. Forma ───────────────────────────────────────────────────────────────────────────────────

def test_la_migracion_es_idempotente_y_lleva_su_ancla(sql):
    assert "tooltip-anchor: P1-PLAN-LOTE-791" in sql
    assert "ADD COLUMN IF NOT EXISTS container_source text" in sql
    assert "DROP CONSTRAINT IF EXISTS master_ingredients_envase_requires_weight" in sql
    assert sql.count("RAISE EXCEPTION") >= 5
    assert sql.count("AND m.container_weight_g IS NULL") == 2, "los dos UPDATE sólo llenan el hueco"
    assert "available_sizes_g  = jsonb_build_array(v.gramos)" in sql
    # Un `$$` dentro de un comentario confunde a cualquier herramienta que trocee por bloques DO.
    cabecera = sql.split("ALTER TABLE", 1)[0]
    assert "$$" not in cabecera


# ── B. Cobertura exacta ────────────────────────────────────────────────────────────────────────

def test_cubre_exactamente_las_64_filas_beta_envasadas(sql, foto):
    beta, _do = _bloques(sql)
    esperadas = {r["name"] for r in foto if _sin_precio(r) and (r.get("default_unit") or "") in _PK}
    assert len(esperadas) == 64, "la foto ya no tiene 64 filas beta envasadas: re-medir antes de tocar"
    assert {f[0] for f in beta} == esperadas
    assert len(beta) == 64


def test_el_bloque_DO_es_exactamente_el_hueco_de_Datiles_y_Curcuma(sql, foto):
    import shopping_calculator as sc
    _beta, do = _bloques(sql)
    hueco_do = {r["name"] for r in foto
                if not _sin_precio(r)
                and str(r.get("default_unit") or "").strip().lower() in sc._CONTAINER_UNIT_ALIASES
                and not (r.get("container_weight_g") or 0) > 0}
    assert hueco_do == {"Dátiles", "Cúrcuma"}
    assert {f[0] for f in do} == hueco_do
    assert {f[3] for f in do} == {"DO"}


# ── C. Jamás el precio ─────────────────────────────────────────────────────────────────────────

def test_no_escribe_precio_ni_market_packages(sql):
    for bloque in re.findall(r"UPDATE public\.master_ingredients AS m\s+SET(.*?)\nFROM", sql, re.S):
        assert "price" not in bloque and "market_packages" not in bloque, bloque
    beta_where = sql.split(") AS v(name, envase, gramos, pais, fuente)", 1)[1].split(";", 1)[0]
    assert "COALESCE(m.price_per_lb, 0) = 0" in beta_where
    assert "COALESCE(m.price_per_unit, 0) = 0" in beta_where


def test_el_envase_es_el_que_la_fila_ya_declara(sql, foto):
    unidad = {r["name"]: r.get("default_unit") for r in foto}
    beta, do = _bloques(sql)
    for n, env, g, _p, _f in beta + do:
        assert 0 < g <= 1500, (n, g)
        if n == "Champús":
            assert unidad[n] == "litro" and env == "botella"
        elif n == "Sofrito":
            # [revisión ronda 1, defecto 4] la fila dice «paquete», pero todas las muestras con marca
            # (Goya, Iberia, Loisa) son frascos de 12 oz: «1 paquete (1.5 lbs)» rotulaba un frasco.
            assert unidad[n] == "paquete" and env == "frasco"
        else:
            assert env == unidad[n], (n, env, unidad[n])


def test_la_regla_del_tamano_es_una_sola(sql):
    """[revisión ronda 1, defecto 4] La regla se aplicaba a medias: en Anchoas y Sazonador ganaba la
    moda minorista frente a una mediana arrastrada por formatos grandes, pero en Adobo (moda 227 g,
    Goya 8 oz ×4, frente a una mediana 340 g arrastrada por los 2 lb de Badia) y Sofrito (moda 340 g,
    los frascos de 12 oz ×9, frente a 680 g arrastrada por las tarrinas de 32-64 oz) ganaba la mediana."""
    beta, _do = _bloques(sql)
    v = {f[0]: f for f in beta}
    assert v["Adobo"][2] == 227 and "moda" in v["Adobo"][4] and "-> 227 g" in v["Adobo"][4]
    assert v["Sofrito"][2] == 340 and v["Sofrito"][1] == "frasco" and "-> 340 g" in v["Sofrito"][4]
    cabecera = sql.split("ALTER TABLE", 1)[0]
    assert "REGLA DEL TAMAÑO" in cabecera


def test_la_cabecera_no_promete_lo_que_el_bloque_DO_no_cumple(sql):
    """[revisión ronda 1, defecto 3] Decía «JAMÁS TOCA UNA FILA DO CON PRECIO» y el bloque 2 llena el
    envase de Dátiles y Cúrcuma, que son DO con precio. La excepción se declara, con lo que falta."""
    cabecera = sql.split("ALTER TABLE", 1)[0]
    assert "JAMÁS TOCA UNA FILA DO CON PRECIO" not in cabecera
    assert "OK DEL DUEÑO" in cabecera and "Dátiles" in cabecera and "Cúrcuma" in cabecera


# ── D. Procedencia ─────────────────────────────────────────────────────────────────────────────

def test_cada_fila_dice_de_donde_sale_su_envase(sql):
    beta, do = _bloques(sql)
    for n, _e, _g, pais, fuente in beta + do:
        assert fuente.startswith("[P1-PLAN-LOTE-791] Open Food Facts (ODbL)"), n
        assert "OFF" in fuente, n
        if pais == "PR":
            assert "PR usa US declarado" in fuente, n
    assert "© colaboradores de Open Food Facts, ODbL" in sql, "la atribución ODbL es obligatoria"


def test_el_sanity_nombra_las_mismas_filas_que_los_values(sql):
    beta, do = _bloques(sql)
    arr = lambda nombre: re.findall(r"'((?:[^']|'')*)'", re.search(nombre + r"\s+text\[\] := ARRAY\[(.*?)\];", sql, re.S).group(1))
    assert arr("_beta") == [f[0] for f in beta]
    assert arr("_do") == [f[0] for f in do]
    assert f"IF _n <> {len(beta) + len(do)} THEN" in sql


# ── E. El CHECK ────────────────────────────────────────────────────────────────────────────────

def _alias_del_check(sql):
    listas = re.findall(r"<> ALL \(ARRAY\[(.*?)\]::text\[\]\)", sql, re.S)
    assert len(listas) == 2, "el predicado vive en la guarda previa y en la constraint"
    conjuntos = [set(re.findall(r"'([^']*)'", l)) for l in listas]
    assert conjuntos[0] == conjuntos[1]
    return conjuntos[0]


def test_las_unidades_de_envase_del_check_son_las_del_agregador(sql):
    import shopping_calculator as sc
    assert _alias_del_check(sql) == set(sc._CONTAINER_UNIT_ALIASES), (
        "_CONTAINER_UNIT_ALIASES cambió: actualiza el CHECK (nueva migración), no este test")


def test_el_check_es_null_seguro(sql):
    """Un CHECK que evalúa a NULL PASA. `container_weight_g > 0` a secas no mordería una fila sin peso."""
    pred = sql.split("ADD CONSTRAINT master_ingredients_envase_requires_weight", 1)[1].split(";", 1)[0]
    assert pred.count("COALESCE(container_weight_g, 0) > 0") == 2
    assert "container_weight_g > 0" not in pred.replace("COALESCE(container_weight_g, 0) > 0", "")


def _viola(r, alias):
    cw = r.get("container_weight_g") or 0
    du = str(r.get("default_unit") or "").strip().lower()
    return (r.get("market_container") is not None and not cw > 0) or (du in alias and not cw > 0)


def test_aplicada_a_la_foto_no_deja_ninguna_fila_que_viole_el_check(sql, foto):
    alias = _alias_del_check(sql)
    antes = sorted(r["name"] for r in foto if _viola(r, alias))
    assert len(antes) == 65, antes  # 63 beta (Champús vende por «litro») + Dátiles + Cúrcuma
    beta, do = _bloques(sql)
    por_nombre = {r["name"]: dict(r) for r in foto}
    for n, env, g, _p, _f in beta:
        r = por_nombre[n]
        if r.get("container_weight_g") is None and _sin_precio(r):
            r.update(market_container=env, container_weight_g=g, available_sizes_g=[g])
    for n, env, g, _p, _f in do:
        r = por_nombre[n]
        if r.get("container_weight_g") is None and r.get("market_container") is None and not r.get("market_packages_n"):
            r.update(market_container=env, container_weight_g=g, available_sizes_g=[g])
    despues = sorted(n for n, r in por_nombre.items() if _viola(r, alias))
    assert despues == []


# ── F. Datos + código (lote 790): la lista rotula el envase curado en el sistema del país ───────
#
# [revisión ronda 1, defecto 5] El replay del implementador calculaba la etiqueta ESPERADA con el mismo
# `_sku_size_label` que se probaba: «66/66 correctas» sólo decía que el envase se conservaba. Aquí el
# oráculo es una tabla LITERAL, revisada a mano fila por fila contra la regla de cada sistema: ES/MX/CO
# en g/ml/L con coma; DO/US/PR como el rótulo dominicano de siempre (lb/oz/ml, «454ml» incluido). Si una
# etiqueta cambia, cambia porque alguien lo decidió aquí, no porque la función cambió.

_ETIQUETAS_66 = [
    # (fila, país de la lista, línea de la receta, rótulo esperado, envase esperado)
    ("Azafrán", "ES", "1 sobre de Azafrán", "0,4 g", "sobre"),
    ("Alioli", "ES", "1 frasco de Alioli", "180 g", "frasco"),
    ("Anchoas", "ES", "1 lata de Anchoas", "50 g", "lata"),
    ("Mazapán", "ES", "1 paquete de Mazapán", "200 g", "paquete"),
    ("Membrillo dulce", "ES", "1 paquete de Membrillo dulce", "400 g", "paquete"),
    ("Turrón", "ES", "1 paquete de Turrón", "200 g", "paquete"),
    ("Nata", "ES", "1 botella de Nata", "200 ml", "botella"),
    ("Aceite de achiote", "MX", "1 botella de Aceite de achiote", "280 ml", "botella"),
    ("Achiote", "MX", "1 sobre de Achiote", "125 g", "sobre"),
    ("Chile ancho", "MX", "1 paquete de Chile ancho", "85 g", "paquete"),
    ("Chile chipotle", "MX", "1 paquete de Chile chipotle", "85 g", "paquete"),
    ("Chile de árbol", "MX", "1 paquete de Chile de árbol", "85 g", "paquete"),
    ("Chile guajillo", "MX", "1 paquete de Chile guajillo", "85 g", "paquete"),
    ("Chile mulato", "MX", "1 paquete de Chile mulato", "85 g", "paquete"),
    ("Chile pasilla", "MX", "1 paquete de Chile pasilla", "85 g", "paquete"),
    ("Chocolate de mesa", "MX", "1 paquete de Chocolate de mesa", "630 g", "paquete"),
    ("Flor de Jamaica", "MX", "1 paquete de Flor de Jamaica", "227 g", "paquete"),
    ("Frijoles refritos", "MX", "1 lata de Frijoles refritos", "430 g", "lata"),
    ("Huitlacoche", "MX", "1 lata de Huitlacoche", "186 g", "lata"),
    ("Panela", "MX", "1 paquete de Panela", "227 g", "paquete"),
    ("Tortilla de maíz", "MX", "1 paquete de Tortilla de maíz", "680 g", "paquete"),
    ("Crema mexicana", "MX", "1 pote de Crema mexicana", "450 g", "pote"),
    ("Arequipe", "CO", "1 pote de Arequipe", "400 g", "pote"),
    ("Natilla", "CO", "1 pote de Natilla", "300 g", "pote"),
    ("Suero costeño", "CO", "1 botella de Suero costeño", "200 ml", "botella"),
    ("Champús", "CO", "1 litro de Champús", "1 L", "botella"),
    ("Aderezo ranch", "US", "1 botella de Aderezo ranch", "454ml", "botella"),
    ("Jarabe de arce", "US", "1 botella de Jarabe de arce", "354ml", "botella"),
    ("Kétchup", "US", "1 botella de Kétchup", "567ml", "botella"),
    ("Salsa barbacoa", "US", "1 botella de Salsa barbacoa", "510ml", "botella"),
    ("Salsa inglesa", "US", "1 botella de Salsa inglesa", "296ml", "botella"),
    ("Crema agria", "US", "1 envase de Crema agria", "1 lb", "envase"),
    ("Crema mitad y mitad", "US", "1 envase de Crema mitad y mitad", "473ml", "envase"),
    ("Ensalada de macarrones", "US", "1 envase de Ensalada de macarrones", "1 lb", "envase"),
    ("Suero de mantequilla", "US", "1 envase de Suero de mantequilla", "946ml", "envase"),
    ("Chile en polvo", "US", "1 frasco de Chile en polvo", "71g", "frasco"),
    ("Arándanos rojos", "US", "1 funda de Arándanos rojos", "340g", "funda"),
    ("Bolitas de papa", "US", "1 funda de Bolitas de papa", "2 lbs", "funda"),
    ("Malvaviscos", "US", "1 funda de Malvaviscos", "284g", "funda"),
    ("Papas ralladas", "US", "1 funda de Papas ralladas", "1.8 lbs", "funda"),
    ("Pretzels", "US", "1 funda de Pretzels", "340g", "funda"),
    ("Chili con carne", "US", "1 lata de Chili con carne", "425g", "lata"),
    ("Frijoles horneados", "US", "1 lata de Frijoles horneados", "1 lb", "lata"),
    ("Salsa de salchicha", "US", "1 lata de Salsa de salchicha", "425g", "lata"),
    ("Bagels", "US", "1 paquete de Bagels", "510g", "paquete"),
    ("Galletas Graham", "US", "1 paquete de Galletas Graham", "408g", "paquete"),
    ("Masa para pie", "US", "1 paquete de Masa para pie", "425g", "paquete"),
    ("Mezcla para panqueques", "US", "1 paquete de Mezcla para panqueques", "1.8 lbs", "paquete"),
    ("Pan de maíz", "US", "1 paquete de Pan de maíz", "1 lb", "paquete"),
    ("Panecillos de mantequilla", "US", "1 paquete de Panecillos de mantequilla", "425g", "paquete"),
    ("Panecillos ingleses", "US", "1 paquete de Panecillos ingleses", "340g", "paquete"),
    ("Pepperoni", "US", "1 paquete de Pepperoni", "170g", "paquete"),
    ("Queso en hebras", "US", "1 paquete de Queso en hebras", "½ lb", "paquete"),
    ("Sémola de maíz", "US", "1 paquete de Sémola de maíz", "2 lbs", "paquete"),
    ("Wafles", "US", "1 paquete de Wafles", "255g", "paquete"),
    ("Sazonador para tacos", "US", "1 sobre de Sazonador para tacos", "28g", "sobre"),
    ("Aceitunas rellenas", "PR", "1 frasco de Aceitunas rellenas", "12 oz", "frasco"),
    ("Adobo", "PR", "1 frasco de Adobo", "8 oz", "frasco"),
    ("Alcaparrado", "PR", "1 frasco de Alcaparrado", "¼ lb", "frasco"),
    ("Especias para arroz con dulce", "PR", "1 sobre de Especias para arroz con dulce", "28g", "sobre"),
    ("Harina de yuca", "PR", "1 paquete de Harina de yuca", "1 lb", "paquete"),
    ("Pique", "PR", "1 botella de Pique", "148ml", "botella"),
    ("Ron de cocina", "PR", "1 botella de Ron de cocina", "750ml", "botella"),
    ("Sofrito", "PR", "1 paquete de Sofrito", "12 oz", "frasco"),   # la receta dice paquete; se compra el frasco
    ("Dátiles", "DO", "1 paquete de Dátiles", "340g", "paquete"),
    ("Cúrcuma", "DO", "1 frasco de Cúrcuma", "57g", "frasco"),
]


def test_la_tabla_de_etiquetas_cubre_las_66_filas(sql):
    beta, do = _bloques(sql)
    assert [t[0] for t in _ETIQUETAS_66] == [f[0] for f in beta + do]
    pais_de = {f[0]: f[3] for f in beta + do}
    for n, pais, *_ in _ETIQUETAS_66:
        assert pais == pais_de[n], n


@pytest.mark.parametrize("nombre,pais,linea,rotulo,envase", _ETIQUETAS_66, ids=[t[0] for t in _ETIQUETAS_66])
def test_cada_envase_curado_llega_a_la_lista_con_el_rotulo_de_su_pais(sql, foto, monkeypatch,
                                                                      nombre, pais, linea, rotulo, envase):
    import envase_pais as ep
    import shopping_calculator as sc
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    fila = _fila_del_lote(nombre, foto, sql)
    if pais == "DO":
        fila["price_per_lb"] = 99.0  # Dátiles y Cúrcuma son filas DO CON precio
    monkeypatch.setattr(sc, "_master_cache", [fila])
    monkeypatch.setattr(sc, "_master_cache_ts", time.time() + 10 ** 6)
    monkeypatch.setattr(sc, "_VERIFIED_SHOPPING_NAMES", None, raising=False)
    with ep.lista_de_pais(pais):
        res = sc.aggregate_and_deduct_shopping_list([linea], structured=True, categorize=False,
                                                    cycle_days=7, num_days=7)
    it = next((i for i in res if isinstance(i, dict) and i.get("name") == nombre), None)
    assert it is not None, (nombre, [i.get("name") for i in res if isinstance(i, dict)])
    assert it["display_string"] == f"1 {envase} ({rotulo}) de {nombre}", it["display_string"]
    assert str(it.get("market_unit")) == envase
    if pais in ("ES", "MX", "CO"):
        assert not re.search(r"\b(lbs?|oz)\b", it["display_string"]), it["display_string"]


# ── H. Los chiles secos del lote son COMIDA, no especiero (revisión ronda 1, defecto 1) ─────────
#
# El tope de condimentos (P1-SHOPLIST-SANITY-CAP) capa lo que es Despensa con envase ≤ 120 g. Con
# los envases de este lote, los seis chiles secos (paquete de 85 g) entraban: 4 × «60 g de Chile
# guajillo» en una semana mexicana (240 g) salían «1 paquete de Chile guajillo» — 85 g de 240, sin
# nota de cobertura y sin el tamaño. Antes del lote salía «½ lb». El especiero (sobre, frasco, pote)
# dura meses; un PAQUETE es comida: los chiles de una salsa, las nueces de una merienda.

_ESPECIERO_791 = {"Azafrán", "Chile en polvo", "Sazonador para tacos", "Especias para arroz con dulce",
                  "Alcaparrado", "Cúrcuma"}
_COMIDA_EN_PAQUETE_791 = {"Chile ancho", "Chile chipotle", "Chile de árbol", "Chile guajillo",
                          "Chile mulato", "Chile pasilla"}


def _fila_del_lote(nombre, foto, sql):
    beta, do = _bloques(sql)
    v = {f[0]: f for f in beta + do}[nombre]
    r = {x["name"]: x for x in foto}[nombre]
    return {"name": nombre, "category": r["category"], "aliases": [], "default_unit": r["default_unit"],
            "price_per_lb": 0.0, "price_per_unit": 0.0, "market_container": v[1],
            "container_weight_g": v[2], "available_sizes_g": [v[2]], "market_packages": None,
            "density_g_per_unit": None, "density_g_per_cup": None, "shelf_life_days": 180, "name_en": None}


def test_cada_fila_pequena_del_lote_sabe_si_es_especiero_o_comida(sql, foto):
    """Todas las filas del lote que el tope podría capar (Despensa, envase ≤ 120 g) están clasificadas
    a propósito, y el tope real (`_apply_condiment_sanity_cap`) sólo capa el especiero."""
    import shopping_calculator as sc
    beta, do = _bloques(sql)
    cat = {r["name"]: r["category"] for r in foto}
    pequenas = {n for n, _e, g, _p, _f in beta + do
                if str(cat[n]).lower().startswith("despensa") and g <= sc._CONDIMENT_MAX_CONTAINER_G}
    assert pequenas == _ESPECIERO_791 | _COMIDA_EN_PAQUETE_791, (
        "una fila nueva del lote cae bajo el tope de condimentos: clasifícala (especiero o comida)")
    for n in sorted(pequenas):
        fila = _fila_del_lote(n, foto, sql)
        obj = {"name": n, "market_qty_numeric": 5.0, "market_qty": "5",
               "market_unit": fila["market_container"], "display_qty": f"5 {fila['market_container']}s"}
        capo = sc._apply_condiment_sanity_cap(obj, fila, "DESPENSA", 7)
        assert capo is (n in _ESPECIERO_791), (n, fila["market_container"], obj)


def test_cuatro_guajillos_de_60_g_no_se_quedan_en_un_paquete(sql, foto, monkeypatch):
    import envase_pais as ep
    import shopping_calculator as sc
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    monkeypatch.setattr(sc, "_master_cache", [_fila_del_lote("Chile guajillo", foto, sql),
                                              _fila_del_lote("Chile en polvo", foto, sql)])
    monkeypatch.setattr(sc, "_master_cache_ts", time.time() + 10 ** 6)
    monkeypatch.setattr(sc, "_VERIFIED_SHOPPING_NAMES", None, raising=False)
    with ep.lista_de_pais("MX"):
        res = sc.aggregate_and_deduct_shopping_list(
            ["60 g de Chile guajillo"] * 4 + ["1 cdta de Chile en polvo"] * 30,
            structured=True, categorize=False, cycle_days=7, num_days=7)
    por_nombre = {i.get("name"): i for i in res if isinstance(i, dict)}
    chile = por_nombre["Chile guajillo"]
    assert chile["market_qty_numeric"] == 3, chile["display_string"]  # 240 g / 85 g → 3 paquetes
    assert "85 g" in chile["display_string"], chile["display_string"]
    # El especiero sigue capado: 30 cucharaditas de chile en polvo en una semana no son 2 frascos.
    assert por_nombre["Chile en polvo"]["market_qty_numeric"] == 1


def test_el_knob_devuelve_el_tope_a_los_paquetes(sql, foto, monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setenv("MEALFIT_CONDIMENT_CAP_FOOD_PACKAGE_EXEMPT", "false")
    fila = _fila_del_lote("Chile guajillo", foto, sql)
    obj = {"name": "Chile guajillo", "market_qty_numeric": 3.0, "market_unit": "paquete",
           "display_qty": "3 paquetes"}
    assert sc._apply_condiment_sanity_cap(obj, fila, "DESPENSA", 7) is True


# ── I. Licencia y doc (revisión ronda 1, defectos 7 y 10) ──────────────────────────────────────

def test_la_licencia_del_uso_nuevo_esta_documentada():
    """`docs/data_provenance_licenses.md` sólo autorizaba Open Food Facts para embutidos: el uso nuevo
    (tamaños de envase de 66 filas) se declara con su atribución y lo que queda por decidir."""
    doc = (_BACKEND / "docs" / "data_provenance_licenses.md").read_text(encoding="utf-8")
    fila = next((ln for ln in doc.splitlines() if "P1-PLAN-LOTE-791" in ln), "")
    assert "Open Food Facts" in fila and "envase" in fila and "ODbL" in fila, fila
    assert "atribución pública" in doc


def test_el_lote_tiene_su_doc():
    doc = (_BACKEND / "docs" / "envases_y_catalogo_por_pais.md").read_text(encoding="utf-8")
    for ancla in ("P1-PLAN-LOTE-790", "P1-PLAN-LOTE-791", "MEALFIT_COUNTRY_CATALOG_FOREIGN_FLAG",
                  "MEALFIT_CONDIMENT_CAP_FOOD_PACKAGE_EXEMPT", "MEALFIT_UNIT_SYSTEM_BY_COUNTRY",
                  "catalogo_de_otro_pais", "p1_plan_lote_791_envases_beta_2026_09_28.sql"):
        assert ancla in doc, ancla
    assert "docs/envases_y_catalogo_por_pais.md" in (_BACKEND / "envase_pais.py").read_text(encoding="utf-8")


# ── G. SSOT dual-dir ───────────────────────────────────────────────────────────────────────────

def test_la_copia_del_repo_raiz_es_identica_si_existe():
    raiz = _BACKEND.parent / "migrations" / _MIG_NAME
    if not (_BACKEND.parent / "migrations").is_dir():
        pytest.skip("workspace raíz ausente: la copia en migrations/ la sube el integrador")
    assert raiz.exists(), "P3-MIGRATIONS-SSOT: falta la copia en migrations/ del repo raíz"
    assert raiz.read_bytes() == _MIG.read_bytes()
