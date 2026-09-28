"""[P1-PLAN-LOTE-791 · 2026-09-28] G58 (datos): las filas de catálogo-país envasadas nacen con envase.

LO QUE SE MIDIÓ. Las 130 filas sin precio del registro de catálogo-país no tenían NINGÚN dato de
envase; las 64 que se venden en envase perdían el envase en la lista («1 sobre de Azafrán» → «1 lb de
Azafrán»): replay del agregador sobre una copia de la tabla, 0 de 64. Con esta migración aplicada a la
copia, 64 de 64 (más Dátiles y Cúrcuma, el mismo hueco en DO) conservan su envase con la etiqueta de su
país (lote 790): «1 sobre (0,4 g) de Azafrán», «1 lata (50 g) de Anchoas», «1 frasco (12 oz) de Adobo».

QUÉ ANCLA ESTE TEST (parser-based, sin DB — el ancla del DATO es el CHECK de la migración):
  A. idempotencia y forma (P3-MIGRATION-IDEMPOTENCE-DOC);
  B. cubre EXACTAMENTE las 64 filas beta envasadas de la foto de producción, y el bloque DO son
     exactamente las dos filas DO con unidad de envase sin peso;
  C. jamás escribe una columna de precio ni `market_packages`, y el bloque beta exige precio 0;
  D. la procedencia (ODbL) va fila por fila; PR declara US;
  E. la lista de unidades de envase del CHECK es la del agregador (paridad) y es NULL-segura;
  F. aplicada a la foto, deja 0 filas que violen el CHECK (hoy, 65).
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
        else:
            assert env == unidad[n], (n, env, unidad[n])


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

def test_los_envases_curados_llegan_a_la_lista_con_la_etiqueta_de_su_pais(sql, foto, monkeypatch):
    import envase_pais as ep
    import shopping_calculator as sc
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    beta, do = _bloques(sql)
    valores = {f[0]: f for f in beta + do}
    por_nombre = {r["name"]: r for r in foto}
    casos = [("Azafrán", "ES", "1 sobre de Azafrán", "(0,4 g)"),
             ("Anchoas", "ES", "1 lata de Anchoas", "(50 g)"),
             ("Adobo", "PR", "1 frasco de Adobo", "(12 oz)"),
             ("Champús", "CO", "1 litro de Champús", "(1 L)")]
    filas = []
    for n, *_ in casos:
        r = por_nombre[n]
        env, g = valores[n][1], valores[n][2]
        filas.append({"name": n, "category": r["category"], "aliases": [], "default_unit": r["default_unit"],
                      "price_per_lb": 0.0, "price_per_unit": 0.0, "market_container": env,
                      "container_weight_g": g, "available_sizes_g": [g], "market_packages": None,
                      "density_g_per_unit": None, "density_g_per_cup": None, "shelf_life_days": 180, "name_en": None})
    monkeypatch.setattr(sc, "_master_cache", filas)
    monkeypatch.setattr(sc, "_master_cache_ts", time.time() + 10 ** 6)
    monkeypatch.setattr(sc, "_VERIFIED_SHOPPING_NAMES", None, raising=False)
    for n, pais, linea, etiqueta in casos:
        with ep.lista_de_pais(pais):
            res = sc.aggregate_and_deduct_shopping_list([linea], structured=True)
        it = next((i for i in res if i.get("name") == n), None)
        assert it is not None, (n, [i.get("name") for i in res])
        assert etiqueta in it["display_string"], (n, it["display_string"])
        assert str(it.get("market_unit")).rstrip("s") == valores[n][1], (n, it.get("market_unit"))


# ── G. SSOT dual-dir ───────────────────────────────────────────────────────────────────────────

def test_la_copia_del_repo_raiz_es_identica_si_existe():
    raiz = _BACKEND.parent / "migrations" / _MIG_NAME
    if not (_BACKEND.parent / "migrations").is_dir():
        pytest.skip("workspace raíz ausente: la copia en migrations/ la sube el integrador")
    assert raiz.exists(), "P3-MIGRATIONS-SSOT: falta la copia en migrations/ del repo raíz"
    assert raiz.read_bytes() == _MIG.read_bytes()
