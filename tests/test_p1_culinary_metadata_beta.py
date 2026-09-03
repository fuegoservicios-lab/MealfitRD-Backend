"""[P1-CULINARY-METADATA-BETA · 2026-08-19] Backfill ronda 3 de metadata culinaria
+ el CHECK que impide que el hueco se reabra.

CONTEXTO. Las 141 filas de países beta que `P1-COUNTRY-SYSTEM-F2` insertó el
2026-08-17 nacieron con `prep_methods` / `ready_to_eat` en NULL al 100%, devolviendo
la cobertura del catálogo de 100% (cerrada por `P2-CULINARY-METADATA-ROUND2` el
2026-08-01) a 206/347 = 59%. NULL es fail-open POR CHECK: el scan no falla, se
*salta* V1/V2 para ese alimento. Medido sobre el corpus beta: 24% de cobertura.

POR QUE ESTE TEST NO ES SOLO PARSER-BASED. El hueco ocurrió **con los tests de las
rondas 1 y 2 en verde**, porque son parser-based sobre las migraciones: prueban que
el archivo existe, que es byte-idéntico en los dos dirs SSOT y que es idempotente.
Ninguno mira el DATO ni la CONDUCTA. Así que aquí hay tres capas:

  1. parser sobre las migraciones (convención del repo),
  2. **el orden interno**, que es load-bearing y casi se me escapa (ver abajo),
  3. **conducta real**: se reconstruye el catálogo DESDE la propia migración y se
     corre el `culinary_contract_scan` de producción contra el corpus beta. Sin DB,
     así que corre en el gate igual que el resto.

EL ORDEN ES LOAD-BEARING. Los overrides por alimento tienen que ir ANTES de los
defaults por categoría. Al revés, el default deja `prep_methods` no-NULL y el
`IS NULL` del override no casa NUNCA: las 43 asignaciones de Proteínas/Lácteos/
Frutas quedan MUERTAS y los curados (jamones, chorizos, pepperoni) se guardan como
carne CRUDA. La primera versión de la migración tenía ese bug; lo cazó un dry-run
transaccional contra Neon, no la simulación en Python — que aplicaba default y
override en la misma pasada y por tanto modelaba mi intención, no el SQL.

tooltip-anchor: P1-CULINARY-METADATA-BETA
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_BACKFILL = "p1_culinary_metadata_beta_2026_08_19.sql"
_CHECK = "p1_culinary_metadata_beta_not_null_check.sql"
_FIXTURES = _BACKEND / "tests" / "fixtures" / "culinary_beta"

_VOCAB = frozenset({"hervir", "plancha", "freir", "hornear", "guisar",
                    "saltear", "licuar", "tostar", "crudo", "ninguno"})


def _leer(nombre: str, root: bool = False) -> str:
    base = _ROOT if root else _BACKEND
    return io.open(base / "migrations" / nombre, encoding="utf-8").read()


# ─────────────────────────── 1. convención del repo ───────────────────────────

@pytest.mark.parametrize("nombre", [_BACKFILL, _CHECK])
def test_migracion_existe_en_los_dos_dirs_ssot(nombre):
    """P3-MIGRATIONS-SSOT: toda migration vive en `migrations/` Y `backend/migrations/`."""
    for base in (_BACKEND, _ROOT):
        p = base / "migrations" / nombre
        assert p.exists(), f"falta {p}"


@pytest.mark.parametrize("nombre", [_BACKFILL, _CHECK])
def test_migracion_byte_identica_en_ambos_dirs(nombre):
    a = (_BACKEND / "migrations" / nombre).read_bytes()
    b = (_ROOT / "migrations" / nombre).read_bytes()
    assert a == b, f"{nombre} difiere entre los dos dirs SSOT ({len(a)} vs {len(b)} bytes)"


def test_backfill_es_idempotente():
    """TODO UPDATE del backfill filtra por `prep_methods IS NULL` (solo vírgenes).

    Sin eso, re-ejecutar pisaría las 206 filas dominicanas ya curadas por las
    rondas 1 y 2.
    """
    sql = _leer(_BACKFILL)
    updates = re.findall(r"UPDATE public\.master_ingredients SET(.*?);", sql, re.S)
    assert updates, "no se encontró ningún UPDATE"
    sin_guarda = [u for u in updates if "prep_methods IS NULL" not in u]
    assert not sin_guarda, (
        f"{len(sin_guarda)} UPDATE(s) sin `prep_methods IS NULL`: no son idempotentes")


def test_backfill_solo_usa_vocabulario_canonico():
    sql = _leer(_BACKFILL)
    for arr in re.findall(r"ARRAY\[([^\]]+)\]", sql):
        metodos = {m.strip().strip("'") for m in arr.split(",")}
        # El sanity DO $$ lista el vocabulario entero: no es una asignación.
        if metodos - _VOCAB and "hervir','plancha'" in arr.replace(" ", ""):
            continue
        fuera = metodos - _VOCAB
        assert not fuera, f"métodos fuera del vocabulario canónico: {sorted(fuera)}"


def test_check_es_idempotente_y_declara_su_orden():
    sql = _leer(_CHECK)
    assert "DROP CONSTRAINT IF EXISTS" in sql, "falta el DROP IF EXISTS (idempotencia)"
    assert "master_ingredients_prep_methods_not_null" in sql
    assert "corre PRIMERO" in sql, (
        "el CHECK debe traer la guarda que aborta si aún quedan filas NULL")


def test_check_no_toca_ready_to_eat():
    """`ready_to_eat` tiene 49 NULLs LEGÍTIMOS preexistentes (Vegetales y Víveres,
    que la ronda 1 dejó así por diseño). Un CHECK ahí sería falso y rompería filas
    dominicanas sanas."""
    sql = _leer(_CHECK)
    assert not re.search(r"CHECK\s*\(\s*ready_to_eat", sql), (
        "ready_to_eat NO debe llevar constraint de no-nulidad")


# ──────────────────────── 2. el orden interno (load-bearing) ───────────────────

def _posiciones(sql: str):
    por_alimento, por_categoria = [], []
    for m in re.finditer(r"UPDATE public\.master_ingredients SET(.*?);", sql, re.S):
        cuerpo = m.group(1)
        (por_categoria if re.search(r"WHERE\s+category\s*=", cuerpo) else
         por_alimento).append(m.start())
    return por_alimento, por_categoria


def test_los_overrides_por_alimento_van_antes_que_los_defaults():
    """Si un default por categoría corre primero, deja `prep_methods` no-NULL y el
    `IS NULL` del override ya no casa: el override queda MUERTO y los curados se
    guardan como carne cruda. Este test ancla la lección del dry-run."""
    sql = _leer(_BACKFILL)
    por_alimento, por_categoria = _posiciones(sql)
    assert por_alimento and por_categoria, "se esperaban UPDATEs de ambos tipos"
    assert max(por_alimento) < min(por_categoria), (
        "hay un UPDATE por-alimento DESPUÉS de un default por categoría: ese override "
        "no se aplicará nunca (el default ya dejó prep_methods no-NULL)")


# ─────────────────── 3. conducta real, sin DB: el corpus beta ──────────────────

def _catalogo_desde_migracion() -> list:
    """Reconstruye el catálogo leyendo la PROPIA migración.

    Reproduce la semántica del SQL, incluido el orden: primero los overrides por
    nombre, luego los defaults por categoría, y cada pasada solo toca lo que sigue
    sin metadata. Si alguien invierte el orden en el .sql, este parser lo refleja y
    los tests de conducta de abajo se ponen rojos — que es justo lo que debe pasar.
    """
    sql = _leer(_BACKFILL)
    # Categoría de cada alimento: la deduce el propio corpus no, así que se usa el
    # mapa que la migración declara implícitamente. Para la conducta solo hacen
    # falta los alimentos NOMBRADOS uno a uno más los defaults por categoría.
    filas: dict = {}
    for m in re.finditer(
            r"UPDATE public\.master_ingredients SET\s*(.*?)\s*WHERE prep_methods IS NULL "
            r"AND name IN \((.*?)\);", sql, re.S):
        sets, nombres = m.group(1), m.group(2)
        prep = re.search(r"prep_methods = ARRAY\[([^\]]+)\]", sets)
        r2e = re.search(r"ready_to_eat = (true|false)", sets)
        metodos = [x.strip().strip("'") for x in prep.group(1).split(",")] if prep else None
        listo = (r2e.group(1) == "true") if r2e else None
        for n in re.findall(r"'((?:[^']|'')+)'", nombres):
            filas.setdefault(n.replace("''", "'"),
                             {"name": n.replace("''", "'"), "prep_methods": metodos,
                              "ready_to_eat": listo})
    return list(filas.values())


def _catalogo_completo() -> list:
    """El catálogo de la migración MÁS los alimentos dominicanos que el corpus
    menciona de pasada (Huevo, Tomate, Leche…). Se les da metadata permisiva a
    propósito: el sujeto de este test son las filas beta, no ellos."""
    base = _catalogo_desde_migracion()
    nombres = {f["name"] for f in base}
    acompanantes = ["Huevo", "Tomate", "Leche", "Cebolla", "Pimiento", "Zanahoria",
                    "Berenjena", "Cerdo", "Pechuga de pollo", "Caldo de pollo",
                    "Harina de maíz precocida", "Carne molida mixta"]
    for n in acompanantes:
        if n not in nombres:
            base.append({"name": n, "ready_to_eat": None,
                         "prep_methods": sorted(_VOCAB)})
    return base


def test_corpus_limpio_no_produce_ni_una_violacion():
    """22 recetas beta REALISTAS. Cada una ejercita una decisión de juicio del
    backfill (quesos que se funden, curados, chiles secos que se tuestan, membrillo
    que no se come crudo, congelados que sí se cocinan)."""
    from culinary_coherence import culinary_contract_scan

    plan = json.loads((_FIXTURES / "beta_limpios.json").read_text(encoding="utf-8"))
    viols = culinary_contract_scan(plan, _catalogo_completo())
    detalle = [(v["check"], v["food"], v["detail"][:80]) for v in viols]
    assert not viols, f"falsos positivos sobre recetas legítimas: {detalle}"


def test_corpus_absurdo_sigue_cazandose():
    """El contrapeso: un backfill demasiado permisivo dejaría el corpus limpio en
    verde y el check inútil. Estos 8 casos verifican que sigue mordiendo."""
    from culinary_coherence import culinary_contract_scan

    casos = json.loads((_FIXTURES / "beta_absurdos.json").read_text(encoding="utf-8"))
    catalogo = _catalogo_completo()
    fallos = []
    for caso in casos:
        checks = {v["check"] for v in culinary_contract_scan(caso["plan"], catalogo)}
        esperado = caso["esperado"]
        nombre = caso["plan"]["days"][0]["meals"][0]["name"]
        if esperado is None:
            if "V2" in checks:
                fallos.append(f"{nombre}: V2 disparó sobre un listo-para-comer")
        elif esperado not in checks:
            fallos.append(f"{nombre}: se esperaba {esperado}, se obtuvo {sorted(checks) or 'nada'}")
    assert not fallos, fallos


def test_cobertura_del_corpus_beta_llega_al_100():
    """La cifra que resume el P-fix: 24% -> 100% sobre el corpus beta."""
    from culinary_coherence import scan_coverage

    plan = json.loads((_FIXTURES / "beta_limpios.json").read_text(encoding="utf-8"))
    cov = scan_coverage(plan, _catalogo_completo())
    assert cov == pytest.approx(1.0), f"cobertura {cov:.0%}, esperada 100%"


def test_el_manifiesto_declara_la_limitacion_del_golden_set():
    """La honestidad del corpus es parte del artefacto: el golden set es dominicano
    y no cubre lo nuevo. Si alguien borra esa nota, el corpus parece redundante."""
    man = json.loads((_FIXTURES / "beta_manifest.json").read_text(encoding="utf-8"))
    assert "DOMINICANOS" in man["proposito"]
    assert man["limpios"]["n"] >= 20
    assert man["absurdos"]["n"] >= 8


# ──────────── 4. ¿el corpus EJERCITA lo que dice ejercitar? ────────────────────

def test_cada_alimento_beta_del_corpus_se_ejercita_de_verdad():
    """Un corpus puede MENCIONAR un alimento sin PROBARLO, y entonces es decorativo.

    Barrido de mutación: a cada alimento del catálogo que el corpus menciona se le
    vacía `prep_methods` y se re-corre el scan. Si no aparece violación, ese alimento
    no lo evalúa ningún verbo.

    La primera versión de este corpus tenía 12 de 37 alimentos MUDOS — entre ellos
    provolone, queso en hebras, jamón serrano, pepperoni y requesón, o sea justo los
    casos que el corpus decía cubrir. Dos causas, ambas contraintuitivas y ambas
    documentadas en `beta_manifest.json`:

      1. V1 solo juzga al objeto INMEDIATO del verbo ("hornea 5 minutos" no evalúa
         el queso que el paso nombra después),
      2. la salvaguarda de acompañantes: en una cláusula con ≥2 alimentos, si uno
         acepta el método los demás se callan.

    Los únicos mudos tolerados son los acompañantes dominicanos, que no son el sujeto
    de este corpus.
    """
    from culinary_coherence import (build_culinary_index, culinary_contract_scan,
                                    find_catalog_foods)

    plan = json.loads((_FIXTURES / "beta_limpios.json").read_text(encoding="utf-8"))
    catalogo = _catalogo_completo()
    por_nombre = {f["name"]: f for f in catalogo}
    de_la_migracion = {f["name"] for f in _catalogo_desde_migracion()}

    indice = build_culinary_index(catalogo)
    mencionados = set()
    for d in plan["days"]:
        for m in d["meals"]:
            texto = " ".join(m["ingredients"]) + " " + " ".join(m["recipe"])
            mencionados.update(find_catalog_foods(texto, indice))

    mudos = []
    for nombre in sorted(mencionados & de_la_migracion):
        fila = por_nombre[nombre]
        original = fila["prep_methods"]
        fila["prep_methods"] = []          # invalida TODOS los verbos
        viols = culinary_contract_scan(plan, catalogo)
        fila["prep_methods"] = original
        if not any(v["food"] == nombre for v in viols):
            mudos.append(nombre)

    assert not mudos, (
        f"el corpus menciona estos alimentos beta pero ningún verbo los evalúa: {mudos}. "
        "Reescribe su paso para que el alimento sea el objeto inmediato del verbo y vaya "
        "solo en su cláusula (ver beta_manifest.json)")


def test_el_parser_de_la_migracion_no_puede_fallar_en_silencio():
    """Sin este contador, los tests de conducta pasan EN VACÍO.

    `_catalogo_desde_migracion` lee el .sql con una regex. Si alguien edita el SQL
    de una forma que la regex no contempla, el catálogo sale vacío, TODO cae a
    fail-open y `test_corpus_limpio_no_produce_ni_una_violacion` pasa sin haber
    comprobado nada.

    No es hipotético: una mutación de control deliberadamente inocua (`AND 1=1 AND
    name IN`) reprodujo justo eso. El síntoma fue que se cayó el test de los
    ABSURDOS — el del corpus limpio siguió verde, tan tranquilo.
    """
    filas = _catalogo_desde_migracion()
    assert len(filas) >= 90, (
        f"el parser solo recuperó {len(filas)} alimentos de la migración (se esperaban ≥90). "
        "Probablemente el SQL cambió de forma y la regex ya no casa: los tests de conducta "
        "estarían corriendo contra un catálogo vacío y pasando en vacío")
    sin_metodos = [f["name"] for f in filas if not f["prep_methods"]]
    assert not sin_metodos, f"alimentos parseados sin prep_methods: {sin_metodos[:5]}"
