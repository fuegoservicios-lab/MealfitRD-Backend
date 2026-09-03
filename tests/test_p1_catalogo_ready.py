"""[P1-CATALOGO-READY · 2026-08-19] El catálogo, listo para producción en los 6 países.

Ancla las cuatro migraciones que cerraron las dimensiones que le faltaban —coherencia
interna, densidad, alias y micros— y, sobre todo, **las decisiones de método**, que es lo
que un test puede proteger y una cifra no.

LO QUE ESTA TANDA ENSEÑÓ SOBRE MEDIR

1. `Atwater` NO es una métrica de calidad para este catálogo. Marcaba 14 filas con la
   fórmula 4/4/9 y 32 con carbohidratos disponibles: ninguna tenía razón. USDA usa
   factores específicos por alimento, el ron son 231 kcal de alcohol que P/C/G no ven, y
   en un bok choy de 13 kcal un porcentaje no significa nada. Atwater sirve como guard de
   PARSEO contra una fuente única (así validó BEDCA y la TCAC), no para auditar datos ya
   publicados. Se sustituyó por **imposibilidades aritméticas**, que no dependen de la
   fuente.

2. El gap de densidad no eran 139 filas: eran **3**. De los 73 alimentos que los planes
   reales miden por volumen, 70 ya tenían densidad. Contar NULLs mide el esquema; contar
   contra el uso real mide el problema.

3. «Ningún alimento tiene 0 fósforo» era **falso**, y lo refutó la propia USDA: los
   aceites, la sal y el vinagre reportan 0,0 mg. La regla correcta distingue alimento
   entero de aislado refinado.

tooltip-anchor: P1-CATALOGO-READY
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIGS = ["p1_catalogo_coherencia_interna_2026_08_19.sql",
         "p1_catalogo_densidad_y_alias_2026_08_19.sql",
         "p1_catalogo_micros_completos_2026_08_19.sql",
         "p1_catalogo_micros_ronda2.sql",
         "p1_catalogo_sinonimos_2026_08_19.sql",
         "p1_catalogo_sinonimos_fix_refs.sql"]


def _sql(nombre: str) -> str:
    return io.open(_BACKEND / "migrations" / nombre, encoding="utf-8").read()


@pytest.mark.parametrize("nombre", _MIGS)
def test_migracion_en_los_dos_dirs_ssot_y_byte_identica(nombre):
    a = (_BACKEND / "migrations" / nombre).read_bytes()
    b = (_ROOT / "migrations" / nombre).read_bytes()
    assert a and a == b, f"{nombre} difiere entre los dos dirs SSOT"


# ───────────────── coherencia interna: las 4 constraints ──────────────────────

def test_las_cuatro_imposibilidades_quedan_ancladas_en_DB():
    """Un CHECK, no un test: la invariante vive donde vive el dato. Un de-proxy futuro
    que deje un sub-componente huérfano falla en el UPDATE, no meses después dentro de
    un guard clínico."""
    sql = _sql("p1_catalogo_coherencia_interna_2026_08_19.sql")
    for c in ("fibra_no_supera_carbos", "azucares_no_superan_carbos",
              "saturada_no_supera_grasa", "macros_no_superan_100g"):
        assert f"master_ingredients_{c}" in sql, f"falta la constraint {c}"
    assert sql.count("DROP CONSTRAINT IF EXISTS") >= 4, "deben ser idempotentes"


def test_documenta_el_mecanismo_que_creo_las_imposibilidades():
    """Las 4 filas imposibles no vinieron de fuera: las creó el de-proxy de esta misma
    tanda, al dejar sub-componentes de un alimento junto a totales de otro. Si esa
    explicación se borra, el siguiente de-proxy repite el patrón."""
    sql = _sql("p1_catalogo_coherencia_interna_2026_08_19.sql")
    assert "Suero costeno" in sql and "6,6" in sql and "1,5" in sql, (
        "debe conservar el caso que mejor lo ilustra: saturada > grasa total")
    assert "sub-componente" in sql.lower() or "SUB-COMPONENTES" in sql


# ───────────────── densidad: la lección de medir contra el uso ────────────────

def test_la_densidad_documenta_que_el_gap_eran_3_y_no_139():
    """Si esta nota se pierde, alguien vuelve a ver «136 filas sin densidad» y se pone a
    rellenarlas. La cifra que importa no es cuántos NULL hay, es cuántos estorban."""
    sql = _sql("p1_catalogo_densidad_y_alias_2026_08_19.sql")
    assert "139" in sql and "12,2%" in sql, "debe conservar la medición contra planes reales"
    assert "70 de esos 73" in sql or "70 de los 73" in sql


def test_las_tres_densidades_vienen_de_foodPortions_no_de_estimacion():
    sql = _sql("p1_catalogo_densidad_y_alias_2026_08_19.sql")
    assert "foodPortions" in sql
    for v in ("36.0", "275.0", "240.0"):
        assert v in sql, f"falta la densidad {v}"


def test_china_no_puede_entrar_como_alias():
    """Guard explícito de un descarte: «china» es el nombre dominicano de la naranja y
    sería útil, pero colisiona con «col china» (Bok choy). Cinco letras ambiguas son la
    clase de alias que ya costó dos incidentes aquí (`sal`⊂`salsa`, `pollo`⊂`repollo`)."""
    sql = _sql("p1_catalogo_densidad_y_alias_2026_08_19.sql")
    assert "col china" in sql and "Bok choy" in sql
    assert "'china' = ANY(aliases)" in sql, "debe haber un sanity que lo impida"


# ───────────────── micros: la premisa que la DB me corrigió ───────────────────

def test_distingue_colesterol_de_fosforo():
    """Los dos NULL parecen el mismo problema y no lo son: el colesterol es un esterol
    ANIMAL (0 en vegetales es un hecho, no una estimación) y el fósforo está en todo
    alimento entero (un 0 sería falso)."""
    sql = _sql("p1_catalogo_micros_completos_2026_08_19.sql")
    assert "esterol ANIMAL" in sql
    assert "FOSFORO esta en todo alimento ENTERO" in sql


def test_conserva_la_correccion_sobre_el_fosforo_cero():
    """La versión fuerte del guard («ningún alimento tiene 0 fósforo») era falsa y la
    refutó USDA con los aceites, la sal y el vinagre. Si alguien borra esta nota, vuelve
    a escribir el guard imposible."""
    sql = _sql("p1_catalogo_micros_completos_2026_08_19.sql")
    assert "MATIZ QUE ME CORRIGIO LA PROPIA USDA" in sql
    assert "aceites" in sql and "0,0 mg" in sql
    # y el sanity vigente debe estar acotado a alimentos enteros
    assert "category IN ('Proteínas', 'Lácteos', 'Frutas', 'Vegetales', 'Víveres')" in sql


def test_los_cinco_fosforos_que_faltan_estan_justificados_no_rellenados():
    """La TCAC SÍ tiene tabla de minerales con Borojó y Chontaduro. No se usó porque el
    número de columnas varía por fila y no hay un validador equivalente a Atwater: el
    mapeo ingenuo daba 1,5 mg en el borojó y 359 en el chontaduro, ambos implausibles.
    Cinco huecos declarados valen más que cinco números inventados."""
    sql = _sql("p1_catalogo_micros_ronda2.sql")
    assert "no hay un validador" in sql or "NO hay un validador" in sql
    assert "implausibles" in sql
    assert "1,5 mg" in sql and "359" in sql, "debe citar los valores que delatan el mapeo malo"


# ───────────────── sinónimos: por qué NO se borran filas ──────────────────────

def test_los_sinonimos_se_sincronizan_pero_no_se_borran():
    sql = _sql("p1_catalogo_sinonimos_2026_08_19.sql")
    assert not re.search(r"\bDELETE\s+FROM\b", sql, re.I)
    assert "CERO referencias" in sql and "nombres ESPANOLES" in sql, (
        "debe explicar que cero referencias en un país que no ha arrancado no es «no se usa»")


def test_la_fusion_preserva_informacion_en_ambas_direcciones():
    """Un «gana el canónico» a secas habría borrado la densidad de Habichuelas blancas,
    que solo tenía el sinónimo."""
    sql = _sql("p1_catalogo_sinonimos_2026_08_19.sql")
    assert "el canonico no lo tiene" in sql
    assert "densidad" in sql.lower()


def test_ninguna_fila_es_sinonimo_de_si_misma():
    sql = _sql("p1_catalogo_sinonimos_fix_refs.sql")
    assert "sinonimo de si misma" in sql
    assert "'sinonimo:' || name" in sql, "el sanity debe detectar la referencia circular"


def test_conserva_la_leccion_de_orden_entre_migraciones():
    """«Idempotente» no es «seguro de re-ejecutar en cualquier orden»: re-correr una
    migración vieja después de una posterior revierte a la posterior. Pasó dos veces
    en esta tanda."""
    sql = _sql("p1_catalogo_sinonimos_fix_refs.sql")
    assert "ORDEN, NO SOLO IDEMPOTENCIA" in sql
    assert "en cualquier orden" in sql
