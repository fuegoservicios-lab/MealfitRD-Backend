"""[P1-BEDCA-DEPROXY-ES · 2026-08-19] Los embutidos y curados españoles dejan de vivir
sobre una fila de USDA que no es la suya.

`fdc 173859` (USDA *Sausage, pork, chorizo, raw*, 296 kcal) hacía de sustituto de SIETE
embutidos a la vez — Sobrasada incluida, que en realidad ronda 595 kcal. La fuente nueva
es BEDCA (AESAN/MICINN).

DOS TRAMPAS QUE ESTE TEST ANCLA:

  1. **BEDCA publica la energía en kJ, no en kcal.** Sin dividir entre 4.184 todos los
     valores entrarían ~4.2x inflados. La comprobación independiente es Atwater: los 11
     alimentos cruzan con <2% de divergencia, que es lo que confirma que la conversión
     y el mapeo de componentes son correctos.
  2. **BEDCA no reporta azúcares, vitamina A (RAE), vitamina D ni K.** Esas cuatro
     columnas conservan el valor heredado del proxy de USDA. Deuda declarada — y
     `vitamin_a_mcg_rae` / `vitamin_k_mcg` son además NOT NULL, así que sobrescribirlas
     exigiría inventarlas.

tooltip-anchor: P1-BEDCA-DEPROXY-ES
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIG = "p1_bedca_deproxy_es_2026_08_19.sql"

_EMBUTIDOS = ("Chistorra", "Chorizo español", "Chorizo mexicano", "Chorizo santarrosano",
              "Chorizo verde", "Longaniza puertorriqueña", "Sobrasada")


def _sql(root: bool = False) -> str:
    base = _ROOT if root else _BACKEND
    return io.open(base / "migrations" / _MIG, encoding="utf-8").read()


def test_migracion_en_los_dos_dirs_ssot_y_byte_identica():
    a = (_BACKEND / "migrations" / _MIG).read_bytes()
    b = (_ROOT / "migrations" / _MIG).read_bytes()
    assert a and a == b


def test_extiende_el_enum_de_procedencia_antes_de_usarlo():
    """El CHECK vigente solo permitía usda|off|faoinfoods|manual. Sin extenderlo
    PRIMERO, todos los UPDATE de abajo fallan."""
    sql = _sql()
    i_check = sql.index("master_ingredients_nutrition_source_check")
    i_primer_update = sql.index("UPDATE public.master_ingredients SET")
    assert i_check < i_primer_update, (
        "el ALTER del enum debe ir ANTES del primer UPDATE que escribe 'bedca'")
    assert "'bedca'::text" in sql


def test_cada_fila_declara_su_referencia_de_procedencia():
    """`fdc_id` solo sabe hablar de USDA. Sin `nutrition_source_ref` estas filas
    nacerían imposibles de re-verificar — repitiendo a sabiendas el problema del
    fdc 330137 (compartido y además HTTP 404)."""
    sql = _sql()
    assert "nutrition_source_ref" in sql
    assert "ADD COLUMN IF NOT EXISTS nutrition_source_ref" in sql, "debe ser idempotente"
    refs = re.findall(r"nutrition_source_ref\s*=\s*'bedca:(\d+)'", sql)
    assert len(refs) >= 11, f"solo {len(refs)} filas traen referencia BEDCA"
    assert len(set(refs)) == len(refs), (
        f"dos alimentos apuntan al MISMO f_id de BEDCA: {refs}. Es exactamente el "
        "error que este P-fix corrige — no se puede reintroducir con otra fuente")


def test_ninguna_fila_bedca_conserva_su_fdc_id():
    """Dejar el fdc_id de USDA en una fila cuyos valores ya son de BEDCA es
    procedencia que miente."""
    for bloque in re.findall(r"UPDATE public\.master_ingredients SET(.*?);", _sql(), re.S):
        if "nutrition_source            = 'bedca'" in bloque:
            assert re.search(r"fdc_id\s*=\s*NULL", bloque), (
                f"fila bedca que no limpia fdc_id: {bloque[:120]}")


def test_documenta_la_conversion_de_kilojulios():
    """Es LA trampa de esta fuente. Si alguien reescribe el extractor sin saberlo,
    mete valores 4.2x inflados."""
    sql = _sql()
    assert "kJ" in sql and "4.184" in sql, (
        "la migración debe documentar que BEDCA publica energía en kJ y el divisor")


def test_declara_las_cuatro_columnas_que_bedca_no_cubre():
    """La deuda tiene que estar por escrito Y ser real: las cuatro columnas se
    NOMBRAN en la prosa y NINGUNA se escribe en un SET."""
    sql = _sql()
    for col in ("azucares", "vitamina A", "vitamina D", "vitamina K"):
        assert col.lower() in sql.lower(), f"no declara el hueco de {col}"
    no_escribibles = ("sugars_g_per_100g", "vitamin_a_mcg_rae_per_100g",
                      "vitamin_d_mcg_per_100g", "vitamin_k_mcg_per_100g")
    for col in no_escribibles:
        assert not re.search(rf"^\s*{col}\s*=", sql, re.M), (
            f"{col} NO tiene fuente en BEDCA: escribirla sería inventarla")


def test_el_cluster_de_embutidos_deja_de_ser_un_bloque_identico():
    """Sanity DIRECTO del efecto: antes las 7 filas tenían las MISMAS kcal."""
    sql = _sql()
    for nombre in _EMBUTIDOS:
        assert nombre in sql, f"el sanity del cluster no menciona {nombre}"
    assert "COUNT(DISTINCT kcal_per_100g)" in sql


def test_trae_sanity_de_atwater():
    """Es la verificación INDEPENDIENTE de que la conversión kJ→kcal es correcta:
    si el divisor estuviera mal, 4P+4C+9G no cuadraría con las kcal."""
    sql = _sql()
    assert "4*protein_g_per_100g" in sql and "9*fats_g_per_100g" in sql
    assert "0.12" in sql, "el umbral de Atwater debe ser el mismo 12% de los scripts de alta"


def test_lomo_embuchado_documenta_por_que_entra():
    """No estaba en ningún grupo de fdc_id compartido: tenía su PROPIO fdc, apuntando
    a lomo de cerdo crudo (110 kcal) en vez de al curado (321). Es la prueba de que la
    clase de error es más amplia que los ids compartidos, y la auditoría por
    duplicados NO la ve."""
    sql = _sql()
    assert "Lomo embuchado" in sql
    assert "CRUDO" in sql.upper() and "compartido" in sql, (
        "debe explicar que un fdc_id único también puede apuntar al alimento equivocado")
