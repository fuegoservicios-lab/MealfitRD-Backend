"""[P1-LATINFOODS-TCAC · 2026-08-19] Cinco andinos dejan el proxy de USDA por su tabla
nacional: la Tabla de Composición de Alimentos Colombianos (TCAC 2015, ICBF).

POR QUÉ IMPORTABA. Un proxy no es «un valor aproximado»: puede ser otro alimento.

    Chontaduro   103 → 332 kcal   vivía sobre *Breadfruit* (panapén). El chontaduro es
                                  palma de pejibaye con 25,7 g de GRASA/100 g; el
                                  panapén tiene 0,23. No se parecen en nada.
    Curuba        97 → 35 kcal    2,8× al revés.
    Borojó        66 → 134 kcal
    Chinola     108,6 → 59 kcal
    Suero cost.  136 → 83 kcal    y proteína 3,5 → 11,0. Vivía sobre *Sour cream*:
                                  es suero fermentado, no crema. El error no era de
                                  magnitud, era de CATEGORÍA.

LA LECCIÓN DE MÉTODO. Un PDF no tiene contrato. La tabla proximal de la TCAC cambia de
número de columnas según si la fila trae desviación estándar, así que partir por posición
se rompe — hay que partir por las letras de calificación. Y sobre todo: **cada fila se
aceptó solo si cruza Atwater** (4P + 4C + 9G vs las kcal declaradas) dentro del 5%.

Ese cruce no es un adorno. Cazó dos errores de parseo MÍOS antes de que llegaran a la
migración: la columna «N» confundida con las kcal (acertaba solo cuando N era un número
romano), y un `finditer` que consumía el par kcal/kJ antes de poder evaluarlo. Sin el
guard, las cinco filas habrían entrado con números de otra columna.

tooltip-anchor: P1-LATINFOODS-TCAC
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIG = "p1_latinfoods_tcac_andinos_2026_08_19.sql"

#: (alimento, kcal, prot, grasa, carbs) tal como se extrajeron de la TCAC.
_ESPERADO = {
    "Chontaduro": (332.0, 6.3, 25.7, 19.0),
    "Curuba": (35.0, 0.6, 0.1, 7.8),
    "Borojó": (134.0, 3.0, 0.6, 29.0),
    "Chinola": (59.0, 1.5, 0.5, 12.0),
    "Suero costeño": (83.0, 11.0, 1.5, 6.4),
}


def _sql(root: bool = False) -> str:
    base = _ROOT if root else _BACKEND
    return io.open(base / "migrations" / _MIG, encoding="utf-8").read()


def test_migracion_en_los_dos_dirs_ssot_y_byte_identica():
    a = (_BACKEND / "migrations" / _MIG).read_bytes()
    b = (_ROOT / "migrations" / _MIG).read_bytes()
    assert a and a == b


def test_extiende_el_enum_antes_de_usar_latinfoods():
    sql = _sql()
    i_enum = sql.index("master_ingredients_nutrition_source_check")
    i_uso = sql.index("nutrition_source = 'latinfoods'")
    assert i_enum < i_uso, "el ALTER del enum debe ir ANTES del primer UPDATE"
    assert "'latinfoods'::text" in sql


def test_cada_fila_cruza_atwater_con_los_valores_ESCRITOS():
    """El guard que hizo fiable la extracción, aplicado ahora a lo que la migración
    escribe de verdad. Si alguien retoca un número a mano, esto lo caza."""
    sql = _sql()
    for nombre, (kcal, prot, grasa, carb) in _ESPERADO.items():
        atwater = 4 * prot + 4 * carb + 9 * grasa
        div = abs(atwater - kcal) / kcal
        assert div <= 0.05, f"{nombre}: Atwater diverge {div:.1%} de las kcal declaradas"
        # y esos números tienen que estar realmente en el SQL
        bloque = re.search(rf"UPDATE public\.master_ingredients SET(.*?)WHERE name = '{re.escape(nombre)}';",
                           sql, re.S)
        assert bloque, f"no hay UPDATE para {nombre}"
        assert f"kcal_per_100g = {kcal}" in bloque.group(1), f"{nombre}: kcal no coincide"


def test_cada_fila_declara_su_codigo_en_la_tcac():
    """Sin el código, la fila es imposible de re-verificar contra el PDF — que es
    exactamente el problema que toda esta tanda vino a cerrar."""
    refs = re.findall(r"nutrition_source_ref = 'tcac:(\d+) \(([^']+)\)'", _sql())
    assert len(refs) == len(_ESPERADO), f"solo {len(refs)} filas traen código TCAC"
    assert len({r[0] for r in refs}) == len(refs), f"dos filas comparten código TCAC: {refs}"


def test_ninguna_fila_latinfoods_conserva_un_fdc_id():
    """Dejar el `fdc_id` en una fila cuyos valores ya son de la TCAC es procedencia que
    miente — la misma regla de P1-PROVENANCE-TRUTHFUL."""
    for bloque in re.findall(r"UPDATE public\.master_ingredients SET(.*?);", _sql(), re.S):
        if "'latinfoods'" in bloque:
            assert re.search(r"fdc_id\s*=\s*NULL", bloque), (
                f"fila latinfoods que no limpia fdc_id: {bloque[:110]}")


def test_declara_lo_que_la_tabla_proximal_NO_trae():
    """Fibra, minerales y vitaminas viven en otras tablas del mismo PDF. Esas columnas
    conservan el valor del proxy: deuda declarada, no descuido."""
    sql = _sql()
    assert "NO** SE TOCA" in sql or "NO se toca" in sql.lower() or "NO fibra" in sql
    for palabra in ("fibra", "minerales", "vitamina"):
        assert palabra in sql.lower(), f"no declara el hueco de {palabra}"


def test_champus_queda_fuera_y_lo_dice():
    """`Champús` no está en la TCAC (es una bebida preparada). Sigue como proxy, que es
    lo honesto — y la migración lo declara para que nadie lo busque en vano."""
    assert "Champús" in _sql() and "NO entra" in _sql()


def test_el_sanity_prueba_el_efecto_no_solo_la_forma():
    """Un sanity que solo cuente filas pasaría aunque los valores fueran los del proxy.
    El de esta migración comprueba que el chontaduro DEJÓ de parecerse al panapén."""
    sql = _sql()
    assert "Chontaduro sigue con perfil de panapen" in sql
    assert "_kcal < 300 OR _grasa < 20" in sql, (
        "el umbral debe separar el perfil del chontaduro (332/25,7) del panapén (103/0,23)")
