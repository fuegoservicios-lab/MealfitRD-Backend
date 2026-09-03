"""[P1-YOGURT-NATURAL · 2026-08-19] La fila `Yogurt` cargaba el perfil del griego.

Diagnóstico en Neon prod (2026-08-19): `Yogurt` (alias literal `yogurt regular`) y
`Yogurt griego sin azúcar` compartían `fdc_id` 330137 y tenían valores BYTE-IDÉNTICOS
— 59.1 kcal / 10.3 g proteína / 0.37 g grasa. Ese es el perfil del griego 0%. Un yogur
natural entero ronda 3.5 g de proteína: la fila sobreestimaba proteína ~3x, en tráfico
dominicano real y en la columna que más pesa en el `portion_solver`.

CORRECCIÓN del mismo día: la primera versión de este test afirmaba que el `fdc_id`
viejo (330137) estaba MUERTO porque `/fdc/v1/food/330137` devuelve 404. El 404 es real,
la conclusión era falsa — 330137 es de tipo `Foundation` y el endpoint de detalle no
sirve ese tipo; el buscador sí lo conoce. Un barrido de los 288 ids del catálogo dio
**cero muertos**. *Un 404 dice que tu petición falló, no que la cosa no exista.*

Parser-based a propósito (no toca DB): el gate deselecciona los tests `e2e`, así que un
test contra Neon no correría y sería un guard incapaz de fallar.

tooltip-anchor: P1-YOGURT-NATURAL
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIG = "p1_yogurt_natural_no_es_griego_2026_08_19.sql"


def _sql(root: bool = False) -> str:
    base = _ROOT if root else _BACKEND
    return io.open(base / "migrations" / _MIG, encoding="utf-8").read()


def test_migracion_en_los_dos_dirs_ssot_y_byte_identica():
    a = (_BACKEND / "migrations" / _MIG).read_bytes()
    b = (_ROOT / "migrations" / _MIG).read_bytes()
    assert a and a == b, "la migración debe existir e ser idéntica en ambos dirs SSOT"


def test_la_proteina_baja_del_rango_griego():
    """3.47 g es yogur natural entero (USDA fdc 171284). >5 g sería seguir en griego."""
    m = re.search(r"protein_g_per_100g\s*=\s*([\d.]+)", _sql())
    assert m, "la migración no fija protein_g_per_100g"
    assert float(m.group(1)) < 5.0, (
        f"proteína {m.group(1)} g sigue en rango griego; el objetivo del fix era ~3.5")


def test_apunta_al_fdc_vivo_y_no_al_muerto():
    sql = _sql()
    assert "171284" in sql, "falta el fdc_id nuevo (Yogurt, plain, whole milk)"
    assert re.search(r"fdc_id\s*=\s*171284", sql)
    # 330137 puede citarse en la prosa (es el diagnóstico), pero NO asignarse.
    assert not re.search(r"fdc_id\s*=\s*330137", sql), (
        "el fdc_id viejo apuntaba al GRIEGO: no puede volver a asignarse a la fila natural")


def test_documenta_la_correccion_sobre_el_404():
    """La migración debe conservar la CORRECCIÓN, no la afirmación original.

    Si alguien borra esta nota, el siguiente lector puede volver a concluir que un 404
    del endpoint de detalle significa «el id no existe» — y ese razonamiento ya produjo
    una afirmación falsa en esta misma migración."""
    sql = _sql()
    assert "CORRECCION" in sql or "CORRECCIÓN" in sql, (
        "la migración debe conservar la corrección sobre el 404")
    assert "Foundation" in sql, (
        "debe quedar escrito POR QUÉ el detalle devolvía 404: es un registro Foundation")


def test_corrige_los_azucares_imposibles():
    """Ambas filas tenían `sugars_g = 0`, imposible en un yogur (lactosa)."""
    m = re.search(r"sugars_g_per_100g\s*=\s*([\d.]+)", _sql())
    assert m and float(m.group(1)) > 0, "los azúcares deben dejar de ser 0"


def test_no_toca_la_fila_del_griego():
    """Los valores del griego son un perfil 0% correcto y conserva legítimamente su
    fdc 330137, que —corregida la sonda— es «Yogurt, Greek, plain, nonfat»."""
    sql = _sql()
    updates = re.findall(r"UPDATE public\.master_ingredients SET(.*?);", sql, re.S)
    for u in updates:
        assert "slug = 'yogurt'" in u, f"UPDATE que no filtra por slug='yogurt': {u[:120]}"
        assert "yogurt-griego" not in u.split("WHERE")[-1], "no debe escribir sobre el griego"


def test_declara_el_residual_de_omega3():
    """SR Legacy no reporta el nutriente 851 para fdc 171284, así que omega3 conserva
    el valor heredado del proxy. Deuda declarada, no descuido: inventar el número
    sería peor."""
    sql = _sql()
    assert "omega3" in sql.lower() and "RESIDUAL" in sql, (
        "la migración debe declarar explícitamente el residual de omega3")
    assert not re.search(r"omega3_ala_g_per_100g\s*=", sql), (
        "omega3 NO debe escribirse: no hay fuente para él")


def test_trae_sanity_que_verifica_el_resultado():
    sql = _sql()
    assert sql.count("DO $$") >= 2, "se esperaban al menos 2 bloques de sanity"
    assert "RAISE EXCEPTION" in sql
    assert "siguen con macros identicos" in sql or "identicos" in sql, (
        "debe verificar que las dos filas dejaron de ser gemelas")
