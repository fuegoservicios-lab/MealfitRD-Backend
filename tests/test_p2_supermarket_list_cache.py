"""[P2-BACKEND-SUPERMARKET-CACHE · 2026-08-14 · parte a] El listado público del
catálogo no tenía caché en NINGUNA capa.

MEDIDO CONTRA PRODUCCIÓN (2026-08-14): `total = 1739`. La página 1 son 518.884 B
crudos / 75.607 gzip y la 2 otros 386.423 / 57.356 — o sea **905 KB crudos y
133 KB gzip en dos peticiones serializadas por visita**, para pintar 48 tarjetas
(`PAGE_SIZE = 48`). Y `_fetch` ejecuta TRES consultas por petición (el SELECT, un
`count(*)` y un `GROUP BY category`), así que son **seis por visita**, con el
count y el group-by devolviendo siempre lo mismo.

Lo llamativo es que la caché ya existía: `_CATALOG_CACHE`, que introdujo
P1-SUPERMARKET-CATALOG-CACHE. Pero sólo la consultaba `/match`. El listado —la
única API de datos del landing, y la que un visitante anónimo dispara al abrir
`/supermercado`— nunca la miró.

⚠️ CORRECCIÓN AL PLAN ORIGINAL, que proponía recortar del payload `notes`,
`description`, `created_at` y `updated_at`: **`notes` y `description` SÍ los usa
el frontend** (los pinta el formulario de edición), así que quitarlos rompería la
página. Verificado con grep sobre `SupermarketPage.jsx`. Sólo `created_at` y
`updated_at` no tienen un solo consumidor — y son los dos únicos que se recortan.
Tampoco se tocan `category`, `brand` ni `image_url`: la página deriva de ellos
las facetas del filtro.

Tooltip-anchor: P2-SUPERMARKET-LIST-CACHE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROUTER = Path(__file__).resolve().parent.parent / "routers" / "supermarket.py"
_FRONT = (
    Path(__file__).resolve().parent.parent.parent
    / "frontend" / "src" / "pages" / "SupermarketPage.jsx"
)


@pytest.fixture()
def sm():
    import routers.supermarket as modulo
    modulo._CATALOG_CACHE.update({"at": 0.0, "rows": None, "master": None, "gen": 0})
    return modulo


# ---------------------------------------------------------------------------
# 1. El caso común se sirve de caché
# ---------------------------------------------------------------------------

def test_el_listado_publico_consulta_la_cache(sm):
    """`/match` la usaba desde agosto; el listado, que es el que ve el visitante, no."""
    assert hasattr(sm, "_cached_active_rows"), (
        "[P2-SUPERMARKET-LIST-CACHE] Falta el helper que sirve el listado de caché. "
        "El catálogo cambia sólo cuando un admin lo edita, y esas rutas ya "
        "invalidan explícitamente: pegarle a la DB en cada visita es trabajo "
        "repetido para devolver siempre lo mismo."
    )


def test_solo_cachea_el_caso_COMUN(sm):
    """Con búsqueda, filtro o modo edición se va a la DB: cachear eso es cachear ruido."""
    src = _ROUTER.read_text(encoding="utf-8")
    m = re.search(r"def api_supermarket_list.*?(?=\n@router|\n# ──)", src, re.DOTALL)
    assert m, "[P2-SUPERMARKET-LIST-CACHE] No se encontró el handler del listado."
    cuerpo = m.group(0)
    assert "_cached_active_rows" in cuerpo, (
        "[P2-SUPERMARKET-LIST-CACHE] El handler no consulta la caché."
    )
    # La condición tiene que excluir los tres casos que no son el común.
    for pieza in ("include_inactive", "q", "category"):
        assert pieza in cuerpo, (
            f"[P2-SUPERMARKET-LIST-CACHE] El gate de la caché no contempla `{pieza}`."
        )


def test_el_modo_edicion_JAMAS_se_sirve_de_cache(sm):
    """`include_inactive=1` es la vista del admin: tiene que ver lo que acaba de escribir.

    Anclado a la SEMÁNTICA, no a una forma concreta de escribir el `if`: la
    primera versión de este guard buscaba el literal `if not include_inactive and`
    y fallaba contra un código correcto que nombra la condición en una variable.
    Lo que importa es que la lectura de caché quede DEBAJO de una condición que
    incluya `not include_inactive`.
    """
    src = _ROUTER.read_text(encoding="utf-8")
    gate = re.search(r"(\w+)\s*=\s*not include_inactive\b[^\n]*", src)
    assert gate, (
        "[P2-SUPERMARKET-LIST-CACHE] No hay ninguna condición construida sobre "
        "`not include_inactive`.\n"
        "Servirle caché al admin le mostraría el catálogo de antes de su propia "
        "edición, que es justo el bug que P1-SUPERMARKET-CATALOG-CACHE evitó "
        "invalidando en cada mutación."
    )
    nombre = gate.group(1)
    # La LLAMADA, no la definición: un `.index()` a secas encuentra primero el
    # `def _cached_active_rows()`, que vive muy por encima del handler, y el trozo
    # a inspeccionar salía vacío — o sea el guard fallaba contra código correcto.
    lectura = src.index("_cached_active_rows()", gate.end())
    assert re.search(rf"if {nombre}\b", src[gate.end(): lectura]), (
        f"[P2-SUPERMARKET-LIST-CACHE] `_cached_active_rows()` no está bajo "
        f"`if {nombre}`: la caché podría servirse también en modo edición."
    )


# ---------------------------------------------------------------------------
# 2. Cabeceras de caché
# ---------------------------------------------------------------------------

def test_la_respuesta_publica_es_cacheable_por_el_navegador():
    src = _ROUTER.read_text(encoding="utf-8")
    assert "Cache-Control" in src, (
        "[P2-SUPERMARKET-LIST-CACHE] La respuesta no declara `Cache-Control`. Sin "
        "él, un usuario que navega dentro de /supermercado re-descarga el catálogo "
        "entero en cada vuelta."
    )
    assert "max-age" in src and "no-store" in src, (
        "[P2-SUPERMARKET-LIST-CACHE] Faltan las DOS caras: `max-age` para la vista "
        "pública y `no-store` para la de edición. Cachear la del admin en el "
        "navegador le escondería su propia escritura."
    )


# ---------------------------------------------------------------------------
# 3. El recorte del payload no puede llevarse por delante lo que se usa
# ---------------------------------------------------------------------------

def test_solo_se_recortan_las_columnas_sin_consumidor():
    """⚠️ El plan proponía recortar `notes` y `description`: los usa el editor."""
    src = _ROUTER.read_text(encoding="utf-8")
    front = _FRONT.read_text(encoding="utf-8")
    for campo in ("notes", "description", "category", "brand", "image_url"):
        assert f"p.{campo}" in front or f"{campo}," in src, (
            f"[P2-SUPERMARKET-LIST-CACHE] `{campo}` desapareció del payload pero el "
            "frontend lo consume — la página se rompería en silencio."
        )
    for campo in ("created_at", "updated_at"):
        assert f"p.{campo}" not in front, (
            f"[P2-SUPERMARKET-LIST-CACHE] `{campo}` pasó a usarse en el frontend, "
            "así que ya no se puede recortar del listado. Devuélvelo al SELECT."
        )


def test_la_cache_se_llena_con_el_CATALOGO_ENTERO_no_con_una_pagina():
    """⚠️ El bug que este P-fix produjo y tuvo que corregir EN CALIENTE.

    La primera versión cacheaba el resultado del `_fetch` paginado cuando
    `limit >= _MAX_LIMIT`. Suena razonable — «pidió el máximo, luego lo tiene
    todo» — y es FALSO: `_MAX_LIMIT` son 1.000 y el catálogo tiene 1.739 filas.
    La caché quedó con 1.000 y el endpoint empezó a responder `total: 1000`, o
    sea un supermercado TRUNCADO con toda la pinta de funcionar. Se detectó
    midiendo el payload contra producción, no leyendo el código.

    *«Pidió mucho» no es «esto es todo».* La única condición honesta para cachear
    algo como catálogo completo es haberlo traído SIN límite.
    """
    src = _ROUTER.read_text(encoding="utf-8")
    m = re.search(r"def _fetch_todas_activas.*?(?=\ndef |\n@)", src, re.DOTALL)
    assert m, (
        "[P2-SUPERMARKET-LIST-CACHE] No existe `_fetch_todas_activas`: la caché "
        "se estaría llenando desde la consulta PAGINADA."
    )
    # Fuera el docstring antes de mirar el SQL: el de esta función dice «Sin
    # `LIMIT` a propósito», así que un `in` sobre el texto crudo se dispara contra
    # la frase que EXPLICA la ausencia. Sexta vez en esta sesión que una prosa que
    # describe código confunde a un guard que lo busca.
    cuerpo = re.sub(r'""".*?"""', "", m.group(0), count=1, flags=re.DOTALL)
    assert "LIMIT" not in cuerpo.upper(), (
        "[P2-SUPERMARKET-LIST-CACHE] La carga que llena la caché lleva LIMIT. "
        "Cachear una página y llamarla catálogo es el bug original: el endpoint "
        "responde un `total` que no es el total."
    )
    # Sin comentarios: la explicación del bug CITA la condición que prohíbe.
    # Es el mismo tropiezo que dos líneas arriba, por la otra puerta — un guard
    # que lee prosa acaba prohibiendo que se documente lo que vigila.
    codigo = re.sub(r"#.*$", "", src, flags=re.MULTILINE)
    assert not re.search(r"limit\s*>=\s*_MAX_LIMIT", codigo), (
        "[P2-SUPERMARKET-LIST-CACHE] Volvió la condición `limit >= _MAX_LIMIT` "
        "para decidir si cachear. `_MAX_LIMIT` (1.000) es MENOR que el catálogo "
        "(1.739): esa condición se cumple sin que las filas sean todas."
    )


def test_las_columnas_de_grilla_no_incluyen_los_timestamps():
    src = _ROUTER.read_text(encoding="utf-8")
    m = re.search(r"_LIST_COLS\s*=\s*\"\"\"(.*?)\"\"\"", src, re.DOTALL)
    assert m, (
        "[P2-SUPERMARKET-LIST-CACHE] No existe `_LIST_COLS`. El listado seguiría "
        "arrastrando `created_at`/`updated_at`, que no consume nadie."
    )
    for campo in ("created_at", "updated_at"):
        assert campo not in m.group(1), (
            f"[P2-SUPERMARKET-LIST-CACHE] `{campo}` volvió a las columnas del listado."
        )
