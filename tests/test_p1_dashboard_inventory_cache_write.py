"""[P1-DASH-INV-CACHE-WRITE · 2026-08-14] El Dashboard LEÍA la caché de inventario
sin escribirla nunca, así que en cada refresco volvía a arrancar sin datos.

EL SÍNTOMA que lo destapó: el aviso «Tu Nevera ya cubre la lista (46 ítems de la
compra)» desaparecía unos milisegundos en CADA refresco de la página.

LA CADENA. `shoppingDeltaMeta` —el memo que alimenta ese aviso— se abre con
`if (liveInventory !== null && …)`, o sea que mientras el inventario sea `null`
devuelve `null` y el aviso no se pinta. Hasta ahí es correcto: `null` significa
«todavía no sé», y esconder el aviso es preferible a afirmar una cobertura que no
se ha comprobado.

El problema estaba un piso más abajo. `P1-DASHBOARD-CACHE-INVENTORY · 2026-05-20`
hizo que el Dashboard HIDRATARA desde `pantryCache` en el `useState` inicial, con
esta frase en su comentario: «el cache ya almacenaba el inventory tras cada visita
a Nevera PERO Dashboard NO lo leía al mount — Dashboard solo guardaba sin leer».
Se añadió la lectura… y el único `setCachedInventory` del fichero vive dentro del
flujo de RESTOCK. **El fetch de arranque nunca repone la caché.**

Resultado: la caché es de un solo sentido. El Dashboard consume lo que produce la
Nevera, y quien no entra a `/dashboard/pantry` —o entró hace más de 10 min, el
TTL— arranca sin nada. Cada F5: `liveInventory = null` → sin aviso → llega el
fetch → aparece de golpe.

QUINTA APARICIÓN DE LA MISMA CLASE en este repo (aviso rojo de urgentes, chips de
marca, CTA de escanear, banner ámbar de nevera baja). La variante nueva es
instructiva: aquí el default no mentía, **faltaba la mitad del circuito**.

    Una caché que se lee y no se escribe está vacía la primera vez, siempre.

Tooltip-anchor: P1-DASH-INV-CACHE-WRITE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASH = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def _read() -> str:
    if not _DASH.exists():
        pytest.fail("[P1-DASH-INV-CACHE-WRITE] No existe Dashboard.jsx")
    return _DASH.read_text(encoding="utf-8")


def _sin_comentarios(t: str) -> str:
    """El comentario que EXPLICA el arreglo nombra las funciones que vigila."""
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", t, flags=re.MULTILINE)


def _cuerpo_del_fetch_de_arranque(t: str) -> str:
    """El `fetchLiveInventory` del efecto de mount, hasta su cierre."""
    i = t.find("const fetchLiveInventory = async () => {")
    assert i != -1, (
        "[P1-DASH-INV-CACHE-WRITE] No se encontró `fetchLiveInventory`. Si el fetch "
        "de arranque se renombró, mueve también este guard."
    )
    fin = t.find("fetchLiveInventory();", i)
    return t[i:fin if fin != -1 else i + 3000]


def test_el_fetch_de_arranque_repone_la_cache():
    cuerpo = _sin_comentarios(_cuerpo_del_fetch_de_arranque(_read()))
    assert "setCachedInventory(" in cuerpo, (
        "[P1-DASH-INV-CACHE-WRITE] El fetch de arranque del Dashboard NO escribe "
        "`setCachedInventory`.\n"
        "El `useState` inicial SÍ lee de esa caché (P1-DASHBOARD-CACHE-INVENTORY), "
        "así que sin la escritura el circuito queda a medias: quien no visita "
        "/dashboard/pantry (o entró hace más del TTL) arranca con `liveInventory = "
        "null` en CADA refresco, y el aviso «Tu Nevera ya cubre la lista» aparece "
        "tarde.\n"
        "Una caché que se lee y no se escribe está vacía la primera vez, siempre."
    )


def test_solo_se_cachea_una_respuesta_FRESCA():
    """Cachear un timeout convertiría un fallo puntual en 10 minutos de mentira."""
    cuerpo = _cuerpo_del_fetch_de_arranque(_read())
    i_guard = cuerpo.find("if (!result.stale)")
    i_write = cuerpo.find("setCachedInventory(")
    assert i_guard != -1 and i_write > i_guard, (
        "[P1-DASH-INV-CACHE-WRITE] La escritura de caché no está bajo "
        "`if (!result.stale)`.\n"
        "`fetchFreshInventoryWithTimeout` marca `stale` en timeout, error o "
        "respuesta vacía, y en ese caso el código NO toca `liveInventory` a "
        "propósito. Persistir esa respuesta guardaría el fallo durante todo el "
        "TTL (10 min) y lo serviría como si fuera el inventario del usuario."
    )


def test_el_dashboard_sigue_hidratando_desde_la_cache():
    """La otra mitad del circuito: si se pierde la lectura, el arreglo es inútil."""
    t = _sin_comentarios(_read())
    assert re.search(r"useState\(\s*_cachedInv\s*\|\|\s*null\s*\)", t), (
        "[P1-DASH-INV-CACHE-WRITE] El Dashboard dejó de hidratar `liveInventory` "
        "desde `getCachedInventory()` (P1-DASHBOARD-CACHE-INVENTORY). Escribir la "
        "caché sin leerla es el mismo circuito a medias, por el otro extremo."
    )


def test_el_aviso_sigue_exigiendo_saber_el_inventario():
    """⚠️ El arreglo NO es relajar el gate a `liveInventory?.length`.

    Con `null` significando «todavía no sé», el memo devuelve `null` y el aviso no
    se pinta — y eso es CORRECTO. Afirmar «tu Nevera ya cubre la lista» sin haber
    comprobado el inventario sería peor que el parpadeo: convertiría un «no sé» en
    una promesa. Lo que se arregla es que ese «no sé» dure menos, no que se
    interprete como un dato.
    """
    t = _sin_comentarios(_read())
    assert "liveInventory !== null" in t, (
        "[P1-DASH-INV-CACHE-WRITE] `shoppingDeltaMeta` dejó de exigir que el "
        "inventario se conozca. El parpadeo NO se arregla afirmando cobertura sin "
        "datos: se arregla haciendo que los datos estén en el primer frame."
    )
