# -*- coding: utf-8 -*-
"""[ARQ27-P1-07 · 2026-09-06] `CatalogCapabilitySnapshot`: qué alimentos existen en el mercado de un país.

`compile_policy` tiene desde F2 un paso de disponibilidad real —el nº 3, entre la dieta y el
presupuesto— que descarta un ancla que el mercado no vende. Ese paso pide `known_ingredients` en su
contexto y **`compile_from_form` nunca se lo pasaba**: el único constructor de política que usa
producción dejaba el contexto vacío, así que el paso se anotaba `market_check_skipped` y no corría
jamás. El registro de F3 lo publicó como 57 de 57 planes.

Consecuencia: la cultura podía pedir un plato cuyo ingrediente ancla el país no vende, y nadie se
enteraba hasta mucho más abajo. Es especialmente caro en perfiles veganos y en cocinas cruzadas —una
cocina dominicana comprando en Estados Unidos— que es justo lo que el sistema de países abrió.

**Cultura ≠ mercado (I16).** Este snapshot es del MERCADO: dice qué se puede comprar, no qué se
cocina. La biblioteca cultural sigue viniendo por su lado.

**Ausente ≠ vacío.** Si el catálogo no se puede leer, esto devuelve `None` y el compilador conserva
`market_check_skipped` — un desconocido explícito. Devolver una lista vacía diría «este país no vende
NADA» y borraría todas las anclas del usuario. La diferencia entre no saber y saber que no hay es
exactamente el gap que ARQ27-P0-03 cierra del lado de los nutrientes.

Alcance: identidad y pertenencia al catálogo del país. SKU, stock, precio e incertidumbre por mercado
son ARQ30-P2-03 y no se prometen aquí.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Positivos cacheados por país (el catálogo ya viene cacheado aguas arriba; esto evita rehacer el set
# y la huella en cada ancla). Los negativos NO se cachean —la DB puede volver— pero su aviso se emite
# una sola vez por país y proceso: repetirlo por llamada convierte una medición en un muro de logs.
_CACHE: dict[str, dict] = {}
_AVISADOS: set = set()


def _avisar_una_vez(cc: str, msg: str) -> None:
    if cc not in _AVISADOS:
        _AVISADOS.add(cc)
        logger.warning(msg)


def reset_cache() -> None:
    """Vacía los caches de proceso. Existe para los tests: un snapshot construido con un catálogo
    falso sobrevive al `monkeypatch` que lo puso —el mock se revierte, la caché no— y el siguiente
    test lee tres filas donde hay trescientas."""
    _CACHE.clear()
    _DISPONIBLE.clear()
    _AVISADOS.clear()


def _country(country: Any) -> str:
    try:
        from constants import canonicalize_country
        return str(canonicalize_country(country) or "DO").upper()
    except Exception:
        return str(country or "DO").strip().upper() or "DO"


def _predicado_de_pais():
    """¿Este país puede COMPRAR esta fila? Se apoya en `_BETA_CATALOG_DO_EXCLUSIVE_NAMES`, la lista de
    alimentos que solo existen en el mercado dominicano (Casabe, Orégano dominicano, Longaniza
    dominicana, Queso de hoja…).

    **No usa el filtro completo del catálogo cerrado**, y esa distinción es el matiz que costó un test
    en rojo. `_verified_catalog_name_allowed_for_country` quita además los `_COUNTRY_CATALOG_SHADOWED_TWINS`
    —«Gambas», «Judías blancas», «Judías pintas», «Requesón» en ES— que NO son alimentos ausentes: son
    el MISMO alimento con su nombre español, ocultos para no mostrar la fila dos veces. Es una decisión
    de PRESENTACIÓN. Tratarla como disponibilidad decía que en España no se venden gambas, y borraba
    del pool la paella de la propia biblioteca española.

    Se resuelve UNA vez por construcción de snapshot: hacer el `import` dentro del bucle costaba 1,85 s
    por país —347 sentencias de import— para un predicado que es una pertenencia a conjunto."""
    try:
        from graph_orchestrator import _BETA_CATALOG_DO_EXCLUSIVE_NAMES as _solo_do
        return lambda nm, cc: cc == "DO" or nm not in _solo_do
    except Exception:
        return lambda nm, cc: True


def catalog_capability(country: Any) -> Optional[dict]:
    """Snapshot de capacidad del mercado, o `None` si el catálogo no se pudo leer.

    Devuelve `{"market_country", "names", "aliases", "count", "source", "fingerprint"}`. `names` son
    los nombres canónicos de las filas que ese país puede comprar; `aliases` amplía el matcheo sin
    contar como alimentos distintos (un alias no es diversidad — ARQ27-P1-05)."""
    cc = _country(country)
    hit = _CACHE.get(cc)
    if hit is not None:
        return hit
    try:
        from shopping_calculator import get_master_ingredients
        rows = list(get_master_ingredients() or [])
    except Exception as e:
        _avisar_una_vez(cc, f"[ARQ27-P1-07] catálogo no legible para {cc}: {e!r} → capacidad desconocida")
        return None
    if not rows:
        # Vacío no es «este país no vende nada»: es que no lo sabemos. Con una lista vacía el paso 3
        # de `compile_policy` borraría TODAS las anclas del usuario y lo llamaría evidencia.
        _avisar_una_vez(cc, f"[ARQ27-P1-07] catálogo VACÍO para {cc} → capacidad desconocida, no cero")
        return None
    permitido = _predicado_de_pais()
    names, aliases, filas_ok = [], [], []
    for r in rows:
        nm = str(r.get("name") or "").strip()
        if not nm or not permitido(nm, cc):
            continue
        names.append(nm)
        filas_ok.append(r)
        # Los alias van DENTRO del filtro de país, no fuera: el alias de una fila que este mercado no
        # vende haría pasar por disponible justo lo que el filtro acaba de quitar.
        for a in (r.get("aliases") or []):
            a = str(a or "").strip()
            if a:
                aliases.append(a)
    if not names:
        _avisar_una_vez(cc, f"[ARQ27-P1-07] ninguna fila habilitada para {cc} → capacidad desconocida")
        return None
    names = sorted(set(names))
    aliases = sorted(set(aliases))
    fp = hashlib.sha256(("|".join(names)).encode("utf-8")).hexdigest()[:12]
    snap = {"market_country": cc, "names": names, "aliases": aliases,
            "count": len(names), "source": "master_ingredients", "fingerprint": fp,
            # Las filas habilitadas se conservan para poder construir el índice del catálogo bajo
            # demanda: `resolve_constituent` es el SSOT de «este nombre resuelve a una fila» y ya
            # sabe de alias, acentos y singular/plural.
            "_rows": filas_ok}
    _CACHE[cc] = snap
    return snap


def known_ingredient_names(country: Any) -> Optional[list]:
    """Lo que el paso de mercado de `compile_policy` consume: nombres + alias, o `None` si no se sabe."""
    snap = catalog_capability(country)
    if not snap:
        return None
    return list(snap["names"]) + list(snap["aliases"])


# Verdicto por (alimento, mercado). Los constituyentes distintos de las 690 plantillas son ~700, así
# que la tabla se llena una vez; sin ella cada plantilla recorrería 1.500 nombres por ingrediente.
_DISPONIBLE: dict[tuple, bool] = {}


def is_available(name: Any, country: Any) -> Optional[bool]:
    """¿Se puede comprar este alimento en ese mercado? `None` = no se sabe (catálogo ilegible).

    Resuelve con `dish_registry.resolve_constituent` sobre el índice del catálogo del país — el mismo
    SSOT que usa el compilador, que ya sabe de alias, acentos y singular/plural. Solo los nombres que
    ese índice NO resuelve caen en el barrido lineal de `_matches`; hacerlo al revés costaba 4,5 s en
    frío en la primera generación."""
    nm = str(name or "").strip()
    if not nm:
        return None
    snap = catalog_capability(country)
    if not snap:
        return None
    key = (nm.lower(), snap["fingerprint"])
    hit = _DISPONIBLE.get(key)
    if hit is not None:
        return hit
    idx = snap.get("_index")
    if idx is None:
        import dish_registry as _dr
        idx = _dr.build_catalog_index(snap["_rows"])
        snap["_index"] = idx
    import dish_registry as _dr
    ok = _dr.resolve_constituent(nm, idx) is not None
    if not ok:
        # Último recurso: el matcher de la política, más laxo (acepta «Harina de yuca» para «Yuca»).
        # Solo lo pagan los nombres que el índice no resuelve, que son unos pocos.
        try:
            from plan_policy import _matches
            ok = any(_matches(nm, k) for k in (list(snap["names"]) + list(snap["aliases"])))
        except Exception:
            return None
    _DISPONIBLE[key] = ok
    return ok


def template_buyable_in(constituent_names, country: Any) -> bool:
    """¿Todos los constituyentes de esta plantilla se compran en ese mercado? Desconocido ⇒ `True`:
    sin catálogo no se recorta nada — la conducta previa, no una escasez inventada."""
    for nm in (constituent_names or ()):
        if is_available(nm, country) is False:
            return False
    return True
