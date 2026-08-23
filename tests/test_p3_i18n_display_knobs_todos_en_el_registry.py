"""[P3-I18N-DISPLAY-KNOBS-TODOS-EN-EL-REGISTRY · 2026-08-23] El bloque que declara los knobs
del `_display` en el import era una LISTA A MANO, así que se quedó atrás en cuanto nacieron
knobs nuevos.

`P3-I18N-DISPLAY-KNOBS-PEREZOSOS` (22-ago) cerró el problema de fondo —`knobs._env_*` sólo
registra al ser LLAMADO, y estos accesores viven dentro de funciones que sólo corren cuando
hay algo que traducir, así que `get_knobs_registry_snapshot()` no los conocía— declarándolos
en el import. Correcto. Pero lo hizo enumerando cinco funciones a mano, y el módulo tiene
SIETE knobs: `MAX_INFLIGHT` y `MAX_LOCALES` se quedaron fuera. El segundo nació el 23-ago,
un día después del cierre, y el guard de aquel P-fix clava esas mismas cinco — o sea que un
knob nuevo entra invisible y con el guard en verde.

Este test no enumera nada: **deriva** la lista del propio fuente del módulo y la compara con
el registry VIVO. Un knob nuevo queda cubierto sin tocar el test — que es la diferencia entre
un guard y una lista que alguien tiene que acordarse de actualizar.

Por qué importa aunque sea P3: el registry es lo que un operador consulta para saber qué
puede mover sin redeploy. Un knob que no sale ahí es, en la práctica, una constante — y
`MAX_LOCALES` es justo el que se querría tocar con el jsonb creciendo.

tooltip-anchor: P3-I18N-DISPLAY-KNOBS-TODOS-EN-EL-REGISTRY
"""
from __future__ import annotations

import re
from pathlib import Path

_MARKER = "P3-I18N-DISPLAY-KNOBS-TODOS-EN-EL-REGISTRY"
_MODULO = Path(__file__).resolve().parents[1] / "plan_display_i18n.py"

# Los knobs del módulo se leen SIEMPRE con un literal como primer argumento de `_env_*`.
# Se extraen de ahí y no de una lista escrita a mano: es la única forma de que esto no
# vuelva a quedarse atrás.
_RX = re.compile(r'_env_(?:int|float|bool|str)\(\s*"(MEALFIT_[A-Z0-9_]+)"')


def _knobs_del_fuente() -> set:
    return set(_RX.findall(_MODULO.read_text(encoding="utf-8")))


def test_el_extractor_encuentra_algo() -> None:
    """Un guard que deriva su universo de un `re.findall` tiene que reventar cuando ese
    universo sale vacío, ANTES de comparar la nada contra la nada. El repo ya pagó esta
    lección con `_LM_DISPLAY_GROUPS`, cuyo parser devolvía un set vacío y acusaba al sitio
    equivocado."""
    encontrados = _knobs_del_fuente()
    assert len(encontrados) >= 5, (
        f"el extractor de knobs encontró {len(encontrados)} en {_MODULO.name}: o el módulo "
        f"cambió de forma de leer el entorno, o este guard quedó midiendo el vacío. "
        f"[{_MARKER}]"
    )


def test_todos_los_knobs_del_modulo_estan_en_el_registry_vivo() -> None:
    """CONDUCTA: se importa el módulo y se le pregunta al registry de verdad."""
    import plan_display_i18n  # noqa: F401,PLC0415 — el import es lo que los declara
    from knobs import get_knobs_registry_snapshot  # noqa: PLC0415

    registrados = set(get_knobs_registry_snapshot() or {})
    faltan = sorted(_knobs_del_fuente() - registrados)
    assert not faltan, (
        f"estos knobs del `_display` no aparecen en el registry vivo tras importar el "
        f"módulo: {faltan}. El registry es lo que un operador consulta para saber qué puede "
        f"mover sin redeploy, así que un knob ausente es una constante con nombre de knob. "
        f"[{_MARKER}]"
    )


def test_la_declaracion_eager_cubre_todos_los_accesores() -> None:
    """La otra dirección: que el bloque de declaración no se quede corto otra vez.

    Se cuenta cuántas funciones enumera el `for _declarar in (...)` y se compara con el
    número de knobs del módulo. No es una cifra clavada: se derivan las dos.
    """
    fuente = _MODULO.read_text(encoding="utf-8")
    bloque = re.search(r"for _declarar in \((.*?)\):", fuente, re.S)
    assert bloque, f"desapareció el bloque de declaración eager de knobs [{_MARKER}]"
    enumeradas = [x.strip() for x in bloque.group(1).split(",") if x.strip()]
    assert len(enumeradas) >= len(_knobs_del_fuente()), (
        f"el bloque declara {len(enumeradas)} accesores y el módulo tiene "
        f"{len(_knobs_del_fuente())} knobs. Una lista a mano se queda atrás en cuanto nace "
        f"uno nuevo — que es exactamente lo que pasó con MAX_INFLIGHT y MAX_LOCALES. "
        f"[{_MARKER}]"
    )
