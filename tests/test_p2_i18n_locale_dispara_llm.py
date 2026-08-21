"""[P2-I18N-LOCALE-DISPARA-LLM · 2026-08-21] Cambiar de idioma dispara gasto LLM, y
`PATCH /api/profile` era el único write de su router sin limitador.

QUÉ PASA. Desde `P1-PLAN-DISPLAY-I18N` (2026-08-19), `PATCH /api/profile` no es sólo
un UPDATE escalar: cuando el `locale` nuevo no es `es-DO`, mira si al plan activo le
falta `_display[locale]` en su primer o último día y, si falta, despacha
`schedule_plan_display_enrichment` — una traducción LLM del plan entero, por lotes.

Y ese endpoint recibía exactamente dos dependencias, `Body` y `get_verified_user_id`.
Sus tres hermanos del MISMO router sí declaran limitador (`_CATALOG_LIMITER`,
`_PLAN_MODE_LIMITER`, `_TARGETS_LIMITER`). O sea: el único write del router sin
limitador resultó ser justo el que cuesta dinero.

POR QUÉ UN LIMITADOR Y NO LA CUOTA. Es la doctrina que la sección
«Historial-quota-exemption» de CLAUDE.md deja fija: el paywall mensual es para el
COSTE del producto, y `RateLimiter` per-bucket es la herramienta contra el spam.
Aplicarle `verify_api_quota` a esto sería absurdo por partida doble — al llegar al cap
el usuario no podría cambiar de idioma, y cada cambio le quemaría crédito de PLANES,
porque `get_monthly_api_usage` cuenta toda fila de `api_usage` sin filtrar endpoint.

QUÉ ANCLA. No este endpoint: la CLASE. Todo método que no sea GET en
`routers/user_data.py` tiene que declarar un `Depends(<algo>_LIMITER)`. Un guard por
endpoint concreto no habría visto éste, que nació sin limitador el día que el router
creció; y no lo verá tampoco el siguiente.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROUTER = _BACKEND / "routers" / "user_data.py"

_MARKER = "P2-I18N-LOCALE-DISPARA-LLM"

# Métodos que MUTAN. Un GET puede ir sin limitador si es barato y está autenticado
# (varios del Historial lo están, a propósito y documentado); un write no.
_METODOS_ESCRITURA = ("post", "put", "patch", "delete")


def _fuente() -> str:
    if not _ROUTER.exists():
        pytest.skip(f"{_ROUTER} no existe en este checkout")
    return _ROUTER.read_text(encoding="utf-8")


def _endpoints_de_escritura(src: str) -> list[tuple[str, str, str]]:
    """(método, ruta, firma) de cada endpoint que muta, del decorador al `):`."""
    fuera = []
    patron = re.compile(
        r"@router\.(" + "|".join(_METODOS_ESCRITURA) + r")\(\s*[\"']([^\"']+)[\"']",
    )
    for m in patron.finditer(src):
        # La firma va del decorador hasta el `):` que la cierra. Se corta ahí y no en
        # el docstring: un `Depends` mencionado en la prosa no es una dependencia.
        resto = src[m.end():]
        fin = resto.find("):")
        firma = resto[:fin] if fin != -1 else resto[:800]
        fuera.append((m.group(1).upper(), m.group(2), firma))
    return fuera


# [P2-I18N-LOCALE-DISPARA-LLM · 2026-08-21] Los writes que HOY siguen sin limitador.
#
# LA AUDITORÍA SE EQUIVOCÓ EN ESTO y conviene dejarlo escrito: reportó que
# `PATCH /profile` era «el ÚNICO write de su router sin RateLimiter». Medido: el
# router tiene DOCE endpoints de escritura y sólo TRES limitadores definidos, de los
# que sólo `plan-mode` se usa. O sea que faltaban diez, no uno.
#
# Se arregla el que la auditoría identificó —`PATCH /profile`, el único que dispara
# gasto LLM— y los demás quedan aquí, como deuda ANOTADA con su forma exacta en vez
# de como un guard apagado. La razón de no cerrarlos de una tacada: son los writes de
# la Nevera, que admiten ráfagas legítimas (añadir la compra entera item a item, el
# escaneo por foto), así que su ventana hay que medirla contra uso real y no
# inventarla. Ponerles un número a ojo es la clase de guard que revienta al primer
# usuario normal y acaba desactivado.
#
# TRINQUETE: esta lista puede ENCOGER, nunca crecer. Un endpoint de escritura nuevo
# nace con limitador o el test lo para.
_SIN_LIMITADOR_PENDIENTES = {
    "POST /inventory/items",
    "POST /inventory/increment",
    "PATCH /inventory/items/{item_id}/unit",
    "PATCH /inventory/items/{item_id}",
    "DELETE /inventory/items/{item_id}",
    "DELETE /inventory/items",
    "POST /inventory/photo-scan",
    "PUT /user/preferences/super-personalization",
    "PUT /user/preferences/clinical-profile",
    "PUT /user/preferences/staple-foods",
}


def test_ningun_write_nuevo_nace_sin_limitador() -> None:
    """El guard por CLASE, con trinquete sobre la deuda existente.

    Un guard por endpoint concreto no habría visto `PATCH /profile`, que nació sin
    limitador el día que el router creció, y no verá el siguiente. Por eso se ancla la
    clase; y por eso la deuda va en una lista explícita en vez de relajar la regla.
    """
    src = _fuente()
    endpoints = _endpoints_de_escritura(src)
    assert endpoints, (
        "No encontré ningún endpoint de escritura en user_data.py — si cambió el "
        "estilo de los decoradores, actualiza este test."
    )

    sin_limitador = {
        f"{metodo} {ruta}"
        for metodo, ruta, firma in endpoints
        if not re.search(r"Depends\(\s*_\w*LIMITER\s*\)", firma)
    }

    nuevos = sin_limitador - _SIN_LIMITADOR_PENDIENTES
    assert not nuevos, (
        "Endpoints de escritura NUEVOS sin `Depends(<algo>_LIMITER)`: "
        + ", ".join(sorted(nuevos))
        + ". El paywall NO es la herramienta (al llegar al cap el usuario no podría "
        "tocar un ajuste suyo, y quemaría crédito de planes); el limitador per-bucket "
        f"sí. [{_MARKER}]"
    )

    arreglados = _SIN_LIMITADOR_PENDIENTES - sin_limitador
    assert not arreglados, (
        "Estos ya tienen limitador y siguen en `_SIN_LIMITADOR_PENDIENTES`: "
        + ", ".join(sorted(arreglados))
        + ". Quítalos de la lista — un trinquete que no se aprieta deja de medir la "
        f"deuda que dice medir. [{_MARKER}]"
    )


def test_el_patch_de_perfil_tiene_el_suyo() -> None:
    """El caso concreto, nombrado, porque es el que cuesta dinero.

    Se ancla aparte del guard de clase para que el mensaje diga POR QUÉ este importa
    más que los demás: es el que despacha la traducción LLM del plan.
    """
    src = _fuente()
    patch = next(
        (firma for metodo, ruta, firma in _endpoints_de_escritura(src)
         if metodo == "PATCH" and ruta == "/profile"),
        None,
    )
    assert patch is not None, "No encuentro `@router.patch(\"/profile\")` en user_data.py."
    assert re.search(r"Depends\(\s*_PROFILE_PATCH_LIMITER\s*\)", patch), (
        "`PATCH /profile` no declara `_PROFILE_PATCH_LIMITER`. Es el endpoint que "
        "despacha `schedule_plan_display_enrichment` al cambiar de `locale`: una "
        "traducción LLM del plan entero, por lotes. Sus tres hermanos del mismo "
        f"router sí tienen limitador. [{_MARKER}]"
    )


def test_el_despacho_llm_sigue_colgando_del_cambio_de_locale() -> None:
    """Si el despacho se mueve o desaparece, el limitador deja de tener esta razón.

    No se ancla para congelar el despacho —puede moverse a un cron mañana— sino para
    que quien lo mueva vea que este limitador se justificaba por él, y decida a
    conciencia si sigue haciendo falta.
    """
    src = _fuente()
    assert "schedule_plan_display_enrichment" in src, (
        "`schedule_plan_display_enrichment` ya no se despacha desde user_data.py. "
        "Si el enriquecimiento se movió a otro sitio, revisa si `_PROFILE_PATCH_LIMITER` "
        f"sigue teniendo motivo, y actualiza este test. [{_MARKER}]"
    )


def test_el_limitador_no_es_la_cuota() -> None:
    """`verify_api_quota` en un ajuste de perfil sería el anti-patrón documentado.

    Al llegar al cap el usuario no podría cambiar de idioma —ni volver al suyo— y cada
    cambio le quemaría crédito de PLANES, porque `get_monthly_api_usage` cuenta toda
    fila de `api_usage` sin filtrar endpoint. Es literalmente el razonamiento por el
    que `/restock` y `/inventory/consume` salieron del paywall.
    """
    src = _fuente()
    patch = next(
        (firma for metodo, ruta, firma in _endpoints_de_escritura(src)
         if metodo == "PATCH" and ruta == "/profile"),
        "",
    )
    assert "verify_api_quota" not in patch, (
        "`PATCH /profile` pasó a depender de `verify_api_quota`. Al llegar al cap el "
        "usuario quedaría ATRAPADO en su idioma actual, y cada cambio de un ajuste "
        f"suyo le quemaría crédito de planes. Usa el RateLimiter. [{_MARKER}]"
    )
