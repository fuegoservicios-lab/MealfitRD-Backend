"""[P1-I18N-PUSH-LOCALE-SIEMPRE-NULO · 2026-08-23] El catálogo de push (43 mensajes × 4
idiomas) era código muerto: la consulta del idioma devolvía una LISTA y el código la leía
como diccionario.

`P1-I18N-PUSH-CRON-ESPANOL` (22-ago) eligió bien el cuello de botella y escribió la tabla
completa. Lo que no funcionaba era la única línea que decide EN QUÉ IDIOMA:

    _perfil = execute_sql_query("SELECT locale FROM user_profiles WHERE id = %s",
                                (user_id,), fetch_all=False)
    if _perfil:
        _locale = _perfil.get("locale") if hasattr(_perfil, "get") else None

Pasar `fetch_all=False` SIN `fetch_one=True` no significa «una fila»: cae en la rama por
defecto del helper, que hace `cursor.fetchall()` y devuelve una LISTA de dicts. Una lista no
tiene `.get`, así que el `hasattr` sale False y `_locale` queda en **None para todos los
usuarios, siempre**. `translate_push_text(texto, None)` es fail-open y devuelve el español.

MEDIDO contra Neon con la llamada EXACTA del fichero:

    WARNING db_core: Caller no pasó fetch_one/fetch_all pero la query devolvió 1 filas
    tipo devuelto: list -> [{'locale': 'es-DO'}]
    hasattr(r,'get'): False
    _locale que utils_push calcularia: None

O sea que las 43×4 traducciones no se han pintado JAMÁS, en la superficie que el propio
P-fix llamó «la MENOS perdonable de todas — llega sin que la pidas, se lee de un vistazo y
no hay dónde cambiar el idioma». Y de paso cada push emitía un WARNING de `db_core` en
producción.

POR QUÉ ESTE GUARD Y NO EL QUE HABÍA: el existente comprueba que el fichero llama a
`translate_push_text`. Eso seguía siendo cierto con el bug puesto — la llamada estaba, con
`None` de segundo argumento. Un guard que mira si el fichero MENCIONA la traducción no puede
distinguir «traduce» de «tiene la palabra». Éste ejecuta `send_push_notification` de verdad,
con un doble de `execute_sql_query` que devuelve **la forma REAL del helper** (una lista), y
mira qué sale en el payload. El doble infiel es lo que dejaba pasar el defecto.

tooltip-anchor: P1-I18N-PUSH-LOCALE-SIEMPRE-NULO
"""
from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import patch

import pytest

_MARKER = "P1-I18N-PUSH-LOCALE-SIEMPRE-NULO"


@pytest.fixture()
def _pywebpush_stub():
    """`pywebpush` no está instalado en CI y además NO queremos que nada salga a la red.
    El stub captura el payload, que es lo que este guard mide."""
    capturado = {}

    def _webpush(subscription_info=None, data=None, **_kw):
        capturado["data"] = data
        return types.SimpleNamespace(status_code=201)

    mod = types.ModuleType("pywebpush")
    mod.webpush = _webpush
    mod.WebPushException = type("WebPushException", (Exception,), {})
    anterior = sys.modules.get("pywebpush")
    sys.modules["pywebpush"] = mod
    # Las VAPID gatean el envío antes de llegar a la traducción; sin ellas el guard mediría
    # el gate y no el idioma. Se ponen y se quitan, nunca se dejan puestas (la lección de
    # los stubs de `sys.modules` sin deshacer).
    with patch.dict(os.environ, {"VAPID_PRIVATE_KEY": "k-de-test",
                                 "VAPID_CLAIM_EMAIL": "mailto:test@test.local"}):
        try:
            yield capturado
        finally:
            if anterior is None:
                sys.modules.pop("pywebpush", None)
            else:
                sys.modules["pywebpush"] = anterior


def _mensaje_del_catalogo():
    """Un par (español, traducción) que exista de verdad, para no clavar copy en el test."""
    from push_i18n import _TITULOS  # noqa: PLC0415

    for clave, por_idioma in _TITULOS.items():
        fr = (por_idioma or {}).get("fr-FR")
        if isinstance(clave, str) and isinstance(fr, str) and fr and fr != clave:
            return clave, fr
    pytest.skip("el catálogo de push no tiene ninguna entrada fr-FR distinta del español")


@pytest.mark.parametrize("forma_del_helper", ["lista", "dict"])
def test_el_push_sale_en_el_idioma_del_usuario(_pywebpush_stub, forma_del_helper) -> None:
    """La CONDUCTA: un usuario con `locale='fr-FR'` recibe el push en francés.

    Se prueban las DOS formas que `execute_sql_query` puede devolver, porque el defecto era
    exactamente asumir una de ellas: si mañana el helper cambia de forma, este guard lo dice
    en vez de volver a fallar en silencio.
    """
    import utils_push  # noqa: PLC0415

    es, fr = _mensaje_del_catalogo()
    fila = {"locale": "fr-FR"}
    devuelve = [fila] if forma_del_helper == "lista" else fila

    with patch.object(utils_push, "execute_sql_query") as _q:
        # 1ª llamada: las suscripciones. 2ª: el locale.
        _q.side_effect = lambda sql, *a, **k: (
            [{"subscription_data": json.dumps({"endpoint": "https://x", "keys": {}})}]
            if "push_subscriptions" in str(sql).lower() else devuelve
        )
        utils_push.send_push_notification("u-1", es, es, "/dashboard")

    data = _pywebpush_stub.get("data")
    assert data, f"no se envió ningún push (forma={forma_del_helper}) [{_MARKER}]"
    payload = json.loads(data)
    assert payload["title"] == fr, (
        f"el push salió en «{payload['title']}» y el usuario tiene locale fr-FR: el catálogo "
        f"de push está inerte (forma del helper = {forma_del_helper}). Es la superficie que "
        f"llega sin que la pidas y no tiene dónde cambiar el idioma. [{_MARKER}]"
    )


def test_sin_locale_sigue_saliendo_en_espanol(_pywebpush_stub) -> None:
    """Fail-open deliberado: una notificación en español es una degradación, una que no sale
    es un fallo. Se ancla para que el arreglo del idioma no convierta el fallo en silencio."""
    import utils_push  # noqa: PLC0415

    es, _fr = _mensaje_del_catalogo()
    with patch.object(utils_push, "execute_sql_query") as _q:
        _q.side_effect = lambda sql, *a, **k: (
            [{"subscription_data": json.dumps({"endpoint": "https://x", "keys": {}})}]
            if "push_subscriptions" in str(sql).lower() else []
        )
        utils_push.send_push_notification("u-1", es, es, "/dashboard")

    payload = json.loads(_pywebpush_stub["data"])
    assert payload["title"] == es, (
        f"sin locale el push debe salir en español y salió «{payload['title']}» [{_MARKER}]"
    )
