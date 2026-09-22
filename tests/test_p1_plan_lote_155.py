"""[P1-PLAN-LOTE-155 · 2026-09-22] Tres cierres del coach para la beta, los tres NATIVOS.

Contexto: el 22-sep el primer APK de Android corrió en un teléfono real (log de nginx: origen
`https://localhost`, con su preflight). Revisando el coach de punta a punta antes de repartirlo a
5 testers salieron tres cosas que en la WEB no se ven y en la APP sí.

A · LAS CABECERAS DE LA CUOTA NO SE PUBLICABAN.
    `verify_coach_quota` manda `X-Coach-Quota-Limit/Used/Resets-At` con cada 402 desde el 2-sep
    (P1-COACH-QUOTA-METER) para que el chat diga «llegaste a tus 60 mensajes, se renueva el 1 de
    octubre». Ninguna estaba en `expose_headers`, y una cabecera que no está en esa lista NO
    EXISTE para el JS: el navegador la entrega al túnel y no al `fetch`.

    En la web nunca se notó porque nginx sirve API y frontend en el MISMO origen — sin CORS no
    hay filtro. En la app nativa toda llamada es cross-origin, así que la función nacía muerta
    justo donde más falta hacía.

      *Mandar una cabecera personalizada no es publicarla: sin `expose_headers` el emisor cree
      que informa y el receptor no ve nada.*

    Es la MISMA familia que `P1-IOS-CORS-NATIVE-ORIGIN`, que en agosto encontró `X-MF-Session`
    mandado y no permitido, con el comentario de la lista pidiendo extenderla al añadir un `X-*`.
    La nota estaba escrita; el segundo caso llegó igual. Por eso este test no vigila una cabecera:
    vigila LA CLASE — toda `X-*` que el frontend lea tiene que estar publicada.

B · el copy de respaldo del 402 invitaba a mejorar de plan DENTRO de la app (Apple 3.1.1).
C · un motor web viejo no rompe: ignora `color-mix()` y el usuario ve una pantalla descolorida.
    (B y C se miden en `frontend/src/__tests__/lote155.test.jsx`, ejecutando el aviso de verdad;
    aquí solo queda el ancla de que existen.)

Tooltip-anchor: P1-PLAN-LOTE-155
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_REPO_ROOT = _BACKEND_ROOT.parent
_FRONT = _REPO_ROOT / "frontend"

_CABECERAS_DE_CUOTA = ("X-Coach-Quota-Limit", "X-Coach-Quota-Used", "X-Coach-Quota-Resets-At")


def _expose_headers_block() -> str:
    """El literal `expose_headers=[...]` del `CORSMiddleware`, tal cual."""
    src = _APP_PY.read_text(encoding="utf-8")
    inicio = src.find("expose_headers=[")
    assert inicio != -1, "No se encontró `expose_headers=[` en app.py"
    fin = src.find("]", inicio)
    assert fin > inicio
    return src[inicio:fin + 1]


def test_las_tres_cabeceras_de_la_cuota_del_coach_estan_publicadas():
    """Sin esto, el medidor del chat es inerte en la app nativa."""
    bloque = _expose_headers_block()
    for cabecera in _CABECERAS_DE_CUOTA:
        assert f'"{cabecera}"' in bloque, (
            f"{cabecera} la manda `verify_coach_quota` con cada 402 y el chat la LEE "
            f"(AgentPage.jsx). Fuera de `expose_headers` el navegador se la come en cualquier "
            f"contexto cross-origin — o sea, en toda la app nativa."
        )


def test_el_correlation_id_sigue_publicado():
    """La lista crece; lo que ya estaba no se cae por el camino."""
    assert '"X-Correlation-ID"' in _expose_headers_block()


def test_toda_cabecera_x_que_el_frontend_lea_esta_publicada():
    """El guard de la CLASE, no de estas tres.

    Escanea el frontend buscando `headers.get('X-...')` y exige que cada nombre esté en
    `expose_headers`. Así, la próxima cabecera personalizada que alguien añada y lea no puede
    repetir esta historia: el test la caza antes de que un teléfono la sufra.
    """
    if not _FRONT.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({_FRONT}).")

    patron = re.compile(r"""headers\.get\(\s*['"](X-[A-Za-z0-9-]+)['"]\s*\)""")
    leidas: dict[str, str] = {}
    for ruta in (_FRONT / "src").rglob("*.js*"):
        if "__tests__" in ruta.parts or "node_modules" in ruta.parts:
            continue
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for nombre in patron.findall(texto):
            leidas.setdefault(nombre, str(ruta.relative_to(_FRONT)))

    bloque = _expose_headers_block()
    sin_publicar = {n: d for n, d in leidas.items() if f'"{n}"' not in bloque}
    assert not sin_publicar, (
        "El frontend lee cabeceras que el backend no publica en `expose_headers`; en la app "
        f"nativa (todo cross-origin) leerá `null`: {sin_publicar}"
    )


def test_el_tope_del_coach_no_invita_a_comprar_dentro_de_la_app():
    """B, desde este lado: el copy de respaldo del 402 se bifurca por plataforma.

    El mensaje bueno («llegaste a tus 60 mensajes, se renueva el…») ya respetaba el gate; el
    de respaldo no, y un respaldo es exactamente lo que se ve cuando algo falla.
    """
    agente = _FRONT / "src" / "pages" / "AgentPage.jsx"
    if not agente.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({agente}).")
    src = agente.read_text(encoding="utf-8")
    bloque = src[src.index("    402: {"):src.index("    409: {")]
    assert "nativeHidesCommerce()" in bloque, (
        "El copy del 402 del coach tiene que bifurcarse por plataforma: dentro de la app no "
        "existe ninguna superficie de comercio (Apple 3.1.1, P1-IOS-NATIVE-SHELL)."
    )
    assert "Llegaste al límite mensual de mensajes con el coach." in bloque


def test_cada_codigo_de_error_del_stream_tiene_fila_en_el_cliente():
    """D · el `code` que este backend clasifica tiene que significar algo al otro lado.

    `_chat_stream_error_payload` distingue `rate_limited`, `unavailable`, `timeout` e
    `internal` desde el 14-sep para no filtrar el detalle interno y aun así decir QUÉ pasó.
    El chat los recibía y escribía `status: 500` a pelo: los cuatro salían con el mismo
    «tuvo un problema, puedes reintentar». El caso que duele es `unavailable` —el
    cortacircuitos—, donde el copy correcto pide ESPERAR y el genérico invita a martillear
    al proveedor que está caído.

      *Clasificar sin que nadie lea la clasificación es no clasificar.*

    Esta paridad se comprueba desde aquí, que es el único sitio que ve los dos árboles: si
    alguien añade un quinto código, esto lo acusa antes de que salga como genérico.
    """
    agente_py = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
    inicio = agente_py.find("def _chat_stream_error_payload")
    assert inicio != -1
    cuerpo = agente_py[inicio:agente_py.find("\n\n\n", inicio)]
    codigos = set(re.findall(r'code,\s*msg\s*=\s*"([a-z_]+)"', cuerpo))
    assert codigos, "No se pudo extraer ningún código de `_chat_stream_error_payload`."

    agente_jsx = _FRONT / "src" / "pages" / "AgentPage.jsx"
    if not agente_jsx.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({agente_jsx}).")
    src = agente_jsx.read_text(encoding="utf-8")
    desde = src.find("const ESTADO_POR_CODIGO_DE_ERROR = {")
    assert desde != -1, "El chat perdió la tabla `code` → estado HTTP."
    tabla = src[desde:src.find("};", desde)]

    faltan = sorted(c for c in codigos if f"{c}:" not in tabla)
    assert not faltan, (
        f"Códigos que este backend emite y el chat no sabe traducir (saldrían con el copy "
        f"genérico): {faltan}"
    )
    assert "ESTADO_POR_CODIGO_DE_ERROR[dataObj.code]" in src, (
        "La tabla existe pero la rama del evento `error` no la usa."
    )


def test_el_aviso_de_motor_viejo_existe_y_corre_antes_del_bundle():
    """C, desde este lado. Lo que HACE se mide ejecutándolo en `lote155.test.jsx`."""
    index = _FRONT / "index.html"
    if not index.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({index}).")
    html = index.read_text(encoding="utf-8")
    marca = html.find("aviso.id = 'mf-motor-viejo'")
    assert marca != -1, (
        "Desapareció el aviso de motor web viejo. Sin él, un WebView anterior a `color-mix()` "
        "(478 usos) pinta la app descolorida y el usuario solo puede reportar «no funciona»."
    )
    assert marca < html.find('<script type="module" src="/src/main.jsx">'), (
        "El aviso tiene que ir ANTES del módulo: si el motor no parsea el bundle, es lo único "
        "que el usuario llegará a ver."
    )
