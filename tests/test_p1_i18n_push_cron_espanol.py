"""[P1-I18N-PUSH-CRON-ESPANOL · 2026-08-22] Las notificaciones push salian en espanol duro.

`P2-I18N-PUSH-SIN-LOCALE` (2026-08-21) tradujo UNA --el nudge del coach-- y su guard solo
abria `proactive_agent.py`. El resto salia con titulo Y cuerpo en espanol desde
`cron_tasks.py` y `routers/plans.py`. MEDIDO con AST sobre los call sites: 25 titulos y 18
cuerpos literales distintos, mas 26 dinamicos.

Un usuario con la app en ingles recibia en la pantalla de bloqueo «Tu plan necesita una
revision — Detectamos ingredientes que ya no estan en tu nevera…». Es la superficie MENOS
perdonable: llega sin que la pidas, se lee de un vistazo y no hay donde cambiar el idioma.

POR QUE SE TRADUCE EN EL CUELLO DE BOTELLA. `utils_push.send_push_notification` es el punto
por el que pasa TODO push (`_dispatch_push_notification` es su envoltorio). Traducir ahi ata
la invariante al ACTO en vez de a 35 llamadas que hay que acordarse de tocar -- la leccion
que ya pagaron `P2-DISPLAY-POP-VECINO` (el pop colgaba de siete funciones con nombre) y
`P1-COUNTRY-SYSTEM-F1` («gatear call sites uno a uno es el agujero, no el cierre»).

LO QUE ESTE GUARD MIDE, y por que NO mira los call sites uno a uno: comprueba (a) que la
traduccion ocurre en el cuello de botella y no en un envoltorio, (b) que la CONDUCTA es
correcta para los cinco idiomas, y (c) que el catalogo no se queda atras de los literales
que los call sites emiten. Lo tercero es lo que hace que anadir un push nuevo sin traducir
salga a la cara -- porque la clave ES el texto espanol, y cambiar el copy huerfana su
traduccion EN SILENCIO.

tooltip-anchor: P1-I18N-PUSH-CRON-ESPANOL
"""
from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

from push_i18n import push_catalog_keys, translate_push_text

_BACKEND = Path(__file__).resolve().parent.parent
_MARKER = "P1-I18N-PUSH-CRON-ESPANOL"
_LOCALES_NO_ES = ("en-US", "pt-BR", "fr-FR", "it-IT")

_FUNCIONES_DE_PUSH = {"_dispatch_push_notification", "send_push_notification"}


# ---------------------------------------------------------------------------
# 1 · La conducta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("locale", _LOCALES_NO_ES)
def test_un_titulo_del_catalogo_sale_traducido(locale: str) -> None:
    original = "Tu plan está en pausa"
    salida = translate_push_text(original, locale)
    assert salida != original, (
        f"«{original}» sale en español con locale={locale}. [{_MARKER}]"
    )
    assert salida.strip(), f"traducción vacía para {locale} [{_MARKER}]"


def test_es_DO_no_toca_el_texto() -> None:
    """El español es la CLAVE y no lleva catálogo: tiene que salir idéntico."""
    original = "Tu plan está en pausa"
    assert translate_push_text(original, "es-DO") == original


@pytest.mark.parametrize("locale_raro", [None, "", "xx-YY", 42, {"a": 1}])
def test_un_locale_invalido_degrada_al_espanol_sin_lanzar(locale_raro) -> None:
    """Fail-open TOTAL. Una notificación en español es una degradación; una que no sale
    —o que sale con una clave técnica en la pantalla de bloqueo— es un fallo."""
    original = "Tu plan está en pausa"
    assert translate_push_text(original, locale_raro) == original


@pytest.mark.parametrize("texto_raro", [None, "", 0, [], {"a": 1}])
def test_una_entrada_no_texto_vuelve_tal_cual(texto_raro) -> None:
    assert translate_push_text(texto_raro, "en-US") == texto_raro


def test_un_texto_fuera_del_catalogo_cae_al_espanol() -> None:
    """Los 26 mensajes dinámicos (f-strings con cifras) viven aquí. Es conducta
    DECLARADA, no fallo: traducir una plantilla compuesta en el call site exigiría
    reestructurar cada uno."""
    suelto = "Te quedan 3 días y 400 kcal por registrar."
    assert translate_push_text(suelto, "fr-FR") == suelto


# ---------------------------------------------------------------------------
# 2 · Dónde vive la traducción
# ---------------------------------------------------------------------------

def test_la_traduccion_vive_en_el_cuello_de_botella() -> None:
    """Si mañana alguien la mueve a `_dispatch_push_notification`, los pushes que llamen
    directo a `send_push_notification` vuelven a salir en español — y ésa es exactamente la
    forma del gap que este P-fix cierra."""
    src = io.open(_BACKEND / "utils_push.py", encoding="utf-8").read()
    assert "translate_push_text" in src, (
        f"`utils_push.send_push_notification` ya no traduce. Es el ÚNICO punto por el que "
        f"pasa todo push; moverlo a un envoltorio deja fuera a quien llame directo. "
        f"[{_MARKER}]"
    )
    assert "user_profiles" in src and "locale" in src, (
        f"`utils_push` ya no resuelve el locale del usuario [{_MARKER}]"
    )


def test_la_resolucion_del_locale_es_best_effort() -> None:
    """Un fallo consultando el perfil no puede impedir que la notificación salga."""
    src = io.open(_BACKEND / "utils_push.py", encoding="utf-8").read()
    i = src.find("SELECT locale FROM user_profiles")
    assert i > 0, f"no encontré la consulta del locale [{_MARKER}]"
    ventana = src[max(0, i - 400): i + 600]
    assert "except" in ventana, (
        f"la consulta del locale no está protegida: si falla, el push no sale. "
        f"[{_MARKER}]"
    )


# ---------------------------------------------------------------------------
# 3 · El catálogo no se queda atrás de los call sites
# ---------------------------------------------------------------------------

def _literales_de_los_call_sites() -> set:
    """Los `title=`/`body=` LITERALES de todos los call sites de push, vía AST.

    Se usa AST y no grep porque el nombre de la función puede llegar por alias de import
    (`from utils_push import send_push_notification as _p`), y un selector textual no lo
    ve — que es exactamente por qué el guard anterior sólo vigilaba UNA notificación.
    """
    fuera = set()
    ficheros = list(_BACKEND.glob("*.py")) + list((_BACKEND / "routers").glob("*.py"))
    for p in ficheros:
        if p.name in ("push_i18n.py",):
            continue
        try:
            arbol = ast.parse(io.open(p, encoding="utf-8").read())
        except Exception:  # noqa: BLE001
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            nombre = getattr(nodo.func, "id", None) or getattr(nodo.func, "attr", None)

            # [P1-I18N-PUSH-GUARD-CIEGO-AL-THREAD · 2026-08-23] La forma envuelta.
            #
            # Ocho call sites no llaman al push directamente: son
            # `threading.Thread(target=send_push_notification, kwargs={"title": …})`, donde
            # el nodo `Call` se llama `Thread` y no `send_push_notification`. El guard veía
            # 41 literales, los comparaba con el catálogo y reportaba CERO faltantes —
            # mientras 16 textos (8 títulos + 8 cuerpos) salían en español. Entre ellos,
            # palabra por palabra, el ejemplo que el propio P-fix usó para justificarse:
            # «Tu plan necesita una revisión».
            #
            # El plan v2 pedía explícitamente resolver esta forma («resolver alias de import
            # y `threading.Thread(kwargs={'title':…})`») y no se hizo.
            if nombre == "Thread":
                kw_t = {k.arg: k.value for k in nodo.keywords if k.arg}
                objetivo = kw_t.get("target")
                nombre_objetivo = (getattr(objetivo, "id", None)
                                   or getattr(objetivo, "attr", None))
                if nombre_objetivo not in _FUNCIONES_DE_PUSH:
                    continue
                candidatos = []
                # `kwargs={"title": …, "body": …}`
                dic = kw_t.get("kwargs")
                if isinstance(dic, ast.Dict):
                    for clave, valor in zip(dic.keys, dic.values):
                        if isinstance(clave, ast.Constant) and clave.value in ("title", "body"):
                            candidatos.append(valor)
                # `args=(user_id, title, body, url)`
                tup = kw_t.get("args")
                if isinstance(tup, (ast.Tuple, ast.List)):
                    candidatos += list(tup.elts[1:3])
                for c in candidatos:
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.strip():
                        fuera.add(c.value)
                continue

            if nombre not in _FUNCIONES_DE_PUSH:
                continue
            kw = {k.arg: k.value for k in nodo.keywords if k.arg}
            candidatos = [kw.get("title"), kw.get("body")]
            candidatos += list(nodo.args[1:3])
            for c in candidatos:
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.strip():
                    fuera.add(c.value)
    return fuera


def test_todo_literal_que_emiten_los_call_sites_esta_en_el_catalogo() -> None:
    """LA invariante que hace que un push nuevo sin traducir salga a la cara.

    La clave ES el texto español, así que cambiar el copy en el call site huérfana su
    traducción EN SILENCIO: el push sale en español y nadie se entera. Esto es lo único que
    lo convierte en un rojo.
    """
    literales = _literales_de_los_call_sites()
    if not literales:
        pytest.skip("no se encontró ningún call site literal (¿refactor?)")

    # [P1-I18N-PUSH-GUARD-CIEGO-AL-THREAD · 2026-08-23] El ALCANCE del extractor, aseverado
    # en positivo.
    #
    # Sin esto, el guard es inmune a su propio agujero: si el extractor deja de ver una
    # forma, encuentra MENOS literales, y «menos literales, todos en el catálogo» sale
    # verde. Lo comprobé mutando —cegando la rama de `Thread`— y el test pasó igual. Un
    # guard cuyo universo puede encogerse en silencio no vigila el universo, vigila lo que
    # le queda.
    #
    # `Renovación pausada` vive SÓLO dentro de un `threading.Thread(kwargs={...})`
    # (`cron_tasks.py`), así que su presencia demuestra que esa forma se alcanza.
    assert "Renovación pausada" in literales, (
        "el extractor dejó de ver los push envueltos en `threading.Thread(target=…, "
        f"kwargs={{'title': …}})`. Encuentra {len(literales)} literales y le falta al menos "
        f"uno que sólo existe en esa forma: el guard volvió a ser ciego justo donde lo era. "
        f"[{_MARKER}]"
    )
    faltan = sorted(literales - push_catalog_keys())
    assert not faltan, (
        f"{len(faltan)} texto(s) de push no están en `push_i18n`: {faltan[:6]}"
        f"{'…' if len(faltan) > 6 else ''}. Saldrán en español en los cuatro idiomas. Si es "
        f"copy nuevo, añádelo al catálogo; si cambiaste el copy, su traducción quedó "
        f"huérfana. [{_MARKER}]"
    )


@pytest.mark.parametrize("locale", _LOCALES_NO_ES)
def test_el_catalogo_esta_completo_en_los_cuatro_idiomas(locale: str) -> None:
    """Media traducción es peor que ninguna: la pantalla de bloqueo mezclaría idiomas."""
    incompletas = [k for k in push_catalog_keys() if translate_push_text(k, locale) == k]
    assert not incompletas, (
        f"{len(incompletas)} clave(s) del catálogo de push no tienen {locale}: "
        f"{sorted(incompletas)[:5]}. [{_MARKER}]"
    )
