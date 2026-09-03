"""[P3-I18N-SW-PUSH-FALLBACK · 2026-08-23] La refutación era correcta. Este guard la sostiene.

EL GAP, Y POR QUÉ NO SE CIERRA TRADUCIENDO

«Los tres textos de reserva de la notificación push están en español y viven donde `t()` no
existe.» Una dimensión de la auditoría lo reportó; otra lo refutó por inalcanzable, y el plan
lo dejó como nota sin decidir.

MEDIDO: la refutación es correcta. Los fallbacks del service worker

    const title = data.title || "Tu coach nutricional IA";
    const body  = data.body  || "Tienes un nuevo mensaje.";

sólo se pintan si el payload llega SIN esos campos. Y no llega: los dos emisores del backend
los pasan siempre — `_dispatch_push_notification` en `cron_tasks.py` (**34 llamadas reales,
verificadas por AST, ninguna omite `title` ni `body`**) y `send_push_notification` desde
`proactive_agent.py`, cuya firma los exige como posicionales.

Traducir esas cadenas habría sido trabajo sobre código muerto — y encima imposible de hacer
bien, porque el service worker corre fuera de React sin motor de i18n. El tercer literal, el
del `catch` de un payload que no parsea, ya lleva su `[I18N-EXEMPT]` con esa razón escrita.

LO QUE SÍ HACE FALTA

Que la refutación siga siendo cierta. Es inalcanzable **hoy**, por una propiedad de los
emisores que nada defendía: el día que alguien añada la llamada 35 sin `title`, el usuario ve
español en su pantalla de bloqueo en los cinco idiomas — y en la pantalla de bloqueo el
título es lo único que se lee de un vistazo, o sea la parte que decide si abre la app.

Un «refutado por inalcanzable» sin guard es una refutación con fecha de caducidad que nadie
apuntó.

tooltip-anchor: P3-I18N-SW-PUSH-FALLBACK
"""
from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MARKER = "P3-I18N-SW-PUSH-FALLBACK"
_CRON = _BACKEND / "cron_tasks.py"
_UTILS = _BACKEND / "utils_push.py"
_SW = _BACKEND.parent / "frontend" / "src" / "custom-sw.js"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _llamadas_sin_copy(src: str, nombre: str) -> list[int]:
    """Líneas donde `nombre(...)` se llama sin `title` o sin `body`.

    Se usa AST y no texto porque estas llamadas se reparten en 34 sitios con estilos
    distintos (posicional, keyword, multilínea): un regex acertaría en unos y fallaría en
    otros, y un guard que sólo ve la mitad de sus sujetos es peor que ninguno.
    """
    fallan = []
    for nodo in ast.walk(ast.parse(src)):
        if not isinstance(nodo, ast.Call):
            continue
        if getattr(nodo.func, "id", None) != nombre:
            continue
        kw = {k.arg for k in nodo.keywords}
        pos = len(nodo.args)
        # Firma: (user_id, title, body, url=...). `title` es el 2.º, `body` el 3.º.
        if not ((pos >= 2 or "title" in kw) and (pos >= 3 or "body" in kw)):
            fallan.append(nodo.lineno)
    return fallan


def test_ninguna_emision_de_push_del_cron_omite_titulo_ni_cuerpo():
    src = _leer(_CRON)
    fallan = _llamadas_sin_copy(src, "_dispatch_push_notification")
    assert not fallan, (
        f"Estas llamadas a `_dispatch_push_notification` no pasan `title` o `body` "
        f"(líneas {fallan}). El service worker cae entonces a sus literales en ESPAÑOL, y "
        "corre fuera de React: ahí no hay `t()` que valga. En la pantalla de bloqueo el "
        "título es lo único que se lee de un vistazo — es la parte que decide si el usuario "
        f"abre la app [{_MARKER}]"
    )


def test_hay_emisiones_de_verdad_que_vigilar():
    """Un guard cuyo conjunto de sujetos se vacía pasa a ser decorativo, y eso no se nota:
    seguiría en verde para siempre. Se ancla que sigue habiendo llamadas que revisar."""
    src = _leer(_CRON)
    n = sum(
        1 for nodo in ast.walk(ast.parse(src))
        if isinstance(nodo, ast.Call)
        and getattr(nodo.func, "id", None) == "_dispatch_push_notification"
    )
    assert n >= 20, (
        f"sólo encuentro {n} emisiones de push en el cron (medido 2026-08-23: 34). Si el "
        f"dispatcher se renombró, este guard dejó de vigilar nada [{_MARKER}]"
    )


def test_el_emisor_directo_exige_titulo_y_cuerpo_en_su_firma():
    """`send_push_notification` los pide como posicionales sin default: es imposible
    llamarla sin ellos, y esa es la defensa más fuerte — la que no depende de un test."""
    src = _leer(_UTILS)
    firma = [l for l in src.splitlines() if l.startswith("def send_push_notification(")]
    assert firma, f"desapareció `send_push_notification` [{_MARKER}]"
    assert "title: str" in firma[0] and "body: str" in firma[0], (
        "`title`/`body` dejaron de ser obligatorios en la firma. Con un default se puede "
        f"llamar sin ellos y el fallback español pasa a ser alcanzable [{_MARKER}]"
    )
    assert "title: str = " not in firma[0] and "body: str = " not in firma[0], (
        f"`title` o `body` ganaron un valor por defecto [{_MARKER}]"
    )


def test_el_fallback_del_service_worker_sigue_declarado_como_tal():
    """No se traduce —el SW corre sin motor de i18n— pero tiene que seguir siendo evidente
    que es un fallback y no copy normal, o alguien lo tomará por texto vivo sin traducir."""
    if not _SW.exists():
        # Convención del repo: `frontend/` es un repo HERMANO. Falta en un checkout del
        # backend solo y en un worktree, y ahí el test no tiene sujeto — saltar es honesto;
        # inventarse una ruta alternativa sería medir otro fichero y llamarlo el mismo.
        pytest.skip(f"no existe {_SW} (¿repo hermano sin clonar?)")
    sw = _leer(_SW)
    assert 'data.title || "Tu coach nutricional IA"' in sw, (
        f"cambió el fallback del título del service worker [{_MARKER}]"
    )
    assert "I18N-EXEMPT" in sw, (
        "desapareció el marcador que explica por qué estos literales no pasan por `t()`. "
        "Sin él, el próximo barrido de i18n los cuenta como deuda y alguien intenta "
        f"traducirlos donde no hay motor que los resuelva [{_MARKER}]"
    )
