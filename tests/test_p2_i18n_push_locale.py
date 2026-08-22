"""[P2-I18N-PUSH-SIN-LOCALE · 2026-08-21] El nudge del coach llegaba BILINGÜE.

El cuerpo lo escribe el LLM bajo `build_language_directive(_nudge_locale)`, así que sigue
el idioma del usuario. El título era un literal español pegado tres líneas más abajo, en
el call site. Resultado: una notificación con **el título en español y el cuerpo en
francés**.

Y en una notificación eso duele más que en una pantalla: en el bloqueo del móvil el
título es lo único que se lee de un vistazo, así que la mitad que no se traducía era
justamente la que decide si el usuario abre.

LA CIFRA DEL PLAN ESTABA MAL Y SE CORRIGE AQUÍ. Decía «36 de 44 call sites pasan un
`title=` literal español». Medido en este checkout: `send_push_notification` tiene **UN**
call site. Los 47 `title=` que salen por grep son en su mayoría otra cosa — kwargs de
gráficas, de PDF, de tests. El defecto es real y es exactamente el descrito; el alcance
era una línea, no treinta y seis. Se anota porque una cifra inflada en un plan hace que
el siguiente lector busque un problema que no está.

EL GUARD mira el call site, no el helper: `send_push_notification` seguirá recibiendo un
`title` string —no tiene por qué saber de idiomas— y lo que no puede pasar es que ese
string nazca literal.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

_MARKER = "P2-I18N-PUSH-SIN-LOCALE"

# Un call site puede eximirse si su título de verdad no depende del idioma. Necesita
# razón: una whitelist sin motivo es indistinguible de un olvido.
_EXENTOS: dict[str, str] = {}


def _ficheros_con_push() -> list[Path]:
    fuera = []
    for p in _BACKEND.rglob("*.py"):
        if "tests" in p.parts or ".venv" in p.parts:
            continue
        try:
            s = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "send_push_notification(" in s and p.name != "utils_push.py":
            fuera.append(p)
    return fuera


# ============================================================
# 1 · El mapa de títulos, junto a su hermano
# ============================================================

def test_el_titulo_se_traduce_a_los_cuatro_idiomas() -> None:
    from prompts.chat_agent import push_nudge_title

    es = push_nudge_title("es-DO")
    for locale in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        assert push_nudge_title(locale) != es, (
            f"el título del nudge en {locale} es idéntico al español. La notificación "
            f"llega con el título en un idioma y el cuerpo en otro. [{_MARKER}]"
        )


@pytest.mark.parametrize("basura", ["de-DE", "", None, 42, "es-DO"])
def test_un_locale_desconocido_cae_al_espanol(basura) -> None:
    """Fallback = conducta de hoy. Un `KeyError` aquí tumbaría el envío entero del nudge
    por no saber traducir un título."""
    from prompts.chat_agent import push_nudge_title, _PUSH_NUDGE_TITLE_ES

    assert push_nudge_title(basura) == _PUSH_NUDGE_TITLE_ES


def test_el_mapa_vive_junto_a_los_nombres_de_idioma_del_coach() -> None:
    """Es el mismo hecho —«en qué idioma le hablamos a este usuario»— y separarlo es
    como acaban divergiendo dos tablas: la lección de `P1-DIET-CANON-SSOT`."""
    src = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    i_nombres = src.index("_COACH_LANGUAGE_NAMES = {")
    i_titulos = src.index("_PUSH_NUDGE_TITLES = {")
    assert abs(i_titulos - i_nombres) < 3000, (
        f"`_PUSH_NUDGE_TITLES` se alejó de `_COACH_LANGUAGE_NAMES`. Son la misma "
        f"decisión y tienen que envejecer juntos. [{_MARKER}]"
    )


def test_los_titulos_cubren_los_mismos_idiomas_que_el_coach() -> None:
    from prompts.chat_agent import _COACH_LANGUAGE_NAMES, _PUSH_NUDGE_TITLES

    faltan = set(_COACH_LANGUAGE_NAMES) - set(_PUSH_NUDGE_TITLES)
    assert not faltan, (
        f"el coach habla {sorted(faltan)} pero su notificación no: el cuerpo llegaría "
        f"traducido y el título en español. [{_MARKER}]"
    )


# ============================================================
# 2 · Ningún call site pasa un título literal
# ============================================================

def test_ningun_call_site_pasa_un_titulo_literal() -> None:
    """Se mira el CALL SITE, no el helper: `send_push_notification` seguirá recibiendo un
    string —no tiene por qué saber de idiomas— y lo que no puede pasar es que ese string
    nazca literal.

    Con AST y no con regex: `title="…"` dentro de un comentario o de un docstring no es
    una llamada, y este repo ya se ha comido siete veces un guard derrotado por prosa.
    """
    culpables = []
    for p in _ficheros_con_push():
        rel = p.relative_to(_BACKEND).as_posix()
        if rel in _EXENTOS:
            continue
        arbol = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            nombre = getattr(nodo.func, "id", None) or getattr(nodo.func, "attr", None)
            if nombre != "send_push_notification":
                continue
            for kw in nodo.keywords:
                if kw.arg == "title" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    culpables.append(f"{rel}:{nodo.lineno}")

    assert not culpables, (
        "Estos envíos de notificación pasan un título LITERAL, así que llegan en español "
        f"aunque el cuerpo vaya traducido: {culpables}. En la pantalla de bloqueo el "
        f"título es lo único que se lee de un vistazo. Usa `push_nudge_title(locale)`. "
        f"[{_MARKER}]"
    )


def test_el_detector_veria_un_literal_de_verdad() -> None:
    """MUTACIÓN DE CONTROL. Un walker que no encuentre la llamada da verde pasando en
    vacío — el modo de fallo de `P1-CULINARY-METADATA-BETA`."""
    fuente = (
        "send_push_notification(user_id=u, title='Aviso literal', body=b, url='/x')\n"
    )
    arbol = ast.parse(fuente)
    encontrados = [
        kw for n in ast.walk(arbol) if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "send_push_notification"
        for kw in n.keywords
        if kw.arg == "title" and isinstance(kw.value, ast.Constant)
    ]
    assert len(encontrados) == 1, "el detector no ve un título literal evidente"


def test_hay_al_menos_un_call_site_que_analizar() -> None:
    """El otro control: si un renombre dejara `_ficheros_con_push()` vacío, el test de
    arriba pasaría sin mirar nada."""
    assert _ficheros_con_push(), (
        f"no encontré ningún fichero que llame a `send_push_notification`. ¿Se renombró? "
        f"[{_MARKER}]"
    )


def test_el_call_site_del_nudge_usa_el_locale_que_ya_tenia_delante() -> None:
    """`_nudge_locale` se resuelve unas líneas antes para la directiva del LLM. Que el
    título lo ignorara era lo que hacía la notificación bilingüe."""
    src = (_BACKEND / "proactive_agent.py").read_text(encoding="utf-8")
    assert "push_nudge_title(_nudge_locale)" in src, (
        f"el nudge no usa `_nudge_locale` para su título, teniéndolo resuelto unas "
        f"líneas antes para `build_language_directive`. [{_MARKER}]"
    )
    i_dir = src.index("build_language_directive(_nudge_locale)")
    i_push = src.index("push_nudge_title(_nudge_locale)")
    assert i_dir < i_push, "el orden esperado es: directiva del cuerpo, luego el envío"
