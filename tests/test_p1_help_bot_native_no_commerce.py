"""[P1-HELP-BOT-NATIVE-NO-COMMERCE · 2026-08-22] El bot de ayuda («Obtener ayuda») se
presentaba con «planes y precios» y sugería «¿Qué incluye cada plan y cuánto cuesta?»
también dentro de la app nativa de iPhone — y contestaba precios si se lo pedían. Apple
3.1.1/3.1.3(b): en la app nativa no puede haber comercio ni ENLACES a él. Un chatbot que
recita precios es comercio. Es la misma familia que P1-IOS-NATIVE-SHELL-2 (las CTAs):
un revisor no navega por URL, pulsa — o pregunta.

Contrato en DOS capas (la del cliente puede fallar si el gate no se aplica; la del
servidor es la que manda):
  - Cliente: en nativo, saludo sin «planes y precios» y sin la sugerencia de precios.
  - Servidor: `hide_commerce: true` en el body ⇒ el prompt de sistema lleva la directiva
    de no citar precios/planes/suscripción y remitir a la web SIN cifras.
Y la cabecera de la hoja móvil reserva el notch (pisaba el reloj, medido en el iPhone).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_FRONT = _REPO / "frontend" / "src" / "components" / "dashboard"


def test_el_prompt_lleva_la_directiva_sin_comercio_cuando_se_pide():
    from prompts.help_bot import help_bot_system_prompt

    con = help_bot_system_prompt("es-DO", hide_commerce=True)
    sin = help_bot_system_prompt("es-DO")
    assert "NO menciones precios" in con
    assert "NO menciones precios" not in sin
    # compat: sin el flag el prompt es byte-idéntico al de siempre
    assert sin == help_bot_system_prompt("es-DO", hide_commerce=False)


def test_el_endpoint_lee_hide_commerce_del_body():
    src = (_REPO / "backend" / "routers" / "help_chat.py").read_text(encoding="utf-8")
    assert re.search(r"hide_commerce\s*=\s*bool\(\(data or \{\}\)\.get\(['\"]hide_commerce['\"]\)\)", src)
    assert "help_bot_system_prompt(" in src and "hide_commerce=hide_commerce" in src


def test_el_widget_gatea_saludo_sugerencias_y_manda_el_flag():
    src = (_FRONT / "HelpChatWidget.jsx").read_text(encoding="utf-8")
    assert "nativeHidesCommerce" in src.split("\n", 12)[0:12].__str__() or "from '../../config/platform'" in src
    assert "hide_commerce: nativeHidesCommerce()" in src
    # la sugerencia de precios solo existe fuera de nativo
    i = src.index("¿Qué incluye cada plan y cuánto cuesta?")
    assert "nativeHidesCommerce()" in src[max(0, i - 300):i]


def test_la_hoja_movil_reserva_el_notch():
    css = (_FRONT / "HelpChatWidget.module.css").read_text(encoding="utf-8")
    movil = css[css.index("@media (max-width: 640px)"):]
    assert re.search(r"\.header\s*\{[^}]*env\(safe-area-inset-top", movil), "la cabecera móvil debe sumar safe-area-inset-top"
    assert re.search(r"env\(safe-area-inset-bottom", movil), "el pie móvil debe sumar safe-area-inset-bottom"


def test_la_hoja_movil_sigue_al_teclado():
    """[P1-HELP-BOT-KEYBOARD · 2026-08-22] En iOS el teclado no encoge el layout viewport;
    una hoja fixed a 100dvh deja el campo de texto debajo del teclado. El panel sigue al
    visualViewport (alto + offsetTop) mientras hay teclado, como la hoja de la Nevera."""
    src = (_FRONT / "HelpChatWidget.jsx").read_text(encoding="utf-8")
    assert "window.visualViewport" in src
    assert re.search(r"style=\{vvBox \? \{ top: vvBox\.top, bottom: 'auto', height: vvBox\.height \}", src)
