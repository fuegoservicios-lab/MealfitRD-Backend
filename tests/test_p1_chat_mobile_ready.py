"""[P1-CHAT-MOBILE-READY · 2026-08-10] El chat del Agente en el teléfono.

Tres defectos que el dueño reportó como «se ve mal» y «salir es incómodo», más el
que encontró la auditoría y nadie había visto.

1. LA BURBUJA ILEGIBLE (`P1-CHAT-MOBILE-CONTRAST`). `.msg-bubble-user` fijaba el
   FONDO con `!important` y NO el color. El color lo pone el inline de
   `MessageBubble` (`var(--text-main)`), que en tema oscuro es #F1F5F9: casi blanco
   sobre lavanda casi blanco, **1,0:1** cuando WCAG pide 4,5:1. Y solo pasaba en el
   móvil porque esa clase NO EXISTE fuera de `@media (max-width: 1024px)`.
   *Color y fondo son un PAR*: fijar uno con `!important` y dejar que el otro lo
   herede del tema es fabricar una combinación que nadie eligió.

2. LA NAVEGACIÓN QUE DESAPARECE (`P1-CHAT-TABBAR-BACK`). La barra de pestañas se
   renderizaba con `{!noPaddingMobile && !isSettings && ...}`. `noPaddingMobile` es
   una bandera de RELLENO (su único trabajo declarado es `padding: 0`) y estaba
   decidiendo de rebote si el usuario tiene navegación. El chat era la única
   sección del dashboard sin barra: entrar 1 toque, salir 2.

3. EL TURNO QUE NO EXISTÍA (`P1-CHAT-TURN-ACTIVE`) — el más grave, y no reportado.
   Vigilado en `test_p1_chat_stop_power.py`.

Parser-based sobre el source: este chat son 3.700 líneas con un `<style>` JSX
gigante, y estas reglas viven en literales que ningún test de render alcanza.
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_FRONT = os.path.join(_ROOT, "frontend", "src")


def _read(*partes: str) -> str:
    ruta = os.path.join(_FRONT, *partes)
    assert os.path.exists(ruta), f"no existe {ruta} — ¿se renombró sin actualizar el test?"
    with open(ruta, encoding="utf-8") as f:
        return f.read()


_AP = _read("pages", "AgentPage.jsx")
_DL = _read("components", "dashboard", "DashboardLayout.jsx")


def _bloque_movil() -> str:
    """El <style> de AgentPage donde viven las reglas móviles del chat.

    Se toma el bloque ENTERO y no una ventana desde `@media (max-width: 1024px)`:
    esa cadena aparece también dentro de comentarios que explican el defecto, así
    que recortar desde su primera aparición devolvía texto que no contiene las
    reglas — un guard que se mide a sí mismo. Lo detectó su propia aserción
    anti-vacío, que es exactamente para lo que está."""
    i = _AP.rfind("<style>{")
    assert i > 0, "desapareció el bloque <style> del chat"
    j = _AP.find("</style>", i)
    assert j > i, "bloque <style> sin cierre"
    return _AP[i:j]


# --------------------------------------------------------------------------
# 1. Contraste
# --------------------------------------------------------------------------
def test_toda_burbuja_que_fija_fondo_fija_tambien_color():
    """La regla general, no solo el caso de la captura: si una regla pinta el fondo
    de una burbuja con `!important`, tiene que traer su color. Sin eso, el color
    sale del tema y aparece una pareja que nadie eligió."""
    bloque = _bloque_movil()
    reglas = re.findall(r"(\.msg-bubble-[a-z]+\s*\{[^}]*\})", bloque)
    # Un bucle sobre cero elementos SIEMPRE pasa — el repo ya pagó esa lección
    # (P1-MOBILE-FIT). Se afirma primero que hay algo que mirar.
    assert len(reglas) >= 2, f"se esperaban las reglas de burbuja; encontradas {len(reglas)}"
    revisadas = 0
    for regla in reglas:
        fondo = re.search(r"background:([^;]*)!important", regla)
        if not fondo:
            continue
        # Un fondo `transparent` NO forma pareja: el texto queda sobre el fondo de la
        # página, que el tema ya resuelve. La pareja rota aparece cuando la regla
        # PINTA algo debajo del texto y deja el texto al tema.
        if "transparent" in fondo.group(1) or "none" in fondo.group(1):
            continue
        revisadas += 1
        assert re.search(r"(?<!-)\bcolor:\s*[^;]+!important", regla), (
            "esta regla pinta un fondo con !important y deja el color al tema — "
            f"exactamente el defecto de la burbuja ilegible:\n{regla.strip()[:220]}"
        )
    assert revisadas >= 1, (
        "ninguna regla de burbuja pinta fondo: o cambió el diseño, o este guard dejó "
        "de mirar donde importa"
    )


def test_la_burbuja_del_usuario_tiene_pareja_en_oscuro():
    bloque = _bloque_movil()
    assert re.search(r'html\[data-theme="dark"\]\s*\.msg-bubble-user\s*\{', bloque), (
        "sin variante oscura, la burbuja clara queda incrustada en un chat negro"
    )
    m = re.search(r'html\[data-theme="dark"\]\s*\.msg-bubble-user\s*\{([^}]*)\}', bloque)
    assert "background" in m.group(1) and "color" in m.group(1), (
        "la variante oscura también debe traer el PAR completo"
    )


def test_el_texto_del_usuario_no_se_pinta_de_blanco_puro_sobre_lavanda():
    """Ancla el caso exacto de la captura: fondo lavanda ⇒ tinta oscura."""
    m = re.search(r"\.msg-bubble-user\s*\{([^}]*)\}", _bloque_movil())
    assert m, "desapareció la regla de la burbuja del usuario"
    cuerpo = m.group(1)
    assert "#EEF2FF" in cuerpo, "cambió el fondo: revisa que el color siga haciendo pareja"
    color = re.search(r"(?<!-)\bcolor:\s*(#[0-9A-Fa-f]{6})", cuerpo)
    assert color, "el color del texto debe ser explícito, no heredado del tema"
    r, g, b = (int(color.group(1)[i:i + 2], 16) for i in (1, 3, 5))
    # Luminancia relativa (WCAG). El fondo mide ~0.80; con tinta oscura el ratio
    # supera de sobra 4,5:1, así que basta con exigir que sea oscura de verdad.
    def _c(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * _c(r) + 0.7152 * _c(g) + 0.0722 * _c(b)
    ratio = (0.80 + 0.05) / (lum + 0.05)
    assert ratio >= 4.5, f"contraste insuficiente sobre el lavanda: {ratio:.1f}:1"


# --------------------------------------------------------------------------
# 2. Navegación
# --------------------------------------------------------------------------
def test_la_barra_de_pestanas_no_depende_de_una_bandera_de_relleno():
    m = re.search(r"\{([^}]*)<BottomTabBar\s*/>\}", _DL)
    assert m, "no se encontró el render de BottomTabBar"
    cond = m.group(1)
    assert "noPaddingMobile" not in cond, (
        "la navegación volvió a colgar de una bandera de PADDING: el chat se queda "
        "sin barra de pestañas y salir pasa de 1 toque a 2"
    )
    assert "isSettings" in cond, (
        "Settings SÍ tiene razón escrita para ir sin barra (página standalone) — "
        "ese gate se conserva"
    )


def test_el_chat_reserva_el_alto_de_la_barra():
    """Devolver la barra sin reservar su alto la deja TAPANDO la caja de escribir —
    un defecto peor que el que arregla. Por eso van juntos."""
    bloque = _bloque_movil()
    m = re.search(r"\.input-wrapper\s*\{([^}]*)\}", bloque)
    assert m, "desapareció la regla de la barra de entrada"
    pad = re.search(r"padding:[^;]*", m.group(1))
    assert pad and "64px" in pad.group(0), (
        "la barra de entrada debe reservar los 64px de la barra de pestañas"
    )
    assert "env(safe-area-inset-bottom" in pad.group(0), (
        "y el safe-area del iPhone, que la propia barra también añade"
    )
    m_sb = re.search(r"\.agent-sidebar\s*\{([^}]*)\}", bloque)
    assert m_sb and "64px" in m_sb.group(1), (
        "el cajón de conversaciones llega al borde inferior: sin la misma reserva, "
        "sus últimas filas quedan debajo de la barra y no se pueden tocar"
    )


# --------------------------------------------------------------------------
# 3. El hueco vertical
# --------------------------------------------------------------------------
def test_los_mensajes_se_apilan_desde_abajo_en_movil():
    """Con un solo turno quedaban ~700px de vacío entre lo último dicho y la caja de
    escribir: el contenedor repartía ARRIBA todo el alto sobrante."""
    bloque = _bloque_movil()
    assert re.search(r"\.msg-log\s*\{[^}]*margin-top:\s*auto", bloque), (
        "la lista de mensajes debe anclarse abajo en móvil"
    )
    assert 'className="msg-log"' in _AP, "el contenedor del log debe llevar la clase"
    # El anclaje va en el HIJO: con justify-content:flex-end en el contenedor, al
    # desbordar el contenido el principio del scroll queda inalcanzable.
    i = bloque.find(".messages-container")
    assert "justify-content: flex-end" not in bloque[i:i + 400], (
        "anclar el CONTENEDOR con flex-end deja el inicio del scroll inalcanzable "
        "cuando el contenido desborda"
    )


# --------------------------------------------------------------------------
# 4. Zonas táctiles
# --------------------------------------------------------------------------
def test_los_botones_de_la_cabecera_se_pueden_tocar():
    """24px de icono + 2×6,4 de relleno daban 36,8px. El propio repo se impuso 44
    por escrito en BottomTabBar.module.css."""
    for icono, etiqueta in (("<History size={24}", "historial"), ("<Menu size={24}", "menú")):
        i = _AP.find(icono)
        assert i > 0, f"desapareció el botón de {etiqueta}"
        win = _AP[max(0, i - 1200):i]
        assert "padding: '0.625rem'" in win, (
            f"el botón de {etiqueta} volvió por debajo de 44px de zona táctil"
        )
        assert "aria-label=" in win, (
            f"el botón de {etiqueta} no tiene nombre accesible: lucide marca su svg "
            "como aria-hidden, así que un lector de pantalla no anuncia nada"
        )
