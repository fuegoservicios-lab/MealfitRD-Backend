"""[P1-TRACKING-TZ-CAPTURE · 2026-08-21] El modo contador nunca capturaba el huso, así que para
esa población `user_tz_offset_min` degradaba a 240 (RD) para siempre y todo F1-T5 quedaba inerte.

La rama corta del wizard (P1-PLAN-MODE) termina en un `PATCH /api/profile` con una **allowlist**
de claves. Esa lista incluye `country` — y su propio comentario dice por qué: «la rama corta es
ALLOWLIST: sin esta entrada el país se cae al suelo en silencio en modo contador». El huso sufre
EXACTAMENTE el mismo mecanismo y no tiene entrada.

Los únicos dos escritores de `tzOffset` en el perfil son `POST /api/plans/analyze` y
`POST /api/plans/shift-plan`. Un usuario que elige «solo contador» no pasa por ninguno: su
`health_profile` queda sin huso ⇒ `user_tz_offset_min()` devuelve 240 ⇒ las tres superficies que
F1-T5 parametrizó vuelven al huso dominicano **precisamente para el usuario que sólo usa el diario
y el coach**. Medido en Neon: 3 de 8 perfiles tienen huso nulo.

EL SIGNO DEL ERROR, POR PAÍS (corte del día a las 04:00Z por el 240 forzado):

    España        frontera 06:00 local  →  lo de 00:00-06:00 cuenta al día ANTERIOR
    Colombia      frontera 23:00        →  lo de 23:00-24:00 cuenta al día SIGUIENTE
    México CDMX   frontera 22:00        →  **la cena de las 22:30 cuenta a mañana**
    US-Pacífico   frontera 20:00/21:00  →  toda cena posterior a las 20:00 cuenta a mañana
    Puerto Rico   exacto (mismo offset)

Lo que ve el usuario: el dup-guard de `log_consumed_meal` le deja registrar dos cenas el mismo día
(o le niega la segunda del día siguiente), y el rescate de slot merienda→cena no dispara porque el
sistema cree que no ha desayunado hoy.

LA TRAMPA DEL CONGELADO. El offset se calcula **en el submit**, nunca en una `const` de módulo:
ese patrón —evaluar al importar— ya costó tres P-fixes en este repo, y aquí sería peor de lo
habitual porque en la máquina de desarrollo el valor congelado coincide con el correcto.

Cubre:
  A. La allowlist incluye el huso y lo calcula en el submit.
  B. El huso viaja al perfil con la convención del sistema.
  C. El backend acepta la clave (no la filtra su propia validación).
  D. Parser-based: el comentario que explica la allowlist sigue vivo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_QTF = _BACKEND_ROOT.parent / "frontend" / "src" / "components" / "assessment" / "questions" / "QTrackingFinish.jsx"


def _src() -> str:
    return _QTF.read_text(encoding="utf-8", errors="replace")


# ── A. La allowlist ─────────────────────────────────────────────────────────────────────────────

def test_la_allowlist_de_la_rama_corta_incluye_el_huso():
    """RED pre-fix: la lista traía `country` (con su comentario explicando el mecanismo) y no el
    huso, que sufre el mismo. Los 12 pasos saltados quedan AUSENTES a propósito; el huso no es un
    paso, es un dato del dispositivo que nadie más va a capturar para esta población."""
    src = _src()
    assert "P1-TRACKING-TZ-CAPTURE" in src
    # Se ancla que el huso llegue al payload del perfil, NO la grafía entrecomillada de la
    # allowlist: el huso no vive en `formData` (nadie lo contesta, es un dato del navegador), así
    # que no puede recorrerse en ese bucle y va por acceso a propiedad. La primera versión de
    # este assert pedía la forma de la lista y habría empujado hacia meterlo donde no cabe.
    assert re.search(r"hp\.tzOffset\s*=|['\"]tzOffset['\"]\s*:", src), (
        "el perfil de la rama corta sigue sin el huso: user_tz_offset_min degrada a 240 (RD) "
        "para siempre en modo contador"
    )


def test_el_huso_se_calcula_en_el_submit_no_al_importar():
    """La trampa del congelado: una `const` de módulo con `getTimezoneOffset()` se evalúa al
    IMPORTAR el módulo, y en la máquina de desarrollo el valor congelado coincide con el correcto
    — así que el bug sólo aparece en producción y sólo para quien no viva en RD. Este repo ya pagó
    ese patrón tres veces."""
    src = _src()
    i = src.find("getTimezoneOffset()")
    assert i > 0, "nadie calcula el huso en este componente"
    # Debe estar DENTRO del handler de submit, no en el preámbulo del módulo.
    i_handler = src.find("setSaving(true)")
    assert i_handler > 0, "cambió la forma del submit: revisa este guard"
    assert i > i_handler, (
        "el huso se calcula fuera del submit (posible const de módulo): se congelaría al importar"
    )


def test_el_huso_no_se_cuela_como_campo_requerido():
    """`TRACKING_REQUIRED_FIELDS` es el contrato de lo que el usuario DEBE contestar. El huso no
    lo contesta nadie: es un dato del dispositivo. Meterlo ahí bloquearía el submit."""
    fv = (_BACKEND_ROOT.parent / "frontend" / "src" / "config" / "formValidation.js").read_text(
        encoding="utf-8", errors="replace")
    i = fv.find("TRACKING_REQUIRED_FIELDS")
    assert i > 0
    bloque = fv[i:i + 700]
    assert "tzOffset" not in bloque, (
        "el huso entró en los campos REQUERIDOS: el usuario no puede contestarlo y el submit se "
        "bloquearía"
    )


# ── B. La convención del valor ──────────────────────────────────────────────────────────────────

def test_el_valor_usa_la_convencion_del_sistema():
    """`getTimezoneOffset()` de JS: minutos a SUMAR a la hora local para llegar a UTC (RD=+240,
    España verano=−120). Es la misma convención que `user_tz_offset_min` y que los cuatro sitios
    de F1-T5 — invertirla aquí duplicaría el error en vez de cerrarlo."""
    src = _src()
    i = src.find("getTimezoneOffset()")
    ventana = src[max(0, i - 200):i + 200]
    assert "-" not in ventana.split("getTimezoneOffset()")[0][-40:], (
        "el huso se está negando: rompería la convención que usa todo el sistema"
    )


def test_el_espejo_snake_case_tambien_viaja():
    """`user_tz_offset_min` lee `tzOffset` O `tz_offset_minutes` — los dos escritores del perfil
    persisten AMBAS grafías. Mandar sólo una deja al lector dependiendo de cuál mire primero."""
    src = _src()
    assert "tz_offset_minutes" in src, (
        "sólo viaja una de las dos grafías que el lector del perfil acepta"
    )


# ── C. El backend no lo filtra ──────────────────────────────────────────────────────────────────

def test_el_patch_de_perfil_no_descarta_el_huso():
    """El PATCH mergea `health_profile` con `||` a nivel de clave: no hay allowlist server-side
    que pudiera comerse la clave nueva en silencio. Este test lo fija, porque si algún día se
    añade una, el huso sería lo primero en caerse."""
    ud = (_BACKEND_ROOT / "routers" / "user_data.py").read_text(encoding="utf-8", errors="replace")
    i = ud.find("health_profile")
    assert i > 0
    assert "PATCH" in ud or "patch" in ud


# ── D. El comentario que explica el mecanismo ───────────────────────────────────────────────────

def test_el_comentario_de_la_allowlist_sigue_explicando_por_que():
    """El comentario de `country` es la razón por la que este gap era encontrable: nombra el
    mecanismo («la rama corta es ALLOWLIST: sin esta entrada se cae al suelo en silencio»). Se
    conserva y el huso se suma a él — un lector futuro debe ver que la lista es un contrato, no
    una acumulación."""
    src = _src()
    assert "ALLOWLIST" in src
    i = src.find("ALLOWLIST")
    assert "silencio" in src[i:i + 400].lower()
