"""[P3-CONSULTAR-DIA-USER-TODAY · 2026-08-22] La tool con la que el coach responde «¿qué toca hoy?»
resolvía «hoy» en hora dominicana, teniendo el `user_id` delante.

`consultar_dia_del_plan` llama a `find_plan_day_for_date(plan, target, rd_today())`. `rd_today()`
hace `now(utc) − 4h` — la fecha de República Dominicana, para todo el mundo. Y esa tool sí tiene
`user_id`: el override de `P0-AGENT-1` se lo garantiza en el tope del bucle `execute_tools`, así
que la información para hacerlo bien estaba disponible en la misma línea.

QUÉ SE ROMPE, y no es simétrico:

  · **México (offset 360, UTC−6).** A las 22:30 del día 21 en Ciudad de México son las 00:30 del
    22 en RD. El usuario pregunta «¿qué me toca hoy?» a la hora de cenar y el coach le describe
    el menú de MAÑANA — el plato que aún no ha comido desaparece de la respuesta.
  · **España (offset −120 en verano).** A las 00:30 del día 22 en Madrid son las 18:30 del 21 en
    RD: le describe el día de AYER, ya consumido.

Es el mismo mecanismo que `P2-LOCAL-DATE-STR-UTC4` cerró para el diario y el contexto temporal del
chat, en la superficie que aquella pasada dejó fuera. Y es la **cuarta** aparición de la misma
pregunta —«¿qué día es hoy para este usuario?»— resuelta a mano: `P3-TZ-FALLBACK-SSOT` unificó tres
esta misma sesión y ésta seguía suelta.

LO QUE NO SE TOCA, y la línea importa. `rd_today()` no se cambia: su nombre dice lo que devuelve
—la fecha de RD— y hay sitios donde eso es exactamente lo correcto. Lo que se corrige es **quién la
llama**. Sí se corrige su docstring, que llamaba a UTC−4 «convención del repo»: no lo es. La
convención es que la fecha sigue AL USUARIO; UTC−4 es el fallback cuando no se sabe quién es, y
llamarlo convención es lo que invita a reusar esta función como si fuera «hoy».

Los otros dos llamantes ya estaban bien: `agent.py` y `chat_history_context.py` prefieren la fecha
que reciben y sólo caen a `rd_today()` si no hay ninguna. Ese fallback se mantiene — lo que se
ancla es que sigan prefiriendo lo del usuario.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tools_src() -> str:
    return (_BACKEND / "tools.py").read_text(encoding="utf-8", errors="replace")


def _codigo(src: str) -> str:
    """Sin comentarios: la zona está llena de prosa que NOMBRA `rd_today` para explicar por qué no
    hay que usarla, y un guard textual la acusaría (comentario-vence-guard, once veces en esta
    ola)."""
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def test_la_tool_no_resuelve_hoy_en_hora_dominicana(tools_src):
    """EL CASO. `rd_today()` dentro de `consultar_dia_del_plan` le da la fecha de RD a un mexicano
    que pregunta a la hora de cenar."""
    codigo = _codigo(tools_src)
    i = codigo.index("find_plan_day_for_date(")
    ventana = codigo[max(0, i - 400): i + 200]
    assert "rd_today()" not in ventana, (
        "`consultar_dia_del_plan` vuelve a resolver «hoy» en hora dominicana. Tiene `user_id` "
        "delante: la fecha tiene que seguir al usuario"
    )


def test_la_tool_usa_la_fecha_local_del_usuario(tools_src):
    """No basta con quitar `rd_today()`: si se quedara sin ninguna fecha, la tool dejaría de saber
    qué día es «hoy» y el coach respondería sobre el día equivocado igual."""
    codigo = _codigo(tools_src)
    i = codigo.index("find_plan_day_for_date(")
    ventana = codigo[max(0, i - 400): i + 200]
    assert "_local_date_str_for_user" in ventana, (
        "la tool no deriva «hoy» del huso del usuario. El helper vive en este mismo módulo"
    )


def test_rd_today_se_presenta_como_fallback_y_no_como_la_norma():
    """La docstring era la invitación: presentar UTC−4 como la norma del proyecto es lo que hace
    que el siguiente la reuse como si fuera «hoy».

    ⚠️ EN POSITIVO, y la primera versión enseñó por qué. Prohibía la frase «convención del repo»
    a secas — y mi propia corrección la usa para enunciar la convención CORRECTA («la convención
    del repo es que la fecha sigue AL USUARIO»). El guard se ponía rojo sobre el texto que lo
    satisface. Un guard que prohíbe un vocablo mide vocabulario; lo que importa es que la función
    se declare FALLBACK y señale a la puerta buena."""
    src = (_BACKEND / "chat_history_context.py").read_text(encoding="utf-8", errors="replace")
    i = src.index("def rd_today")
    doc = src[i:i + 900]
    assert "FALLBACK" in doc, (
        "`rd_today` no se declara FALLBACK. Sin eso, el siguiente lector la toma por «hoy» y le "
        "da la fecha dominicana a un usuario que no vive ahí"
    )
    assert "_local_date_str_for_user" in doc, (
        "`rd_today` no señala cuál es la puerta correcta cuando SÍ se sabe quién pregunta"
    )


def test_los_llamantes_legitimos_siguen_prefiriendo_la_fecha_del_usuario():
    """Ancla en positivo de lo que YA estaba bien, para que la limpieza no se lleve por delante el
    fallback correcto: `agent.py` y `chat_history_context.py` sólo caen a RD si no reciben nada."""
    agente = (_BACKEND / "agent.py").read_text(encoding="utf-8", errors="replace")
    assert "if local_date_str:" in agente, (
        "`agent.py` dejó de preferir la fecha local que recibe antes de caer a RD"
    )
    ctx = (_BACKEND / "chat_history_context.py").read_text(encoding="utf-8", errors="replace")
    assert re.search(r"_today\s*=\s*today\s+or\s+rd_today\(\)", ctx), (
        "`chat_history_context` dejó de preferir el `today` que le pasan"
    )


def test_el_fallback_de_agent_usa_el_ssot():
    """La cuarta copia del 240 a mano, en el mismo fichero: `tz_offset_mins = ... else 240`.
    `P3-TZ-FALLBACK-SSOT` unificó tres respuestas a esta pregunta; ésta se quedó suelta."""
    agente = _codigo((_BACKEND / "agent.py").read_text(encoding="utf-8", errors="replace"))
    # EN POSITIVO y sobre la expresión COMPLETA, no por línea. La primera versión buscaba
    # `if tz_offset is not None else 240` en una sola línea; el arreglo dejó la expresión partida
    # en dos y la mutación se le escapó limpiamente. Un guard atado al salto de línea mide
    # formato, no contrato — y aquí el contrato es de dónde sale el número.
    m = re.search(r"tz_offset_mins\s*=\s*\(?[^\n]*(?:\n[^\n]*)?\)?", agente)
    assert m, "desapareció la asignación de `tz_offset_mins`"
    expr = m.group(0)
    assert "_DEFAULT_TZ_OFFSET_MIN" in expr, (
        f"`agent.py` sigue resolviendo el fallback de huso sin el SSOT: {expr!r}"
    )
    assert not re.search(r"\belse\s+240\b", expr), (
        f"vuelve el 240 clavado como fallback de huso: {expr!r}"
    )
