"""[P2-LOCAL-DATE-STR-UTC4 · 2026-08-21] `_local_date_str_for_user()` no sabía de qué usuario.

El nombre promete «del usuario» y el cuerpo hacía `datetime.now(timezone.utc) - timedelta(hours=4)`:
la fecha dominicana, para todo el mundo. Fase 1 (T5) conectó `user_tz_offset_min` en tres sitios de
`tools.py` y dejó éste, que es el que deciden las tools de hidratación y el contexto temporal del
`/api/chat` no-stream.

QUÉ SE ROMPE, EN LOS DOS SENTIDOS — y el propio comentario del call site en `agent.py` describe el
bug para RD sin darse cuenta de que seguía abierto para los demás:

  · **España (UTC+1/+2, offset −60/−120).** A las 00:30 del día 22 en Madrid son las 18:30 del 21
    en UTC−4. La función dice «21». El vaso de agua que el usuario acaba de registrar cae en el
    cubo de AYER, y «¿cuánta agua llevo hoy?» contesta con el día equivocado.
  · **México (UTC−6, offset 360).** A las 22:30 del 21 en Ciudad de México son las 00:30 del 22 en
    UTC−4. La función dice «22». El registro se va al cubo de MAÑANA y el contador de hoy lee 0 —
    el agente puede regañar a alguien que sí bebió.

Es la misma clase que `P1-PROACTIVE-TZ` y que los tres relojes que esta ola ya cerró: una hora
local calculada con el huso de otro.

EL SIGNO, QUE ES DONDE ESTO SE EQUIVOCA. La convención es la de `Date.getTimezoneOffset()`:
**positivo = oeste de UTC** (RD = 240, España en invierno = −60). Así que la hora local es
`utc − offset`, no `utc + offset`. Invertirlo da el doble del error y encima parece correcto para
República Dominicana, que es donde se prueba — exactamente lo que pasó en `P1-AVG-MEAL-HOUR-SIGN`,
donde el signo llevaba invertido desde siempre y nadie lo vio porque el caso dominicano lo tapaba.

FAIL-SAFE HACIA LA CONDUCTA DE HOY: sin `user_id`, sin perfil o con un huso ilegible, se usa 240
(RD). Un usuario sin huso registrado no puede quedarse sin fecha.

Cubre:
  A. La fecha sigue el huso del usuario.
  B. El signo.
  C. Fail-safe a la conducta histórica.
  D. Los call sites pasan el usuario (si no, el arreglo nace inerte).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(scope="module")
def tools():
    import tools as _t
    return _t


def _fecha_esperada(offset_min: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=offset_min)).date().isoformat()


# ── A. La fecha sigue al usuario ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("offset", [240, -60, -120, 360, 300, 0])
def test_la_fecha_usa_el_huso_del_usuario(tools, monkeypatch, offset):
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: offset)
    assert tools._local_date_str_for_user("u1") == _fecha_esperada(offset)


def test_dos_usuarios_en_husos_distintos_no_comparten_fecha(tools, monkeypatch):
    """El síntoma medible: con el bug, los dos recibían la fecha dominicana. Se elige un par de
    husos separados por EXACTAMENTE 24 horas (±720) para que el día difiera SIEMPRE, no según a
    qué hora corra el test — un caso que sólo falla a ciertas horas es un test intermitente, no
    una defensa.

    [2026-08-22, tumbó un deploy a las 13:36 UTC] Decía «20 horas» y usaba 720/-600: 22 horas,
    que dejan una ventana DIARIA de 12:00 a 14:00 UTC en la que los dos comparten fecha contra
    el código correcto. La docstring enunciaba el principio y los números lo incumplían — el
    mismo patrón que el test del signo, dos funciones más abajo, ya confesaba."""
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 720 if uid == "oeste" else -720)
    assert tools._local_date_str_for_user("oeste") != tools._local_date_str_for_user("este")


# ── B. El signo ─────────────────────────────────────────────────────────────────────────────────

def test_el_signo_es_el_de_getTimezoneOffset(tools, monkeypatch):
    """Positivo = OESTE de UTC. Invertirlo duplica el error y encima parece correcto en República
    Dominicana, que es donde se prueba — el modo de fallo exacto de P1-AVG-MEAL-HOUR-SIGN.

    ⚠️ 720, no 240. La primera versión usaba ±240: ocho horas de separación, así que los dos
    instantes caen en el MISMO día salvo en una ventana de 8 h de cada 24. El test pasaba por la
    hora a la que lo corrí y se puso rojo a las 04:25 UTC de otro día, contra el código correcto.
    ±720 deja los dos instantes a exactamente 24 h: la fecha difiere SIEMPRE.

    Lo doloroso es que el principio estaba escrito dos funciones más arriba, en
    `test_dos_usuarios_en_husos_distintos_no_comparten_fecha`, con estas palabras: «un caso que
    sólo falla a ciertas horas es un test intermitente, no una defensa». Lo enuncié y acto seguido
    lo incumplí."""
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 720)
    assert tools._local_date_str_for_user("u1") == _fecha_esperada(720)
    assert tools._local_date_str_for_user("u1") != _fecha_esperada(-720)


# ── C. Fail-safe ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("uid", [None, "", "guest"])
def test_sin_usuario_conserva_la_fecha_dominicana(tools, uid):
    """Conducta histórica: el helper se llamaba sin argumentos y devolvía UTC−4."""
    assert tools._local_date_str_for_user(uid) == _fecha_esperada(240)


def test_se_sigue_pudiendo_llamar_sin_argumentos(tools):
    """La firma vieja no se rompe: hay tres call sites y uno vive en otro módulo."""
    assert tools._local_date_str_for_user() == _fecha_esperada(240)


@pytest.mark.parametrize("devuelto", [None, "basura", float("nan"), object()])
def test_un_huso_ilegible_no_deja_al_usuario_sin_fecha(tools, monkeypatch, devuelto):
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: devuelto)
    assert tools._local_date_str_for_user("u1") == _fecha_esperada(240)


def test_si_el_lookup_revienta_tampoco(tools, monkeypatch):
    def _boom(uid):
        raise RuntimeError("DB caída")
    monkeypatch.setattr(tools, "user_tz_offset_min", _boom)
    assert tools._local_date_str_for_user("u1") == _fecha_esperada(240)


# ── D. Los call sites lo pasan ──────────────────────────────────────────────────────────────────

def test_las_tools_de_hidratacion_pasan_el_usuario(tools):
    """Sin esto el arreglo nace INERTE: la función sabría el huso y nadie se lo diría. Es la
    trampa que esta misma ola ya vio dos veces (el catálogo de F2 y el título del plan: la función
    existía y nadie la llamaba)."""
    import inspect
    import re
    src = inspect.getsource(tools)
    llamadas = re.findall(r"_local_date_str_for_user\(([^)]*)\)", src)
    # La primera coincidencia es la propia definición (`def ...`); se filtra por posición.
    llamadas = [c for c in llamadas if "user_id" in c or c.strip() == ""]
    reales = [c for c in llamadas if c.strip() != ""]
    assert reales, "ninguna tool le pasa el user_id: el helper no puede saber el huso"


def test_el_contexto_del_chat_pasa_el_usuario():
    """`agent.py::_build_hydration_context` ya recibe `user_id` — es el path no-stream de
    `/api/chat`, que es justo el que cae a este helper cuando el cliente no manda su fecha."""
    import inspect
    import re
    import agent
    src = inspect.getsource(agent._build_hydration_context)
    m = re.search(r"_local_date_str_for_user\(([^)]*)\)", src)
    assert m, "el contexto de hidratación ya no usa el helper (¿renombrado?)"
    assert "user_id" in m.group(1), (
        "el chat no-stream sigue pidiendo la fecha SIN decir de quién: para un español a las 00:30 "
        "leería el cubo de ayer"
    )
