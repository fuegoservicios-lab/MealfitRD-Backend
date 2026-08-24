"""[P3-CHAT-NOSTREAM-CONTEXTO-TEMPORAL-RD · 2026-08-23] `POST /api/chat` (no-stream) construia
TODO su contexto temporal en hora dominicana.

QUE SE MIDIO ANTES DE TOCAR NADA
--------------------------------
El diagnostico del hallazgo tenia DOS premisas que resultaron FALSAS al medirlas (las lineas del
plan estaban desplazadas):

  * "`chat_with_agent` no admite `local_date`/`tz_offset`" -> FALSO: su firma ya los admitia.
  * "`routers/chat.py` no los lee del body" -> FALSO: el handler ya los leia y ya se los pasaba.

Lo que SI estaba roto era el ultimo tramo, el que hace que todo eso sirva de algo: DENTRO de
`chat_with_agent` los argumentos se dejaban caer. Es la forma exacta de "cablear un paso no es
ejecutarlo": el dato viajaba entero desde el cliente hasta la funcion y ahi se tiraba.

  agent.py  build_temporal_context()                          <- sin argumentos (x2 ramas)
  agent.py  get_consumed_meals_today(user_id)                 <- sin ventana local
  agent.py  _build_today_remaining_context(..., None)         <- "lo que te queda HOY"
  agent.py  _build_past_days_context(user_id, current_plan)   <- sin fecha ni huso

`build_temporal_context` NO usa `_local_date_str_for_user`: tiene su PROPIO default UTC-4
(`prompts/chat_agent.py`). Eran DOS defaults dominicanos independientes dentro del mismo system
prompt. Para un usuario en Madrid a las 00:30 el coach cree que es el dia anterior a las 18:30.

`_build_hydration_context` NO estaba roto (con `None` resuelve el dia server-side desde el
`user_id`); se le pasa `local_date` por paridad, no por defecto.

POBLACION AFECTADA HOY: cero — el frontend solo llama a `/api/chat/stream`. Pero el endpoint esta
registrado, autenticado, tarifado (`log_api_usage`) y alcanzable, asi que es superficie de coste.
De las dos salidas que ofrecia el hallazgo se eligio (a), resolver server-side: el endpoint tiene
un `verified_user_id` delante y no debe depender de que el cliente colabore.

QUE VIGILA ESTE FICHERO
  1. `_resolve_chat_local_time` (routers/chat.py), por CONDUCTA, incluida su promesa de no poder
     empeorar ningun caso.
  2. Que los dos handlers lo llamen ANTES de invocar al agente (orden, no mera presencia — el
     patron de `P0-AGENT-1`).
  3. PARIDAD ESTRUCTURAL entre `chat_with_agent` y `chat_with_agent_stream`: cada llamada de
     contexto temporal tiene que recibir los MISMOS argumentos en los dos paths. Este es el guard
     que de verdad cierra la clase entera de defecto — no una lista de cuatro sitios que el
     proximo bloque temporal volveria a dejar incompleta.

ZONA HORARIA DE LA MAQUINA: no interviene. El unico assert que mira una fecha la deriva del MISMO
offset inyectado que la produccion, asi que el resultado es igual en Santo Domingo, en Madrid y en
un runner de CI en UTC. Dos tests de este repo solo pasaban donde vivia su autor; este no puede.
"""

import ast
import io

import pytest

import agent
import routers.chat as chat_router


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _func(modulo, nombre: str) -> ast.FunctionDef:
    src = io.open(modulo.__file__, encoding="utf-8").read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == nombre:
            return n
    raise AssertionError(f"`{modulo.__name__}.{nombre}` no existe — este guard la sigue por nombre.")


#: Fragmentos que identifican una llamada "de contexto temporal". Se compara por SUBSTRING del
#: nombre y no por lista cerrada a proposito: un bloque temporal NUEVO entra solo en la paridad.
_TEMPORALES = (
    "temporal", "today", "hydration", "past_days", "circadian",
    "consumed", "plan_today", "cycle",
)


def _llamadas_temporales(nodo: ast.AST) -> dict:
    """{nombre: [firma_desnormalizada, ...]} de las llamadas de contexto temporal."""
    out = {}
    for n in ast.walk(nodo):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        nombre = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if not nombre or not any(k in nombre.lower() for k in _TEMPORALES):
            continue
        args = [ast.unparse(a) for a in n.args]
        kwargs = sorted(f"{k.arg}={ast.unparse(k.value)}" for k in n.keywords)
        out.setdefault(nombre, []).append(f"{nombre}({', '.join(args + kwargs)})")
    return {k: sorted(v) for k, v in out.items()}


# ===========================================================================
# 1. `_resolve_chat_local_time` — conducta
# ===========================================================================

@pytest.fixture
def perfil_madrid(monkeypatch):
    """Perfil con huso de Madrid en invierno (-60). Se INYECTA: nunca se lee el reloj del que
    corre el test."""
    import db
    monkeypatch.setattr(db, "user_tz_offset_min", lambda uid: -60, raising=True)
    return -60


def test_si_el_cliente_manda_ambos_no_se_toca_nada(monkeypatch):
    """Ni un round-trip a la DB: si el cliente colaboro, su dato manda."""
    import db

    def _explota(uid):
        raise AssertionError("no debio consultarse el perfil: el cliente mando ambos campos")

    monkeypatch.setattr(db, "user_tz_offset_min", _explota, raising=True)
    assert chat_router._resolve_chat_local_time("2026-03-04", 240, "u-1") == ("2026-03-04", 240)


def test_cero_es_un_offset_legitimo_y_no_se_reemplaza(monkeypatch):
    """`0` es UTC — y Canarias en invierno. La resolucion es por `is not None`, nunca por
    truthiness. Un `if tz_offset:` aqui reintroduciria el huso del servidor para esos usuarios."""
    import db
    monkeypatch.setattr(db, "user_tz_offset_min", lambda uid: 240, raising=True)
    fecha, off = chat_router._resolve_chat_local_time("2026-03-04", 0, "u-1")
    assert off == 0, "el offset 0 del cliente fue tratado como ausente."
    assert fecha == "2026-03-04"


def test_sin_nada_del_cliente_se_resuelve_del_perfil(perfil_madrid):
    """El caso del hallazgo: cliente mudo, usuario verificado. La fecha sale del huso del PERFIL."""
    from datetime import datetime, timedelta, timezone

    fecha, off = chat_router._resolve_chat_local_time(None, None, "u-1")
    assert off == perfil_madrid
    esperada = (datetime.now(timezone.utc) - timedelta(minutes=perfil_madrid)).date().isoformat()
    assert fecha == esperada, (
        f"se resolvio {fecha} y el dia local del offset {perfil_madrid} es {esperada}."
    )


def test_el_offset_del_cliente_gana_al_del_perfil(perfil_madrid):
    """El usuario viaja: el huso que reporta su navegador AHORA vale mas que el persistido.

    Con `tz_offset=240` (RD) y sin fecha, la fecha derivada tiene que ser la de RD, no la de
    Madrid. Ambas se calculan desde el mismo instante UTC, asi que el assert es valido a
    cualquier hora y en cualquier maquina."""
    from datetime import datetime, timedelta, timezone

    fecha, off = chat_router._resolve_chat_local_time(None, 240, "u-1")
    assert off == 240
    assert fecha == (datetime.now(timezone.utc) - timedelta(minutes=240)).date().isoformat()


def test_invitado_sin_identidad_conserva_la_conducta_previa(monkeypatch):
    """Sin `verified_user_id` no hay perfil que leer. Devolver `(None, None)` deja aguas abajo
    exactamente donde estaba: este helper no puede empeorar ningun caso."""
    import db

    def _explota(uid):
        raise AssertionError("no hay identidad verificada: no debio consultarse ningun perfil")

    monkeypatch.setattr(db, "user_tz_offset_min", _explota, raising=True)
    assert chat_router._resolve_chat_local_time(None, None, None) == (None, None)
    assert chat_router._resolve_chat_local_time(None, None, "") == (None, None)


def test_si_la_lectura_de_perfil_revienta_devuelve_lo_que_llego(monkeypatch):
    """Fail-open: un fallo de DB no puede tumbar un turno de chat."""
    import db

    def _revienta(uid):
        raise RuntimeError("simulado")

    monkeypatch.setattr(db, "user_tz_offset_min", _revienta, raising=True)
    assert chat_router._resolve_chat_local_time(None, None, "u-1") == (None, None)
    assert chat_router._resolve_chat_local_time("2026-03-04", None, "u-1") == ("2026-03-04", None)


def test_un_offset_basura_no_revienta(monkeypatch):
    import db
    monkeypatch.setattr(db, "user_tz_offset_min", lambda uid: 240, raising=True)
    assert chat_router._resolve_chat_local_time(None, "manana", "u-1") == (None, "manana")


# ===========================================================================
# 2. Los handlers lo llaman ANTES de invocar al agente
# ===========================================================================

@pytest.mark.parametrize(
    "handler,invocacion",
    [
        ("api_chat", "chat_with_agent"),
        ("api_chat_stream", "chat_with_agent_stream"),
    ],
)
def test_el_handler_resuelve_el_huso_antes_de_invocar_al_agente(handler, invocacion):
    """ORDEN, no mera presencia (patron `P0-AGENT-1`): resolver el huso DESPUES de invocar al
    agente seria una linea que no hace nada, y el guard estaria verde igual."""
    nodo = _func(chat_router, handler)
    resoluciones = [
        n.lineno for n in ast.walk(nodo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_resolve_chat_local_time"
    ]
    agentes = [
        n.lineno for n in ast.walk(nodo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == invocacion
    ]
    assert resoluciones, (
        f"`{handler}` ya no llama a `_resolve_chat_local_time`: el contexto temporal vuelve a "
        "depender de que el cliente mande su fecha."
    )
    assert agentes, f"`{handler}` ya no invoca a `{invocacion}` — ¿se renombro?"
    assert min(resoluciones) < min(agentes), (
        f"en `{handler}` el huso se resuelve DESPUES de invocar a `{invocacion}`: la resolucion "
        "es inerte."
    )


@pytest.mark.parametrize("handler", ["api_chat", "api_chat_stream"])
def test_el_agente_recibe_los_dos_campos(handler):
    nodo = _func(chat_router, handler)
    llamadas = [
        n for n in ast.walk(nodo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("chat_with_agent", "chat_with_agent_stream")
    ]
    assert llamadas
    for c in llamadas:
        kw = {k.arg for k in c.keywords}
        assert {"local_date", "tz_offset"} <= kw, (
            f"`{handler}` invoca al agente sin {sorted({'local_date', 'tz_offset'} - kw)}."
        )


# ===========================================================================
# 3. Paridad estructural stream <-> no-stream (el guard que cierra la clase)
# ===========================================================================

def test_los_dos_paths_arman_el_contexto_temporal_igual():
    """Cada llamada de contexto temporal recibe los MISMOS argumentos en los dos paths.

    Este es el guard de verdad. Una lista cerrada de "los cuatro sitios rotos" se queda corta en
    cuanto alguien anade un quinto bloque temporal al stream y se olvida del no-stream — que es
    literalmente como nacio este defecto (`_build_past_days_context` llevaba un comentario que
    DECLARABA paridad con el stream mientras omitia los dos argumentos).

    Se compara la firma desnormalizada (kwargs ordenados), asi que reordenar argumentos no lo
    rompe; cambiar u omitir uno, si.
    """
    non = _llamadas_temporales(_func(agent, "chat_with_agent"))
    stream = _llamadas_temporales(_func(agent, "chat_with_agent_stream"))

    solo_stream = sorted(set(stream) - set(non))
    assert not solo_stream, (
        f"el path stream arma bloques temporales que el no-stream no arma: {solo_stream}. "
        "Si la divergencia es deliberada, documentala aqui antes de excluirla."
    )

    divergentes = {
        nombre: {"non_stream": non[nombre], "stream": stream[nombre]}
        for nombre in sorted(set(non) & set(stream))
        if non[nombre] != stream[nombre]
    }
    assert not divergentes, (
        "los dos paths del chat arman el mismo bloque temporal con argumentos distintos:\n"
        + "\n".join(
            f"  {k}\n    NON-STREAM: {v['non_stream']}\n    STREAM:     {v['stream']}"
            for k, v in divergentes.items()
        )
    )


def test_build_temporal_context_nunca_se_llama_sin_argumentos_en_el_chat():
    """El defecto literal del hallazgo, fijado aparte de la paridad: si alguien lo rompiera en
    LOS DOS paths a la vez, la paridad seguiria verde y este assert no."""
    for fn in ("chat_with_agent", "chat_with_agent_stream"):
        nodo = _func(agent, fn)
        for n in ast.walk(nodo):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "build_temporal_context":
                kw = {k.arg for k in n.keywords}
                assert {"local_date", "tz_offset"} <= kw, (
                    f"`{fn}` llama a `build_temporal_context` sin {sorted({'local_date', 'tz_offset'} - kw)}: "
                    "ese bloque tiene su PROPIO default UTC-4 y volveria a decir el dia dominicano."
                )


def test_la_ventana_del_diario_de_hoy_es_local_en_el_no_stream():
    """`DIARIO DE HOY` con la ventana del servidor le presenta al coach la cena de anoche como
    "registrado hoy" — el defecto que `P1-DIARY-TZ-DEFAULT-RD` cerro en el stream."""
    nodo = _func(agent, "chat_with_agent")
    llamadas = [
        n for n in ast.walk(nodo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "get_consumed_meals_today"
    ]
    assert llamadas, "`chat_with_agent` ya no arma `DIARIO DE HOY` — ¿se movio el bloque?"
    for c in llamadas:
        kw = {k.arg for k in c.keywords}
        assert {"date_str", "tz_offset_mins"} <= kw, (
            f"falta {sorted({'date_str', 'tz_offset_mins'} - kw)} en `get_consumed_meals_today`."
        )
