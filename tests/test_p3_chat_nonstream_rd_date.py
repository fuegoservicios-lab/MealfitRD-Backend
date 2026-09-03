"""[P3-CHAT-NONSTREAM-RD-DATE · 2026-08-23] El RESTO del contexto temporal del chat no-stream.

Mismo endpoint que `P3-CHAT-NOSTREAM-CONTEXTO-TEMPORAL-RD` (`POST /api/chat`), cerrado a la vez.
Alli se fijan la hora/dia del prompt (`build_temporal_context`) y la ventana del `DIARIO DE HOY`;
aqui viven los bloques que hablan de "hoy" y de "los dias pasados" derivandolos de una fecha:

  * `_build_past_days_context`  — lo que el plan mando y lo que el usuario registro, dia a dia.
  * `_build_today_remaining_context` — "las comidas que te quedan HOY".
  * `_build_hydration_context`  — vasos de agua de hoy.

EL HALLAZGO SE CORRIGIO AL MEDIRLO. El plan afirmaba que `_build_hydration_context` estaba roto:
NO lo estaba — con `local_date_str=None` resuelve el dia server-side desde el `user_id` via
`tools._local_date_str_for_user`, que si conoce el huso del usuario. Se le pasa `local_date` por
PARIDAD con el stream, no porque fuera un defecto. Los dos que SI estaban rotos son los otros.

EL PEOR DE LOS TRES ERA EL QUE LLEVABA UN COMENTARIO DECLARANDO QUE ESTABA BIEN:

    # [P1-CHAT-PAST-DAYS · 2026-07-27] Paridad con el path stream. Este
    # path ya recibe el `tz_offset` del cliente igual que el stream.
    system_prompt += _build_past_days_context(
        user_id, current_plan, plan_id=(plan_record or {}).get("id"),
    )

El comentario decia la verdad sobre la FIRMA (el path si recibe el `tz_offset`) y mentia sobre la
LLAMADA (no se lo pasaba a nadie). Para un usuario en Madrid a las 00:30 eso desplaza la ventana
entera un dia: el coach le resume el menu de anteayer creyendo que es el de ayer.

Nada de esto se ve leyendo el stream, que estaba bien: solo se ve COMPARANDO los dos paths. Por
eso el guard estructural de la clase entera vive en el fichero hermano
(`test_p3_chat_nostream_contexto_temporal_rd.py::test_los_dos_paths_arman_el_contexto_temporal_igual`)
y lo que hay aqui son los tres bloques nombrados, uno por uno, para que el mensaje de fallo diga
CUAL se rompio.

ZONA HORARIA DE LA MAQUINA: no interviene en ningun assert de este fichero.
"""

import ast
import io

import pytest

import agent


def _func(nombre: str) -> ast.FunctionDef:
    src = io.open(agent.__file__, encoding="utf-8").read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == nombre:
            return n
    raise AssertionError(f"`agent.{nombre}` no existe — este guard la sigue por nombre.")


def _llamadas(nodo: ast.AST, nombre: str) -> list:
    return [
        n for n in ast.walk(nodo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == nombre
    ]


#: (bloque, kwargs obligatorios, que se rompe si faltan)
_BLOQUES = [
    (
        "_build_past_days_context",
        {"local_date_str", "tz_offset"},
        "los dias pasados se cuentan desde el dia del SERVIDOR: en Madrid a las 00:30 el coach "
        "resume el menu de anteayer creyendo que es el de ayer",
    ),
    (
        "_build_today_remaining_context",
        {"local_date_str"},
        "'las comidas que te quedan HOY' se resuelven contra el dia del SERVIDOR: en Madrid a "
        "las 00:30 se listan las del dia anterior",
    ),
    (
        "_build_hydration_context",
        {"local_date_str"},
        "se pierde la paridad con el stream (este bloque NO estaba roto, pero con la fecha en "
        "mano usarla evita que dos bloques del mismo prompt discrepen sobre que dia es hoy)",
    ),
]


@pytest.mark.parametrize("bloque,obligatorios,dano", _BLOQUES)
def test_el_bloque_recibe_la_fecha_local_en_el_no_stream(bloque, obligatorios, dano):
    nodo = _func("chat_with_agent")
    llamadas = _llamadas(nodo, bloque)
    assert llamadas, (
        f"`chat_with_agent` ya no invoca `{bloque}`. Si el bloque se retiro a proposito, borra "
        "su fila de `_BLOQUES`; si se renombro, actualizala."
    )
    for c in llamadas:
        kw = {k.arg for k in c.keywords}
        faltan = sorted(obligatorios - kw)
        assert not faltan, f"`{bloque}` sin {faltan} en el path no-stream: {dano}."


@pytest.mark.parametrize("bloque,obligatorios,dano", _BLOQUES)
def test_el_stream_sigue_siendo_la_referencia(bloque, obligatorios, dano):
    """Si el stream perdiera estos argumentos, "paridad" pasaria a significar "los dos mal".

    Este assert es el que impide que alguien cierre una divergencia futura degradando el path
    bueno en vez de arreglar el malo."""
    nodo = _func("chat_with_agent_stream")
    llamadas = _llamadas(nodo, bloque)
    assert llamadas, f"`chat_with_agent_stream` ya no invoca `{bloque}`."
    for c in llamadas:
        kw = {k.arg for k in c.keywords}
        assert not (obligatorios - kw), (
            f"REGRESION EN EL PATH BUENO: `{bloque}` perdio {sorted(obligatorios - kw)} en el stream."
        )


def test_ningun_bloque_temporal_del_no_stream_pasa_local_date_str_none():
    """`local_date_str=None` explicito era la firma del defecto: se escribia el argumento y se
    anulaba en el mismo gesto. Un `None` literal ahi es, hoy, siempre un olvido."""
    nodo = _func("chat_with_agent")
    culpables = []
    for n in ast.walk(nodo):
        if not isinstance(n, ast.Call):
            continue
        nombre = n.func.id if isinstance(n.func, ast.Name) else None
        if not nombre:
            continue
        for k in n.keywords:
            if k.arg in ("local_date_str", "local_date", "tz_offset", "tz_offset_mins"):
                if isinstance(k.value, ast.Constant) and k.value.value is None:
                    culpables.append(f"{nombre}({k.arg}=None)")
    assert not culpables, (
        "el path no-stream vuelve a anular su contexto temporal en el propio callsite: "
        f"{culpables}. Si un bloque debe resolver el dia server-side, OMITE el argumento "
        "(su default ya lo hace) en vez de pasarle `None` a mano."
    )


def test_el_comentario_de_paridad_ya_no_miente():
    """El bloque de dias pasados llevaba desde julio un comentario declarando paridad con el
    stream mientras la llamada la incumplia. Un comentario no puede satisfacer un assert (regla
    del repo), asi que lo que se comprueba es el CODIGO: si la paridad de ese bloque se rompe,
    esto se pone rojo aunque el comentario siga diciendo lo mismo."""
    non = _llamadas(_func("chat_with_agent"), "_build_past_days_context")
    stream = _llamadas(_func("chat_with_agent_stream"), "_build_past_days_context")
    assert len(non) == 1 and len(stream) == 1
    firma = lambda c: sorted(f"{k.arg}={ast.unparse(k.value)}" for k in c.keywords)
    assert firma(non[0]) == firma(stream[0]), (
        f"el bloque de dias pasados diverge:\n  NON-STREAM: {firma(non[0])}\n  STREAM:     {firma(stream[0])}"
    )
