"""[P1-CHAT-DIARY-CORRECT · 2026-07-29] El coach puede CORREGIR una fila
mal registrada en `consumed_meals` sin crear una segunda.

Incidente real, verificado en la DB (transcript de producción, 09:58-10:00
local):
    El usuario preguntó "¿qué comí ayer?" — el coach listó desayuno/cena de
    ayer y preguntó "Del almuerzo no hay registro, ¿lo recuerdas?". El
    usuario respondió describiendo una comida (sin decir día ni comida). El
    coach ATRIBUYÓ esa respuesta al almuerzo de AYER — el TEMA que el coach
    mismo había abierto, no algo que el usuario hubiera afirmado — e
    insertó una fila. Dos turnos después el usuario aclaró "me referia al
    desayuno de hoy que me lo comi" y el coach volvió a malinterpretar antes
    de "corregir". La única herramienta disponible para corregir era
    `log_consumed_meal`, que SOLO puede insertar: la DB terminó con DOS
    filas (`consumed_at` 07-28 13:58Z/almuerzo + 07-29 14:00Z/desayuno) para
    UNA sola comida real. El coach afirmó "quedó corregido ✅" sin haber
    hecho ningún UPDATE.

Diagnóstico del owner (ver brief): la atribución de día/comida debe salir
de lo que el USUARIO dijo explícitamente, o ser la única lectura posible —
NUNCA de qué estaba preguntando el coach. Cuando genuinamente no está claro,
preguntar UNA vez es barato; adivinar mal es caro (3 turnos + fila fantasma
para deshacerlo).

Este fix NO reabre `P1-CHAT-ACT-DONT-ASK` (2026-07-28): esa regla prohíbe
pedir PERMISO para registrar algo que el usuario ya afirmó en pasado. Esta
regla prohíbe INVENTAR el día/la comida — son ejes ortogonales. Sección 5
de este archivo (`test_over_asking_guard_named_slot_still_acts`) ancla
específicamente que nombrar la comida sigue disparando acción inmediata,
no una pregunta.

Cobertura:
    1. `correct_consumed_meal` (tools.py) — shape + cubierta por
       agent_tools + doc de paridad.
    2. `db_facts.update_consumed_meal` — semántica UPDATE-only: la
       corrección deja EXACTAMENTE una fila; un `meal_id` de OTRO usuario
       no toca nada (IDOR) y la fila víctima sobrevive intacta.
    3. P0-AGENT-1 — el override genérico de `execute_tools` cubre la tool
       nueva (funcional, mock de tool_call con user_id ajeno).
    4. Las reglas de atribución día/comida + "no anunciar sin ejecutar
       extendido a correcciones" viven en los 4 prompts base compartidos
       (`_CHAT_BREVITY_RULES`), no en 3 de 4.
    5. Guard anti-sobre-preguntar: nombrar la comida explícitamente sigue
       llevando a actuar, no a preguntar — la regresión que desharía
       P1-CHAT-ACT-DONT-ASK.

Tooltip-anchor: P1-CHAT-DIARY-CORRECT
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_CHAT_AGENT_PROMPTS_PY = _BACKEND_ROOT / "prompts" / "chat_agent.py"
_TOOLS_DOC = _BACKEND_ROOT / "docs" / "agent_tools_user_id_table.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _read(_TOOLS_PY)


@pytest.fixture(scope="module")
def prompts_src() -> str:
    return _read(_CHAT_AGENT_PROMPTS_PY)


@pytest.fixture(scope="module")
def tools_doc_src() -> str:
    return _read(_TOOLS_DOC)


# ===========================================================================
# 1. `correct_consumed_meal` — shape + wiring
# ===========================================================================

def test_correct_consumed_meal_is_a_tool_in_agent_tools():
    from tools import correct_consumed_meal, agent_tools

    names = [getattr(t, "name", None) for t in agent_tools]
    assert "correct_consumed_meal" in names, (
        "P1-CHAT-DIARY-CORRECT regresión: `correct_consumed_meal` no está "
        "en `agent_tools` — el chat-agent nunca la vería como opción."
    )
    assert correct_consumed_meal.name == "correct_consumed_meal"


def test_correct_consumed_meal_requires_meal_id_param():
    """El schema de la tool DEBE exponer `meal_id` — sin él la LLM no tiene
    forma de nombrar la fila específica a corregir."""
    from tools import correct_consumed_meal

    params = set(correct_consumed_meal.args.keys())
    assert "meal_id" in params, (
        "P1-CHAT-DIARY-CORRECT regresión: falta `meal_id` en el schema de "
        "`correct_consumed_meal` — sin id no hay forma de apuntar a UNA "
        "fila concreta (degeneraría a insertar otra vez)."
    )
    # Campos correctivos opcionales — todos deben poder omitirse (UPDATE
    # parcial: solo se corrige lo que cambió).
    for optional in ("meal_name", "calories", "meal_type", "days_ago"):
        assert optional in params, (
            f"P1-CHAT-DIARY-CORRECT regresión: falta el parámetro corregible "
            f"`{optional}` en `correct_consumed_meal`."
        )


def test_log_consumed_meal_still_returns_row_id_for_later_correction(tools_src: str):
    """`log_consumed_meal` debe seguir devolviendo el ID_REGISTRO_DIARIO en
    el ToolMessage — es la ÚNICA forma en que la LLM puede nombrar la fila
    en un turno posterior. Si esto desaparece, `correct_consumed_meal`
    queda inutilizable en la práctica (la LLM no tendría ningún id para
    pasarle)."""
    assert "ID_REGISTRO_DIARIO" in tools_src, (
        "P1-CHAT-DIARY-CORRECT regresión: `ID_REGISTRO_DIARIO` desapareció "
        "de tools.py. `log_consumed_meal` debe seguir devolviendo el id de "
        "la fila insertada para que `correct_consumed_meal` tenga algo que "
        "corregir."
    )


def test_doc_documents_correct_consumed_meal_idor_defense(tools_doc_src: str):
    row_re = re.compile(
        r"\|\s*\d+\s*\|\s*`correct_consumed_meal`\s*\|([^\n]*)\|"
    )
    m = row_re.search(tools_doc_src)
    assert m is not None, (
        "P1-CHAT-DIARY-CORRECT regresión: `correct_consumed_meal` no tiene "
        "row en `docs/agent_tools_user_id_table.md` — "
        "`test_p2_chat_cleanup.py::test_doc_lists_all_agent_tools` fallará."
    )
    assert "user_id" in m.group(1), (
        "P1-CHAT-DIARY-CORRECT regresión: la row del doc no explica la "
        "mutación cross-user (IDOR) que el override impide."
    )


# ===========================================================================
# 2. `db_facts.update_consumed_meal` — semántica UPDATE-only + IDOR
# ===========================================================================
#
# Fake in-memory de `consumed_meals` que interpreta el WHERE/SET reales de
# la query (no mockeamos el resultado a ciegas — reproducimos la semántica
# de Postgres para que el test muerda un regresión real). CERO conexión a
# DB real (ver runbook: la suite NO debe escribir en producción).

class _FakeConsumedMealsTable:
    """Mini-intérprete GENÉRICO del `UPDATE consumed_meals SET ... WHERE
    ... RETURNING id` real — deliberadamente NO asume que el WHERE incluye
    `user_id` (eso convertiría al fake en el propio guard de seguridad,
    enmascarando una regresión real en vez de reproducirla: si el WHERE
    sabotedo solo trae `id`, este intérprete filtra únicamente por `id` —
    tal como haría Postgres — permitiendo que el test de IDOR de verdad
    muerda la regresión en vez de que la muerda un `assert` de forma."""

    def __init__(self):
        self.rows: dict = {}

    def seed(self, row_id: str, user_id: str, **fields):
        self.rows[row_id] = {"user_id": user_id, **fields}

    def fake_execute_sql_write(self, query, params, returning=False):
        m = re.search(
            r"UPDATE consumed_meals SET (.*) WHERE (.*) RETURNING id",
            query, re.S,
        )
        assert m, f"forma de query inesperada (no es un UPDATE...RETURNING id): {query!r}"
        set_clause, where_clause = m.group(1), m.group(2)

        cols = [c.split("=")[0].strip() for c in set_clause.split(",")]
        set_params = params[: len(cols)]
        where_params = params[len(cols):]

        conditions = [c.strip() for c in where_clause.split(" AND ")]
        assert len(conditions) == len(where_params), (
            f"mismatch condiciones/params del WHERE: {conditions!r} vs "
            f"{where_params!r} — query: {query!r}"
        )

        matches = set(self.rows.keys())
        for cond, val in zip(conditions, where_params):
            col = cond.split("=")[0].strip()
            if col == "id":
                matches &= {rid for rid in matches if rid == val}
            elif col == "user_id":
                matches &= {rid for rid in matches if self.rows[rid]["user_id"] == val}
            else:
                raise AssertionError(f"columna de WHERE no soportada por el fake: {col!r}")

        if not matches:
            return []
        row_id = next(iter(matches))
        row = self.rows[row_id]
        for col, val in zip(cols, set_params):
            row[col] = getattr(val, "obj", val)  # unwrap Jsonb(...)
        return [{"id": row_id}]


@pytest.fixture()
def fake_table(monkeypatch):
    import db_facts

    table = _FakeConsumedMealsTable()
    monkeypatch.setattr(db_facts, "connection_pool", MagicMock())
    monkeypatch.setattr(db_facts, "execute_sql_write", table.fake_execute_sql_write)
    return table


def test_correction_leaves_exactly_one_row(fake_table):
    """El incidente real: 3 panes registrados como almuerzo de ayer,
    debían ser desayuno de hoy. La corrección debe dejar UNA fila con los
    datos correctos — nunca una segunda."""
    import db_facts

    fake_table.seed(
        "row-1", "user-A",
        meal_name="3 panes con queso y jamon", calories=900, protein=30,
        carbs=90, healthy_fats=20, meal_type="almuerzo",
        consumed_at="2026-07-28T13:58:00+00:00",
    )

    result = db_facts.update_consumed_meal(
        "user-A", "row-1",
        meal_type="desayuno",
        consumed_at_override="2026-07-29T14:00:00+00:00",
    )

    assert result == "row-1"
    assert len(fake_table.rows) == 1, (
        "P1-CHAT-DIARY-CORRECT regresión CRÍTICA: la corrección creó una "
        "fila nueva en vez de actualizar la existente — exactamente el "
        "modo de fallo del incidente (2 filas para 1 comida real)."
    )
    row = fake_table.rows["row-1"]
    assert row["meal_type"] == "desayuno"
    assert row["consumed_at"] == "2026-07-29T14:00:00+00:00"
    # Campos NO mencionados en la corrección deben sobrevivir intactos.
    assert row["meal_name"] == "3 panes con queso y jamon"
    assert row["calories"] == 900


def test_correction_via_tool_wraps_days_ago_and_meal_type(fake_table, monkeypatch):
    """La tool (`tools.correct_consumed_meal`) debe traducir `days_ago` a
    un `consumed_at_override` real y normalizar `meal_type`, delegando el
    UPDATE atómico a `db_facts.update_consumed_meal` — sin reimplementar la
    escritura ella misma (dos paths a la misma tabla es el patrón que ya
    causó el bug de doble fila)."""
    from tools import correct_consumed_meal
    import tools as _tools_mod

    fake_table.seed(
        "row-2", "user-B",
        meal_name="Almuerzo viejo", calories=500, protein=20, carbs=40,
        healthy_fats=10, meal_type="almuerzo",
        consumed_at="2026-07-20T12:00:00+00:00",
    )
    # El wiring real es tools.db_update_consumed_meal → db_facts.update_consumed_meal.
    monkeypatch.setattr(
        _tools_mod, "db_update_consumed_meal",
        __import__("db_facts").update_consumed_meal,
    )

    out = correct_consumed_meal.func(
        user_id="user-B",
        meal_id="row-2",
        meal_type="desayuno",
        days_ago=0,
    )

    assert "ERROR" not in out
    assert len(fake_table.rows) == 1
    row = fake_table.rows["row-2"]
    assert row["meal_type"] == "desayuno"
    # days_ago=0 ⇒ hoy ⇒ el override debe ser una fecha de HOY (no la vieja
    # del 2026-07-20).
    assert not row["consumed_at"].startswith("2026-07-20")


def test_idor_cannot_touch_other_users_row(fake_table):
    """El test de seguridad: un `meal_id` real pero de OTRO usuario no debe
    ni corregirse ni reportar éxito — y la fila víctima debe sobrevivir
    intacta, byte a byte, para probar que NO hubo escritura alguna."""
    import db_facts

    fake_table.seed(
        "victim-row", "victim-user",
        meal_name="Cena de la víctima", calories=650, protein=40,
        carbs=30, healthy_fats=15, meal_type="cena",
        consumed_at="2026-07-15T23:00:00+00:00",
    )

    result = db_facts.update_consumed_meal(
        "attacker-user", "victim-row",
        meal_name="HACKEADO", calories=1,
    )

    assert result is None, (
        "P1-CHAT-DIARY-CORRECT regresión CRÍTICA (IDOR): "
        "`update_consumed_meal` reportó éxito sobre una fila que NO es del "
        "caller — un `meal_id` adivinado/alucinado por la LLM podría "
        "corromper el diario de OTRO usuario."
    )
    victim = fake_table.rows["victim-row"]
    assert victim["user_id"] == "victim-user"
    assert victim["meal_name"] == "Cena de la víctima", (
        "P1-CHAT-DIARY-CORRECT regresión CRÍTICA: la fila víctima fue "
        "modificada pese al mismatch de user_id."
    )
    assert victim["calories"] == 650
    assert len(fake_table.rows) == 1, "no debe haberse creado ninguna fila nueva"


def test_idor_via_tool_layer_reports_error_not_success(fake_table, monkeypatch):
    """Mismo ataque pero pasando por la capa de la tool (`tools.py`), que es
    lo que la LLM invoca realmente."""
    from tools import correct_consumed_meal
    import tools as _tools_mod

    fake_table.seed(
        "victim-row-2", "victim-user-2",
        meal_name="Desayuno de la víctima", calories=300, protein=15,
        carbs=20, healthy_fats=5, meal_type="desayuno",
        consumed_at="2026-07-10T12:00:00+00:00",
    )
    monkeypatch.setattr(
        _tools_mod, "db_update_consumed_meal",
        __import__("db_facts").update_consumed_meal,
    )

    out = correct_consumed_meal.func(
        user_id="attacker",
        meal_id="victim-row-2",
        meal_name="HACKEADO",
    )

    assert "ERROR" in out, (
        "la tool debe reportar ERROR (no éxito silencioso) cuando el "
        "UPDATE no tocó ninguna fila."
    )
    assert fake_table.rows["victim-row-2"]["meal_name"] == "Desayuno de la víctima"


def test_no_fields_to_correct_is_a_noop_not_an_error_masking_write(fake_table):
    """Sin ningún campo a corregir, `update_consumed_meal` no debe ejecutar
    un UPDATE vacío (ni tocar la fila) — retorna None, tratado como
    'nada que hacer' por el caller."""
    import db_facts

    fake_table.seed("row-3", "user-C", meal_name="X", calories=100)
    result = db_facts.update_consumed_meal("user-C", "row-3")
    assert result is None
    assert fake_table.rows["row-3"]["meal_name"] == "X"


# ===========================================================================
# 3. P0-AGENT-1 — el override cubre `correct_consumed_meal`
# ===========================================================================

@pytest.fixture
def execute_tools_callable(monkeypatch):
    _stub_targets = (
        "db", "db_inventory", "db_profiles", "db_plans", "db_facts",
        "memory_manager", "vision_agent", "fact_extractor", "cpu_tasks",
        "graph_orchestrator", "ai_helpers",
    )
    for mod_name in _stub_targets:
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, MagicMock())
    try:
        from agent import execute_tools
    except Exception as e:
        pytest.skip(f"No se pudo importar agent.execute_tools (deps): {e}")
    return execute_tools


def _make_tool_call(name: str, args: dict, call_id: str = "tc_1") -> dict:
    return {"name": name, "args": dict(args), "id": call_id}


def _make_state(user_id: str, session_id: str, tool_calls: list) -> dict:
    last_msg = MagicMock()
    last_msg.tool_calls = tool_calls
    return {
        "messages": [last_msg],
        "user_id": user_id,
        "session_id": session_id,
        "form_data": {},
        "current_plan": {},
        "updated_fields": {},
        "new_plan": None,
        "sys_prompt": "",
    }


def test_p0_agent_1_override_covers_correct_consumed_meal(
    execute_tools_callable, monkeypatch
):
    """Smoke funcional: `correct_consumed_meal` cae en el branch `else:
    t.invoke(tool_args)` de `execute_tools` — el override genérico (antes
    de cualquier if/elif) debe reemplazar un `user_id` ajeno inyectado por
    la LLM por el `state['user_id']` autenticado, exactamente como ya hace
    con `log_consumed_meal`."""
    import agent as _agent

    trusted_uid = "AAAAAAAA-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    hostile_uid = "ZZZZZZZZ-zzzz-zzzz-zzzz-zzzzzzzzzzzz"
    captured = {}

    class _FakeTool:
        name = "correct_consumed_meal"

        def invoke(self, tool_args):
            captured["args"] = dict(tool_args)
            return "ok"

    monkeypatch.setattr(_agent, "agent_tools", [_FakeTool()])

    tool_calls = [
        _make_tool_call(
            "correct_consumed_meal",
            {
                "user_id": hostile_uid,  # LLM inyecta uid ajeno
                "meal_id": "some-row-id",
                "meal_type": "desayuno",
            },
        )
    ]
    state = _make_state(trusted_uid, "session_x", tool_calls)
    execute_tools_callable(state)

    assert "args" in captured, (
        "P1-CHAT-DIARY-CORRECT regresión funcional: `correct_consumed_meal` "
        "nunca fue invocada por execute_tools."
    )
    assert captured["args"]["user_id"] == trusted_uid, (
        f"P0-AGENT-1 regresión CRÍTICA sobre la tool nueva: recibió "
        f"`user_id={captured['args'].get('user_id')!r}` (LLM-supplied) en "
        f"vez del trusted `{trusted_uid!r}` — reabre IDOR de escritura "
        f"sobre `consumed_meals` de otro usuario vía `meal_id` adivinado."
    )


# ===========================================================================
# 4. Regla de atribución día/comida + "no anunciar" extendido a
#    correcciones — presente en LOS 4 prompts base, no en 3 de 4.
# ===========================================================================

_ATTRIBUTION_MARKER = "ATRIBUCIÓN DE DÍA/COMIDA"
_ATTRIBUTION_NOT_FROM_TOPIC = "NUNCA de qué era el TEMA de tu pregunta anterior"
_ATTRIBUTION_ACT_WHEN_CLEAR = "actúa directo"
_ANNOUNCE_EXTENDED_TO_CORRECTIONS = "corregir o borrar un registro que ya existe"


@pytest.fixture(scope="module")
def four_base_prompts() -> dict:
    """Importa los 4 prompts BASE (evaluados, no texto crudo de archivo) —
    así un comentario de código nunca puede satisfacer el assert; solo el
    contenido real que se manda a la LLM cuenta."""
    from prompts.chat_agent import (
        CHAT_SYSTEM_PROMPT_BASE,
        CHAT_STREAM_SYSTEM_PROMPT_BASE,
        CHAT_AGENT_INLINE_PROMPT,
        CHAT_STREAM_INLINE_PROMPT,
    )
    return {
        "CHAT_SYSTEM_PROMPT_BASE": CHAT_SYSTEM_PROMPT_BASE,
        "CHAT_STREAM_SYSTEM_PROMPT_BASE": CHAT_STREAM_SYSTEM_PROMPT_BASE,
        "CHAT_AGENT_INLINE_PROMPT": CHAT_AGENT_INLINE_PROMPT,
        "CHAT_STREAM_INLINE_PROMPT": CHAT_STREAM_INLINE_PROMPT,
    }


@pytest.mark.parametrize(
    "prompt_name",
    [
        "CHAT_SYSTEM_PROMPT_BASE",
        "CHAT_STREAM_SYSTEM_PROMPT_BASE",
        "CHAT_AGENT_INLINE_PROMPT",
        "CHAT_STREAM_INLINE_PROMPT",
    ],
)
def test_each_of_the_four_prompts_carries_attribution_rule(
    four_base_prompts: dict, prompt_name: str
):
    """Parametrizado sobre las 4 constantes individualmente — si una futura
    edición copy-pasteara la regla en 3 prompts y se olvidara del 4to (el
    fallo exacto que `_CHAT_BREVITY_RULES` existe para prevenir), el caso
    parametrizado de ESE prompt falla solo, señalando cuál."""
    text = four_base_prompts[prompt_name]
    assert _ATTRIBUTION_MARKER in text, (
        f"[{prompt_name}] falta la regla de atribución día/comida "
        f"({_ATTRIBUTION_MARKER!r}) — P1-CHAT-DIARY-CORRECT."
    )
    assert _ATTRIBUTION_NOT_FROM_TOPIC in text, (
        f"[{prompt_name}] la regla no prohíbe explícitamente atribuir "
        f"desde el TEMA de la propia pregunta del coach — esto es "
        f"precisamente lo que causó el incidente real."
    )
    assert _ATTRIBUTION_ACT_WHEN_CLEAR in text, (
        f"[{prompt_name}] falta la salvaguarda de que, si ya es inequívoco, "
        f"se actúa directo — sin esto la regla degenera en preguntar "
        f"SIEMPRE, revirtiendo P1-CHAT-ACT-DONT-ASK."
    )


@pytest.mark.parametrize(
    "prompt_name",
    [
        "CHAT_SYSTEM_PROMPT_BASE",
        "CHAT_STREAM_SYSTEM_PROMPT_BASE",
        "CHAT_AGENT_INLINE_PROMPT",
        "CHAT_STREAM_INLINE_PROMPT",
    ],
)
def test_each_of_the_four_prompts_extends_no_announce_to_corrections(
    four_base_prompts: dict, prompt_name: str
):
    text = four_base_prompts[prompt_name]
    assert _ANNOUNCE_EXTENDED_TO_CORRECTIONS in text, (
        f"[{prompt_name}] la regla 'no anuncies sin ejecutar' no se "
        f"extendió explícitamente a corregir/borrar un registro existente "
        f"— sin esto el coach puede decir 'quedó corregido' sin haber "
        f"llamado `correct_consumed_meal`."
    )


def test_shared_constant_used_by_all_four_not_pasted_four_times(prompts_src: str):
    """Ancla estructural: la regla vive en `_CHAT_BREVITY_RULES` (constante
    ÚNICA) y los 4 prompts la referencian por `+ _CHAT_BREVITY_RULES`, no
    por 4 copias pegadas del texto. Si alguien 'arregla' 3 prompts a mano
    en vez de tocar la constante compartida, este test no lo detecta
    directamente, pero los 4 tests parametrizados de arriba sobre el
    VALOR evaluado de cada prompt sí — este test es una defensa adicional
    de que la ARQUITECTURA sigue siendo 1 constante, no 4 copias."""
    assert prompts_src.count(_ATTRIBUTION_MARKER) == 1, (
        "P1-CHAT-DIARY-CORRECT regresión: el marker de atribución aparece "
        "más de una vez en el source — señal de que alguien pegó la regla "
        "por separado en vez de compartir `_CHAT_BREVITY_RULES`, "
        "reabriendo el riesgo de 'arreglar 3 de 4'."
    )
    assert prompts_src.count("+ _CHAT_BREVITY_RULES") == 4, (
        "P1-CHAT-NARRATION-KEPT/P1-CHAT-DIARY-CORRECT regresión: alguno de "
        "los 4 prompts base dejó de sumar `_CHAT_BREVITY_RULES` — recibiría "
        "ni esta regla ni las anteriores."
    )


def test_marker_present_for_traceability(prompts_src: str):
    assert "P1-CHAT-DIARY-CORRECT" in prompts_src


# ===========================================================================
# 5. Guard anti-sobre-preguntar: nombrar la comida sigue disparando acción.
#    Esta es LA regresión que desharía P1-CHAT-ACT-DONT-ASK si la regla de
#    atribución se escribiera como "siempre pregunta el día/comida".
# ===========================================================================

def test_over_asking_guard_named_slot_still_acts(prompts_src: str):
    """Si el usuario NOMBRA la comida ('me comí el desayuno'), la regla de
    atribución no debe forzar una pregunta — el marcador de
    P1-CHAT-ACT-DONT-ASK (frase en pasado ⇒ MISMO TURNO, sin preguntar)
    debe seguir intacto Y la nueva regla debe decir explícitamente que
    cuando es inequívoco se actúa directo, no que SIEMPRE hay que
    preguntar el día/la comida."""
    # P1-CHAT-ACT-DONT-ASK: sigue exigiendo actuar en pasado sin pedir
    # permiso — la nueva regla NO debe haber diluido esto.
    assert "sigues actuando en el mismo turno sin pedir permiso" in prompts_src

    # La regla nueva explícitamente NO es un mandato universal de preguntar:
    # debe contener la cláusula de excepción para el caso inequívoco.
    attribution_start = prompts_src.find(_ATTRIBUTION_MARKER)
    assert attribution_start != -1
    attribution_block = prompts_src[attribution_start: attribution_start + 900]
    assert "no preguntes algo que ya sabes" in attribution_block, (
        "la regla de atribución debe declarar explícitamente que, si el "
        "usuario ya nombró el día/comida (o no hay otra lectura posible), "
        "NO hay que preguntar — de lo contrario el mandato es 'siempre "
        "pregunta', que rompe el caso donde el usuario SÍ fue explícito "
        "('me comí el desayuno') y así revierte P1-CHAT-ACT-DONT-ASK."
    )
    # Negativo: NO debe existir una frase tipo "siempre pregunta" a secas
    # (sin condicionarla a la ambigüedad) en el bloque de atribución.
    assert not re.search(r"siempre\s+pregunta", attribution_block, re.IGNORECASE), (
        "la regla de atribución no debe mandar preguntar incondicionalmente "
        "— solo cuando el día/comida esté genuinamente indeterminado."
    )


def test_log_consumed_meal_bullet_ties_slot_to_user_words_not_topic(prompts_src: str):
    """El bullet operativo de `log_consumed_meal` (ambos builders) también
    debe recordar que `days_ago`/`meal_type` salen de lo que el usuario
    dijo, no del tema — refuerzo a nivel de instrucción de tool, no solo de
    regla general."""
    assert prompts_src.count("[P1-CHAT-DIARY-CORRECT]") >= 2, (
        "se esperaba el marker [P1-CHAT-DIARY-CORRECT] en AMBOS bullets de "
        "`log_consumed_meal` (inline + stream)."
    )
