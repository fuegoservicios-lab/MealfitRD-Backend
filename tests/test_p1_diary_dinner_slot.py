"""[P1-DIARY-DINNER-SLOT + P1-CHAT-FILLER-STRIP · 2026-07-30] Tres defectos del chat en una sesión
real del owner: el slot equivocado, un bloque que decía "Registrando…", y ~150 palabras para ~60 de
información.

Caso vivo (2026-07-29, diario del owner leído en la base):
    desayuno 10:00 (1070 kcal) · almuerzo 15:06 (525) · **snack 22:47 (420)** · cena AUSENTE
El usuario escribió "me comí dos sandwiches con queso gooda" a las 22:47 RD sin decir qué comida
era. El coach lo razonó por NARRATIVA — "ya veo que hoy has comido bastante, esto parece un snack
nocturno adicional" — teniendo delante los dos datos que lo resolvían: la hora y que la cena seguía
vacía. Confirmado por el usuario: era su cena.

⚠️ La derivación NO puede ser una banda horaria (`hora >= 19 → cena`): el prompt del coach protege
explícitamente al usuario de turno nocturno ("si tiene turno nocturno, las 5 AM es su cena"). Se usa
el HUECO desde su última comida principal, que sale de sus propios datos y se adapta a cualquier
horario.

Y el `Registrando…`: `P1-CHAT-NARRATION-KEPT` preserva la narración a propósito (descartarla perdía
contenido real que el usuario ya había visto en vivo), así que cada preámbulo queda grabado. El
prompt ya lo prohibía y llevaba dos días ignorado **porque estaba escrito como lista negra de
frases** ("lo registro", "lo anoto") y el modelo la esquiva con sinónimos. Un prompt es una petición;
el filtro es la garantía.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import agent
import tools

_BACKEND = Path(__file__).resolve().parents[1]
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_PROMPT = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")


# ══════════════════════════════════ 1. el slot ══════════════════════════════════

class _FakeAhora:
    """Fija "ahora" para medir el hueco sin depender del reloj real."""


def _fake_rows(slots_horas):
    """slots_horas: {'desayuno': datetime, ...} -> filas como las devuelve la query."""
    return [{"meal_type": k, "ultima": v} for k, v in slots_horas.items()]


@pytest.fixture()
def diario(monkeypatch):
    """Sustituye la consulta del diario por filas controladas."""
    estado = {"rows": []}

    def _fake_esq(sql, params=None, fetch_all=False, fetch_one=False):
        return estado["rows"]
    import db
    monkeypatch.setattr(db, "execute_sql_query", _fake_esq, raising=False)
    return estado


def _dt(h, m=0):
    from datetime import datetime, timezone, timedelta
    # 22:47 RD = 02:47 UTC del día siguiente; aquí se trabaja en UTC directo.
    return datetime.now(timezone.utc) - timedelta(hours=h, minutes=m)


def test_el_caso_vivo_snack_2247_pasa_a_cena(diario):
    """Reproducción del diario real: desayuno + almuerzo puestos, cena vacía, 7,7h de hueco."""
    diario["rows"] = _fake_rows({"desayuno": _dt(12, 47), "almuerzo": _dt(7, 41)})
    assert tools._rescue_dinner_slot("u1", "snack", 420, 0) == "cena"


def test_si_ya_ceno_un_picoteo_nocturno_sigue_siendo_snack(diario):
    """No convertimos en cena lo que viene DESPUÉS de una cena registrada."""
    diario["rows"] = _fake_rows({"desayuno": _dt(13), "almuerzo": _dt(8), "cena": _dt(2)})
    assert tools._rescue_dinner_slot("u1", "snack", 420, 0) == "snack"


def test_merienda_pegada_al_almuerzo_no_se_toca(diario):
    """La protección del falso positivo: una merienda a 1h del almuerzo es una merienda."""
    diario["rows"] = _fake_rows({"desayuno": _dt(6), "almuerzo": _dt(1)})
    assert tools._rescue_dinner_slot("u1", "merienda", 400, 0) == "merienda"


def test_dia_sin_ancla_no_se_toca(diario):
    """Sin desayuno ni almuerzo registrados no hay ancla en SU ritmo — no inventamos."""
    diario["rows"] = _fake_rows({"desayuno": _dt(9)})
    assert tools._rescue_dinner_slot("u1", "snack", 500, 0) == "snack"
    diario["rows"] = []
    assert tools._rescue_dinner_slot("u1", "snack", 500, 0) == "snack"


def test_picoteo_pequeno_no_es_cena(diario):
    diario["rows"] = _fake_rows({"desayuno": _dt(12), "almuerzo": _dt(7)})
    assert tools._rescue_dinner_slot("u1", "snack", 120, 0) == "snack"


def test_una_comida_principal_explicita_jamas_se_reescribe(diario):
    """Si el usuario (o el modelo) dijo `almuerzo`, no se toca: solo se rescata snack/merienda."""
    diario["rows"] = _fake_rows({"desayuno": _dt(12), "almuerzo": _dt(7)})
    for slot in ("desayuno", "almuerzo", "cena"):
        assert tools._rescue_dinner_slot("u1", slot, 600, 0) == slot


def test_backdate_no_se_toca(diario):
    """En un backdate la hora de AHORA no dice nada del hueco de aquel día."""
    diario["rows"] = _fake_rows({"desayuno": _dt(12), "almuerzo": _dt(7)})
    assert tools._rescue_dinner_slot("u1", "snack", 420, 1) == "snack"


def test_turno_nocturno_funciona_igual(diario):
    """El punto de usar el HUECO y no el reloj: alguien que desayuna a las 20:00 y almuerza a las
    02:00 y come a las 09:00 también tiene su cena rescatada, sin banda horaria que lo rompa."""
    diario["rows"] = _fake_rows({"desayuno": _dt(13), "almuerzo": _dt(7)})
    assert tools._rescue_dinner_slot("u1", "snack", 450, 0) == "cena"


def test_knob_de_rollback(diario, monkeypatch):
    diario["rows"] = _fake_rows({"desayuno": _dt(12), "almuerzo": _dt(7)})
    monkeypatch.setattr(tools, "_DINNER_RESCUE_ENABLED", False)
    assert tools._rescue_dinner_slot("u1", "snack", 420, 0) == "snack"


def test_fallo_de_db_no_rompe_el_registro(monkeypatch):
    """Fail-open: un hiccup de base no puede impedir que se guarde la comida."""
    import db

    def _boom(*a, **k):
        raise RuntimeError("db caida")
    monkeypatch.setattr(db, "execute_sql_query", _boom, raising=False)
    assert tools._rescue_dinner_slot("u1", "snack", 420, 0) == "snack"


def test_log_consumed_meal_invoca_el_rescate():
    """[P1-PLAN-LOTE-980] El dup-guard por franja ya no existe (2c6f3af0, 2026-10-12: se permiten
    varios platos en la misma comida). Lo que sigue en pie: log_consumed_meal pasa el slot pedido
    por el rescate antes de escribir."""
    assert "_rescue_dinner_slot(user_id, _meal_type" in _TOOLS


def test_la_regla_3_ya_no_es_una_lista_negra_de_frases():
    """La regla existía y el modelo la esquivó con sinónimos ("Registrando…" no estaba en la lista).
    Una regla anclada a frases se evade con otra frase — la enésima vez de esta clase en el repo."""
    i = _PROMPT.index("CERO TEXTO ANTES DE UNA HERRAMIENTA")
    seg = _PROMPT[i:i + 1200]
    assert "no va ningún texto delante de la llamada" in seg, (
        "la regla debe ser de COMPORTAMIENTO (cero texto antes de la tool), no un diccionario")
    for evasion in ("vamos a registrarlo", "dame un segundo", "registrando", "estimando"):
        assert evasion in seg.lower(), f"falta {evasion!r} como EJEMPLO de la evasión medida"


def test_existe_regla_de_derivacion_del_slot():
    """[P1-PLAN-LOTE-980] La regla 6-bis del prompt salió el 2026-10-12 (2c6f3af0): el slot ya no
    lo deriva el modelo, lo deriva el CÓDIGO con la hora y lo que falta por registrar. El caso vivo
    (dos sándwiches a las 22:47 con la cena vacía) tiene que seguir siendo cena."""
    from coach_day_context import franja_por_hora
    assert franja_por_hora(22 + 47 / 60, faltan=["cena"]) == "cena"
    assert franja_por_hora(22 + 47 / 60) == "cena"
    import agent
    assert "infer_current_meal_slot(" in Path(agent.__file__).read_text(encoding="utf-8")


def test_las_reglas_siguen_compartidas_por_los_4_prompts():
    """P1-CHAT-NARRATION-KEPT las hizo constante compartida para que una edición no arregle 3 de 4.
    Si alguien vuelve a copiarlas, este test cae."""
    assert _PROMPT.count("CERO TEXTO ANTES DE UNA HERRAMIENTA") == 1
    assert _PROMPT.count("_CHAT_BREVITY_RULES") >= 5      # 1 def + 4 usos


def test_reconstruccion_inmune_a_clases_de_otro_modulo():
    """[P1-CHAT-MSG-DUCK-TYPE] Estos 3 tests estuvieron verdes en aislamiento y ROJOS solo en la
    corrida completa con el arbol CONGELADO: otro test deja en sys.modules un langchain distinto y
    `isinstance(m, AIMessage)` deja de casar — todos los mensajes se saltan y el turno sale vacio.
    La reconstruccion ahora acepta tambien el duck-type `m.type == "ai"/"human"`, que es como
    fallaria en produccion con versiones mixtas. Se simula con clases FALSAS que no heredan de
    langchain."""

    class _FakeHuman:
        type = "human"
        def __init__(self, content): self.content = content

    class _FakeAI:
        type = "ai"
        def __init__(self, content): self.content = content

    msgs = [_FakeHuman("me comi dos sandwiches"),
            _FakeAI("Registrando..."),
            _FakeAI("Listo, quedo anotado como cena: 420 kcal.")]
    out = agent._build_final_content_from_messages(msgs)
    assert "420 kcal" in out, "con clases ajenas la reconstruccion no puede salir vacia"
    assert "Registrando..." not in out, "el filtro de relleno debe operar igual sobre duck-types"
