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


def test_corre_antes_del_dup_guard():
    """Si reclasifica a cena, la cena tiene que pasar por SU comprobación de duplicado. Al revés se
    colaría una segunda cena en el mismo día."""
    i_resc = _TOOLS.index("_rescue_dinner_slot(user_id, _meal_type")
    i_guard = _TOOLS.index("if _meal_type in _CONSUMED_MAIN_MEAL_TYPES and not force:")
    assert i_resc < i_guard, "el rescate debe correr ANTES del dup-guard"


def test_el_slot_escrito_se_loguea_a_info():
    """El único log de qué slot se escribía era `logger.debug`, invisible en producción — no había
    forma de diagnosticar esta clase desde el journal."""
    assert 'logger.info(f"🍽️ [DIARY] slot=' in _TOOLS


def test_knobs_registrados():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    for k in ("MEALFIT_DIARY_DINNER_RESCUE", "MEALFIT_DIARY_DINNER_RESCUE_MIN_KCAL",
              "MEALFIT_DIARY_DINNER_RESCUE_MIN_GAP_H"):
        assert k in snap, f"{k} sin registrar en el registry de knobs"


# ══════════════════════════════ 2. el "Registrando…" ══════════════════════════════

@pytest.mark.parametrize("t", [
    "Registrando...", "Registrando…", "Estimando macros...", "Calculando...",
    "Dame un segundo...", "Un momento...", "Vamos a registrarlo.", "Estimado aproximado:",
])
def test_relleno_detectado(t):
    assert agent._is_pure_filler(t), f"{t!r} es relleno de espera"


@pytest.mark.parametrize("t", [
    "Dame un segundo para estimar los macros de 2 sándwiches (asumo 40 g de queso).",
    "Listo, quedó anotado como cena. Cierras el día en 2015 kcal de 2050.",
    "Son casi las 11 de la noche y ya veo que hoy has comido bastante.",
    "Ojo: las grasas se te fueron altas (92g de ~57g meta).",
    "Registrando tu cena de 420 kcal",
])
def test_contenido_nunca_se_descarta(t):
    """Descartar de más es el bug que P1-CHAT-NARRATION-KEPT cerró: cualquier bloque con cifras es
    información (de dónde salió el estimado, cuánto llevas, qué se pasó)."""
    assert not agent._is_pure_filler(t), f"{t!r} lleva información y NO puede descartarse"


def test_el_ultimo_bloque_es_intocable():
    """Aunque el turno acabara en algo que parece relleno, es la respuesta real: nunca se tira."""
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = [HumanMessage(content="me comí dos sándwiches"),
            AIMessage(content="Registrando..."),
            AIMessage(content="Calculando...")]
    out = agent._build_final_content_from_messages(msgs)
    assert "Calculando..." in out, "el último bloque no se descarta jamás"
    assert "Registrando..." not in out, "el relleno intermedio sí"


def test_el_turno_del_owner_queda_limpio():
    """El turno real, reconstruido: se va el 'Registrando…' y se queda todo lo demás."""
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = [
        HumanMessage(content="me comi dos sandwiches con queso gooda"),
        AIMessage(content="Son casi las 11 de la noche. 2 sándwiches (~80 g queso): ~420 kcal, "
                          "20 g proteína, 36 g carbohidratos, 22 g grasas"),
        AIMessage(content="Registrando..."),
        AIMessage(content="Listo, quedó anotado como cena. Cierras el día en ~2,015 kcal "
                          "de 2,050 kcal."),
    ]
    out = agent._build_final_content_from_messages(msgs)
    assert "Registrando..." not in out
    assert "420 kcal" in out and "2,015" in out


def test_knob_de_rollback_del_filtro(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    monkeypatch.setattr(agent, "FILLER_STRIP_ENABLED", False)
    msgs = [HumanMessage(content="x"), AIMessage(content="Registrando..."),
            AIMessage(content="Listo.")]
    assert "Registrando..." in agent._build_final_content_from_messages(msgs)


# ═══════════════════════════════ 3. las reglas del prompt ═══════════════════════════════

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
    i = _PROMPT.index("EL SLOT SE DERIVA DE LOS DATOS")
    seg = _PROMPT[i:i + 1200]
    assert "QUÉ COMIDAS DEL DÍA SIGUEN SIN REGISTRAR" in seg
    assert "22:47" in seg, "el caso vivo hace la regla concreta en vez de abstracta"
    assert "no es el cajón por defecto" in seg, (
        "hay que decir explícitamente que snack NO es el default de 'no sé cuál era'")


def test_las_reglas_siguen_compartidas_por_los_4_prompts():
    """P1-CHAT-NARRATION-KEPT las hizo constante compartida para que una edición no arregle 3 de 4.
    Si alguien vuelve a copiarlas, este test cae."""
    assert _PROMPT.count("CERO TEXTO ANTES DE UNA HERRAMIENTA") == 1
    assert _PROMPT.count("_CHAT_BREVITY_RULES") >= 5      # 1 def + 4 usos
