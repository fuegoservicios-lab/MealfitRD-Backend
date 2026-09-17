"""[P1-PLAN-LOTE-56 · 2026-09-15] Lo que quedó abierto tras la batería del coach (LOTE-53).

1. Brevedad: topes de longitud por TIPO de pregunta (v5: brevedad 1,56-1,70/2; las respuestas de
   seguridad iban por 100-155 palabras). Medido con dos corridas en docs/coach_bateria_2026_09_15.md.
2. Purga de cuenta: `llm_usage_events` es la contabilidad del gasto de IA — se ANONIMIZA (user_id,
   plan_id y `corr` fuera), no se borra. Decisión delegada por el dueño.
3. Visión real (VPS, Gemini 3.8 Flash): el tope de 2.500 kcal por plato baja también las macros
   (test en `test_p1_scanner_audit.py`).
4. Gate: -n 2 por defecto y faulthandler_timeout=300 (el -n 3 se colgó al 99 % con la CPU a 0).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def test_marcador_del_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 56 and m.group(2) >= "2026-09-15"
    assert "[P1-PLAN-LOTE-56 · 2026-09-15]" in app


# ─────────────────────────────────────────── 1. topes de longitud
def test_los_topes_de_longitud_van_por_tipo_en_los_cuatro_prompts():
    from prompts import chat_agent as P
    for nombre in ("CHAT_SYSTEM_PROMPT_BASE", "CHAT_STREAM_SYSTEM_PROMPT_BASE",
                   "CHAT_AGENT_INLINE_PROMPT", "CHAT_STREAM_INLINE_PROMPT"):
        txt = getattr(P, nombre)
        for marca in ("TOPES POR TIPO", "MÁXIMO 65 palabras", "Queja o frustración", "MÁXIMO 90 palabras",
                      "MÁXIMO 120 palabras", "recorta en este orden", "NUNCA se recorta"):   # [P1-PLAN-LOTE-79] topes medidos
            assert marca in txt, f"{nombre}: falta «{marca}»"
    assert "Máximo unas 6 frases" not in P._CHAT_VOICE_RULES, "el tope de riesgo se cuenta en palabras"


# ─────────────────────────────────────────── 2. purga: anonimizar el gasto de IA
class _FakeDB:
    def __init__(self):
        self.writes: list = []

    def execute_sql_write(self, query, params=None, returning=False, lock_timeout_ms=None):
        self.writes.append((" ".join(str(query).split()), params))
        return [{"id": 1}, {"id": 2}] if returning else True


@pytest.fixture
def db(monkeypatch):
    import db_profiles
    fake = _FakeDB()
    monkeypatch.setattr(db_profiles, "execute_sql_write", fake.execute_sql_write)
    monkeypatch.setattr(db_profiles, "execute_sql_query", lambda *a, **k: [] if k.get("fetch_all") else None,
                        raising=False)
    monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)
    monkeypatch.setattr(db_profiles, "_purge_visual_diary_storage", lambda uid: 0, raising=False)
    return fake


_UID = "11111111-2222-3333-4444-555555555555"


@pytest.mark.parametrize("include_profile", [True, False])
def test_la_purga_anonimiza_el_gasto_de_ia_en_vez_de_borrarlo(db, include_profile):
    import db_profiles
    r = db_profiles.delete_account_data(_UID, include_profile=include_profile)
    qs = [q for q, _ in db.writes]
    assert not any(q.startswith("DELETE FROM llm_usage_events") for q in qs), "el gasto de IA no se borra"
    upd = [(q, p) for q, p in db.writes if q.startswith("UPDATE llm_usage_events")]
    assert len(upd) == 1
    q, p = upd[0]
    assert "user_id = NULL" in q and "plan_id = NULL" in q and "metadata - 'corr'" in q
    assert "WHERE user_id = %s" in q and p == (_UID,)
    assert r["anonymized"] == {"llm_usage_events": 2} and "llm_usage_events" not in r["deleted"]


def test_llm_usage_events_sale_de_la_lista_de_borrado():
    import db_profiles
    assert "llm_usage_events" not in db_profiles._USER_SCOPED_TABLES_USERID
    assert db_profiles._USER_SCOPED_TABLES_ANONYMIZE == ("llm_usage_events",)
    # Lo demás de telemetría sigue borrándose (no es anónimo).
    for t in ("api_usage", "pipeline_metrics", "user_facts", "meal_plans"):
        assert t in db_profiles._USER_SCOPED_TABLES_USERID


# ─────────────────────────────────────────── 1-bis. el día que nombra el usuario
@pytest.mark.parametrize("texto,esperado", [
    ("ayer me comí un chimi en la calle", 1),
    ("anoche cené 2 panes con queso", 1),
    ("antier me comí un sancocho", 2),
    ("anteayer almorcé arroz", 2),
    ("me comí un mangú con 2 huevos", None),
    ("hoy desayuné avena y ayer también", None),  # dos días: no decide el guard
    ("desayuné avena con guineo", None),          # «desayuné» no contiene «ayer» como palabra
])
def test_el_dia_nombrado_por_el_usuario(texto, esperado):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    import agent as A
    msgs = [HumanMessage(content="hola"), AIMessage(content="¡Hola!"), HumanMessage(content=texto),
            SystemMessage(content="nota interna de ayer")]
    assert A._days_ago_named_by_user(msgs) == esperado


def test_el_guard_de_dias_va_despues_del_override_de_identidad():
    """Batería v6 B, caso B6: days_ago=0 con «ayer» y la respuesta afirmaba «la cena de ayer»."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i_uid = src.index('tool_args["user_id"] = _trusted_uid')
    i_dia = src.index("_days_ago_named_by_user(state.get(\"messages\") or [])")
    assert i_uid < i_dia < src.index('tool_result = ""', i_uid)
    assert 'tool_name == "log_consumed_meal" and not tool_args.get("days_ago")' in src


# ─────────────────────────────────────────── 4. gate
def test_el_gate_corre_a_dos_workers_con_volcado_del_test_colgado():
    src = (_BACKEND / "scripts" / "run_ci.ps1").read_text(encoding="utf-8")
    assert 'if (-not $workers) { $workers = "2" }' in src
    assert "faulthandler_timeout=300" in src and "PYTEST_ADDOPTS" in src
