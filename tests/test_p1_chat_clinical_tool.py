"""[P1-CHAT-CLINICAL-TOOL · 2026-07-12] Perfil clínico bajo demanda + dieta de tokens del plan.

Pedido del owner: el agente debe poder hablar de sus laboratorios. Diseño: los
labs NO se inyectan en cada turno (dato más sensible del sistema + tokens);
`check_clinical_profile` los trae SOLO cuando el usuario pregunta, reusando el
builder SSOT de plan-gen (`build_clinical_profile_context` — mismos umbrales y
flags que el generador y el reviewer médico ven).

Mismo turno (audit del plan vivo): el plan podado que viaja al chat arrastraba
reportes internos de QA (dish_quality_report, variety_report, _review_issues_raw,
lecciones de chunks…) que pueden ser KBs por turno — añadidos a
_CHAT_PLAN_PRUNE_KEYS. Lo user-facing (micronutrient_report/advice, insights,
budget_reconciliation, goal_eta, _review_disclaimer) se queda: el agente SÍ
conoce micros, duración y presupuesto por esa vía.
tooltip-anchor: P1-CHAT-CLINICAL-TOOL
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

from tools import agent_tools, check_clinical_profile  # noqa: E402
from agent import _CHAT_PLAN_PRUNE_KEYS, _prune_plan_for_chat  # noqa: E402


def test_tool_registered():
    names = [t.name for t in agent_tools]
    assert "check_clinical_profile" in names
    assert "user_id" in check_clinical_profile.args


def test_tool_reuses_ssot_builder():
    with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.find("def check_clinical_profile(")
    body = src[i:i + 3500]
    assert "from prompts.plan_generator import build_clinical_profile_context" in body, \
        "mismos umbrales/flags que plan-gen y el reviewer médico (SSOT)"
    assert "no sustituye" in body.lower() or "NO sustituye" in body, \
        "el disclaimer médico es parte del contrato de la tool"


def test_tool_empty_profile_invites(monkeypatch):
    import db

    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {}})
    out = check_clinical_profile.func("11111111-1111-1111-1111-111111111111")
    assert "NO ha llenado" in out and "Perfil Clínico Avanzado" in out


def test_tool_formats_labs(monkeypatch):
    import db

    # OJO: los nombres de campos son los del builder SSOT (plan_generator._lab):
    # glucosa_ayunas / hba1c / ldl / tfg / vitamina_d — no traducciones inglesas.
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {
        "health_profile": {"clinical_profile": {"labs": {"glucosa_ayunas": 92, "hba1c": 5.4}}}
    })
    out = check_clinical_profile.func("11111111-1111-1111-1111-111111111111")
    assert "5.4" in out, "los valores del lab deben viajar formateados al agente"
    assert "profesional" in out, "cierre con recomendación de confirmar con profesional"


def test_prune_sheds_internal_qa_keeps_user_facing():
    for k in ("_review_issues_raw", "dish_quality_report", "variety_report",
              "_recipe_coherence_errors", "_recent_chunk_lessons"):
        assert k in _CHAT_PLAN_PRUNE_KEYS, f"{k} es QA interno — no debe viajar al chat"
    plan = {
        "micronutrient_report": {"x": 1}, "budget_reconciliation": {"tier": "moderado"},
        "goal_eta": {"pace": "decidido"}, "insights": [], "calories": 2100,
        "dish_quality_report": {"heavy": "x" * 100}, "variety_report": {"heavy": True},
    }
    pruned = _prune_plan_for_chat(plan)
    assert "micronutrient_report" in pruned and "budget_reconciliation" in pruned \
        and "goal_eta" in pruned, "lo user-facing sobrevive (micros/presupuesto/meta)"
    assert "dish_quality_report" not in pruned and "variety_report" not in pruned


def test_prompts_teach_clinical_tool():
    with open(os.path.join(_BACKEND, "prompts", "chat_agent.py"), encoding="utf-8") as f:
        prompts = f.read()
    assert prompts.count("check_clinical_profile") >= 2, "ambos builders"
    assert "NO diagnostiques" in prompts


def test_doc_table_has_row_14():
    with open(os.path.join(_BACKEND, "docs", "agent_tools_user_id_table.md"),
              encoding="utf-8") as f:
        doc = f.read()
    assert "`check_clinical_profile`" in doc and "Las 14 tools" in doc
