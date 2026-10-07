import ast
from pathlib import Path

from health_evidence import HEALTH_EVIDENCE_RULES

ROOT = Path(__file__).resolve().parents[1]


def test_all_four_chat_modes_apply_evidence_after_their_style_rules():
    tree = ast.parse((ROOT / 'prompts/chat_agent.py').read_text(encoding='utf-8'))
    names = {'CHAT_SYSTEM_PROMPT_BASE', 'CHAT_STREAM_SYSTEM_PROMPT_BASE',
             'CHAT_AGENT_INLINE_PROMPT', 'CHAT_STREAM_INLINE_PROMPT'}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            found.update(t.id for t in node.targets if isinstance(t, ast.Name) and t.id in names)
            assert isinstance(node.value, ast.BinOp)
            assert isinstance(node.value.right, ast.Name) and node.value.right.id == 'HEALTH_EVIDENCE_RULES'
    assert found == names
    needed = names | {'_CHAT_BREVITY_RULES', '_CHAT_VOICE_RULES', '_CHAT_RESOLVE_RULES'}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in needed for target in node.targets)]
    namespace = {'HEALTH_EVIDENCE_RULES': HEALTH_EVIDENCE_RULES}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<chat-modes>', 'exec'), namespace)
    for name in names:
        assert HEALTH_EVIDENCE_RULES in namespace[name]
        assert namespace[name].endswith(HEALTH_EVIDENCE_RULES)


def test_old_unsupported_digestive_orders_are_removed():
    source = (ROOT / 'prompts/chat_agent.py').read_text(encoding='utf-8')
    assert 'Toma 5 horas digerir' not in source
    assert 'Nutriólogo Crítico' not in source
    assert 'guiarlos como nutricionista profesional' not in source


def test_sources_have_scope_and_do_not_turn_partial_diary_into_a_diagnosis():
    assert 'https://pubmed.ncbi.nlm.nih.gov/2305711/' in HEALTH_EVIDENCE_RULES
    assert 'https://fdc.nal.usda.gov/' in HEALTH_EVIDENCE_RULES
    assert 'https://ods.od.nih.gov/' in HEALTH_EVIDENCE_RULES
    assert 'No extrapoles una dosis deportiva' in HEALTH_EVIDENCE_RULES
    assert 'no prueban una carencia' in HEALTH_EVIDENCE_RULES
    assert 'no es una fuente médica' in HEALTH_EVIDENCE_RULES


def test_live_voice_uses_same_scope_but_never_reads_urls_aloud():
    tree = ast.parse((ROOT / 'coach_live.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'instrucciones')
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {'HEALTH_EVIDENCE_RULES': HEALTH_EVIDENCE_RULES}
    exec(compile(module, '<voice-instructions>', 'exec'), namespace)
    instructions = namespace['instrucciones']('es-DO')
    assert HEALTH_EVIDENCE_RULES in instructions
    assert 'No leas URLs en voz alta' in instructions
