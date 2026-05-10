"""[P6-CRITIQUE-VS-SKELETON] Tests para la regla de precedencia inviolable
añadida al corrector LLM (self_critique_node + surgical_marker_regen_node).

Bug observable (corrida 2026-05-05 14:53-15:01):
  Self-critique detectó "Día 3: almuerzo y cena comparten proteína (pavo)".
  Critique pidió: "cambiar la proteína de la cena para coherencia de slot".
  Corrector OBEDECIÓ → removió pavo de la cena.
  Skeleton fidelity validator: "Día 3 omitió pavo asignado" → HIGH.
  Reviewer rechazó. Retry attempt #2 idéntico fallo. Max attempts → plan
  con violación tolerado y entregado al usuario.

Causa raíz:
  Conflicto mutuamente excluyente entre dos defensas:
    - Skeleton fidelity: "Día 3 debe usar pavo (asignado por planner)"
    - Slot coherence: "almuerzo y cena no deben compartir proteína"
  Cuando planner asigna 1 protein/día, no hay solución que satisfaga
  ambas. El corrector no sabía cuál priorizar.

Fix:
  Regla explícita en el corrector prompt: skeleton GANA. Si critique
  pide cambiar protein asignada, mantener y resolver slot por otro
  medio (cambiar carbo, técnica, vegetal, presentación).

Cobertura:
  - Prompt incluye "REGLA DE PRECEDENCIA INVIOLABLE"
  - Menciona explícitamente "skeleton" y "ASIGNACIÓN" como hard constraint
  - Sugiere alternativas: cambiar carbo / técnica / vegetal / presentación
  - Aplica a ambos nodos (self_critique_node y surgical_marker_regen_node)
"""
import inspect


# ---------------------------------------------------------------------------
# 1. self_critique_node prompt incluye la regla de precedencia
# ---------------------------------------------------------------------------
class TestSelfCritiquePrecedenceRule:
    def test_prompt_has_precedence_rule_marker(self):
        """[P6-CRITIQUE-VS-SKELETON] El prompt del corrector debe incluir
        la sección 'REGLA DE PRECEDENCIA INVIOLABLE'."""
        import graph_orchestrator as go
        src = inspect.getsource(go.self_critique_node)
        assert "REGLA DE PRECEDENCIA INVIOLABLE" in src

    def test_prompt_marks_skeleton_as_hard_constraint(self):
        """El prompt debe explicar que la asignación del planner es
        HARD CONSTRAINT — no soft hint."""
        import graph_orchestrator as go
        src = inspect.getsource(go.self_critique_node)
        # Buscar mención de hard constraint sobre el skeleton
        assert "HARD CONSTRAINT" in src
        assert "ASIGNACIÓN DEL PLANIFICADOR" in src

    def test_prompt_explicitly_warns_against_changing_assigned_protein(self):
        """El prompt debe instruir al LLM: NO cambiar la proteína
        asignada aunque el critique lo pida."""
        import graph_orchestrator as go
        src = inspect.getsource(go.self_critique_node)
        # Verificar que menciona el escenario y la solución
        prompt_section = src[src.find("REGLA DE PRECEDENCIA"):src.find("REGLA BIDIRECCIONAL")]
        assert "MANTÉN" in prompt_section, (
            "Prompt debe instruir a MANTENER la proteína asignada"
        )
        assert "slot" in prompt_section.lower(), (
            "Prompt debe mencionar slot coherence como caso de conflicto"
        )

    def test_prompt_offers_alternative_dimensions_to_change(self):
        """El prompt debe sugerir alternativas para resolver slot
        coherence sin tocar la proteína: carbohidrato, técnica, vegetal,
        presentación."""
        import graph_orchestrator as go
        src = inspect.getsource(go.self_critique_node)
        prompt_section = src[src.find("REGLA DE PRECEDENCIA"):src.find("REGLA BIDIRECCIONAL")]
        # Al menos 3 de las 4 dimensiones alternativas deben mencionarse
        alternatives = ["carbohidrato", "técnica", "vegetal", "presentación"]
        mentioned = sum(1 for a in alternatives if a.lower() in prompt_section.lower())
        assert mentioned >= 3, (
            f"Prompt debe sugerir ≥3 dimensiones alternativas para resolver slot. "
            f"Mencionadas: {mentioned}/4. Sin alternativas, el LLM no tiene path "
            f"de escape al conflicto y caerá en el mismo loop."
        )


# ---------------------------------------------------------------------------
# 2. surgical_marker_regen_node prompt incluye la misma regla
# ---------------------------------------------------------------------------
class TestSurgicalMarkerRegenPrecedenceRule:
    def test_surgical_regen_prompt_has_precedence_rule(self):
        """surgical_marker_regen también recibe el `original_issue` que
        pudo originarse en el critique. Misma regla aplica."""
        import graph_orchestrator as go
        src = inspect.getsource(go.surgical_marker_regen_node)
        assert "REGLA DE PRECEDENCIA INVIOLABLE" in src

    def test_surgical_regen_prompt_marks_skeleton_hard_constraint(self):
        import graph_orchestrator as go
        src = inspect.getsource(go.surgical_marker_regen_node)
        assert "HARD CONSTRAINT" in src
        assert "ASIGNACIÓN DEL PLANIFICADOR" in src


# ---------------------------------------------------------------------------
# 3. Sanity: ambas rutas tienen marker de la fix
# ---------------------------------------------------------------------------
def test_both_correctors_marked_with_fix_label():
    """Sanity guard: el marker `P6-CRITIQUE-VS-SKELETON` debe estar en
    los comentarios de AMBAS rutas para alertar regresión."""
    import graph_orchestrator as go
    src_self = inspect.getsource(go.self_critique_node)
    src_surgical = inspect.getsource(go.surgical_marker_regen_node)
    assert "P6-CRITIQUE-VS-SKELETON" in src_self, (
        "self_critique_node debe estar marcado con P6-CRITIQUE-VS-SKELETON"
    )
    assert "P6-CRITIQUE-VS-SKELETON" in src_surgical, (
        "surgical_marker_regen_node debe estar marcado con P6-CRITIQUE-VS-SKELETON"
    )


# ---------------------------------------------------------------------------
# 4. Sanity: la regla viene ANTES de la regla bidireccional
# ---------------------------------------------------------------------------
def test_precedence_rule_comes_before_bidirectional_in_self_critique():
    """Orden importa para LLM: la regla de precedencia (más específica
    sobre conflict resolution) debe venir ANTES de la regla bidireccional
    (más genérica). Sin orden correcto, el LLM puede priorizar mal."""
    import graph_orchestrator as go
    src = inspect.getsource(go.self_critique_node)
    precedence_pos = src.find("REGLA DE PRECEDENCIA INVIOLABLE")
    bidirectional_pos = src.find("REGLA BIDIRECCIONAL CRÍTICA")
    assert precedence_pos > 0 and bidirectional_pos > 0
    assert precedence_pos < bidirectional_pos, (
        "Precedence rule debe venir ANTES de bidirectional para guiar al "
        "LLM en orden lógico (conflicto skeleton↔critique antes que "
        "consistencia recipe↔ingredients)."
    )


# ---------------------------------------------------------------------------
# 5. Repro corrida 14:53 — verificar que el prompt cubre el caso exacto
# ---------------------------------------------------------------------------
def test_repro_corrida_14_53_pavo_skeleton_violation():
    """Repro escenario: critique pide 'cambiar proteína de cena (Pavo)
    por res/pescado'. Prompt debe instruir mantener pavo + resolver
    slot por otro medio."""
    import graph_orchestrator as go
    src = inspect.getsource(go.self_critique_node)
    section = src[src.find("REGLA DE PRECEDENCIA"):src.find("REGLA BIDIRECCIONAL")]
    # El escenario debe estar cubierto:
    # - menciona "proteína" (target del critique)
    # - menciona "slot" (causa de la sugerencia)
    # - menciona "MANTÉN" (acción correcta)
    # - sugiere "carbohidrato" como alt (path de escape común)
    assert "proteína" in section.lower()
    assert "slot" in section.lower()
    assert "mantén" in section.lower()
    assert "carbohidrato" in section.lower() or "carbo" in section.lower()
