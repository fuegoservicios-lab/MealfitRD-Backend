"""[P3-PROTEIN-CAP] Tests para `_RESTRICTED_PROTEIN_KEYS` y la prohibición
de jamón de pavo procesado en `build_day_assignment_context`.

Bug observable (corridas 2026-05-05 múltiples):
  El planner asignaba proteínas distintas (Atún, Lentejas, Huevos) pero el
  day_generator LLM ignoraba la asignación e insertaba "pechuga de pavo en
  lonjas" / "jamón de pavo" en casi todas las comidas. Resultado:
    - 41+ lbs de jamón de pavo en lista mensual
    - Rechazo HIGH del revisor médico ("repetición excesiva, alto sodio,
      nitritos, falta de variedad")
    - Plan entregado degradado al usuario

Causa: `_RESTRICTED_PROTEIN_KEYS` listaba atún/salami/longaniza/chorizo pero
no incluía variantes de jamón de pavo. El `prohibited_block` no se generaba
para esa proteína, así que el LLM la insertaba libremente como complemento.

Fix:
  1. Añadidas variantes ('jamón de pavo', 'jamon de pavo', 'pavo en lonjas',
     'lonjas de pavo', 'pavo procesado', 'pavo molido') con label común.
  2. Lógica de dedup-by-label corregida: dos pasos (allowed pass + prohibited
     pass) con `strip_accents` para tolerar variantes con/sin tilde.
  3. Rule 12 del system prompt clarifica que jamón de pavo cuenta como
     embutido procesado, no como proteína fresca.

Cobertura:
  - Pavo NO asignado → prohibido en el block
  - Jamón de pavo CON tilde asignado → allowed
  - Jamon de pavo SIN tilde asignado → allowed (acento-tolerant)
  - Pavo genérico/fresh asignado → jamón de pavo SIGUE prohibido
  - Atún asignado → atún allowed pero jamón sigue prohibido
  - Otros restringidos (salami, etc.) sin regresión
"""
import pytest

from prompts.day_generator import (
    _RESTRICTED_PROTEIN_KEYS,
    build_day_assignment_context,
)


def _prohibited_section(ctx: str) -> str:
    """Extrae la sección 'PROHIBIDO ABSOLUTO' del context."""
    if "PROHIBIDO ABSOLUTO" not in ctx:
        return ""
    return ctx.split("PROHIBIDO ABSOLUTO")[1].split("---")[0].lower()


# ---------------------------------------------------------------------------
# 1. Configuración del set de keys
# ---------------------------------------------------------------------------
class TestRestrictedKeysSet:
    def test_pavo_processed_variants_present(self):
        """[P3-PROTEIN-CAP] El set debe incluir variantes de pavo procesado."""
        keys_lower = {k.lower() for k in _RESTRICTED_PROTEIN_KEYS}
        assert "jamón de pavo" in keys_lower
        assert "jamon de pavo" in keys_lower
        assert "pavo en lonjas" in keys_lower
        assert "lonjas de pavo" in keys_lower

    def test_legacy_restricted_proteins_unchanged(self):
        """No debemos perder las restricciones originales."""
        keys_lower = {k.lower() for k in _RESTRICTED_PROTEIN_KEYS}
        for legacy in ["atún", "atun", "salami", "longaniza", "chorizo"]:
            assert legacy in keys_lower

    def test_pavo_variants_share_label(self):
        """Todas las variantes de pavo procesado mapean al mismo label
        (para que la dedup funcione)."""
        pavo_keys = ["jamón de pavo", "jamon de pavo", "pavo en lonjas",
                     "lonjas de pavo", "pavo procesado"]
        labels = {_RESTRICTED_PROTEIN_KEYS[k] for k in pavo_keys}
        assert len(labels) == 1, (
            f"Variantes de pavo procesado deberían compartir label, "
            f"recibido {len(labels)} labels distintos: {labels}"
        )


# ---------------------------------------------------------------------------
# 2. build_day_assignment_context: prohibición de pavo procesado
# ---------------------------------------------------------------------------
class TestProhibitedBlockPavo:
    def test_pavo_no_asignado_aparece_prohibido(self):
        """Caso del incidente real: planner asigna pollo/lentejas/huevos.
        Jamón de pavo debe estar en prohibited_block."""
        ctx = build_day_assignment_context({
            "protein_pool": ["Pechuga de pollo fresca", "Lentejas", "Huevos"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=1)
        section = _prohibited_section(ctx)
        assert "jamón de pavo" in section or "jamon de pavo" in section, (
            f"Pavo procesado debió estar prohibido: {section}"
        )

    def test_jamon_de_pavo_asignado_NO_aparece_prohibido(self):
        """Si planner asigna explícitamente 'Jamón de pavo' (con tilde),
        no debe estar en prohibited."""
        ctx = build_day_assignment_context({
            "protein_pool": ["Jamón de pavo", "Lentejas"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=2)
        section = _prohibited_section(ctx)
        assert "jamón de pavo" not in section
        assert "jamon de pavo" not in section
        # Pero los demás restringidos siguen prohibidos
        assert "atún" in section or "atun" in section

    def test_jamon_de_pavo_asignado_sin_tilde_NO_prohibido(self):
        """Acento tolerance: 'Jamon de pavo' (sin tilde) también debe
        marcarse como allowed."""
        ctx = build_day_assignment_context({
            "protein_pool": ["Jamon de pavo"],  # Sin tilde
            "carb_pool": [], "fruit_pool": [],
        }, day_num=3)
        section = _prohibited_section(ctx)
        assert "jamón de pavo" not in section
        assert "jamon de pavo" not in section

    def test_pavo_generico_NO_libera_jamon_procesado(self):
        """Caso clave: planner asigna 'Pavo' (genérico = pechuga fresca).
        Jamón de pavo procesado SIGUE prohibido — no es lo mismo."""
        ctx = build_day_assignment_context({
            "protein_pool": ["Pavo", "Habichuelas Rojas", "Queso Blanco"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=4)
        section = _prohibited_section(ctx)
        assert "jamón de pavo" in section or "jamon de pavo" in section, (
            "Pavo genérico (fresh) NO debe permitir jamón de pavo procesado"
        )

    def test_pechuga_de_pavo_fresca_NO_libera_jamon(self):
        """Pechuga de pavo fresca asignada → jamón de pavo procesado
        sigue prohibido."""
        ctx = build_day_assignment_context({
            "protein_pool": ["Pechuga de pavo fresca", "Lentejas"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=5)
        section = _prohibited_section(ctx)
        # "pechuga de pavo" no contiene "jamón de pavo" como substring
        assert "jamón de pavo" in section or "jamon de pavo" in section


# ---------------------------------------------------------------------------
# 3. Sin regresión en otros restringidos
# ---------------------------------------------------------------------------
class TestNoRegressionOnOtherRestrictions:
    def test_atun_allowed_when_assigned(self):
        ctx = build_day_assignment_context({
            "protein_pool": ["Atún en agua", "Lentejas"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=1)
        section = _prohibited_section(ctx)
        assert "atún" not in section and "atun" not in section
        # Otros sigue prohibidos
        assert "salami" in section or "longaniza" in section

    def test_salami_prohibited_when_not_assigned(self):
        ctx = build_day_assignment_context({
            "protein_pool": ["Pollo", "Lentejas"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=1)
        section = _prohibited_section(ctx)
        assert "salami" in section

    def test_salami_allowed_when_assigned(self):
        ctx = build_day_assignment_context({
            "protein_pool": ["Salami dominicano", "Lentejas"],
            "carb_pool": [], "fruit_pool": [],
        }, day_num=1)
        section = _prohibited_section(ctx)
        assert "salami" not in section


# ---------------------------------------------------------------------------
# 4. System prompt mention de jamón de pavo
# ---------------------------------------------------------------------------
def test_system_prompt_clarifies_jamon_de_pavo_is_processed():
    """[P3-PROTEIN-CAP] El system prompt debe mencionar jamón de pavo en
    la rule 12 para que el LLM entienda que cuenta como embutido procesado."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT
    text = DAY_GENERATOR_SYSTEM_PROMPT.lower()
    assert "jamón de pavo" in text, (
        "System prompt rule 12 debe mencionar 'jamón de pavo' explícitamente"
    )
    # También debe clarificar que cuenta como procesado
    assert "procesad" in text  # 'procesado' o 'procesados'


# ---------------------------------------------------------------------------
# 5. Repro de las corridas 2026-05-05
# ---------------------------------------------------------------------------
def test_repro_corrida_2026_05_05_atun_lentejas_huevos_skel():
    """Reproduce el escenario del incidente: planner asignó
    ['Atún', 'Lentejas', 'Huevos'] pero LLM puso pavo procesado en todas
    las comidas. Post-fix: prohibited_block lista jamón de pavo
    explícitamente → LLM tiene constraint claro."""
    ctx = build_day_assignment_context({
        "protein_pool": ["Atún", "Lentejas", "Huevos"],
        "carb_pool": ["Yuca", "Arroz"], "fruit_pool": ["Limón"],
    }, day_num=1)

    section = _prohibited_section(ctx)
    # Jamón de pavo / pavo en lonjas / pavo procesado deben estar prohibidos
    assert "jamón de pavo" in section or "pavo en lonjas" in section, (
        "Repro fallido: jamón de pavo debe estar prohibido cuando el "
        "planner no lo asignó"
    )
    # Atún asignado → no en prohibited
    assert "atún" not in section and "atun" not in section
