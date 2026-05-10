"""[P6-EGGS-CAP] Tests para el cap implícito de huevos en el prompt
del day_generator.

Bug observable (corrida 2026-05-05 14:01):
  Día 1 emitió "5 huevos grandes" como ingrediente. Acumulado del ciclo
  de 3 días: 11.5 huevos. Reviewer médico rechazó:
    "Carga excesiva de huevos (11.5 unidades en el ciclo de 3 días)"
  Pipeline costó retry: ~210s extra (skeleton + 3 días + assemble + review).

Causa raíz:
  El prompt del day_generator no tenía cap explícito sobre cantidad de
  huevos. El reviewer médico tiene su threshold (~3-4/día); cuando el LLM
  generaba 5+ huevos en una comida, el revisor flageaba y forzaba retry.

Fix:
  Añadir cap explícito al prompt del day_generator:
    "MÁXIMO 3 unidades enteras EN ESTE DÍA + máximo 6 claras"
  Para gain_muscle, claras son alternativa sin colesterol.
  Reduce retries por sobre-carga de huevos.

Cobertura:
  - El prompt incluye el cap explícito y mencionsa el reviewer
  - El cap está en la sección 12 (SEGURIDAD ALIMENTARIA)
  - El número 3 es el cap explícito (testeable por substring)
  - Mención de claras como alternativa para alta proteína
"""
import pytest


def test_day_generator_prompt_has_eggs_cap():
    """[P6-EGGS-CAP] El prompt debe incluir cap explícito de 3 huevos
    enteros/día para evitar rechazos por carga excesiva."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    # Cap explícito presente
    assert "HUEVOS" in DAY_GENERATOR_SYSTEM_PROMPT
    assert "MÁXIMO 3" in DAY_GENERATOR_SYSTEM_PROMPT, (
        "El cap de 3 huevos enteros debe estar explícito en el prompt"
    )


def test_eggs_cap_mentions_claras_as_alternative():
    """Para gain_muscle, claras son alternativa válida sin colesterol.
    El prompt debe enseñar al LLM esta opción para no quedarse corto en
    proteína cuando se aplica el cap."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    assert "claras" in DAY_GENERATOR_SYSTEM_PROMPT.lower() or \
           "CLARAS" in DAY_GENERATOR_SYSTEM_PROMPT, (
        "El prompt debe mencionar claras como alternativa al cap de huevos"
    )
    # Cap de claras (6 es lo que pusimos)
    assert "6 claras" in DAY_GENERATOR_SYSTEM_PROMPT.lower() or \
           "máximo 6 claras" in DAY_GENERATOR_SYSTEM_PROMPT.lower(), (
        "El prompt debe especificar cap de claras (no ilimitado)"
    )


def test_eggs_cap_explains_reviewer_rejection_reason():
    """El prompt debe explicar el "porqué" — que el reviewer médico
    rechaza por exceso. Sin contexto, el LLM podría re-emitir 5+ huevos
    pensando que es OK para gain_muscle."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    # Mencionar que reviewer rechaza
    assert "revisor médico" in DAY_GENERATOR_SYSTEM_PROMPT.lower()
    # Específicamente sobre huevos: la palabra "carga excesiva" o "colesterol"
    eggs_section_lower = DAY_GENERATOR_SYSTEM_PROMPT.lower()
    eggs_idx = eggs_section_lower.find("huevos — cap")
    if eggs_idx == -1:
        eggs_idx = eggs_section_lower.find("huevos")
    # Buscar dentro de los siguientes 500 chars
    section = DAY_GENERATOR_SYSTEM_PROMPT[eggs_idx:eggs_idx + 600].lower()
    has_rejection_context = (
        "carga excesiva" in section
        or "colesterol" in section
        or "rechaza" in section
    )
    assert has_rejection_context, (
        "Sección de huevos debe explicar por qué el reviewer rechaza "
        "(carga excesiva / colesterol)"
    )


def test_eggs_cap_in_safety_section():
    """El cap debe vivir en la sección 12 (SEGURIDAD ALIMENTARIA — CAPS),
    junto con embutidos, atún, galletas — coherencia organizacional."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    # Encontrar dónde empieza la sección 12 y dónde termina (siguiente número)
    s12_idx = DAY_GENERATOR_SYSTEM_PROMPT.find("12. SEGURIDAD ALIMENTARIA")
    s13_idx = DAY_GENERATOR_SYSTEM_PROMPT.find("13. ", s12_idx)
    assert s12_idx != -1, "Sección 12 debe existir"

    s12_block = DAY_GENERATOR_SYSTEM_PROMPT[s12_idx:s13_idx if s13_idx > 0 else None]
    assert "HUEVOS" in s12_block, (
        "El cap de huevos debe vivir dentro de la sección 12 (CAPS de seguridad)"
    )


def test_eggs_cap_distribution_guidance():
    """El prompt debe sugerir distribución típica para que el LLM no
    concentre todos los huevos en una sola comida."""
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    # Buscar mención de distribución (desayuno + otra comida o similar)
    eggs_section_idx = DAY_GENERATOR_SYSTEM_PROMPT.find("HUEVOS")
    section = DAY_GENERATOR_SYSTEM_PROMPT[eggs_section_idx:eggs_section_idx + 600].lower()
    has_distribution_hint = (
        "desayuno" in section or "repartición" in section or "distribución" in section
    )
    assert has_distribution_hint, (
        "El cap debe incluir hint de distribución para evitar concentración"
    )
