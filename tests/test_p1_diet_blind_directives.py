# [P1-DIET-BLIND-DIRECTIVES · 2026-08-08] El stack de prompts ordenaba proteína ANIMAL sin mirar
# la dieta declarada — y el modelo obedecía. Evidencia de producción (benchmark issue #9, corridas
# 31230621539/31232856541 + journal 01:53-02:00 UTC):
#
#   - El retry informado del perfil vegana_dm2 llevaba INYECTADO el rechazo del piso de proteína:
#     "Cada comida PRINCIPAL DEBE incluir una fuente animal de alta densidad (pollo, pescado,
#     cerdo, res, huevos, queso)" — una orden de violar la dieta dentro de la directiva que debía
#     corregirla. El intento 2 salió con pechuga de pollo/atún: el modelo obedeció la contradicción.
#   - El prompt BASE del day-gen nombra pollo/pescado/atún como rotación OBLIGATORIA de huevo
#     (§12), como "proteína fresca" y como patrón cultural del almuerzo — 4+ órdenes contrarias a
#     la directiva de dieta PRIORIDAD-1 (que perdía: atún en el desayuno vegetariano del intento 1,
#     con PROTEIN-POOL-SCRUB limpio en los 3 días).
#   - SLOT_POSITIVE_HINT sugería "pescado/pollo a la plancha" hasta en el rechazo de un plan vegano.
#
# Contrato: TODA directiva/sugerencia de fuentes de proteína que viaja a un prompt debe derivar de
# `constants.diet_protein_suggestions` (SSOT, canonicaliza vía canonicalize_diet_type — cero 4ª
# tabla, P1-DIET-CANON-SSOT) y las superficies balanced deben quedar BYTE-IDÉNTICAS (prompt-cache).
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

# Word-boundary (memoria: 'res'⊂'fresas', 'pollo'⊂'repollo' — jamás substring puro). Sobre texto
# acent-stripped lowercase.
_ANIMAL_RX = re.compile(
    r"\b(pollo|res|cerdo|pescado|atun|camarones|camaron|salmon|sardinas|pavo|carne|higado|"
    r"calamar|chivo|arenque|bacalao|jamon|salami|longaniza|chorizo|mariscos?)\b"
)
_DAIRY_EGG_RX = re.compile(r"\b(huevos?|queso|yogur|yogurt|lacteos?|claras?)\b")


def _norm(s: str) -> str:
    from constants import strip_accents
    return strip_accents((s or "").lower())


def _animal_hits(s: str) -> list:
    return sorted(set(_ANIMAL_RX.findall(_norm(s))))


# ---------------------------------------------------------------------------
# 1. SSOT: constants.diet_protein_suggestions
# ---------------------------------------------------------------------------

def test_ssot_vegan_sin_animal_ni_lacteo_ni_huevo():
    from constants import diet_protein_suggestions
    s = diet_protein_suggestions("vegana")  # forma del wizard, no canónica — el SSOT canonicaliza
    assert s, "vegan debe tener sugerencias propias"
    assert _animal_hits(s) == [], f"sugerencias veganas con animal: {_animal_hits(s)}"
    assert not _DAIRY_EGG_RX.search(_norm(s)), "sugerencias veganas con lácteo/huevo"
    assert "tofu" not in _norm(s), "P3-TOFU-REMOVE: no se vende, no se sugiere"


def test_ssot_vegetarian_permite_huevo_lacteo_pero_no_carne():
    from constants import diet_protein_suggestions
    s = diet_protein_suggestions("vegetariana")
    assert s, "vegetarian debe tener sugerencias propias"
    assert _animal_hits(s) == [], f"sugerencias vegetarianas con carne/pescado: {_animal_hits(s)}"
    assert _DAIRY_EGG_RX.search(_norm(s)), "vegetarian debe sugerir huevo/queso/yogur"


def test_ssot_balanced_devuelve_none():
    from constants import diet_protein_suggestions
    assert diet_protein_suggestions("balanced") is None
    assert diet_protein_suggestions(None) is None
    assert diet_protein_suggestions("") is None


# ---------------------------------------------------------------------------
# 2. Directiva del piso de proteína (el veneno del retry, probado en journal 02:00:10)
# ---------------------------------------------------------------------------

def test_protein_floor_directive_vegan_sin_fuente_animal():
    import graph_orchestrator as go
    txt = go._protein_floor_directive_text(
        "Día 1: 92g de 188g", 0.9, 188.0, "vegan", "adecuación proteica diaria")
    assert "fuente animal" not in _norm(txt)
    assert _animal_hits(txt) == [], f"directiva vegana ordena animal: {_animal_hits(txt)}"
    assert "188" in txt and "92g de 188g" in txt, "conserva los números medidos"


def test_protein_floor_directive_vegetarian_sin_carne_pescado():
    import graph_orchestrator as go
    txt = go._protein_floor_directive_text(
        "Día 3: 98g de 116g", 0.9, 116.0, "vegetariana", "adecuación proteica diaria")
    assert "fuente animal" not in _norm(txt)
    assert _animal_hits(txt) == [], f"directiva vegetariana ordena carne/pescado: {_animal_hits(txt)}"
    assert _DAIRY_EGG_RX.search(_norm(txt)), "vegetarian puede subir proteína con huevo/queso/yogur"


def test_protein_floor_directive_balanced_texto_exacto_preservado():
    # Byte-anchor: la rama balanced NO cambia ni una letra (es la que ven el 95% de los usuarios
    # y hay tests/telemetría ancladas a este texto).
    import graph_orchestrator as go
    txt = go._protein_floor_directive_text(
        "Día 2: 100g de 150g", 0.9, 150.0, None, "ganancia muscular")
    esperado = (
        "DÉFICIT DE PROTEÍNA (rechazo clínico — ganancia muscular): el plan no "
        "alcanza el piso de proteína en Día 2: 100g de 150g. Cada comida PRINCIPAL (almuerzo y "
        "cena) DEBE incluir una fuente animal de alta densidad (pollo, pescado, cerdo, "
        "res, huevos, queso) dimensionada en gramos para que cada día sume al menos "
        "90% del target (150g). NO dependas solo "
        "de leguminosas/almidón en las comidas principales."
    )
    assert txt == esperado


# ---------------------------------------------------------------------------
# 3. Prompt estático del day-gen parametrizado por dieta
# ---------------------------------------------------------------------------

def test_day_system_prompt_balanced_byte_identico():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, build_day_generator_system_prompt
    assert build_day_generator_system_prompt(None) == DAY_GENERATOR_SYSTEM_PROMPT
    assert build_day_generator_system_prompt("balanced") == DAY_GENERATOR_SYSTEM_PROMPT


def test_day_system_prompt_vegan_sin_rotacion_animal():
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, build_day_generator_system_prompt
    vegan = build_day_generator_system_prompt("vegan")
    assert vegan != DAY_GENERATOR_SYSTEM_PROMPT
    # Las SUGERENCIAS animales del balanced no pueden sobrevivir en el render veg* (las
    # PROHIBICIONES que mencionan atún/embutidos SÍ se quedan — son guardas, no sugerencias):
    assert "res molida magra" not in vegan, "lista de rotación de huevo balanced sobrevivió"
    assert "proteína (carne/pollo/pescado)" not in vegan, "patrón Bandera balanced sobrevivió"
    assert "para proteína fresca usa pollo" not in vegan
    assert "Locrio (pollo, cerdo, gandules" not in vegan
    assert "Pescado/pollo/cerdo a la plancha" not in vegan
    assert "Sopa ligera de pollo" not in vegan
    assert "Pinchitos sencillos (pollo/queso)" not in vegan
    # Y las prohibiciones de seguridad siguen vivas:
    assert "PROHIBIDO usar atún en más de 1 comida" in vegan


def test_day_system_prompt_vegan_conserva_caps_de_seguridad():
    # Los CAPS de seguridad (embutidos/sodio) NO se relajan al parametrizar: son guardas, no
    # sugerencias. Si esta línea desaparece del render vegano, el fix borró de más.
    from prompts.day_generator import build_day_generator_system_prompt
    vegan = build_day_generator_system_prompt("vegan")
    assert "PRESUPUESTO DE SODIO" in vegan
    assert "PREPARACIONES TRANSFORMADAS" in vegan


def test_day_system_prompt_vegetarian_sin_carne_en_rotacion():
    from prompts.day_generator import build_day_generator_system_prompt
    veg = build_day_generator_system_prompt("vegetarian")
    assert "res molida magra" not in veg
    assert "proteína (carne/pollo/pescado)" not in veg
    # vegetarian SÍ conserva huevo/queso/yogur como fuentes:
    assert _DAIRY_EGG_RX.search(_norm(veg))


# ---------------------------------------------------------------------------
# 4. protein_diversity_block del assignment context
# ---------------------------------------------------------------------------

def _esqueleto():
    return {
        "protein_pool": ["Habichuelas Negras", "Queso fresco"],
        "carb_pool": ["Arroz", "Batata"],
        "fruit_pool": ["Guineo"],
        "meal_types": ["Desayuno", "Almuerzo", "Cena", "Merienda"],
    }


def test_assignment_context_default_byte_identico():
    # Sin diet_type el contexto es byte-idéntico al actual (callers no migrados intactos).
    from prompts.day_generator import build_day_assignment_context
    a = build_day_assignment_context(_esqueleto(), 1)
    b = build_day_assignment_context(_esqueleto(), 1, diet_type=None)
    assert a == b


def test_assignment_context_vegan_diversidad_sin_animal():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_esqueleto(), 1, diet_type="vegan")
    # El bloque de diversidad no puede ordenar "proteína animal magra (pollo, pescado...)"
    assert "proteína animal magra" not in ctx
    assert "pechuga/muslo sin piel" not in ctx


def test_assignment_context_vegetarian_diversidad_sin_carne():
    from prompts.day_generator import build_day_assignment_context
    ctx = build_day_assignment_context(_esqueleto(), 1, diet_type="vegetariana")
    assert "proteína animal magra" not in ctx


# ---------------------------------------------------------------------------
# 5. SLOT_POSITIVE_HINT por dieta (fluye a rechazos del gate S1 y prompts de update)
# ---------------------------------------------------------------------------

def test_slot_hint_vegan_almuerzo_sin_pescado_carne():
    from constants import slot_positive_hint
    hint = slot_positive_hint("almuerzo", "vegan")
    assert hint, "vegan almuerzo debe tener hint propio"
    assert _animal_hits(hint) == [], f"hint vegano sugiere animal: {_animal_hits(hint)}"
    assert not _DAIRY_EGG_RX.search(_norm(hint)), "hint vegano sugiere lácteo/huevo"


def test_slot_hint_vegan_cena_y_merienda_sin_animal_ni_lacteo():
    from constants import slot_positive_hint
    for slot in ("cena", "merienda", "desayuno"):
        hint = slot_positive_hint(slot, "vegana")
        assert hint
        assert _animal_hits(hint) == [], f"hint vegano de {slot}: {_animal_hits(hint)}"
        assert not _DAIRY_EGG_RX.search(_norm(hint)), f"hint vegano de {slot} con lácteo/huevo"


def test_slot_hint_vegetarian_almuerzo_cena_sin_carne():
    from constants import slot_positive_hint
    for slot in ("almuerzo", "cena"):
        hint = slot_positive_hint(slot, "vegetarian")
        assert hint
        assert _animal_hits(hint) == [], f"hint vegetariano de {slot}: {_animal_hits(hint)}"


def test_slot_hint_balanced_passthrough_exacto():
    from constants import slot_positive_hint, SLOT_POSITIVE_HINT
    for slot, base in SLOT_POSITIVE_HINT.items():
        assert slot_positive_hint(slot, None) == base
        assert slot_positive_hint(slot, "balanced") == base
        # pescatarian: pescado permitido → hints base válidos
        assert slot_positive_hint(slot, "pescatarian") == base


# ---------------------------------------------------------------------------
# 6. Wiring: los call sites pasan la dieta (parser-based con tooltip-anchor)
# ---------------------------------------------------------------------------

_GO_SRC = None


def _go_src():
    global _GO_SRC
    if _GO_SRC is None:
        p = os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py")
        _GO_SRC = open(p, encoding="utf-8").read()
    return _GO_SRC


def test_wiring_slot_gate_usa_hint_por_dieta():
    # Los mensajes COMIDA FUERA DE HORARIO deben construirse con slot_positive_hint(slot, dieta),
    # no con el dict crudo SLOT_POSITIVE_HINT (que ignora la dieta). tooltip-anchor en el call site:
    # P1-DIET-BLIND-DIRECTIVES.
    src = _go_src()
    assert src.count("P1-DIET-BLIND-DIRECTIVES") >= 3, (
        "faltan anchors P1-DIET-BLIND-DIRECTIVES en graph_orchestrator (piso de proteína + "
        "gate de slots + day-gen system prompt por dieta)")


def test_wiring_daygen_system_prompt_por_dieta():
    # El SystemMessage del day-gen debe seleccionar el render por dieta (patrón
    # P2-VERIFIED-CATALOG-NOT-FILTERED: cache por variante, byte-idéntico para balanced).
    src = _go_src()
    assert "_day_system_instruction_for_diet" in src, (
        "el day-gen sigue usando el system prompt único diet-blind")


def test_fragmentos_balanced_existen_verbatim_en_la_constante():
    # Si un edit del prompt deriva un fragmento balanced, el .replace() del builder se vuelve
    # no-op EN SILENCIO y el render veg* vuelve a ordenar proteína animal (clase "guard huérfano
    # por regex", 3ª vez en memoria). Este test convierte ese drift en rojo de CI.
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, _DIET_FRAGMENT_TABLE
    assert len(_DIET_FRAGMENT_TABLE) >= 7
    for i, row in enumerate(_DIET_FRAGMENT_TABLE):
        assert row[0] in DAY_GENERATOR_SYSTEM_PROMPT, (
            f"fragmento balanced #{i} ya no existe verbatim en DAY_GENERATOR_SYSTEM_PROMPT — "
            f"actualiza _DIET_FRAGMENT_TABLE junto con el prompt: {row[0][:80]!r}")
        assert row[1] != row[0] and row[2] != row[0], f"fila #{i} sin variante real"


def test_micronutrients_prioridad_muscle_vegan_sin_fuente_animal():
    # micronutrients.py:850 — la línea PRIORIDAD de gain_muscle no puede ordenar "fuente animal"
    # a un usuario veg*.
    import micronutrients as mn
    src = open(os.path.join(os.path.dirname(__file__), "..", "micronutrients.py"),
               encoding="utf-8").read()
    assert "P1-DIET-BLIND-DIRECTIVES" in src, (
        "micronutrients.py: la línea PRIORIDAD de proteína sigue diet-blind (sin anchor)")
