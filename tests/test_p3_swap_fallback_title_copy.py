"""[P3-SWAP-FALLBACK-TITLE-COPY · 2026-05-22] Cuando el swap-meal cae al
fallback de seguridad (LLM agotó retries sin pasar validators), el
title que el user veía en la card del Dashboard era hostil:

  ``Opción Segura: 1 Cabeza (~400G) Lechuga Y 1 Cabeza (~500G) Brócoli``

Tres bugs concretos:

  1. ``fallback_ing`` venía del empty-pantry-fallback que prefería
     ``display_string`` sobre ``name`` — exponía las cantidades formateadas
     del aggregator (peso, paréntesis, mayúsculas) crudas al usuario.
  2. El title aplicaba ``.title()`` sobre los strings ya formateados, lo
     que capitalizaba unidades (``g`` → ``G``) y el ``"y"`` mayúscula del
     join — visualmente roto.
  3. El prefijo ``"Opción Segura:"`` + desc ``"autogenerado como medida
     de seguridad..."`` era jargon técnico que sonaba a error del sistema
     en vez de una sugerencia amigable.

Caso productivo verificado 2026-05-22 23:04-23:05: swap de "Crema de
Auyama" con ``swap_reason=variety`` falló por ``P1-SWAP-RECIPE-COHERENCE``
(LLM mencionó "dorado" en receta sin listar "pescado" en ingredients,
3 intentos), cayó al fallback con título "Opción Segura: 1 Cabeza (~400G)
Lechuga Y 1 Cabeza (~500G) Brócoli". User feedback: "el titulo de primera
vista lo veo mal".

Fix:

  1. ``agent.py::swap_meal`` (P1-SWAP-EMPTY-PANTRY-FALLBACK reader)
     prefiere ``name`` sobre ``display_string`` → ``clean_ingredients``
     contiene tokens limpios ("Lechuga", "Brócoli") tanto para el prompt
     del LLM como para el fallback title.
  2. El title construye ``f"{meal_type} con {ing1} y {ing2}"`` sin
     ``.title()`` (preserva acentos + unidades como están) y con guard
     defensivo para tokens vacíos.
  3. La desc pasa a copy cálido y simple — "Plato simple armado con
     ingredientes que tienes en casa".

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_fallback_title_copy`` ↔ filename
``test_p3_swap_fallback_title_copy.py``.
"""
import pathlib
import re

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
AGENT_PY = (BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Empty-pantry-fallback prefiere `name` sobre `display_string`
# ---------------------------------------------------------------------------

def test_empty_pantry_fallback_prefers_name_over_display_string():
    """El reader del ``aggregated_shopping_list`` debe leer ``name`` PRIMERO
    para que el prompt del LLM y el fallback title reciban tokens limpios
    (sin pesos/unidades) en vez de display strings formateados."""
    # Match sobre el bloque de extracción que pre-fix prefería display_string
    m = re.search(
        r"_item\.get\(\s*[\"']name[\"']\s*\)\s*\n?\s*or\s+_item\.get\(\s*[\"']display_string[\"']",
        AGENT_PY,
    )
    assert m, (
        "El reader del empty-pantry-fallback debe preferir `name` sobre "
        "`display_string` (orden: name → display_string). Pre-fix era al "
        "revés y exponía '1 Cabeza (~400g) Lechuga' al fallback title."
    )


def test_empty_pantry_fallback_does_not_prefer_display_string_first():
    """Defensa-en-depth: el patrón pre-fix `display_string` → `name` NO
    debe re-aparecer si alguien revierte por error."""
    bad_pattern = re.search(
        r"_item\.get\(\s*[\"']display_string[\"']\s*\)\s*\n?\s*or\s+_item\.get\(\s*[\"']name[\"']",
        AGENT_PY,
    )
    assert not bad_pattern, (
        "Patrón pre-fix `display_string → name` detectado en agent.py. "
        "Revierte el orden a `name → display_string`."
    )


# ---------------------------------------------------------------------------
# Section B — Title del fallback usa formato friendly sin `.title()`
# ---------------------------------------------------------------------------

def test_fallback_title_drops_opcion_segura_prefix():
    """El prefijo ``"Opción Segura:"`` debe estar eliminado del title.
    Era jargon técnico que sonaba a error del sistema."""
    # Match sobre la f-string `name` que construye el title del fallback
    fallback_section = _extract_fallback_response_block(AGENT_PY)
    assert "Opción Segura" not in fallback_section, (
        "El prefijo `Opción Segura:` aún presente en el title del fallback. "
        "Debe estar reemplazado por un formato más amigable."
    )


def test_fallback_title_does_not_call_dot_title():
    """El title NO debe aplicar ``.title()`` sobre los ingredientes —
    mangla unidades (``g`` → ``G``) cuando upstream emite display strings."""
    fallback_section = _extract_fallback_response_block(AGENT_PY)
    # El .title() problemático estaba en `' y '.join(fallback_ing[:2]).title()`
    assert "fallback_ing[:2]).title()" not in fallback_section, (
        "El title del fallback aún invoca `.title()` sobre el join de "
        "ingredientes. Pre-fix mangleba `g` → `G` y `y` → `Y`."
    )


def test_fallback_title_uses_meal_type_plus_ingredients_pattern():
    """El title debe seguir el patrón ``f"{meal_type} con {ings}"`` —
    natural y no expone el prefijo técnico."""
    fallback_section = _extract_fallback_response_block(AGENT_PY)
    # Buscar la f-string que construye el name
    assert re.search(
        r'"name":\s*f"\{meal_type\}\s+con\s+\{[^}]+\}"',
        fallback_section,
    ), (
        "El title del fallback debe seguir el patrón "
        "`f\"{meal_type} con {_title_ings}\"`. Si renombraste la variable, "
        "actualiza este test."
    )


def test_fallback_title_guards_against_empty_ingredients():
    """El title debe tener un guard defensivo cuando ``fallback_ing[:2]``
    no produce tokens válidos — fallback a un copy genérico para evitar
    titles rotos como ``"Cena con  y "``."""
    # El guard ``if _ing_title_tokens else "ingredientes de tu nevera"``
    assert 'ingredientes de tu nevera' in AGENT_PY, (
        "Falta el copy genérico de respaldo `'ingredientes de tu nevera'` "
        "cuando no hay tokens válidos en fallback_ing[:2]."
    )


# ---------------------------------------------------------------------------
# Section C — Description del fallback con copy cálido
# ---------------------------------------------------------------------------

def test_fallback_desc_uses_warm_copy_not_technical_jargon():
    """La descripción del fallback NO debe contener jargon técnico como
    ``"autogenerado como medida de seguridad"`` — sonaba a error del
    sistema."""
    fallback_section = _extract_fallback_response_block(AGENT_PY)
    assert "autogenerado como medida de seguridad" not in fallback_section, (
        "Desc técnica pre-fix aún presente. Debe estar reemplazada por "
        "un copy cálido tipo `Plato simple armado con ingredientes que "
        "tienes en casa`."
    )
    assert "Plato simple armado con ingredientes que tienes en casa" in fallback_section, (
        "Falta el copy cálido nuevo de la desc del fallback."
    )


# ---------------------------------------------------------------------------
# Section D — Helper para extraer el bloque del response del fallback
# ---------------------------------------------------------------------------

def _extract_fallback_response_block(src: str) -> str:
    """Extrae SOLO el dict literal ``response = { ... }`` del fallback en
    ``swap_meal`` (sin comentarios narrativos que puedan mencionar
    frases viejas como 'Opción Segura' / 'medida de seguridad' como
    contexto histórico). Localiza por la asignación al field ``"name"``
    + el patrón único de la f-string del title nuevo."""
    # Buscar el bloque ``response = { ... "name": f"{meal_type} con ..."``
    # — el dict del fallback es el único con esa firma.
    m = re.search(
        r"response\s*=\s*\{[^{}]*?\"name\":\s*f\"\{meal_type\}[^\"]*\"[^{}]*?\}",
        src,
        re.DOTALL,
    )
    assert m, (
        "No se encontró el dict literal `response = { ..., \"name\": "
        "f\"{meal_type} con ...\", ... }` en agent.py. Si renombraste el "
        "patrón del title, actualiza este helper."
    )
    return m.group(0)
