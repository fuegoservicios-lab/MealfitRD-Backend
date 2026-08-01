"""[P2-VEG-VOLUME-TOKENS-2 · 2026-08-01] Ola de fixes derivada de 2 diagnósticos SOLO-LECTURA
sobre los 17 planes de prod (`.superpowers/esparragos-600g-diagnostico.md`).

## Sección 1 — 4 vegetales acuosos faltantes en `_REALISM_VOLUME_VEG_TOKENS`

`_REALISM_VOLUME_VEG_TOKENS` (cap de 250 g/línea, `P1-VEG-VOLUME-TOKENS`) ya había tenido que
ampliarse una vez (2026-07-26) y el forense midió que seguía incompleta: 3/17 planes de prod
(17.6%) tenían ≥1 línea de vegetal ACUOSO en 300-600 g para una sola porción, con 4 vegetales
DISTINTOS que el allowlist no cubría — ninguno repetido, la firma clásica de "lista a mano que
drifea" que el propio diagnóstico documenta:

    600 g de espárragos           (plan 5f4bb17e — el solver LSQ per-meal infló el único "carbs
                                    driver" barato en gramos del plato; 100g→343g reproducido en
                                    vivo, factor 3.43×. El 600 exacto además evade
                                    LINE_GRAM_HARD_CAP, que exige `>600` estricto)
    400 g de vainitas cortadas    (plan cf3a81fb)
    2.75 tazas de coles de Bruselas (413g)   (plan 7c545d59)
    21.5 molondrones medianos (≈322 g)       (plan 7c545d59)

Fix: añadir `esparrago`, `vainita`, `coles de bruselas`, `molondron` a la tuple
(`graph_orchestrator.py`). `LINE_GRAM_HARD_CAP` (el techo genérico de 600g) NO se toca — el
diagnóstico lo desaconseja explícitamente (bajarlo rompería porciones legítimas de arroz/pollo en
household grande; el backstop correcto para vegetales acuosos es el cap específico, no el
genérico). El refactor estructural que el diagnóstico propone (derivar el set desde
`master_ingredients WHERE category='Vegetales'` en vez de una tuple a mano, mismo principio que
P1-DIET-CANON-SSOT) queda documentado como follow-up, no implementado aquí.

## Sección 2 — qty-sync ciego a "lonjas/pedazos" (`_STEP_QTY_UNITS` + `_STEP_QTY_MENTION_RE`)

Causa del caso queso 30↔45 del mismo diagnóstico (§3, plan 5f4bb17e): `ingredients` decía
"30 g de queso" pero el paso de Mise en place decía "desmenuza 1¾ lonjas/pedazos de queso (45 g)"
— `_sync_recipe_step_quantities` (P1-RECIPE-QTY-SYNC) EXISTE precisamente para reconciliar esto,
pero (a) `pedazo(?:s)?` no estaba en `_STEP_QTY_UNITS`, y (b) aunque "lonja" sí estaba, el slash
PEGADO ("lonjas/pedazos", sin espacio) rompía el match: tras la unidad el patrón exige
`\\.?\\s+de\\s+` inmediatamente, y lo que sigue a "lonjas" es "/pedazos", no espacio+"de". Fix:
`pedazo(?:s)?` añadido a `_STEP_QTY_UNITS` + `_STEP_QTY_MENTION_RE` tolera un 2º token de unidad
opcional pegado por slash (`(?:/(?:...))?`) sin cambiar qué unidad queda en el grupo `unit`
(sigue siendo la primera) — la reescritura reemplaza la mención COMPLETA por
"<qty_actual> <unit_actual> de <food>", así que la forma compuesta del ORIGEN no necesita
sobrevivir. Solo se tocó `_STEP_QTY_MENTION_RE` (lado pasos/receta) — `_ING_LEAD_QTY_RE`/
`_ING_LEAD_QTY_MIXED_RE` (lado `ingredients[]`) no reciben la forma "unidad/unidad" en la
práctica, así que se dejaron sin cambios.
"""
from __future__ import annotations

import os
import re

import graph_orchestrator as g

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ═══════════════════════ Sección 1 — veg volume tokens ═══════════════════════

_NUEVOS_TOKENS = ("esparrago", "vainita", "coles de bruselas", "molondron")


def test_parser_los_4_tokens_nuevos_estan_en_la_tuple():
    for tok in _NUEVOS_TOKENS:
        assert tok in g._REALISM_VOLUME_VEG_TOKENS, (
            f"falta {tok!r} en _REALISM_VOLUME_VEG_TOKENS — regresión del fix P2-VEG-VOLUME-TOKENS-2"
        )


def test_parser_marker_y_evidencia_anclados_en_fuente():
    assert "P2-VEG-VOLUME-TOKENS-2" in _GO
    assert "esparragos-600g-diagnostico.md" in _GO or "esparragos-600g" in _GO


class _DB:
    """Espejo del dummy de `test_p1_apio_stalk_cap.py`/`test_p1_recipe_blockers_2.py`: solo
    necesita responder a `_ingredient_macro_group` (via `macros_from_ingredient_string`) para que
    la clasificación de grupo (protein/carbs/fats) del cap de LINE_GRAM_HARD_CAP no interfiera
    con la rama del cap de volumen vegetal, que es la que este test ejercita."""

    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()

    def _ingredient_macro_group(self, *a, **k):
        return None


def _cap(line):
    meal = {"name": "Batata Majada Gratinada con Queso de Hoja, Tomate y Espárragos",
            "meal": "Cena", "ingredients": [line], "ingredients_raw": [line],
            "protein": 24, "carbs": 62, "fats": 13, "cals": 464}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    return meal["ingredients"][0]


def test_funcional_esparragos_600g_capeado_a_250():
    """Caso real literal del diagnóstico (plan 5f4bb17e): "600 g de espárragos" para 1 persona
    debe quedar recortado al techo de REALISM_VEG_VOLUME_CAP_G (250g, default)."""
    out = _cap("600 g de esparragos")
    assert out.startswith("250"), out
    assert g.REALISM_VEG_VOLUME_CAP_G == 250


def test_funcional_vainitas_400g_capeado():
    assert _cap("400 g de vainitas cortadas").startswith("250")


def test_funcional_coles_de_bruselas_capeado_pese_a_ser_multi_palabra():
    """'coles de bruselas' es un token de 3 palabras en la tuple — el matcher del consumidor
    (`_re.search(r"\\b" + t, il)`) hace substring sobre texto lowercased+accent-stripped, así
    que un token con espacios matchea igual que uno de una sola palabra. Verificado en runtime,
    no solo como afirmación de parser."""
    assert _cap("280 g de coles de bruselas").startswith("250")


def test_funcional_molondron_singular_matchea_forma_plural():
    """El token es singular ('molondron'); la línea real usa plural ('molondrones') — substring
    search sin límite de fin de palabra, así que el singular matchea dentro del plural."""
    assert _cap("350 g de molondrones").startswith("250")


def test_funcional_porcion_razonable_no_se_toca():
    """Control negativo: una porción realista (por debajo del cap) queda intacta."""
    out = _cap("150 g de esparragos")
    assert out.startswith("150"), out


def test_la_auyama_sigue_fuera_decision_explicita():
    """No-regresión de la decisión P1-VEG-VOLUME-TOKENS: la auyama es base de carbohidrato, no
    relleno de volumen — no debe entrar nunca sin su propia medición de impacto en macros."""
    assert not any(re.search(r"\b" + t, "auyama") for t in g._REALISM_VOLUME_VEG_TOKENS)


def test_line_gram_hard_cap_no_tocado():
    """El diagnóstico desaconseja bajar el techo genérico de 600g — este fix es SOLO el
    allowlist de vegetales, LINE_GRAM_HARD_CAP debe seguir en su default."""
    assert g.LINE_GRAM_HARD_CAP == 600


# ═══════════════════════ Sección 2 — qty-sync "lonjas/pedazos" ═══════════════════════

def test_parser_pedazo_en_step_qty_units():
    assert "pedazo(?:s)?" in _GO, "falta pedazo(?:s)? en _STEP_QTY_UNITS"


def test_caso_real_lonjas_slash_pedazos_se_sincroniza(monkeypatch):
    """Texto REAL del diagnóstico (plan 5f4bb17e, §3): ingrediente re-escalado a "30 g de queso"
    (post-solver) pero el paso de Mise en place seguía congelado en "1¾ lonjas/pedazos de queso
    (45 g)" — el slash pegado rompía el match del qty-sync. Debe sincronizarse a "30 g de queso"."""
    monkeypatch.setattr(g, "RECIPE_STEP_QTY_SYNC_ENABLED", True)
    meal = {
        "name": "Batata Majada Gratinada con Queso de Hoja, Tomate y Espárragos",
        "ingredients": ["30 g de queso", "600 g de esparragos"],
        "recipe": [
            "Mise en place: Pela y aplasta ½ batata mediana (145 g); desmenuza 1¾ "
            "lonjas/pedazos de queso (45 g); corta los espárragos en trozos de 4 cm.",
            "El Toque de Fuego: saltea los espárragos con el aceite de oliva a fuego medio "
            "durante 5-6 minutos.",
        ],
    }
    n = g._sync_recipe_step_quantities(meal)
    assert n >= 1, "esperada ≥1 mención reescrita"
    assert "30 g de queso" in meal["recipe"][0], meal["recipe"][0]
    assert "1¾ lonjas/pedazos" not in meal["recipe"][0], meal["recipe"][0]


def test_lonjas_slash_pedazos_aislado_via_step_qty_mention_re():
    """El regex compilado matchea la mención compuesta directamente (sin pasar por el pipeline
    completo de _sync_recipe_step_quantities) — aísla la causa exacta que el diagnóstico
    documentó: unit='lonjas', el '/pedazos' pegado ya no rompe el `\\.?\\s+de\\s+` posterior."""
    texto = "desmenuza 1¾ lonjas/pedazos de queso (45 g)"
    m = g._STEP_QTY_MENTION_RE.search(texto)
    assert m is not None, "el regex debe matchear la forma compuesta con slash"
    assert m.group("unit") == "lonjas"
    assert m.group("food") == "queso"


def test_pedazos_solo_sin_slash_tambien_matchea():
    """Control: 'pedazos' solo (sin compuesto con slash) ahora es una unidad reconocida por sí
    misma, no solo como 2º término de un compuesto."""
    texto = "corta 2 pedazos de queso"
    m = g._STEP_QTY_MENTION_RE.search(texto)
    assert m is not None
    assert m.group("unit") == "pedazos"
    assert m.group("food") == "queso"
