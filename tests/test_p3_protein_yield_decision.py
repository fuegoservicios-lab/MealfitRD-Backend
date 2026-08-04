"""[P3-PROTEIN-YIELD-DECISION · 2026-08-04] Decisión delegada de Task 14
(P2-PROTEIN-YIELD-CANONICAL): el knob `MEALFIT_PROTEIN_YIELD_ON_CANONICAL` nació OFF
esperando un A/B — este P-fix MIDE el delta de costo real y decide.

## Punto de partida (medido por el operador contra prod, read-only, 2026-08-03)

12 líneas de `ingredients_raw` matchean la regla #2 de `_calculate_yield_multiplier`
(proteína cocida → 1.35× crudo) en 5/23 planes (~22%). De esas 12, **1 es línea de REUSO**
(«205 g de pollo cocido y desmenuzado (del almuerzo o preparado extra)») ya excluida por
`_PROTEIN_REUSE_PAREN_RE` desde Task 14 → **11 líneas realmente afectadas**.

## Medición OFFLINE de este P-fix (ejecutando el agregador real, no inventada)

Replicando la MISMA convención que ancla `test_p2_protein_yield_canonical.py`
(`num_days=1` → `base_duration_scale=7`, "cantidad semanal" para una línea que aparece una
vez), se corrió `get_shopping_list_delta` con el knob OFF vs ON para las 5 líneas de las
que tenemos texto exacto (4 no-reuso + la de reuso, como control negativo):

    | línea (texto real)                              | alimento          | Δg/semana | precio RD$/lb (catálogo) | ΔRD$/semana |
    |---------------------------------------------------|-------------------|-----------|---------------------------|-------------|
    | «160 g de pescado cocido»                          | Pescado           | 392.00    | 127.5 (Filete pescado blanco) | ~110.2  |
    | «100 g de cerdo magro cocido y desmenuzado»         | Cerdo             | 245.00    | 115 (Cerdo genérico)          | ~62.1   |
    | «45 g de costilla de cerdo cocida y desmenuzada»    | Costilla de cerdo | 110.25    | 189 (Costilla de cerdo)       | ~45.9   |
    | «40g de pechuga de pollo cocido»                    | Pechuga de pollo  | 98.00     | 135 (Filete pechuga de pollo) | ~29.2   |
    | «205 g ... (del almuerzo o preparado extra)»        | Pollo (REUSO)     | 0.00      | —                             | 0.0 (control) |

Precios RD$/lb: catálogo VERSIONADO del repo (grep `price_per_lb` en `scripts/`) —
`scripts/add_foods_batch1_2026_06_26.py` (Muslo de pollo 68, Tilapia 130, Mero 290),
`scripts/add_foods_batch2_2026_06_26.py` (Costilla de cerdo 189),
`seed_supermarket_2026_07_02.py` (Cerdo genérico 115, Filete pechuga de pollo 135;
"Filete de pescado blanco" Paquete 32 Oz RD$255 → RD$127.5/lb, línea 145 del mismo
archivo — misma derivación citada en el comentario `P1-BUDGET-PREMIUM-SHELLFISH` dentro
de `_BUDGET_DRIVER_FAMILIES`, graph_orchestrator.py; se cita por símbolo/marker y no por
número de línea porque el archivo se sigue editando y la línea drifea). Rango real
medido: **RD$68–290/lb** según corte/proteína.

Promedio por línea no-reuso: (110.2+62.1+45.9+29.2)/4 ≈ **RD$61.85/línea**. Con 11 líneas
no-reuso repartidas en 5 planes afectados (2.2 líneas/plan promedio) ⇒ delta semanal
**PROMEDIO por plan afectado ≈ RD$136**. Peor caso CONSTRUIDO (cota superior: las 4
proteínas DISTINTAS sumadas — ningún plan de los 5 medidos tuvo de hecho las 4 a la vez,
el promedio real es 2.2 líneas/plan) ≈ **RD$247**. Ambos números son una fracción menor
(<10%) del costo semanal típico de una lista (RD$3.000–6.000, CLAUDE.md) — incluso la
cota superior roza pero no cruza de forma significativa el umbral ~RD$200 de la decisión
delegada, y sigue siendo ruido frente al presupuesto semanal real.

## Decisión

**FLIP a `True`.** El mecanismo cierra un under-buy real (~26% menos proteína cruda de la
necesaria por línea matcheada) a un costo marginal (RD$136 promedio / RD$247 cota
superior construida, por semana, solo en el ~22% de planes que matchean) — y el sello
`protein_yield_applied` (`_protein_yield_seal_applied`, ronda 1 de Task 14) ya blinda al
guard de coherencia contra
divergencias fantasma en cualquier dirección del A/B (verificado por
`TestGuardSealNotLiveKnob` en `test_p2_protein_yield_canonical.py`). El knob se mantiene
como rollback: `MEALFIT_PROTEIN_YIELD_ON_CANONICAL=false` sin redeploy.

tooltip-anchor: P3-PROTEIN-YIELD-DECISION
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import shopping_calculator as sc
from knobs import get_knobs_registry_snapshot

_BACKEND = Path(__file__).resolve().parents[1]
_SRC = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")

_LB_TO_G = 453.592


@pytest.fixture(autouse=True)
def no_master_db():
    """OFFLINE: mismo patrón que test_p2_protein_yield_canonical.py."""
    with patch.object(sc, "get_master_ingredients", return_value=[]):
        yield


def _delta(plan, **kw):
    return sc.get_shopping_list_delta(
        None, plan, kw.pop("is_new_plan", True), False, True, kw.pop("multiplier", 1.0), **kw
    )


def _item(items, needle: str):
    return next(i for i in items if needle in str(i.get("name", "")).lower())


def _plan_de(linea: str) -> dict:
    return {"days": [{"meals": [{"meal": "almuerzo", "ingredients_raw": [linea]}]}]}


# ---------------------------------------------------------------------------
# 0. Marker
# ---------------------------------------------------------------------------
def test_marker_present():
    assert "P3-PROTEIN-YIELD-DECISION" in _SRC


# ---------------------------------------------------------------------------
# 1. El default FLIPEA a True
# ---------------------------------------------------------------------------
def test_default_es_ahora_true(monkeypatch):
    monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
    assert sc._protein_yield_on_canonical_enabled() is True


def test_knob_registrado_default_true():
    sc._protein_yield_on_canonical_enabled()
    reg = get_knobs_registry_snapshot()
    assert reg["MEALFIT_PROTEIN_YIELD_ON_CANONICAL"]["default"] is True


# ---------------------------------------------------------------------------
# 2. El knob sigue siendo un rollback lever explícito (sin redeploy)
# ---------------------------------------------------------------------------
def test_rollback_explicito_a_false_sigue_funcionando(monkeypatch):
    monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "false")
    assert sc._protein_yield_on_canonical_enabled() is False


def test_encendido_explicito_a_true_sigue_funcionando(monkeypatch):
    monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
    assert sc._protein_yield_on_canonical_enabled() is True


# ---------------------------------------------------------------------------
# 3. Réplica EXACTA de la medición: las 5 líneas reales, con el DEFAULT (sin
#    setear el env var — así se ejercita el default nuevo, no un override).
# ---------------------------------------------------------------------------
class TestMedicionReplicadaConDefault:
    def test_pescado_160g(self, monkeypatch):
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        items = _delta(_plan_de("160 g de pescado cocido"),
                        inventory_override=[], consumed_override=[])
        it = _item(items, "pescado")
        assert it["base_qty"] == pytest.approx(1512.0, abs=0.5)
        assert it["base_qty"] - (160.0 * 7) == pytest.approx(392.0, abs=0.5)
        assert it.get("protein_yield_applied") is True

    def test_cerdo_100g(self, monkeypatch):
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        items = _delta(_plan_de("100 g de cerdo magro cocido y desmenuzado"),
                        inventory_override=[], consumed_override=[])
        it = _item(items, "cerdo")
        assert it["base_qty"] == pytest.approx(945.0, abs=0.5)
        assert it["base_qty"] - (100.0 * 7) == pytest.approx(245.0, abs=0.5)

    def test_costilla_de_cerdo_45g(self, monkeypatch):
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        items = _delta(_plan_de("45 g de costilla de cerdo cocida y desmenuzada"),
                        inventory_override=[], consumed_override=[])
        it = _item(items, "costilla")
        assert it["base_qty"] == pytest.approx(425.25, abs=0.5)
        assert it["base_qty"] - (45.0 * 7) == pytest.approx(110.25, abs=0.5)

    def test_pechuga_de_pollo_40g(self, monkeypatch):
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        items = _delta(_plan_de("40 g de pechuga de pollo cocido"),
                        inventory_override=[], consumed_override=[])
        it = _item(items, "pechuga de pollo")
        assert it["base_qty"] == pytest.approx(378.0, abs=0.5)
        assert it["base_qty"] - (40.0 * 7) == pytest.approx(98.0, abs=0.5)

    def test_reuso_control_negativo_delta_cero(self, monkeypatch):
        """La línea de REUSO medida en prod: con el default ahora en True, la exclusión
        `_PROTEIN_REUSE_PAREN_RE` debe seguir dando delta CERO — es el control negativo
        de esta medición. (El sello `protein_yield_applied` refleja si el PASE estuvo
        activo, no si ESTA línea en particular recibió el multiplicador — mismo criterio
        ya usado por `test_linea_de_reuso_sin_yield_e2e` en test_p2_protein_yield_canonical.py,
        que tampoco asevera sobre el sello en el caso de reuso. Fuera de alcance de esta
        decisión: lo que importa aquí es que la CANTIDAD comprada no cambió.)"""
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        linea = "205 g de pollo cocido y desmenuzado (del almuerzo o preparado extra)"
        items = _delta(_plan_de(linea), inventory_override=[], consumed_override=[])
        it = _item(items, "pollo")
        assert it["base_qty"] == pytest.approx(1435.0, abs=0.5), (
            "una línea de REUSO no debe recibir el multiplicador de yield, aunque el "
            "default sea True"
        )


# ---------------------------------------------------------------------------
# 4. La decisión (números) queda documentada en el comentario del knob
# ---------------------------------------------------------------------------
def test_comentario_del_knob_documenta_la_medicion():
    """[M-1 · review final de audit-v7-p3] Delimitar por conteo fijo de chars (2400) es la
    misma clase de fragilidad que el repo ya documentó y corrigió en otros parser-tests
    (`test_p3_oregano_display_name` pasó de 700 chars a un delimitador estructural): una
    corrección legítima del docstring (alinear los "ejemplos reales" con la tabla medida, ver
    M-1) alargó el bloque y empujó "RD$136" fuera de una ventana arbitraria. Delimita por el
    siguiente `def` de columna 0 — el fin REAL de la función — en vez de un número mágico."""
    i = _SRC.index("def _protein_yield_on_canonical_enabled")
    j = _SRC.index("\ndef ", i + 10)
    doc = _SRC[i:j]
    for pieza in ("P3-PROTEIN-YIELD-DECISION", "RD$136", "RD$247", "FLIP", "True"):
        assert pieza in doc, (
            f"el docstring de `_protein_yield_on_canonical_enabled` debe citar {pieza!r} "
            f"— la decisión tiene que quedar auditable desde el propio código, no solo "
            f"desde el reporte de la tarea."
        )


def test_callsite_del_aggregator_ya_no_dice_default_false():
    """El comentario junto al callsite real (`get_shopping_list_delta`) todavía decía
    '(default False)' — quedaría mintiendo tras el flip."""
    i = _SRC.index("_apply_protein_yield = bool(is_new_plan) and _protein_yield_on_canonical_enabled()")
    ventana = _SRC[max(0, i - 400):i]
    assert "(default False)" not in ventana, (
        "el comentario junto al callsite real sigue anclando el default viejo tras el flip"
    )


# ---------------------------------------------------------------------------
# 5. Marker bump — patrón fecha-floor (NO literal, ver higiene de este mismo lote)
# ---------------------------------------------------------------------------
def test_last_known_pfix_bumpeado():
    import re
    from datetime import date, datetime

    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "No se encontró _LAST_KNOWN_PFIX en app.py."
    marker = m.group(1)
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha, f"Marker sin fecha ISO: {marker!r}"
    marker_date = datetime.strptime(fecha.group(1), "%Y-%m-%d").date()
    floor = date(2026, 8, 4)
    assert marker_date >= floor, (
        f"_LAST_KNOWN_PFIX={marker!r} (fecha={marker_date}) anterior al floor {floor} "
        f"de cierre de P3-PROTEIN-YIELD-DECISION."
    )
