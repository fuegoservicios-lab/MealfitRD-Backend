"""[P1-PDF-LIST-POLISH · 2026-09-02] Pulido visual del PDF de la lista de compras.

Vivo (plan 82da3be1 renovado con la Nevera llena, 6 ítems por comprar):
  1. «Esta compra RD$435» y debajo «Costo real del ciclo RD$8.951» sin una línea que
     explicara el salto; el backend (budget_reconciliation) decía 12.768 para el ciclo
     completo — DOS definiciones del ciclo, la del frontend re-derivada con
     perecederos × (semanas−1).
  2. Recuadro ROJO (#fef2f2/#dc2626) para una instrucción neutra («compra esta semana»);
     rojo es el color del plan VENCIDO (P2-SHOPPING-2).
  3. «3 funda (…)»: 'funda' no estaba en la tabla de plurales, que vivía inline.
  4. «Generado: 2/9/2026» (¿febrero o septiembre?) → «2 sept 2026».
  5. «(referencia estimada)» — jerga interna junto al presupuesto.
  6. La leyenda de Smart Engine ocupaba la cabecera; ahora es pie de página.
Parser-based sobre Dashboard.jsx + funcional sobre `get_plural_unit`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DASH_PATH = _ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_DASH = ""
_LOCALES = _ROOT / "frontend" / "src" / "i18n" / "locales"


@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # [P2-CI-BACKEND-CERO-TESTS] la fixture compartida salta el módulo si falta el hermano;
    # la lectura ya no ocurre al importar (el checkout del backend no trae ../frontend).
    _ = frontend_repo_path
    global _DASH
    _DASH = _DASH_PATH.read_text(encoding="utf-8")


def _win_after(anchor: str, span: int = 2600) -> str:
    i = _DASH.find(anchor)
    assert i != -1, f"ancla desaparecida: {anchor!r}"
    return _DASH[i:i + span]


# ---------------------------------------------------------------- 1. ciclo SSOT
def test_pdf_cycle_future_weeks_come_from_backend():
    win = _win_after("_deltaAware = (deltaItemsRemoved || 0) > 0")
    assert "_backendCostSummary.cycle_total_rd - _backendCostSummary.trip_total_rd" in win, \
        "semanas 2..N = cycle_total_rd − trip_total_rd del backend (paridad con budget_reconciliation)"
    assert "_fullCycleCostFinal = _stableCost + _perishableCost + _futureFreshRdPdf" in win
    # fallback sin resumen: sigue la derivación local (planes legacy)
    assert re.search(r"_fullPerishableRd \* Math\.max\(0, _cycleCostMultiplier - 1\)", win)


def test_banner_cycle_uses_same_backend_rule():
    win = _win_after("const shoppingDeltaMeta = useMemo(", 4000)
    assert "_bs.cycle_total_rd - _bs.trip_total_rd" in win, "el banner in-app usa la MISMA regla que el PDF"
    assert "const deltaCycleRd = deltaTripRd + _futureFreshRd;" in win


def test_pdf_cycle_guard_rejects_inconsistent_summary():
    """cycle < trip (resumen corrupto) ⇒ fallback local, nunca un ciclo negativo."""
    win = _win_after("_deltaAware = (deltaItemsRemoved || 0) > 0")
    assert "_backendCostSummary.cycle_total_rd >= _backendCostSummary.trip_total_rd" in win
    win_b = _win_after("const shoppingDeltaMeta = useMemo(", 4000)
    assert "_bs.cycle_total_rd >= _bs.trip_total_rd" in win_b


# ---------------------------------------------------------------- 1b. frase puente
_BRIDGE = "Incluye ≈{monto} para recomprar en las semanas siguientes los frescos que hoy ya tienes en la Nevera"


def test_bridge_line_only_when_delta_aware():
    i = _DASH.find("Costo real del ciclo de {duracion}")
    assert i != -1
    win = _DASH[i:i + 900]
    assert f"_deltaAware ? t('{_BRIDGE}'" in win, "la frase puente explica el salto SOLO cuando la Nevera descontó"
    assert "Math.round(_futureFreshRdPdf)" in win, "el importe de la frase es el MISMO que se sumó al ciclo"
    assert "Despensa 1× + perecederos de {duracion} (recompra cada 7 días)" in win, "sin descuento, el copy de siempre"


@pytest.mark.parametrize("locale", ["en-US", "fr-FR", "it-IT", "pt-BR"])
def test_new_keys_translated_and_old_key_gone(locale):
    cat = (_LOCALES / f"{locale}.json").read_text(encoding="utf-8")
    head = f'"{_BRIDGE}": "'
    assert head in cat
    assert "{monto}" in cat.split(head, 1)[1][:300], "el placeholder sobrevive a la traducción"
    assert '" (estimado según tus metas)": "' in cat
    assert '" (referencia estimada)": "' not in cat, "clave huérfana (npm run i18n:check la rechaza)"


# ---------------------------------------------------------------- 2. ámbar, no rojo
def test_perishables_box_is_amber_not_red():
    i = _DASH.find("<!-- Prioridad Alta")
    assert i != -1
    block = _DASH[i:i + 1800]
    assert "#fffbeb" in block and "#fcd34d" in block and "#d97706" in block and "#92400e" in block
    for red in ("#fef2f2", "#fca5a5", "#dc2626", "#991b1b", "#b91c1c"):
        assert red not in block, f"{red}: el rojo es del plan VENCIDO (P2-SHOPPING-2), no de una instrucción"


def test_expired_banner_keeps_red():
    m = re.search(r"isPlanExpired\s*\?\s*`([^`]+)`", _DASH, re.DOTALL)
    assert m and "#fef2f2" in m.group(1), "el banner de plan vencido sigue rojo (contraste semántico)"


# ---------------------------------------------------------------- 3. plurales
def test_unit_plurals_single_table_and_new_nouns():
    import shopping_calculator as sc

    src = Path(sc.__file__).read_text(encoding="utf-8")
    assert "    PLURALS = {" not in src, "la tabla inline volvió — el SSOT es UNIT_PLURALS"
    assert "UNIT_PLURALS.get(u_lower, u)" in src
    assert sc.get_plural_unit(3, "funda") == "fundas"
    assert sc.get_plural_unit(2, "Funda") == "Fundas"
    assert sc.get_plural_unit(1, "funda") == "funda"
    for noun, plural in (("malla", "mallas"), ("manojo", "manojos"), ("libra", "libras"),
                         ("litro", "litros")):
        assert sc.get_plural_unit(2, noun) == plural
    assert sc.get_plural_unit(2, "cartón (30 uds.)") == "cartones (30 uds.)", "sufijo parentético preservado"


def test_measured_master_containers_all_pluralize():
    """Los envases medidos en master_ingredients (2026-09-02) tienen plural, todos."""
    import shopping_calculator as sc

    measured = ["paquete", "pote", "lata", "frasco", "botella", "tarro", "carton", "cartón",
                "unidad", "sobre", "funda", "malla", "mazo", "envase", "cabeza", "manojo", "libra", "litro"]
    stuck = [u for u in measured if sc.get_plural_unit(2, u) == u]
    assert not stuck, f"sin plural: {stuck}"


# ---------------------------------------------------------------- 4. fecha
def test_generated_date_is_not_ambiguous_numeric():
    i = _DASH.find("Generado: {fecha}")
    assert i != -1
    win = _DASH[i:i + 200]
    assert "formatDate(new Date(), { day: 'numeric', month: 'short', year: 'numeric' })" in win, \
        "«2/9/2026» era ambiguo (¿febrero?) → mes abreviado por locale"


# ---------------------------------------------------------------- 5. copy
def test_reference_copy_is_plain_spanish():
    assert "t(' (referencia estimada)')" not in _DASH
    assert _DASH.count("t(' (estimado según tus metas)')") == 3, "app (banner + Config) y PDF comparten la clave"


# ---------------------------------------------------------------- 6. leyenda al pie
def test_legend_is_a_footnote_after_the_totals():
    legend = _DASH.find("<!-- Disclaimer de Cantidades")
    totals = _DASH.find("Costo real del ciclo de {duracion}")
    footer = _DASH.find("<!-- Footer -->")
    assert -1 not in (legend, totals, footer)
    assert totals < legend < footer, "la leyenda va DESPUÉS de los totales y ANTES del pie"
    # anclas históricas intactas (P3-SHOPPING-DISCLAIMER-EXPAND / P3-STABLES-NO-SCALE-UX)
    assert "conversión aproximada" in _DASH and "realismo de almacenamiento" in _DASH
    assert "${isUltraDense ? '' :" in _DASH
