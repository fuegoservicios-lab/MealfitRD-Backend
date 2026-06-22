"""[P2-SEASONING-CATALOG-KEEP · 2026-06-22] Un sazón del CATÁLOGO verificado emitido por el LLM
solo en cantidad nominal (pizca/al gusto, sin peso) se LISTA con 1 empaque mínimo en vez de dropearse.

Caso en vivo (2026-06-22, plan de angelobrito500): un plan de ganancia muscular salió excelente
(macros band 1.0) pero la lista de compras NO incluía cilantro ni orégano dominicano — el LLM los
emitió como "pizca", y el AGGREGATE-DROP los descartaba por no tener peso. Ambos SÍ están en el
catálogo de 119 (son alimentos reales y comprables). Es la misma clase que el bug de la mantequilla
de maní: la receta dice X, la lista no tiene X.

Fix: en el drop por cantidad-nominal, si el ingrediente resuelve al catálogo verificado, asignarle
el peso de 1 empaque (container_weight_g → density → default) y dejarlo caer al path normal de peso
→ apply_smart_market_units lo lista como "1 frasco/mazo". Los NO-catálogo siguen el drop (+ la
observabilidad VERIFIED-ONLY de la Fase 1).
"""
import shopping_calculator as sc


def _src():
    return open(sc.__file__, encoding="utf-8").read()


def test_marker_y_gate_presentes():
    src = _src()
    assert "P2-SEASONING-CATALOG-KEEP" in src
    assert "def _seasoning_catalog_keep_enabled" in src
    assert "_SEASONING_DEFAULT_G" in src


def test_gate_default_on():
    assert sc._seasoning_catalog_keep_enabled() is True


def test_gate_respeta_knob_off(monkeypatch):
    monkeypatch.setenv("MEALFIT_SEASONING_CATALOG_KEEP", "false")
    assert sc._seasoning_catalog_keep_enabled() is False


def test_keep_solo_si_verificado_y_con_precio():
    # El keep DEBE estar gateado por _is_verified_for_shopping + precio > 0 (no aplica a inventados).
    src = _src()
    idx = src.find("_keep_seasoning = _is_verified_for_shopping(name)")
    assert idx > -1, "El keep debe condicionarse a que el ingrediente resuelva al catálogo verificado."
    region = src[idx: idx + 120]
    assert "price_per_lb > 0 or price_per_unit > 0" in region


def test_drop_no_catalogo_se_preserva():
    # Un ingrediente NO-catálogo emitido nominal debe SEGUIR dropeándose (el AGGREGATE-DROP sigue
    # en la rama else); el keep es solo para catálogo.
    src = _src()
    assert "[AGGREGATE-DROP]" in src, "El drop de no-catálogo debe seguir existiendo (rama else)."
    # El keep asigna peso (1 empaque) y NO hace continue → cae al path de peso normal.
    assert "weight_in_lbs = _seas_g / 453.592" in src
    assert "has_weight = True" in src
