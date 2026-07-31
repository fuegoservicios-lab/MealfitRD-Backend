"""[P1-LUNA-PRICING · 2026-07-26] Precio de la familia gpt-5.6 en la tabla de costo.

Sin fila de precio, `compute_llm_cost_micros` devuelve `None` y el evento se persiste con tokens
pero SIN costo. Medido: la primera corrida del canario dejó 3 llamadas de `gpt-5.6-luna` sin
costear, y la tabla del A/B las sumaba como `0.0000` — el modelo caro apareciendo como gratis
justo donde se decide si vale su precio.

Se registran los TRES hermanos aunque hoy sólo `luna` esté en uso: añadir el precio antes de
probar un modelo, no después de descubrir que no se midió.

## Pricing oficial OpenAI, tier STANDARD (consultado 2026-07-26, USD/1M tokens)

    modelo          input   cached   cache-write   output
    gpt-5.6-luna    1.00     0.10       1.25        6.00
    gpt-5.6-terra   2.50     0.25       3.125      15.00
    gpt-5.6-sol     5.00     0.50       6.25       30.00

⚠️ El `cached` es **10× más barato** que la entrada normal, y el day-gen trae ~42% de aciertos de
caché medidos. Ignorarlo sobreestimaría el costo de Luna un 22% (medido sobre los tokens reales
del plan b01baf9c: USD 0.1016 con caché contra 0.1303 sin ella).

⚠️ NO cubierto: la tarifa de ESCRITURA de caché. `usage_metadata` no la separa de la entrada
normal, así que la primera llamada de cada prompt nuevo queda ligeramente subestimada.

El tier PRIORITY cuesta ~2× y no se usa.
"""
import pytest

from db_profiles import _DEFAULT_LLM_PRICING_MICROS_PER_M as TABLA, compute_llm_cost_micros


# [P1-REVIEWER-TIER-MODELS · 2026-07-31] Recorte oficial de OpenAI re-consultado:
# luna -80% ($1.00/$6.00 → $0.20/$1.20), terra -20% ($2.50/$15.00 → $2.00/$12.00).
# sol sin cambio. `cached` mantiene el ratio 10% del input.
_FAMILIA = {
    "gpt-5.6-luna":  (200_000, 1_200_000, 20_000),
    "gpt-5.6-terra": (2_000_000, 12_000_000, 200_000),
    "gpt-5.6-sol":   (5_000_000, 30_000_000, 500_000),
}


@pytest.mark.parametrize("modelo,esperado", _FAMILIA.items())
def test_los_tres_hermanos_tienen_precio(modelo, esperado):
    fila = TABLA.get(modelo)
    assert fila, f"{modelo} sin precio → sus llamadas se persisten sin costo"
    assert (fila["input"], fila["output"], fila["cached"]) == esperado


@pytest.mark.parametrize("modelo", _FAMILIA)
def test_ningun_modelo_de_la_familia_se_queda_sin_costear(modelo):
    assert compute_llm_cost_micros(modelo, 1000, 100, 0) is not None


def test_el_cache_es_diez_veces_mas_barato():
    """Si alguien iguala `cached` a `input` por descuido, el costo de Luna se infla un 22%."""
    f = TABLA["gpt-5.6-luna"]
    assert f["cached"] * 10 == f["input"]


def test_costo_del_plan_medido_en_produccion():
    """Tokens REALES de las 3 llamadas del plan b01baf9c (2026-07-26 07:24). Ancla el orden de
    magnitud: si un cambio de tabla mueve esto, se ve aquí antes que en la factura.
    [P1-REVIEWER-TIER-MODELS · 2026-07-31] OpenAI recortó luna -80% ($1.00/$6.00 →
    $0.20/$1.20): el mismo plan pasa de ~$0.102 a ~$0.020 (×0.2 exacto — cross-check
    del recorte)."""
    llamadas = [(25346, 2670, 10658), (25347, 2977, 10658), (25340, 3405, 10658)]
    total = sum(compute_llm_cost_micros("gpt-5.6-luna", i, o, c) for i, o, c in llamadas)
    assert 0.019 < total / 1e6 < 0.022, f"USD {total/1e6:.5f}"


def test_luna_es_el_mas_barato_de_la_familia():
    assert (TABLA["gpt-5.6-luna"]["input"] < TABLA["gpt-5.6-terra"]["input"]
            < TABLA["gpt-5.6-sol"]["input"])


def test_deepseek_sigue_intacto():
    """El canario no debe haber tocado el pricing del provider que corre en producción."""
    assert TABLA["deepseek-v4-flash"] == {"input": 140_000, "output": 280_000, "cached": 2_800}
    assert TABLA["deepseek-v4-pro"] == {"input": 435_000, "output": 870_000, "cached": 3_625}


def test_relacion_de_costo_contra_deepseek():
    """El número que decide: mismos tokens, cuánto más caro es Luna.
    [P1-REVIEWER-TIER-MODELS · 2026-07-31] Con el -80%, la relación se DESPLOMÓ:
    de 11.6× flash / 3.7× pro (2026-07-26) a ~2.3× flash y ~0.75× pro — Luna es
    ahora MÁS BARATO que deepseek-v4-pro, lo que hace viable usarlo de reviewer
    clínico hasta en el tier gratis."""
    llamadas = [(25346, 2670, 10658), (25347, 2977, 10658), (25340, 3405, 10658)]
    def _t(m):
        return sum(compute_llm_cost_micros(m, i, o, c) for i, o, c in llamadas)
    assert 2.1 < _t("gpt-5.6-luna") / _t("deepseek-v4-flash") < 2.5
    assert 0.65 < _t("gpt-5.6-luna") / _t("deepseek-v4-pro") < 0.85
