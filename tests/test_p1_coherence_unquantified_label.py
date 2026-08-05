"""[P1-COHERENCE-UNQUANTIFIED-LABEL · 2026-07-26] `unknown` era la etiqueta de casi todo.

## El caso

Medido con el guard REAL sobre 19 planes vivos: **800 de 879 divergencias** salían con
`hypothesis="unknown"`, que es literalmente "no sé qué pasó aquí". El operador que abre la
alerta diaria no puede distinguir señal de ruido conocido en esa lista.

`_classify_divergence_hypothesis` cubría el caso `exp_qty > 0` con cuatro hipótesis, pero
dejaba caer TODO el lado `exp_qty == 0` (el fantasma con `delta_pct = inf`) al `unknown` final.
Y ese lado tiene dos causas bien distintas:

  1. **`recipe_unquantified`** — el alimento está en la lista pero las recetas no le ponen
     cantidad. Es el condimento: `"Sal al gusto"` parsea a `0.0 pizca`, cantidad cero, así que
     no entra en `expected`. Medido: 39 casos.

  2. **`unit_mismatch`** — el alimento SÍ está en las recetas, pero en otra unidad (receta en
     `taza`, lista en `pote`). Medido: ~676 casos, casi todos de planes persistidos ANTES de
     `base_qty` — el desajuste que P1-COHERENCE-BASE-QTY ya cierra para los planes nuevos.

El caso 2 es el **simétrico** del caso ya cubierto en la rama 2 del clasificador, que solo
miraba `exp_qty > 0`. Etiquetarlo bien lo enruta al filtro `P1-COHERENCE-PACKAGING-NOISE` que
ya existía desde 2026-07-07 y que descarta el ruido de granularidad de envase con esta misma
razón escrita: *"ya se excluye del block; es igual de ruidoso en warn"*.

## El efecto, medido (19 planes vivos, guard REAL en block)

    divergencias reportadas:  879 -> 203
    planes que bloquearían:    18 -> 18   (sin cambio)
    fbe53a5b (pipeline moderno completo):  6 -> 2 divergencias, NO bloquea

⚠️ NO es un simple reetiquetado: al nombrar bien el caso 2 entra en un filtro preexistente y
deja de reportarse. Lo que NO cambia es qué bloquea — el subset crítico es idéntico. Se
preservan `yield_uncovered`, `pantry_overdeduct`, el `unknown` POR DEBAJO (falta real), la
sobre-oferta de proteína y toda la capa presence.

tooltip-anchor: P1-COHERENCE-UNQUANTIFIED-LABEL
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc

_clf = sc._classify_divergence_hypothesis


# ───────────── 1. las dos causas nuevas ─────────────

def test_condimento_sin_cantidad():
    """"Sal al gusto" → 0.0 pizca → la sal no entra en expected. La lista SÍ la trae."""
    assert _clf(0.0, 454.0, {}, {"g": 454.0}, food="Sal") == "recipe_unquantified"


def test_alimento_presente_en_otra_unidad():
    """La receta pide tazas, la lista vende potes: no hay magnitud comparable."""
    assert _clf(0.0, 1.0, {"taza": 4.0}, {"pote": 1.0}, food="Yogurt") == "unit_mismatch"


def test_es_el_simetrico_de_la_rama_ya_existente():
    """La rama 2 del clasificador ya devolvía `unit_mismatch` para `exp_qty > 0`. El lado
    `exp_qty == 0` es la MISMA situación vista desde el otro lado y caía a `unknown`."""
    assert _clf(4.0, 0.0, {"taza": 4.0}, {"pote": 1.0}) == "unit_mismatch"
    assert _clf(0.0, 1.0, {"taza": 4.0}, {"pote": 1.0}) == "unit_mismatch"


# ───────────── 2. no se pisó ninguna hipótesis previa ─────────────

def test_ausente_de_la_lista_sigue_siendo_cap_swallowed():
    """La capa que SÍ bloquea no se toca: receta pide pollo, lista no lo trae."""
    assert _clf(500.0, 0.0, {"g": 500.0}, {}) == "cap_swallowed_modifier"


def test_yield_sigue_detectandose():
    assert _clf(100.0, 135.0, {"g": 100.0}, {"g": 135.0}) == "yield_uncovered"


def test_sobrededuccion_sigue_detectandose():
    assert _clf(100.0, 20.0, {"g": 100.0}, {"g": 20.0}) == "pantry_overdeduct"


def test_unknown_sigue_existiendo_para_lo_que_no_encaja():
    """No se agota el espacio: sigue habiendo divergencias sin nombre.

    [P1-COHERENCE-MILD-SHORT · 2026-08-05] El caso que usaba este test (ratio 0.7,
    compra POR DEBAJO de la receta) ya no es `unknown`: era el 98,5% del bucket
    —128 de 130 incógnitas medidas sobre 25 planes— y ahora se llama
    `magnitude_mild_short`. Lo que queda sin nombre es la SOBRE-oferta (act > exp),
    que es lo que el propio guard afirma en `_has_critical_divergence`: «`unknown`
    de magnitud es SIEMPRE sobre-oferta». Esa nota era falsa mientras el
    sub-suministro leve vivía dentro; ahora es cierta, y este test lo ancla.

    La intención original —el bucket no se vacía— se conserva intacta.
    """
    # Sobre-oferta: la lista compra el doble de lo que piden las recetas.
    assert _clf(100.0, 200.0, {"g": 100.0}, {"g": 200.0}) == "unknown"
    # Y el caso de antes ya no cae aquí: tiene nombre propio.
    assert _clf(100.0, 70.0, {"g": 100.0}, {"g": 70.0}) == "magnitude_mild_short"


@pytest.mark.parametrize("act", [0.0, -1.0])
def test_sin_nada_en_la_lista_no_entra_a_las_ramas_nuevas(act):
    assert _clf(0.0, act, {}, {}) == "unknown"


# ───────────── 3. el efecto sobre lo que BLOQUEA (lo que importa) ─────────────

def test_el_condimento_no_bloquea():
    """`recipe_unquantified` es fantasma delta=inf: excluido del subset crítico por diseño."""
    p = {
        "days": [{"day": 1, "meals": [{"name": "Almuerzo",
                                       "ingredients": ["100g de pechuga de pollo", "Sal al gusto"]}]}],
        "aggregated_shopping_list_weekly": [
            {"name": "Pechuga de pollo", "base_qty": 700.0, "base_unit": "g"},
            {"name": "Sal", "base_qty": 454.0, "base_unit": "g"},
        ],
    }
    sc.run_shopping_coherence_guard(p, mode_override="block", multiplier=1.0)
    assert not p.get("_shopping_coherence_block"), p.get("_shopping_coherence_block")


def test_una_falta_REAL_sigue_bloqueando():
    """El test que separa 'quitar ruido' de 'apagar el guard': si la lista NO trae un alimento
    que la receta pide, el plan sigue bloqueando."""
    p = {
        "days": [{"day": 1, "meals": [{"name": "Almuerzo",
                                       "ingredients": ["100g de pechuga de pollo", "200g de pescado"]}]}],
        "aggregated_shopping_list_weekly": [
            {"name": "Pechuga de pollo", "base_qty": 700.0, "base_unit": "g"},
        ],
    }
    sc.run_shopping_coherence_guard(p, mode_override="block", multiplier=1.0)
    assert p.get("_shopping_coherence_block"), "una ausencia real DEBE seguir bloqueando"


# ───────────── 4. ancla ─────────────

def test_las_ramas_nuevas_van_antes_del_unknown_final():
    import inspect
    src = inspect.getsource(_clf)
    assert src.index('return "recipe_unquantified"') < src.rindex('return "unknown"')
    assert src.index('"unit_mismatch"') < src.rindex('return "unknown"')


def test_el_filtro_de_ruido_de_envase_sigue_gobernado_por_su_knob():
    """La supresión de `unit_mismatch` NO es nueva ni de este P-fix: vive en
    P1-COHERENCE-PACKAGING-NOISE desde 2026-07-07 y es reversible sin redeploy."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    assert "MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER" in src
    i = src.index("def _is_packaging_noise")
    assert 'd.get("hypothesis") == "unit_mismatch"' in src[i:i + 400]
