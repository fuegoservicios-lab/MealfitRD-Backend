"""[P1-COHERENCE-DAY-BASIS · 2026-07-26] El guard comparaba 3 días de recetas contra 7 de lista.

## El caso

`get_shopping_list_delta` PROYECTA a propósito, y lo dice en su propio comentario
(shopping_calculator.py ~10172):

    # Si hay 3 días generados, representan un ciclo rotativo. Promediamos por día
    # y proyectamos a 7 días.
    num_days = max(1, len(days))
    base_duration_scale = 7.0 / num_days
    effective_multiplier = multiplier * base_duration_scale

`expected_sum_from_recipes` NO espejaba ese factor. Así que el guard sumaba los días que
EXISTEN y los comparaba contra una lista proyectada a una semana: todo divergía por
7/num_days.

Medido sobre 19 planes vivos — los 19 con ≤3 días materializados, porque el guard corre en
`assemble_plan_node` ANTES de que los chunks llenen los días 4+ — y el factor encaja **al
decimal**, no aproximadamente:

    Pescado    574.7 / 3 × 7 = 1341.0    ← la lista dice 1341.0
    Cangrejo   225.0 / 3 × 7 =  525.0    ← la lista dice  525.0

Dos alimentos sin relación entre sí con el ratio idéntico ⇒ factor estructural, no
incoherencia. Tras el fix ambos desaparecen de las divergencias.

## Por qué importaba

La premisa de `P2-COH-WEEKLY-BASIS` ("la lista semanal ES la misma base que expected") solo es
cierta con la semana COMPLETA materializada. En modo `block` esto rechazaba casi cualquier
plan por un factor que no es un defecto — la razón por la que el guard estaba forzado a `warn`
en producción.

## Lo que destapó (seguimiento abierto)

Con el factor fuera, en el plan `fbe53a5b` (nevera **vacía**, 0 items, así que
`pantry_overdeduct` está descartado) afloran desajustes emparejados que antes quedaban tapados:

    Yogur    esperado=2324.8  lista=907.2   (39%)   ← 907.18 g = exactamente un pote de 2 lb
    Cebolla  esperado=1459.5  lista=600.0   (41%)
    Tomate   esperado=1575.0  lista=750.0   (48%)

No es redondeo de mercado: `apply_smart_market_units("Yogurt", 2324 g)` devuelve 6 Ud.
correctamente. El agregador calcula una cantidad distinta a la que piden las recetas. Ese es el
siguiente hilo, y es exactamente la clase que el guard existe para cazar.

El conteo total sube (858 → 891 divergencias en 19 planes) porque el ruido estructural se
convierte en señal emparejada; las 833 sin pareja no se mueven (condimentos "al gusto", clase
aparte). Subir el conteo NO es una regresión aquí: antes el guard no reportaba estos casos
porque el 7/3 los dejaba dentro de tolerancia por casualidad.

tooltip-anchor: P1-COHERENCE-DAY-BASIS
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc



def _magnitudes(divs) -> list:
    """⚠️ NO filtrar por `food == "Pechuga de pollo"`: el guard canonicaliza y devuelve
    **"Pollo"**. Mi primera versión filtraba por el nombre crudo, salía lista vacía siempre, y
    los tests positivos pasaban EN VACÍO mientras los negativos fallaban — el rojo fue lo único
    que delató que el fixture no medía nada."""
    return [d for d in (divs or []) if d.get("magnitude")]


def _plan(n_dias: int, gramos_por_dia: float = 100.0, con_semanal: bool = True) -> dict:
    """Plan sintético: `n_dias` días, cada uno con 1 comida de `gramos_por_dia` g de pollo."""
    p = {
        "days": [
            {"day": i + 1, "meals": [{
                "name": f"Almuerzo {i + 1}",
                "ingredients": [f"{gramos_por_dia:.0f}g de pechuga de pollo"],
            }]}
            for i in range(n_dias)
        ],
    }
    lista = [{"name": "Pechuga de pollo", "base_qty": gramos_por_dia * 7, "base_unit": "g",
              "market_qty_numeric": 1.0, "market_unit": "lb"}]
    if con_semanal:
        p["aggregated_shopping_list_weekly"] = lista
    else:
        p["aggregated_shopping_list"] = lista
    return p


# ───────────── 1. el efecto: 3 días de recetas emparejan con 7 de lista ─────────────

@pytest.mark.parametrize("n_dias", [1, 2, 3, 4])
def test_los_dias_parciales_emparejan_con_la_lista_semanal(n_dias):
    """La lista trae 7 días de pollo; el plan tiene `n_dias`. Con la normalización, el lado
    esperado se proyecta igual que el agregador y NO debe haber divergencia de magnitud."""
    p = _plan(n_dias)
    divs = sc.run_shopping_coherence_guard(p, mode_override="warn", multiplier=1.0) or []
    mags = _magnitudes(divs)
    assert not mags, f"{n_dias} días no deben diverger contra la lista semanal: {mags}"


def test_sin_la_normalizacion_diverge(monkeypatch):
    """Rojo del contrato: apagando el knob vuelve la divergencia estructural 7/3."""
    monkeypatch.setenv("MEALFIT_COHERENCE_DAY_BASIS_NORM", "false")
    divs = sc.run_shopping_coherence_guard(_plan(3), mode_override="warn", multiplier=1.0) or []
    mags = _magnitudes(divs)
    assert mags, "sin el knob, 3 días contra lista de 7 DEBE diverger (es el bug que se cierra)"
    d = mags[0]
    assert d["actual_qty"] / d["expected_qty"] == pytest.approx(7.0 / 3.0, rel=0.02)


def test_siete_dias_es_no_op():
    """Con la semana completa el factor es 1.0: el fix no debe tocar el caso ya correcto."""
    divs = sc.run_shopping_coherence_guard(_plan(7), mode_override="warn", multiplier=1.0) or []
    assert not _magnitudes(divs), divs


# ───────────── 2. una divergencia REAL sigue reportándose ─────────────

def test_no_silencia_un_desajuste_genuino():
    """El fix no es un mute: si la lista trae la mitad de lo proyectado, se reporta."""
    p = _plan(3)
    p["aggregated_shopping_list_weekly"][0]["base_qty"] = 100.0 * 7 / 2.0
    divs = sc.run_shopping_coherence_guard(p, mode_override="warn", multiplier=1.0) or []
    mags = _magnitudes(divs)
    assert mags, "la mitad de lo necesario DEBE seguir divergiendo"
    assert mags[0]["actual_qty"] == pytest.approx(mags[0]["expected_qty"] / 2.0, rel=0.05)


# ───────────── 3. el alcance del fix ─────────────

def test_no_aplica_a_la_lista_ACTIVA():
    """Solo se espeja contra la lista SEMANAL. La activa (quincenal/mensual) tiene otra base —
    es justo el caso que `P2-COH-WEEKLY-BASIS` evita usando la semanal cuando existe."""
    p = _plan(3, con_semanal=False)
    divs = sc.run_shopping_coherence_guard(p, mode_override="warn", multiplier=1.0) or []
    mags = _magnitudes(divs)
    assert mags, "sin lista semanal NO se escala (base desconocida): la divergencia se conserva"


def test_plan_sin_dias_no_revienta():
    p = {"days": [], "aggregated_shopping_list_weekly": [
        {"name": "Pechuga de pollo", "base_qty": 700.0, "base_unit": "g"}]}
    assert isinstance(sc.run_shopping_coherence_guard(p, mode_override="warn", multiplier=1.0), list)


# ───────────── 4. el knob ─────────────

def test_el_knob_nace_en_true():
    """Nace ON, no OFF como los gates de calidad: no añade rechazos, elimina falsos positivos
    de una comparación mal planteada."""
    import os
    os.environ.pop("MEALFIT_COHERENCE_DAY_BASIS_NORM", None)
    assert sc._get_coherence_day_basis_norm_knob() is True


@pytest.mark.parametrize("valor,esperado", [("false", False), ("0", False), ("true", True)])
def test_el_knob_se_puede_apagar(monkeypatch, valor, esperado):
    monkeypatch.setenv("MEALFIT_COHERENCE_DAY_BASIS_NORM", valor)
    assert sc._get_coherence_day_basis_norm_knob() is esperado


# ───────────── 5. ancla: la fórmula es la MISMA del agregador ─────────────

def test_espeja_la_formula_del_agregador():
    """Ancla de la clase: si alguien cambia la proyección del agregador y no la de aquí, el
    guard vuelve a comparar bases distintas — el bug que este P-fix cierra. Ambos lados deben
    seguir escribiendo `7.0 /`."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    assert "base_duration_scale = 7.0 / num_days" in src, (
        "cambió la proyección del agregador: revisa el espejo del guard"
    )
    i = src.index("P1-COHERENCE-DAY-BASIS · 2026-07-26] El guard comparaba")
    bloque = src[i:i + 2600]
    assert "7.0 / float(_n_days_basis)" in bloque, "el guard debe usar la misma fórmula"
