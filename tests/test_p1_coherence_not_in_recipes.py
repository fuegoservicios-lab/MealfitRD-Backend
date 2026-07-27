"""[2026-07-27] Los dos casos de "expected=0" YA se distinguen. No añadas una etiqueta.

## La pregunta

Con el guard ya en `block`, la única clase de divergencia que quedaba viva era
`recipe_unquantified` — 36 de 84 (43%) sobre 19 planes. Parecía mezclar dos cosas muy
distintas:

    "Sal al gusto"   ->  la receta SÍ nombra la sal, sin cantidad     BENIGNO
    <nada>           ->  ninguna receta la menciona                   DEFECTO: se compra y no se usa

`_normalize_food_dict_to_grams` descarta las cantidades <= 0, así que tras el filtro los dos
llegan al comparador igual: en la lista, con `expected_qty = 0`.

## La respuesta, medida

**Ya están separados, por `side`, en capas distintas del guard:**

    Azúcar  side=aggregated_only  magnitude=False  (capa de PRESENCIA)
    Sal     side=magnitude        magnitude=True   recipe_unquantified

La capa de presencia compara NOMBRES contra `expected_raw` SIN filtrar por cantidad, así que un
alimento que ninguna receta menciona no llega nunca al comparador de magnitudes: se reporta
antes como `aggregated_only`. Y la sal, que sí está nombrada, no la ve la capa de presencia y
cae en magnitud con cantidad cero.

## Lo que se intentó y se descartó

Se llegó a implementar una etiqueta `not_in_recipes` en el comparador de magnitudes, capturando
los nombres antes del filtro de cantidades. Medida sobre 19 planes vivos: **0 ocurrencias**. Era
código INALCANZABLE — el caso que pretendía marcar lo intercepta la capa de presencia primero.
Revertido.

Este archivo es lo que queda: la prueba de que la distinción existe, para que nadie (yo incluido)
vuelva a "arreglar" algo que ya funciona.

⚠️ Ninguno de los dos bloquea: son fantasmas `delta=inf`, excluidos del subset crítico por
diseño. Importa ahora que el guard corre en `block`.
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


def _plan(lineas_receta: list[str], lista: list[dict]) -> dict:
    return {
        "days": [{"day": 1, "meals": [{"name": "Almuerzo", "ingredients": lineas_receta}]}],
        "aggregated_shopping_list_weekly": lista,
    }


_LISTA_CON_AMBOS = [
    {"name": "Pechuga de pollo", "base_qty": 1400.0, "base_unit": "g"},
    {"name": "Sal", "base_qty": 454.0, "base_unit": "g"},      # nombrada "al gusto"
    {"name": "Azúcar", "base_qty": 900.0, "base_unit": "g"},   # nadie la menciona
]
_RECETA = ["200g de pechuga de pollo", "Sal al gusto"]


def _por_alimento(divs):
    return {str(d.get("food")): d for d in (divs or [])}


# ───────────── 1. la distinción existe ─────────────

def test_los_dos_casos_caen_en_capas_distintas():
    """El discriminador es `side`, no la hipótesis. Si algún día los dos cayeran en la misma
    capa, el fantasma real quedaría enterrado entre los condimentos (43% de las divergencias)."""
    divs = sc.run_shopping_coherence_guard(
        _plan(_RECETA, _LISTA_CON_AMBOS), mode_override="warn", multiplier=1.0) or []
    d = _por_alimento(divs)
    _sal = next((v for k, v in d.items() if "sal" in k.lower()), None)
    _azu = next((v for k, v in d.items() if "car" in k.lower()), None)
    assert _sal is not None and _azu is not None, f"faltan divergencias: {list(d)}"
    assert _azu.get("side") == "aggregated_only", (
        f"el alimento que NINGUNA receta menciona debe salir en la capa de PRESENCIA "
        f"(`aggregated_only`), no confundido con un condimento. Salió: {_azu}"
    )
    assert _sal.get("side") == "magnitude" and _sal.get("hypothesis") == "recipe_unquantified", (
        f"el condimento nombrado sin cantidad debe caer en magnitud como "
        f"`recipe_unquantified`. Salió: {_sal}"
    )
    assert _sal.get("side") != _azu.get("side"), "los dos casos deben ser distinguibles"


def test_la_capa_de_presencia_no_filtra_por_cantidad():
    """La razón POR LA QUE funciona: la capa de presencia compara nombres contra `expected_raw`
    sin pasar por el filtro de cantidades > 0. Por eso la sal (0.0 pizca) NO aparece ahí."""
    divs = sc.run_shopping_coherence_guard(
        _plan(_RECETA, _LISTA_CON_AMBOS), mode_override="warn", multiplier=1.0) or []
    presencia = [d for d in divs if d.get("side") == "aggregated_only"]
    nombres = {str(d.get("food")).lower() for d in presencia}
    assert not any("sal" == n for n in nombres), (
        "la sal está NOMBRADA en la receta ('Sal al gusto'): no puede aparecer como ausente "
        f"de las recetas. Presencia: {nombres}"
    )


# ───────────── 2. no cambian el veredicto ─────────────

def test_ninguno_de_los_dos_bloquea():
    p = _plan(_RECETA, _LISTA_CON_AMBOS)
    sc.run_shopping_coherence_guard(p, mode_override="block", multiplier=1.0)
    assert not p.get("_shopping_coherence_block"), (
        f"ni un condimento ni un fantasma deben rechazar el plan: "
        f"{p.get('_shopping_coherence_block')}"
    )


def test_una_falta_REAL_sigue_bloqueando():
    """Guarda contra ablandar el guard: si la lista NO trae un alimento que la receta pide, el
    plan debe seguir bloqueando. Importa ahora que el modo es `block` en producción."""
    p = _plan(["200g de pechuga de pollo", "150g de pescado"],
              [{"name": "Pechuga de pollo", "base_qty": 1400.0, "base_unit": "g"}])
    sc.run_shopping_coherence_guard(p, mode_override="block", multiplier=1.0)
    assert p.get("_shopping_coherence_block"), "una ausencia real DEBE seguir bloqueando"


# ───────────── 3. ancla contra la "mejora" que ya se descartó ─────────────

def test_no_hay_etiqueta_not_in_recipes():
    """Se implementó y se revirtió: 0 ocurrencias en 19 planes vivos porque el caso que
    pretendía marcar lo intercepta la capa de presencia ANTES. Si alguien la reintroduce, este
    test se lo dice con el motivo."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    assert "not_in_recipes" not in src, (
        "vuelve a estar la etiqueta `not_in_recipes` en el comparador de magnitudes. Es código "
        "INALCANZABLE: un alimento que ninguna receta menciona lo reporta antes la capa de "
        "presencia como `side=aggregated_only`. Medido: 0 ocurrencias en 19 planes."
    )
