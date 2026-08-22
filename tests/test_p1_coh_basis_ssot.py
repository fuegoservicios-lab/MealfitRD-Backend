"""[P1-COH-BASIS-SSOT · 2026-08-22] Arreglé la mitad de un espejo y dejé la otra mirando a otro sitio.

## Qué rompí

`P0-SHOPPING-CYCLE-DAYS` cambió la fuente de días del agregador Y de
`expected_sum_from_recipes` al SSOT `shopping_source_days` (`days + _archived_days`).

Pero el guard tiene una TERCERA lectura: `_basis_scale`, que espeja la proyección
`base_duration_scale = 7.0 / num_days` del agregador para poder comparar contra la lista
SEMANAL. Esa seguía leyendo `plan_result["days"]` a pelo.

Resultado en el plan real `2245eb45` (3 días vivos + 4 archivados):

- agregador: `num_days = 7` → proyección ×1.0 → lista = suma de 7 días.
- guard: `_basis_scale = 7/3 = 2.333` → esperado = suma de 7 días **× 2.333**.

Ratio comprado/esperado = **3/7 = 0.4286** en **46 alimentos**, con mínimo 0.424 y máximo
0.431. Dos alimentos sin relación con ratio idéntico ⇒ factor estructural, no incoherencia
— que es *literalmente* el criterio que el comentario de ese bloque ya usaba para
diagnosticar su propio bug original.

Se manifiesta SOLO en planes con días archivados, o sea justo en los que el shift ha
tocado. Un plan recién generado (`_archived_days` vacío) da los dos lados iguales y no
enseña nada: el plan nuevo `7af0499b` marcaba 0 undersupply mientras el viejo marcaba 46.

## Por qué importa aunque no bloquee

`MEALFIT_GUARD_UNDERSUPPLY_SEVERE` está en `False`, así que hoy no fuerza reintentos. Pero
el knob existe para encenderse «tras medir el history», y el history estaba llenándose de
39-46 fantasmas por plan. Encenderlo con esto dentro habría rechazado todo plan que
hubiera vivido un shift.

## La lección

Es la MISMA de `P0-SHOPPING-CYCLE-DAYS` una vuelta más adentro: **las dos mitades de un
espejo tienen que leer de la misma fuente.** Moví una y no busqué las demás. El test
`test_las_tres_lecturas_usan_el_ssot` existe para que la tercera no se quede atrás otra
vez — y para que una cuarta falle aquí antes que en producción.
"""
import ast
import io
import os

import pytest

import shopping_calculator
from shopping_calculator import run_shopping_coherence_guard


# Catálogo mínimo: con `get_master_ingredients() == []` el filtro-espejo
# `_is_verified_for_shopping` tumba TODO el lado esperado y cada ítem sale como
# `aggregated_only` — nunca se llega a comparar una magnitud. La primera versión de este
# fixture stubeaba a `[]` y el test PASABA contra el código roto: vacuo, la misma clase de
# guard-que-no-puede-fallar que este archivo cierra.
_CATALOGO = [{
    "name": "Arroz", "category": "Despensa", "price_per_lb": 30.0, "price_per_unit": None,
    "density_g_per_cup": 185.0, "density_g_per_unit": None, "default_unit": "lb",
    "container_weight_g": 453.6, "shelf_life_days": 365, "aliases": [],
}]


@pytest.fixture(autouse=True)
def catalogo_minimo(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: list(_CATALOGO))


def _dia(fecha):
    return {"day": 1, "date": fecha,
            "meals": [{"name": f"Comida {fecha}", "ingredients": ["70 g de arroz"]}]}


def _plan(vivos, archivados, comprado=490):
    """7 días × 70 g = 490 g de recetas; la lista lleva EXACTAMENTE esa suma."""
    lista = [{"name": "Arroz", "market_qty_numeric": comprado, "market_unit": "g",
              "base_qty": comprado, "base_unit": "g", "category": "Despensa"}]
    return {
        "cycle_start_date": "2026-08-18T00:00:00+00:00",
        "total_days_requested": 30,
        "days": [_dia(f"2026-08-{d:02d}") for d in vivos],
        "_archived_days": [_dia(f"2026-08-{d:02d}") for d in archivados],
        "aggregated_shopping_list": list(lista),
        "aggregated_shopping_list_weekly": [dict(x) for x in lista],
    }


def _magnitudes(plan):
    divs = run_shopping_coherence_guard(plan, mode_override="warn", multiplier=1.0) or []
    return [d for d in divs if d.get("side") == "magnitude"]


class TestElEspejoNoInventaEscasez:
    def test_plan_con_dias_archivados_no_inventa_divergencia(self):
        """El caso del plan 2245eb45: 3 vivos + 4 archivados, lista = suma de los 7.

        Contra el código roto esto daba `expected_qty=1143.33` (490 × 7/3) contra
        `actual_qty=490` y `delta_pct=0.5714` — la firma exacta de los 46 alimentos de
        producción. Se asserta sobre `side == "magnitude"` y no sobre una hipótesis
        concreta: la etiqueta varía (`magnitude_undersupply` / `pantry_overdeduct`) según
        el contexto de nevera, pero el defecto es el mismo.
        """
        ms = _magnitudes(_plan([22, 23, 24], [18, 19, 20, 21]))
        assert not ms, (
            "escasez fantasma por el factor 7/len(days): "
            f"{[(d['food'], d.get('expected_qty'), d.get('actual_qty')) for d in ms]}"
        )

    def test_plan_sin_archivados_sigue_igual_que_antes(self):
        """Los 7 días vivos y ninguno archivado: la conducta previa, intacta."""
        assert not _magnitudes(_plan([18, 19, 20, 21, 22, 23, 24], []))

    def test_una_escasez_REAL_se_sigue_viendo(self):
        """El guard no se ha vuelto ciego: si la lista compra de menos, lo dice."""
        assert _magnitudes(_plan([22, 23, 24], [18, 19, 20, 21], comprado=100)), (
            "con 100 g contra 490 g esperados el guard TIENE que protestar"
        )


class TestLasTresLecturasUsanElSSOT:
    """`shopping_source_days` es la fuente; quien la esquive vuelve a desincronizar."""

    def _fuente(self):
        ruta = os.path.join(os.path.dirname(__file__), "..", "shopping_calculator.py")
        return io.open(ruta, encoding="utf-8").read()

    @pytest.mark.parametrize("fn", [
        "get_shopping_list_delta",          # el que construye la lista
        "expected_sum_from_recipes",        # el lado esperado del guard
        "run_shopping_coherence_guard",     # la escala de base del espejo
    ])
    def test_cada_lectura_llama_al_ssot(self, fn):
        src = self._fuente()
        arbol = ast.parse(src)
        nodo = next(n for n in ast.walk(arbol)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        cuerpo = ast.get_source_segment(src, nodo) or ""
        assert "shopping_source_days" in cuerpo, (
            f"{fn} decide su base de días por su cuenta; las dos mitades de un espejo "
            f"tienen que leer de la misma fuente (la lección de P0-SHOPPING-CYCLE-DAYS)"
        )

    def test_la_escala_de_base_ya_no_lee_la_ventana_viva(self):
        src = self._fuente()
        arbol = ast.parse(src)
        nodo = next(n for n in ast.walk(arbol)
                    if isinstance(n, ast.FunctionDef) and n.name == "run_shopping_coherence_guard")
        cuerpo = ast.get_source_segment(src, nodo) or ""
        assert '_n_days_basis = len(plan_result.get("days")' not in cuerpo, (
            "`_n_days_basis` volvió a contar sólo la ventana viva: con días archivados eso "
            "escala el lado esperado ×7/len(days) y fabrica escasez en TODOS los alimentos"
        )
