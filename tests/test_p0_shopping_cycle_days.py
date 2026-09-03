"""[P0-SHOPPING-CYCLE-DAYS · 2026-08-22] La lista de compras no puede nacer de una ventana que encoge.

## El incidente que ancla estos tests

Plan real `2245eb45` (30 días, country=US). El generador entregó 3 días de menú y una
lista de **48 alimentos** — con Pechuga de pollo, Cebolla, Habichuelas negras, Pan
integral y Champiñones. El shift rodante fue podando los días vividos hacia
`_archived_days` hasta dejar `days == []`. Cualquier recálculo posterior reconstruyó la
lista desde esa ventana ya erosionada y la SOBRESCRIBIÓ: quedaron **25 alimentos**, que
son *exactamente* el conjunto canónico del último día superviviente (medido: día 1 → 21
ítems, día 2 → 22, día 3 → 25 = lo publicado, los tres juntos → 48).

El usuario pulsó «ya compré la lista» y su nevera nació como espejo fiel de esa lista
mutilada: 25 filas con UNA sola proteína (Huevo), sin cebolla y sin almidón básico. El
chunk siguiente intentó cocinar 3 días nuevos contra esa nevera, falló el gate de
despensa y quedó en `pending_user_action` — el usuario se quedó sin menú.

## Por qué nadie lo vio

`expected_sum_from_recipes` (el lado ESPERADO del coherence guard) leía el MISMO
`plan_data["days"]` encogido que el lado COMPRADO. Los dos lados se recortan a la vez y
la divergencia se cancela: la telemetría del plan pasó de 31 divergencias a 6 justo
después de la amputación. Es decir, **mutilar la lista MEJORÓ la métrica del guard**.

## El contrato que fijan estos tests

`shopping_source_days(plan_data)` es el ÚNICO sitio que decide desde qué días se agrega
la lista, y lo usan LOS DOS lados (builder y guard) para que no puedan volver a
divergir. La unión se acota al ciclo vivo del plan porque `_archived_days` nunca se
vacía, ni al renovar (`chat_history_context.py:204`) — sin ese filtro un plan renovado
arrastraría alimentos de la temporada anterior.

Nota sobre la aritmética: agregar MÁS días no infla la compra. El total es
`Σ(ingredientes) × (7/num_days) × cycle_qty_multiplier`, o sea
`promedio_por_día × días_del_ciclo` — invariante en `num_days`. Con más días el promedio
es mejor estimador, no mayor.
"""
import pytest

import shopping_calculator
from shopping_calculator import (
    shopping_source_days,
    expected_sum_from_recipes,
)


@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    """Mismo stub que el resto de la suite del agregador: sin DB, fallback inline."""
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


def _dia(fecha, nombre_comida, ingredientes, day=1):
    return {
        "day": day,
        "date": fecha,
        "meals": [{"name": nombre_comida, "ingredients": list(ingredientes)}],
    }


# El plan del incidente, reducido a su esqueleto: lo que se comió ya está archivado y
# la ventana viva quedó vacía.
PLAN_INCIDENTE = {
    "cycle_start_date": "2026-08-18T22:52:25.610005+00:00",
    "total_days_requested": 30,
    "days": [],
    "_archived_days": [
        _dia("2026-08-18", "Guiso de Habichuelas Negras", ["25 g de habichuelas negras secas", "1 cebolla"]),
        _dia("2026-08-19", "Horneado con Pechuga", ["100 g de pechuga de pollo", "250 g de champiñones"]),
        _dia("2026-08-20", "Estofado de Papa", ["2 papas medianas", "1 taza de espinaca"]),
    ],
}


class TestFuenteDeDias:
    def test_ventana_viva_vacia_no_significa_plan_sin_menu(self):
        """El caso exacto del incidente: days=[] con 3 días archivados NO es un plan vacío."""
        assert shopping_source_days(PLAN_INCIDENTE) != []
        assert len(shopping_source_days(PLAN_INCIDENTE)) == 3

    def test_une_archivados_y_vivos_en_orden_cronologico(self):
        plan = dict(PLAN_INCIDENTE)
        plan["days"] = [_dia("2026-08-21", "Avena", ["45 g de avena"])]
        dias = shopping_source_days(plan)
        assert len(dias) == 4
        # los archivados van primero; la ventana viva cierra
        assert dias[-1]["date"] == "2026-08-21"
        assert dias[0]["date"] == "2026-08-18"

    def test_plan_sano_sin_archivados_es_byte_identico_al_comportamiento_previo(self):
        """Sin `_archived_days` el helper devuelve exactamente `days` — cero cambio."""
        plan = {"days": [_dia("2026-08-21", "Avena", ["45 g de avena"])]}
        assert shopping_source_days(plan) == plan["days"]

    def test_no_arrastra_dias_de_una_renovacion_anterior(self):
        """`_archived_days` nunca se vacía: los días previos al ciclo vivo quedan fuera."""
        plan = dict(PLAN_INCIDENTE)
        plan["_archived_days"] = [
            _dia("2026-07-02", "Plan viejo", ["200 g de salmón"]),  # temporada anterior
        ] + PLAN_INCIDENTE["_archived_days"]
        nombres = " ".join(
            i for d in shopping_source_days(plan)
            for m in d["meals"] for i in m["ingredients"]
        )
        assert "salmón" not in nombres
        assert "pechuga de pollo" in nombres

    def test_recorta_al_total_de_dias_pedidos(self):
        plan = {
            "cycle_start_date": "2026-08-01T00:00:00+00:00",
            "total_days_requested": 3,
            "days": [],
            "_archived_days": [_dia(f"2026-08-{d:02d}", "X", ["10 g de arroz"]) for d in range(2, 12)],
        }
        assert len(shopping_source_days(plan)) == 3

    def test_knob_apagado_restaura_la_conducta_previa(self, monkeypatch):
        monkeypatch.setenv("MEALFIT_SHOPPING_SOURCE_INCLUDES_ARCHIVED", "false")
        assert shopping_source_days(PLAN_INCIDENTE) == []

    @pytest.mark.parametrize("basura", [None, [], {}, "no soy un plan", {"days": "tampoco"}])
    def test_fail_safe_ante_entrada_corrupta(self, basura):
        assert shopping_source_days(basura) == []


class TestElGuardDejaDeSerCiego:
    """`expected_sum_from_recipes` es el lado ESPERADO del coherence guard.

    Mientras leyera sólo `days`, un plan post-shift le daba `{}` y NINGUNA ausencia
    podía producir divergencia `expected_only` — el guard no podía ver que faltaba el
    pollo ni aunque faltara.
    """

    def test_ve_los_ingredientes_de_dias_archivados(self):
        esperado = expected_sum_from_recipes(PLAN_INCIDENTE)
        assert esperado, "con days=[] y 3 archivados el lado esperado no puede salir vacío"
        blob = " ".join(esperado.keys()).lower()
        assert "pollo" in blob
        assert "habichuela" in blob

    def test_plan_realmente_vacio_sigue_devolviendo_vacio(self):
        """No inventamos señal donde no hay menú: sin días de ninguna clase, {}."""
        assert expected_sum_from_recipes({"days": [], "_archived_days": []}) == {}


class TestListaDeCompras:
    """El builder y el guard deben mirar la MISMA fuente de días."""

    def test_builder_y_guard_comparten_la_fuente(self):
        """Si divergen, la lista vuelve a poder encoger sin que el guard lo note."""
        import inspect
        cuerpo_builder = inspect.getsource(shopping_calculator.get_shopping_list_delta)
        cuerpo_guard = inspect.getsource(shopping_calculator.expected_sum_from_recipes)
        assert "shopping_source_days" in cuerpo_builder, (
            "get_shopping_list_delta debe tomar los días del SSOT, no de plan_result['days']"
        )
        assert "shopping_source_days" in cuerpo_guard, (
            "expected_sum_from_recipes debe tomar los días del SSOT, no de plan_data['days']"
        )

    def test_alimento_que_solo_vive_en_un_dia_archivado_sobrevive_al_recalculo(self):
        """La regresión literal del incidente: el pollo no puede desaparecer de la lista."""
        salida = shopping_calculator.get_shopping_list_delta(
            "guest", PLAN_INCIDENTE, is_new_plan=True, structured=True,
            multiplier=4.29, inventory_override=[], consumed_override=[],
        )
        nombres = " ".join(
            str(i.get("name") or "") for i in (salida or []) if isinstance(i, dict)
        ).lower()
        assert "pollo" in nombres, f"el pollo se perdió; lista={nombres}"
        assert "habichuela" in nombres
