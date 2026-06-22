"""[P2-SHOPPING-COMPLETENESS · 2026-06-21] Mínimo de completitud de la lista de compras escalado
por duración (Fase 5 del build "todo terreno").

El owner: "la lista de compras debe tener un mínimo de alimentos para que los planes estén al 100%,
dependiendo si es de 7, 15 o 30 días". La lista DERIVA del plan (cuyos gates de variedad/proteína/
calorías ya evitan planes degenerados), así que el piso es defensa-en-profundidad + observabilidad:
- is_empty (lista vacía CON recetas) = bug claro del builder/drop → reject (gate independiente del
  modo del coherence guard).
- is_sparse (pocos distintos vs el mínimo escalado por días) = señal suave → telemetría.
"""
import graph_orchestrator as go


def _plan(items, with_recipes=True):
    p = {"aggregated_shopping_list": list(items)}
    p["days"] = [{"meals": [{"ingredients": ["algo"] if with_recipes else None}]}]
    return p


def _items(n):
    return [{"name": f"Alimento {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# distinct + exclusión de urgentes
# ---------------------------------------------------------------------------
def test_cuenta_distintos_excluye_urgentes():
    plan = _plan([
        {"name": "Pollo"},
        {"name": "Arroz"},
        {"name": "Suplemento X", "category": "🚨 Compra Urgente"},  # excluido
    ])
    sc = go._shopping_list_completeness(plan, {"groceryDuration": "weekly"})
    assert sc["distinct"] == 2, "Los ítems urgentes (supplement) no cuentan para la lista base."


# ---------------------------------------------------------------------------
# expected_min escala con la duración (conservador, con tope)
# ---------------------------------------------------------------------------
def test_expected_min_escala_por_duracion():
    base = go.SHOPPING_MIN_DISTINCT_BASE
    per = go.SHOPPING_MIN_DISTINCT_PER_WEEK
    cap = go.SHOPPING_MIN_DISTINCT_CAP
    w = go._shopping_list_completeness(_plan(_items(1)), {"groceryDuration": "weekly"})["expected_min"]
    b = go._shopping_list_completeness(_plan(_items(1)), {"groceryDuration": "biweekly"})["expected_min"]
    m = go._shopping_list_completeness(_plan(_items(1)), {"groceryDuration": "monthly"})["expected_min"]
    assert w == base                      # 7d → base
    assert b == min(cap, base + per)      # 15d (2 semanas) → base + 1×per
    assert m == min(cap, base + 3 * per)  # 30d (≈4 semanas) → base + 3×per
    assert w <= b <= m, "El mínimo debe crecer (o mantenerse) con la duración."
    assert m <= cap, "El mínimo está capeado (la variedad no escala linealmente con los días)."


# ---------------------------------------------------------------------------
# is_empty (señal dura)
# ---------------------------------------------------------------------------
def test_is_empty_con_recetas_y_lista_vacia():
    sc = go._shopping_list_completeness(_plan([], with_recipes=True), {"groceryDuration": "weekly"})
    assert sc["is_empty"] is True
    assert sc["is_sparse"] is False  # vacía no es "sparse" (es el caso dura aparte)


def test_lista_vacia_sin_recetas_no_es_empty():
    # Sin ingredientes en las recetas no hay nada que comprar → lista vacía es legítima.
    sc = go._shopping_list_completeness(_plan([], with_recipes=False), {"groceryDuration": "weekly"})
    assert sc["is_empty"] is False


# ---------------------------------------------------------------------------
# is_sparse vs completa
# ---------------------------------------------------------------------------
def test_is_sparse_pocos_distintos():
    sc = go._shopping_list_completeness(_plan(_items(2)), {"groceryDuration": "weekly"})
    assert sc["is_sparse"] is True   # 2 < base(6)
    assert sc["is_empty"] is False


def test_lista_completa_no_marca_nada():
    sc = go._shopping_list_completeness(_plan(_items(go.SHOPPING_MIN_DISTINCT_BASE + 3)),
                                        {"groceryDuration": "weekly"})
    assert sc["is_sparse"] is False
    assert sc["is_empty"] is False


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def test_marker_y_gate_en_source():
    src = open(go.__file__, encoding="utf-8").read()
    assert "P2-SHOPPING-COMPLETENESS" in src
    assert "SHOPPING_EMPTY_LIST_REJECT" in src
    assert "_shopping_completeness" in src  # assemble setea, review lee
