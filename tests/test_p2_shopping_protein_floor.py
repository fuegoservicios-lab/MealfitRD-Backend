"""[P2-SHOPPING-PROTEIN-FLOOR · 2026-08-22] El piso de la lista contaba nombres, no comida.

## El hueco

`_shopping_list_completeness` medía UNA cosa: cuántos nombres distintos hay en
`aggregated_shopping_list`, contra un mínimo escalado por semanas (12 para 30 días). Era
ciego a la CATEGORÍA.

La lista publicada del plan real `2245eb45` tenía 25 ítems y pasaba limpia — ni
`is_empty` ni `is_sparse` — con este desglose contra `master_ingredients.category`:

| Categoría | Ítems |
|---|---|
| Despensa | 12 |
| Lácteos | 4 |
| Vegetales | 4 |
| Frutas | 2 |
| **Proteínas** | **1** (Huevo) |
| Víveres | 1 (Papa) |

25 ≥ 12, así que el sistema declaró la lista completa. Con esa lista el usuario llenó su
nevera, y el bloque siguiente tenía que componer tres días de comidas con **una sola
proteína**. El sistema sabía contar alimentos pero no sabía preguntar «¿con esto se
puede comer?».

## El segundo hueco: el veredicto caducaba

`_shopping_completeness` se calculaba SOLO en `assemble_plan_node` y nadie lo volvía a
medir. `grep _shopping_completeness backend/routers/plans.py` daba **0 matches**: el
recálculo reescribía `aggregated_shopping_list*` y jamás re-medía. El plan quedó
persistido afirmando `distinct: 49` mientras publicaba 25 — un veredicto que describe
una lista que ya no existe es peor que ninguno, porque un operador lo cree.

## Alcance deliberado

Esto MIDE y AVISA; no bloquea la generación. Un piso de proteínas que rechace planes
rompería dietas legítimamente bajas en variedad (veganas estrictas, presupuestos
mínimos), y el modo de fallo que nos ocupa no era «se generó mal» sino «la lista se
erosionó después» — eso lo cierra `P0-SHOPPING-CYCLE-DAYS`. Aquí ponemos el termómetro
que faltaba y hacemos que deje de mentir.
"""
import pytest

import graph_orchestrator as go


def _lista(*pares):
    return [{"name": n, "category": c} for n, c in pares]


# La lista real del incidente, con las categorías del catálogo.
LISTA_DEL_INCIDENTE = _lista(
    ("Huevo", "Proteínas"),
    ("Queso blanco", "Lácteos"), ("Queso provolone", "Lácteos"),
    ("Yogurt", "Lácteos"), ("Leche descremada", "Lácteos"),
    ("Papa", "Víveres"),
    ("Espinacas", "Vegetales"), ("Zanahoria", "Vegetales"),
    ("Ajo", "Vegetales"), ("Ají morrón", "Vegetales"),
    ("Pera", "Frutas"), ("Uva", "Frutas"),
    *[(f"Condimento {i}", "Despensa") for i in range(12)],
)

LISTA_SANA = LISTA_DEL_INCIDENTE + _lista(
    ("Pechuga de pollo", "Proteínas"),
    ("Habichuelas negras", "Proteínas"),
    ("Soya texturizada", "Proteínas"),
)

_PLAN_CON_RECETAS = {"days": [{"meals": [{"ingredients": ["1 huevo"]}]}]}


def _medir(lista, duracion="monthly"):
    plan = dict(_PLAN_CON_RECETAS)
    plan["aggregated_shopping_list"] = lista
    return go._shopping_list_completeness(plan, {"groceryDuration": duracion})


class TestCuentaProteinas:
    def test_el_caso_del_incidente_queda_marcado(self):
        veredicto = _medir(LISTA_DEL_INCIDENTE)
        assert veredicto["distinct_proteins"] == 1
        assert veredicto["is_protein_starved"] is True, (
            "25 ítems con UNA proteína para 30 días pasaba como lista completa"
        )

    def test_la_metrica_vieja_seguia_diciendo_que_estaba_bien(self):
        """El punto entero: los contadores previos NO veían el problema."""
        veredicto = _medir(LISTA_DEL_INCIDENTE)
        assert veredicto["is_empty"] is False
        assert veredicto["is_sparse"] is False
        assert veredicto["distinct"] >= veredicto["expected_min"]

    def test_una_lista_sana_no_se_marca(self):
        veredicto = _medir(LISTA_SANA)
        assert veredicto["distinct_proteins"] == 4
        assert veredicto["is_protein_starved"] is False

    def test_los_lacteos_no_cuentan_como_proteina_principal(self):
        """4 lácteos no son 4 proteínas: si contaran, el incidente habría pasado."""
        solo_lacteos = _lista(*[(f"Queso {i}", "Lácteos") for i in range(6)])
        assert _medir(solo_lacteos)["distinct_proteins"] == 0

    def test_el_piso_es_un_knob(self, monkeypatch):
        monkeypatch.setattr(go, "SHOPPING_MIN_PROTEINS", 1)
        assert _medir(LISTA_DEL_INCIDENTE)["is_protein_starved"] is False

    def test_lista_vacia_no_se_marca_como_hambrienta(self):
        """`is_empty` ya cubre ese caso; marcar dos veces confunde el diagnóstico."""
        veredicto = _medir([])
        assert veredicto["is_empty"] is True
        assert veredicto["is_protein_starved"] is False

    @pytest.mark.parametrize("basura", [None, "no soy lista", [{"sin": "nombre"}]])
    def test_fail_safe(self, basura):
        plan = dict(_PLAN_CON_RECETAS)
        plan["aggregated_shopping_list"] = basura
        v = go._shopping_list_completeness(plan, {})
        assert isinstance(v, dict) and "distinct_proteins" in v

    def test_acepta_la_categoria_sin_acento(self):
        """El catálogo escribe «Proteínas»; un recálculo o un espejo pueden traerla plana."""
        sin_acento = _lista(("Pollo", "Proteinas"), ("Atún", "PROTEÍNAS"))
        assert _medir(sin_acento)["distinct_proteins"] == 2


class TestElVeredictoDejaDeCaducar:
    def test_el_recalculo_vuelve_a_medir(self):
        """`grep _shopping_completeness routers/plans.py` daba 0 matches.

        El plan quedaba afirmando `distinct: 49` mientras publicaba 25.
        """
        import io
        import os
        ruta = os.path.join(os.path.dirname(__file__), "..", "routers", "plans.py")
        fuente = io.open(ruta, encoding="utf-8").read()
        assert "_shopping_completeness" in fuente, (
            "/recalculate-shopping-list reescribe las 4 listas pero no re-mide la "
            "completitud: el veredicto persistido describe una lista que ya no existe"
        )

    def test_el_bloque_no_referencia_nombres_inexistentes(self):
        """Un `NameError` dentro del `try` fail-open sería INVISIBLE.

        Al escribir este fix puse `plan_id_for_lock`, que no existe en ese scope: los
        tests pasaron igual porque el `except` se lo habría tragado y el veredicto
        habría seguido caducando en silencio. El fail-open protege el recalc, pero
        convierte un typo en un no-op indistinguible del éxito — el mismo modo de fallo
        que la auditoría de PayPal encontró en un `UPDATE` sin `RETURNING`.
        """
        import ast
        import io
        import os
        ruta = os.path.join(os.path.dirname(__file__), "..", "routers", "plans.py")
        fuente = io.open(ruta, encoding="utf-8").read()
        arbol = ast.parse(fuente)

        externa = next(
            n for n in ast.walk(arbol)
            if isinstance(n, ast.FunctionDef) and n.name == "api_recalculate_shopping_list"
        )
        ligados = {a.arg for a in externa.args.args}
        for nodo in ast.walk(externa):
            if isinstance(nodo, ast.Name) and isinstance(nodo.ctx, ast.Store):
                ligados.add(nodo.id)
            elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
                ligados.update((a.asname or a.name).split(".")[0] for a in nodo.names)
            elif isinstance(nodo, ast.FunctionDef):
                ligados.add(nodo.name)
                ligados.update(a.arg for a in nodo.args.args)
            elif isinstance(nodo, ast.ExceptHandler) and nodo.name:
                ligados.add(nodo.name)

        modulo = {n.id for n in ast.walk(arbol)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        modulo.update(n.name for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef))
        for n in ast.walk(arbol):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                modulo.update((a.asname or a.name).split(".")[0] for a in n.names)
        import builtins
        conocidos = ligados | modulo | set(dir(builtins))

        usados = set()
        for nodo in ast.walk(externa):
            if isinstance(nodo, ast.Name) and isinstance(nodo.ctx, ast.Load):
                usados.add(nodo.id)
        huerfanos = {u for u in usados if u not in conocidos}
        assert not huerfanos, (
            f"nombres usados en api_recalculate_shopping_list que nadie liga: "
            f"{sorted(huerfanos)} — un NameError ahí dentro sería silencioso"
        )
