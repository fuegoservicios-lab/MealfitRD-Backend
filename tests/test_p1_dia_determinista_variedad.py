# -*- coding: utf-8 -*-
"""[P1-DIA-DETERMINISTA-VARIEDAD · 2026-09-09] La palanca de variedad estaba anulada por su consumidor.

## De dónde sale

Antes de encender `MEALFIT_DETERMINISTIC_DAY` para el dueño, le armé sus 14 días con su perfil real
(2.100 kcal, presupuesto `low`) y los medí. El resultado técnico era bueno y el producto, no:

```
14/14 días armados · 0 comidas sucias · calorías +0,0 %
variedad: 7 platos distintos en 56 comidas — el MISMO almuerzo los 14 días
```

**Causa:** `template_candidates` ya recibía `rotate=day_num` y rotaba la lista, pero
`elegir_plantilla` recorría la lista ENTERA y devolvía `cands[0]`, el mejor por macros. Rotar una
lista que el consumidor recorre entera no cambia quién gana: la rotación era **inerte**.

*Una palanca de variedad que el consumidor de la lista anula no es una palanca.* Es la misma forma
que `P1-PROTEIN-FLOOR-LAST-WORD` (un pase cableado en un solo camino) y que
`P1-I18N-DEAD-VEREDICTO` (un no-op que parece vivo): **algo que existe, corre y no tiene efecto**.

## La ventana de empate va MEDIDA, no supuesta

Sobre el perfil real, contando cuántos candidatos caen a menos de X del mejor score:

```
            +0,05   +0,10   +0,20   +0,30   +0,50
desayuno      1       1       2       4       5
almuerzo      1       1       2       2       4
cena          1       2       3       3       4
merienda      2       3       3       4       8
```

Con +0,05 hay UN elegible en tres de las cuatro franjas: por eso el plan era un bucle. El default
es **0,50**. El score es `2·|Δp|/p + |Δc|/c + |Δf|/f`, así que 0,50 es del orden de un 15 % de
desvío en proteína más un 10 % en los otros dos — y **ese desvío tiene quien lo recoja**
(`reconcile_protein_band_post_finalize`, el cerrador de banda). Comer mofongo catorce días no lo
recoge nadie.

## Resultado medido tras el arreglo

```
7 → 22 platos distintos en 56 comidas · peor repetición 9× → 5×
14/14 días · 0 comidas sucias · calorías −0,1 % · 0 platos repetidos dentro del mismo día
determinismo intacto: dos corridas, hash idéntico
```
"""
import pytest

import deterministic_day as dd


# ── El defecto exacto, anclado ───────────────────────────────────────────────
def test_la_rotacion_cambia_la_eleccion():
    """Si esto cae, `rotate` volvió a ser decorativo y el plan vuelve a ser un bucle."""
    cat = {"A": {"kcal_per_100g": 100, "protein_g_per_100g": 10, "carbs_g_per_100g": 10,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
           "B": {"kcal_per_100g": 100, "protein_g_per_100g": 9.5, "carbs_g_per_100g": 10.5,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
           "C": {"kcal_per_100g": 100, "protein_g_per_100g": 9.0, "carbs_g_per_100g": 11,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0}}

    def _t(tid, ing):
        return {"template_id": tid, "name": tid,
                "constituents": [{"name": ing, "grams": 100.0}]}

    por_id = {"t1": _t("t1", "A"), "t2": _t("t2", "B"), "t3": _t("t3", "C")}
    obj = {"kcal": 100.0, "protein_g": 10.0, "carbs_g": 10.0, "fats_g": 1.0}
    tids = ["t1", "t2", "t3"]

    vistos = {dd.elegir_plantillas(tids, obj, cat, por_id, "almuerzo", rotacion=r)[0][0]["template_id"]
              for r in range(3)}
    assert len(vistos) >= 2, (
        f"tres rotaciones dieron {vistos}: la rotación volvió a no cambiar nada — es exactamente el "
        f"defecto que este P-fix cerró (7 platos distintos en 56 comidas)")


def test_sin_rotacion_gana_el_mejor_por_macros():
    """La calidad no se negocia: rotación 0 sigue siendo el mejor candidato. Es lo que fija el
    contrato de `elegir_plantilla`, que se conserva como cabeza de la lista."""
    cat = {"A": {"kcal_per_100g": 100, "protein_g_per_100g": 10, "carbs_g_per_100g": 10,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
           "B": {"kcal_per_100g": 100, "protein_g_per_100g": 5, "carbs_g_per_100g": 15,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0}}
    por_id = {"bueno": {"template_id": "bueno", "name": "bueno",
                        "constituents": [{"name": "A", "grams": 100.0}]},
              "malo": {"template_id": "malo", "name": "malo",
                       "constituents": [{"name": "B", "grams": 100.0}]}}
    obj = {"kcal": 100.0, "protein_g": 10.0, "carbs_g": 10.0, "fats_g": 1.0}
    el = dd.elegir_plantilla(["malo", "bueno"], obj, cat, por_id, "almuerzo")
    assert el and el[0]["template_id"] == "bueno"


def test_la_ventana_de_empate_deja_fuera_lo_que_es_MUCHO_peor():
    """Ampliar la ventana sin tope convertiría «variedad» en «cualquier cosa»."""
    cat = {"A": {"kcal_per_100g": 100, "protein_g_per_100g": 10, "carbs_g_per_100g": 10,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
           "Z": {"kcal_per_100g": 100, "protein_g_per_100g": 0.1, "carbs_g_per_100g": 24,
                 "fats_g_per_100g": 0.1, "fiber_g_per_100g": 0}}
    por_id = {"ok": {"template_id": "ok", "name": "ok",
                     "constituents": [{"name": "A", "grams": 100.0}]},
              "horrible": {"template_id": "horrible", "name": "horrible",
                           "constituents": [{"name": "Z", "grams": 100.0}]}}
    obj = {"kcal": 100.0, "protein_g": 10.0, "carbs_g": 10.0, "fats_g": 1.0}
    ids = {t["template_id"] for t, _ in
           dd.elegir_plantillas(["ok", "horrible"], obj, cat, por_id, "almuerzo")}
    assert ids == {"ok"}, "entró un candidato lejísimos del objetivo: la ventana dejó de acotar"


def test_el_tope_de_elegibles_se_respeta(monkeypatch):
    cat = {f"I{i}": {"kcal_per_100g": 100, "protein_g_per_100g": 10 - i * 0.01,
                     "carbs_g_per_100g": 10, "fats_g_per_100g": 1, "fiber_g_per_100g": 0}
           for i in range(12)}
    por_id = {f"t{i}": {"template_id": f"t{i}", "name": f"t{i}",
                        "constituents": [{"name": f"I{i}", "grams": 100.0}]}
              for i in range(12)}
    obj = {"kcal": 100.0, "protein_g": 10.0, "carbs_g": 10.0, "fats_g": 1.0}
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_TIE_MAX", "4")
    assert len(dd.elegir_plantillas(list(por_id), obj, cat, por_id, "almuerzo")) == 4


def test_sin_candidatos_devuelve_lista_vacia_no_revienta():
    assert dd.elegir_plantillas([], {"kcal": 100.0}, {}, {}, "almuerzo") == []
    assert dd.elegir_plantilla([], {"kcal": 100.0}, {}, {}, "almuerzo") is None


# ── El desfase por franja ────────────────────────────────────────────────────
def test_las_franjas_no_rotan_en_formacion():
    """Rotar las cuatro con el mismo `day_num` las mueve juntas, y una plantilla que sale en dos
    franjas —«Arepitas de maíz» está en desayuno Y merienda— se repite el mismo día."""
    d0 = {s: dd._rotacion_de(0, s) for s in ("desayuno", "almuerzo", "cena", "merienda")}
    assert len(set(d0.values())) == 4, f"dos franjas comparten rotación: {d0}"


def test_la_rotacion_por_franja_es_ESTABLE_entre_procesos():
    """`hash()` de Python está salado por proceso: usarlo daría un plan distinto en cada arranque,
    y este módulo promete determinismo. Los valores de abajo son los de sha256, fijos para siempre.
    """
    assert dd._rotacion_de(0, "desayuno") == dd._rotacion_de(0, "Desayuno") == dd._rotacion_de(0, "DESAYUNO")
    esperados = {s: dd._rotacion_de(0, s) for s in ("desayuno", "almuerzo", "cena", "merienda")}
    for s, v in esperados.items():
        assert dd._rotacion_de(7, s) == v + 7, f"la rotación de {s} dejó de avanzar con el día"
        assert isinstance(v, int) and v >= 0


@pytest.mark.parametrize("dn", [None, 0, 3, "no soy un número"])
def test_rotacion_tolera_dias_raros(dn):
    try:
        v = dd._rotacion_de(dn, "cena")
    except Exception as e:
        pytest.fail(f"_rotacion_de reventó con day_num={dn!r}: {e!r}")
    assert isinstance(v, int)


# ── Lo que hace el armador con la lista ──────────────────────────────────────
def test_el_armador_consume_la_LISTA_y_no_el_ganador():
    """Dos cosas cuelgan de esto: la variedad, y que un plato sin receta congelada no tire el DÍA
    ENTERO al LLM (`construir_comida` devuelve `None` sin receta)."""
    import inspect

    src = inspect.getsource(dd.build_day_for_skeleton)
    assert "elegir_plantillas(" in src, (
        "el armador volvió a pedir un solo candidato: vuelven el bucle de 7 platos y el día "
        "perdido por un plato sin receta")
    assert "usadas_hoy" in src, "se perdió el guard de no repetir plantilla dentro del mismo día"


def test_una_plantilla_ya_usada_hoy_queda_de_RESPALDO_no_descartada():
    """Quedarse sin día por no repetir es peor que repetir: el orden cambia, el conjunto no."""
    import inspect

    src = inspect.getsource(dd.build_day_for_skeleton)
    assert "sorted(" in src and "usadas_hoy" in src, (
        "si esto pasó a ser un filtro en vez de un reorden, un día con pocos elegibles se cae "
        "entero al LLM por no querer repetir un plato")
