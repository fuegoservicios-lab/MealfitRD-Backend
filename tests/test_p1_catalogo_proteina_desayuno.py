# -*- coding: utf-8 -*-
"""[P1-CATALOGO-PROTEINA-DESAYUNO · 2026-09-09] El desayuno criollo es carbohidrato casi puro.

## De dónde sale

Con la variedad ya arreglada (`P1-DIA-DETERMINISTA-VARIEDAD`), 14 días deterministas daban 22
platos distintos en 56 comidas y el dueño pidió ~40. El embudo dijo que **el catálogo no era el
cuello**: tras todos los filtros sobrevivían 23-34 plantillas por franja y sólo 5-6 pasaban la
última puerta. Y el barrido dijo que ensanchar esa puerta tampoco era la vía:

```
ventana   platos   días bajo el piso de proteína   peor corto
  0,50      22                  2                    0,7 g
  0,75      27                  2                   18,7 g
  1,00      33                  5                   27,7 g
  1,25      44                  9                   27,7 g
```

Llegar a 44 platos por ahí cuesta 9 de 14 días bajo el piso clínico. **La ventana ya estaba en su
sitio.** Lo que faltaba era materia prima con proteína:

| franja | plantillas | mediana g prot/100 kcal | ≥ 85 % del objetivo (4,98) |
|---|---|---|---|
| desayuno | 45 | **3,86** | **7** |
| almuerzo | 66 | 5,59 | 41 |
| cena | 54 | 5,74 | 35 |
| merienda | 40 | **3,37** | **6** |

Mangú, casabe, yuca y plátano: cocina real y correcta, y casi sin proteína. De ahí 20 platos
nuevos (10 desayunos, 10 meriendas) construidos SOBRE esa cocina —claras, atún, sardinas, arenque,
hígado, tilapia, queso cottage, jamón de pavo— y no en su contra.

## Los dos defectos que la ampliación destapó, que valen más que los platos

**1. La ventana se estrechaba sola.** El techo era `mejor + margen`: se mueve con el mejor
candidato, así que **añadir un plato bueno EXPULSA a otros**. Medido: el fondo pasó de 25 a 31
supervivientes por franja y los elegibles se quedaron en 5. *Un criterio relativo al rival no mide
al plato: mide la competencia.* La segunda puerta es ABSOLUTA y usa el piso clínico que el resto
del sistema ya usa. **Unión, jamás sustitución**: la puerta de proteína sola daba 3 elegibles en
merienda donde el score daba 6.

**2. Un guard que descartaba el conjunto en vez del elemento.** `verifica_comida` corría FUERA del
bucle de candidatos, así que un plato que se construía pero no pasaba el escáner tiraba el DÍA
ENTERO. Sólo se vio al ensanchar los elegibles: 14/14 días pasaron a **12/14 justo cuando había más
donde elegir**. *Un guard que castiga la abundancia está mal puesto.*

## Medido al cerrar (perfil vivo del dueño, 2.100 kcal, `gain_muscle`, presupuesto `low`)

```
platos distintos ....... 7 → 22 → 35 en 56 comidas
días armados ........... 14/14 · comidas sucias 0/56 · calorías −0,1 %
días bajo el piso ...... 4 (2 tras el cerrador) → 0
proteína mediana ....... 131 g sobre un objetivo de 123
coste del ciclo ........ RD$5.230 (suelo `low` RD$13.650)
determinismo ........... dos corridas, hash idéntico
```
"""
import json
import os

import pytest

import deterministic_day as dd

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REG = os.path.join(_BACKEND, "data", "registry")

#: Los 20 que entraron. Si alguien los borra, que sea a sabiendas de que vuelve el bucle de 22.
PLATOS_NUEVOS = (
    "Revoltillo de claras con espinaca y casabe",
    "Mangú de plátano verde con atún y cebolla encurtida",
    "Avena cocida con claras de huevo y maní",
    "Yogurt griego con avena tostada y maní",
    "Sardinas guisadas con casabe y tomate",
    "Tortilla de yuca con atún y queso blanco",
    "Hígado de res encebollado con plátano hervido",
    "Batida de leche con claras, avena y guineo",
    "Queso cottage con casabe, tomate y aguacate",
    "Tilapia al horno con yuca y cebolla",
    "Huevo duro con aguacate y sal mínima",
    "Atún en agua con casabe y cebolla",
    "Yogurt griego con guineo",
    "Queso cottage con lechosa",
    "Batida de leche con claras y avena",
    "Sardinas en lata con casabe",
    "Queso cottage con casabe y tomate",
    "Pechuga desmenuzada con casabe y cebolla",
    "Tofu salteado con salsa de soya y cebolla",
    "Queso de hoja con tomate y cebolla",
)


def _snapshot():
    p = os.path.join(_REG, "dish_registry_do_v1.json")
    if not os.path.exists(p):
        pytest.skip("sin snapshot DO en este entorno")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _por_nombre():
    return {t["name"]: t for t in (_snapshot().get("templates") or [])}


# ── El catálogo ──────────────────────────────────────────────────────────────
def test_los_20_platos_estan_compilados_y_servibles():
    idx = _por_nombre()
    faltan = [n for n in PLATOS_NUEVOS if n not in idx]
    assert not faltan, f"salieron del snapshot compilado: {faltan}"
    malos = [n for n in PLATOS_NUEVOS if idx[n].get("status") != "ok"]
    assert not malos, f"compilados con status != ok (nacen invisibles al selector): {malos}"


def test_cada_plato_nuevo_tiene_su_receta_congelada():
    """Sin receta, `construir_comida` devuelve `None` y el plato es decorativo."""
    p = os.path.join(_REG, "recipe_library_do_v1.json")
    if not os.path.exists(p):
        pytest.skip("sin biblioteca de recetas")
    with open(p, encoding="utf-8") as f:
        lib = json.load(f)
    idx = _por_nombre()
    sin = [n for n in PLATOS_NUEVOS
           if not ((lib.get("por_id") or {}).get(idx[n]["template_id"]) or {}).get("pasos")]
    assert not sin, f"platos sin pasos escritos: {sin}"


def test_ningun_constituyente_quedo_sin_resolver():
    """Un constituyente `excluded` es un ingrediente que la lista de compras no comprará."""
    idx = _por_nombre()
    sucios = {n: idx[n].get("excluded") for n in PLATOS_NUEVOS if idx[n].get("excluded")}
    assert not sucios, f"constituyentes sin resolver contra el catálogo: {sucios}"


def test_la_densidad_proteica_de_los_nuevos_cumple_su_razon_de_ser():
    """Entraron para cerrar un hueco de PROTEÍNA. Si uno baja del umbral, entró por otra cosa."""
    idx = _por_nombre()
    flojos = []
    for n in PLATOS_NUEVOS:
        np_ = idx[n].get("nutrition_per_serving") or {}
        kcal, prot = np_.get("kcal"), np_.get("protein_g")
        if not kcal or prot is None:
            flojos.append((n, "sin macros"))
            continue
        d = float(prot) / float(kcal) * 100
        if d < 5.0:
            flojos.append((n, round(d, 2)))
    assert not flojos, f"por debajo de 5,0 g de proteína/100 kcal: {flojos}"


def test_la_mediana_del_desayuno_y_la_merienda_subio():
    """La cifra que motivó el trabajo: 3,86 y 3,37 contra un objetivo de 5,86."""
    import statistics

    snap = _snapshot()
    for slot, antes in (("desayuno", 3.86), ("merienda", 3.37)):
        dens = []
        for t in snap["templates"]:
            if t.get("status") != "ok" or slot not in (t.get("slots") or []):
                continue
            np_ = t.get("nutrition_per_serving") or {}
            if np_.get("kcal") and np_.get("protein_g") is not None:
                dens.append(float(np_["protein_g"]) / float(np_["kcal"]) * 100)
        assert dens, slot
        assert statistics.median(dens) > antes, (
            f"la mediana de {slot} volvió a {statistics.median(dens):.2f}, por debajo del "
            f"{antes} de partida: se deshizo la ampliación")


# ── La segunda puerta del selector ───────────────────────────────────────────
def _cat_sintetico():
    return {
        "Proteico": {"kcal_per_100g": 100, "protein_g_per_100g": 20, "carbs_g_per_100g": 2,
                     "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
        "Casi": {"kcal_per_100g": 100, "protein_g_per_100g": 19, "carbs_g_per_100g": 3,
                 "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
        "Carbo": {"kcal_per_100g": 100, "protein_g_per_100g": 2, "carbs_g_per_100g": 22,
                  "fats_g_per_100g": 1, "fiber_g_per_100g": 0},
    }


def _tpl(tid, ing):
    return {"template_id": tid, "name": tid, "constituents": [{"name": ing, "grams": 100.0}]}


def test_el_piso_de_proteina_es_UNA_SEGUNDA_puerta_no_un_reemplazo():
    """La unión nunca puede devolver menos que el criterio de score solo.

    Sobre merienda, la puerta de proteína SOLA daba 3 elegibles donde el score daba 6: sustituir
    una por otra habría empeorado justo la franja más pobre.
    """
    cat = _cat_sintetico()
    por_id = {k: _tpl(k, v) for k, v in
              (("prot", "Proteico"), ("casi", "Casi"), ("carbo", "Carbo"))}
    obj = {"kcal": 100.0, "protein_g": 20.0, "carbs_g": 2.0, "fats_g": 1.0}
    tids = list(por_id)
    union = {t["template_id"] for t, _ in dd.elegir_plantillas(tids, obj, cat, por_id, "almuerzo")}
    assert "prot" in union and "casi" in union, (
        f"un plato que llega al piso clínico quedó fuera: {union}")


def test_anadir_un_plato_MEJOR_no_expulsa_a_los_que_ya_cumplian():
    """El defecto exacto: un techo `mejor + margen` se estrecha al mejorar el mejor."""
    cat = _cat_sintetico()
    cat["Perfecto"] = {"kcal_per_100g": 100, "protein_g_per_100g": 20.0, "carbs_g_per_100g": 2,
                       "fats_g_per_100g": 1, "fiber_g_per_100g": 0}
    obj = {"kcal": 100.0, "protein_g": 20.0, "carbs_g": 2.0, "fats_g": 1.0}

    sin = {"casi": _tpl("casi", "Casi"), "carbo": _tpl("carbo", "Carbo")}
    antes = {t["template_id"] for t, _ in dd.elegir_plantillas(list(sin), obj, cat, sin, "almuerzo")}

    con = dict(sin, perfecto=_tpl("perfecto", "Perfecto"))
    despues = {t["template_id"] for t, _ in dd.elegir_plantillas(list(con), obj, cat, con, "almuerzo")}

    assert antes <= despues, (
        f"añadir un plato mejor expulsó a {antes - despues}: la ventana volvió a estrecharse sola")


def test_el_piso_relativo_no_es_un_CUARTO_numero():
    """`PROTEIN_FLOOR_REL` tiene que valer lo mismo que el piso del resto del sistema. Escribir un
    número propio es la lección de `P1-DIET-CANON-SSOT` (tres tablas, drifearon, y la del filtro
    servía pollo a vegetarianas)."""
    import graph_orchestrator as go

    assert dd.PROTEIN_FLOOR_REL == pytest.approx(float(go.PROTEIN_FLOOR_HARD_PCT)), (
        f"el piso del selector ({dd.PROTEIN_FLOOR_REL}) se separó del clínico "
        f"({go.PROTEIN_FLOOR_HARD_PCT})")


# ── El guard que castigaba la abundancia ─────────────────────────────────────
def test_un_plato_sucio_cede_el_turno_en_vez_de_tirar_el_dia():
    """`verifica_comida` va DENTRO del bucle de candidatos.

    Estaba fuera y sólo se notó al ensanchar los elegibles: 14/14 días pasaron a 12/14 **justo
    cuando había más donde elegir**. Un guard que descarta el conjunto en vez del elemento castiga
    la abundancia.
    """
    import inspect

    src = inspect.getsource(dd.build_day_for_skeleton)
    cuerpo = src[src.index("for _t, _f in sorted("):]
    corte = cuerpo.index("if not comida:")
    assert "verifica_comida(" in cuerpo[:corte], (
        "la verificación volvió a correr FUERA del bucle: un plato sucio tira el día entero en vez "
        "de ceder el turno al siguiente candidato")
