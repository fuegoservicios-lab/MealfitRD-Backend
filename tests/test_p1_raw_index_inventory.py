# -*- coding: utf-8 -*-
"""[P1-RAW-INDEX-INVENTORY · 2026-09-07] Inventario congelado de las escrituras a `ingredients_raw`.

Escribir `raw[i]` cruzando `ingredients` con `ingredients_raw` **por índice** es la clase de fallo
que este repo lleva persiguiendo desde julio: el reconciliador reconstruye `raw` como
`[conservadas] + [añadidas]` —preserva el largo y cambia el orden— así que «mismo largo» nunca fue
«mismo alimento». Cuando falla, el usuario COMPRA otra cosa:

    humanizador  ·  «corta 115 g de lechosa»   →  «corta 25 g de avena (115 g)»
    piso         ·  sube el aguacate           →  la papa de ½ a 1,5   (×3)
    lácteo       ·  «2 huevos»                 →  «30 huevos»

**Por qué existe este test y no bastaban los P-fixes.** `P2-RAW-PAIR-BY-FOOD` (29-jul) migró tres
superficies. `P1-DM-RAW-BY-FOOD` (31-jul) migró una más y se llamó *«el último que quedaba fuera»*.
No lo era: el 2026-09-07 aparecieron cuatro sitios más con daño medido, en tres ficheros distintos.
Nada vigilaba que no naciera el siguiente.

**Y por qué el inventario sale del AST.** Ese mismo día conté los sitios cuatro veces con grep —8,
12, 28— y las tres primeras estaban mal: un grep se pierde continuaciones de línea, cuenta
comentarios y depende de qué patrón se te ocurra buscar. El AST no opina. *Buscar el síntoma que
conoces encuentra sólo los casos que ya conocías.*

**Qué hacer si este test falla.** No subas el número sin más. Una escritura nueva a `raw[i]` debe
resolver la línea por ALIMENTO —`_raw_idx_for_display`, `_rescale_raw_by_food` o
`_remove_one_raw_line_by_food`, según sea reescritura, escalado o borrado— y sólo entonces se
actualiza el inventario. Si de verdad puede ir por índice (porque corre antes de los
reconciliadores que appendean), dilo en la columna `veredicto` con la traza que lo demuestra.
"""
import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

# Foto AST del 2026-09-07 tras cerrar los cuatro sitios con daño medido.
# {fichero: {función: (nº de escrituras a raw[...], veredicto)}}
#
# «resuelto»  la función pasa por el resolvedor por alimento.
# «antes»     corre ANTES de los reconciliadores que appendean → el índice es válido ahí (trazado).
# «0 medido»  expuesta, medida sobre la flota con 0 comidas dañadas — y CON QUÉ SONDA:
#             una que sobreescribe la línea entera sólo puede fallar RUIDOSAMENTE (escribe sobre
#             otro alimento) y para eso la sonda del 07-sep servía. Una que escribe bajo `sub` o
#             `rescale` falla en SILENCIO (no arregla la línea que tocaba) y ahí era ciega: así se
#             coló `_apply_budget_driver_aware_pass` con un 0 falso hasta el 08-sep.
#             Y el razonamiento «sobreescribe entera ⇒ la sonda ruidosa aplica» tampoco basta si
#             esa sonda no se verificó: `_cap_daily_whole_eggs` llevaba ese sello y escribía
#             «3 claras de huevo» sobre «Sal al gusto» (P1-EGGCAP-RAW-BY-FOOD). Un veredicto vale
#             lo que valga el instrumento, aunque el razonamiento sea bueno.
# «no dispara» ni una activación sobre las 1.172 comidas vivas. NO es «escritura segura»: es que
#             nunca corre. Un detector que no dispara nunca sale perfecto.
_INVENTARIO = {
    "graph_orchestrator.py": {
        "_remove_one_raw_line_by_food":            (1, "resuelto"),
        "_sync_one_raw_line":                      (1, "resuelto"),
        "_cap_unrealistic_portions":               (3, "resuelto (2 migrados + fallback documentado)"),
        "_floor_subservible_portions":             (9, "resuelto"),
        "_repair_name_phantom_dairy":              (1, "resuelto"),
        "_scale_congruent_protein_line":           (1, "antes de los appenders"),
        "_try_scale_existing_protein":             (1, "antes de los appenders"),
        "_ensure_ingredient_quantities":           (1, "antes de los appenders"),
        "_swap_fat_dense_protein_to_lean_for_day": (1, "antes de los appenders"),
        "_consolidate_duplicate_gram_lines":       (4, "antes de los appenders"),
        "_day_sodium_autofix":                     (1, "0 medido 07-sep · sobreescribe entera (sonda ruidosa aplica)"),
        "_cap_cheese_dumps_final":                 (1, "0 activaciones 08-sep · no dispara"),
        "_single_trip_fresh_substitute":           (1, "0 medido 07-sep · sobreescribe entera (sonda ruidosa aplica)"),
        "_cap_daily_whole_eggs":                   (2, "resuelto"),
        "_apply_budget_cheapen_pass":              (1, "resuelto"),
        "_apply_budget_driver_aware_pass":         (1, "resuelto"),
        "_baking_powder_cap_pass":                 (1, "0 activaciones 08-sep · no dispara"),
        # Trazada tras publicar el inventario: corre en `finalize_plan_data_coherence:29061`,
        # DESPUÉS de los appenders → expuesta por posición. Pero su escritura pasa por
        # `_EGG_ONE_LINE_RX.sub`, y su condición («el otro huevo» + exactamente una línea de
        # huevo con lead 1) no se cumple en NINGUNA de las 1.172 comidas vivas: 0 activaciones.
        "_egg_count_step_sync":                    (1, "0 medido 07-sep · expuesta por posición"),
    },
    "portion_solver.py": {
        "refine_day_portions_integer":             (1, "resuelto"),
    },
    # [P1-NIGHTRICE-RAW-BY-FOOD · 2026-09-08] Este ratchet cazó su propio helper el día después de
    # nacer: escribe `raw[_ri]`, pero `_ri` sale de `_raw_idx_for_display` — por ALIMENTO, y sin
    # pareja no toca nada. Es exactamente la forma que el inventario existe para distinguir.
    "constants.py": {
        "sustituye_display_y_raw":                 (1, "resuelto"),
    },
}

_NOMBRES = {"raw", "_raw"}


def _es_subindice_de_raw(nodo) -> bool:
    return (isinstance(nodo, ast.Subscript) and isinstance(nodo.value, ast.Name)
            and nodo.value.id in _NOMBRES)


def _escrituras_por_funcion(path: Path) -> dict:
    """Cuenta con AST: asignación, aumento, `del` y `.pop()` sobre `raw[...]`.

    Con AST y no con regex a propósito: las cuentas por grep del 07-sep salieron mal tres veces
    seguidas — continuaciones de línea, comentarios, y el patrón que no se te ocurre buscar.
    """
    arbol = ast.parse(path.read_text(encoding="utf-8"))
    fuera = {}
    for top in arbol.body:
        if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        n = 0
        for nodo in ast.walk(top):
            if isinstance(nodo, ast.Assign) and any(_es_subindice_de_raw(t) for t in nodo.targets):
                n += 1
            elif isinstance(nodo, ast.AugAssign) and _es_subindice_de_raw(nodo.target):
                n += 1
            elif isinstance(nodo, ast.Delete) and any(_es_subindice_de_raw(t) for t in nodo.targets):
                n += 1
            elif (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
                  and nodo.func.attr == "pop" and isinstance(nodo.func.value, ast.Name)
                  and nodo.func.value.id in _NOMBRES):
                n += 1
        if n:
            fuera[top.name] = n
    return fuera


@pytest.mark.parametrize("fichero", sorted(_INVENTARIO))
def test_ninguna_funcion_NUEVA_escribe_raw_por_indice(fichero):
    """El ratchet: una función que no estaba en el inventario no puede empezar a escribir raw.

    Es el hueco por el que esta clase sobrevivió desde julio — cada P-fix arreglaba los sitios que
    conocía y nada impedía que naciera el siguiente.
    """
    real = _escrituras_por_funcion(_BACKEND / fichero)
    nuevas = sorted(set(real) - set(_INVENTARIO[fichero]))
    assert not nuevas, (
        f"{fichero}: funciones NUEVAS que escriben `raw[i]`: {nuevas}.\n"
        "Resuelve la línea por ALIMENTO (`_raw_idx_for_display` para reescribir, "
        "`_rescale_raw_by_food` para escalar, `_remove_one_raw_line_by_food` para borrar) y "
        "añádela al inventario con su veredicto. Si de verdad puede ir por índice, deja la traza "
        "que lo demuestra.")


@pytest.mark.parametrize("fichero", sorted(_INVENTARIO))
def test_el_conteo_por_funcion_no_crece_en_silencio(fichero):
    """Una escritura MÁS dentro de una función ya inventariada también cuenta.

    `_cap_unrealistic_portions` tenía sitios migrados y sin migrar a la vez: alguien arregló uno en
    julio y dejó dos al lado. Sin contar por función, eso vuelve a pasar sin que nadie lo vea.
    """
    real = _escrituras_por_funcion(_BACKEND / fichero)
    difs = []
    for fn, (esperado, veredicto) in _INVENTARIO[fichero].items():
        visto = real.get(fn, 0)
        if visto != esperado:
            difs.append(f"{fn}: {visto} escrituras, el inventario dice {esperado} ({veredicto})")
    assert not difs, f"{fichero}:\n  " + "\n  ".join(difs)


def test_el_inventario_declara_un_veredicto_por_funcion():
    """Cada fila dice POR QUÉ es aceptable. Un inventario sin razones es una lista de deuda."""
    # «SIN TRAZAR» sigue siendo válido a propósito: quien añada una fila sin haber trazado
    # su cadena de llamada debe poder decirlo, en vez de inventarse un veredicto cómodo.
    # [P1-DRIVER-RAW-BY-FOOD · 2026-09-08] «0 activaciones» es un veredicto DISTINTO de «0 medido»:
    # el primero dice que la función no llegó a correr ni una vez, el segundo que corrió y no dañó.
    # Colapsarlos deja creer que una escritura está probada cuando nadie la ha ejercido nunca.
    validos = ("resuelto", "antes de los appenders", "0 medido", "0 activaciones", "SIN TRAZAR")
    for fichero, filas in _INVENTARIO.items():
        for fn, (_, veredicto) in filas.items():
            assert any(veredicto.startswith(v) for v in validos), \
                f"{fichero}:{fn} tiene un veredicto no reconocido: {veredicto!r}"


def test_el_resolvedor_canonico_sigue_existiendo():
    """Si desaparece, el inventario de arriba pierde su salida y hay que rehacer el contrato."""
    import graph_orchestrator as go

    for nombre in ("_raw_idx_for_display", "_rescale_raw_by_food",
                   "_remove_one_raw_line_by_food", "_raw_display_parallel_by_food"):
        assert callable(getattr(go, nombre, None)), f"falta {nombre}"


# ─────────────────────────────────────────────────────────────────────────────
# [P1-RAW-INDEX-INVENTORY-FICHEROS · 2026-09-08] El ratchet vigilaba DOS ficheros
#
# El inventario de arriba congela `graph_orchestrator.py` y `portion_solver.py`, y sus dos guards
# preguntan «¿nació una función nueva AQUÍ?» y «¿creció el conteo AQUÍ?». Ninguno preguntaba
# «¿empezó OTRO fichero a hacerlo?» — que es exactamente la forma en que esta clase sobrevivió desde
# julio: cada P-fix cerraba los sitios que conocía y nada impedía que el siguiente naciera al lado.
#
# Escanear los 119 ficheros de producción con el detector desnudo da un falso positivo:
# `cultural_benchmark._profile_lexicons` tiene un `raw: dict[str, set[str]]` de léxicos, sin ninguna
# relación con `ingredients_raw`. Por eso el guard exige que la función TAMBIÉN mencione
# `ingredients_raw`.
#
# Discriminador verificado antes de adoptarlo: no pierde NINGUNA de las 32 escrituras ya
# inventariadas, y descarta el falso positivo. *Un detector que no se prueba contra los casos que ya
# conoces no sabes si detecta o si reparte.*


def _menciona_ingredients_raw(path: Path, nombre_funcion: str) -> bool:
    try:
        arbol = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for top in arbol.body:
        if getattr(top, "name", None) == nombre_funcion:
            try:
                return "ingredients_raw" in ast.unparse(top)
            except Exception:
                return False
    return False


def test_ningun_fichero_NUEVO_empieza_a_escribir_raw_por_indice():
    """El tercer guard: la clase no puede mudarse a un fichero que el inventario no mira.

    Si esto falla, decide una de dos y hazlo explícito:
      · la escritura resuelve por ALIMENTO (`_raw_idx_for_display` y hermanos) → arréglala; o
      · el fichero entra al `_INVENTARIO` con su veredicto y su traza.
    Subir el número no es una opción aquí: no hay número que subir.
    """
    fuera = {}
    for path in sorted(_BACKEND.rglob("*.py")):
        rel = path.relative_to(_BACKEND).as_posix()
        if rel.startswith(("tests/", "scripts/", "migrations/")) or "site-packages" in rel:
            continue
        if rel in _INVENTARIO:
            continue
        funcs = {fn: n for fn, n in _escrituras_por_funcion(path).items()
                 if _menciona_ingredients_raw(path, fn)}
        if funcs:
            fuera[rel] = funcs
    assert not fuera, (
        "ficheros que empezaron a escribir `raw[i]` y el inventario no vigila:\n  "
        + "\n  ".join(f"{f}: {d}" for f, d in fuera.items()))
