# -*- coding: utf-8 -*-
"""[P1-DETERMINISTIC-DAY-WIRED · 2026-09-08] El enganche al pipeline, anclado aparte.

`test_p1_deterministic_day.py` prueba que el ensamblador FUNCIONA. Este prueba que está
ENCHUFADO, y son dos contratos distintos: hoy mismo encontramos `recipe_for_dish_name` con
cero call sites — una función correcta, probada y desplegada que no servía a nadie. Separarlos
hace que borrar el enganche falle el test del enganche, no el del algoritmo.
"""
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def test_el_ensamblador_esta_enchufado_al_generador_de_dias():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "from deterministic_day import build_day_for_skeleton as _det_day" in src, (
        "sin el import, el ensamblador vuelve a ser una feature inerte")
    assert ("_det_day(nutrition, form_data, skel_day, day_num) or "
            "await _generate_day_hedged") in src, (
        "el determinista va PRIMERO y el LLM es el `or`. Invertirlo deja el día determinista "
        "inalcanzable — inerte con toda la apariencia de estar enchufado")


def test_el_enganche_no_engorda_el_god_file():
    """`graph_orchestrator.py` tiene techo duro y superarlo NO se arregla subiendo el número: se
    arregla extrayendo. Este enganche cabe en 0 líneas netas — import colgado de una línea que ya
    existía y una línea sustituida por otra."""
    n = len((_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").splitlines())
    assert n <= 53100, f"el god-file subió a {n} líneas: extraer, no subir el cap"


def test_el_modulo_no_importa_el_grafo():
    """`deterministic_day` no puede depender de `graph_orchestrator`: sería un ciclo, y además lo
    haría imposible de probar sin montar el grafo entero.

    Se mira el AST, no el texto. Las menciones narrativas en comentarios son historia legítima —
    el módulo explica en su docstring que un día que sale de ahí bypasea `assemble_plan_node`, y
    eso es exactamente lo que el lector necesita saber. Es la misma distinción que el blanket
    anti-Gemini ya hace entre usar un API y contar por qué no se usa.
    """
    import ast

    arbol = ast.parse((_BACKEND / "deterministic_day.py").read_text(encoding="utf-8"))
    nivel_modulo = set()
    for n in arbol.body:                       # SOLO el nivel de módulo, no `ast.walk`
        if isinstance(n, ast.Import):
            nivel_modulo.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            nivel_modulo.add(n.module.split(".")[0])
    assert "graph_orchestrator" not in nivel_modulo, (
        f"importa el grafo A NIVEL DE MÓDULO: eso es el ciclo. Importa: {sorted(nivel_modulo)}")


def test_el_backstop_clinico_se_invoca_de_verdad():
    """[P1-DETERMINISTIC-DAY-BACKSTOP] Mi propio docstring prometía «`build_day` re-verifica antes
    de devolver» y el código NO verificaba nada — la misma clase de defecto que este repo lleva el
    día entero produciendo: una afirmación escrita que el código no honra.

    Y no es cosmético. `P0-DEGRADED-SAFETY-SCAN` existe porque un día armado sin LLM bypasea
    `assemble_plan_node`: ni reviewer médico, ni scans de alérgeno y dieta. El día determinista es
    exactamente esa clase de superficie.
    """
    src = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    assert "def verifica_comida" in src
    assert "clinical_backstop_for_meal" in src, (
        "sin el backstop clínico, un alérgeno que se cuele por el filtro de candidatos llega al "
        "plato: es la lección literal de P0-DEGRADED-SAFETY-SCAN")
    assert "_viol = verifica_comida(" in src, (
        "la función existe pero nadie la llama — que es justo lo que hoy encontramos en "
        "`recipe_for_dish_name`: correcta, probada y con cero call sites")
    assert "return None" in src.split("_viol = verifica_comida(")[1][:1200], (
        "una violación tiene que RECHAZAR el día y caer al LLM, no sólo registrarse")


def test_el_rechazo_dice_el_MOTIVO_con_las_dos_formas():
    """[P1-DETERMINISTIC-DAY-BACKSTOP] Las dos capas hablan idiomas distintos: el escáner culinario
    devuelve dicts con `check`, el backstop clínico devuelve STRINGS legibles
    («alérgeno 'huevo' en el ingrediente '200 g de Huevo'»).

    La primera versión hacía `.get()` sobre ambos. Sobre la cadena revienta, la excepción del
    `try` exterior se traga el aviso, y el operador se queda sin saber POR QUÉ se rechazó el día —
    el mismo modo de fallo que `P2-ALERT-MESSAGE-REFRESH` cerró esta misma mañana: un diagnóstico
    que no llega a quien investiga.
    """
    src = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    bloque = src.split("_viol = verifica_comida(")[1][:900]
    assert "isinstance(v, dict)" in bloque, (
        "el log asume una sola forma: con la otra revienta y el motivo se pierde")
    assert "else str(v)" in bloque
    assert "logger.warning" in bloque, "un rechazo silencioso es indistinguible de que no pase nada"


def test_los_filtros_clinicos_VIAJAN_al_selector():
    """[P1-DETERMINISTIC-DAY-BACKSTOP] El docstring decía que «el CandidateSet ya filtra por
    alergia y dieta». Es cierto de `_registry_slice` y **falso de esta función**, que llama a
    `template_candidates` directamente y no le pasaba nada.

    Lo cazó el backstop clínico rechazando un desayuno con huevo a un alérgico al huevo. La defensa
    en profundidad funcionó — y por eso mismo había que cerrar el hueco de arriba: *una última
    línea de defensa que trabaja sola dejó de ser defensa en profundidad*.

    Medido tras el arreglo: 14/14 días para perfil normal, alérgico a huevo y vegetariana, sin que
    el backstop rechace ninguno.
    """
    src = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    bloque = src.split("dr.template_candidates(")[1][:400]
    assert "exclude_allergens=" in bloque, (
        "el selector elige candidatos sin saber las alergias: el backstop acaba siendo la ÚNICA "
        "defensa, que es justo lo que la defensa en profundidad no debe ser")
    assert "diet=" in bloque, "lo mismo con la dieta: servía pollo a vegetarianas en P1-DIET-CANON-SSOT"
