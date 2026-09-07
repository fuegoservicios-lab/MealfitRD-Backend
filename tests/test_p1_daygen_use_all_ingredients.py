# -*- coding: utf-8 -*-
"""[P1-DAYGEN-USE-ALL-INGREDIENTS · 2026-09-07] La frase que ganó por ablación.

El 2026-09-07 se evaluó la arquitectura «Bioboros 3.0» (solver de cantidades + narrador sin
cifras). El brazo B hundió los huérfanos de V3 de 13 a 0 sobre 37 comidas y parecía mérito del
solver. El brazo C —**sólo la frase «usa TODOS los ingredientes», sin solver**— lo igualó o
superó en las tres columnas. El mérito era de la frase; la maquinaria no aportaba.

De ahí que esto sea un párrafo del prompt y no un roadmap. Estos tests anclan las cuatro cosas
que un refactor puede romper en silencio:

1. **El fichero.** `GENERATOR_SYSTEM_PROMPT` arma el esqueleto del plan; el que escribe recetas
   —ingredientes y pasos, donde nace el defecto— es `DAY_GENERATOR_SYSTEM_PROMPT`. Estuve a punto
   de editar el equivocado, y ahí la frase habría sido INERTE con el test en verde.
2. **Las dos direcciones.** V3 es «la lista compra algo que ningún paso toca»; V5 es su espejo,
   «un paso usa algo que la lista no compró». Media frase cierra medio defecto.
3. **Estático a import-time.** El append vive a nivel de módulo, como §17/§18/§19: el
   SystemMessage sale byte-idéntico en cada petición y el prompt-cache (P1-PROMPT-CACHE) queda
   intacto. Mover el knob a una lectura POR PETICIÓN cambiaría los bytes entre usuarios y mataría
   el cache sin que nada fallara.
4. **El knob apaga de verdad.** Un rollback que no revierte es peor que no tenerlo.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FUENTE = _BACKEND / "prompts" / "day_generator.py"
_KNOB = "MEALFIT_DAYGEN_USE_ALL_INGREDIENTS"


@pytest.fixture(scope="module")
def prompt() -> str:
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    return DAY_GENERATOR_SYSTEM_PROMPT


def _subproceso(valor: str | None) -> str:
    """Importa el módulo LIMPIO en otro proceso con el knob puesto.

    `importlib.reload` sobre un módulo de prompts es contaminación irreversible para el resto de
    la suite (lección de 2026-07-27): el subproceso es la única forma honesta de medir el efecto
    de un knob que se lee a import-time.
    """
    env = dict(os.environ)
    env.pop(_KNOB, None)
    if valor is not None:
        env[_KNOB] = valor
    codigo = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as P\n"
        "print('SI' if '20. USA TODOS LOS INGREDIENTES' in P else 'NO')\n" % str(_BACKEND)
    )
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                       env=env, cwd=str(_BACKEND), timeout=180)
    assert r.returncode == 0, f"el import falló: {r.stderr[-800:]}"
    return r.stdout.strip().splitlines()[-1]


# ---------------------------------------------------------------- 1. el fichero correcto

def test_la_seccion_vive_en_el_prompt_que_escribe_recetas(prompt):
    """El generador de DÍA es el que emite `ingredients` y `recipe`; el del plan, no."""
    assert "20. USA TODOS LOS INGREDIENTES" in prompt


def test_no_se_coló_en_el_prompt_del_esqueleto():
    """Si acaba en `plan_generator`, la frase no llega a ninguna receta y el test 1 sigue verde
    sólo si además se puso aquí — este test evita el 'puesta en los dos por si acaso'."""
    from prompts.plan_generator import GENERATOR_SYSTEM_PROMPT

    assert "20. USA TODOS LOS INGREDIENTES" not in GENERATOR_SYSTEM_PROMPT


def test_va_despues_del_19_y_no_duplica_numeral(prompt):
    assert prompt.index("\n19. ") < prompt.index("\n20. ")
    assert prompt.count("\n20. ") == 1


# ---------------------------------------------------------------- 2. las dos direcciones

def test_exige_que_cada_ingrediente_aparezca_en_un_paso(prompt):
    """La dirección de V3: la lista compra algo que ningún paso toca."""
    bloque = prompt[prompt.index("\n20. "):]
    assert "`ingredients` DEBE aparecer en al menos un paso" in bloque


def test_prohibe_pasos_que_pidan_lo_que_no_esta_en_la_lista(prompt):
    """La dirección de V5, el espejo. Sin ella la frase cierra medio defecto."""
    bloque = prompt[prompt.index("\n20. "):]
    assert "no esté en" in bloque and "`ingredients`" in bloque
    assert "el paso no puede pedirlo" in bloque


# ------------------------------------------------- 3. estático a import-time (prompt-cache)

def test_el_append_es_estatico_a_nivel_de_modulo():
    """El append NO puede vivir dentro de una función.

    Si alguien lo mueve a un builder por-petición el prompt deja de ser byte-idéntico entre
    usuarios, el prompt-cache muere y NADA falla: el plan se sigue generando, sólo que más caro.
    Por eso el ancla es estructural (AST) y no textual.
    """
    arbol = ast.parse(_FUENTE.read_text(encoding="utf-8"))
    de_modulo = []
    for nodo in arbol.body:                     # sólo el cuerpo del módulo, sin descender
        for sub in ast.walk(nodo):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and "20. USA TODOS LOS INGREDIENTES" in sub.value:
                de_modulo.append(type(nodo).__name__)
    assert de_modulo, "la §20 no aparece en el cuerpo del módulo: ¿se movió a una función?"
    assert all(n in ("If", "Assign", "AugAssign") for n in de_modulo), \
        f"la §20 se emite desde {de_modulo} — debe ser estática a import-time"


def test_el_knob_se_lee_una_sola_vez_y_no_por_peticion():
    """`_env_bool(MEALFIT_DAYGEN_...)` se lee una sola vez, fuera de toda función.

    Se cuentan literales del AST, no apariciones del texto: el nombre del knob también vive en el
    comentario de rollback de arriba, y contar texto crudo mediría la documentación en vez del
    código (lo aprendí fallando este mismo test).
    """
    arbol = ast.parse(_FUENTE.read_text(encoding="utf-8"))
    literales = [n for n in ast.walk(arbol)
                 if isinstance(n, ast.Constant) and n.value == _KNOB]
    assert len(literales) == 1, f"el knob se lee {len(literales)} veces"
    dentro_de_funcion = any(
        isinstance(sub, ast.Constant) and sub.value == _KNOB
        for nodo in ast.walk(arbol)
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))
        for sub in ast.walk(nodo)
    )
    assert not dentro_de_funcion, "el knob se lee por petición: mata el prompt-cache"


# ---------------------------------------------------------------- 4. el rollback funciona

def test_el_default_es_encendido():
    assert _subproceso(None) == "SI"


@pytest.mark.parametrize("apagado", ["0", "false", "off", "no"])
def test_el_knob_lo_apaga_de_verdad(apagado):
    """Un rollback que no revierte es peor que no tenerlo: el operador cree que apagó algo."""
    assert _subproceso(apagado) == "NO"


def test_el_knob_queda_registrado_en_el_snapshot():
    """Los knobs se auto-registran vía `_env_bool` (P3-NEW-D); si no aparece, el operador no
    puede descubrirlo sin leer el código."""
    import prompts.day_generator  # noqa: F401  — fuerza el import que registra el knob
    from graph_orchestrator import get_knobs_registry_snapshot

    assert _KNOB in {str(k) for k in get_knobs_registry_snapshot()}
