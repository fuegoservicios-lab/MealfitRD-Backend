# -*- coding: utf-8 -*-
"""[P1-DAYGEN-DONENESS-INSIDE · 2026-09-07] La señal de que está listo POR DENTRO.

El dueño lo pidió 8 veces en las 37 notas del duelo y 9 en las 20 de la biblioteca, siempre igual:
«no usar el dorado como criterio final», «comprobar cocción interior». Mi primera lectura fue
«falta temperatura y tiempo» — **y la flota dijo que no**:

    comidas vivas con cocción ............ 801 de 1.194
      al horno SIN temperatura ...........   1  (0,1 %)   ← NO era el hueco
      sin señal de cocción INTERIOR ....... 571 (71,3 %)  ← este sí

La temperatura ya la escribe producción; faltaba en el experimento de la biblioteca porque *ese*
prompt prohibía toda cifra. **Medir cambió qué implementar**, no solo si implementarlo.

El 71,3 % bruto tampoco es el objetivo: a un sofrito de cebolla no le falta nada. Acotado a donde
importa, y por eso la §21 nombra tres familias y **excluye explícitamente el salteado**:

    A · masa formada cocida en seco, sin señal interior ... 142 (11,9 %)
    B · pollo/cerdo/pavo en seco, sin señal interior ......  38 ( 3,2 %)

Evidencia de que el modelo obedece cuando se le pide: en la biblioteca pasó de 0/20 recetas con
parámetros a 9/20, y los dos platos que el dueño había rechazado por cocción quedaron correctos.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FUENTE = _BACKEND / "prompts" / "day_generator.py"
_KNOB = "MEALFIT_DAYGEN_DONENESS_INSIDE"
_TITULO = "21. CÓMO SE SABE QUE ESTÁ LISTO POR DENTRO"


@pytest.fixture(scope="module")
def prompt() -> str:
    from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT

    return DAY_GENERATOR_SYSTEM_PROMPT


@pytest.fixture(scope="module")
def bloque(prompt) -> str:
    return prompt[prompt.index("\n21. "):]


def _subproceso(valor: str | None) -> str:
    """Importa el módulo LIMPIO en otro proceso con el knob puesto.

    `importlib.reload` sobre un módulo de prompts contamina al resto de la suite de forma
    irreversible (lección de 2026-07-27): el subproceso es la única medición honesta de un knob
    que se lee a import-time.
    """
    env = dict(os.environ)
    env.pop(_KNOB, None)
    if valor is not None:
        env[_KNOB] = valor
    codigo = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT as P\n"
        "print('SI' if %r in P else 'NO')\n" % (str(_BACKEND), _TITULO)
    )
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                       env=env, cwd=str(_BACKEND), timeout=180)
    assert r.returncode == 0, f"el import falló: {r.stderr[-800:]}"
    return r.stdout.strip().splitlines()[-1]


# ---------------------------------------------------------------- el fichero correcto

def test_vive_en_el_prompt_que_escribe_recetas(prompt):
    assert _TITULO in prompt


def test_no_se_colo_en_el_prompt_del_esqueleto():
    """`GENERATOR_SYSTEM_PROMPT` arma el esqueleto y no escribe pasos: ahí sería inerte."""
    from prompts.plan_generator import GENERATOR_SYSTEM_PROMPT

    assert _TITULO not in GENERATOR_SYSTEM_PROMPT


def test_va_despues_del_20_y_no_duplica_numeral(prompt):
    assert prompt.index("\n20. ") < prompt.index("\n21. ")
    assert prompt.count("\n21. ") == 1


# ---------------------------------------------------------------- las tres familias medidas

def test_nombra_la_masa_formada(bloque):
    """La familia A: 142 comidas vivas. Aquí el dorado engaña — fuera antes que el centro."""
    for pieza in ("croquetas", "tortitas", "bollitos", "arepitas", "panqueques"):
        assert pieza in bloque, f"falta «{pieza}» entre los ejemplos de masa formada"
    assert "palillo" in bloque, "falta la señal concreta (el palillo)"


def test_la_carne_se_comprueba_por_TEMPERATURA_no_por_color(bloque):
    """[P0-DONENESS-TEMPERATURA · 2026-09-08] Este test pedía la palabra «rosadas» — o sea, exigía
    el criterio EQUIVOCADO, y lo hacía mientras la frase de al lado decía «es seguridad».

    El color no es un criterio de seguridad: la carne se dora antes de llegar a temperatura segura y
    puede seguir rosada después. Lo cazó el juicio a ciegas del dueño sobre almuerzos y cenas —
    **13 de sus 16 «dudoso» pedían exactamente esto**— y los desayunos habían sacado 17/20 sólo
    porque casi no llevan carne.

    *Un test puede anclar una regla equivocada con la misma firmeza que una correcta; lo que lo
    delató no fue el test, fue un humano probando el producto.*
    """
    assert "74 °C" in bloque, "falta la temperatura del ave, que es la que importa"
    assert "71 °C" in bloque, "falta la de la carne molida"
    assert "seguridad" in bloque
    assert "COLOR NO SIRVE" in bloque, (
        "volvió a admitirse el color como criterio: es lo que este P-fix cerró")


def test_exige_hervir_los_viveres_antes_de_majar(bloque):
    """El rechazo textual del dueño: «equipara yuca cruda rallada con puré sin indicar cocción»."""
    assert "yuca" in bloque and "Rallar no es cocinar" in bloque


def test_excluye_el_salteado_a_proposito(bloque):
    """Sin esta salvedad la regla se dispara donde no hay centro crudo.

    El 71,3 % bruto de comidas sin señal interior incluye sofritos, donde no falta nada: pedir ahí
    una comprobación de centro es ruido, y el ruido es lo que hace que una regla se ignore entera.
    """
    assert "sofrito" in bloque or "salteado" in bloque


def test_no_prohibe_el_dorado_como_acompanante(bloque):
    """«hasta dorar» sigue siendo válido junto a la señal real.

    Prohibirlo del todo chocaría con el resto del prompt, que lo usa para describir la técnica.
    """
    assert "puede acompañar" in bloque


# ------------------------------------------- estático a import-time (prompt-cache)

def test_el_append_es_estatico_a_nivel_de_modulo():
    """Moverlo a un builder por petición mata el prompt-cache sin que NADA falle: el plan se
    seguiría generando, solo que más caro. Por eso el ancla es estructural, no textual."""
    arbol = ast.parse(_FUENTE.read_text(encoding="utf-8"))
    de_modulo = []
    for nodo in arbol.body:
        for sub in ast.walk(nodo):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and _TITULO in sub.value:
                de_modulo.append(type(nodo).__name__)
    assert de_modulo, "la §21 no está en el cuerpo del módulo: ¿se movió a una función?"
    assert all(n in ("If", "Assign", "AugAssign") for n in de_modulo), \
        f"la §21 se emite desde {de_modulo} — debe ser estática a import-time"


def test_el_knob_se_lee_una_sola_vez_y_no_por_peticion():
    """Se cuentan literales del AST, no apariciones del texto: el nombre del knob también vive en
    el comentario de rollback, y contar texto crudo mediría la documentación."""
    arbol = ast.parse(_FUENTE.read_text(encoding="utf-8"))
    literales = [n for n in ast.walk(arbol)
                 if isinstance(n, ast.Constant) and n.value == _KNOB]
    assert len(literales) == 1, f"el knob se lee {len(literales)} veces"
    dentro = any(
        isinstance(sub, ast.Constant) and sub.value == _KNOB
        for nodo in ast.walk(arbol)
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))
        for sub in ast.walk(nodo))
    assert not dentro, "el knob se lee por petición: mata el prompt-cache"


# ---------------------------------------------------------------- el rollback funciona

def test_el_default_es_encendido():
    assert _subproceso(None) == "SI"


@pytest.mark.parametrize("apagado", ["0", "false", "off"])
def test_el_knob_lo_apaga_de_verdad(apagado):
    assert _subproceso(apagado) == "NO"


def test_convive_con_la_seccion_20(prompt):
    """§20 y §21 son independientes: apagar una no puede llevarse la otra por delante."""
    assert "20. USA TODOS LOS INGREDIENTES" in prompt and _TITULO in prompt
