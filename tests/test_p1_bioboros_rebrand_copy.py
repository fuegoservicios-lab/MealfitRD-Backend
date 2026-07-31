"""[P1-BIOBOROS-REBRAND-COPY · 2026-07-30] La marca vieja no puede volver al copy.

El producto pasó de MealfitRD a Bioboros y de `mealfitrd.com` a `bioboros.com`.
El rebrand visual se hizo, pero sobrevivieron cadenas que SÍ llegan al usuario:

  - `prompts/help_bot.py` mandaba a `mealfitrd.com/supermercado` y
    `mealfitrd.com/medical` — enlaces a un dominio MUERTO, dictados por el bot
    de ayuda a cualquiera que preguntase.
  - Tres notificaciones push decían "Abre Mealfit".
  - El prompt del coach hablaba del "diario de Mealfit".
  - El CTA de una push decía "Abrir Mealfit".

Ninguna auditoría las vio, y el patrón es siempre el mismo: son cadenas que se
escriben una vez y nadie vuelve a leer. Su única prueba de corrección era que
alguien pasara por encima. Este test es ese alguien.

## Qué se prohíbe y qué NO

Se escanean SÓLO los literales de cadena (vía AST), porque son lo que llega al
usuario. Quedan fuera a propósito:

  - **Comentarios y docstrings**: son el registro histórico. Reescribir "antes
    decía Mealfit V1.0" para que diga "Bioboros" convierte una nota verdadera
    en una mentira. Un replace masivo ya hizo justo eso una vez en esta base.
  - **`MEALFIT_*`** (knobs) y **`mealfit_*`** (claves de localStorage / KV):
    son IDENTIFICADORES, no marca. Renombrarlos rompería el `.env` del VPS
    (~676 knobs) y borraría el estado local de los usuarios (plan en curso,
    formulario a medias, sesión). El patrón es sensible a mayúsculas
    precisamente para dejarlos pasar: ni `MEALFIT_X` ni `mealfit_x` contienen
    `Mealfit` ni `mealfitrd`.

Si alguna vez hace falta una cadena con la marca vieja (una migración de datos
que compara contra valores históricos, por ejemplo), añade el marker
`# [P1-BIOBOROS-REBRAND-COPY EXEMPT: <razón>]` en las 3 líneas previas.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

BACKEND = pathlib.Path(__file__).resolve().parent.parent

# Sensible a mayúsculas A PROPÓSITO: deja pasar `MEALFIT_*` y `mealfit_*`.
MARCA_VIEJA = re.compile(r"Mealfit|mealfitrd")

EXEMPT = "P1-BIOBOROS-REBRAND-COPY EXEMPT:"

# Directorios que NO son código de producción.
EXCLUIDOS = {"tests", "scratch", "venv", "venv-test", "test_venv", ".venv",
             "migrations", "node_modules", "__pycache__", "docs"}


def ficheros_prod() -> list[pathlib.Path]:
    out = []
    for p in BACKEND.rglob("*.py"):
        if any(parte in EXCLUIDOS for parte in p.relative_to(BACKEND).parts):
            continue
        out.append(p)
    return sorted(out)


def _ids_de_docstring(arbol: ast.AST) -> set[int]:
    """`id()` de los Constant que son docstring (Module/Class/Function).

    Un docstring es sintácticamente indistinguible de un literal suelto, así
    que hay que identificarlos por POSICIÓN: primer statement del cuerpo.
    """
    ids = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        cuerpo = getattr(nodo, "body", None)
        if not cuerpo:
            continue
        primero = cuerpo[0]
        if (isinstance(primero, ast.Expr)
                and isinstance(primero.value, ast.Constant)
                and isinstance(primero.value.value, str)):
            ids.add(id(primero.value))
    return ids


def violaciones(path: pathlib.Path) -> list[tuple[int, str]]:
    fuente = path.read_text(encoding="utf-8", errors="ignore")
    lineas = fuente.splitlines()
    try:
        arbol = ast.parse(fuente)
    except SyntaxError:  # pragma: no cover - los .py de prod compilan
        return []

    docstrings = _ids_de_docstring(arbol)
    fuera = []
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)):
            continue
        if id(nodo) in docstrings:
            continue
        m = MARCA_VIEJA.search(nodo.value)
        if not m:
            continue
        # Exención declarada en las 3 líneas previas.
        ini = max(0, nodo.lineno - 4)
        if any(EXEMPT in l for l in lineas[ini:nodo.lineno]):
            continue
        fuera.append((nodo.lineno, m.group(0)))
    return fuera


def test_ningun_literal_de_prod_lleva_la_marca_vieja():
    ficheros = ficheros_prod()
    # Sanity del vehículo: si el glob se rompe, el test pasaría en vacío.
    assert len(ficheros) > 20, f"solo {len(ficheros)} ficheros escaneados"

    fallos = []
    for p in ficheros:
        for linea, hallado in violaciones(p):
            fallos.append(f"  {p.relative_to(BACKEND)}:{linea} -> {hallado!r}")

    assert not fallos, (
        "Cadenas con la marca vieja en literales de producción "
        f"({len(fallos)}):\n" + "\n".join(fallos)
        + "\n\nSon texto que llega al usuario. Cámbialas a Bioboros / "
          "bioboros.com, o declara el marker "
          f"`# [{EXEMPT} <razón>]` en las 3 líneas previas."
    )


def test_el_escaner_detecta_una_marca_plantada():
    """El guard falla cuando debe — si no, el test de arriba no prueba nada."""
    tmp = BACKEND / "tests" / "_tmp_rebrand_probe.py"
    tmp.write_text('URL = "visita mealfitrd.com hoy"\n', encoding="utf-8")
    try:
        assert violaciones(tmp) == [(1, "mealfitrd")]
    finally:
        tmp.unlink()


def test_el_escaner_deja_pasar_knobs_y_claves_locales():
    """`MEALFIT_*` y `mealfit_*` son identificadores, no marca."""
    tmp = BACKEND / "tests" / "_tmp_rebrand_probe2.py"
    tmp.write_text(
        'K = "MEALFIT_SHOPPING_COHERENCE_GUARD"\n'
        'L = "mealfit_plan_in_progress"\n',
        encoding="utf-8")
    try:
        assert violaciones(tmp) == []
    finally:
        tmp.unlink()


def test_el_escaner_ignora_comentarios_y_docstrings():
    """El registro histórico se conserva: reescribirlo lo vuelve falso."""
    tmp = BACKEND / "tests" / "_tmp_rebrand_probe3.py"
    tmp.write_text(
        '"""Modulo que antes se llamaba MealfitRD."""\n'
        '# historico: antes decia Mealfit V1.0\n'
        'X = 1\n',
        encoding="utf-8")
    try:
        assert violaciones(tmp) == []
    finally:
        tmp.unlink()


# --------------------------------------------------------------------------
# Anclas sobre el fichero que estaba mal. El blanket de arriba cubre la clase;
# esto fija los hechos concretos que el bot de ayuda dicta a los usuarios.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prompt_help_bot() -> str:
    import sys
    sys.path.insert(0, str(BACKEND))
    from prompts.help_bot import HELP_BOT_SYSTEM_PROMPT
    return HELP_BOT_SYSTEM_PROMPT


def test_help_bot_no_dicta_el_dominio_muerto(prompt_help_bot):
    assert "mealfitrd.com" not in prompt_help_bot
    # Y sigue citando URLs: si alguien las borra en vez de corregirlas, el bot
    # deja de saber a dónde mandar al usuario y este test lo nota.
    assert "bioboros.com/supermercado" in prompt_help_bot
    assert "bioboros.com/medical" in prompt_help_bot


def test_help_bot_llama_Max_al_tier_top_y_no_le_inventa_precio_anual(prompt_help_bot):
    # La UI muestra 'Suscripción Max' (Upgrade.jsx); el bot decía "Ultra".
    assert "**Max**" in prompt_help_bot
    # $449.99/año nunca se vendió: `ANNUAL_DISABLED_TIERS` excluye `ultra`
    # desde 2026-06-19 y ese plan no existe en PayPal. El bot lo anunciaba.
    assert "449.99" not in prompt_help_bot
