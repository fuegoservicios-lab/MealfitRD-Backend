"""[P2-CI-COUNTRY-COVERAGE · 2026-08-21] El gate corría los tests de países… y nadie lo sabía.

La auditoría lo midió así: «el CI remoto lleva 94 de 100 corridas en rojo en los tres repos: ni un
test del sistema de países se ha ejecutado nunca allí».

Al abrir el workflow, la causa NO era que faltaran del gate — `pytest tests/ -m "not e2e"` los
incluye a todos por construcción. Era que el job venía fallando por otra cosa y **`-x` paraba en el
primer fallo**, así que cada corrida devolvía UN dato en vez del mapa y lo que venía detrás no se
ejecutaba nunca. Exactamente la misma forma del defecto que esta semana costó 25 corridas seguidas:
eslint abortaba los ocho pasos siguientes del job y nadie vio que el gate de i18n no había corrido
jamás.

DOS CAMBIOS, y los dos van contra el mismo modo de fallo — el veredicto que no distingue:

  1. **Sin `-x` ni `--maxfail`.** G30 demostró que un techo de 25 errores de colección volvió a
     ocultar toda la suite. El gate debe devolver el mapa completo.

  2. **Los e2e dejan de callar.** No se ejecutan aquí —necesitan Neon y este job corre sin base de
     datos— y eso está bien; lo que no estaba bien es que no se dijera. «Añadirlos al gate daría
     verde sin verificar nada»: un verde que no distingue «pasó» de «no se ejecutó» es
     `P1-CI-GATE-INCONCLUSIVE`, y su lección es que «no concluyente» no puede colapsar a ninguno de
     los dos lados. El paso nuevo NO falla el build: imprime el recuento para que el número esté a
     la vista de quien lee el log, en vez de vivir en la cabeza de quien escribió el filtro.

Lo que este P-fix NO hace: montar Neon en CI para ejecutar los 65 e2e de verdad. Eso es
infraestructura y presupuesto, decisión del dueño. Aquí se cierra la parte que sí es un defecto —
que el gate mintiera por omisión.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parent
_WORKFLOW_CANDIDATES = (
    BACKEND_ROOT / ".github" / "workflows" / "ci.yml",
    WORKSPACE_ROOT / ".github" / "workflows" / "ci.yml",
)
_WORKFLOWS = tuple(path for path in _WORKFLOW_CANDIDATES if path.is_file())


def test_hay_al_menos_un_workflow_real_que_auditar():
    assert _WORKFLOWS, "no encontré ningún workflow de CI que auditar"


@pytest.fixture(
    scope="module",
    params=_WORKFLOWS or (None,),
    ids=lambda path: path.parents[2].name if path else "workflow-ausente",
)
def ci_path(request) -> Path:
    assert request.param is not None, "no encontré ningún workflow de CI que auditar"
    return request.param


@pytest.fixture(scope="module")
def ci(ci_path) -> str:
    return ci_path.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def paso_pytest(ci) -> str:
    """El paso del gate SIN sus comentarios.

    Sin el filtro, el guard de `-x` acusaba a mi propio comentario del YAML —que dice literalmente
    «`-x` fuera» para explicar por qué se quitó—. Enésima vez que un comentario derrota a un guard
    en este repo, y van varias mías en esta ola: el remedio, siempre el mismo, es mirar sólo lo que
    se EJECUTA."""
    i = ci.index("Run pytest (parser-based + functional)")
    j = ci.index("\n      - name:", i)
    return "\n".join(l for l in ci[i:j].split("\n") if not l.lstrip().startswith("#"))


# ── El gate ejecuta los tests de países ─────────────────────────────────────────────────────────

def test_el_gate_corre_toda_la_carpeta_de_tests(paso_pytest):
    """`pytest tests/` los incluye por construcción: no hay lista que mantener, así que un test
    nuevo entra en el gate el día que se escribe. Si alguien lo acota a una lista de ficheros,
    vuelve el problema de «un test que existe y el gate no corre»."""
    assert re.search(r"pytest\s+tests/", paso_pytest), (
        "el gate dejó de correr la carpeta entera: un test nuevo ya no entraría solo"
    )


def test_un_fallo_ya_no_esconde_el_resto(paso_pytest):
    """`-x` paraba en el primer rojo. Cada corrida devolvía UN dato en vez del mapa — la misma
    forma del defecto que costó 25 corridas seguidas cuando eslint abortaba el job entero."""
    assert not re.search(r"(?<![\w-])-x(?![\w-])", paso_pytest), (
        "volvió el `-x`: un fallo cualquiera esconde todo lo que venga detrás"
    )
    assert "--maxfail" not in paso_pytest, (
        "volvió un techo de errores: G30 demostró que 25 fallos de colección pueden impedir "
        "que se ejecute una sola prueba"
    )


# ── El silencio de los e2e ──────────────────────────────────────────────────────────────────────

def test_el_gate_dice_que_los_e2e_no_se_han_ejecutado(ci):
    """Un verde que no distingue «pasó» de «no se ejecutó» es P1-CI-GATE-INCONCLUSIVE. El gate
    filtra los e2e por diseño (no hay DB); lo que no puede es callárselo."""
    assert "e2e" in ci and re.search(r"NO se ejecutan|sin verificar", ci), (
        "el workflow no dice en ningún sitio que los e2e no se ejecutan: el verde sigue "
        "significando dos cosas distintas"
    )


def test_ese_aviso_no_falla_el_build(ci):
    """Deliberado: convertirlo en un fallo dejaría el gate rojo para siempre sin que nadie pueda
    arreglarlo desde el código — la DB es infraestructura. Informar sí; bloquear, no."""
    i = ci.index("Report unverified e2e count")
    bloque = ci[i:i + 600]
    assert "::notice" in bloque, "el aviso dejó de ser informativo"
    assert "exit 1" not in bloque and "|| false" not in bloque, (
        "el paso informativo empezó a fallar el build: la DB en CI es infraestructura, no código"
    )


# ── Que el filtro siga significando lo que dice ─────────────────────────────────────────────────

def test_hay_e2e_de_verdad_que_quedan_fuera():
    """El guard del guard: si nadie marcara ya `@pytest.mark.e2e`, el aviso hablaría de un conjunto
    vacío y este fichero entero sería decorativo. Se mide sobre los ficheros reales."""
    tests_dir = Path(__file__).resolve().parent
    marcados = sum(
        1 for f in tests_dir.glob("test_*.py")
        if "pytest.mark.e2e" in f.read_text(encoding="utf-8", errors="replace")
    )
    assert marcados >= 5, (
        f"sólo {marcados} ficheros usan @pytest.mark.e2e: o el filtro ya no excluye nada, o los "
        f"e2e se marcaron de otra forma y el aviso del gate quedó obsoleto"
    )
