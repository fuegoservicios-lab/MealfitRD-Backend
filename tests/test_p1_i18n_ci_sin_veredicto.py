"""[P1-I18N-CI-SIN-VEREDICTO · 2026-08-23] Ningún CI ejecutaba los guards de idiomas — ni el
P0 clínico de alérgenos.

MEDIDO con `gh run view --log-failed`: el CI del backend hacía `pytest tests/ --maxfail=25`
sobre un checkout SOLO-backend, y 400 de los 1.932 tests (20 %) construyen una ruta a
`frontend/src` vía `Path(__file__).parents[2]`. En ese checkout fallaban en COLECCIÓN con
`FileNotFoundError` — no son fallos de test, son ficheros que no se pueden ni importar — y
a los 25 el job moría sin ejecutar UN test. Semanas en rojo sin veredicto. Entre lo que no
corría: 55 de los 96 guards de idiomas, incluido `test_p0_i18n_alergenos_fr_it_pt.py`.

El CI del frontend sí ejecutaba pero llevaba 68 corridas en rojo por otras causas, y como el
paso de i18n va el PRIMERO, cuando fallaba se saltaban los 8 siguientes. El de la raíz
moría en `Set up Node`. Resultado neto: los ~64 guards de idiomas sólo se verificaban en la
máquina del dueño, vía `run_ci.ps1` — que además se salta con `-SkipTests`, la válvula que
se usa justo cuando hay prisa.

TRES COSAS, y por qué cada una:
  1. El backend se clona en `backend/` y el frontend en `frontend/`, hermanos: es la
     disposición que `parents[2]` asume. El repo del frontend es PRIVADO y el token del
     job no lo alcanza: hace falta el secret `SIBLING_REPO_TOKEN` (acción del dueño).
  2. SIN el secret, la degradación se VE: el resumen del job dice cuántos ficheros se
     deseleccionan y por qué, y se deseleccionan con `--ignore` DERIVADO del árbol (el
     mismo grep que esta medición), no con una lista a mano.
  3. `--maxfail` FUERA. Un CI que para a los N rojos devuelve N datos; el mapa entero es
     lo que distingue «un test roto» de «el entorno está mal».

Este guard ancla las tres en el fichero del workflow. No puede ejecutar el CI, pero sí
impedir que alguien vuelva a clonar en la raíz o a poner un `--maxfail` que lo deje mudo.

tooltip-anchor: P1-I18N-CI-SIN-VEREDICTO
"""
from __future__ import annotations

import re
from pathlib import Path

_MARKER = "P1-I18N-CI-SIN-VEREDICTO"
_CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _ci() -> str:
    return _CI.read_text(encoding="utf-8")


def test_el_backend_se_clona_como_hermano_en_backend() -> None:
    ci = _ci()
    assert re.search(r"uses:\s*actions/checkout@[0-9a-f]+.*\n(?:\s+#.*\n)*\s+with:\s*\n\s+path:\s*backend\b", ci), (
        f"el backend ya no se clona en `backend/`: los 400 tests que resuelven el frontend "
        f"por `parents[2]` volverán a reventar en colección. [{_MARKER}]"
    )


def test_el_frontend_hermano_se_clona_en_frontend_con_el_secret() -> None:
    ci = _ci()
    assert "repository: fuegoservicios-lab/MealfitRD" in ci, (
        f"desapareció el checkout del frontend hermano [{_MARKER}]"
    )
    assert re.search(r"token:\s*\$\{\{\s*secrets\.SIBLING_REPO_TOKEN\s*\}\}", ci), (
        f"el checkout del frontend no usa `SIBLING_REPO_TOKEN`: el repo es privado y el "
        f"GITHUB_TOKEN del job no lo alcanza. [{_MARKER}]"
    )
    assert re.search(r"path:\s*frontend\b", ci), f"el frontend no se clona en `frontend/` [{_MARKER}]"


def test_sin_el_secret_la_degradacion_es_visible_y_derivada() -> None:
    """Sin el token, el job NO puede fingir cobertura: lo dice en el resumen y deselecciona
    por una lista DERIVADA del árbol, no escrita a mano."""
    ci = _ci()
    assert "GITHUB_STEP_SUMMARY" in ci, (
        f"el job ya no declara en su resumen si corrió con o sin el frontend hermano. "
        f"Una degradación que no se ve es una cobertura que se da por buena. [{_MARKER}]"
    )
    assert re.search(r"grep -lE .*frontend.* tests/test_\*\.py", ci), (
        f"la deselección de los tests que leen el frontend ya no se deriva del árbol con "
        f"grep: una lista a mano se queda atrás con el primer test nuevo. [{_MARKER}]"
    )
    assert "--ignore=" in ci, f"sin `--ignore`, los ficheros que leen el frontend revientan en colección [{_MARKER}]"


def test_el_pytest_del_ci_no_lleva_maxfail() -> None:
    """`--maxfail` convertía 25 errores de COLECCIÓN en «0 tests ejecutados»."""
    ci = _ci()
    run = re.search(r"name: Run pytest.*?run: \|(.*?)(?:\n\s{0,6}- name:|\Z)", ci, re.S)
    assert run, f"desapareció el paso de pytest [{_MARKER}]"
    cuerpo = "\n".join(l for l in run.group(1).splitlines() if not l.strip().startswith("#"))
    assert "--maxfail" not in cuerpo, (
        f"`--maxfail` ha vuelto al pytest del CI. Con errores de colección agota el cupo "
        f"antes del primer test y la corrida termina SIN veredicto. [{_MARKER}]"
    )
    assert re.search(r"(?m)^\s*-x\b|\s-x\s", cuerpo) is None, f"`-x` ha vuelto [{_MARKER}]"


def test_el_secret_no_se_evalua_en_un_if_de_step() -> None:
    """`secrets.*` dentro de un `if` de step llega vacío; va por `env` del job."""
    ci = _ci()
    assert not re.search(r"if:\s*\$\{\{\s*secrets\.", ci), (
        f"hay un `if` de step leyendo `secrets.*` directamente: GitHub lo deja vacío ahí y "
        f"el checkout del hermano nunca correría. [{_MARKER}]"
    )
    assert re.search(r"HAS_SIBLING_TOKEN:\s*\$\{\{\s*secrets\.SIBLING_REPO_TOKEN\s*!=\s*''\s*\}\}", ci), (
        f"el puente `env.HAS_SIBLING_TOKEN` desapareció [{_MARKER}]"
    )
