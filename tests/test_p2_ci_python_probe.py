"""[P2-CI-PYTHON-PROBE · 2026-08-15] El gate elige el intérprete PROBÁNDOLO.

QUÉ PASÓ. `deploy-mealfit.ps1` dejaba la suite del backend fuera del gate con este
motivo escrito: «la suite tiene una BASELINE ROJA, 43 fallos medidos el
2026-08-14, así que un gate que la incluyera abortaría TODOS los despliegues». El
razonamiento era bueno — un gate que no puede pasar entrena a saltárselo — pero la
premisa no se sostuvo al medirla: **17.898 passed, 0 failed** con el entorno real.

Y al buscar por qué figuraba roja apareció algo más útil que la cifra:
`run_ci.ps1` elegía el intérprete con «el primer fichero que exista», y en la
máquina del dueño eso era `backend/venv/bin/python.exe` — un venv a medio
provisionar que tiene `pytest` pero **no** `fastapi`, `langgraph`, `psycopg` ni
`pydantic`. Con ese python la suite no falla: **ni siquiera colecciona**.

O sea que el gate no podía distinguir «los tests fallan» de «este python no tiene
las dependencias». Esa ambigüedad es lo que convierte una deuda en permanente: el
síntoma es idéntico en los dos casos, así que nadie puede saber si arreglarlo
cuesta cinco minutos o cinco días, y la respuesta por defecto pasa a ser dejarlo.

QUÉ ANCLA ESTE TEST. Las tres propiedades que hacen que el gate signifique algo:

  1. La elección se hace con una SONDA, no con `Test-Path` a secas.
  2. La sonda importa algo que el venv roto NO tiene. Sondar con `pytest` sería
     inútil: es justo lo que el venv roto sí tiene, y por lo que pasaba el filtro.
  3. El gate del deploy ya no salta el backend incondicionalmente.

Y una cuarta de higiene: las dos copias de `run_ci.ps1` (raíz y backend) no
derivan — misma regla que P3-MIGRATIONS-SSOT, por el mismo motivo (repos hermanos).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_CI_BACKEND = _REPO_ROOT / "backend" / "scripts" / "run_ci.ps1"
_RUN_CI_ROOT = _REPO_ROOT / "scripts" / "run_ci.ps1"
_DEPLOY = _REPO_ROOT / "deploy-mealfit.ps1"

# El paquete que el venv a medias NO tenía. Sondar con él es lo que separa
# «entorno provisionado» de «hay un python.exe ahí».
_PAQUETE_SONDA = "fastapi"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def test_el_interprete_se_elige_probandolo() -> None:
    """Nada de «el primer fichero que exista»."""
    src = _leer(_RUN_CI_BACKEND)

    assert "Resolve-BackendPython" in src, (
        "`run_ci.ps1` perdió `Resolve-BackendPython`. Sin la sonda, la elección "
        "vuelve a ser «el primer python.exe que exista» — y el que existe en la "
        "máquina del dueño no puede importar la app."
    )
    assert re.search(rf"-c\s+[\"'][^\"']*\b{_PAQUETE_SONDA}\b", src), (
        f"La sonda ya no importa `{_PAQUETE_SONDA}`. Ese import es TODO el valor "
        "del mecanismo: es lo que el venv a medio provisionar no tiene."
    )


def test_la_sonda_no_se_conforma_con_pytest() -> None:
    """Sondar sólo con `pytest` reintroduce el bug entero.

    El venv roto tiene pytest. Si la sonda se relajara a `import pytest`, volvería
    a aceptarlo y el gate volvería a fallar por ImportError disfrazado de test roto.
    """
    src = _leer(_RUN_CI_BACKEND)
    m = re.search(r"-c\s+[\"']import ([^\"']+)[\"']", src)
    assert m, "No encuentro el import de la sonda en run_ci.ps1."
    modulos = {x.strip() for x in m.group(1).split(",")}
    assert modulos - {"pytest"}, (
        f"La sonda importa sólo {modulos}. `pytest` está en el venv roto: sondar "
        "con él es exactamente lo que dejaba pasar al intérprete equivocado."
    )


def test_falla_ruidosamente_si_ningun_python_sirve() -> None:
    """Sin candidato válido, el paso revienta con instrucciones — no corre a ciegas."""
    src = _leer(_RUN_CI_BACKEND)
    assert "MEALFIT_PYTHON" in src, (
        "Desapareció la escotilla `MEALFIT_PYTHON`. Es la salida para CI, "
        "contenedores y máquinas donde la heurística no acierte."
    )
    assert re.search(r"throw\s*\(", src), (
        "`Resolve-BackendPython` ya no lanza cuando ningún candidato sirve. "
        "Degradar en silencio a un python roto es el bug original."
    )


def test_el_deploy_ya_no_salta_el_backend_incondicionalmente() -> None:
    """`-SkipBackend` pasa a depender del target, no a estar siempre puesto."""
    src = _leer(_DEPLOY)

    assert not re.search(r"^\s*\$ciArgs\s*=\s*@\('-SkipBackend'\)\s*$", src, re.M), (
        "`deploy-mealfit.ps1` volvió a saltar el backend SIEMPRE. Si la suite se "
        "puso roja de verdad, la salida no es esto en silencio: es arreglarla, o "
        "congelar la baseline en un fichero y comparar contra ella."
    )
    assert "tocaBackend" in src, (
        "Se perdió la decisión por target. Desplegar sólo el frontend no tiene por "
        "qué esperar ~20 min de suite del backend; desplegarlo sí."
    )


def test_las_dos_copias_de_run_ci_no_derivan() -> None:
    """Misma regla que P3-MIGRATIONS-SSOT: repos hermanos, dos copias, cero deriva."""
    if not _RUN_CI_ROOT.exists():
        pytest.skip("checkout sólo-backend: la copia de la raíz no está aquí")
    a = _leer(_RUN_CI_ROOT)
    b = _leer(_RUN_CI_BACKEND)
    assert a == b, (
        "`scripts/run_ci.ps1` y `backend/scripts/run_ci.ps1` han divergido. La "
        "raíz no tiene remote, así que la copia del backend es la que viaja: si "
        "sólo tocas una, el gate que corre en otra máquina es el viejo."
    )
