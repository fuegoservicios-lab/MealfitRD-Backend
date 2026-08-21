"""[P1-CI-QUALITY-ABORTADO · 2026-08-21] El job `quality` tenía que poder LLEGAR
a sus pasos, no solo declararlos.

QUÉ PASÓ. `frontend/.github/workflows/ci.yml` llevaba desde el 2026-08-18 sin un
solo verde, y por DOS defectos de orden apilados:

  1. `npx eslint . --max-warnings 66` es el PRIMER paso del job y medía 67. Un
     paso que falla aborta el job, así que los OCHO siguientes no se ejecutaban
     nunca — entre ellos `npm run i18n:check:strict`, cableado dos días antes y
     descrito en su propio comentario como «lа única defensa que existe» contra
     que un cambio de copy huerfane su traducción en los 4 idiomas EN SILENCIO.

  2. `npm run check:bundle-size` corría ANTES de `npm run build` y lee
     `dist/assets/AgentPage-*.js`. En un checkout limpio ese directorio no existe,
     así que el paso no podía pasar NUNCA: el job habría seguido rojo aun con
     eslint dentro del techo.

Medido con `gh run list`: 25 runs consecutivos en `failure`. El último verde es
del 2026-08-18, el commit ANTERIOR al que añadió esos pasos. Los doce P-fixes de
i18n del 19 y 20 de agosto entraron todos con CI en rojo, y todos llegaron por
captura del dueño — exactamente lo que el gate se añadió para impedir.

LA LECCIÓN, y es la que estos tests anclan: **cablear un paso no es ejecutarlo.**
Al añadir un guard a CI, la verificación no es que el paso aparezca en el YAML
sino que haya un run VERDE después. Es hermano de `P1-CI-GATE-INCONCLUSIVE` (la
fase A daba verde sin ejecutar la suite) y de `P1-GATE-SCAFFOLDING`: un gate que
no puede pasar entrena a leer el rojo como ruido, y entonces el rojo de verdad
tampoco se lee.

QUÉ ANCLA. Tres propiedades, ninguna de ellas un número:

  · Ningún paso que lea `dist/` precede al `build` que lo crea. Se resuelve por el
    FUENTE de cada script de `package.json`, no por una lista a mano: una lista a
    mano no se entera del script que se añada mañana.
  · El gate de i18n sigue en el job y en `--strict`.
  · El techo global de warnings no queda por debajo de la suma de los techos por
    regla — si se contradicen, el job falla con todas las reglas en su techo y el
    mensaje no dice qué arreglar. Es justo el estado en que estuvo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND = _REPO_ROOT / "frontend"

_CI_YML = _FRONTEND / ".github" / "workflows" / "ci.yml"
_LINT_COUNT = _FRONTEND / "scripts" / "lint-count.mjs"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


_PACKAGE_JSON = _FRONTEND / "package.json"

_MARKER_ORDEN = "P1-CI-QUALITY-ABORTADO"


def _pasos_del_job(ci: str, job: str) -> list[str]:
    """Los `- run:` del job pedido, en orden, sin salir a un parser de YAML.

    `ci.yml` tiene los jobs a dos espacios y sus claves a cuatro, así que el
    bloque de un job va desde su cabecera hasta la siguiente línea con exactamente
    dos espacios de sangría — el mismo criterio que usa el resto de este fichero.
    """
    m = re.search(rf"^  {re.escape(job)}:$", ci, re.M)
    assert m, f"No encuentro el job `{job}` en ci.yml."
    resto = ci[m.end():]
    fin = re.search(r"^  \w[\w-]*:$", resto, re.M)
    bloque = resto[: fin.start()] if fin else resto
    return [ln.strip()[len("- run:"):].strip() for ln in bloque.splitlines()
            if ln.strip().startswith("- run:")]


def _script_lee_dist(nombre: str) -> bool:
    """¿El script `nombre` de package.json lee `dist/`?

    Se resuelve por el FUENTE del fichero que invoca, no por su nombre: quien
    decide si un paso necesita el build es lo que el script abre, y eso solo se
    sabe leyéndolo.
    """
    import json as _json

    pkg = _json.loads(_leer(_PACKAGE_JSON))
    cmd = pkg.get("scripts", {}).get(nombre)
    if not cmd:
        return False
    m = re.search(r"(scripts/[\w.-]+\.mjs)", cmd)
    if not m:
        return False
    fuente = _FRONTEND / m.group(1)
    if not fuente.exists():
        return False
    return "dist" in fuente.read_text(encoding="utf-8")


def test_ningun_paso_que_lee_dist_precede_al_build() -> None:
    """Un paso que mide `dist/` no puede correr antes de crearlo.

    Es el segundo de los dos defectos que tuvieron el job en rojo: aun con eslint
    dentro del techo, `check:bundle-size` no podía pasar en un checkout limpio
    porque el directorio que abre todavía no existía.
    """
    ci = _leer(_CI_YML)
    pasos = _pasos_del_job(ci, "quality")

    idx_build = next((i for i, p in enumerate(pasos) if p == "npm run build"), None)
    assert idx_build is not None, (
        "El job `quality` ya no tiene un paso `npm run build`. Si el build se "
        "movió a otro job, este test tiene que seguirlo: sin build no hay `dist/` "
        "y los gates que lo miden no pueden pasar."
    )

    culpables = []
    for i, paso in enumerate(pasos):
        m = re.fullmatch(r"npm run ([\w:-]+)", paso)
        if not m or i > idx_build:
            continue
        if _script_lee_dist(m.group(1)):
            culpables.append((i, paso))

    assert not culpables, (
        "Estos pasos del job `quality` leen `dist/` y corren ANTES de "
        f"`npm run build` (índice {idx_build}): "
        + ", ".join(f"{p!r} (índice {i})" for i, p in culpables)
        + ". En un checkout limpio `dist/` no existe, así que el job no puede "
        "pasar y aborta a todos los pasos siguientes. Moverlos DESPUÉS del build. "
        f"[{_MARKER_ORDEN}]"
    )


def test_el_gate_de_i18n_sigue_en_el_job_y_en_estricto() -> None:
    """`i18n:check:strict` es el paso que los dos defectos de orden apagaron.

    Se ancla aquí, junto al contrato de orden, porque el modo de fallo que lo dejó
    sin ejecutar no fue que alguien lo borrara: fue que un paso anterior abortaba
    el job. El paso puede estar escrito y no correr nunca.
    """
    ci = _leer(_CI_YML)
    pasos = _pasos_del_job(ci, "quality")

    assert "npm run i18n:check:strict" in pasos, (
        "El job `quality` perdió `npm run i18n:check:strict`. El motor de "
        "traducción usa el texto español COMO clave, así que editar un copy "
        "huerfana su traducción en los 4 idiomas EN SILENCIO. Sin `--strict` el "
        "comprobador no exige cobertura y da verde justo en ese caso. "
        f"[{_MARKER_ORDEN}]"
    )


def test_el_techo_de_warnings_no_puede_estar_por_debajo_de_su_propia_regla() -> None:
    """El techo global y la suma de los techos por regla tienen que ser coherentes.

    `P2-LINT-RATCHET-POR-REGLA` puso techo por regla porque el global es fungible.
    Pero si el global queda por DEBAJO de lo que los techos por regla permiten, el
    job falla en el paso 1 con todos los techos por regla respetados — que es
    exactamente el estado en que estuvo: 67 warnings, techo global 66, y el
    trinquete por regla también a 1 de distancia. Un gate cuyos dos números se
    contradicen no dice qué arreglar.
    """
    ci = _leer(_CI_YML)
    lint_count = _leer(_LINT_COUNT)

    m_ci = re.search(r"--max-warnings\s+(\d+)", ci)
    assert m_ci, "No encuentro `--max-warnings <n>` en ci.yml."
    techo_global = int(m_ci.group(1))

    por_regla = {
        k: int(v)
        for k, v in re.findall(r"['\"]([\w-]+/[\w-]+)['\"]\s*:\s*(\d+)", lint_count)
    }
    if not por_regla:
        pytest.skip(
            "No encuentro los techos por regla en lint-count.mjs — si cambió el "
            "formato del mapa, actualizá este test."
        )

    suma = sum(por_regla.values())
    assert techo_global >= suma, (
        f"El techo global de ci.yml ({techo_global}) es MENOR que la suma de los "
        f"techos por regla ({suma}: {por_regla}). Con todas las reglas dentro de "
        "su techo el job seguiría fallando en el paso 1, y el mensaje no diría "
        f"cuál arreglar. Subí el global o bajá los de regla. [{_MARKER_ORDEN}]"
    )
