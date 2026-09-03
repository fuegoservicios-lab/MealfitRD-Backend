"""[P1-CI-GATE-PASSABLE · 2026-08-14] Los gates de CI del frontend tienen que
poder pasar.

QUÉ PASÓ. Los dos jobs de `frontend/.github/workflows/ci.yml` estaban en rojo:

  · `quality` — `npx eslint . --max-warnings 148`, contra 180 warnings reales en
    un checkout limpio. El techo se congeló el 2026-07-12 como «estado actual»;
    cuando el lockfile subió `eslint-plugin-react-hooks` a 7.0.1, la regla nueva
    `set-state-in-effect` aportó 16 warnings sobre código que nadie había tocado.
    El job fallaba sin que existiera ni un defecto nuevo.

  · `audit` — `nanoid GHSA-28wg-ghj8-5hjv` (high, cadena de producción) nunca se
    trió, así que `scripts/audit-gate.mjs` salía con exit 1.

POR QUÉ ES UN DEFECTO Y NO COSMÉTICA. Un gate que no puede pasar deja de ser un
gate: entrena a leer el rojo como ruido, y entonces el rojo de verdad tampoco se
lee. Es la misma lección que el commit ac042bd escribió para el gate del deploy
(«El gate del deploy tiene que poder pasar, o entrena a saltárselo»), aplicada al
gate de CI.

QUÉ ANCLA ESTE TEST. No el número —el número sube y baja legítimamente— sino las
tres propiedades que hacen que el número signifique algo:

  1. El techo de `ci.yml` y la constante de `scripts/lint-count.mjs` no derivan.
     Son dos copias del mismo dato en ficheros distintos; sin este ancla, el día
     que alguien recalibre una sola, el reporte informativo empieza a mentir.
  2. `coverage/` está fuera del lint. Es lo que hacía que el conteo LOCAL y el de
     CI difirieran (en checkout limpio el directorio no existe), y una
     recalibración medida en la orilla equivocada nace desfasada.
  3. Toda recalibración del techo deja su causa escrita. Sin causa, el siguiente
     que lo mire no puede saber si puede bajarlo.

Además comprueba que `--max-warnings` sigue existiendo: quitarlo «para que pase»
sería la forma más rápida de convertir el gate en decorado, y es justo el atajo
que este P-fix existe para cerrar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND = _REPO_ROOT / "frontend"

_CI_YML = _FRONTEND / ".github" / "workflows" / "ci.yml"
_LINT_COUNT = _FRONTEND / "scripts" / "lint-count.mjs"
_ESLINT_CONFIG = _FRONTEND / "eslint.config.js"
_AUDIT_GATE = _FRONTEND / "scripts" / "audit-gate.mjs"

_MARKER = "P1-CI-GATE-PASSABLE"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def test_ci_conserva_el_gate_de_warnings() -> None:
    """`--max-warnings` sigue en el job `quality`.

    Quitarlo haría pasar el job siempre, que es exactamente el atajo que este
    P-fix cierra: el objetivo era un gate VERDE, no un gate ausente.
    """
    ci = _leer(_CI_YML)
    assert "--max-warnings" in ci, (
        "El job `quality` perdió `--max-warnings`. Un lint sin techo no bloquea "
        "warnings nuevos: pasa siempre. Si el techo estorbaba, recalibralo — no "
        "lo borres."
    )
    assert "npx eslint ." in ci, "El job `quality` ya no ejecuta eslint sobre el árbol."


def test_techo_de_warnings_sincronizado_entre_ci_y_el_reporte() -> None:
    """El techo vive en DOS ficheros; este test impide que deriven.

    `ci.yml` lo usa para gatear y `lint-count.mjs` para reportar «cuánto margen
    queda». Si sólo se recalibra uno, el reporte informativo pasa a mentir — y su
    única razón de existir es que el número sea visible y fiable.
    """
    ci = _leer(_CI_YML)
    lint_count = _leer(_LINT_COUNT)

    m_ci = re.search(r"--max-warnings\s+(\d+)", ci)
    assert m_ci, "No encuentro `--max-warnings <n>` en ci.yml."
    techo_ci = int(m_ci.group(1))

    m_script = re.search(r"const\s+CEILING\s*=\s*(\d+)", lint_count)
    assert m_script, (
        "No encuentro `const CEILING = <n>` en scripts/lint-count.mjs. Si "
        "renombraste la constante, actualiza este test — el ancla es el "
        "sincronismo, no el nombre."
    )
    techo_script = int(m_script.group(1))

    assert techo_ci == techo_script, (
        f"Deriva del techo de warnings: ci.yml dice {techo_ci} y "
        f"scripts/lint-count.mjs dice {techo_script}. Son el mismo dato en dos "
        "sitios; recalibrá los dos o el reporte miente sobre el margen."
    )


def test_toda_recalibracion_deja_su_causa_escrita() -> None:
    """El techo va acompañado de la razón, no sólo del número.

    148 se puso con «Tope = estado actual» y nadie pudo saber después si el
    exceso venía de código nuevo (arreglable) o de una regla nueva (recalibrable).
    Esa ambigüedad es la que dejó el job rojo sin diagnóstico.
    """
    ci = _leer(_CI_YML)
    assert _MARKER in ci, (
        f"El marker {_MARKER} desapareció de ci.yml. Es el hilo que conecta el "
        "número con la razón por la que vale ese número."
    )
    assert "set-state-in-effect" in ci, (
        "Se perdió la causa documentada de la última recalibración (la regla que "
        "trajo eslint-plugin-react-hooks 7.0.1). Al recalibrar, sustituí la causa "
        "vieja por la nueva; no la borres sin más."
    )


def test_coverage_fuera_del_lint() -> None:
    """`coverage/` ignorado: sin esto, el conteo local y el de CI no coinciden.

    En un checkout limpio el directorio no existe, así que CI medía un número y
    el desarrollador otro. Un techo sólo sirve si las dos orillas cuentan lo mismo.
    """
    cfg = _leer(_ESLINT_CONFIG)
    m = re.search(r"globalIgnores\(\[([^\]]*)\]\)", cfg, re.S)
    assert m, "No encuentro `globalIgnores([...])` en eslint.config.js."
    assert "'coverage'" in m.group(1) or '"coverage"' in m.group(1), (
        "`coverage` salió de globalIgnores. Vuelve a introducir la asimetría "
        "local/CI que hacía imposible recalibrar el techo con confianza."
    )


def _cuerpo_de_excepciones(gate: str) -> str:
    """El bloque de excepciones de `audit-gate.mjs`, sea cual sea su envoltorio.

    [P0-AUDIT-EXCEPCIONES · 2026-08-18] Ese bloque dejó de ser
    `const ALLOWLIST = new Set([...])` —una lista de GHSA sueltos— y pasó a ser
    `const EXCEPCIONES = { ... }`, donde cada entrada lleva dueño, caducidad y
    mitigación, y una excepción VENCIDA pone el gate en rojo. La `ALLOWLIST`
    sigue existiendo, pero derivada: `new Set(Object.keys(EXCEPCIONES))`.

    Estos dos tests seguían buscando la forma vieja con una expresión regular y
    por eso reventaron: no porque la protección hubiera desaparecido, sino porque
    se había REFORZADO. Es el precio conocido de anclar un guard a la sintaxis de
    producción —y el precio que el repo acepta a cambio de que un renombre falle
    aquí antes de cambiar la conducta en silencio—.

    Se acepta cualquiera de las dos formas para que el fallo, si vuelve a
    cambiar, diga QUÉ falta en vez de «no encuentro nada».
    """
    m = re.search(r"const\s+EXCEPCIONES\s*=\s*\{(.*?)\n\};", gate, re.S)
    if m:
        return m.group(1)
    m = re.search(r"const\s+ALLOWLIST\s*=\s*new\s+Set\(\[(.*?)\]\)", gate, re.S)
    assert m, (
        "No encuentro ni `const EXCEPCIONES = {...}` ni `const ALLOWLIST = new Set([...])` "
        "en audit-gate.mjs. Si el bloque cambió de forma otra vez, enseñale la nueva "
        "a este helper: es el único sitio que hay que tocar."
    )
    return m.group(1)


def test_cada_excepcion_declara_dueno_caducidad_y_mitigacion() -> None:
    """La forma nueva permite exigir más que «que haya un comentario».

    Una excepción de seguridad sin fecha de caducidad se vuelve permanente por
    omisión: nadie decide dejarla para siempre, simplemente nadie vuelve. Con
    `caduca` el propio gate la reabre sola el día marcado.
    """
    gate = _leer(_AUDIT_GATE)

    # Se buscan los OBJETOS que declaran una excepcion, no las palabras sueltas.
    #
    # [P2-GUARD-LITERAL] La primera version de este test hacia `re.search("dueno",
    # gate)` sobre el fichero entero, y por eso NO detecto la mutacion que renombra
    # el campo: la palabra `dueno` tambien aparece en el mensaje de error que el
    # propio script imprime («... con dueno, caduca y mitigacion verificable»). Un
    # texto que CITA el literal satisfacia al guard que lo EXIGE. Es la tercera vez
    # que este repo se come esa trampa; la salida es mirar la estructura, no el texto.
    bloques = []
    for m in re.finditer(r"caduca\s*:", gate):
        abre = gate.rfind("{", 0, m.start())
        cierra = gate.find("}", m.end())
        assert abre != -1 and cierra != -1, "objeto de excepcion mal formado alrededor de `caduca`"
        bloques.append(gate[abre:cierra + 1])

    assert bloques, (
        "Ninguna excepcion declara `caduca`. Los tres campos (dueno / caduca / "
        "mitigacion) son el motivo de que esta estructura sustituyera a una lista "
        "de GHSA sueltos."
    )

    for bloque in bloques:
        for campo in ("dueno", "caduca", "mitigacion"):
            assert re.search(rf"\b{campo}\s*:", bloque), (
                f"Una excepcion no declara `{campo}`:\n{bloque[:200]}"
            )
        fecha = re.search(r"caduca\s*:\s*'([^']*)'", bloque)
        assert fecha and re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha.group(1)), (
            f"`caduca` no es una fecha YYYY-MM-DD en:\n{bloque[:200]}\n"
            "El gate la compara contra hoy; un texto libre no caduca nunca."
        )


def test_el_advisory_de_react_router_no_vuelve_a_la_allowlist() -> None:
    """`GHSA-qwww-vcr4-c8h2` se retiró porque 7.18.2 lo cierra hacia delante.

    Reañadirlo sería enmascarar un advisory que YA tiene remedio. Si un día
    reaparece, el arreglo es el bump, no la excepción.
    """
    gate = _leer(_AUDIT_GATE)
    cuerpo = _cuerpo_de_excepciones(gate)

    entradas_activas = [
        ln for ln in cuerpo.splitlines()
        if "GHSA-qwww-vcr4-c8h2" in ln and not ln.strip().startswith("//")
    ]
    assert not entradas_activas, (
        "GHSA-qwww-vcr4-c8h2 volvió a la allowlist. react-router 7.18.2 lo cierra "
        "hacia delante: la vía es actualizar, no allowlistear. (La MENCIÓN en un "
        "comentario está bien — es el registro de por qué se retiró.)"
    )


def test_la_allowlist_no_crece_sin_justificacion_inline() -> None:
    """Cada GHSA allowlisteado lleva comentario en su bloque.

    La allowlist es la única puerta por la que un high/critical de producción
    llega a `main`. Una entrada sin razón escrita es indistinguible de un
    silenciamiento por prisa.
    """
    gate = _leer(_AUDIT_GATE)
    cuerpo = _cuerpo_de_excepciones(gate)

    ids = set(re.findall(r"'(GHSA-[a-z0-9-]+)'", cuerpo))
    assert ids, "Las EXCEPCIONES quedaron vacías o cambiaron de formato — revisá este test."

    # El bloque entero tiene que llevar prosa: el criterio del repo es que la
    # razón viva junto a la excepción, no sólo en el doc de triage.
    lineas_comentario = [ln for ln in cuerpo.splitlines() if ln.strip().startswith("//")]
    assert len(lineas_comentario) >= 3, (
        "La ALLOWLIST perdió sus comentarios de triage. Cada excepción tiene que "
        "decir por qué está ahí; el doc docs/security/deps-triage.md es el detalle, "
        "no el sustituto."
    )
