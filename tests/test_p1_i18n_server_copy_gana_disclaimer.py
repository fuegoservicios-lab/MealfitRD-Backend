"""[P1-I18N-SERVER-COPY-GANA-SIGUE-ABIERTO · 2026-08-23] El guard «por propiedad» seguía
siendo una enumeración de cinco nombres de campo, y `_review_disclaimer` no estaba en ella.

`P1-I18N-SERVER-COPY-GANA` (22-ago) prometía «reescribir el guard por propiedad: cualquier
`<expresión del servidor> || t(…)` en posición de copy, sea cual sea el nombre del campo». Lo
entregado ensanchó de 1 canal a 5 —`detail|message|error_message|ai_interrupted_message|error`—
y arregló los números de línea, pero siguió siendo una lista cerrada.

MEDIDO mutando el regex de entonces sobre el árbol de hoy:

    False | description: generatedPlan?._review_disclaimer || t('El plan se ajusto...')
    False | toast.error(data?.motivo || t('Algo fallo.'))
    True  | toast.error(data?.detail || t('Algo fallo.'))

El invisible que más costaba es `_review_disclaimer`: el backend lo compone SIEMPRE, y
siempre en español, así que el `|| t(…)` de al lado era RAMA MUERTA — la traducción existía,
estaba revisada, y no se pintaba jamás. En la rama de rechazo médico el usuario leía el
título traducido «Plan ajustado por seguridad médica» y debajo, en español: «El sistema
detectó violaciones críticas (alergias o condiciones médicas)…».

DOS COSAS SE ARREGLARON, y la segunda es la que evita el siguiente:

1. Las seis variantes se glosan al imprimir (`glossReviewDisclaimer`), reusando el motor de
   la nota clínica en vez de inventar uno.
2. El canal del guard pasa a ser CUALQUIER campo, con lista blanca razonada de los que son
   DATO y no prosa. Al abrirlo apareció un falso positivo real —`p.description || ''` en
   `SupermarketPage`, que es un campo de formulario— y eso obligó a afinar la propiedad: el
   fallback tiene que ser una TRADUCCIÓN (`|| … t(`). Sin esa mitad, el guard acusa a un
   sitio correcto, y un guard que acusa de más se acaba silenciando.

ESTE fichero ancla la paridad del disclaimer, que es la mitad que el guard de posición no
puede ver: que las seis variantes del backend sigan siendo reconocibles por el glosador.
Mismo modo de fallo que la nota clínica —el copy vive en `graph_orchestrator.py` y su
traducción en otro repo— y misma defensa bidireccional.

tooltip-anchor: P1-I18N-SERVER-COPY-GANA-SIGUE-ABIERTO
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_MARKER = "P1-I18N-SERVER-COPY-GANA-SIGUE-ABIERTO"
_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_ORQ = _BACKEND / "graph_orchestrator.py"
_GLOSADOR = _ROOT / "frontend" / "src" / "utils" / "clinicalNoteGloss.js"


def _fuente(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def _norm(s: str) -> str:
    return " ".join(s.split())


def _disclaimers_del_backend() -> set:
    """Las variantes que el backend asigna a `_review_disclaimer`.

    Se derivan del código —nada de listas a mano, que es lo que dejó fuera `_med_note` en el
    guard de la nota clínica— buscando las asignaciones y fundiendo la concatenación
    implícita de literales adyacentes, que es como están escritas.
    """
    src = _fuente(_ORQ)
    fuera = set()
    for m in re.finditer(r'"_review_disclaimer"\]?\s*[:=]\s*\(', src):
        # Desde el paréntesis, tomar los literales hasta cerrarlo.
        i, prof, trozos = m.end(), 1, []
        while i < len(src) and prof:
            if src[i] == "(":
                prof += 1
            elif src[i] == ")":
                prof -= 1
            i += 1
        bloque = src[m.end():i]
        trozos = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', bloque)
        junto = "".join(trozos).strip()
        if len(junto) > 40:
            fuera.add(_norm(junto))
    return fuera


def _claves_del_glosador() -> set:
    src = _fuente(_GLOSADOR)
    ini = src.find("const _DISCLAIMERS")
    assert ini > 0, f"desapareció `_DISCLAIMERS` de {_GLOSADOR.name} [{_MARKER}]"
    fin = src.find("];", ini)
    return {_norm(x) for x in re.findall(r"t\('([^']+)'\)", src[ini:fin])}


def test_el_extractor_del_backend_encuentra_algo() -> None:
    """Centinela: si la derivación sale vacía, el test de abajo aprueba la nada contra la
    nada. Es la lección del parser de `_LM_DISPLAY_GROUPS`."""
    d = _disclaimers_del_backend()
    assert len(d) >= 4, (
        f"sólo extraje {len(d)} disclaimers de {_ORQ.name}: cambió la forma de componerlos y "
        f"este guard se quedó midiendo el vacío. [{_MARKER}]"
    )


def test_toda_variante_del_backend_la_reconoce_el_glosador() -> None:
    """El sentido que importa: si alguien cambia el copy en el backend, el glosador queda
    inerte y el usuario vuelve a leer español bajo un título traducido, sin que nada avise."""
    faltan = sorted(_disclaimers_del_backend() - _claves_del_glosador())
    assert not faltan, (
        f"{len(faltan)} variante(s) de `_review_disclaimer` que el backend emite y el "
        f"glosador NO reconoce: {[x[:70] + '…' for x in faltan]}. Saldrán en español en los "
        f"cuatro idiomas, debajo de un título traducido. [{_MARKER}]"
    )


def test_ninguna_clave_del_glosador_quedo_fosil() -> None:
    """El sentido inverso: una clave que el backend ya no emite es una traducción muerta ×4
    que el gate de catálogos da por viva, porque para él sigue siendo una clave usada."""
    sobran = sorted(_claves_del_glosador() - _disclaimers_del_backend())
    assert not sobran, (
        f"{len(sobran)} clave(s) del glosador que el backend ya no emite: "
        f"{[x[:70] + '…' for x in sobran]}. [{_MARKER}]"
    )
