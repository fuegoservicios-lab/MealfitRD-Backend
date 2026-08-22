"""[P3-COUNTRY-DOC-TRUTH · 2026-08-22] Tres documentos del sistema de países dicen cosas que
dejaron de ser ciertas — y las tres mienten en la dirección que manda a alguien a trabajar donde
no duele.

Este repo ya tiene nombre para el modo de fallo: **comentario-vence-guard**, siete veces en la
ola de agosto. La forma es siempre la misma: la prosa describe un mundo anterior, alguien la lee
como si fuera el contrato, y actúa sobre el mundo que la prosa describe. Aquí va en la dirección
más cara de todas, porque los tres textos son lo PRIMERO que lee quien va a tocar el sistema.

LO MEDIDO HOY:

1. **`country_system_f1.md` dice que el flip NO se ejecutó.** El runbook (§«Runbook del flip»)
   afirma literalmente «el flip **NO se ejecutó** en Fase 2 — este runbook es la guía completa
   para cuando el dueño decida encenderlo». Ciento diez líneas más abajo, el MISMO documento
   titula «Incidente del día del flip» y narra dos incidentes fechados el 2026-08-18. El flip
   está vivo en producción desde entonces (`MEALFIT_COUNTRY_SYSTEM=true` en el `.env` del VPS).
   Quien lea la primera afirmación tratará todo gap de países como código en oscuro, y no lo es:
   es producción viva con dos planes beta reales persistidos.

2. **`landing_benchmarks.md` nunca recibió la anotación que la spec prometió por escrito.** La
   spec declara «el benchmark del landing queda scoped a RD» **con la condición de anotarlo en su
   doc canónica**. Cero menciones de país en todo el fichero. Es una limitación aceptada que nadie
   documentó: exactamente el patrón de `P2-VISION-COUNTRY`, donde una decisión se aceptó «con una
   condición escrita» y la condición no se cumplió. *Una decisión que nadie defendió por escrito
   vuelve sin que nadie lo note* — la lección ya está en la memoria de este repo.

3. **El comentario de `COUNTRY_POOLS` dice «Solo ES tiene pool propio hoy».** El dict tiene
   CINCO claves (ES, MX, CO, PR, US). El código está mejor que su documentación, que es la
   dirección menos peligrosa de las dos pero no es inocua: quien lea el comentario creerá que
   México, Colombia, Puerto Rico y EE. UU. caen al pool dominicano en el camino degradado, y ese
   es justo el diagnóstico equivocado que le haría «arreglar» algo que ya funciona.

POR QUÉ ESTE FICHERO NO SE LIMITA A BUSCAR LA FRASE MALA. Anclar «que no diga *NO se ejecutó*» se
satisface renombrando la frase, y el tercer caso enseña por qué eso no basta: el comentario no
estaba mal escrito, estaba **desactualizado respecto a un dato que crece**. Así que el ancla del
caso 3 es de PARIDAD (cada clave del dict tiene que aparecer en su comentario), no de literal: una
sexta entrada sin documentar falla igual que la cuarta sin documentar habría fallado en su día.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_F1 = _BACKEND / "docs" / "country_system_f1.md"
_BENCH = _BACKEND / "docs" / "landing_benchmarks.md"
_CONSTANTS = _BACKEND / "constants.py"

#: La fecha del flip, verificada contra el `.env` del VPS y contra los dos incidentes que el
#: propio documento narra. Si alguien la mueve, que sea a sabiendas.
_FECHA_FLIP = "2026-08-18"


def _leer(p: Path) -> str:
    if not p.is_file():
        pytest.skip(f"{p.name} no está en este árbol")
    return p.read_text(encoding="utf-8", errors="replace")


# ── 1 · El runbook del flip ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def f1() -> str:
    return _leer(_F1)


def test_el_runbook_no_declara_pendiente_un_flip_ya_ejecutado(f1):
    """El defecto literal. La frase vivía en la cabecera del runbook, que es lo primero que lee
    quien va a operar el sistema."""
    seccion = f1[f1.index("## Runbook del flip"):]
    assert not re.search(r"flip\s+\*\*NO se ejecut", seccion), (
        "el runbook volvió a declarar el flip como pendiente. Está vivo en producción desde el "
        f"{_FECHA_FLIP} y el propio documento narra dos incidentes posteriores"
    )


def test_el_runbook_dice_cuando_se_ejecuto(f1):
    """No basta con borrar la mentira: sin fecha, el lector no puede saber si los incidentes que
    vienen después son de antes o de después del flip — que es justo lo que el documento tiene que
    resolverle."""
    seccion = f1[f1.index("## Runbook del flip"):f1.index("### 1. Backend")]
    assert _FECHA_FLIP in seccion, (
        f"el runbook no dice cuándo se ejecutó el flip ({_FECHA_FLIP}); sin fecha, la sección de "
        f"incidentes de más abajo queda sin ancla temporal"
    )


def test_la_contradiccion_interna_esta_cerrada(f1):
    """El guard del guard, y la razón por la que este fichero existe: la contradicción era
    INTERNA, entre dos secciones del mismo documento. Si mañana desaparece la sección de
    incidentes, el caso de arriba dejaría de tener contra qué contradecirse y este test avisa en
    vez de quedarse mudo."""
    assert "Incidente del día del flip" in f1, (
        "desapareció la sección de incidentes post-flip: era la prueba interna de que el flip "
        "sucedió. Si se movió a otro documento, re-ancla este test allí"
    )


# ── 2 · La anotación que la spec prometió ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bench() -> str:
    return _leer(_BENCH)


def test_el_benchmark_declara_su_alcance_de_pais(bench):
    """La condición escrita de la spec, sin cumplir desde que se escribió. Un benchmark cuyas
    cifras alimentan el landing y que se evalúa entero como dominicano tiene que decirlo en el
    único sitio donde alguien iría a comprobarlo."""
    assert re.search(r"scoped a RD|acotad\w+ a RD|alcance.{0,40}RD", bench, re.I), (
        "`landing_benchmarks.md` sigue sin la anotación de alcance que la spec prometió por "
        "escrito. Sus 20 perfiles clínicos se evalúan todos como dominicanos"
    )


def test_el_benchmark_documenta_el_eje_de_pais_que_ya_tiene(bench):
    """Y la otra mitad: desde `P2-LANDING-BENCH-COUNTRY` el modo `structural` SÍ tiene eje de
    país. Documentar sólo la limitación dejaría el documento a medias en la dirección contraria —
    alguien volvería a añadir el eje que ya existe."""
    assert "structural_facts_por_pais" in bench or re.search(
        r"eje de pa[ií]s", bench, re.I), (
        "el doc no menciona el eje de país del modo `structural`; sin eso, la anotación de "
        "alcance se lee como «no hay nada por país» y ya no es cierto"
    )


# ── 3 · La paridad del comentario de COUNTRY_POOLS ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def constantes() -> str:
    return _leer(_CONSTANTS)


@pytest.fixture(scope="module")
def bloque_country_pools(constantes) -> tuple[str, tuple[str, ...]]:
    """Devuelve (comentario que precede al dict, claves declaradas dentro del dict)."""
    i = constantes.index("COUNTRY_POOLS: dict[str, dict[str, list]] = {")
    # El comentario es el bloque contiguo de líneas `#` inmediatamente anterior.
    lineas = constantes[:i].splitlines()
    comentario = []
    for linea in reversed(lineas):
        if linea.lstrip().startswith("#"):
            comentario.append(linea)
        elif linea.strip() == "":
            continue
        else:
            break
    j = constantes.index("\n}", i)
    claves = tuple(re.findall(r'^\s{4}"([A-Z]{2})":', constantes[i:j], re.M))
    return "\n".join(reversed(comentario)), claves


def test_el_dict_sigue_teniendo_pools(bloque_country_pools):
    """Sanity: sin claves, la paridad de abajo pasaría por vacuidad. Es el modo de fallo que esta
    ola pagó tres veces — un guard que no puede fallar es una coartada."""
    _, claves = bloque_country_pools
    assert len(claves) >= 5, f"COUNTRY_POOLS bajó a {len(claves)} países: {claves}"


def test_el_comentario_no_declara_un_unico_pais_con_pool(bloque_country_pools):
    """El defecto literal: «Solo ES tiene pool propio hoy» sobre un dict de cinco entradas."""
    comentario, _ = bloque_country_pools
    assert not re.search(r"[Ss]olo\s+[A-Z]{2}\s+tiene pool", comentario), (
        "el comentario volvió a declarar un único país con pool propio"
    )


def test_el_comentario_declara_su_cobertura_en_una_linea_comprobable(bloque_country_pools):
    """La paridad necesita una FRASE que declare, no un bloque de prosa donde buscar. Sin esta
    línea el test de abajo no tiene qué leer y pasaría por vacuidad."""
    comentario, _ = bloque_country_pools
    assert re.search(r"^#\s*COBERTURA HOY:.*$", comentario, re.M), (
        "desapareció la línea `# COBERTURA HOY:` del comentario de COUNTRY_POOLS. Es la única "
        "afirmación estructurada del bloque y de ella cuelga la paridad"
    )


def test_la_cobertura_declarada_coincide_exactamente_con_los_pools(bloque_country_pools):
    """La paridad, que es lo que de verdad protege: un sexto pool sin documentar falla aquí, y
    también falla documentar un país cuyo pool ya no existe.

    DOS TRAMPAS QUE ESTE TEST YA PAGÓ, las dos en su propia escritura:

    1. **Por subcadena.** La primera versión hacía `c not in comentario` y declaró que sólo faltaba
       MX: `US` ⊂ «USDA-sourced», `ES` ⊂ «RESUELVE-BIEN», `PR` ⊂ «propio» y `CO` ⊂ «cocina».
       Cuatro de cinco pasaban por accidente sobre un comentario que no nombraba a ninguno. Un
       código de país de dos letras es el sustrato perfecto para la colisión que este repo lleva
       pagando desde `"sal"⊂"salsa"`.

    2. **Sobre el bloque entero.** Corregido a `\\b`, la mutación lo destapó igual: la prosa que
       EXPLICA el arreglo nombra «MX/CO/PR/US», así que satisfacía la paridad aunque la línea de
       cobertura hubiera perdido un país. Comentario-vence-guard por novena vez. Por eso se lee
       **sólo la línea que declara**: la explicación puede decir lo que necesite sin poner en verde
       lo que no lo está."""
    comentario, claves = bloque_country_pools
    linea = re.search(r"^#\s*COBERTURA HOY:(.*)$", comentario, re.M).group(1)
    declarados = set(re.findall(r"\b([A-Z]{2})\b", linea))
    assert declarados == set(claves), (
        f"la cobertura declarada {sorted(declarados)} no coincide con los pools reales "
        f"{sorted(set(claves))}. El comentario es lo que lee quien decide si el camino degradado "
        f"de un país cae al pool dominicano"
    )
