"""[P1-APEX-PRECACHE-BLIND + P1-APEX-ENTRY-DIET · 2026-08-14] Lo que el apex
descarga sin poder ejecutarlo.

DOS MEDICIONES DEL MISMO DÍA, sobre el bundle real:

  · Precache del apex: 119 entradas / 2.252 KiB raw / **721,7 KiB por la red**.
    237,0 KiB gz de eso —un tercio— eran tres chunks que la portada tiene
    PROHIBIDO ejecutar: `@sentry-internal/replay` (115,5), `@neondatabase/auth`
    + zod (87,1) y la cadena unified/micromark (34,4). Tras el fix: 485,4 KiB gz.

  · Entry síncrono: 1.146.913 B de fuente, de los cuales `@sentry/*` eran
    **427.010 B = 37,2%** — en el recurso #1 del critical path de una página de
    marketing. Tras diferir el `init`: 730.194 B y 0% de Sentry, 86,5 → 53,6 kB gz.

POR QUÉ HACEN FALTA GUARDS PARSER-BASED Y NO BASTAN LOS TESTS DE JS. Las dos
propiedades que sostienen estos arreglos no son de comportamiento — son de
CABLEADO, y se rompen en silencio:

  · Si alguien quita `manifestTransforms` de `vite.config.js`, no falla ningún
    test de unidad: simplemente vuelven los 237 KiB.
  · Si alguien vuelve a importar `@sentry/react` desde un módulo eager (los error
    boundaries y `analytics.js` lo hacían), los 427.010 B regresan al entry sin
    que cambie una sola aserción de comportamiento.

Ambos son fallos donde el sistema sigue funcionando perfectamente y sólo se
vuelve más caro, que es exactamente la clase de defecto que nadie reporta.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND = _REPO_ROOT / "frontend"

_VITE_CONFIG = _FRONTEND / "vite.config.js"
_SW = _FRONTEND / "src" / "custom-sw.js"
_GUARD = _FRONTEND / "scripts" / "precache-guard.mjs"
_AUDIENCE = _FRONTEND / "scripts" / "precacheAudience.mjs"
_PACKAGE_JSON = _FRONTEND / "package.json"
_SRC = _FRONTEND / "src"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- P1-APEX-PRECACHE-BLIND

def test_el_transform_del_manifest_sigue_cableado() -> None:
    """`manifestTransforms` es lo único que aplica las exclusiones.

    `globIgnores` no puede expresarlas: los chunks llevan hash de contenido y un
    patrón literal caducaría en el siguiente deploy, fallando en silencio.
    """
    cfg = _leer(_VITE_CONFIG)
    # Clave ACTIVA, no subcadena. La primera versión de este test usaba `in` y la
    # mutación de verificación —renombrar a `_manifestTransformsDesactivado`— pasó
    # limpiamente, porque el nombre nuevo contiene al viejo. Un `in` no distingue
    # «cableado» de «renombrado para desactivarlo», que es justo la forma en que
    # esto se apagaría de verdad.
    assert re.search(r"(?<![\w$])manifestTransforms\s*:", cfg), (
        "vite.config.js no tiene una clave `manifestTransforms:` activa. Sin ella "
        "vuelven al precache del apex los 237,0 KiB gz de replay + SDK de auth + "
        "markdown, y ningún test de comportamiento se entera."
    )
    assert "excluidosDelPrecache" in cfg, (
        "El transform ya no lee el Set que llena `generateBundle`. La cadena "
        "plugin → Set → transform es lo que convierte nombres hasheados en "
        "exclusiones; rota, el filtro queda vacío y no avisa."
    )
    assert "precacheAudiencePlugin" in cfg, (
        "El plugin `bioboros-precache-audience` no está registrado: nadie llena "
        "el Set y el transform filtra un conjunto vacío."
    )


def test_el_guard_de_peso_corre_en_el_build() -> None:
    """El guard es lo que descubre al SIGUIENTE intruso.

    Las exclusiones arreglan los tres de hoy. Sin el guard en `postbuild`, el
    cuarto crece igual de invisible que crecieron estos.
    """
    pkg = _leer(_PACKAGE_JSON)
    assert "precache-guard.mjs" in pkg, (
        "`scripts/precache-guard.mjs` salió del pipeline de build. Es la única "
        "pieza que hace VISIBLE el peso del precache: sin ella, el coste vuelve "
        "a ser un número que nadie tiene delante."
    )
    assert re.search(r'"postbuild"\s*:\s*"[^"]*precache-guard', pkg), (
        "El guard existe pero ya no cuelga de `postbuild`; un guard que no corre "
        "es documentación."
    )


def test_filtro_por_host_sincronizado_entre_sw_y_guard() -> None:
    """La lista de chunks app-only vive en DOS sitios y no puede derivar.

    `custom-sw.js` la usa para filtrar en el navegador; `precache-guard.mjs` la
    replica para medir lo mismo que se despliega. Si divergen, el guard aprueba
    un precache distinto del real — que es peor que no tener guard, porque da
    una garantía falsa.
    """
    sw = _leer(_SW)
    guard = _leer(_GUARD)

    def nombres(texto: str, variable: str) -> set[str]:
        m = re.search(rf"{variable}\s*=\s*/\(\?:\^\|\\/\)\(([^)]+)\)", texto)
        assert m, f"No encuentro la regex `{variable}`; si la renombraste, actualizá este test."
        return set(m.group(1).split("|"))

    del_sw = nombres(sw, "_APP_ONLY_CHUNKS")
    del_guard = nombres(guard, "APP_ONLY")

    assert del_sw == del_guard, (
        "Deriva del filtro por host.\n"
        f"  sólo en custom-sw.js:      {sorted(del_sw - del_guard)}\n"
        f"  sólo en precache-guard.mjs:{sorted(del_guard - del_sw)}\n"
        "Actualizá las dos: el guard tiene que medir exactamente el precache que "
        "el Service Worker va a construir."
    )


def test_cada_familia_excluida_declara_su_gate() -> None:
    """Una exclusión sin el gate escrito es una afirmación sin respaldo.

    El gate («¿qué garantiza que el apex no ejecuta esto?») es lo que permite
    comprobar meses después si la exclusión sigue siendo cierta.
    """
    aud = _leer(_AUDIENCE)
    m = re.search(r"FAMILIAS_NO_PRECACHEABLES\s*=\s*\[(.*?)\n\];", aud, re.S)
    assert m, "No encuentro `FAMILIAS_NO_PRECACHEABLES` en precacheAudience.mjs."
    cuerpo = m.group(1)

    ids = re.findall(r"id:\s*'([^']+)'", cuerpo)
    gates = re.findall(r"gate:\s*'([^']+)'", cuerpo)
    assert ids, "La lista de familias quedó vacía o cambió de formato."
    assert len(ids) == len(gates), (
        f"{len(ids)} familias pero {len(gates)} gates declarados. Toda familia "
        "excluida tiene que decir QUÉ garantiza que el apex no la ejecuta."
    )


def test_no_se_vuelve_a_clasificar_por_dominancia() -> None:
    """La regla es por MARCADOR, no por volumen — y la diferencia ya costó un intento.

    El primer diseño midió «≥50% de los módulos del chunk son de la familia» y no
    atrapó el chunk de replay: Rollup mete ahí medio `@sentry/core` y el core
    diluía la proporción. En un chunk mixto, la parte pesada y la parte que lo
    identifica no son la misma.
    """
    aud = _leer(_AUDIENCE)
    assert "familiaMarcada" in aud, (
        "`familiaMarcada` desapareció. Si volviste a una métrica de volumen, lee "
        "el comentario de FAMILIAS_NO_PRECACHEABLES: ya se probó y no atrapa el "
        "chunk de replay."
    )
    assert "marcadores" in aud, "Las familias ya no declaran `marcadores`."


# ------------------------------------------------------------------ P1-APEX-ENTRY-DIET

def test_una_sola_puerta_a_sentry_en_todo_el_arbol() -> None:
    """`@sentry/*` sólo se importa desde `utils/sentryBoot.js`.

    ESTA es la propiedad que sostiene los −32,9 kB gz del entry, y la que se
    rompe sin que falle nada visible. Antes había CINCO puertas —main.jsx,
    GlobalErrorBoundary, RouteErrorBoundary, analytics.js y AgentPage— y bastaba
    con que UNA siguiera abierta para que `@sentry/core` volviera al chunk
    síncrono: los boundaries y `analytics.js` son eager.

    El import DINÁMICO de `main.jsx` (`await import('@sentry/react')` para las
    integraciones) es legítimo y no cuenta: no ata nada al entry.
    """
    permitidos = {"src/utils/sentryBoot.js"}
    infractores: list[str] = []

    for ruta in _SRC.rglob("*"):
        if ruta.suffix not in {".js", ".jsx"}:
            continue
        rel = ruta.relative_to(_FRONTEND).as_posix()
        if "__tests__" in rel or rel in permitidos:
            continue
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        # Sólo imports ESTÁTICOS: `from '@sentry/...'`.
        if re.search(r"""from\s+['"]@sentry/""", texto):
            infractores.append(rel)

    assert not infractores, (
        "Import estático de `@sentry/*` fuera de `utils/sentryBoot.js`:\n  "
        + "\n  ".join(infractores)
        + "\n\nCada uno devuelve 427.010 B de fuente al entry síncrono si el "
        "módulo es eager (o alcanzable desde uno). Usá la fachada "
        "`utils/observability.js`, que además ENCOLA lo que llegue antes del "
        "init en vez de perderlo."
    )


def test_la_fachada_no_importa_sentry() -> None:
    """`utils/observability.js` es eager: una arista suya a Sentry lo deshace todo.

    La importan los dos error boundaries y `analytics.js`. Si adquiere un import
    de `@sentry/*`, arrastra el SDK al entry por la puerta de atrás y el test de
    arriba —que exceptúa a `sentryBoot`— no lo vería.
    """
    fachada = _leer(_SRC / "utils" / "observability.js")
    # Se busca el IMPORT, no la mención: el fichero explica en prosa por qué no
    # importa Sentry, y prohibir la palabra prohibiría documentar la decisión.
    # (Primera versión de este test buscaba la subcadena suelta y fallaba contra
    # su propio comentario — el guard acusaba a la explicación de ser el defecto.)
    imports = re.findall(r"""(?:from|import\s*\()\s*['"](@sentry[^'"]*)['"]""", fachada)
    assert not imports, (
        f"`utils/observability.js` importa {imports}. Su razón de existir es NO "
        "tener @sentry en su grafo: es el módulo eager que usan los dos error "
        "boundaries y `analytics.js`, así que una arista aquí devuelve el SDK al "
        "entry por la puerta de atrás."
    )


def test_la_cola_de_errores_tempranos_sigue_existiendo() -> None:
    """Diferir el init sin encolar no es optimizar: es quedarse ciego al arranque.

    La cola —y el arranque inmediato ante el primer error— es el precio pagado
    por los −32,9 kB. Quitarla dejaría el ahorro y perdería los errores de boot,
    que son los que más duele perder.
    """
    fachada = _leer(_SRC / "utils" / "observability.js")
    for pieza, porque in [
        ("addEventListener('error'", "sin handler temprano no hay nada que encolar"),
        ("unhandledrejection", "las promesas rotas del boot son la mitad de los casos"),
        ("registrarArranqueSentry", "es lo que arranca el SDK ante el primer error"),
        ("MAX_EN_COLA", "una cola sin tope es una fuga con otro nombre"),
    ]:
        assert pieza in fachada, f"Falta `{pieza}` en la fachada: {porque}."
