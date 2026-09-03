"""[P1-I18N-BACKEND-DETAIL · 2026-08-21] El `detail` español del servidor ganaba sobre el
fallback traducido.

El patrón era `toast.error(data?.detail || t('…'))`. El `||` hace que el texto **español
del servidor** se pinte siempre que exista, y el fallback traducido sólo se vea cuando el
backend **no** explica qué pasó. O sea: la traducción estaba escrita, revisada y presente
en los cuatro catálogos — y era exactamente la rama que no llegaba nunca.

Y el gate lo daba verde, porque la clave existe y está traducida. **Nadie medía cuál de
las dos ramas del `||` gana.** Ese es el hueco que este fichero cierra: un test de
cobertura no puede verlo, porque el defecto no está en el catálogo sino en el operador.

MEDIDO: 22 usos de `?.detail ||` en `frontend/src`, de los cuales **5** en posición
inequívoca de copy (`toast.error(...)`, `description:`) y 10 en `throw new Error(...)`.

POR QUÉ NO BASTA CON INVERTIR EL `||`: `t('…') || data.detail` pintaría siempre el
fallback y tiraría lo que el servidor sí sabe —«te faltan 3 ingredientes» degradado a
«inténtalo de nuevo»—. Lo correcto es traducir lo que el servidor sabe, y para eso el
canal es el CÓDIGO, no la prosa. El backend ya emite ocho `error_code` canónicos.

LO QUE NO ENTRA, y por qué no es un olvido: los 10 `throw new Error(detail || …)`. Ahí el
string viaja a un `catch` cuyo destino varía —unos lo pintan, otros lo registran, otros
sólo miran `err.code`—, así que migrarlos a ciegas cambiaría comportamiento que nadie ha
medido. Se documentan aquí como deuda con nombre en vez de dejarlos parecer cubiertos.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MARKER = "P1-I18N-BACKEND-DETAIL"

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_SRC = _ROOT / "frontend" / "src"

# `toast.error(…)` y `description:` son posición de COPY sin ambigüedad: lo que va ahí lo
# lee el usuario. `throw new Error(…)` no lo es — ver el docstring.
#
# [P1-I18N-SERVER-COPY-GANA · 2026-08-22] EL CANAL YA NO ES `detail`, ES LA PROPIEDAD.
#
# La versión anterior anclaba la grafía `detail`, y por eso medía un canal de cuatro.
# MEDIDO sobre `frontend/src` el 2026-08-22, la clase completa era:
#
#     detail ||                  7 sitios      <- los únicos que el guard veía
#     message ||                28 sitios
#     error_message ||           5 sitios
#     ai_interrupted_message ||  1 sitio
#
# 41 en total, 26 de ellos en posición de copy visible. Y `mensajeDeError` —la
# herramienta correcta, escrita el día antes— se usaba en 6.
#
# El caso que lo prueba: `AssessmentContext.jsx` hacía
#
#     if (data.error_code === 'pantry_insufficient_for_goal') {
#         toast.error(t('Faltan ingredientes en tu Nevera'), {
#             description: data.error_message || t('Tu Nevera no alcanza…'),
#
# El call site YA SABÍA el código, `COPY_POR_CODIGO` YA tenía ese código, y aun así
# pintaba el español del servidor debajo de un título traducido. La traducción existía,
# estaba revisada, y era exactamente la rama que no llegaba nunca.
#
# Es la regla que este repo sacó de tres fallos en un día: si el guard puede expresarse
# por la propiedad, que no dependa de cómo se escriba. La propiedad aquí es «una
# expresión que viene del servidor gana sobre un fallback traducido», y el nombre del
# campo es un detalle de implementación del endpoint que la emite.
# [P1-I18N-SERVER-COPY-GANA-SIGUE-ABIERTO · 2026-08-23] Y aun así seguía siendo una
# ENUMERACIÓN. El cierre del 22-ago ensanchó de 1 canal a 5 y arregló los números de línea,
# pero no convirtió el guard en una propiedad: un campo nuevo entra invisible. MEDIDO con el
# regex de entonces, mutándolo sobre el árbol de hoy:
#
#     False | description: generatedPlan?._review_disclaimer || t('El plan se ajusto...')
#     False | toast.error(data?.motivo || t('Algo fallo.'))
#     True  | toast.error(data?.detail || t('Algo fallo.'))
#
# El peor de los invisibles es `_review_disclaimer` (`Plan.jsx`), que el backend compone
# SIEMPRE en español: en la rama crítica el usuario lee el título traducido «Plan ajustado
# por seguridad médica» y debajo, en español, «El sistema detectó violaciones críticas
# (alergias o condiciones médicas)…».
#
# Ahora el canal es CUALQUIER campo. Lo que acota el guard deja de ser una lista de nombres
# y pasa a ser lo que de verdad define el defecto: el RECEPTOR es payload del servidor y la
# posición es copy visible. La lista blanca de abajo es de campos que son DATO y no prosa —
# razonada uno a uno, no por conveniencia.
_CANALES_DEL_SERVIDOR = r"(?!(?:" + r"|".join([
    # Datos, no prosa: pintarlos crudos es correcto porque no son texto que traducir.
    "length", "size", "count", "id", "status", "code", "name", "url", "type",
]) + r")\b)[A-Za-z_$][\w$]*"

# El RECEPTOR importa tanto como el canal. `e.message` / `err.message` dentro de un `catch`
# NO es «el español del servidor»: es lo que puso el `throw`, y si ese throw ya tradujo,
# reemplazarlo por `mensajeDeError` PIERDE información (devolvería el fallback genérico
# porque un `Error` no lleva `error_code`). El defecto vive en el THROW, no en el catch.
#
# Por eso el guard mira los receptores que son PAYLOAD del servidor (`data`, `result`,
# `body`, `status`, `newMealData`…) y deja fuera los nombres canónicos de variable de
# excepción. Es una distinción de significado, no una excepción por conveniencia: los
# throws tienen su propio test más abajo.
_RECEPTOR_DE_EXCEPCION = re.compile(r"\b(?:e|err|error|ex|_e|_err)$")
_POSICION_DE_COPY = re.compile(
    # `(?:\?\.|\.)` y no `\?\.?`: lo segundo exige el signo de interrogacion, asi que
    # `err.detail ||` —que existe en el arbol— era invisible. Lo cazo la mutacion de
    # control; el test principal estaba pasando sin ver una de las dos formas.
    # [P1-I18N-CONSENT-MODAL-SERVIDOR-GANA · 2026-08-23] `message:` como tercera posición
    # de copy. El modal «Tu Nevera no alcanza» recibía `message: newMealData.message || '…'`
    # en un `return` de AssessmentContext —ni toast ni description— y este guard no lo veía.
    # Es copy visible igual: el Dashboard lo pinta tal cual en el modal.
    r"(?:toast\.(?:error|warning|success)?\s*\(|description:\s*|message:\s*)[^;\n]*?"
    r"(?P<receptor>[A-Za-z_$][\w$]*)(?:\?\.|\.)" + _CANALES_DEL_SERVIDOR
    # [P1-I18N-SERVER-COPY-GANA-SIGUE-ABIERTO · 2026-08-23] El fallback tiene que ser una
    # TRADUCCIÓN. Al abrir el canal a cualquier campo apareció
    # `description: p.description || ''` (SupermarketPage), que es un campo de FORMULARIO y
    # no copy: no hay ninguna traducción a la que el servidor le gane.
    #
    # Exigir `t(` cerca del `||` es lo que define el defecto de verdad —«el español del
    # servidor gana sobre un fallback traducido»— en vez de aproximarlo por el nombre del
    # campo. La ventana de 200 cubre el ternario de `Plan.jsx:1281`, cuyo fallback es
    # `|| (Number.isFinite(…) ? t(…) : t(…))`.
    + r"\s*\|\|[^;]{0,200}?\bt\(",
    re.S,
)


def _ficheros():
    if not (_ROOT / "backend").is_dir() or not _SRC.is_dir():
        pytest.skip(f"{_ROOT} no es la raíz del repo (¿worktree?)")
    return [p for p in list(_SRC.rglob("*.jsx")) + list(_SRC.rglob("*.js"))
            if "__tests__" not in p.parts]


def _sin_comentarios(js: str) -> str:
    """Un `detail ||` citado en un comentario no es código.

    Este repo lleva siete guards derrotados por prosa en dos días, varios con el
    comentario escrito por quien escribía el guard.

    [P1-I18N-SERVER-COPY-GANA · 2026-08-22] LA SUSTITUCIÓN PRESERVA LA LONGITUD. Antes
    cada comentario colapsaba a UN espacio, así que el texto que se analiza tiene menos
    saltos de línea que el original y **los números de línea reportados no eran los del
    fichero**: al ensanchar el guard salieron tres «violaciones» en `SupermarketPage.jsx`
    apuntando a líneas que no contienen el patrón (una era `} catch (err) {`). Un guard
    que acusa a la línea equivocada hace perder más tiempo del que ahorra, y es la misma
    familia de `ast.col_offset` cuenta BYTES que este repo ya tiene registrada.
    """
    def _blanquear(m: re.Match) -> str:
        # Se conservan los saltos de línea; todo lo demás pasa a espacios.
        return "".join("\n" if c == "\n" else " " for c in m.group(0))

    js = re.sub(r"/\*.*?\*/", _blanquear, js, flags=re.S)
    return re.sub(r"//[^\n]*", _blanquear, js)


# ============================================================
# 1 · Ninguna posición de copy pinta el `detail` crudo
# ============================================================

def test_ninguna_posicion_de_copy_pinta_el_detail_del_servidor() -> None:
    culpables = []
    for p in _ficheros():
        s = _sin_comentarios(p.read_text(encoding="utf-8"))
        for m in _POSICION_DE_COPY.finditer(s):
            if _RECEPTOR_DE_EXCEPCION.search(m.group("receptor")):
                continue  # `e.message` en un catch: el defecto vive en el throw
            # [P1-I18N-CONSENT-MODAL-SERVIDOR-GANA · 2026-08-23] Al abrir `message:` como
            # posición de copy aparecieron tres sitios en `authClient.js` que NO son
            # víctimas: van bajo el contrato `mfCopy` (`mfCopy: !data?.message`), y
            # `humanizeAuthError` sólo respeta el texto crudo cuando `mfCopy` es true — o
            # sea, cuando GANÓ el fallback traducido. Un `message` del servidor de auth pasa
            # por los heurísticos y se traduce por clase. Es un contrato de dos piezas que
            # una línea sola no enseña; se reconoce por la PROPIEDAD (el `mfCopy` a
            # continuación), no por el nombre del fichero.
            cola = s[m.end():m.end() + 200]
            if re.search(r"\bmfCopy\s*:", cola):
                continue
            linea = s[:m.start()].count("\n") + 1
            culpables.append(f"{p.relative_to(_SRC).as_posix()}:{linea}")

    assert not culpables, (
        "Estos sitios pintan el `detail` del servidor —que viene en ESPAÑOL siempre— y "
        f"dejan el fallback traducido como rama muerta: {culpables}. Usa "
        f"`mensajeDeError(data, t('…'), t)`: traduce por CÓDIGO lo que el servidor sabe y "
        f"manda el `detail` crudo a la consola, no a la cara del usuario. [{_MARKER}]"
    )


def test_el_detector_veria_el_patron_original() -> None:
    """MUTACIÓN DE CONTROL. Si el regex dejara de casar, el test de arriba daría verde
    pasando en vacío — el modo de fallo de `P1-CULINARY-METADATA-BETA`."""
    for muestra in (
        "toast.error(data?.detail || t('Algo falló.'));",
        "toast.error(err.detail || t('Algo falló.'));",
        "  description: data?.detail || t('Reintenta.'),",
    ):
        assert _POSICION_DE_COPY.search(muestra), f"el detector no ve: {muestra!r}"


def test_el_detector_no_marca_un_throw() -> None:
    """La otra mitad del control: si marcara los `throw`, el guard pediría migrar 10
    sitios cuyo destino varía, y un guard que pide lo que no se puede dar se desactiva."""
    assert not _POSICION_DE_COPY.search(
        "throw new Error(data?.detail || t('No se pudo registrar.'));"
    )


# ============================================================
# 2 · El helper traduce por código y no pierde el detalle
# ============================================================

def _errorcopy() -> str:
    p = _SRC / "utils" / "errorCopy.js"
    if not p.exists():
        pytest.fail(
            f"No existe `utils/errorCopy.js`. Sin él, cada call site vuelve a decidir "
            f"por su cuenta si pinta el español del servidor. [{_MARKER}]"
        )
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "codigo",
    [
        "ai_unavailable", "ai_exhausted_retries", "swap_ai_unavailable",
        "swap_llm_retries_exhausted", "swap_clinical_violation",
        "swap_strict_pantry_no_inventory", "pantry_insufficient_for_goal",
        "budget_insufficient", "budget_below_goal_floor",
    ],
)
def test_los_codigos_que_el_backend_emite_tienen_copy(codigo: str) -> None:
    """Los ocho `error_code` canónicos del backend más `budget_insufficient`, que viaja
    como `detail.code`. Si el backend añade uno y aquí no está, ese error cae al fallback
    genérico — degradación aceptable, pero el usuario pierde el motivo concreto."""
    assert f"{codigo}:" in _errorcopy(), (
        f"`{codigo}` no tiene copy traducible. El backend lo emite y el usuario recibirá "
        f"un mensaje genérico en su lugar. [{_MARKER}]"
    )


def test_el_copy_es_una_funcion_de_t_y_no_una_cadena() -> None:
    """LA TRAMPA DEL CONGELADO. Un `t('…')` evaluado en ámbito de módulo se resuelve al
    IMPORTAR y se queda en el idioma de arranque — y en es-DO parece correcto. Por eso la
    tabla guarda funciones."""
    src = _errorcopy()
    m = re.search(r"const COPY_POR_CODIGO = \{(.*?)\n\};", src, re.S)
    assert m, "no encontré la tabla de copy"
    cuerpo = m.group(1)
    assert "(t) => t(" in cuerpo, (
        f"la tabla guarda cadenas y no funciones de `t`: se congelarían en el idioma de "
        f"arranque. [{_MARKER}]"
    )
    for linea in cuerpo.strip().split("\n"):
        if ":" in linea and "t(" in linea:
            # [P2-I18N-PLAN-TOASTS-ERROR-MESSAGE · 2026-08-23] el contrato pasó a `(t, detail)`:
            # una entrada puede necesitar un DATO del servidor (`max`). Sigue siendo función de `t`.
            assert "(t) =>" in linea or "(t, d) =>" in linea, f"esta entrada no es función de `t`: {linea.strip()[:70]}"


def test_el_detail_crudo_no_se_pierde_va_a_la_consola() -> None:
    """El servidor SÍ sabe cosas que el cliente no. Traducir por código no puede
    significar tirar el diagnóstico — sólo sacarlo de la pantalla."""
    src = _errorcopy()
    assert "console.error" in src, (
        f"el `detail` sin código traducible se descarta en silencio. Tiene que ir a la "
        f"consola: los guards del repo preservan `console.error` en producción a "
        f"propósito, y Sentry lo recoge. [{_MARKER}]"
    )


def test_la_deuda_de_los_throw_esta_declarada() -> None:
    """Un alcance que se decide y no se anota vuelve como hallazgo en la siguiente
    auditoría."""
    src = _errorcopy()
    assert "throw new Error" in src and "catch" in src, (
        f"no está escrito por qué los `throw new Error(detail || …)` quedan fuera. Sin "
        f"esa razón, el siguiente auditor los cuenta como olvido — o los migra a ciegas. "
        f"[{_MARKER}]"
    )
