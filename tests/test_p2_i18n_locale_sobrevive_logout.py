"""[P2-I18N-LOCALE-SOBREVIVE-LOGOUT + 5 · 2026-08-22] La segunda tanda de la ola final.

Ancla seis gaps. Lleva el nombre del marcador que bumpea `_LAST_KNOWN_PFIX`.

LO QUE ESTA TANDA ENSEÑA

1. EL ARREGLO OBVIO NO ERA EL ARREGLO. El plan proponía borrar `mealfit_locale` en
   `_clearUserScopedCaches`, y advertía que costaba: de sus SEIS llamadores, dos no son
   cambio de usuario (sesión expirada, entrada en invitado), y ahí el borrado le quita el
   idioma a quien no ha cambiado de cuenta.

   Midiendo salió que el daño no está en heredar la preferencia: está en ESTAMPARLA. Cuando
   el perfil del recién llegado trae `locale = NULL` —lo normal desde que la migración quitó
   el DEFAULT—, `P1-I18N-PROFILE-DEFAULT-PISA` le escribe el idioma ACTIVO. Su comentario
   dice «el que la autodetección puso y el usuario no cambió», y eso es cierto en un
   dispositivo estrenado; en uno COMPARTIDO el activo es la elección del usuario anterior.
   Desde el perfil viaja a todos los dispositivos del recién llegado: una elección que nunca
   hizo se vuelve su preferencia permanente.

   Se distingue el ORIGEN: una preferencia autodetectada no tiene dueño y se puede estampar;
   una elegida a mano lo lleva, y no se hereda.

2. EL ORDEN ES LOAD-BEARING. Reclamar va ANTES de sincronizar. Al revés, el sello del
   usuario anterior sigue puesto cuando corre la comprobación y ésta descartaría el idioma
   que el perfil ACABA de aplicar — el arreglo saldría peor que el defecto.

3. UN AVISO QUE PROMETE LO QUE LA APP INCUMPLE SOLA. Si el PATCH del idioma falla, el toast
   decía que el cambio «queda en este dispositivo». No queda: al volver a entrar,
   `syncLocaleFromProfile` lee el perfil —que conserva el idioma anterior, porque el PATCH
   es justo lo que falló— y lo revierte. El idioma vuelve sin explicación y el siguiente
   diagnóstico del usuario es «el selector no funciona».

4. UN GUARD QUE SÓLO MIRA INGLÉS NO VIGILA EL IDIOMA: VIGILA EL INGLÉS. Los `val` de los
   chips son los identificadores con los que resuelven `pantry_names_match`, el guard de
   coherencia y el backstop de alergias. Su red comparaba sólo contra `en-US`, así que
   `val: "Onion"` fallaba y `val: "Oignon"` pasaba limpio.

5. LA REGLA QUE FALTABA, NO LA MIGRACIÓN. `P2-I18N-GATE-SIN-REGLA-FORMATO-CLAVADO` sonaba a
   deuda pendiente; medido, en `src/` quedan DIEZ apariciones de un locale clavado y NUEVE
   son comentarios que documentan el arreglo. Código real: **una**, y en una superficie
   sólo-español por alcance. Lo que faltaba era la regla — y tuvo que saltarse los
   comentarios, o la documentación del arreglo dispararía el guard que el arreglo instaló.

tooltip-anchor: P2-I18N-LOCALE-SOBREVIVE-LOGOUT
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONT = _ROOT / "frontend"
_MARKER = "P2-I18N-LOCALE-SOBREVIVE-LOGOUT"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _sin_comentarios(src: str) -> str:
    """Sólo las líneas cuyo primer token es `//`.

    No se filtra todo lo que siga a un `//` en cualquier posición: un `//` dentro de una
    cadena (una URL) haría que el filtro se comiera código real, y en una aserción de tipo
    «esto ya no aparece» comerse código es un falso VERDE.
    """
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))


# ───────────────────────── 1. el idioma no se hereda entre cuentas ─────────────────

def test_existe_el_sello_de_dueno_de_la_preferencia():
    src = _leer(_FRONT / "src" / "i18n" / "index.js")
    assert "export async function claimLocaleForUser(" in src, (
        "desapareció `claimLocaleForUser`: el idioma vuelve a heredarse entre cuentas en "
        f"un dispositivo compartido, y el estampado lo graba en el perfil ajeno [{_MARKER}]"
    )
    assert "mealfit_locale_owner" in src, "se perdió la clave del dueño"


def test_reclamar_va_antes_de_sincronizar():
    """El orden es load-bearing (punto 2 del docstring)."""
    src = _leer(_FRONT / "src" / "context" / "AssessmentContext.jsx")
    codigo = _sin_comentarios(src)
    i_claim = codigo.find("claimLocaleForUser(userId)")
    i_sync = codigo.find("syncLocaleFromProfile(data.locale)")
    assert i_claim != -1, f"nadie reclama el idioma para el usuario [{_MARKER}]"
    assert i_sync != -1, "desapareció la sincronización con el perfil"
    assert i_claim < i_sync, (
        "`syncLocaleFromProfile` corre ANTES de `claimLocaleForUser`. En ese orden el sello "
        "del usuario anterior sigue puesto cuando corre la comprobación, y ésta descarta el "
        f"idioma que el perfil ACABA de aplicar [{_MARKER}]"
    )


def test_el_estampado_usa_lo_reclamado_y_no_el_activo_crudo():
    codigo = _sin_comentarios(_leer(_FRONT / "src" / "context" / "AssessmentContext.jsx"))
    bloque = codigo[codigo.index("if (!data.locale) {"):][:600]
    assert "getLocale()" not in bloque, (
        "el estampado volvió a leer el idioma ACTIVO en crudo. En un dispositivo "
        "compartido eso es la elección del usuario ANTERIOR, y acaba en el perfil del "
        f"recién llegado [{_MARKER}]"
    )
    assert "_activo" in bloque


# ───────────────────────── 2. el aviso dice la verdad ──────────────────────────────

def test_el_aviso_del_idioma_no_promete_persistencia_local():
    codigo = _sin_comentarios(_leer(_FRONT / "src" / "pages" / "Settings.jsx"))
    assert "Idioma cambiado en este dispositivo, pero no se pudo guardar" not in codigo, (
        "volvió el aviso que promete que el cambio queda en el dispositivo. No queda: el "
        "siguiente arranque lo revierte desde el perfil, que conserva el idioma anterior "
        f"precisamente porque el PATCH es lo que falló [{_MARKER}]"
    )
    assert "Volverá al anterior la próxima vez que entres" in codigo


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_el_aviso_nuevo_esta_traducido(locale: str):
    import json
    cat = json.loads(_leer(_FRONT / "src" / "i18n" / "locales" / f"{locale}.json"))
    clave = ("Idioma cambiado, pero no se pudo guardar en tu cuenta. "
             "Volverá al anterior la próxima vez que entres.")
    assert cat.get(clave), f"{locale} no traduce el aviso nuevo"


# ───────────────────────── 3. la costura del nombre en el aviso ────────────────────

def test_el_aviso_de_registro_usa_el_nombre_que_el_usuario_ve():
    """`result.meal_name` sigue llegando canónico y sigue siendo lo que el motor resuelve;
    simplemente no se PINTA. Traducirlo en el servidor cruzaría la frontera."""
    codigo = _sin_comentarios(_leer(_FRONT / "src" / "pages" / "Dashboard.jsx"))
    assert "t('{plato} registrado', { plato: result.meal_name })" not in codigo, (
        "el aviso volvió a pintar el nombre canónico del servidor mientras la tarjeta "
        f"muestra el traducido: dos nombres para lo mismo en la misma pantalla [{_MARKER}]"
    )
    assert "mealDisplayName(meal, _dashLocale)" in codigo
    assert "result.meal_name" in codigo, (
        "se dejó de recibir el nombre canónico. Es el respaldo cuando el plato no tiene "
        "traducción, y quitarlo deja el aviso sin nombre"
    )


# ───────────────────────── 4. la regla de los locales clavados ─────────────────────

def test_el_gate_prohibe_un_locale_clavado_en_un_formateador():
    src = _leer(_FRONT / "scripts" / "i18n-check.mjs")
    assert "LOCALE_CLAVADO" in src, (
        "desapareció la regla: un `toLocaleString('es-DO')` nuevo vuelve a entrar con el "
        f"gate en verde [{_MARKER}]"
    )
    assert "FORMATO_EXENTO" in src
    assert "pages/SupermarketPage.jsx" in src, (
        "se perdió la exención de la página del supermercado, que es superficie "
        "sólo-español por alcance. Sin ella el gate se pone rojo por un fichero correcto, "
        "y un falso rojo enseña a apagar el gate"
    )


def test_la_regla_se_salta_los_comentarios():
    """Si no, la documentación del arreglo dispara el guard que el arreglo instaló."""
    src = _leer(_FRONT / "scripts" / "i18n-check.mjs")
    bloque = src[src.index("const LOCALE_CLAVADO"):][:2000]
    assert "startsWith('//')" in bloque, (
        "la regla dejó de saltarse los comentarios. En `src/` hay NUEVE menciones en "
        f"comentarios y UNA en código: sin el filtro, el gate reporta diez [{_MARKER}]"
    )


# ───────────────────────── 5. el techo de hilos del enriquecimiento ────────────────

def _codigo_py(src: str) -> str:
    """Sólo las líneas que no son comentario de Python.

    Comentario-vence-guard, TERCERA dirección y tres veces en este mismo test: la prosa que
    explica por qué el semáforo es `BoundedSemaphore` y por qué el `acquire` no bloquea
    contiene literalmente las dos cadenas que el guard busca, así que las tres mutaciones
    salían VERDES con el defecto puesto. Un guard satisfecho por su propia documentación es
    un guard inerte.
    """
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


def test_el_enriquecimiento_tiene_techo_global_de_hilos():
    src = _codigo_py(_leer(_ROOT / "backend" / "plan_display_i18n.py"))
    # Con `=` detrás: `_INFLIGHT_SEMAPHORE` es subcadena de `_INFLIGHT_SEMAPHORE_X`, y
    # renombrarlo así dejaba el guard verde. Vigésima vez que una subcadena sin frontera
    # me cuesta un guard inerte.
    assert "_INFLIGHT_SEMAPHORE = " in src, (
        "desapareció el techo: cada plan arranca su propio hilo y cada hilo puede vivir "
        f"20-29 minutos hablando con un proveedor pago [{_MARKER}]"
    )
    assert "BoundedSemaphore" in src, (
        "el semáforo dejó de ser acotado. Sin el acotado, un `release()` de más sube el "
        "techo en silencio para siempre — un techo que se relaja solo no es un techo"
    )
    assert "acquire(blocking=False)" in src, (
        "el `acquire` pasó a bloquear: congelaría el hilo del request que programa el "
        "enriquecimiento. Esto es una conveniencia, y una conveniencia no bloquea a quien "
        "la pide"
    )
    cuerpo = src[src.index("_INFLIGHT_SEMAPHORE.acquire"):]
    assert "_INFLIGHT_SEMAPHORE.release()" in cuerpo, (
        "no se suelta el permiso. Sin el `finally`, una excepción convierte el techo en un "
        "candado permanente y la feature se apaga sola tras N fallos"
    )


# ───────────────────────── 6. la exclusión que fallaba en silencio ─────────────────

def test_la_compra_cuenta_lo_excluido_y_no_lo_pedido():
    src = _leer(_ROOT / "backend" / "tools.py")
    assert "len(_excl_casadas)" in src, (
        "el mensaje volvió a contar `len(excluded_items)`, que es lo que el usuario PIDIÓ. "
        "Con cero coincidencias decía «se excluyeron 3» — que es lo que pasa cuando el "
        f"usuario chatea en otro idioma y el coach emite «Avocado» [{_MARKER}]"
    )
    assert "NO encontré en la lista de" in src, (
        "se dejó de avisar de los ítems que no casaron. Un no-op indistinguible del éxito "
        "es peor que un fallo declarado: el usuario cree que excluyó algo que sigue ahí"
    )
