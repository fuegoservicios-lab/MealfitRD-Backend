# backend/cuentas_prueba.py
"""[P1-PLAN-LOTE-830 · 2026-09-29] Cuentas de prueba: quién las marca, quién sale y cuándo se abre su contenido.

Spec docs/superpowers/specs/2026-09-29-admin-cuentas-actividad-pruebas-design.md §3 y §13.4 (raíz del workspace).

Una cuenta de prueba es una cuenta que el dueño marca desde /admin → Cuentas con un motivo. Mientras la marca está viva,
el equipo puede ver su contenido (formulario, comidas, planes, conversaciones) y la propia persona lo sabe —la app se lo
dice una vez— y puede salir cuando quiera. Las reglas viven AQUÍ (SSOT): los routers y el script de consola solo las
llaman. La marca es una fila de `public.cuentas_de_prueba`; una sola viva por cuenta (índice único parcial).

Cuatro decisiones que no son obvias:

1. La marca se comprueba en CADA petición del detalle (`exigir_prueba`), sin caché: si la persona sale, la siguiente
   vista ya falla. Una marca se QUITA, nunca se borra: la tabla es su propio historial.
2. Sin correo previo (decisión del dueño), el contenido no se abre hasta que la persona vio el aviso en la app
   (`aviso_visto_at`, knob `MEALFIT_ADMIN_TEST_REQUIRE_NOTICE`): ese aviso es su notificación y llega ANTES de cualquier
   acceso a su contenido.
3. Salir (`salir`) funciona SIEMPRE, con el interruptor maestro apagado también: como revertir un regalo, dejar de ser
   vigilada nunca depende de un interruptor. Tampoco anota en `admin_access_log`: no es una acción del personal.
4. El interruptor maestro (`MEALFIT_ADMIN_TEST_ACCOUNTS`, apagado por defecto) lo cierra el router con un 404 en cada
   ruta del panel; aquí lo respetan `exigir_prueba` y `para_la_persona`. Las escrituras del admin no lo repiten (como
   `admin_cuentas.revocar`): el router ya no las deja llegar.

Escribir como admin (`marcar`, `quitar`): la fila de `admin_access_log` va ANTES de escribir y, si no se puede anotar, no
hay cambio (503). Si la escritura falla, queda `<accion>_fallo` al lado. El texto libre del rastro va SOLO bajo las claves
`motivo` y `error`: son las que `db_profiles.delete_account_data` quita del rastro al cerrar la cuenta (lote 841), así
que renombrarlas dejaría el motivo de una persona en un log que sobrevive a su cuenta.

`ErrorPrueba.detalle` es un código estable que el router devuelve tal cual: `ya_marcada`, `salio_ella`, `sin_marca`,
`no_existe`, `motivo`, `demasiadas`, `no_es_prueba`, `aviso_pendiente`. Los 5xx llevan una frase, no un código.
"""
from __future__ import annotations

import logging
import uuid

import regalos_cuenta as rc
from admin_acceso import registrar_acceso
from db import execute_sql_query, execute_sql_write
from knobs import _env_bool

logger = logging.getLogger(__name__)

MAX_LOTE = 100
_SIN_RASTRO = "No se pudo registrar la acción; no se hizo ningún cambio."
_NO_GUARDADA = "No se pudo guardar el cambio de la marca."
_NO_LEIDA = "No se pudo comprobar la marca de la cuenta; no se muestra nada."
# Lo que `marcar_varias` cuenta como resultado de UNA cuenta (el resto de errores corta el lote entero).
_RESULTADOS_DE_LOTE = frozenset({(409, "ya_marcada"), (409, "salio_ella"), (404, "no_existe")})


class ErrorPrueba(Exception):
    """Una operación sobre una cuenta de prueba que no procede: el router la devuelve con su código y su `detalle`."""

    def __init__(self, status: int, detalle: str):
        super().__init__(detalle)
        self.status = status
        self.detalle = detalle


def activo() -> bool:
    """Interruptor maestro: apagado, el panel no ve cuentas de prueba y la app no le enseña nada a nadie."""
    return _env_bool("MEALFIT_ADMIN_TEST_ACCOUNTS", False)


def requiere_aviso() -> bool:
    """Con True (default) el contenido de la cuenta espera a que la persona vea el aviso en la app (§13.4)."""
    return _env_bool("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", True)


def _uuid(v) -> str | None:
    """El id en forma canónica, o None si no es un uuid (un invitado no lo es): no se consulta la base con basura."""
    try:
        return str(uuid.UUID(str(v).strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def _motivo(v) -> str:
    m = " ".join(str(v or "").split())
    if not 3 <= len(m) <= 300:
        raise ErrorPrueba(422, "motivo")
    return m


def _anotar(admin_id, accion, user_id, detalle) -> None:
    try:
        registrar_acceso(admin_id, accion, user_id, detalle)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[P1-PLAN-LOTE-830] rastro no anotado ({accion} para {user_id}): {e!r}")
        raise ErrorPrueba(503, _SIN_RASTRO) from e


def _anotar_fallo(admin_id, accion, user_id, detalle) -> None:
    """El rastro ya dice `accion` (se anota ANTES de escribir); si el cambio no se hizo, deja `accion_fallo` al lado.
    Best-effort: el fallo ya está en el log."""
    try:
        registrar_acceso(admin_id, f"{accion}_fallo", user_id, detalle)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-830] rastro de fallo no anotado ({accion}_fallo para {user_id}): {e!r}")


def _registrar_fallo(admin_id, accion, user_id, marca_id, e) -> None:
    logger.error(f"[P1-PLAN-LOTE-830] {accion} no guardado para {user_id}: {e!r}")
    _anotar_fallo(admin_id, accion, user_id, {"marca_id": marca_id, "error": type(e).__name__})


def marca_viva(user_id) -> dict | None:
    """La marca viva de la cuenta (`{id, marcada_por, marcada_at, motivo, aviso_visto_at}`), o None. Sin caché."""
    uid = _uuid(user_id)
    if not uid:
        return None
    fila = execute_sql_query(
        "SELECT id::text AS id, marcada_por::text AS marcada_por, marcada_at, motivo, aviso_visto_at "
        "FROM public.cuentas_de_prueba WHERE user_id = %s AND quitada_at IS NULL LIMIT 1",
        (uid,), fetch_one=True)
    return dict(fila) if fila else None


def historial(user_id) -> list[dict]:
    """Todas las marcas que tuvo la cuenta, la más nueva primero: las quitadas también (quitar no borra)."""
    uid = _uuid(user_id)
    if not uid:
        return []
    filas = execute_sql_query(
        "SELECT id::text AS id, marcada_por::text AS marcada_por, marcada_at, motivo, aviso_visto_at, quitada_at, "
        "quitada_por::text AS quitada_por, quitada_por_la_persona, motivo_quitar "
        "FROM public.cuentas_de_prueba WHERE user_id = %s ORDER BY marcada_at DESC",
        (uid,), fetch_all=True)
    return [dict(f) for f in filas or []]


def _existe_perfil(uid: str) -> bool:
    fila = execute_sql_query("SELECT id::text AS id FROM public.user_profiles WHERE id = %s", (uid,), fetch_one=True)
    return bool(fila)


def _salio_ella(uid: str) -> bool:
    """¿La ÚLTIMA marca de la cuenta la quitó la propia persona? Solo cuenta la última: si una marca posterior la
    quitó un admin, ya no hay nada que confirmar."""
    fila = execute_sql_query(
        "SELECT quitada_por_la_persona FROM public.cuentas_de_prueba WHERE user_id = %s "
        "ORDER BY marcada_at DESC LIMIT 1", (uid,), fetch_one=True)
    return bool(fila and fila.get("quitada_por_la_persona"))


def marcar(admin_id, user_id, motivo, confirmar_vuelta: bool = False) -> dict:
    """Marca la cuenta como de prueba. 422 `motivo`, 404 `no_existe`, 409 `ya_marcada`, y 409 `salio_ella` si la
    última marca la quitó ELLA y el admin no confirma que fue la persona quien pidió volver (`confirmar_vuelta`)."""
    motivo = _motivo(motivo)
    uid = _uuid(user_id)
    if not uid or not _existe_perfil(uid):
        raise ErrorPrueba(404, "no_existe")
    if marca_viva(uid):
        raise ErrorPrueba(409, "ya_marcada")
    salio = _salio_ella(uid)
    if salio and not confirmar_vuelta:
        raise ErrorPrueba(409, "salio_ella")
    mid = str(uuid.uuid4())
    _anotar(admin_id, "marcar_prueba", uid, {"marca_id": mid, "motivo": motivo, "vuelta_tras_salir": salio})
    try:
        execute_sql_write(
            "INSERT INTO public.cuentas_de_prueba (id, user_id, marcada_por, motivo) VALUES (%s, %s, %s, %s)",
            (mid, uid, admin_id, motivo))
    except Exception as e:  # noqa: BLE001
        _registrar_fallo(admin_id, "marcar_prueba", uid, mid, e)
        nombre = type(e).__name__
        if nombre == "UniqueViolation":        # dos admins a la vez: el índice único parcial corta la carrera
            raise ErrorPrueba(409, "ya_marcada") from e
        if nombre == "ForeignKeyViolation":    # la cuenta se borró entre la lectura y la escritura
            raise ErrorPrueba(404, "no_existe") from e
        raise ErrorPrueba(500, _NO_GUARDADA) from e
    return {"id": mid, "user_id": uid, "motivo": motivo}


def marcar_varias(admin_id, user_ids, motivo) -> list[dict]:
    """`marcar` para cada id, en su orden. Devuelve `{user_id, resultado}` con `marcada | ya_marcada | salio_ella |
    no_existe` (una que salió ella exige el flujo individual con `confirmar_vuelta`). Una fila de rastro por cuenta
    marcada. Más de `MAX_LOTE` ids ⇒ 422 `demasiadas`. Si el rastro o la base fallan a mitad, el lote se corta con ese
    error: las ya marcadas quedan con su rastro y volver a lanzarlo las devuelve como `ya_marcada`."""
    ids = list(user_ids or [])
    if len(ids) > MAX_LOTE:
        raise ErrorPrueba(422, "demasiadas")
    motivo = _motivo(motivo)
    resultados = []
    for uid in ids:
        try:
            marcar(admin_id, uid, motivo)
            resultado = "marcada"
        except ErrorPrueba as e:
            if (e.status, e.detalle) not in _RESULTADOS_DE_LOTE:
                raise
            resultado = e.detalle
        resultados.append({"user_id": str(uid), "resultado": resultado})
    return resultados


def quitar(admin_id, user_id, motivo) -> dict:
    """Quita la marca como admin (la fila se queda: es el historial). 422 `motivo`, 409 `sin_marca`."""
    motivo = _motivo(motivo)
    uid = _uuid(user_id)
    marca = marca_viva(uid) if uid else None
    if not marca:
        raise ErrorPrueba(409, "sin_marca")
    _anotar(admin_id, "quitar_prueba", uid, {"marca_id": marca["id"], "motivo": motivo})
    try:
        filas = execute_sql_write(
            "UPDATE public.cuentas_de_prueba SET quitada_at = now(), quitada_por = %s, "
            "quitada_por_la_persona = false, motivo_quitar = %s WHERE id = %s AND quitada_at IS NULL RETURNING id",
            (admin_id, motivo, marca["id"]), returning=True)
    except Exception as e:  # noqa: BLE001
        _registrar_fallo(admin_id, "quitar_prueba", uid, marca["id"], e)
        raise ErrorPrueba(500, _NO_GUARDADA) from e
    if not filas:
        # Carrera perdida: entre la lectura y el UPDATE la persona salió (o la quitó otro admin). Este admin no quitó nada.
        logger.warning(f"[P1-PLAN-LOTE-830] quitar_prueba {marca['id']}: ya estaba quitada (carrera perdida)")
        _anotar_fallo(admin_id, "quitar_prueba", uid, {"marca_id": marca["id"], "error": "ya_quitada"})
        raise ErrorPrueba(409, "sin_marca")
    return {"id": marca["id"], "user_id": uid}


def salir(user_id) -> bool:
    """La propia persona sale del modo de prueba. Funciona SIEMPRE (también con el interruptor apagado) y es
    idempotente: sin marca viva devuelve False. `quitada_por` es ella misma; no hay fila de admin_access_log (no la
    hizo el personal)."""
    uid = _uuid(user_id)
    if not uid:
        return False
    filas = execute_sql_write(
        "UPDATE public.cuentas_de_prueba SET quitada_at = now(), quitada_por = user_id, "
        "quitada_por_la_persona = true WHERE user_id = %s AND quitada_at IS NULL RETURNING id",
        (uid,), returning=True)
    if filas:
        logger.info(f"[P1-PLAN-LOTE-830] la cuenta {uid} salió del modo de prueba por su cuenta")
    return bool(filas)


def aviso_visto(user_id) -> None:
    """La app le enseñó el aviso a la persona. Solo se anota la primera vez y solo en una marca viva."""
    uid = _uuid(user_id)
    if not uid:
        return
    execute_sql_write(
        "UPDATE public.cuentas_de_prueba SET aviso_visto_at = now() "
        "WHERE user_id = %s AND aviso_visto_at IS NULL AND quitada_at IS NULL", (uid,))


def exigir_prueba(user_id) -> dict:
    """La marca viva de la cuenta, o `ErrorPrueba`: 403 `no_es_prueba` (interruptor apagado o sin marca) y 409
    `aviso_pendiente` (marca viva, pero la persona aún no vio el aviso). Se llama en CADA petición del detalle y NO
    tiene caché: si la persona sale entre dos peticiones, la segunda ya falla. Si la base no responde, tampoco hay
    vista (503)."""
    if not activo():
        raise ErrorPrueba(403, "no_es_prueba")
    try:
        marca = marca_viva(user_id)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[P1-PLAN-LOTE-830] marca no legible para {user_id}: {e!r}")
        raise ErrorPrueba(503, _NO_LEIDA) from e
    if not marca:
        raise ErrorPrueba(403, "no_es_prueba")
    if requiere_aviso() and not marca.get("aviso_visto_at"):
        raise ErrorPrueba(409, "aviso_pendiente")
    return marca


def estado_de(marca: dict | None) -> str | None:
    """`None` sin marca, `"aviso_pendiente"` si el detalle aún no se abre y `"activa"` si ya se abre: refleja lo que
    `exigir_prueba` haría (sin exigir el aviso, una marca sin aviso visto ya está activa)."""
    if not marca:
        return None
    if requiere_aviso() and not marca.get("aviso_visto_at"):
        return "aviso_pendiente"
    return "activa"


def para_la_persona(user_id) -> dict | None:
    """Lo que ve la PERSONA en su perfil: `{desde, aviso_visto}` o None (sin marca, o con el interruptor apagado).
    Nunca rompe el perfil: si la lectura falla devuelve None (el contenido sigue cerrado, la app lo reintenta)."""
    if not activo():
        return None
    try:
        marca = marca_viva(user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-830] marca no legible para el perfil de {user_id}: {e!r}")
        return None
    if not marca:
        return None
    return {"desde": rc.iso(marca.get("marcada_at")), "aviso_visto": bool(marca.get("aviso_visto_at"))}
