"""[P3-TZ-FALLBACK-SSOT · 2026-08-22] «¿Qué huso asumimos cuando no lo sabemos?» tenía TRES
respuestas distintas en el mismo backend, y ninguna sabía de las otras.

    routers/plans._resolve_request_tz_offset   → 0    (UTC)
    tools._LOCAL_DATE_FALLBACK_OFFSET_MIN      → 240  (RD)
    schemas.HealthProfileSchema.tzOffset       → 0    (UTC), y encima nadie la lee

Es exactamente la forma del defecto que este repo bautizó en `P1-DIET-CANON-SSOT`: tres tablas a
mano para la misma pregunta, drifearon, y la que decidía en el sitio equivocado servía Pollo a
vegetarianas. Aquí el daño es más callado pero de la misma familia — el mismo usuario, sin huso
registrado, es dominicano para el helper de fechas del chat y está en UTC para el resolutor de
`/analyze`. Cuatro horas de diferencia decidiendo a qué DÍA pertenece lo que acaba de registrar.

CUÁL DE LOS TRES GANA, y por qué no es una moneda al aire. Medido en producción hoy: de los cinco
perfiles reales, **los cinco** tienen `tz_offset_minutes = 240` (RD). Cero usuarios en UTC. Un
fallback a 0 no es «el neutral»: es una elección, y es la equivocada para el 100% de la población
medida — le corre el día cuatro horas a quien más probablemente lo esté usando. 240 gana porque
describe a los usuarios que hay, no porque sea más bonito.

Va como knob (`MEALFIT_DEFAULT_TZ_OFFSET_MIN`) porque es un cambio de conducta en un camino de
petición: si algún día la población deja de ser dominicana, se mueve sin redeploy.

EL TERCERO NO SE UNIFICA: SE BORRA. `HealthProfileSchema` declaraba también
`timezone = 'America/Santo_Domingo'`. Medido: **cero lectores** en todo el código y **cero filas**
con esa clave en la base. No es un default: es una invitación. Cualquiera que escriba
`profile.timezone` creyendo que lee el huso del usuario obtiene «Santo Domingo» para un noruego,
y el código parecerá correcto en RD, que es donde se prueba. Una constante que DESCRIBE puede
mentir; esta ni siquiera describía.

Y `tzOffset` en ese mismo esquema pasa de `0` a `None`. No se borra porque la clave SÍ existe en
datos reales (5 filas), pero su default deja de fabricar un valor plausible: quien la lea sin
haberla resuelto obtiene `None`, que no se puede confundir con un huso.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def plans_src() -> str:
    return (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def schemas_src() -> str:
    return (_BACKEND / "schemas.py").read_text(encoding="utf-8", errors="replace")


# ── El SSOT ─────────────────────────────────────────────────────────────────────────────────────

def test_existe_una_sola_constante_para_el_huso_desconocido():
    import constants

    assert hasattr(constants, "DEFAULT_TZ_OFFSET_MIN"), (
        "no existe `constants.DEFAULT_TZ_OFFSET_MIN`. Sin un SSOT, cada camino vuelve a inventar "
        "su propia respuesta a «¿qué huso asumimos?» — que es como se llegó a tres"
    )
    assert isinstance(constants.DEFAULT_TZ_OFFSET_MIN, int)


def test_el_valor_describe_a_los_usuarios_que_hay():
    """Positivo = oeste. 240 = UTC−4 = RD, que es donde están los cinco perfiles reales medidos."""
    import constants

    assert constants.DEFAULT_TZ_OFFSET_MIN == 240


def test_es_un_knob_revertible():
    """Cambia conducta en un camino de petición: tiene que poder moverse sin redeploy."""
    import constants

    src = (_BACKEND / "constants.py").read_text(encoding="utf-8", errors="replace")
    assert "MEALFIT_DEFAULT_TZ_OFFSET_MIN" in src, (
        "`DEFAULT_TZ_OFFSET_MIN` se cableó como literal en vez de como knob"
    )


# ── Los tres consumidores ───────────────────────────────────────────────────────────────────────

def test_el_helper_de_fecha_usa_el_ssot():
    import constants
    import tools

    assert tools._LOCAL_DATE_FALLBACK_OFFSET_MIN == constants.DEFAULT_TZ_OFFSET_MIN


def test_el_resolutor_de_peticion_usa_el_ssot(plans_src):
    """EL CAMBIO DE CONDUCTA. `_resolve_request_tz_offset` devolvía 0 y le pasaba 0 a
    `_get_user_tz_live`: dos sitios, los dos en UTC, para una población que está entera en RD."""
    i = plans_src.index("def _resolve_request_tz_offset")
    cuerpo = plans_src[i:plans_src.index("\ndef ", i + 10)]
    codigo = "\n".join(l for l in cuerpo.splitlines() if not l.strip().startswith("#"))
    assert "DEFAULT_TZ_OFFSET_MIN" in codigo, (
        "`_resolve_request_tz_offset` no usa el SSOT; sigue teniendo su propia respuesta"
    )
    assert not re.search(r"fallback_minutes\s*=\s*0\b", codigo), (
        "sigue pasándole 0 a `_get_user_tz_live`: un usuario con perfil pero sin huso vuelve a "
        "salir en UTC"
    )
    assert not re.search(r"^\s*return 0\s*$", codigo, re.M), (
        "el `return 0` final sigue ahí: es la tercera respuesta a la misma pregunta"
    )


# ── La mentira inerte ───────────────────────────────────────────────────────────────────────────

def test_el_esquema_no_declara_un_huso_dominicano_que_nadie_lee(schemas_src):
    """Cero lectores y cero filas con esa clave. No era un default: era una invitación a cablear
    «Santo Domingo» para un noruego, con el código pareciendo correcto en RD."""
    i = schemas_src.index("class HealthProfileSchema")
    cuerpo = schemas_src[i:schemas_src.index("class Config", i)]
    assert "America/Santo_Domingo" not in cuerpo, (
        "`HealthProfileSchema` vuelve a declarar un huso dominicano por defecto. Nadie lo lee y "
        "ninguna fila lo trae: lo único que puede hacer es engañar a quien lo descubra"
    )


def test_el_tzoffset_del_esquema_no_fabrica_un_valor_plausible(schemas_src):
    """`0` se puede confundir con un huso de verdad (UTC); `None` no se puede confundir con nada."""
    i = schemas_src.index("class HealthProfileSchema")
    cuerpo = schemas_src[i:schemas_src.index("class Config", i)]
    m = re.search(r"tzOffset:\s*Optional\[int\]\s*=\s*(\S+)", cuerpo)
    assert m, "desapareció `tzOffset` del esquema"
    assert m.group(1) == "None", (
        f"`tzOffset` vuelve a tener un default numérico ({m.group(1)}): quien lo lea sin resolverlo "
        f"obtiene un huso que nadie eligió"
    )


def test_el_esquema_sigue_aceptando_claves_extra(schemas_src):
    """El límite: `health_profile` es texto libre por diseño. Quitar el campo declarado no puede
    convertirse en quitar el DATO — `extra = 'allow'` es lo que garantiza que una fila con
    `timezone` guardado siga pasando por el validador sin perderla."""
    assert "extra = 'allow'" in schemas_src or 'extra = "allow"' in schemas_src
