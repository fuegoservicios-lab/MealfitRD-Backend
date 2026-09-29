# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-846 · 2026-09-29] La edad mínima de 18 años — SSOT del backend.

POR QUÉ
    Los Términos (§2) y la Privacidad (§11) dicen «solo mayores de 18, sin excepción», y el formulario aceptaba de 12 a
    100 años: recogía datos de salud de chicos de 12 a 17 (también de menores de 13, COPPA en EE. UU.; en España el
    art. 8 del RGPD fija los 14) y los mandaba a la IA. Auditoría 2026-09-29, fila 16.2 y §A.9.

QUÉ HACE CUMPLIR
    - `routers/plans.py::_BIO_RANGES["age"] = (18, 100)`, espejo de `BIO_RANGES.age` del formulario
      (`test_p3_5_bio_ranges_parity.py`) y de `tools._CHAT_BIO_RANGES` (el coach no guarda una edad menor).
    - 422 `underage` (este módulo) ANTES que cualquier otra validación en las puertas por las que una edad entra a la
      generación (`/analyze`, `/analyze/stream`, `/generation-runs`) y en las que regeneran contenido del plan con la
      edad del perfil (cambiar plato, regenerar día, arreglar sodio, reintentar y regenerar bloques). También en
      `PATCH /api/profile`: un perfil nunca guarda una edad menor.
    - P1-MINOR-SAFETY-GATE (`nutrition_calculator`, FS9) se queda como capa extra: si un menor se colara por un camino
      que nadie cubre, el plan sigue sin déficit y con revisión profesional.

QUÉ ES «MENOR»
    Una edad LEGIBLE (un número limpio: coma o punto decimal, nada más; «15 años» no lo es) cuya parte entera está
    entre 1 y 17: la cuenta `int(float(...))` del router y el `0 < age < 18` del gate de `nutrition_calculator`. Una
    edad ausente o ilegible (0, negativa, texto) no es «menor»: eso lo rechaza el rango (`invalid_biometric_range`). El
    espejo del formulario es `esMenorDeEdad` (`frontend/src/config/formValidation.js`), con la MISMA expresión.

tooltip-anchor: EDAD_MINIMA, _EDAD_LIMPIA, es_menor_de_edad, rechazar_si_menor, rechazar_si_menor_en_perfil,
validar_edad_del_perfil, sin_edad_de_menor
(tests/test_p1_plan_lote_846_edad.py)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

#: Edad mínima para usar Bioboros. Espejo: `_BIO_RANGES["age"][0]` (router) y `BIO_RANGES.age.min` (formulario).
EDAD_MINIMA = 18
#: Techo del rango (erratas), espejo de `_BIO_RANGES["age"][1]` y `BIO_RANGES.age.max`.
EDAD_MAXIMA = 100

#: El código de error del 422, para el cliente (`errorCopy.js` lo traduce).
CODIGO_MENOR = "underage"

#: La frase del formulario (base es-DO); el cliente pinta la suya traducida a partir del código.
MENSAJE_MENOR = f"Bioboros es solo para mayores de {EDAD_MINIMA} años."

# [ronda 1] Una edad «limpia»: un número y nada más (coma o punto decimal, espacios alrededor). La MISMA expresión
# que `EDAD_LIMPIA` de `esMenorDeEdad` (formValidation.js): «15 años», «1e1», «1_5» o «inf» no son edades en ninguno de
# los dos lados (antes JS leía «15 años» como 15 con `parseFloat` y Python como ilegible).
_EDAD_LIMPIA = re.compile(r"^\s*[+-]?\d+(?:[.,]\d+)?\s*$")


def edad_declarada(valor: Any) -> Optional[int]:
    """La parte entera de una edad legible, o None. Acepta int/float finitos y cadenas LIMPIAS (`_EDAD_LIMPIA`)."""
    if valor is None or isinstance(valor, bool):
        return None
    try:
        if isinstance(valor, str):
            if not _EDAD_LIMPIA.match(valor):
                return None
            valor = valor.strip().replace(",", ".")
        elif not isinstance(valor, (int, float)):
            return None
        return int(float(valor))
    except (TypeError, ValueError, OverflowError):
        return None


def es_menor_de_edad(valor: Any) -> bool:
    """True si la edad es legible y su parte entera está entre 1 y `EDAD_MINIMA - 1`."""
    edad = edad_declarada(valor)
    return edad is not None and 0 < edad < EDAD_MINIMA


def detalle_menor() -> dict:
    """El cuerpo del 422. `code` sigue la forma de los demás 422 del formulario (`detail.code`); `error_code` es el
    nombre que pide la auditoría; `field` devuelve el formulario al campo de la edad (`campoDelRechazo`, Plan.jsx)."""
    return {
        "code": CODIGO_MENOR,
        "error_code": CODIGO_MENOR,
        "field": "age",
        "min_age": EDAD_MINIMA,
        "message": MENSAJE_MENOR,
    }


def rechazar_si_menor(*edades: Any, origen: str) -> None:
    """422 `underage` si CUALQUIERA de las edades dadas es de un menor. Sin edades legibles no hace nada."""
    if any(es_menor_de_edad(e) for e in edades):
        logger.warning(f"[P1-PLAN-LOTE-846] {origen}: edad menor de {EDAD_MINIMA} -> 422 {CODIGO_MENOR}")
        raise HTTPException(status_code=422, detail=detalle_menor())


def edad_del_perfil(user_id: Optional[str]) -> Optional[Any]:
    """`health_profile.age` de la cuenta, o None (invitado, sin perfil o la lectura falló).

    Fail-open a propósito: la puerta principal es la edad de la PETICIÓN. Los escritores de `health_profile.age` la
    rechazan o la descartan: `PATCH /api/profile` (422), la generación (422 antes del merge que la vuelca al perfil),
    la herramienta del coach que edita el formulario (no la guarda) y el merge del chat
    (`services.merge_form_data_with_profile`, que la quita antes de escribir). Quedan las cuentas ANTERIORES a este
    lote (0 en producción el 2026-09-29) y cualquier escritor futuro que no pase por aquí: para eso existe esta
    lectura. Un fallo de la base no debe tumbar un cambio de plato de un adulto."""
    if not user_id or user_id == "guest":
        return None
    try:
        from db import execute_sql_query
        fila = execute_sql_query(
            "SELECT health_profile->>'age' AS age FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        return (fila or {}).get("age")
    except Exception as e:  # pragma: no cover - la base caída se ve en otros sitios
        logger.warning(f"[P1-PLAN-LOTE-846] no pude leer la edad del perfil: {type(e).__name__}: {e}")
        return None


def rechazar_si_menor_en_perfil(user_id: Optional[str], *edades: Any, origen: str) -> None:
    """La edad de la petición (si la trae) y la del perfil de la cuenta: 422 `underage` si alguna es de un menor.
    La de la petición se mira primero: si ya es de un menor, no hace falta leer la base."""
    rechazar_si_menor(*edades, origen=origen)
    rechazar_si_menor(edad_del_perfil(user_id), origen=f"{origen} (perfil)")


def validar_edad_del_perfil(valor: Any, *, origen: str) -> None:
    """[ronda 1] La edad que va a GUARDARSE en el perfil: vacía/ausente pasa (no tocarla o borrarla es válido); de un
    menor, 422 `underage`; ilegible, 0, negativa o sobre el techo, el 422 `invalid_biometric_range` de siempre (la
    misma forma que `routers/plans.py::_validate_form_data_ranges`)."""
    if valor is None or (isinstance(valor, str) and not valor.strip()):
        return
    rechazar_si_menor(valor, origen=origen)
    edad = edad_declarada(valor)
    if edad is None or not (EDAD_MINIMA <= edad <= EDAD_MAXIMA):
        logger.warning(f"[P1-PLAN-LOTE-846] {origen}: edad fuera de rango -> 422 invalid_biometric_range")
        raise HTTPException(status_code=422, detail={
            "code": "invalid_biometric_range",
            "errors": [{"field": "age", "value": valor, "accepted_range": [EDAD_MINIMA, EDAD_MAXIMA], "unit": "años"}],
            "message": "Algunos datos biométricos están fuera del rango aceptado: age. Revísalos en el formulario.",
        })


def sin_edad_de_menor(datos: Any, *, origen: str) -> Any:
    """[ronda 1] Copia de `datos` SIN la clave `age` si es la de un menor (y lo anota); si no, `datos` tal cual. Para
    los escritores del perfil que no responden con un error propio (el merge del chat)."""
    if isinstance(datos, dict) and es_menor_de_edad(datos.get("age")):
        logger.warning(f"[P1-PLAN-LOTE-846] {origen}: edad de menor descartada, no se guarda en el perfil")
        return {k: v for k, v in datos.items() if k != "age"}
    return datos
