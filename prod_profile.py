# backend/prod_profile.py
"""[P1-ARQ27-F3-BATERIA · 2026-09-06] El perfil de knobs EQUIVALENTE A PRODUCCIÓN (ARQ27-P1-06).

`conftest.py` apaga gates a propósito para que los tests que prueban el camino OFF no dependan del
entorno. El efecto lateral es que **la suite entera mide un producto distinto del que se entrega**:
seis knobs corren en un estado que producción no usa. El más caro es `MEALFIT_COUNTRY_SYSTEM`, que en
producción vale `true` y en la suite `False` — con él apagado los seis países colapsan a DO, así que
ningún test de la suite ha visto nunca el catálogo de ES, US, MX, PR ni CO.

**Procedencia.** Los valores salen de `/opt/mealfit/backend/.env` del VPS, leído el 2026-09-06. No son
una suposición ni el default del código: son lo que el binario servido tiene delante. Cuando el
operador cambie un knob en producción, este fichero queda obsoleto y hay que releerlo — por eso lleva
la fecha y por eso `PROFILE_READ_AT` es parte del contrato, no un comentario.

**Lo que NO está aquí, a sabiendas:**

- Secretos (`*_SECRET`, `*_TOKEN`, `*_KEY`, `*_PASSWORD`, `*_URL`). Nunca salen del VPS.
- Las listas de usuarios de canary (`MEALFIT_INITIAL_VIA_QUEUE_USERS`,
  `MEALFIT_PLAN_POLICY_ENFORCE_USERS`). Son identificadores de personas reales; un perfil de pruebas
  no es sitio para ellos. Su ausencia cambia el comportamiento del canary, y eso se declara en
  `EXCLUIDOS_A_SABIENDAS` en vez de disimularse.
- Knobs de infraestructura (tamaños de pool, timeouts de red, modelos de LLM): no cambian QUÉ se
  entrega, solo cuánto tarda o cuánto cuesta, y fijarlos en una batería determinista sería ruido.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

PROFILE_READ_AT = "2026-09-06"
PROFILE_SOURCE = "/opt/mealfit/backend/.env (VPS Oracle, producción)"

#: Knobs que cambian QUÉ se entrega. Valor tal cual en producción.
PROD_KNOBS: dict[str, str] = {
    # ── el sistema de países: el más caro de los divergentes ──────────────────────────────────
    "MEALFIT_COUNTRY_SYSTEM": "true",
    # ── gates clínicos y de contrato ──────────────────────────────────────────────────────────
    "MEALFIT_VERIFIED_INGREDIENTS_ONLY": "true",
    "MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS": "true",
    "MEALFIT_SHOPPING_COHERENCE_GUARD": "block",
    "MEALFIT_CULINARY_JUDGE_GUARD": "warn",
    "MEALFIT_FIDELITY_GATE": "warn",
    "MEALFIT_DISH_QUALITY_SOFT_GATE": "true",
    "MEALFIT_PANTRY_SUFFICIENCY_GATE": "true",
    # ── política del plan ─────────────────────────────────────────────────────────────────────
    "MEALFIT_PLAN_POLICY_MODE": "enforce",
    "MEALFIT_PLAN_JOBS_ENABLED": "1",
    "MEALFIT_INITIAL_VIA_QUEUE": "true",
    "MEALFIT_MAX_ATTEMPTS": "3",
    # ── composición del plato ─────────────────────────────────────────────────────────────────
    "MEALFIT_MACRO_SOLVER_ENABLED": "true",
    "MEALFIT_MACRO_AWARE_RECONCILE": "True",
    "MEALFIT_MICRONUTRIENT_CLOSER": "true",
    "MEALFIT_CARB_TARGET_TRIM": "true",
    "MEALFIT_CARB_TO_PROTEIN_SWAP": "true",
    "MEALFIT_SPICE_UNIT_AS_TSP": "true",
    "MEALFIT_HARDEN_MAIN_ARITY": "true",
    # `MEALFIT_HARDEN_SAMEDAY_PROTEIN` salió del perfil en P1-PLAN-LOTE-7: nunca tuvo rama. Queda huérfano en el
    # .env del VPS (inerte) hasta que el operador lo limpie.
    "MEALFIT_HARDEN_POOLS_ENABLED": "true",
    # ── swap y renovación ─────────────────────────────────────────────────────────────────────
    "MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER": "true",
    "MEALFIT_SWAP_TARGET_FROM_SLOT": "true",
    "MEALFIT_UPDATE_MACRO_REBALANCE": "true",
    "MEALFIT_RENEWAL_PANTRY_AWARE_ENABLED": "true",
    "MEALFIT_PANTRY_COMPLETION_LIST_ENABLED": "true",
    # ── memoria ───────────────────────────────────────────────────────────────────────────────
    "MEALFIT_DREAMING_ENABLED": "true",
    "MEALFIT_DREAMING_RETRIEVAL_ENABLED": "true",
    "MEALFIT_DREAMING_INJECT_PLAN_ENABLED": "true",
}

#: Knobs ausentes del `.env` de producción: allí corren con el DEFAULT DEL CÓDIGO, que es `True`.
#: `conftest` los fuerza a `false`, así que la suite prueba el camino apagado de tres gates que en
#: producción están encendidos. Un knob que no aparece en un `.env` no está apagado — está en su
#: default, y esa distinción es justo la que este bloque existe para no perder.
PROD_DEFAULTS_ON: dict[str, str] = {
    "MEALFIT_SODIUM_EXCESS_GATE": "true",
    "MEALFIT_RECIPE_CONTRACT_GATE": "true",
    "MEALFIT_MICRO_CLOSER_PERDAY": "true",
}

#: Lo que producción tiene y este perfil deja fuera a propósito, con el motivo.
EXCLUIDOS_A_SABIENDAS: dict[str, str] = {
    "MEALFIT_INITIAL_VIA_QUEUE_USERS": "lista de user_id reales — no van a un perfil de pruebas",
    "MEALFIT_PLAN_POLICY_ENFORCE_USERS": "lista de user_id reales — no van a un perfil de pruebas",
    "MEALFIT_DAYGEN_CANARY_PCT": "canary a 0 en producción; una batería determinista no lo ejerce",
    "MEALFIT_DB_POOL_MAX_SIZE": "infraestructura: cambia cuánto tarda, no qué se entrega",
    "MEALFIT_FLASH_MODEL": "modelo de LLM: la batería es determinista y no llama al proveedor",
    "MEALFIT_VISION_MODEL": "modelo de LLM: fuera del alcance de la batería",
}


def perfil_completo() -> dict[str, str]:
    """Los knobs de producción más los que allí corren en su default `True`."""
    return {**PROD_KNOBS, **PROD_DEFAULTS_ON}


def aplicar(entorno: dict | None = None) -> dict[str, str]:
    """Escribe el perfil en `os.environ` (o en el dict que se le pase) y devuelve lo que cambió.

    No usa `setdefault`: el sentido de esta función es SOBRESCRIBIR lo que `conftest` dejó apagado.
    Quien quiera el comportamiento contrario tiene el dict y puede componerlo a mano.

    ⚠️ **Es irreversible por sí sola.** Dentro de un proceso compartido —pytest, por ejemplo— usa
    `perfil_aplicado()`, que restaura al salir. Aplicar esto en tiempo de import contaminó la suite
    entera en su primera versión: 17 tests de coherencia, compras y hierbas caían porque un módulo
    importado les había encendido los gates por debajo. Una herramienta que existe para denunciar que
    la suite corre con otras banderas no puede ser quien se las cambie."""
    env = os.environ if entorno is None else entorno
    cambios = {}
    for k, v in perfil_completo().items():
        if env.get(k) != v:
            cambios[k] = f"{env.get(k)!r} → {v!r}"
        env[k] = v
    return cambios


@contextmanager
def perfil_aplicado(entorno: dict | None = None):
    """El perfil, y al salir el entorno vuelve EXACTO a como estaba — incluidas las claves que no
    existían, que se borran en vez de quedarse con el valor de producción."""
    env = os.environ if entorno is None else entorno
    previo = {k: env.get(k) for k in perfil_completo()}
    cambios = aplicar(env)
    try:
        yield cambios
    finally:
        for k, v in previo.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v


def divergencias(entorno: dict | None = None) -> Iterator[tuple[str, str, str]]:
    """`(knob, valor_actual, valor_produccion)` para cada knob que el entorno actual tiene distinto.

    Es la medida del gap: si esto devuelve algo dentro de la suite, la suite está midiendo otro
    producto. Se publica en la cabecera de la batería en vez de asumirse."""
    env = os.environ if entorno is None else entorno
    for k, v in perfil_completo().items():
        actual = env.get(k)
        if actual is None or str(actual).strip().lower() != str(v).strip().lower():
            yield (k, "(ausente)" if actual is None else str(actual), str(v))


__all__ = ["PROFILE_READ_AT", "PROFILE_SOURCE", "PROD_KNOBS", "PROD_DEFAULTS_ON",
           "EXCLUIDOS_A_SABIENDAS", "perfil_completo", "aplicar", "perfil_aplicado", "divergencias"]
