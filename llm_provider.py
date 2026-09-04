"""[P0-LLM-PROVIDER-MIGRATION · 2026-06-12] SSOT del provider LLM (GLM) +
router de modelos por tier de suscripción.

Este módulo reemplaza TODA la dependencia previa de `langchain_google_genai`
(Gemini). Decisión de producto 2026-06-12: migrar a GLM-5.3 (API
OpenAI-compatible, base `https://api.z.ai/api/paas/v4`) para reducir costo de
producción, con enrutamiento por plan de pago:

    - Tier `gratis` (free) / guests / desconocido → `glm-5.3-flash`
    - Tiers `basic` / `plus` / `ultra` (pagados)   → `glm-5.3-flash`

[P1-FLASH-PRIMARY · 2026-07-31] Decisión del owner: `glm-5.3-flash` es
actualmente MEJOR que `glm-5.3` (los providers actualizan modelos bajo
el mismo ID — la premisa "pro > flash" de 2026-06-12 caducó). Flash pasa a ser
el modelo PRIMARIO de TODAS las superficies, incluidos tiers pagados y el
reviewer clínico risk-tier (`graph_orchestrator._REVIEWER_RISK_TIER_DEFAULT`).
Pro NO desaparece: queda exclusivamente como RED post-fallo (2º en cadena del
day-gen, fallback del planner con breaker independiente, escalada del corrector
quirúrgico — `MEALFIT_PRO_MODEL`), donde su valor es ser un modelo DISTINTO
con circuit breaker propio, no ser "mejor". Rollback sin redeploy:
`MEALFIT_MODEL_PAID_TIER=glm-5.3`.

Precios oficiales 2026-06 (por 1M tokens): flash $0.14 in / $0.28 out;
pro $0.435 in / $0.87 out. Ambos: 1M contexto, 384K max output, JSON mode,
function calling, thinking nativo (gestionado por el provider — el antiguo
`thinking_budget` de Gemini NO aplica y se swallow-ea en el wrapper).

Contratos:
  - `ChatGLM` — drop-in del antiguo `ChatGoogleGenerativeAI`: acepta y
    descarta kwargs legacy (`google_api_key`, `safety_settings`,
    `thinking_budget`) y traduce `max_output_tokens` → `max_tokens`, para que
    los ~37 callsites migren sin cirugía de kwargs. API key SIEMPRE desde env
    `ZAI_API_KEY` (NUNCA hardcodeada — test blanket
    `test_p0_glm_migration.py` lo enforza).
  - `resolve_model_for_tier(tier)` / `resolve_model_for_user(user_id)` — el
    router. Fail-cheap: cualquier duda (guest, DB blip, tier desconocido)
    resuelve al modelo FREE. Un fallo de lookup jamás puede COSTAR dinero
    (simétrico a fail-secure en auth: acá el riesgo es billing, no IDOR).
  - Tier lookup con cache in-process TTL (`MEALFIT_TIER_CACHE_TTL_S`,
    default 300s) para no añadir un roundtrip DB por cada LLM call.

Knobs (auto-registrados en `_KNOBS_REGISTRY` vía `_env_*`):
  - `MEALFIT_ZAI_BASE_URL`  (default `https://api.z.ai/api/paas/v4`)
  - `MEALFIT_MODEL_FREE_TIER`    (default `glm-5.3-flash`)
  - `MEALFIT_MODEL_PAID_TIER`    (default `glm-5.3-flash`, P1-FLASH-PRIMARY)
  - `MEALFIT_TIER_CACHE_TTL_S`   (default 300, clamp [10, 3600])

Rollback operacional sin redeploy: ambos modelos son swappeables vía knob
(convención P3-PREVIEW-MODEL-KNOB). Si GLM deprecia los IDs V4
(`glm-5.3-flash`/`glm-5.3` legacy mueren 2026-07-24), basta con
setear los knobs al ID nuevo y reiniciar el worker.

Tooltip-anchor: P0-LLM-PROVIDER-MIGRATION.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from langchain_openai import ChatOpenAI

from knobs import _env_int, _env_str, _env_bool

logger = logging.getLogger(__name__)

# [P0-GLM-MIGRATION · 2026-09-02] IDs oficiales del API Z.ai (docs.z.ai, verificados
# EN VIVO 2026-09-02): `glm-5.3-flash` (320B MoE/18B activos, multimodal, 1M ctx,
# $0.15/$0.50 por 1M in/out) y `glm-5.3` (flagship, $1.4/$4.4). Los dos piensan
# SIEMPRE: `thinking.type=disabled` responde 400 (código 1210); la latencia se
# gobierna con `reasoning_effort` ∈ {low, high, max} (ver `_glm_reasoning_effort`).
GLM_FLASH = "glm-5.3-flash"
GLM_PRO = "glm-5.3"

# [P1-NET-LUNA · P1-REVIEWER-TIER-MODELS · P1-REVIEWER-SOL-HARD · 2026-07-31]
# IDs OpenAI gpt-5.6 en uso: luna = red cross-provider del pipeline + reviewer
# clínico free; terra = reviewer clínico tiers pagados; sol = reviewer clínico
# plus/ultra en casos clínicos DIFÍCILES (bariátrico / ≥2 reglas activas).
# SSOT del ID — no repetir literales.
GPT56_LUNA = "gpt-5.6-luna"
GPT56_TERRA = "gpt-5.6-terra"
GPT56_SOL = "gpt-5.6-sol"

# [P0-GLM-MIGRATION · 2026-09-02] GLM-5.3 razona SIEMPRE (no existe "thinking off":
# el API responde 400/1210). Lo que sí se gobierna es el ESFUERZO, y el esfuerzo es
# latencia y tokens de salida facturados (medido 2026-09-02 con un prompt trivial:
# effort=max → 6,3 s / 173 tokens; effort=low → 1,1 s / 14 tokens). El default del
# wrapper es `low` para TODOS los runnables (títulos, extractores, memoria, router,
# structured-output): son relleno de esquema, no deliberación. Las superficies que
# quieren razonar de verdad (day-gen por tier, reviewer con riesgo, corrector Pro,
# juez culinario) pasan su effort explícito y SIEMPRE gana sobre este default.
# Knob de rollback sin redeploy: `MEALFIT_GLM_REASONING_EFFORT=high|max`.
_GLM_EFFORT_VALID = frozenset({"low", "high", "max"})
_GLM_DEFAULT_REASONING_EFFORT = _env_str("MEALFIT_GLM_REASONING_EFFORT", "low", choices=set(_GLM_EFFORT_VALID))


def _glm_reasoning_effort(value) -> str:
    """Traduce cualquier vocabulario de effort del pipeline al de Z.ai (low|high|max).

    `medium` (OpenAI) → `high`; `xhigh` (OpenAI) → `max`; `none`/`off`/vacío → `low`
    (GLM no puede apagar el razonamiento: `low` es lo más barato que existe).
    """
    v = str(value or "").strip().lower()
    if v in _GLM_EFFORT_VALID:
        return v
    if v in ("medium",):
        return "high"
    if v in ("xhigh", "very_high", "ultra"):
        return "max"
    return "low"

# Tiers de pago canónicos (columna `user_profiles.plan_tier`, ver
# routers/billing.py P0-BILLING-1). Todo lo demás («gratis», NULL, guests,
# strings corruptos) enruta a FREE — fail-cheap.
PAID_TIERS = frozenset({"basic", "plus", "ultra"})

_MISSING_KEY_PLACEHOLDER = "MISSING_ZAI_API_KEY"
_warned_missing_key = False


def _zai_base_url() -> str:
    """Base URL OpenAI-compatible de GLM. Knob para entornos proxy/test."""
    return _env_str("MEALFIT_ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")


def _is_glm_provider(base_url: Optional[str] = None) -> bool:
    """True si el `base_url` efectivo apunta a GLM.

    [P1-SINGLE-PROVIDER-RESTORE · 2026-07-04] El `extra_body={"thinking": ...}`
    que este wrapper inyecta es un parámetro ESPECÍFICO del API GLM: otros
    back-ends OpenAI-compatibles rechazan campos desconocidos con HTTP 400.
    Este guard evita inyectarlo si un entorno de test apunta el knob
    `MEALFIT_ZAI_BASE_URL` a un proxy/back-end distinto. GLM es el
    ÚNICO provider soportado en producción — el plumbing multi-provider
    (override global de modelo + detección Ollama) fue eliminado a pedido del
    owner 2026-07-04; si se re-introduce un provider alterno, debe nacer con
    knob + test ancla propios.

    [P1-PROVIDER-INSTANCE-GUARD · 2026-07-28] La asunción anterior ("los
    callsites productivos NUNCA pasan un `base_url` propio") YA NO es cierta:
    `ChatGLM` (subclase de `ChatOpenAI`) se construye cada vez más contra
    back-ends OpenAI-compatibles DISTINTOS de GLM (p.ej. el meal-photo
    scanner apuntado a OpenAI). Llamar esta función sin argumento SIEMPRE
    inspecciona el knob global `MEALFIT_ZAI_BASE_URL` — que SIEMPRE
    contiene "z.ai" — y retorna `True` sin importar a dónde apunte la
    instancia real. Todo callsite que resuelva el provider de una instancia YA
    construida (p.ej. `ChatGLM.with_structured_output`) DEBE pasar el
    `base_url` efectivo de ESA instancia (`self.openai_api_base` en
    `ChatOpenAI` — `base_url` NO es atributo de instancia, solo kwarg del
    constructor), nunca confiar en el default implícito. El fallback al knob
    global sigue vigente para: (a) el path del constructor (`base_url` aún no
    resuelto, puede venir `None`) y (b) cualquier caller sin forma de leer el
    atributo de instancia — preserva el comportamiento GLM-only de hoy.
    """
    resolved = (base_url or _zai_base_url() or "").lower()
    return ("z.ai" in resolved) or ("bigmodel" in resolved)


def _zai_api_key() -> str:
    """API key desde env `ZAI_API_KEY`.

    Si falta, retorna un placeholder NO-vacío: la construcción del cliente
    nunca debe tirar el boot (hay LLMs construidos a module-import, e.g.
    `agent.py::llm`); la invocación fallará con 401 explícito y el error-log
    de boot (una sola vez) le dice al operador exactamente qué falta. Misma
    semántica que tenía el constructor legacy con la key ausente (None).
    """
    global _warned_missing_key
    key = (os.environ.get("ZAI_API_KEY") or "").strip()
    if key:
        return key
    if not _warned_missing_key:
        logger.error(
            "❌ [LLM-PROVIDER] ZAI_API_KEY no configurada en el entorno. "
            "Toda invocación LLM fallará con 401 hasta setearla "
            "(.env local / env vars del VPS) y reiniciar el worker."
        )
        _warned_missing_key = True
    return _MISSING_KEY_PLACEHOLDER


def model_free_tier() -> str:
    """Modelo para tier `gratis`, guests y fallback. Default V4 Flash."""
    return _env_str("MEALFIT_MODEL_FREE_TIER", GLM_FLASH) or GLM_FLASH


def model_paid_tier() -> str:
    """Modelo para tiers `basic`/`plus`/`ultra`. Default V4 Flash
    ([P1-FLASH-PRIMARY · 2026-07-31]: el owner midió que flash es actualmente
    mejor que pro; era `GLM_PRO` desde P0-LLM-PROVIDER-MIGRATION). Rollback:
    `MEALFIT_MODEL_PAID_TIER=glm-5.3`."""
    return _env_str("MEALFIT_MODEL_PAID_TIER", GLM_FLASH) or GLM_FLASH


def resolve_model_for_tier(tier: Optional[str]) -> str:
    """Router tier → model ID. Desconocido/None/`gratis` → FREE (fail-cheap)."""
    normalized = (tier or "").strip().lower()
    if normalized in PAID_TIERS:
        return model_paid_tier()
    return model_free_tier()


# ------------------------------------------------------------------
# Tier lookup con cache TTL in-process.
#
# Por qué cache: el pipeline de un plan hace decenas de LLM calls; sin
# cache cada una pagaría un roundtrip a `user_profiles`. TTL corto (5 min
# default) para que un upgrade de tier se refleje rápido sin redeploy.
# Por qué fail-cheap cacheado: durante un blip de DB preferimos servir al
# usuario pagado con el modelo FREE durante ≤TTL segundos antes que
# martillar la DB caída con un lookup por LLM call.
# ------------------------------------------------------------------
_TIER_CACHE: dict = {}
_TIER_CACHE_LOCK = threading.Lock()
_TIER_CACHE_MAX_ENTRIES = 4096


def _tier_cache_ttl_s() -> int:
    return _env_int(
        "MEALFIT_TIER_CACHE_TTL_S", 300, validator=lambda v: 10 <= v <= 3600
    )


def invalidate_tier_cache(user_id: Optional[str] = None) -> None:
    """Invalida el cache de tier (entero o per-user). Llamar tras upgrades
    de billing si se quiere reflejo inmediato sin esperar el TTL."""
    with _TIER_CACHE_LOCK:
        if user_id is None:
            _TIER_CACHE.clear()
        else:
            _TIER_CACHE.pop(str(user_id), None)


def get_user_tier(user_id: Optional[str]) -> str:
    """Resuelve `plan_tier` para `user_id` con cache TTL.

    Guests (None, vacío, prefijo `guest`) y cualquier fallo de lookup →
    `gratis`. El lookup usa `db.get_user_plan_tier` con import lazy para
    no acoplar este módulo (importado a module-init por todo el backend)
    al stack de DB en import-time.
    """
    if not user_id or not isinstance(user_id, str):
        return "gratis"
    uid = user_id.strip()
    if not uid or uid.lower().startswith("guest"):
        return "gratis"

    now = time.monotonic()
    ttl = _tier_cache_ttl_s()
    with _TIER_CACHE_LOCK:
        hit = _TIER_CACHE.get(uid)
        if hit is not None and (now - hit[1]) < ttl:
            return hit[0]

    tier = "gratis"
    try:
        from db import get_user_plan_tier  # lazy: evita ciclo en module-init

        raw = get_user_plan_tier(uid)
        if raw:
            tier = str(raw).strip().lower() or "gratis"
    except Exception as e:
        # Fail-cheap documentado: blip de DB → tier FREE cacheado ≤TTL.
        logger.debug(
            "[LLM-PROVIDER] tier lookup falló (user_id=%s): %s: %s — "
            "fail-cheap a 'gratis'",
            uid[:36],
            type(e).__name__,
            str(e)[:160],
        )

    with _TIER_CACHE_LOCK:
        if len(_TIER_CACHE) >= _TIER_CACHE_MAX_ENTRIES:
            # Evicción simple: limpiar todo. El cache se rellena solo y un
            # clear esporádico (>4k usuarios activos en 5 min) es más barato
            # que mantener LRU exacto en el hot path.
            _TIER_CACHE.clear()
        _TIER_CACHE[uid] = (tier, now)
    return tier


def resolve_model_for_user(user_id: Optional[str] = None) -> str:
    """Router user → model ID. [P1-FLASH-PRIMARY] Ambos tiers resuelven FLASH
    por default; la distinción pagado/gratis se conserva (knobs separados) para
    poder divergir de nuevo sin tocar código."""
    return resolve_model_for_tier(get_user_tier(user_id))


# Kwargs del constructor legacy de Gemini que el wrapper acepta y DESCARTA
# en silencio, para que la migración de callsites sea rename-only:
#   - google_api_key: la key ahora viene de ZAI_API_KEY (env).
#   - safety_settings: HarmCategory/HarmBlockThreshold eran Gemini-only.
#     (La decisión P3-CHAT-SAFETY-OFF queda obsoleta: GLM no aplica
#     content-filters configurables client-side.)
#   - thinking_budget: el thinking de GLM-5.3 es nativo del modelo y no
#     expone budget por request en el API OpenAI-compatible. El costo que
#     motivaba el cap (reasoning a $9/M en Gemini) no existe: output V4
#     cuesta $0.28–0.87/M, 10-30× menos.
_LEGACY_SWALLOWED_KWARGS = (
    "google_api_key",
    "safety_settings",
    "thinking_budget",
    "convert_system_message_to_human",
)


class ChatGLM(ChatOpenAI):
    """Cliente chat GLM (OpenAI-compatible) — reemplazo 1:1 del antiguo
    `ChatGoogleGenerativeAI`.

    Diferencias gestionadas internamente:
      - `model`: ID GLM (`glm-5.3-flash` / `glm-5.3`).
      - `api_key`/`base_url`: defaults desde env/knob; los callsites NO los
        pasan (y NUNCA hardcodean la key — test blanket).
      - `max_output_tokens` → `max_tokens` (naming OpenAI).
      - `stream_usage=True` por default: GLM soporta `include_usage` en
        streaming y el wrapper de instrumentación (`graph_orchestrator.py`)
        depende de `usage_metadata` en el último chunk para llenar
        `llm_usage_events` (P1-COST-INSTRUMENTATION-FIX).
      - kwargs legacy de Gemini: swallow silencioso (ver lista arriba).

    `timeout=` (segundos) y `max_retries=` se pasan tal cual — ChatOpenAI los
    soporta nativamente, preservando los knobs `MEALFIT_*_LLM_TIMEOUT_S`
    (P2-LLM-TIMEOUT-SWEEP) sin cambios.
    """

    def __init__(
        self,
        *,
        model: str,
        max_output_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        for _legacy in _LEGACY_SWALLOWED_KWARGS:
            kwargs.pop(_legacy, None)
        # [P1-SINGLE-PROVIDER-RESTORE · 2026-07-04] El override global de modelo
        # (`MEALFIT_LLM_MODEL_OVERRIDE`, [MULTI-PROVIDER · 2026-07-01]) fue
        # ELIMINADO: permitía colapsar TODOS los modelos —incluido el reviewer
        # médico risk-tier— a un provider de test (Gemini), degradando el gate
        # clínico fail-secure en silencio. El routing de modelos vive SOLO en
        # los knobs MEALFIT_*_MODEL per-feature + el router por tier.
        if max_output_tokens is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_output_tokens
        kwargs.setdefault("stream_usage", True)
        # [P0-GLM-MIGRATION · 2026-09-02] GLM razona siempre: aquí se fija el ESFUERZO.
        # Contrato con los callsites heredados: `extra_body.thinking.effort` (vocabulario
        # del proveedor anterior) y `thinking.type=disabled` se TRADUCEN — el primero a
        # `reasoning_effort`, el segundo a `low` — porque Z.ai rechaza ambos tal cual.
        # `reasoning_effort` explícito del callsite gana; si no hay ninguno, default del knob.
        if _is_glm_provider(base_url):
            _extra = dict(kwargs.get("extra_body") or {})
            _think = _extra.get("thinking")
            _legacy_eff = None
            if isinstance(_think, dict):
                if _think.get("type") == "disabled":
                    _legacy_eff = "low"
                elif _think.get("effort"):
                    _legacy_eff = _think.get("effort")
            _extra["thinking"] = {"type": "enabled"}
            if "reasoning_effort" not in kwargs:
                kwargs["reasoning_effort"] = _glm_reasoning_effort(
                    _legacy_eff if _legacy_eff is not None else _GLM_DEFAULT_REASONING_EFFORT
                )
            else:
                kwargs["reasoning_effort"] = _glm_reasoning_effort(kwargs["reasoning_effort"])
            kwargs["extra_body"] = _extra
        super().__init__(
            model=model,
            api_key=api_key or _zai_api_key(),
            base_url=base_url or _zai_base_url(),
            **kwargs,
        )

    def with_structured_output(self, schema=None, **kwargs):
        """Override del default de langchain-openai, calibrado EN VIVO contra Z.ai
        (2026-09-02, glm-5.3-flash con thinking activo):
        1. `method="json_schema"` (default de langchain-openai 1.3) y `json_mode`:
           GLM devuelve JSON pero IGNORA el esquema (contesta `{"respuesta": ...}` o
           markdown) → `OutputParserException`. Se fuerza `method="function_calling"`
           (tools API + `tool_choice` forzado), que SÍ respeta el esquema y funciona
           CON razonamiento activo (13 s a effort=low para un veredicto de 3 campos).
        2. Un caller que pida `json_mode` explícito contra GLM recibe igualmente
           `function_calling`: en el proveedor anterior `json_mode` era el rodeo para
           "thinking rechaza tool_choice"; en Z.ai es al revés — el rodeo es lo que rompe.
        Cubre los ~15 callsites `.with_structured_output(...)` del pipeline sin tocarlos.
        [P1-PROVIDER-INSTANCE-GUARD · 2026-07-28] El guard resuelve el provider de ESTA
        instancia (`self.openai_api_base`), no del knob global: una instancia apuntada
        a OpenAI conserva el `method` que pidió el caller.
        """
        kwargs.setdefault("method", "function_calling")
        _instance_base_url = getattr(self, "openai_api_base", None)
        if kwargs["method"] == "json_mode" and _is_glm_provider(_instance_base_url):
            kwargs["method"] = "function_calling"
        return ChatOpenAI.with_structured_output(self, schema, **kwargs)


# ============================================================
# [P1-DAYGEN-LUNA-CANARY · 2026-07-26] Fábrica de LLM por PROVEEDOR.
# ============================================================
# `_build_day_llm` (graph_orchestrator) construía siempre `ChatGLM`, aunque su propio
# comentario decía "provider correcto por prefijo" — la intención estaba escrita y no
# implementada. Con un modelo OpenAI en el chain, ese hardcode lo mandaría al base_url de
# GLM con la key equivocada.
#
# Verificado contra el API (2026-07-26) antes de escribir esto:
#   · `gpt-5.6-luna` responde en `/v1/chat/completions` y soporta `response_format=json_object`
#     (el day-gen lo exige con MEALFIT_DAYGEN_JSON_MODE).
#   · A pelo rechaza `temperature != 1` y `max_tokens` (pide `max_completion_tokens`), pero
#     langchain-openai 1.3.0 traduce ambos → los ~37 callsites no necesitan cirugía.
#
# ⚠️ LangChain DESCARTA la temperatura en silencio para estos modelos: sólo aceptan el valor por
# defecto. Cualquier nodo que dependa de `temperature=0` para ser determinista (p.ej. el
# `compressor`, cuyo comentario dice "no inventes nada, solo resume") PIERDE esa garantía sin
# aviso. Por eso el canario se limita al day-gen, donde la temperatura es un empujón y no un
# contrato.
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt")

# [P1-VISION-GEMINI-FLASH · 2026-09-04] Gemini SOLO como provider de VISIÓN (escáner de
# comida y de Nevera), por su endpoint compatible con OpenAI — cero dependencia nueva
# (`langchain-google-genai` sigue fuera, P0-LLM-PROVIDER-MIGRATION). El pipeline de
# planes y el coach NO pasan por aquí: `build_chat_llm` no conoce estos prefijos a
# propósito. Motivo del cambio: en Roboflow Vision Evals (52 modelos, 2026-09) Gemini 3.8
# Flash da 85,1 % (#3) frente a 72,7 % (#19) de gpt-5.6-luna y 65,3 % (#36) de GLM-5V-Turbo,
# con identificación 97,9 % vs 84,4 % y conteo 78,8 % vs 66,2 %: lo que decide un plato.
_GOOGLE_MODEL_PREFIXES = ("gemini",)
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def is_google_model(model: str) -> bool:
    """¿El ID es un modelo Gemini (vía su capa compatible con OpenAI)?"""
    m = str(model or "").strip().lower()
    return any(m.startswith(p) for p in _GOOGLE_MODEL_PREFIXES)


def _google_api_key() -> str:
    """Key de Gemini desde el entorno (la variable propia de Gemini, o la de Google como
    alias — ver las dos líneas marcadas abajo). Fail-loud, mismo contrato que
    `_openai_api_key`: NUNCA argumento del callsite. Las líneas llevan el marker inline que
    `test_p0_llm_provider_migration` exige para la costura de VISIÓN (única excepción viva
    al blanket anti-Gemini; el pipeline de planes sigue vetado)."""
    _k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")  # [P1-VISION-GEMINI-FLASH]
    if not _k:
        raise RuntimeError("modelo Gemini pedido sin la key de Gemini en el entorno")
    return _k


def is_openai_model(model: str) -> bool:
    """¿El ID pertenece al API de OpenAI (y no a GLM)?"""
    m = str(model or "").strip().lower()
    return any(m.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def _openai_api_key() -> str:
    """Key de OpenAI desde el entorno. Fail-loud: mejor una excepción que un call silencioso al
    proveedor equivocado. NUNCA argumento del callsite — mismo contrato que enforza
    `test_p0_glm_migration.py`."""
    _k = os.environ.get("OPENAI_API_KEY")
    if not _k:
        raise RuntimeError("modelo OpenAI pedido sin OPENAI_API_KEY en el entorno")
    return _k


def build_chat_llm(model: str, **kwargs):
    """Devuelve el cliente chat del proveedor que corresponde al `model`.

    OpenAI → `ChatOpenAI` con `OPENAI_API_KEY` y base por defecto. Resto → `ChatGLM`
    (que ya inyecta su propia key/base).

    ⚠️ [P1-LUNA-USAGE-BLIND · 2026-07-26] Devuelve las clases BASE: sin backpressure y **sin
    contabilidad de costo**. Dentro del pipeline de generación NO se usa esta fábrica — allí van
    `graph_orchestrator.ChatGLM` / `ChatOpenAIInstrumented`, que añaden el mixin. Construir
    el day-gen con esta fábrica dejó `llm_usage_events` sin una sola fila de `day_generator`
    (el nodo más caro) durante la primera corrida del canario Luna.
    """
    if is_openai_model(model):
        return ChatOpenAI(model=model, api_key=_openai_api_key(), **kwargs)
    return ChatGLM(model=model, **kwargs)
