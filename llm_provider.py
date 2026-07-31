"""[P0-DEEPSEEK-MIGRATION · 2026-06-12] SSOT del provider LLM (DeepSeek) +
router de modelos por tier de suscripción.

Este módulo reemplaza TODA la dependencia previa de `langchain_google_genai`
(Gemini). Decisión de producto 2026-06-12: migrar a DeepSeek-V4 (API
OpenAI-compatible, base `https://api.deepseek.com`) para reducir costo de
producción, con enrutamiento por plan de pago:

    - Tier `gratis` (free) / guests / desconocido → `deepseek-v4-flash`
    - Tiers `basic` / `plus` / `ultra` (pagados)   → `deepseek-v4-flash`

[P1-FLASH-PRIMARY · 2026-07-31] Decisión del owner: `deepseek-v4-flash` es
actualmente MEJOR que `deepseek-v4-pro` (los providers actualizan modelos bajo
el mismo ID — la premisa "pro > flash" de 2026-06-12 caducó). Flash pasa a ser
el modelo PRIMARIO de TODAS las superficies, incluidos tiers pagados y el
reviewer clínico risk-tier (`graph_orchestrator._REVIEWER_RISK_TIER_DEFAULT`).
Pro NO desaparece: queda exclusivamente como RED post-fallo (2º en cadena del
day-gen, fallback del planner con breaker independiente, escalada del corrector
quirúrgico — `MEALFIT_PRO_MODEL`), donde su valor es ser un modelo DISTINTO
con circuit breaker propio, no ser "mejor". Rollback sin redeploy:
`MEALFIT_MODEL_PAID_TIER=deepseek-v4-pro`.

Precios oficiales 2026-06 (por 1M tokens): flash $0.14 in / $0.28 out;
pro $0.435 in / $0.87 out. Ambos: 1M contexto, 384K max output, JSON mode,
function calling, thinking nativo (gestionado por el provider — el antiguo
`thinking_budget` de Gemini NO aplica y se swallow-ea en el wrapper).

Contratos:
  - `ChatDeepSeek` — drop-in del antiguo `ChatGoogleGenerativeAI`: acepta y
    descarta kwargs legacy (`google_api_key`, `safety_settings`,
    `thinking_budget`) y traduce `max_output_tokens` → `max_tokens`, para que
    los ~37 callsites migren sin cirugía de kwargs. API key SIEMPRE desde env
    `DEEPSEEK_API_KEY` (NUNCA hardcodeada — test blanket
    `test_p0_deepseek_migration.py` lo enforza).
  - `resolve_model_for_tier(tier)` / `resolve_model_for_user(user_id)` — el
    router. Fail-cheap: cualquier duda (guest, DB blip, tier desconocido)
    resuelve al modelo FREE. Un fallo de lookup jamás puede COSTAR dinero
    (simétrico a fail-secure en auth: acá el riesgo es billing, no IDOR).
  - Tier lookup con cache in-process TTL (`MEALFIT_TIER_CACHE_TTL_S`,
    default 300s) para no añadir un roundtrip DB por cada LLM call.

Knobs (auto-registrados en `_KNOBS_REGISTRY` vía `_env_*`):
  - `MEALFIT_DEEPSEEK_BASE_URL`  (default `https://api.deepseek.com`)
  - `MEALFIT_MODEL_FREE_TIER`    (default `deepseek-v4-flash`)
  - `MEALFIT_MODEL_PAID_TIER`    (default `deepseek-v4-flash`, P1-FLASH-PRIMARY)
  - `MEALFIT_TIER_CACHE_TTL_S`   (default 300, clamp [10, 3600])

Rollback operacional sin redeploy: ambos modelos son swappeables vía knob
(convención P3-PREVIEW-MODEL-KNOB). Si DeepSeek deprecia los IDs V4
(`deepseek-chat`/`deepseek-reasoner` legacy mueren 2026-07-24), basta con
setear los knobs al ID nuevo y reiniciar el worker.

Tooltip-anchor: P0-DEEPSEEK-MIGRATION.
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

# IDs oficiales del API DeepSeek (verificados contra api-docs.deepseek.com
# 2026-06-12). Los aliases legacy `deepseek-chat`/`deepseek-reasoner` se
# depredan el 2026-07-24 — NO usarlos como default.
DEEPSEEK_FLASH = "deepseek-v4-flash"
DEEPSEEK_PRO = "deepseek-v4-pro"

# [P1-NET-LUNA · P1-REVIEWER-TIER-MODELS · 2026-07-31] IDs OpenAI gpt-5.6 en
# uso: luna = red cross-provider del pipeline + reviewer clínico free;
# terra = reviewer clínico tiers pagados. SSOT del ID — no repetir literales.
GPT56_LUNA = "gpt-5.6-luna"
GPT56_TERRA = "gpt-5.6-terra"

# [P1-DEEPSEEK-THINKING-OFF · 2026-06-13] DeepSeek-V4 trae "thinking mode"
# (chain-of-thought) NATIVO ENCENDIDO por default. Para la generación de planes
# (tool-calling con `consultar_nutricion` + relleno de macros) el reasoning NO
# aporta calidad y multiplica la latencia: en el test E2E 2026-06-13 el skeleton
# (structured-output, thinking ya OFF) tardó 9.5s pero cada día (tool-calling,
# thinking ON) excedió el techo de 170s → TimeoutError → fallback matemático.
# Por eso lo desactivamos por DEFAULT en TODOS los calls (no solo structured
# output). Knob de rollback sin redeploy: `MEALFIT_DEEPSEEK_THINKING=on`
# re-activa el reasoning (p.ej. si el reviewer clínico lo necesita). Cualquier
# `extra_body.thinking` explícito del callsite SIEMPRE gana sobre este default.
_DEEPSEEK_THINKING_DISABLED = not _env_bool("MEALFIT_DEEPSEEK_THINKING", False)

# Tiers de pago canónicos (columna `user_profiles.plan_tier`, ver
# routers/billing.py P0-BILLING-1). Todo lo demás («gratis», NULL, guests,
# strings corruptos) enruta a FREE — fail-cheap.
PAID_TIERS = frozenset({"basic", "plus", "ultra"})

_MISSING_KEY_PLACEHOLDER = "MISSING_DEEPSEEK_API_KEY"
_warned_missing_key = False


def _deepseek_base_url() -> str:
    """Base URL OpenAI-compatible de DeepSeek. Knob para entornos proxy/test."""
    return _env_str("MEALFIT_DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _is_deepseek_provider(base_url: Optional[str] = None) -> bool:
    """True si el `base_url` efectivo apunta a DeepSeek.

    [P1-DEEPSEEK-ONLY-RESTORE · 2026-07-04] El `extra_body={"thinking": ...}`
    que este wrapper inyecta es un parámetro ESPECÍFICO del API DeepSeek: otros
    back-ends OpenAI-compatibles rechazan campos desconocidos con HTTP 400.
    Este guard evita inyectarlo si un entorno de test apunta el knob
    `MEALFIT_DEEPSEEK_BASE_URL` a un proxy/back-end distinto. DeepSeek es el
    ÚNICO provider soportado en producción — el plumbing multi-provider
    (override global de modelo + detección Ollama) fue eliminado a pedido del
    owner 2026-07-04; si se re-introduce un provider alterno, debe nacer con
    knob + test ancla propios.

    [P1-DEEPSEEK-PROVIDER-INSTANCE · 2026-07-28] La asunción anterior ("los
    callsites productivos NUNCA pasan un `base_url` propio") YA NO es cierta:
    `ChatDeepSeek` (subclase de `ChatOpenAI`) se construye cada vez más contra
    back-ends OpenAI-compatibles DISTINTOS de DeepSeek (p.ej. el meal-photo
    scanner apuntado a OpenAI). Llamar esta función sin argumento SIEMPRE
    inspecciona el knob global `MEALFIT_DEEPSEEK_BASE_URL` — que SIEMPRE
    contiene "deepseek" — y retorna `True` sin importar a dónde apunte la
    instancia real. Todo callsite que resuelva el provider de una instancia YA
    construida (p.ej. `ChatDeepSeek.with_structured_output`) DEBE pasar el
    `base_url` efectivo de ESA instancia (`self.openai_api_base` en
    `ChatOpenAI` — `base_url` NO es atributo de instancia, solo kwarg del
    constructor), nunca confiar en el default implícito. El fallback al knob
    global sigue vigente para: (a) el path del constructor (`base_url` aún no
    resuelto, puede venir `None`) y (b) cualquier caller sin forma de leer el
    atributo de instancia — preserva el comportamiento DeepSeek-only de hoy.
    """
    resolved = (base_url or _deepseek_base_url() or "").lower()
    return "deepseek" in resolved


def _deepseek_api_key() -> str:
    """API key desde env `DEEPSEEK_API_KEY`.

    Si falta, retorna un placeholder NO-vacío: la construcción del cliente
    nunca debe tirar el boot (hay LLMs construidos a module-import, e.g.
    `agent.py::llm`); la invocación fallará con 401 explícito y el error-log
    de boot (una sola vez) le dice al operador exactamente qué falta. Misma
    semántica que tenía el constructor legacy con la key ausente (None).
    """
    global _warned_missing_key
    key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if key:
        return key
    if not _warned_missing_key:
        logger.error(
            "❌ [LLM-PROVIDER] DEEPSEEK_API_KEY no configurada en el entorno. "
            "Toda invocación LLM fallará con 401 hasta setearla "
            "(.env local / env vars del VPS) y reiniciar el worker."
        )
        _warned_missing_key = True
    return _MISSING_KEY_PLACEHOLDER


def model_free_tier() -> str:
    """Modelo para tier `gratis`, guests y fallback. Default V4 Flash."""
    return _env_str("MEALFIT_MODEL_FREE_TIER", DEEPSEEK_FLASH) or DEEPSEEK_FLASH


def model_paid_tier() -> str:
    """Modelo para tiers `basic`/`plus`/`ultra`. Default V4 Flash
    ([P1-FLASH-PRIMARY · 2026-07-31]: el owner midió que flash es actualmente
    mejor que pro; era `DEEPSEEK_PRO` desde P0-DEEPSEEK-MIGRATION). Rollback:
    `MEALFIT_MODEL_PAID_TIER=deepseek-v4-pro`."""
    return _env_str("MEALFIT_MODEL_PAID_TIER", DEEPSEEK_FLASH) or DEEPSEEK_FLASH


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
#   - google_api_key: la key ahora viene de DEEPSEEK_API_KEY (env).
#   - safety_settings: HarmCategory/HarmBlockThreshold eran Gemini-only.
#     (La decisión P3-CHAT-SAFETY-OFF queda obsoleta: DeepSeek no aplica
#     content-filters configurables client-side.)
#   - thinking_budget: el thinking de DeepSeek-V4 es nativo del modelo y no
#     expone budget por request en el API OpenAI-compatible. El costo que
#     motivaba el cap (reasoning a $9/M en Gemini) no existe: output V4
#     cuesta $0.28–0.87/M, 10-30× menos.
_LEGACY_SWALLOWED_KWARGS = (
    "google_api_key",
    "safety_settings",
    "thinking_budget",
    "convert_system_message_to_human",
)


class ChatDeepSeek(ChatOpenAI):
    """Cliente chat DeepSeek (OpenAI-compatible) — reemplazo 1:1 del antiguo
    `ChatGoogleGenerativeAI`.

    Diferencias gestionadas internamente:
      - `model`: ID DeepSeek (`deepseek-v4-flash` / `deepseek-v4-pro`).
      - `api_key`/`base_url`: defaults desde env/knob; los callsites NO los
        pasan (y NUNCA hardcodean la key — test blanket).
      - `max_output_tokens` → `max_tokens` (naming OpenAI).
      - `stream_usage=True` por default: DeepSeek soporta `include_usage` en
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
        # [P1-DEEPSEEK-ONLY-RESTORE · 2026-07-04] El override global de modelo
        # (`MEALFIT_LLM_MODEL_OVERRIDE`, [MULTI-PROVIDER · 2026-07-01]) fue
        # ELIMINADO: permitía colapsar TODOS los modelos —incluido el reviewer
        # médico risk-tier— a un provider de test (Gemini), degradando el gate
        # clínico fail-secure en silencio. El routing de modelos vive SOLO en
        # los knobs MEALFIT_*_MODEL per-feature + el router por tier.
        if max_output_tokens is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_output_tokens
        kwargs.setdefault("stream_usage", True)
        # [P1-DEEPSEEK-THINKING-OFF · 2026-06-13] Desactiva thinking mode por
        # default en TODOS los runnables (no solo structured-output). Merge
        # no-destructivo: si el callsite ya pasó `extra_body.thinking`, gana.
        if _DEEPSEEK_THINKING_DISABLED and _is_deepseek_provider(base_url):
            _extra = dict(kwargs.get("extra_body") or {})
            _extra.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = _extra
        super().__init__(
            model=model,
            api_key=api_key or _deepseek_api_key(),
            base_url=base_url or _deepseek_base_url(),
            **kwargs,
        )

    def with_structured_output(self, schema=None, **kwargs):
        """Override del default de langchain-openai, calibrado EN VIVO contra
        el API DeepSeek (2026-06-12):

        1. `method="json_schema"` (default de langchain-openai 1.3) emite
           `response_format={"type": "json_schema", ...}` → DeepSeek responde
           `400 This response_format type is unavailable`. Se usa
           `method="function_calling"` (tools API, soportado).
        2. El thinking mode (default-ON en V4) NO soporta el `tool_choice`
           forzado que function_calling necesita → `400 Thinking mode does
           not support this tool_choice`. Para estos runnables se desactiva
           el thinking via `extra_body={"thinking": {"type": "disabled"}}` —
           structured output es relleno de schema, no requiere reasoning, y
           de paso se ahorran los reasoning-tokens (facturan como output).

        Cubre los ~15 callsites `.with_structured_output(...)` del pipeline
        sin tocarlos. Un caller puede pasar `method=` explícito si lo necesita.

        [P1-DEEPSEEK-PROVIDER-INSTANCE · 2026-07-28] El guard resuelve el
        provider de ESTA instancia (`self.openai_api_base`), NO del knob
        global: `_is_deepseek_provider()` sin argumento inspecciona
        `MEALFIT_DEEPSEEK_BASE_URL` (que SIEMPRE contiene "deepseek") y
        retornaba `True` para TODA instancia sin importar a dónde apuntara de
        verdad — inyectando `thinking` (parámetro DeepSeek-only) contra
        back-ends que lo rechazan con `400 Unknown parameter: 'thinking'`
        (medido en vivo: el meal-photo scanner con `ChatDeepSeek` contra
        OpenAI). `openai_api_base` es el atributo real que `ChatOpenAI`
        persiste desde el kwarg `base_url` del constructor (verificado
        empíricamente — `base_url` NO es atributo de instancia). Si un futuro
        langchain-openai renombra el atributo, `getattr(..., None)` cae a
        `None` y `_is_deepseek_provider(None)` reproduce el comportamiento de
        HOY (inspecciona el knob global) — el path DeepSeek-only de
        producción (~15 callsites) no puede regresar por este cambio.
        """
        kwargs.setdefault("method", "function_calling")
        base = self
        _instance_base_url = getattr(self, "openai_api_base", None)
        if kwargs["method"] == "function_calling" and _is_deepseek_provider(_instance_base_url):
            merged_extra = dict(self.extra_body or {})
            merged_extra.setdefault("thinking", {"type": "disabled"})
            base = self.model_copy(update={"extra_body": merged_extra})
        return ChatOpenAI.with_structured_output(base, schema, **kwargs)


# ============================================================
# [P1-DAYGEN-LUNA-CANARY · 2026-07-26] Fábrica de LLM por PROVEEDOR.
# ============================================================
# `_build_day_llm` (graph_orchestrator) construía siempre `ChatDeepSeek`, aunque su propio
# comentario decía "provider correcto por prefijo" — la intención estaba escrita y no
# implementada. Con un modelo OpenAI en el chain, ese hardcode lo mandaría al base_url de
# DeepSeek con la key equivocada.
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


def is_openai_model(model: str) -> bool:
    """¿El ID pertenece al API de OpenAI (y no a DeepSeek)?"""
    m = str(model or "").strip().lower()
    return any(m.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def _openai_api_key() -> str:
    """Key de OpenAI desde el entorno. Fail-loud: mejor una excepción que un call silencioso al
    proveedor equivocado. NUNCA argumento del callsite — mismo contrato que enforza
    `test_p0_deepseek_migration.py`."""
    _k = os.environ.get("OPENAI_API_KEY")
    if not _k:
        raise RuntimeError("modelo OpenAI pedido sin OPENAI_API_KEY en el entorno")
    return _k


def build_chat_llm(model: str, **kwargs):
    """Devuelve el cliente chat del proveedor que corresponde al `model`.

    OpenAI → `ChatOpenAI` con `OPENAI_API_KEY` y base por defecto. Resto → `ChatDeepSeek`
    (que ya inyecta su propia key/base).

    ⚠️ [P1-LUNA-USAGE-BLIND · 2026-07-26] Devuelve las clases BASE: sin backpressure y **sin
    contabilidad de costo**. Dentro del pipeline de generación NO se usa esta fábrica — allí van
    `graph_orchestrator.ChatDeepSeek` / `ChatOpenAIInstrumented`, que añaden el mixin. Construir
    el day-gen con esta fábrica dejó `llm_usage_events` sin una sola fila de `day_generator`
    (el nodo más caro) durante la primera corrida del canario Luna.
    """
    if is_openai_model(model):
        return ChatOpenAI(model=model, api_key=_openai_api_key(), **kwargs)
    return ChatDeepSeek(model=model, **kwargs)
