"""[P1-VISION-LUNA · 2026-07-28] Test ancla.

Doc: backend/docs/vision_luna.md
"""
import asyncio
import io
import logging
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_prep import prepare_image_for_vision  # noqa: E402

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as fh:
        return fh.read()


def _png(w, h, mode="RGB"):
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _wh(raw):
    from PIL import Image
    return Image.open(io.BytesIO(raw)).size


def test_foto_de_movil_se_reduce_al_cap():
    """3024x4032 son 12.406 tokens de entrada ($0,0130); a 1024 son 1.374
    ($0,0020). 6,6x, medido contra la API real el 2026-07-28."""
    raw = _png(3024, 4032)
    out, info = prepare_image_for_vision(raw, max_side=1024)
    assert max(_wh(out)) == 1024
    assert info["resized"] is True
    assert len(out) < len(raw)


def test_mantiene_el_aspecto():
    out, _ = prepare_image_for_vision(_png(4000, 2000), max_side=1024)
    w, h = _wh(out)
    assert (w, h) == (1024, 512), (w, h)


def test_imagen_pequena_pasa_intacta():
    """Re-encodear algo que ya cabe solo pierde calidad sin ahorrar nada."""
    raw = _png(800, 600)
    out, info = prepare_image_for_vision(raw, max_side=1024)
    assert out == raw
    assert info["resized"] is False


def test_png_con_alpha_no_revienta_el_encoder_jpeg():
    """RGBA -> JPEG lanza OSError si no se convierte a RGB antes."""
    out, info = prepare_image_for_vision(_png(2000, 2000, mode="RGBA"), max_side=1024)
    assert max(_wh(out)) == 1024
    assert info["resized"] is True


@pytest.mark.parametrize("basura", [b"", b"no soy una imagen", b"\x00" * 100, None])
def test_fail_open_devuelve_los_bytes_originales(basura):
    """Un escaneo NUNCA debe morir porque el resize fallo."""
    out, info = prepare_image_for_vision(basura, max_side=1024)
    assert out == basura
    assert info["resized"] is False
    assert info["skipped_reason"]


def test_knob_max_side_con_clamp():
    from image_prep import vision_max_side_px
    assert vision_max_side_px() == 1024
    os.environ["MEALFIT_VISION_MAX_SIDE_PX"] = "99999"
    try:
        assert vision_max_side_px() == 1024, "fuera de [256,4096] debe caer al default"
    finally:
        os.environ.pop("MEALFIT_VISION_MAX_SIDE_PX", None)


# ---------------------------------------------------------------------------
# Task 2 [P1-VISION-LUNA · 2026-07-28 → simplificado P1-VISION-NO-LOCAL ·
# 2026-07-28]: cablear el resize + despacho al ÚNICO provider en
# vision_agent.py. Tests behaviorales (spy sobre los puntos de extensión
# reales del módulo) en vez de grep de substrings — si se invierte el
# comportamiento (orden, propagación del error), estos deben ponerse rojos.
# La cascada de fallback (`MEALFIT_VISION_FALLBACK_PROVIDER`, reintento
# contra `ollama`) fue ELIMINADA en P1-VISION-NO-LOCAL junto con el provider
# local — sus tests vivían acá y se retiraron con ella.
# ---------------------------------------------------------------------------

def _openai_compatible_env(monkeypatch):
    """Config mínima para que `is_vision_enabled()` acepte el provider cloud."""
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://example.test/v1")


def test_resize_invocado_antes_del_despacho_al_provider(monkeypatch):
    """Behavioral, NO source-parsing: instrumenta el punto de resize real
    (`prepare_image_for_vision`, importado a vision_agent) y el punto de
    despacho real (`_dispatch_openai_compatible_vision` — el ÚNICO provider
    tras P1-VISION-NO-LOCAL) y verifica el ORDEN de invocación en tiempo de
    ejecución. Si alguien mueve el resize después del despacho, esto se pone
    rojo."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)

    calls = []

    def _spy_prepare(raw, max_side=None):
        calls.append("resize")
        return raw, {
            "original_bytes": len(raw) if raw else 0, "sent_bytes": 0,
            "original_wh": None, "sent_wh": None, "resized": False,
            "skipped_reason": None,
        }

    async def _spy_dispatch(image_bytes):
        calls.append("dispatch")
        return {"is_food": False, "description": "ok", "meal_name": "",
                "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}

    monkeypatch.setattr(va, "prepare_image_for_vision", _spy_prepare)
    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _spy_dispatch)

    asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake-jpeg"))

    assert calls == ["resize", "dispatch"], (
        f"el resize debe ocurrir ANTES del despacho al provider: {calls}"
    )


def test_sin_cascada_el_resultado_del_unico_provider_se_retorna_tal_cual(monkeypatch):
    """[P1-VISION-NO-LOCAL · 2026-07-28] La cascada de fallback
    (`MEALFIT_VISION_FALLBACK_PROVIDER`, provider `ollama`) fue eliminada —
    no queda un segundo provider al que reintentar. `process_image_with_vision`
    debe ser un passthrough directo: el resultado (éxito o `analysis_failed`)
    del ÚNICO provider se retorna EXACTAMENTE igual, sin envolver ni mezclar."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)

    primary_failure = {
        "description": "Error analizando imagen (marca-primario-original).",
        "is_food": False, "analysis_failed": True, "meal_name": "",
        "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0,
    }

    async def _fail_openai(image_bytes):
        return dict(primary_failure)

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _fail_openai)

    out = asyncio.run(va.process_image_with_vision(b"\xff\xd8\xfffake"))

    assert out == primary_failure


def test_cascada_infra_eliminada_de_vision_agent():
    """[P1-VISION-NO-LOCAL · 2026-07-28] Blanket anti-regresión: la cascada
    completa (knob + resolver + dispatcher genérico) no debe reaparecer como
    símbolo del módulo — si alguien la reintroduce, este test lo cacha antes
    que el blanket parser genérico de test_p1_vision_no_local.py (que solo
    busca 'ollama'/'gemma', no estos nombres específicos de la cascada)."""
    import vision_agent as va
    for symbol in (
        "_vision_fallback_provider", "_vision_cascade_target",
        "_dispatch_vision_provider", "_vision_result_unusable",
    ):
        assert not hasattr(va, symbol), (
            f"vision_agent.{symbol} debía eliminarse junto con la cascada "
            f"(P1-VISION-NO-LOCAL) — no queda un segundo provider al que cascar."
        )


def test_telemetria_de_resize_se_loguea_a_info(monkeypatch, caplog):
    """La auditoría de ahorro en prod depende de esta línea existiendo y
    siendo greppable — verifica que el marker + los campos de telemetría
    llegan al logger a nivel INFO."""
    import vision_agent as va
    _openai_compatible_env(monkeypatch)

    async def _ok_openai(image_bytes):
        return {"is_food": False, "description": "ok", "meal_name": "",
                "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}

    monkeypatch.setattr(va, "_dispatch_openai_compatible_vision", _ok_openai)

    caplog.set_level(logging.INFO, logger="vision_agent")
    asyncio.run(va.process_image_with_vision(_png(64, 64)))

    telemetry = [
        r.message for r in caplog.records
        if r.name == "vision_agent" and "[P1-VISION-LUNA] resize" in r.message
    ]
    assert telemetry, "falta el log de telemetría del resize (marker greppable)"
    line = telemetry[0]
    for key in ("original_bytes=", "sent_bytes=", "original_wh=",
                "sent_wh=", "resized=", "skipped_reason="):
        assert key in line, f"telemetría incompleta, falta {key!r} en: {line}"


# ---------------------------------------------------------------------------
# Bug real cazado en vivo contra la API de OpenAI (VPS, 2026-07-28):
# `_dispatch_openai_compatible_vision` construía SIEMPRE `ChatDeepSeek`
# (subclase de `ChatOpenAI` que inyecta `extra_body={"thinking": ...}`,
# parámetro PROPIO de DeepSeek) -- con `MEALFIT_VISION_MODEL=gpt-5.6-luna`
# el API real de OpenAI rechazaba el request con
# `400 Unknown parameter: 'thinking'`. Fix: el cliente se elige por MODELO
# (`is_openai_model`, mismo criterio que `llm_provider.build_chat_llm`).
# ---------------------------------------------------------------------------

def test_dispatch_por_modelo_no_hardcodea_chatdeepseek(monkeypatch):
    """El path OpenAI-compatible debe elegir el cliente por MODELO, no
    hardcodear uno. Si alguien revierte a `ChatDeepSeek` incondicional (el
    bug real que causó el HTTP 400 'Unknown parameter: thinking' contra la
    API de OpenAI), este test se pone rojo."""
    import vision_agent as va

    calls = []

    class _FakeResponse:
        description = "ok"
        is_food = False
        meal_name = ""
        calories = 0
        protein = 0
        carbs = 0
        healthy_fats = 0

    class _FakeBoundLLM:
        async def ainvoke(self, messages):
            return _FakeResponse()

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(("ChatOpenAI", kwargs))

        def with_structured_output(self, schema):
            return _FakeBoundLLM()

    class _FakeChatDeepSeek:
        def __init__(self, **kwargs):
            calls.append(("ChatDeepSeek", kwargs))

        def with_structured_output(self, schema):
            return _FakeBoundLLM()

    monkeypatch.setattr(va, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(va, "ChatDeepSeek", _FakeChatDeepSeek)
    monkeypatch.setenv("VISION_API_KEY", "test-key-not-real")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://api.openai.com/v1")

    # Modelo OpenAI real (Luna) -> DEBE usar ChatOpenAI, NUNCA ChatDeepSeek.
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5.6-luna")
    asyncio.run(va._dispatch_openai_compatible_vision(b"fake-jpeg-bytes"))
    assert [c[0] for c in calls] == ["ChatOpenAI"], (
        f"gpt-5.6-luna debe despachar a ChatOpenAI, no a ChatDeepSeek: {calls}"
    )

    calls.clear()
    # Modelo NO-OpenAI (DeepSeek u otro proveedor OpenAI-compatible) ->
    # sigue siendo ChatDeepSeek (el wrapper correcto para esos providers).
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "deepseek-v4-flash")
    asyncio.run(va._dispatch_openai_compatible_vision(b"fake-jpeg-bytes"))
    assert [c[0] for c in calls] == ["ChatDeepSeek"], (
        f"un modelo no-OpenAI debe seguir usando ChatDeepSeek: {calls}"
    )


def test_api_key_precedence_vision_key_gana_sobre_openai_key(monkeypatch):
    """VISION_API_KEY (knob específico de visión) SIEMPRE gana sobre la key
    genérica del proveedor; si no está seteada, cae a `_openai_api_key()`
    (OPENAI_API_KEY) para modelos OpenAI."""
    import vision_agent as va

    calls = []

    class _FakeBoundLLM:
        async def ainvoke(self, messages):
            class _R:
                description = "ok"; is_food = False; meal_name = ""
                calories = 0; protein = 0; carbs = 0; healthy_fats = 0
            return _R()

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def with_structured_output(self, schema):
            return _FakeBoundLLM()

    monkeypatch.setattr(va, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setenv("MEALFIT_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://api.openai.com/v1")

    # Con VISION_API_KEY seteada, debe ganar sobre OPENAI_API_KEY.
    monkeypatch.setenv("VISION_API_KEY", "vision-specific-key")
    monkeypatch.setenv("OPENAI_API_KEY", "generic-openai-key")
    asyncio.run(va._dispatch_openai_compatible_vision(b"fake"))
    assert calls[-1]["api_key"] == "vision-specific-key"

    # Sin VISION_API_KEY, cae a OPENAI_API_KEY (_openai_api_key()).
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    asyncio.run(va._dispatch_openai_compatible_vision(b"fake"))
    assert calls[-1]["api_key"] == "generic-openai-key"


# ---------------------------------------------------------------------------
# [P1-VISION-NO-LOCAL · 2026-07-28] El bug real que `_ollama_model_name()`
# cerraba (retrocompat del modelo dedicado de Ollama vs el knob cloud
# compartido durante la cascada) dejó de aplicar: no queda accessor, knob
# `MEALFIT_OLLAMA_VISION_MODEL`, ni cascada a la que ese modelo alimentara.
# Los 4 tests que vivían acá (nunca-hereda-el-modelo-cloud, retrocompat-
# primario, dedicado-gana-sobre-ambos, default-sin-knob) se retiraron junto
# con `_ollama_model_name()` — ver test_cascada_infra_eliminada_de_vision_agent
# arriba y el blanket parser de test_p1_vision_no_local.py para la cobertura
# de que el símbolo no reaparezca.
# ---------------------------------------------------------------------------


def test_ollama_model_name_symbol_removed():
    """Anti-regresión puntual: `_ollama_model_name` no debe reaparecer como
    símbolo de vision_agent — su única razón de existir era la cascada
    eliminada en P1-VISION-NO-LOCAL."""
    import vision_agent as va
    assert not hasattr(va, "_ollama_model_name")


# ---------------------------------------------------------------------------
# Task 3 [P1-VISION-LUNA · 2026-07-28]: sacar la visión del libro de cuota de
# planes. El bug que se cierra: `diary.py` quemaba un crédito de plan
# (`log_api_usage` → `api_usage`) por cada scan cuando el provider era CLOUD
# pago — con Luna encendido eso agota el cap gratis (15/mes) en 5 días. Los
# tests parsean el bloque real alrededor del gate `if actual_user_id and
# actual_user_id != session_id:` en vez de grepear el archivo entero, para no
# dar falso-verde si el texto viejo sobrevive en un comentario/docstring en
# otra parte del archivo.
#
# [P1-VISION-NO-LOCAL · 2026-07-28] El marker INCLUÍA `and not
# is_vision_local()` — ese sufijo se eliminó junto con `is_vision_local()`
# (sin provider local, la condición nunca podía ser False). Re-anclado al
# gate simplificado.
# ---------------------------------------------------------------------------

_VISION_GATE_MARKER = (
    "if actual_user_id and actual_user_id != session_id:"
)


def _vision_gate_window(src, chars=900):
    """Ventana de código INMEDIATAMENTE posterior al gate
    `if actual_user_id and actual_user_id != session_id:` — el mismo gate
    que ya existía pre-P1-VISION-LUNA (P1-MEAL-SCAN-GEMMA), simplificado en
    P1-VISION-NO-LOCAL (perdió el sufijo `and not is_vision_local()` al
    eliminarse el provider local). Acotar la ventana (en vez de buscar en el
    archivo entero) evita que un comentario explicativo en OTRA parte del
    archivo (p.ej. el bloque de docstring que narra el bug cerrado) haga
    pasar el test en falso."""
    i = src.find(_VISION_GATE_MARKER)
    assert i != -1, (
        "no se encontró el gate `if actual_user_id and actual_user_id != "
        "session_id:` en diary.py — renombrado o eliminado; los tests de "
        "abajo necesitan re-anclarse."
    )
    return src[i + len(_VISION_GATE_MARKER):i + len(_VISION_GATE_MARKER) + chars]


def test_diary_upload_ya_no_quema_credito_de_plan_por_scan_cloud():
    """El call site original NO era `log_api_usage(...)` -- era
    `asyncio.to_thread(log_api_usage, actual_user_id, "llm_vision")`: la
    función se pasaba POR REFERENCIA (bare, sin paréntesis) como argumento
    de `to_thread`, que la invoca después en el thread pool. Un test que
    solo buscara el substring `log_api_usage(` NUNCA habría atrapado ese
    patrón -- verificado manualmente reconstruyendo el call site viejo y
    confirmando que `"log_api_usage(" in texto` da False sobre él. Por eso
    se cubren AMBOS patrones de reaparición (referencia bare a to_thread Y
    llamada directa), más la ausencia del import -- sin import, cualquier
    referencia bare a `log_api_usage` sería un NameError en runtime, así
    que el import es la señal más fuerte."""
    src = _src("routers/diary.py")
    assert "to_thread(log_api_usage" not in src, (
        "el patrón exacto del bug original -- log_api_usage pasado POR "
        "REFERENCIA a asyncio.to_thread -- reapareció. Quemaría crédito de "
        "plan de nuevo por cada scan con provider cloud pago."
    )
    assert "log_api_usage(" not in src, (
        "log_api_usage ya no debe invocarse (llamada directa tampoco) "
        "desde diary.py -- el gasto de visión vive en llm_usage_events "
        "vía log_llm_usage_event."
    )
    assert re.search(r"^\s*from\s+\S+\s+import\s+[^#\n]*\blog_api_usage\b",
                      src, re.MULTILINE) is None, (
        "log_api_usage tampoco debe seguir importado -- sin el import, "
        "cualquier referencia bare al nombre sería un NameError en "
        "runtime; el import muerto es lo que habilitaría reintroducir el "
        "bug sin que ni siquiera falle al arrancar."
    )


def test_diary_upload_llama_log_llm_usage_event_en_el_gate_vision():
    """Behavioral/parser: el MISMO gate que antes llamaba log_api_usage debe
    ahora invocar log_llm_usage_event, vía asyncio.to_thread (mismo patrón
    async-offload del resto del archivo), con node="vision_scan" y el modelo
    real (`_vision_model_name()`, el accessor de vision_agent -- NO un
    string hardcodeado, que se desincronizaría del knob
    MEALFIT_VISION_MODEL). Si el nodo o el offload desaparecen, este test se
    pone rojo."""
    window = _vision_gate_window(_src("routers/diary.py"))
    assert re.search(r"asyncio\.to_thread\(\s*\n?\s*log_llm_usage_event\s*,", window), (
        f"log_llm_usage_event debe invocarse via asyncio.to_thread dentro "
        f"del gate de visión. Ventana analizada:\n{window}"
    )
    assert re.search(r'node\s*=\s*"vision_scan"', window), (
        "falta node=\"vision_scan\" -- sin esto el evento de costo no es "
        "distinguible de cualquier otra llamada LLM en llm_usage_events."
    )
    assert "model=_vision_model_name()" in window, (
        "el modelo debe venir del accessor vivo de vision_agent, no de un "
        "literal hardcodeado (se desincronizaría de MEALFIT_VISION_MODEL)."
    )


def test_diary_upload_import_log_llm_usage_event_desde_facade_db():
    """[P3-DB-IMPORTS-FACADE] `log_llm_usage_event` debe entrar por la
    fachada `from db import ...` (convención del repo para call sites
    nuevos) -- no por `from db_profiles import log_llm_usage_event` directo."""
    src = _src("routers/diary.py")
    assert re.search(r"from\s+db\s+import\s*\([^)]*\blog_llm_usage_event\b",
                      src, re.DOTALL), (
        "log_llm_usage_event debe importarse desde la fachada `db`, no "
        "desde `db_profiles` directo."
    )
    assert re.search(r"from\s+db_profiles\s+import\s*\([^)]*\blog_llm_usage_event\b",
                      src, re.DOTALL) is None


def test_get_monthly_api_usage_sigue_sin_filtro_de_endpoint():
    """ANCLA anti-regresión, actualizada por [P1-COACH-METER · 2026-08-11].

    Contrato ORIGINAL (P1-VISION-LUNA Task 3): cero filtros por endpoint —
    un endpoint que quisiera facturación propia debía usar log_llm_usage_event,
    no reinterpretar el libro compartido. Contrato VIGENTE: el medidor se
    partió en dos A PROPÓSITO (el chat cobraba un crédito de generación
    ~96x más caro que su costo), y el espíritu del ancla se conserva en
    forma nueva:

      1. kind="generation" (el DEFAULT) va en LISTA NEGATIVA
         (`<> 'llm_chat'`): un endpoint nuevo cuenta CARO por defecto.
         Un filtro positivo aquí dejaría endpoints gratis por olvido —
         el modo de fallo que el ancla original prevenía.
      2. `'llm_chat'` es el ÚNICO literal de endpoint permitido en el
         cuerpo. Si aparece un segundo, alguien está whitelisteando un
         endpoint fuera del medidor de generación — revertir y usar
         log_llm_usage_event en el call site nuevo."""
    src = _src("db_profiles.py")
    m = re.search(
        r"^def get_monthly_api_usage\(.*?(?=^def )", src, re.MULTILINE | re.DOTALL
    )
    assert m, "get_monthly_api_usage no encontrada en db_profiles.py."
    body = m.group(0)
    assert 'kind: str = "generation"' in body, (
        "el default del medidor debe ser generation — el caro. Un default "
        "coach regalaría créditos de plan a cualquier caller que olvide "
        "el argumento."
    )
    assert "COALESCE(endpoint, '') <> 'llm_chat'" in body, (
        "la rama generation debe ir en LISTA NEGATIVA (<> 'llm_chat'): en "
        "positivo, un endpoint nuevo queda GRATIS por olvido. Y el COALESCE "
        "importa: endpoint NULL debe contar como generación."
    )
    _literales = set(re.findall(r"endpoint\s*(?:=|<>)\s*'([^']+)'", body))
    assert _literales == {"llm_chat"}, (
        f"literales de endpoint en el libro de cuota: {sorted(_literales)} — "
        "solo 'llm_chat' está permitido. Un segundo literal es un endpoint "
        "colándose fuera del medidor compartido de generación; usar "
        "log_llm_usage_event en su call site en vez de tocar este SELECT."
    )


# ---------------------------------------------------------------------------
# CLAUDE.md: la fila "Historial-quota-exemption" de `/api/diary/upload`
# tiene que reflejar el nuevo hecho (costo vive en llm_usage_events) en LAS
# DOS copias (backend/CLAUDE.md, la única con git, y el espejo de
# workspace-root fuera de cualquier repo) y ambas deben seguir idénticas --
# mismo patrón que test_p1_diary_editable.py::test_claude_md_copies_stay_in_sync.
# ---------------------------------------------------------------------------

_ROOT = os.path.join(_BACKEND, "..")


def _claude_md(which):
    path = os.path.join(_BACKEND, "CLAUDE.md") if which == "backend" \
        else os.path.join(_ROOT, "CLAUDE.md")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _diary_upload_row(claude_md):
    """Extrae la fila de la tabla Historial-quota-exemption para
    `/api/diary/upload` (una sola línea `| ... |`) -- pinchar la fila
    específica, no "en algún lado del archivo de 50k chars", para que el
    test no dé falso-verde si el texto aparece en otra sección."""
    m = re.search(r"^\|\s*`/api/diary/upload`.*\|$", claude_md, re.MULTILINE)
    assert m, "no se encontró la fila de `/api/diary/upload` en Historial-quota-exemption."
    return m.group(0)


@pytest.mark.parametrize("which", ["backend", "workspace-root"])
def test_claude_md_diary_upload_row_documenta_llm_usage_events(which):
    row = _diary_upload_row(_claude_md(which))
    assert "P1-VISION-LUNA" in row, f"{which}: falta el marker P1-VISION-LUNA en la fila."
    assert "llm_usage_events" in row, (
        f"{which}: la fila debe decir que el costo de un scan cloud vive "
        f"en llm_usage_events, no solo que log_api_usage desapareció."
    )
    assert "log_llm_usage_event" in row
    assert "vision_scan" in row
    # La fila vieja decía "log_api_usage solo si el provider es cloud pago" --
    # eso ahora es FALSO (nunca se llama, ni local ni cloud). Si el texto
    # viejo sobrevive literal, la fila miente sobre el comportamiento actual.
    assert "log_api_usage solo si" not in row


def test_claude_md_copies_stay_in_sync_p1_vision_luna():
    """Redundante a propósito con test_p1_diary_editable.py -- el bug real
    (P1-DIARY-EDITABLE) fue exactamente "una sola copia se editó"; este
    P-fix toca la MISMA fila y merece su propio guard independiente del
    archivo de test de otro P-fix."""
    assert _claude_md("backend") == _claude_md("workspace-root"), (
        "Las dos copias de CLAUDE.md divergieron tras el edit de "
        "P1-VISION-LUNA -- sincronizar ambas (solo backend/CLAUDE.md vive "
        "en git)."
    )


# ---------------------------------------------------------------------------
# Task 4 [P1-VISION-LUNA · 2026-07-28]: doc canónica + marker.
#
# El marker `_LAST_KNOWN_PFIX` en app.py DEBE quedar en "P1-VISION-LUNA ·
# 2026-07-28" o más nuevo. El owner cierra P-fixes en PARALELO sobre esta
# rama (el marker cambió dos veces solo en la sesión de hoy, de
# P1-DIARY-EDITABLE a P1-RENAL-TRIM-RESOLVABLE, ANTES de que este Task 4
# tocara nada) -- un `assert marker.startswith("P1-VISION-LUNA")` se rompería
# en horas apenas el owner mergee su siguiente fix. La regla correcta es por
# FECHA: aceptar el marker propio O cualquiera cuya fecha sea >= la nuestra,
# mirror de `test_marker_bumped` en test_p2_audit_v5_batch.py:60-68.
# ---------------------------------------------------------------------------

_OUR_MARKER_SLUG = "P1-VISION-LUNA"
_OUR_MARKER_DATE = "2026-07-28"


def _marker_supersedes_or_matches(marker: str) -> bool:
    """True si `marker` es el propio de este P-fix O tiene fecha >= la
    nuestra. Extraído a función PURA (sin leer app.py) para poder probar el
    comportamiento contra markers SINTÉTICOS -- un `startswith` habría
    rechazado cualquier P-fix futuro con OTRO nombre, sin importar la fecha;
    esta función lo acepta correctamente si la fecha es igual o posterior."""
    if _OUR_MARKER_SLUG in marker:
        return True
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    if not fecha:
        return False
    return fecha.group(1) >= _OUR_MARKER_DATE


def test_marker_bumped_to_vision_luna_or_later():
    """El marker REAL en app.py debe ser el nuestro o uno posterior."""
    src = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX en app.py"
    marker = m.group(1)
    assert _marker_supersedes_or_matches(marker), (
        f"`_LAST_KNOWN_PFIX={marker!r}` no es P1-VISION-LUNA ni tiene fecha "
        f">= {_OUR_MARKER_DATE} -- stale respecto a este P-fix."
    )


@pytest.mark.parametrize("marker_viejo", [
    "P1-DIARY-EDITABLE · 2026-07-27",  # un día ANTES de la nuestra
    "P0-ALGUN-FIX-VIEJO · 2026-01-01",
])
def test_bite_marker_mas_viejo_es_rechazado(marker_viejo):
    """MUERDE de verdad: un marker de OTRO P-fix con fecha ANTERIOR a la
    nuestra debe fallar la validación. Esto es lo que un `startswith`
    también habría rechazado -- pero por la razón EQUIVOCADA (nombre, no
    fecha). Aquí probamos que la razón correcta (fecha) sigue rechazándolo."""
    assert not _marker_supersedes_or_matches(marker_viejo), (
        f"{marker_viejo!r} es más viejo que {_OUR_MARKER_DATE} y NO debería "
        f"pasar la validación de supersession."
    )


@pytest.mark.parametrize("marker_nuevo", [
    "P1-DIARY-EDITABLE · 2026-07-28",  # mismo día, OTRO P-fix
    "P2-ALGUN-FIX-FUTURO · 2026-08-01",  # posterior, OTRO P-fix
])
def test_bite_marker_de_otro_pfix_pero_mas_nuevo_es_aceptado(marker_nuevo):
    """MUERDE de verdad: un marker de OTRO P-fix (nombre completamente
    distinto) pero con fecha >= la nuestra DEBE pasar -- esto es EXACTAMENTE
    lo que un `assert marker.startswith("P1-VISION-LUNA")` NUNCA aceptaría,
    y es el escenario real: el owner cierra su siguiente P-fix en esta misma
    rama en paralelo y bumpea el marker a otro nombre sin que este test deba
    tocarse."""
    assert _marker_supersedes_or_matches(marker_nuevo), (
        f"{marker_nuevo!r} tiene fecha >= {_OUR_MARKER_DATE} -- un marker "
        f"de OTRO P-fix más nuevo debe pasar la validación de supersession."
    )


def test_marker_slug_matches_this_test_file():
    """Cross-link con test_p2_hist_audit_14_marker_test_link.py: el slug
    derivado de `P1-VISION-LUNA` (`p1_vision_luna`) debe matchear el nombre
    de ESTE archivo -- si no, el gate `test_marker_has_associated_test_file`
    del audit-14 fallaría en CI apenas se bumpee el marker."""
    slug = _OUR_MARKER_SLUG.replace("-", "_").lower()
    assert slug == "p1_vision_luna"
    this_file = os.path.basename(__file__)
    assert this_file.startswith(f"test_{slug}"), (
        f"slug {slug!r} no matchea el nombre de este archivo {this_file!r}"
    )
