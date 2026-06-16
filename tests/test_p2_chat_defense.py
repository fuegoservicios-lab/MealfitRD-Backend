"""[P2-CHAT-DEFENSE · 2026-05-19] Test parser-based: bundle de 3 P2 del
audit prod-readiness del Agente (2026-05-19):

  (1) BACKEND — sanitización output server-side (`_sanitize_chat_output_for_wire`)
      aplicada en los 3 puntos donde el LLM emite texto al cliente:
      `chat_with_agent` non-stream return, `chat_with_agent_stream` chunk
      yield, `chat_with_agent_stream` done yield. Defensa-en-profundidad
      sobre rehype-sanitize del frontend — si el cliente falla, los
      vectores XSS más comunes (script/iframe/object/embed/style/form,
      event handlers `on*=`, `javascript:` URIs) siguen neutralizados.

  (2) FRONTEND — `prefers-reduced-motion: reduce` declarado en index.css
      desactiva todas las animaciones a 0.01ms (terminan instantáneamente
      pero los onAnimationEnd handlers siguen disparándose) +
      scroll-behavior: auto (override del smooth scroll). WCAG 2.3.3
      (AAA) y accessibility guidelines.

  (3) FRONTEND — `fetchSessionMessages` clasifica errores por status y
      aplica backoff exponencial con jitter. Pre-fix: 2 retries
      hardcoded con delays fijos (sin distinguir 401/403 transitorio de
      500 server-error de 404 bug). Post-fix: helper
      `_classifyFetchSessionRetry` mapea status→policy (maxRetries +
      baseDelayMs); helper `_computeFetchBackoffMs` calcula exp con
      jitter ±10%.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p2_chat_defense`
matchea este archivo `test_p2_chat_defense.py`.

Tooltip-anchors: P2-CHAT-SANITIZE | P2-REDUCED-MOTION |
P2-FETCH-RETRY-ADAPTIVE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PY = _REPO_ROOT / "backend" / "agent.py"
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_INDEX_CSS = _REPO_ROOT / "frontend" / "src" / "index.css"


@pytest.fixture(scope="module")
def agent_py_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_css_src() -> str:
    return _INDEX_CSS.read_text(encoding="utf-8")


# =============================================================================
# (1) BACKEND — Sanitization helper + 3 callsites
# =============================================================================


def test_sanitize_helper_defined(agent_py_src: str):
    """`_sanitize_chat_output_for_wire` debe definirse a nivel de módulo
    en agent.py."""
    pattern = re.compile(
        r"def\s+_sanitize_chat_output_for_wire\s*\([^)]*\)\s*:",
    )
    assert pattern.search(agent_py_src), (
        "P2-CHAT-SANITIZE regresión: `_sanitize_chat_output_for_wire` no "
        "encontrado en agent.py. Este helper centraliza la neutralización "
        "de tags HTML peligrosas en el wire SSE — defensa-en-profundidad "
        "sobre rehype-sanitize del frontend."
    )


def test_sanitize_dangerous_tags_pattern(agent_py_src: str):
    """El regex `_DANGEROUS_HTML_TAG_RE` debe cubrir al menos las 6 tags
    XSS clásicas: script, iframe, object, embed, style, form."""
    pattern = re.compile(r"_DANGEROUS_HTML_TAG_RE\s*=\s*re\.compile\(\s*r?[\"']([^\"']+)[\"']")
    m = pattern.search(agent_py_src)
    assert m, "P2-CHAT-SANITIZE: `_DANGEROUS_HTML_TAG_RE` no encontrado."
    regex_str = m.group(1).lower()
    required = ["script", "iframe", "object", "embed", "style", "form"]
    missing = [t for t in required if t not in regex_str]
    assert not missing, (
        f"P2-CHAT-SANITIZE: tags peligrosas faltan del regex "
        f"`_DANGEROUS_HTML_TAG_RE`: {missing}. Cada una es un vector XSS "
        f"clásico que rehype-sanitize del frontend cubre; acá replicamos "
        f"para defensa-en-profundidad."
    )


def test_sanitize_on_handler_pattern_present(agent_py_src: str):
    """Debe existir `_ON_HANDLER_RE` para event handlers `on*=`."""
    assert "_ON_HANDLER_RE" in agent_py_src, (
        "P2-CHAT-SANITIZE regresión: `_ON_HANDLER_RE` no declarado — event "
        "handlers (`onclick`, `onerror`, `onload`) no se neutralizan."
    )


def test_sanitize_js_uri_pattern_present(agent_py_src: str):
    """Debe existir `_JS_URI_RE` para `javascript:` URIs."""
    assert "_JS_URI_RE" in agent_py_src, (
        "P2-CHAT-SANITIZE regresión: `_JS_URI_RE` no declarado — "
        "`javascript:` URIs en href/src no se neutralizan."
    )


def test_sanitize_applied_in_chat_with_agent_return(agent_py_src: str):
    """`chat_with_agent` (non-stream) debe aplicar el helper antes del
    return final."""
    # Match: `_sanitize_chat_output_for_wire(...)` antes del `return ...content`
    pattern = re.compile(
        r"_sanitize_chat_output_for_wire\s*\([^)]*content[^)]*\)[\s\S]{0,500}?"
        r"return\s+\w+,\s*final_state\.get",
    )
    assert pattern.search(agent_py_src), (
        "P2-CHAT-SANITIZE regresión: `chat_with_agent` ya no sanitiza "
        "`content` antes del return. El frontend recibiría HTML peligroso "
        "raw si rehype-sanitize fallara."
    )


def test_sanitize_applied_to_chunk_yield(agent_py_src: str):
    """En `chat_with_agent_stream`, el `chunk_content` debe pasar por el
    helper antes del `yield`."""
    pattern = re.compile(
        r"chunk_content\s*=\s*_sanitize_chat_output_for_wire\s*\(\s*chunk_content\s*\)"
        r"[\s\S]{0,200}?yield\s+f[\"'].*?'type':\s*'chunk'",
    )
    assert pattern.search(agent_py_src), (
        "P2-CHAT-SANITIZE regresión: chunk yield no aplica el helper. "
        "Cada chunk SSE debería pasar por `_sanitize_chat_output_for_wire` "
        "antes del json.dumps."
    )


def test_sanitize_applied_to_done_yield(agent_py_src: str):
    """En `chat_with_agent_stream`, el `final_content` debe pasar por el
    helper antes del yield del `done` event."""
    pattern = re.compile(
        r"final_content\s*=\s*_sanitize_chat_output_for_wire\s*\(\s*final_content\s*\)"
        r"[\s\S]{0,300}?yield\s+f[\"'].*?'type':\s*'done'",
    )
    assert pattern.search(agent_py_src), (
        "P2-CHAT-SANITIZE regresión: el `done` yield no sanitiza "
        "`final_content`. Importante: ese payload se persiste a DB via "
        "save_message — sin sanitización, el HTML peligroso queda en la BD."
    )


# Functional smoke test del helper (no necesita fixtures de pytest porque
# importamos el módulo agent.py directo)


def test_sanitize_helper_neutralizes_script_tag(agent_py_src: str):
    """Smoke test funcional: el helper debe escapar `<script>` a `&lt;script`
    en una cadena de ejemplo. Verifica que el patrón funciona end-to-end."""
    # Ejecutar el helper con un input adversario via exec en sandbox
    # (no podemos importar agent.py completo por sus side-effects)
    ns = {}
    import re as _re
    exec(
        "import re\n"
        + re.search(
            r"_DANGEROUS_HTML_TAG_RE\s*=\s*re\.compile\([\s\S]*?\)\s*\n"
            r"_ON_HANDLER_RE\s*=\s*re\.compile\([\s\S]*?\)\s*\n"
            r"_JS_URI_RE\s*=\s*re\.compile\([\s\S]*?\)\s*\n"
            r"\s*\n"
            r"\s*\n"
            r"def\s+_sanitize_chat_output_for_wire[\s\S]*?\n    return text",
            agent_py_src,
        ).group(0),
        ns,
    )
    fn = ns["_sanitize_chat_output_for_wire"]
    # Casos adversarios
    assert "&lt;script" in fn("<script>alert(1)</script>"), (
        "Helper no neutralizó <script>."
    )
    assert "data-stripped-onclick=" in fn("<p onclick=\"alert(1)\">x</p>"), (
        "Helper no neutralizó onclick="
    )
    assert "data-stripped:" in fn("<a href=\"javascript:alert(1)\">x</a>"), (
        "Helper no neutralizó javascript: URI"
    )
    # Casos legítimos NO deben afectarse
    assert fn("# Heading markdown\n- list item") == "# Heading markdown\n- list item", (
        "Helper modificó markdown legítimo."
    )
    assert "if x < 5" in fn("código: `if x < 5: pass`"), (
        "Helper rompió código con `<` literal."
    )


# =============================================================================
# (2) FRONTEND — prefers-reduced-motion en index.css
# =============================================================================


def test_anchor_present_index_css(index_css_src: str):
    assert "P2-REDUCED-MOTION" in index_css_src, (
        "P2-REDUCED-MOTION regresión: anchor textual perdido en index.css."
    )


def test_reduced_motion_media_query_present(index_css_src: str):
    """Debe existir `@media (prefers-reduced-motion: reduce)` en index.css."""
    pattern = re.compile(
        r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)\s*\{",
    )
    assert pattern.search(index_css_src), (
        "P2-REDUCED-MOTION regresión: `@media (prefers-reduced-motion: reduce)` "
        "no encontrado en index.css. Sin él, usuarios con vestibular disorders "
        "o setting de OS activado ven todas las animaciones del chat (pulse, "
        "shimmer, fadeInUp, spin) — riesgo de motion sickness + violación WCAG 2.3.3."
    )


def _reduced_motion_universal_block(index_css_src: str) -> str:
    """Devuelve el cuerpo del bloque `@media (prefers-reduced-motion: reduce)`
    que target el selector universal (`*`).

    [P2-CHAT-DEFENSE drift 2026-05-19→] index.css ahora tiene DOS bloques
    `@media (prefers-reduced-motion: reduce)`: uno pre-existente acotado a
    `.mf-cta-btn` (`transform: none`) y el bloque P2-REDUCED-MOTION universal
    (`*, *::before, *::after { animation-duration: 0.01ms !important; ... }`).
    El `re.search` non-greedy original matcheaba el PRIMER bloque (el de
    `.mf-cta-btn`), que no contiene las reglas universales → falso rojo. Aquí
    enumeramos todos los bloques y seleccionamos el universal."""
    blocks = re.findall(
        r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)\s*\{([\s\S]*?)\n\}",
        index_css_src,
    )
    assert blocks, "Bloque @media (prefers-reduced-motion) no parseable."
    universal = [b for b in blocks if re.search(r"(^|[\s,{])\*[\s,{]", b)]
    assert universal, (
        "P2-REDUCED-MOTION: ningún bloque `@media (prefers-reduced-motion: "
        "reduce)` target el selector universal `*`. El reset global de "
        "animaciones debe aplicar a todo el árbol."
    )
    return universal[0]


def test_reduced_motion_universal_selector_targets_animations(index_css_src: str):
    """El bloque media query debe target `*` (universal) con
    `animation-duration: 0.01ms !important`."""
    block = _reduced_motion_universal_block(index_css_src)
    assert re.search(r"animation-duration\s*:\s*0\.01ms\s*!important", block), (
        "P2-REDUCED-MOTION: la regla `animation-duration: 0.01ms !important` no "
        "está en el bloque. Sin ella, animaciones siguen corriendo full speed."
    )
    assert re.search(r"transition-duration\s*:\s*0\.01ms\s*!important", block), (
        "P2-REDUCED-MOTION: la regla `transition-duration: 0.01ms !important` "
        "no está en el bloque. Transiciones también pueden inducir motion sickness."
    )


def test_reduced_motion_overrides_smooth_scroll(index_css_src: str):
    """El bloque debe override `scroll-behavior: auto` — `.messages-container`
    tiene `scrollBehavior: 'smooth'` inline que dispararía smooth scroll
    incluso con reduced-motion preference."""
    block = _reduced_motion_universal_block(index_css_src)
    assert re.search(r"scroll-behavior\s*:\s*auto\s*!important", block), (
        "P2-REDUCED-MOTION: `scroll-behavior: auto !important` no encontrado. "
        "Sin él, el smooth scroll inline del .messages-container ignora el "
        "preference."
    )


# =============================================================================
# (3) FRONTEND — Retry adaptativo en fetchSessionMessages
# =============================================================================


def test_anchor_present_agent_page(agent_page_src: str):
    assert "P2-FETCH-RETRY-ADAPTIVE" in agent_page_src, (
        "P2-FETCH-RETRY-ADAPTIVE regresión: anchor textual perdido en AgentPage.jsx."
    )


def test_classify_fetch_session_retry_defined(agent_page_src: str):
    """Helper `_classifyFetchSessionRetry` debe declararse."""
    assert re.search(
        r"const\s+_classifyFetchSessionRetry\s*=\s*\(",
        agent_page_src,
    ), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: `_classifyFetchSessionRetry` no "
        "encontrado en AgentPage.jsx. Sin él, todos los errores se manejan "
        "uniforme con delays fijos."
    )


def test_compute_fetch_backoff_defined(agent_page_src: str):
    """Helper `_computeFetchBackoffMs` debe declararse — exponencial con jitter."""
    assert re.search(
        r"const\s+_computeFetchBackoffMs\s*=\s*\(",
        agent_page_src,
    ), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: `_computeFetchBackoffMs` no "
        "encontrado. Sin él, no hay backoff exponencial."
    )


def test_classifier_covers_5xx_bucket(agent_page_src: str):
    """El classifier debe tener una rama para status 5xx (server errors)
    con retryable=true. Pre-fix los 5xx caían al `else` genérico y NO se
    reintentaban — bug clásico que causa welcome message tras un blip
    transitorio del backend."""
    pattern = re.compile(
        r"_classifyFetchSessionRetry\s*=[\s\S]{0,1500}?status\s*>=\s*500"
        r"[\s\S]{0,200}?retryable\s*:\s*true",
    )
    assert pattern.search(agent_page_src), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: el clasificador no maneja "
        "5xx como retryable. Pre-fix 500/502/503/504 fallaban a welcome "
        "message sin un solo retry — esto era el bug original."
    )


def test_classifier_covers_429_with_high_basedelay(agent_page_src: str):
    """429 (rate-limit) debe ser retryable con baseDelayMs alto (>=1000).
    Sin esto, retry agresivo amplifica el rate-limit y empeora el problema."""
    pattern = re.compile(
        r"status\s*===\s*429[\s\S]{0,200}?baseDelayMs\s*:\s*(\d+)",
    )
    m = pattern.search(agent_page_src)
    assert m, (
        "P2-FETCH-RETRY-ADAPTIVE regresión: branch para status 429 no "
        "encontrado. Sin él, rate-limit cae al default no-retryable."
    )
    base_delay = int(m.group(1))
    assert base_delay >= 1000, (
        f"P2-FETCH-RETRY-ADAPTIVE: baseDelayMs para 429 es {base_delay}ms "
        f"— muy bajo. Rate-limit requiere delay >=1s para respetar el server."
    )


def test_classifier_default_not_retryable(agent_page_src: str):
    """4xx genuinos (404, 400, 422) NO deben ser retryable — son bugs del
    cliente, reintentar no resuelve y solo gasta cuota / red."""
    # El último return del classifier debe tener retryable: false
    pattern = re.compile(
        r"_classifyFetchSessionRetry\s*=[\s\S]*?return\s*\{\s*retryable\s*:\s*false",
    )
    assert pattern.search(agent_page_src), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: el classifier no tiene un "
        "default `retryable: false`. 4xx genuinos (404, 400) entrarían a "
        "retry loops infinitos."
    )


def test_backoff_uses_exponential_formula(agent_page_src: str):
    """`_computeFetchBackoffMs` debe usar la fórmula `base * 2^attempt` —
    exponencial, no lineal. Lineal escala mal con retries altos."""
    pattern = re.compile(
        r"_computeFetchBackoffMs\s*=[\s\S]{0,400}?Math\.pow\s*\(\s*2\s*,\s*attempt\s*\)",
    )
    assert pattern.search(agent_page_src), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: `_computeFetchBackoffMs` ya no "
        "usa `Math.pow(2, attempt)` — el backoff dejó de ser exponencial. "
        "Lineal o constant amplifica la carga sobre el server tras blips."
    )


def test_backoff_includes_jitter(agent_page_src: str):
    """El backoff debe incluir jitter (Math.random) — sin él, múltiples
    clientes que recargan en sync atacan al server en burst."""
    pattern = re.compile(
        r"_computeFetchBackoffMs\s*=[\s\S]{0,400}?Math\.random",
    )
    assert pattern.search(agent_page_src), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: `_computeFetchBackoffMs` no "
        "incluye Math.random — sin jitter, thundering herd cuando muchos "
        "clientes recargan tras un downtime del backend."
    )


def test_fetch_session_messages_uses_policy(agent_page_src: str):
    """En `fetchSessionMessages`, la lógica debe invocar
    `_classifyFetchSessionRetry` y `_computeFetchBackoffMs`."""
    assert "_classifyFetchSessionRetry(response.status" in agent_page_src, (
        "P2-FETCH-RETRY-ADAPTIVE regresión: `fetchSessionMessages` no llama "
        "`_classifyFetchSessionRetry(response.status, ...)` en el branch HTTP. "
        "El refactor no se completó."
    )
    assert "_classifyFetchSessionRetry(null, true)" in agent_page_src, (
        "P2-FETCH-RETRY-ADAPTIVE regresión: el catch outer no invoca "
        "`_classifyFetchSessionRetry(null, true)` para clasificar network "
        "error. El refactor no se completó."
    )
    assert "_computeFetchBackoffMs(policy.baseDelayMs" in agent_page_src, (
        "P2-FETCH-RETRY-ADAPTIVE regresión: el delay del retry ya no se "
        "computa con `_computeFetchBackoffMs(policy.baseDelayMs, retryCount)`."
    )


def test_legacy_hardcoded_retry_count_removed(agent_page_src: str):
    """Los delays hardcoded `800` y `600` de la versión pre-fix NO deben
    seguir invocando `setTimeout(...fetchSessionMessages..., 800)` ni
    similar — el classifier ahora controla los baseDelays."""
    forbidden_800 = re.compile(
        r"setTimeout\s*\(\s*\(\)\s*=>\s*fetchSessionMessages\([^)]+\)\s*,\s*800\)",
    )
    forbidden_600 = re.compile(
        r"setTimeout\s*\(\s*\(\)\s*=>\s*fetchSessionMessages\([^)]+\)\s*,\s*600\)",
    )
    assert not forbidden_800.search(agent_page_src), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: setTimeout con delay hardcoded 800ms "
        "volvió a aparecer en `fetchSessionMessages`. Usar policy.baseDelayMs."
    )
    assert not forbidden_600.search(agent_page_src), (
        "P2-FETCH-RETRY-ADAPTIVE regresión: setTimeout con delay hardcoded 600ms "
        "volvió a aparecer en el catch. Usar policy.baseDelayMs."
    )
