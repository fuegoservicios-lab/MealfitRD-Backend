"""[P1-VISION-NO-LOCAL · 2026-07-28] Test ancla: eliminación completa del
provider de visión LOCAL (Ollama/gemma, P1-MEAL-SCAN-GEMMA · 2026-07-12).

Contexto: el owner no podía seguir sosteniendo el modelo local (gemma vía
túnel SSH reverso laptop→VPS) — la máquina no aguantaba el servicio.
"Escanear comida" (Dashboard) ya había migrado a `gpt-5.6-luna` (cloud,
P1-VISION-LUNA · 2026-07-28) horas antes de este fix, pero el escáner de
Nevera (`routers/user_data.py::api_inventory_photo_scan`) tenía su PROPIO
cliente Ollama y seguía gateado en `provider == "ollama"` — con
`MEALFIT_VISION_PROVIDER=openai_compatible` en prod, ese endpoint devolvía
503 (outage). Este P-fix:
  1. Migra el escáner de Nevera al MISMO transporte cloud que el meal-scan
     (`vision_agent.analyze_image_structured`).
  2. Elimina el provider `ollama`, la cascada de fallback
     (`MEALFIT_VISION_FALLBACK_PROVIDER`), el single-flight lock, y
     `is_vision_local()` — código muerto sin un segundo provider real.

Doc: backend/docs/vision_luna.md.
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
import tokenize
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. El escáner de Nevera ya no exige `provider == "ollama"` y no 503 bajo
#    el provider cloud — funcional, no solo parser-based: llama el endpoint
#    real con vision_agent/DB mockeados.
# ---------------------------------------------------------------------------

def test_pantry_scan_endpoint_no_longer_gates_on_ollama_string():
    """Parser: el gate `if provider != "ollama":` desapareció del endpoint —
    reemplazado por `!= "openai_compatible"` (el único provider real)."""
    src = _src("routers/user_data.py")
    i = src.find('@router.post("/inventory/photo-scan")')
    assert i != -1, "endpoint del escáner de Nevera desapareció"
    body = src[i:i + 4000]
    assert 'provider != "ollama"' not in body, (
        "el gate viejo (provider != 'ollama') reapareció -- volvería a "
        "503-ear bajo MEALFIT_VISION_PROVIDER=openai_compatible (el outage "
        "que este P-fix cerró)."
    )
    assert 'provider != "openai_compatible"' in body, (
        "el endpoint debe validar contra el ÚNICO provider real."
    )
    # El 503 de 'provider desconocido' se conserva (fail-secure ante un
    # config typo) -- lo que cambió es CONTRA QUÉ compara, no que exista.
    assert 'if provider == "off":' in body and "503" in body


def test_pantry_scan_endpoint_functional_under_cloud_provider(monkeypatch):
    """Funcional: invoca el endpoint real (bypaseando FastAPI DI, llamada
    directa a la coroutine) bajo MEALFIT_VISION_PROVIDER=openai_compatible,
    con `vision_agent.analyze_image_structured` y `db.execute_sql_query`
    mockeados (sin DB real -- este repo corre sus tests en un venv DB-less).
    Verifica que YA NO 503-ea (el bug en vivo) y que el resultado matchea
    contra el catálogo tal como antes."""
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "openai_compatible")

    import routers.user_data as ud
    import vision_agent as va
    import db as db_facade

    async def _fake_analyze(image_bytes, prompt, schema):
        assert schema is ud._PantryScanResult, (
            "el escáner de Nevera debe pedir SU propio schema de negocio "
            "(items + brand), no el ImageDescription del meal-scan."
        )
        return ud._PantryScanResult(items=[
            ud._PantryScanItem(
                name="pechuga de pollo", quantity=2, unit="lb",
                confidence=0.9, brand=None,
            ),
        ])

    def _fake_catalog_query(*a, **k):
        return [{"id": "abc-123", "name": "Pechuga de pollo",
                  "market_container": None, "default_unit": "lb"}]

    monkeypatch.setattr(va, "analyze_image_structured", _fake_analyze)
    monkeypatch.setattr(db_facade, "execute_sql_query", _fake_catalog_query)

    body = {"image_b64": base64.b64encode(b"fake-jpeg-bytes").decode("ascii")}
    result = asyncio.run(
        ud.api_inventory_photo_scan(body=body, verified_user_id="user-123")
    )

    assert result["provider"] == "openai_compatible"
    assert result["items"], "el scan debió devolver al menos un item detectado"
    item = result["items"][0]
    assert item["detected_name"] == "pechuga de pollo"
    assert item["master_ingredient_id"] == "abc-123", "match contra el catálogo"


def test_pantry_scan_endpoint_still_503s_when_feature_off(monkeypatch):
    """El kill-switch (`MEALFIT_VISION_PROVIDER=off`) sigue funcionando --
    esto NO es el bug: un 503 intencional (feature apagado) es distinto del
    503 accidental (provider mal-gateado) que este P-fix cerró."""
    monkeypatch.setenv("MEALFIT_VISION_PROVIDER", "off")
    import routers.user_data as ud
    from fastapi import HTTPException

    body = {"image_b64": base64.b64encode(b"x").decode("ascii")}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(ud.api_inventory_photo_scan(body=body, verified_user_id="user-123"))
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# 2. Blanket parser: ningún módulo de PRODUCCIÓN de backend/ referencia
#    ollama/gemma/_ollama_ como CÓDIGO VIVO. La guardia más fuerte de que la
#    eliminación es completa.
#
# Precisión (instrucción del plan -- "precisa pero no frágil"): usa
# `tokenize` para distinguir CÓDIGO (identificadores, operadores, strings de
# una/dos comillas usados como valores -- p.ej. un nombre de env var) de
# NARRATIVA (comentarios `#...` y docstrings triple-comillados). Un
# comentario tipo "P1-MEAL-SCAN-GEMMA · 2026-07-12" que documenta que el fix
# ANTERIOR usaba gemma es EXACTAMENTE el patrón que el resto de este repo
# usa para narrar su propia historia (ver CLAUDE.md: "Gemini eliminado por
# completo" sin borrar la palabra "Gemini" de la prosa) -- eso debe
# SOBREVIVIR. Lo que NO debe sobrevivir es un identificador vivo
# (`_ollama_vision_scan`, `_VISION_PROVIDER_OLLAMA`) o un string usado como
# DATO de código (`"MEALFIT_OLLAMA_BASE_URL"` pasado a `os.environ.get`) --
# eso son tokens NAME/STRING-no-triple-comillada, y el test los cacha
# porque NO están dentro de un comentario ni de un docstring.
# ---------------------------------------------------------------------------

_PATTERN = re.compile(r"ollama|gemma", re.IGNORECASE)

# Directorios que NO son código de producción de MealfitRD: la suite de
# tests (con sus propias menciones históricas legítimas -- fixtures,
# markers de P-fixes anteriores, docstrings narrativos) y cualquier
# virtualenv (dependencias de terceros, no nuestras). `test_venv` no
# matchea el glob `venv*` pero SÍ es un virtualenv real (869 .py de
# site-packages, verificado) -- se excluye por la misma razón que `venv`/
# `venv-test`, no por nombre.
_SKIP_DIR_NAMES = {"tests", "__pycache__", ".git", ".github", ".pytest_cache", "node_modules"}


def _is_skippable_dir(name: str) -> bool:
    if name in _SKIP_DIR_NAMES:
        return True
    if name.startswith("."):
        return True
    if "venv" in name.lower():
        return True
    return False


def _iter_backend_py_files():
    for root, dirs, files in os.walk(_BACKEND):
        dirs[:] = [d for d in dirs if not _is_skippable_dir(d)]
        for fname in files:
            if fname.endswith(".py"):
                yield Path(root) / fname


def _is_triple_quoted_string_token(tok_string: str) -> bool:
    """True si el STRING token es un docstring/bloque triple-comillado
    (`\"\"\"...\"\"\"` o `'''...'''`, con prefijos opcionales f/r/b/u).
    En este codebase los triple-comillados son SIEMPRE narrativa (docstrings)
    -- el contenido de negocio (prompts, env var names) usa comillas simples
    o dobles normales, nunca triples."""
    s = tok_string
    i = 0
    while i < len(s) and s[i] in "fFrRbBuU":
        i += 1
    rest = s[i:]
    return rest.startswith('"""') or rest.startswith("'''")


def _live_code_hits(path: Path):
    """Tokeniza `path` y devuelve [(lineno, token_text), ...] para tokens
    FUERA de comentarios/docstrings que matchean ollama/gemma. Archivos no
    parseables (encoding raro, etc.) se saltan silenciosamente -- no es la
    responsabilidad de este test detectar SyntaxErrors."""
    hits = []
    try:
        with open(path, "rb") as fh:
            tokens = list(tokenize.tokenize(fh.readline))
    except Exception:
        return hits
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and _is_triple_quoted_string_token(tok.string):
            continue
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.OP):
            continue
        if _PATTERN.search(tok.string):
            hits.append((tok.start[0], tok.string.strip()[:80]))
    return hits


def test_no_live_code_references_ollama_or_gemma_outside_tests():
    violations = []
    for path in _iter_backend_py_files():
        for lineno, text in _live_code_hits(path):
            rel = path.relative_to(_BACKEND).as_posix()
            violations.append(f"{rel}:{lineno}: {text!r}")
    assert not violations, (
        "Referencias VIVAS (fuera de comentarios/docstrings) a ollama/gemma "
        "sobreviven en backend/ (excluyendo tests/ y virtualenvs) tras "
        "P1-VISION-NO-LOCAL:\n" + "\n".join(violations)
    )


def test_blanket_scan_actually_covers_the_files_that_mattered():
    """Sanity del blanket test mismo: si el helper de exclusión de dirs se
    rompiera (p.ej. excluyera 'routers' por accidente), el blanket de arriba
    daría falso-verde. Ancla que los módulos reales que se tocaron SÍ entran
    al recorrido."""
    scanned = {p.relative_to(_BACKEND).as_posix() for p in _iter_backend_py_files()}
    for must_include in (
        "vision_agent.py", "image_prep.py",
        "routers/user_data.py", "routers/diary.py",
    ):
        assert must_include in scanned, (
            f"{must_include} no entró al recorrido del blanket test -- "
            f"revisar _is_skippable_dir/_iter_backend_py_files."
        )


def test_blanket_scan_would_have_caught_the_removed_symbols():
    """MUERDE de verdad: alimenta el detector con un snippet SINTÉTICO que
    reproduce el patrón que existía pre-fix (`_VISION_PROVIDER_OLLAMA =
    "ollama"` vivo, fuera de comentario/docstring) y confirma que
    `_live_code_hits`-equivalente lo marca. Sin esto, un blanket que nunca
    encuentra nada podría estar roto en vez de estar limpio."""
    import io as _io
    src = (
        '_VISION_PROVIDER_OLLAMA = "ollama"  # una env var real, no un comentario suelto\n'
    )
    hits = []
    tokens = list(tokenize.generate_tokens(_io.StringIO(src).readline))
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and _is_triple_quoted_string_token(tok.string):
            continue
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.OP):
            continue
        if _PATTERN.search(tok.string):
            hits.append(tok.string)
    assert any("_VISION_PROVIDER_OLLAMA" in h for h in hits), (
        "el detector NO cachó el identificador vivo del snippet sintético -- "
        "el blanket test de arriba podría estar dando falso-verde."
    )
    assert any(h == '"ollama"' for h in hits), (
        "el detector NO cachó el string-valor 'ollama' del snippet sintético."
    )


def test_docstring_and_comment_mentions_are_tolerated():
    """MUERDE de verdad (dirección opuesta): un comentario/docstring
    narrando historia ('# P1-MEAL-SCAN-GEMMA · 2026-07-12 -- antes usaba
    gemma') NO debe disparar el detector -- si lo hiciera, el blanket test
    de arriba sería imposible de mantener verde sin borrar la historia del
    repo (contra la convención de CLAUDE.md)."""
    import io as _io
    src = (
        '# [P1-MEAL-SCAN-GEMMA . 2026-07-12] antes usabamos gemma via Ollama\n'
        'def foo():\n'
        '    """Docstring que menciona gemma y Ollama como historia."""\n'
        '    return 1\n'
    )
    hits = []
    tokens = list(tokenize.generate_tokens(_io.StringIO(src).readline))
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and _is_triple_quoted_string_token(tok.string):
            continue
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.OP):
            continue
        if _PATTERN.search(tok.string):
            hits.append(tok.string)
    assert not hits, f"comentario/docstring narrativo disparó el detector: {hits}"


# ---------------------------------------------------------------------------
# 3. Anti-regresión de la doctrina de cuota: el costo de un scan sigue
#    yendo a `llm_usage_events`, NUNCA a `api_usage` -- y `get_monthly_api_usage`
#    sigue contando sin filtro de endpoint (I-doctrina compartida con TODOS
#    los endpoints exentos, no solo visión). Redundante A PROPÓSITO con
#    test_p1_vision_luna.py -- ese archivo ancla el fix de P1-VISION-LUNA;
#    este ancla que P1-VISION-NO-LOCAL no lo reabrió al tocar el mismo gate.
# ---------------------------------------------------------------------------

def test_diary_upload_cost_still_goes_to_llm_usage_events_not_api_usage():
    src = _src("routers/diary.py")
    assert "log_llm_usage_event" in src, (
        "el gasto de un scan debe seguir yendo a llm_usage_events (libro de "
        "COSTO) vía log_llm_usage_event."
    )
    assert re.search(r'node\s*=\s*"vision_scan"', src)
    assert "log_api_usage(" not in src, (
        "log_api_usage NO debe invocarse desde diary.py bajo ningún "
        "provider -- quemaría crédito de PLAN por fotografiar comida "
        "(doctrina P1-NEVERA-QUOTA-EXEMPT)."
    )
    assert re.search(r"^\s*from\s+\S+\s+import\s+[^#\n]*\blog_api_usage\b",
                      src, re.MULTILINE) is None, (
        "log_api_usage no debe estar ni importado en diary.py."
    )


def test_gate_no_longer_references_is_vision_local():
    """El gate simplificado (sin provider local, la exclusión
    `not is_vision_local()` nunca podía ser False) debe emitir la fila
    SIEMPRE que haya usuario autenticado -- ya no hay condición de provider
    que evaluar. Ojo: NO es un substring-check del archivo entero -- ese
    archivo conserva (a propósito, ver blanket test arriba) un comentario
    narrando que el gate ANTES incluía `is_vision_local()`; lo que este test
    ancla es que el `if` VIVO (no el comentario) ya no lo tiene."""
    src = _src("routers/diary.py")
    gate_marker = "if actual_user_id and actual_user_id != session_id:"
    # Encontrar este substring EXACTO (terminado en ':' justo tras
    # session_id) ya prueba que no hay un sufijo `and not is_vision_local()`
    # colgando del `if` -- si lo hubiera, el ':' no vendría inmediatamente
    # después de `session_id` y el find() de abajo fallaría.
    assert gate_marker in src, (
        "el gate del costo de visión desapareció/cambió de forma en "
        "diary.py -- o volvió a traer un sufijo tipo `and not "
        "is_vision_local()`, re-anclar el marker."
    )
    # `is_vision_local` ya no debe ser IMPORTABLE de vision_agent (ver
    # test_vision_agent_symbols_removed) -- si sobreviviera una referencia
    # bare a él fuera de un comentario, sería un NameError en runtime; el
    # blanket parser-based de arriba es el guard correcto para ESO.


def test_get_monthly_api_usage_sigue_sin_filtro_de_endpoint():
    """Redundante a propósito con test_p1_vision_luna.py (ver docstring de
    esa función homónima) -- ancla que P1-VISION-NO-LOCAL, al tocar el mismo
    gate de diary.py, no reintrodujo un filtro por endpoint en el libro de
    cuota compartido. Cambiar este SELECT alteraría la facturación de TODOS
    los endpoints, no solo el de visión."""
    src = _src("db_profiles.py")
    m = re.search(
        r"^def get_monthly_api_usage\(.*?(?=^def )", src, re.MULTILINE | re.DOTALL
    )
    assert m, "get_monthly_api_usage no encontrada en db_profiles.py."
    body = m.group(0)
    assert (
        "SELECT count(*) as total FROM api_usage WHERE user_id = %s "
        "AND created_at >= %s"
    ) in body
    assert "endpoint" not in body


# ---------------------------------------------------------------------------
# 4. Símbolos/knobs eliminados -- anclas puntuales adicionales a las de
#    test_p1_vision_luna.py (esa suite ancla la cascada; esta ancla el
#    single-flight/pantry-scan/knobs, el alcance propio de este P-fix).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol", [
    "get_vision_single_flight_lock", "is_vision_local",
    "_ollama_base_url", "_ollama_model_name", "_ollama_timeout_s",
    "_ollama_meal_scan", "_dispatch_ollama_vision",
])
def test_vision_agent_symbols_removed(symbol):
    import vision_agent as va
    assert not hasattr(va, symbol), f"vision_agent.{symbol} debía eliminarse."


def test_user_data_no_longer_has_own_ollama_client():
    """Ojo: `_ollama_vision_scan` como PALABRA sobrevive en un docstring
    (`_cloud_vision_scan` narra qué reemplazó) -- eso es narrativa legítima,
    ver blanket test arriba. Lo que este test ancla es que ya NO está
    DEFINIDA como función (mismo patrón que
    test_p1_deepseek_only_restore.py::test_no_ollama_helper, que también
    busca `"def _is_ollama_provider"` y no el substring desnudo)."""
    src = _src("routers/user_data.py")
    assert "def _ollama_vision_scan" not in src
    assert "httpx.post" not in src, (
        "el escáner de Nevera no debe mantener su PROPIO roundtrip httpx -- "
        "reutiliza vision_agent.analyze_image_structured."
    )
    assert "get_vision_single_flight_lock" not in src


@pytest.mark.parametrize("env_var", [
    "MEALFIT_OLLAMA_BASE_URL", "MEALFIT_OLLAMA_VISION_MODEL",
    "MEALFIT_VISION_FALLBACK_PROVIDER", "MEALFIT_VISION_TIMEOUT_S",
])
def test_dead_knobs_no_longer_read_in_production(env_var):
    """Los 4 knobs que el controller debe borrar del .env del VPS ya no se
    leen en NINGÚN módulo de producción (`os.environ` ni `knobs._env_*`)."""
    for rel in ("vision_agent.py", "routers/user_data.py", "routers/diary.py",
                "image_prep.py"):
        assert env_var not in _src(rel), f"{env_var} sigue referenciado en {rel}"


# ---------------------------------------------------------------------------
# 5. Marker supersession-proof (mirror de test_p2_audit_v5_batch.py:60-68 /
#    test_p1_vision_luna.py) -- own-name-or-later-date.
# ---------------------------------------------------------------------------

_OUR_MARKER_SLUG = "P1-VISION-NO-LOCAL"
_OUR_MARKER_DATE = "2026-07-28"


def _marker_supersedes_or_matches(marker: str) -> bool:
    if _OUR_MARKER_SLUG in marker:
        return True
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    if not fecha:
        return False
    return fecha.group(1) >= _OUR_MARKER_DATE


def test_marker_bumped_to_vision_no_local_or_later():
    src = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX en app.py"
    marker = m.group(1)
    assert _marker_supersedes_or_matches(marker), (
        f"`_LAST_KNOWN_PFIX={marker!r}` no es P1-VISION-NO-LOCAL ni tiene "
        f"fecha >= {_OUR_MARKER_DATE} -- stale respecto a este P-fix."
    )


@pytest.mark.parametrize("marker_viejo", [
    "P1-VISION-LUNA · 2026-07-27",
    "P0-ALGUN-FIX-VIEJO · 2026-01-01",
])
def test_bite_marker_mas_viejo_es_rechazado(marker_viejo):
    assert not _marker_supersedes_or_matches(marker_viejo)


@pytest.mark.parametrize("marker_nuevo", [
    "P1-VISION-LUNA · 2026-07-28",
    "P2-ALGUN-FIX-FUTURO · 2026-08-01",
])
def test_bite_marker_de_otro_pfix_pero_mas_nuevo_es_aceptado(marker_nuevo):
    assert _marker_supersedes_or_matches(marker_nuevo)


def test_marker_slug_matches_this_test_file():
    """Cross-link con test_p2_hist_audit_14_marker_test_link.py."""
    slug = _OUR_MARKER_SLUG.replace("-", "_").lower()
    assert slug == "p1_vision_no_local"
    this_file = os.path.basename(__file__)
    assert this_file.startswith(f"test_{slug}"), (
        f"slug {slug!r} no matchea el nombre de este archivo {this_file!r}"
    )
