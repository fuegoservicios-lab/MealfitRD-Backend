"""[P1-CHAT-ACT-DONT-ASK · 2026-07-28] El coach ACTÚA sobre lo que el usuario
ya afirmó, en vez de pedir permiso para algo que ya tiene.

Vivo (owner, mangú con los tres golpes): subió una foto y escribió "me
desayuné esto" (pasado, inequívoco). El agente preguntó "¿Te lo comiste
completo...? Si es así, lo registro". El usuario contestó "si". El agente
respondió "Perfecto, lo registro al tiro" — Y APILÓ una segunda pregunta
sobre mover el almuerzo — pero NUNCA llamó `log_consumed_meal`. Verificado en
prod: `consumed_meals` se quedó en 8 filas, la más nueva del 2026-07-13. La
comida jamás se guardó pese a la confirmación explícita.

Causa raíz (2 capas):
  1. `prompts/chat_agent.py` — el bullet de `log_consumed_meal` en AMBOS
     builders enmarcaba una frase en pasado ("me desayuné esto") como algo que
     requería una confirmación aparte ("si el usuario confirma que se la
     comió, USA ESTA HERRAMIENTA"). Una frase en pasado YA ES la confirmación.
  2. `agent.py` — el bloque `DIARIO DE HOY` (stream path, ~línea 4480) abría
     con "Revisa esto ANTES de preguntar si ya comió algo", reforzando el
     marco de "preguntar" incluso cuando lo que protege (no duplicar un
     registro) no requiere preguntar nada.

Fix: ambos bullets ahora exigen registrar EN EL MISMO TURNO ante una frase en
pasado, prohíben afirmar "lo registro"/"lo guardé" sin haber llamado la
herramienta, tratan la divergencia plan-vs-realidad como normal (no exige
permiso), y limitan a UNA sola pregunta de seguimiento por respuesta (la
única legítima: el guard de duplicado día/comida-principal, que sigue vivo
con `force=true`). El opener de `DIARIO DE HOY` deja de primar "preguntar"
pero conserva la instrucción de no duplicar un registro ya existente.

Seguro ahora de un modo en que antes no lo era: una comida mal registrada es
borrable desde la card "Progreso en Tiempo Real" (`DELETE
/api/diary/consumed/{id}`, P1-DIARY-EDITABLE · 2026-07-28) — el prompt nuevo
referencia esa vía de corrección en vez de pedir permiso por adelantado.

Confirmaciones reales que NO se tocan (verificadas, no asumidas):
  - el guard de duplicado (`force=true` cuando ya hay una comida principal
    ese día) — previene doble conteo, así que sigue siendo pregunta legítima;
  - `regenerate_full_day` — cuesta crédito + ~2 min, sigue pidiendo CONFIRMA;
  - cualquier cosa destructiva (fuera del alcance de este bullet).

Los asserts anclan en marcadores semánticos introducidos por este fix (no en
oraciones largas verbatim) para que una edición de copy futura que preserve
el comportamiento no rompa el test por accidente, pero un revert SÍ lo rompa.

tooltip-anchor: P1-CHAT-ACT-DONT-ASK
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_CHAT_AGENT_PROMPTS_PY = _BACKEND_ROOT / "prompts" / "chat_agent.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _read(_AGENT_PY)


@pytest.fixture(scope="module")
def prompts_src() -> str:
    return _read(_CHAT_AGENT_PROMPTS_PY)


@pytest.fixture(scope="module")
def builder_outputs() -> dict:
    """Invoca ambos builders con un user_id dummy. `log_consumed_meal` no
    depende del knob `MEALFIT_CHAT_PLAN_TOOLS_ENABLED` (vive fuera de
    `_plan_tools_bullets_*`), así que el estado del knob en el entorno de
    test no afecta este bullet."""
    from prompts.chat_agent import build_tools_instructions, build_tools_instructions_stream

    uid = "11111111-1111-1111-1111-111111111111"
    return {
        "inline": build_tools_instructions(uid),
        "stream": build_tools_instructions_stream(uid),
    }


def _bullet_line(text: str, tool_name: str) -> str:
    """Cada bullet vive en su propia línea física dentro del f-string
    triple-quoted (una línea de source == una línea del output, sin saltos
    embebidos). Devuelve esa línea completa o falla loud."""
    prefix = f"- Usa `{tool_name}`"
    for line in text.split("\n"):
        if line.strip().startswith(prefix):
            return line
    raise AssertionError(
        f"No encontré el bullet de `{tool_name}` en el output del builder. "
        f"¿Se renombró la tool o cambió el formato del bullet?"
    )


@pytest.fixture(scope="module")
def log_bullets(builder_outputs: dict) -> dict:
    return {
        "inline": _bullet_line(builder_outputs["inline"], "log_consumed_meal"),
        "stream": _bullet_line(builder_outputs["stream"], "log_consumed_meal"),
    }


# ===========================================================================
# 1. Registrar EN EL MISMO TURNO ante una frase en pasado — sin preguntar
# ===========================================================================

def test_both_builders_log_immediately_on_past_tense_statement(log_bullets: dict):
    """Un revert al wording viejo ('si el usuario confirma que se la comió')
    no menciona 'mismo turno' en absoluto — este marker es net-new."""
    for name, bullet in log_bullets.items():
        assert "MISMO TURNO" in bullet, (
            f"[{name}] el bullet de log_consumed_meal ya no exige registrar "
            f"EN EL MISMO TURNO ante una frase en pasado."
        )
        assert re.search(r"en (tiempo )?pasado", bullet), (
            f"[{name}] falta la referencia a 'tiempo pasado' como disparador "
            f"suficiente para registrar."
        )
        assert "YA ES la confirmación" in bullet or "YA es la confirmación" in bullet, (
            f"[{name}] falta la instrucción explícita de que una frase en "
            f"pasado YA ES la confirmación (no algo a solicitar aparte)."
        )


def test_old_confirm_first_framing_is_gone(prompts_src: str, log_bullets: dict):
    """La frase vieja enmarcaba la confirmación como un turno aparte que había
    que solicitar. Debe estar ausente del archivo completo (solo vivía en
    estos 2 bullets) y ausente de cada bullet extraído."""
    forbidden = (
        "el usuario confirma que se la comió",
        "y el usuario confirma que se la comió",
    )
    for phrase in forbidden:
        assert phrase not in prompts_src, (
            f"regresión: reapareció el framing viejo {phrase!r} en "
            f"prompts/chat_agent.py — una frase en pasado debe bastar, no "
            f"debe pedirse confirmación aparte."
        )
    for name, bullet in log_bullets.items():
        assert "usuario confirma" not in bullet, (
            f"[{name}] el bullet todavía enmarca 'confirma' como una "
            f"solicitud de permiso separada."
        )


# ===========================================================================
# 2. Prohibido afirmar que se registró sin haber llamado la herramienta
# ===========================================================================

def test_both_builders_forbid_claiming_logged_without_tool_call(log_bullets: dict):
    """El fallo más grave observado en prod: 'lo registro al tiro' sin
    llamada real a la tool. El wording viejo no tenía NINGUNA cláusula sobre
    esto — es net-new, así que un revert lo pierde por completo."""
    for name, bullet in log_bullets.items():
        assert "NUNCA digas" in bullet, (
            f"[{name}] falta la prohibición explícita de afirmar el registro."
        )
        assert re.search(r"si no llamaste la herramienta", bullet), (
            f"[{name}] la prohibición no está condicionada a 'si no llamaste "
            f"la herramienta en ese turno' — sin esa condición la regla es "
            f"vaga."
        )


# ===========================================================================
# 3. Divergencia plan-vs-realidad es normal — no requiere permiso
# ===========================================================================

def test_both_builders_say_plan_divergence_needs_no_permission(log_bullets: dict):
    for name, bullet in log_bullets.items():
        assert "NO requiere permiso" in bullet, (
            f"[{name}] falta la instrucción de que comer distinto al plan "
            f"prescrito es normal y no exige pedir permiso."
        )


# ===========================================================================
# 4. El guard de duplicado (`force=true`) sigue vivo en ambos builders
# ===========================================================================

def test_duplicate_guard_and_force_true_survive(log_bullets: dict):
    for name, bullet in log_bullets.items():
        assert "force=true" in bullet, (
            f"[{name}] el guard de duplicado (`force=true`) desapareció del "
            f"bullet — regresión sobre P1-CONSUMED-BACKDATE."
        )
        assert re.search(r"ese d[ií]a ya tiene esa comida", bullet, re.IGNORECASE), (
            f"[{name}] falta la referencia al guard 'ese día ya tiene esa "
            f"comida (principal) registrada'."
        )


def test_p1_consumed_backdate_marker_and_days_ago_survive(prompts_src: str):
    """El marker de P1-CONSUMED-BACKDATE + los params days_ago/meal_type
    deben seguir presentes — este fix NO toca ese contrato, solo el framing
    de la confirmación."""
    assert prompts_src.count("P1-CONSUMED-BACKDATE") >= 2, (
        "ambos builders deben conservar el marker [P1-CONSUMED-BACKDATE]"
    )
    assert prompts_src.count("máx 7") >= 2, (
        "ambos builders deben conservar el tope de 7 días (P1-DIARY-EDITABLE)"
    )
    assert "máx 3" not in prompts_src, "el tope viejo (3) no debe reaparecer"


# ===========================================================================
# 5. Como máximo UNA pregunta de seguimiento por respuesta
# ===========================================================================

def test_at_most_one_followup_question_marked_explicit(log_bullets: dict):
    """El bug vivo apiló una segunda pregunta (mover el almuerzo) sobre la
    primera. El bullet ahora marca el guard de duplicado como la ÚNICA
    pregunta permitida en la respuesta — wording net-new, ausente del bullet
    viejo."""
    for name, bullet in log_bullets.items():
        assert "ÚNICA" in bullet or "única" in bullet, (
            f"[{name}] falta la instrucción de limitar a una sola pregunta "
            f"de seguimiento por respuesta."
        )


# ===========================================================================
# 6. `regenerate_full_day` sigue pidiendo confirmación (cuesta crédito+tiempo)
# ===========================================================================

def _bullet_function_body(src: str, fn_name: str) -> str:
    m = re.search(rf"def\s+{re.escape(fn_name)}\s*\([^)]*\)\s*(?:->[^:]+)?:", src)
    assert m is not None, f"no encontré `def {fn_name}(` en prompts/chat_agent.py"
    start = m.end()
    next_def = re.search(r"\ndef\s", src[start:])
    end = start + (next_def.start() if next_def else 4000)
    return src[start:end]


def test_regenerate_full_day_confirmation_survives(prompts_src: str):
    for fn_name in ("_plan_tools_bullets_inline", "_plan_tools_bullets_stream"):
        body = _bullet_function_body(prompts_src, fn_name)
        i = body.find("regenerate_full_day")
        assert i != -1, f"{fn_name}: bullet de regenerate_full_day desapareció"
        window = body[i:i + 400]
        assert "CONFIRMA" in window, (
            f"{fn_name}: regenerate_full_day dejó de exigir CONFIRMA antes de "
            f"llamarla (cuesta 1 crédito + ~2 min) — este fix NO debe tocar "
            f"esta confirmación."
        )


# ===========================================================================
# 7. `DIARIO DE HOY` (agent.py, stream path): ya no prima 'preguntar', pero
#    sigue protegiendo contra duplicar un registro ya existente.
# ===========================================================================

def _diario_hoy_registrado_line(src: str) -> str:
    """Hay DOS callsites con el prefijo 'DIARIO DE HOY: El usuario ya ha
    registrado consumir hoy' (non-stream ~4099, sin opener extra; stream
    ~4480, el que este fix edita). Ancla en el marker `[P1-CHAT-ACT-DONT-ASK]`
    que solo vive en el callsite editado para no matchear el primero."""
    marker = "[P1-CHAT-ACT-DONT-ASK]"
    i = src.find(marker)
    assert i != -1, (
        "no encontré el marker [P1-CHAT-ACT-DONT-ASK] en agent.py — el "
        "bloque DIARIO DE HOY (rama 'ya registró', stream path) debe "
        "llevarlo inline."
    )
    # La línea entera del f-string (system_prompt += f"...") — buscar hasta
    # el cierre de la f-string en esa misma línea física.
    line_start = src.rfind("\n", 0, i) + 1
    line_end = src.find("\n", i)
    return src[line_start:line_end]


def test_diario_de_hoy_no_longer_primes_preguntar(agent_src: str):
    line = _diario_hoy_registrado_line(agent_src)
    assert "ANTES de preguntar" not in line, (
        "regresión: el bloque DIARIO DE HOY volvió a abrir primando el "
        "marco de 'preguntar' en vez de actuar."
    )


def test_diario_de_hoy_still_prevents_duplicate_logging(agent_src: str):
    line = _diario_hoy_registrado_line(agent_src)
    assert re.search(r"NO DUPLICAR|sin volver a registrarlo", line), (
        "el bloque DIARIO DE HOY dejó de instruir no duplicar un registro "
        "ya existente — eso SÍ debe sobrevivir (evita doble conteo)."
    )
    assert "snack nocturno" in line, (
        "se perdió el ejemplo concreto (cena ya registrada + foto nocturna "
        "nueva → asumir snack, no duplicar la cena)."
    )


def test_diario_de_hoy_reinforces_act_dont_ask_generally(agent_src: str):
    line = _diario_hoy_registrado_line(agent_src)
    assert "sin preguntar" in line, (
        "el opener reescrito debe reforzar la regla general (comida nueva en "
        "pasado = regístrala ya) en vez de limitarse a callar sobre duplicados."
    )


# ===========================================================================
# 8. Marker `_LAST_KNOWN_PFIX` — supersession-proof
# ===========================================================================

_MARKER_RE = re.compile(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"')


def test_marker_bumped_or_superseded():
    """Acepta el marker propio de este P-fix O cualquier P-fix POSTERIOR
    (fecha ISO >= 2026-07-28). `_LAST_KNOWN_PFIX` es last-writer-wins por
    diseño (lo compara `/health/version` contra deploy-lag) — atarlo al slug
    exacto de este fix dejaría el test rojo en cuanto el siguiente P-fix lo
    pise en la misma rama. La landmine real es usar SOLO
    `marker.startswith("P1-CHAT-ACT-DONT-ASK")` como criterio de paso: eso
    rechaza cualquier P-fix posterior con un slug distinto (ej.
    'P2-ALGO-NUEVO · 2026-08-01'), que sí debería pasar por venir después.
    Por eso el fallback compara fecha por `>=` sobre el string ISO
    (YYYY-MM-DD ordena lexicográficamente igual que temporalmente), no el
    nombre del slug.
    """
    src = _read(_APP_PY)
    m = _MARKER_RE.search(src)
    assert m, "no encontré _LAST_KNOWN_PFIX en app.py"
    marker = m.group(1)
    if "P1-CHAT-ACT-DONT-ASK" in marker:
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha and fecha.group(1) >= "2026-07-28", (
        f"marker {marker!r} es ANTERIOR a P1-CHAT-ACT-DONT-ASK (2026-07-28): "
        f"sin bump un operador no puede confirmar que este fix está vivo en "
        f"producción."
    )


def test_marker_slug_has_this_test_file():
    """Cross-link P2-HIST-AUDIT-14: si el marker actual ES este P-fix, el
    slug (`p1_chat_act_dont_ask`) debe matchear este mismo archivo — sanity
    trivial, pero documenta la convención para quien edite el marker luego."""
    src = _read(_APP_PY)
    m = _MARKER_RE.search(src)
    assert m
    marker = m.group(1)
    if "P1-CHAT-ACT-DONT-ASK" not in marker:
        pytest.skip("marker actual pertenece a un P-fix posterior")
    slug = marker.split("·", 1)[0].strip().replace("-", "_").lower()
    assert slug == "p1_chat_act_dont_ask"
    assert Path(__file__).name.startswith(f"test_{slug}")
