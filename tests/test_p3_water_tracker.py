"""[P3-WATER-TRACKER · 2026-05-16] Tests parser-based del tracker diario
de hidratacion. Reemplazo del card "Mi Nevera" del Dashboard.

Contrato anclado por estos tests:
  1. Migracion SSOT existe (`migrations/p3_water_tracker_2026_05_16.sql`)
     con CHECK de rango, RLS, y FK ON DELETE CASCADE.
  2. Backend expone GET y POST `/api/plans/water-intake` en `routers/plans.py`
     usando `_WATER_TRACKER_LIMITER` (rate limiter, NO `verify_api_quota` —
     cero costo LLM, patron Historial-quota-exemption).
  3. Frontend tiene componente `WaterTracker.jsx` en `components/dashboard/`
     y el `Dashboard.jsx` lo renderiza en lugar del card "Mi Nevera".
  4. La fecha enviada por el cliente es LOCAL (no UTC) — el helper
     `getLocalDateString` construye YYYY-MM-DD desde `getFullYear/Month/Date`,
     NO desde `toISOString` (que devolveria UTC y rompe el reset a
     medianoche local).

Limitaciones (out of scope):
  - NO testea ejecucion del endpoint contra DB real (eso requiere fixture
    Supabase). Sanity manual: GET ?date=YYYY-MM-DD devuelve {glasses: 0}
    para fechas sin row, POST con glasses=8 cambia el estado.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_WATER_TRACKER_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "WaterTracker.jsx"
_MIGRATION_SQL = _REPO_ROOT / "migrations" / "p3_water_tracker_2026_05_16.sql"


# ---------------------------------------------------------------------------
# 1. Migracion SSOT
# ---------------------------------------------------------------------------
def test_migration_file_exists():
    assert _MIGRATION_SQL.exists(), (
        f"Migracion SSOT ausente: {_MIGRATION_SQL}. "
        "Convencion CLAUDE.md: DDL en runtime prohibido."
    )


def test_migration_idempotent_and_has_constraints():
    sql = _MIGRATION_SQL.read_text(encoding="utf-8")
    # Idempotencia (P3-MIGRATION-IDEMPOTENCE-DOC).
    assert "CREATE TABLE IF NOT EXISTS public.water_intake_log" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_water_intake_log_user_date" in sql
    assert "DROP POLICY IF EXISTS" in sql
    # Constraint defensivo + FK cascade.
    assert re.search(r"CHECK\s*\(\s*glasses\s*>=\s*0\s+AND\s+glasses\s*<=\s*50\s*\)", sql), (
        "CHECK constraint sobre `glasses` ausente — riesgo de click-spam o bug del cliente "
        "metiendo valores absurdos."
    )
    assert "REFERENCES auth.users(id) ON DELETE CASCADE" in sql
    # RLS + 4 policies.
    assert "ENABLE ROW LEVEL SECURITY" in sql
    for cmd in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        assert f"FOR {cmd} TO authenticated" in sql, f"Policy para {cmd} faltante."
    # Sanity check.
    assert "RAISE EXCEPTION" in sql, "Sanity check final del DO $$ ausente."


# ---------------------------------------------------------------------------
# 2. Backend endpoints
# ---------------------------------------------------------------------------
def test_backend_endpoints_registered():
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert '@router.get("/water-intake")' in src, "GET /water-intake no registrado."
    assert '@router.post("/water-intake")' in src, "POST /water-intake no registrado."


def test_backend_uses_rate_limiter_not_quota():
    """Cero costo LLM → patron Historial-quota-exemption: usar rate limiter
    propio, NO `verify_api_quota` (paywall que devuelve 402 al exceder el cap
    mensual — bloquear el tracker de hidratacion por eso seria UX inaceptable).
    """
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert "_WATER_TRACKER_LIMITER = RateLimiter(" in src
    # El limiter debe estar como Depends en ambos endpoints. Buscamos el bloque
    # de la funcion GET y POST y validamos que `_WATER_TRACKER_LIMITER` aparece
    # antes que `verify_api_quota` no aparezca (negative assertion).
    get_block = re.search(
        r'@router\.get\("/water-intake"\).*?(?=@router\.)',
        src,
        re.DOTALL,
    )
    post_block = re.search(
        r'@router\.post\("/water-intake"\).*?(?=@router\.)',
        src,
        re.DOTALL,
    )
    assert get_block, "No se pudo aislar el bloque GET /water-intake."
    assert post_block, "No se pudo aislar el bloque POST /water-intake."
    for name, block in (("GET", get_block.group(0)), ("POST", post_block.group(0))):
        assert "_WATER_TRACKER_LIMITER" in block, (
            f"{name} /water-intake no usa _WATER_TRACKER_LIMITER."
        )
        assert "verify_api_quota" not in block, (
            f"{name} /water-intake usa verify_api_quota — debe usar el rate limiter "
            "(Historial-quota-exemption: cero costo LLM)."
        )


def test_backend_personalized_goal_helper_exists():
    """[P3-WATER-TRACKER · 2026-05-16] Helper `_compute_water_goal` debe:
      1. Existir como funcion en plans.py.
      2. Usar la formula 35ml/kg (constante `_WATER_ML_PER_KG = 35`).
      3. Clamp a [6, 14] vasos (constantes `_WATER_GOAL_MIN/MAX`).
      4. Aplicar bonus por activityLevel (dict `_WATER_ACTIVITY_BONUS_ML`).
      5. Convertir lb → kg via `/ 2.20462` (mismo patron que
         nutrition_calculator.py).
      6. Fallback fail-secure: cualquier exception → default 8 vasos.
    """
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert "def _compute_water_goal(user_id: str) -> dict:" in src
    assert "_WATER_ML_PER_KG = 35" in src
    assert "_WATER_GOAL_MIN = 6" in src
    assert "_WATER_GOAL_MAX = 14" in src
    assert "_WATER_ACTIVITY_BONUS_ML" in src
    # Conversion de libras a kg.
    assert "/ 2.20462" in src, (
        "Conversion lb→kg ausente. Debe usar `/ 2.20462` (mismo factor que "
        "nutrition_calculator.py)."
    )
    # Bonus activity levels canonicos.
    for level in ("sedentary", "moderate", "active", "very_active"):
        assert f'"{level}"' in src, f"Activity level `{level}` no listado en bonus dict."


def test_backend_get_endpoint_returns_goal_basis():
    """El GET /water-intake debe devolver `goal` + `goal_basis` (no
    el _WATER_DEFAULT_GOAL hardcoded). El frontend usa goal_basis.weight_kg
    para el subtitulo dinamico."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    get_block = re.search(
        r'@router\.get\("/water-intake"\).*?(?=@router\.)',
        src,
        re.DOTALL,
    )
    assert get_block, "No se pudo aislar GET /water-intake."
    body = get_block.group(0)
    assert "_compute_water_goal(verified_user_id)" in body, (
        "GET /water-intake no invoca `_compute_water_goal` — devuelve goal "
        "hardcoded en lugar del personalizado."
    )
    assert '"goal_basis"' in body, (
        "GET /water-intake no incluye `goal_basis` en la respuesta — el "
        "frontend no podra renderizar el subtitulo con el peso del usuario."
    )


def test_backend_validates_date_and_glasses():
    """El handler POST debe rechazar fechas / glasses invalidos antes del upsert."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    # Helper de validacion de fecha existe.
    assert "_validate_water_date" in src
    # El POST valida tipo int (no bool — Python bool isinstance int).
    post_block = re.search(
        r'@router\.post\("/water-intake"\).*?(?=@router\.)',
        src,
        re.DOTALL,
    )
    assert post_block
    body = post_block.group(0)
    assert "isinstance(raw_glasses, int)" in body
    assert "isinstance(raw_glasses, bool)" in body, (
        "Sin el check de bool, `True`/`False` pasan como 1/0 (Python: bool subclass int)."
    )
    assert "_WATER_MAX_GLASSES" in body


# ---------------------------------------------------------------------------
# 3. Frontend
# ---------------------------------------------------------------------------
def test_watertracker_component_exists():
    assert _WATER_TRACKER_JSX.exists(), (
        f"Componente WaterTracker.jsx ausente: {_WATER_TRACKER_JSX}"
    )


def test_watertracker_uses_dynamic_goal_and_subtitle():
    """El componente debe leer `goal` + `goal_basis` del response del backend
    (no hardcodear 8 vasos). El subtitulo dinamico debe mencionar el peso
    cuando `goal_basis.default === false`."""
    src = _WATER_TRACKER_JSX.read_text(encoding="utf-8")
    # State para goal + goal_basis.
    assert "const [goal, setGoal]" in src, "Goal debe ser state, no constante."
    assert "const [goalBasis, setGoalBasis]" in src, "goalBasis state ausente."
    # Sanea el goal recibido (defensa contra payloads corruptos).
    assert "sanitizeGoal" in src, (
        "Falta `sanitizeGoal` — sin sanitizar, un backend devolviendo NaN/null/"
        "fuera-de-rango rompe la UI."
    )
    # Subtitulo menciona peso cuando viene de la formula.
    assert "goalBasis.weight_kg" in src or "goalBasis?.weight_kg" in src, (
        "Subtitulo dinamico no usa goal_basis.weight_kg. El usuario no "
        "entiende POR QUE su meta es N vasos."
    )
    # Grid responsivo: columnas dinamicas (ceil(goal/2) si goal>8).
    assert "columnsPerRow" in src, (
        "Variable `columnsPerRow` ausente — la grilla no es responsiva al goal."
    )
    assert "Math.ceil(goal / 2)" in src, (
        "Wrap a 2 filas via Math.ceil(goal/2) ausente. Goals 9-14 saldrian "
        "en 1 sola fila apretada."
    )


def test_watertracker_uses_local_date_not_utc():
    """El reset a medianoche local depende de que el cliente envie su fecha
    LOCAL (YYYY-MM-DD construida desde getFullYear/Month/Date). Usar
    `toISOString` devolveria UTC y romperia el reset en timezones con offset.
    """
    src = _WATER_TRACKER_JSX.read_text(encoding="utf-8")
    assert "getLocalDateString" in src
    assert "getFullYear()" in src and "getMonth()" in src and "getDate()" in src, (
        "WaterTracker debe construir la fecha desde getFullYear/Month/Date "
        "(local), NO desde toISOString (UTC)."
    )
    # Negative: no debe usar toISOString para derivar la fecha del dia.
    assert "toISOString" not in src, (
        "WaterTracker NO debe usar toISOString — devuelve UTC y rompe el "
        "reset a medianoche local en timezones con offset > 0."
    )


def test_dashboard_imports_and_renders_watertracker():
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "import WaterTracker from '../components/dashboard/WaterTracker'" in src, (
        "Dashboard.jsx no importa WaterTracker."
    )
    assert "<WaterTracker" in src, "Dashboard.jsx no renderiza WaterTracker."


def test_dashboard_mobile_renders_water_tracker_above_meals():
    """[P3-WATER-TRACKER · 2026-05-16] En mobile el WaterTracker DEBE
    renderizar ENCIMA del menu de comidas (`.main-grid`). Esto requiere:
      1. Hook `isMobileViewport` definido (matchMedia con breakpoint 768px).
      2. Un render gated por `isMobileViewport && <WaterTracker />` ANTES
         del `<div className="main-grid">`.
      3. Un render gated por `!isMobileViewport && <WaterTracker />`
         DENTRO de la columna derecha (sigue siendo el render de desktop).
    Una sola instancia activa a la vez (mobile XOR desktop) — el render
    condicional evita doble fetch y divergencia de state.
    """
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "isMobileViewport" in src, (
        "Hook `isMobileViewport` ausente — sin el, no hay forma de elegir "
        "donde renderizar WaterTracker."
    )
    # matchMedia con el breakpoint canonico (768px) del resto del Dashboard.
    assert "matchMedia('(max-width: 768px)')" in src, (
        "Detector de mobile debe usar `matchMedia('(max-width: 768px)')` "
        "para alinearse con las media queries existentes del Dashboard."
    )
    # Mobile render ANTES del .main-grid (orden textual = orden visual).
    mobile_marker = "isMobileViewport && <WaterTracker"
    grid_marker = '<div className="main-grid">'
    assert mobile_marker in src, (
        "No se encontro `{isMobileViewport && <WaterTracker />}` (render mobile)."
    )
    assert grid_marker in src, "No se encontro `.main-grid` container."
    assert src.index(mobile_marker) < src.index(grid_marker), (
        "El render mobile de WaterTracker debe aparecer ANTES de "
        "`.main-grid` (textual y visualmente)."
    )
    # Desktop render gated explicitamente por NOT mobile.
    assert "!isMobileViewport && <WaterTracker" in src, (
        "El render dentro de la columna derecha debe estar gated por "
        "`!isMobileViewport` — sin esa guarda se duplicaria con el mobile."
    )


def test_dashboard_removed_mi_nevera_card():
    """El card antiguo "Mi Nevera" (titulo h3 con texto literal "Mi Nevera"
    + parrafo "Lo que tienes en casa hoy") fue removido del Dashboard.
    Esta es la decision de producto que motivo el feature.
    """
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # El string "Mi Nevera" no debe aparecer como contenido renderizado.
    # Permitimos su mencion en comentarios (// o /* */).
    # Strip de comentarios single-line y multi-line para evaluar.
    stripped = re.sub(r"//[^\n]*", "", src)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    assert "Mi Nevera" not in stripped, (
        '"Mi Nevera" sigue apareciendo como texto renderizado en Dashboard.jsx. '
        "El card debe haber sido reemplazado por <WaterTracker />."
    )
    assert "Lo que tienes en casa hoy" not in stripped, (
        '"Lo que tienes en casa hoy" (subtitulo del card removido) sigue presente.'
    )


# ---------------------------------------------------------------------------
# 4. Marker
# ---------------------------------------------------------------------------
def test_marker_water_tracker_test_file_still_present():
    """[P3-WATER-TRACKER · 2026-05-16] One-time guard del bump del marker
    (originalmente exigía `_LAST_KNOWN_PFIX = "P3-WATER-TRACKER · 2026-05-16"`).
    Suavizado tras 4 bumps subsecuentes el mismo día (cost-instrumentation,
    cost-by-node, critique-timeout-v2): solo verifica que ESTE archivo de
    tests existe en la ubicación esperada para que el cross-link de
    P2-HIST-AUDIT-14 lo pueda matchear si alguien revierte el marker a
    `P3-WATER-TRACKER`. Tests test_p3_1 + test_p2_hist_audit_14 hacen
    el enforcement vivo del marker."""
    assert Path(__file__).exists(), "Este archivo debe existir para el cross-link."


# ---------------------------------------------------------------------------
# 5. Toggle de Preferencias (enabled/disabled)
# ---------------------------------------------------------------------------
_PREFERENCES_PY = _BACKEND_ROOT / "routers" / "preferences.py"
_DB_PROFILES_PY = _BACKEND_ROOT / "db_profiles.py"
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"
_TOGGLE_MIGRATION = _REPO_ROOT / "migrations" / "add_water_tracker_enabled_2026_05_16.sql"


def test_toggle_migration_exists_and_idempotent():
    """Migration SSOT para `water_tracker_enabled` debe existir y ser
    idempotente (IF NOT EXISTS) — convencion P3-MIGRATION-IDEMPOTENCE-DOC."""
    assert _TOGGLE_MIGRATION.exists(), f"Migration ausente: {_TOGGLE_MIGRATION}"
    sql = _TOGGLE_MIGRATION.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS water_tracker_enabled BOOLEAN" in sql
    assert "DEFAULT TRUE" in sql, "Default debe ser TRUE (predeterminado activado)."
    assert "RAISE EXCEPTION" in sql, "Sanity check DO $$ ausente."


def test_backend_toggle_helpers_exist():
    """`update_water_tracker_enabled` + `get_water_tracker_enabled` deben
    existir en db_profiles.py con filtros por user_id (invariante I2)."""
    src = _DB_PROFILES_PY.read_text(encoding="utf-8")
    assert "def update_water_tracker_enabled(user_id: str, enabled: bool)" in src
    assert "def get_water_tracker_enabled(user_id: str)" in src
    # Update filtra por id (I2).
    upd_block = re.search(
        r"def update_water_tracker_enabled.*?(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert upd_block and '.eq("id", user_id)' in upd_block.group(0), (
        "update_water_tracker_enabled debe filtrar por id=user_id (I2)."
    )


def test_backend_toggle_endpoints_registered():
    """Endpoints GET + PATCH /api/user/preferences/water-tracker."""
    src = _PREFERENCES_PY.read_text(encoding="utf-8")
    assert '@router.patch("/water-tracker")' in src
    assert '@router.get("/water-tracker")' in src
    # Sin gate de tier (disponible para todos los autenticados).
    # El endpoint NO debe llamar a verify_api_quota (cero costo LLM).
    pref_block = re.search(
        r'@router\.(get|patch)\("/water-tracker"\).*?(?=@router\.|\Z)',
        src,
        re.DOTALL,
    )
    assert pref_block
    assert "verify_api_quota" not in pref_block.group(0), (
        "Endpoint del toggle NO debe usar verify_api_quota (cero costo LLM)."
    )


def test_water_intake_get_includes_enabled():
    """El GET /water-intake debe incluir `enabled` para evitar un fetch
    separado a /preferences/water-tracker en el mount del componente."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    get_block = re.search(
        r'@router\.get\("/water-intake"\).*?(?=@router\.)',
        src,
        re.DOTALL,
    )
    assert get_block, "No se pudo aislar GET /water-intake."
    body = get_block.group(0)
    assert "get_water_tracker_enabled(verified_user_id)" in body, (
        "GET /water-intake no invoca get_water_tracker_enabled — el frontend "
        "tendria que hacer un fetch separado (race + roundtrip extra)."
    )
    assert '"enabled"' in body, "Response no incluye campo `enabled`."


def test_watertracker_gates_on_enabled():
    """El componente debe:
      1. Leer `enabled` de localStorage al mount (cache para evitar flash).
      2. Retornar null si `enabled === false`.
      3. Tener listener `visibilitychange` para Fix 3 (re-GET cross-tab).
      4. Tener listener `storage` event para reaccionar al toggle en otra tab.
    """
    src = _WATER_TRACKER_JSX.read_text(encoding="utf-8")
    assert "LS_ENABLED_KEY" in src and "'mealfit_water_tracker_enabled'" in src
    assert "readEnabledFromCache" in src, (
        "Falta helper para leer cache local — sin esto, hay flash visual en mount."
    )
    assert "if (!enabled) return null" in src, (
        "Gate por enabled ausente — el toggle no surte efecto."
    )
    # Fix 3: cross-tab listeners.
    assert "'visibilitychange'" in src, (
        "Listener visibilitychange ausente (Fix 3 cross-tab refresh)."
    )
    assert "'storage'" in src, (
        "Listener storage event ausente (Fix 3 reaccion al toggle en otra tab)."
    )


def test_watertracker_subtitle_affirmative_when_personalized():
    """[Fix 1] El subtitulo debe usar copy afirmativo (`personalizado para`)
    cuando `goalBasis.default === false`, distinguible del fallback default
    incluso cuando el goal computado coincide con 8."""
    src = _WATER_TRACKER_JSX.read_text(encoding="utf-8")
    assert "personalizado para" in src, (
        "Subtitulo no usa copy afirmativo. Cuando goal computado = 8 (caso "
        "comun: usuarios 50-60kg), el subtitulo es indistinguible del default."
    )


def test_dashboard_no_isPlanExpired_gate_on_watertracker():
    """[Fix 2] El render de <WaterTracker /> NO debe estar gated por
    isPlanExpired. La hidratacion es independiente del ciclo del plan."""
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # Buscar TODAS las apariciones de <WaterTracker /> y validar ninguna
    # esta precedida (en el JSX expr) por `!isPlanExpired`.
    forbidden = "!isPlanExpired && isMobileViewport && <WaterTracker"
    forbidden2 = "!isPlanExpired && !isMobileViewport && <WaterTracker"
    assert forbidden not in src, (
        f"Encontrado gate prohibido `{forbidden}`. La hidratacion es "
        "independiente del plan — quitar el `!isPlanExpired &&`."
    )
    assert forbidden2 not in src, (
        f"Encontrado gate prohibido `{forbidden2}`."
    )


# ---------------------------------------------------------------------------
# 6. Integracion con el chat agent (tools check_hydration_today + log_water_glass)
# ---------------------------------------------------------------------------
_TOOLS_PY = _BACKEND_ROOT / "tools.py"
_CHAT_AGENT_PROMPTS_PY = _BACKEND_ROOT / "prompts" / "chat_agent.py"
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


def test_agent_tools_check_hydration_and_log_glass_exist():
    """Las 2 tools de hidratacion (check + log) deben estar en agent_tools."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    # Decorador @tool aplicado a ambas funciones.
    assert "@tool\ndef check_hydration_today(user_id: str)" in src, (
        "Tool `check_hydration_today` ausente o sin decorador @tool."
    )
    assert "@tool\ndef log_water_glass(user_id: str" in src, (
        "Tool `log_water_glass` ausente o sin decorador @tool."
    )
    # Ambas registradas en la lista agent_tools.
    agent_tools_line = re.search(r"^agent_tools\s*=\s*\[(.*?)\]", src, re.MULTILINE | re.DOTALL)
    assert agent_tools_line, "Lista `agent_tools` no encontrada."
    body = agent_tools_line.group(1)
    assert "check_hydration_today" in body, (
        "`check_hydration_today` no listada en agent_tools — el agente no la "
        "podra invocar."
    )
    assert "log_water_glass" in body, (
        "`log_water_glass` no listada en agent_tools."
    )


def test_water_tools_accept_user_id_first_arg():
    """Por contrato P0-AGENT-1, las tools que aceptan `user_id` reciben el
    force-override del trusted UID. Sin `user_id` como primer arg, el override
    no cubre la tool y abre IDOR cross-user."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    # Signature check explicito (no permitir alias como `uid` o `verified_user`).
    assert "def check_hydration_today(user_id: str)" in src
    # log_water_glass tiene segundo arg con default — primer arg debe ser user_id.
    assert re.search(r"def log_water_glass\(user_id: str,\s*count_delta", src), (
        "log_water_glass debe tener `user_id: str` como PRIMER argumento."
    )


def test_chat_agent_prompt_mentions_water_tools():
    """build_tools_instructions + build_tools_instructions_stream deben mencionar
    las 2 tools nuevas + la regla del UI_ACTION REFRESH_HYDRATION."""
    src = _CHAT_AGENT_PROMPTS_PY.read_text(encoding="utf-8")
    # Tool names en el system prompt (ambos formatos: full + stream).
    assert src.count("check_hydration_today") >= 2, (
        "check_hydration_today debe mencionarse en AMBAS instrucciones "
        "(build_tools_instructions + _stream)."
    )
    assert src.count("log_water_glass") >= 2, (
        "log_water_glass debe mencionarse en ambas instrucciones."
    )
    # UI_ACTION REFRESH_HYDRATION reglado en ambas variantes.
    assert src.count("REFRESH_HYDRATION") >= 2, (
        "Regla `[UI_ACTION: REFRESH_HYDRATION]` debe estar en ambas "
        "instrucciones para que el agente la incluya tras log_water_glass."
    )


def test_agent_page_handles_refresh_hydration():
    """AgentPage.jsx debe:
      1. Stripear el tag `[UI_ACTION: REFRESH_HYDRATION]` del texto visible.
      2. Disparar custom event `mealfit:refresh-hydration` para que el
         WaterTracker refetchee sin necesidad de navegar.
    """
    src = _AGENT_PAGE_JSX.read_text(encoding="utf-8")
    assert "REFRESH_HYDRATION" in src, (
        "AgentPage no estripea el tag REFRESH_HYDRATION — quedara visible en el chat."
    )
    assert "mealfit:refresh-hydration" in src, (
        "AgentPage no dispara el custom event — el card no refrescara sin reload."
    )


def test_water_tracker_listens_to_agent_refresh_event():
    """WaterTracker debe escuchar `mealfit:refresh-hydration` para refrescar
    cuando el chat agent muta el conteo via log_water_glass."""
    src = _WATER_TRACKER_JSX.read_text(encoding="utf-8")
    assert "'mealfit:refresh-hydration'" in src, (
        "WaterTracker no escucha el evento del agente — log_water_glass "
        "modificara el DB pero el card no se actualizara hasta el proximo "
        "reload o visibilitychange."
    )


def test_settings_toggle_ui_present():
    """Settings.jsx debe:
      1. Importar GlassWater de lucide-react.
      2. Tener state waterTrackerEnabled + handler handleToggleWaterTracker.
      3. Hacer fetch al endpoint GET /api/user/preferences/water-tracker.
      4. Hacer PATCH al mismo endpoint en el handler.
      5. Sincronizar localStorage para evitar flash en Dashboard.
    """
    src = _SETTINGS_JSX.read_text(encoding="utf-8")
    assert "GlassWater" in src, "GlassWater no importado en Settings."
    assert "waterTrackerEnabled" in src and "setWaterTrackerEnabled" in src
    assert "handleToggleWaterTracker" in src
    assert "'/api/user/preferences/water-tracker'" in src
    # PATCH del toggle.
    handler_block = re.search(
        r"handleToggleWaterTracker\s*=\s*async\s*\(\)\s*=>\s*\{.*?\n\s{4}\}",
        src,
        re.DOTALL,
    )
    assert handler_block
    body = handler_block.group(0)
    assert "method: 'PATCH'" in body, "Handler no hace PATCH."
    assert "mealfit_water_tracker_enabled" in body, (
        "Handler no sincroniza localStorage — el Dashboard tendra flash al "
        "regresar tras toggle off."
    )
