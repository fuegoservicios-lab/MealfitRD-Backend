"""[P1-DIARY-EDITABLE · 2026-07-28] El diario de comidas ("consumed_meals")
deja de ser append-only e inmutable-al-presente:

  1. `POST /api/diary/consumed` acepta `days_ago` (0..7) para registrar un
     día ANTERIOR (antes SIEMPRE escribía `NOW()` — un usuario que olvidó
     loguear el fin de semana no podía corregirlo retroactivamente).
  2. `DELETE /api/diary/consumed/{meal_id}` permite borrar una fila mal
     registrada (antes vivía para siempre hasta purgar la cuenta).
  3. `get_consumed_meals_today` expone `id` (`db_facts.py`) — sin esa
     columna, ningún caller puede identificar QUÉ fila borrar.
  4. `tools._CONSUMED_MAX_DAYS_AGO` sube de 3 a 7 para que el chat-agent y
     el endpoint manual acuerden el mismo tope.

Por qué importa: el chat coach recién aprendió a leer el diario
(`agent.py::get_consumed_meals_today`) — un diario que la gente abandona
porque no puede corregir un error de tap significa que el coach responde
"SIN REGISTRO" para siempre.

Este archivo cubre, en el espíritu de `test_p3_next_1_i2_user_id_filter_contract.py`
(pinnear SEMÁNTICA, no solo presencia de substring):

  - El clamp de `days_ago` nunca produce un timestamp futuro ni más de 7
    días atrás, para input sano Y basura (garbage/negativo/futuro → hoy).
  - `get_consumed_meals_today` selecciona `id` como columna real (no en un
    comentario) y la query realmente usa `_COLUMNS`.
  - El DELETE de `db_facts.delete_consumed_meal` lleva `AND user_id = %s`
    en el MISMO statement que el filtro por `id` (invariante I2).
  - `delete_consumed_meal` retorna falsy cuando `RETURNING id` viene vacío
    (nada borrado) y truthy cuando sí borra — el endpoint depende de esta
    distinción para 404 honesto vs 200.
  - El endpoint DELETE está gateado por un `RateLimiter` (quota-exempt,
    doctrina "Historial-quota-exemption" de CLAUDE.md), NUNCA por
    `verify_api_quota`.
  - El wiring end-to-end del handler: el DELETE se ejecuta con el
    `verified_user_id` AUTENTICADO como scope — nunca con un id que el
    cliente pudiera inyectar (no hay `user_id` en el body/path del DELETE,
    solo `meal_id` + la identidad que resolvió `get_verified_user_id`).

Tooltip-anchor: P1-DIARY-EDITABLE
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures: source de los archivos tocados por este P-fix.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_facts_src() -> str:
    return (_BACKEND / "db_facts.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diary_src() -> str:
    return (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return (_BACKEND / "tools.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_agent_prompts_src() -> str:
    return (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    """Devuelve el cuerpo de `def <fn_name>(` hasta el siguiente `\\ndef `
    top-level (o EOF). Tolerante a async def."""
    i = src.find(f"def {fn_name}(")
    assert i != -1, f"`{fn_name}` no encontrada en el source."
    j = src.find("\ndef ", i + 1)
    return src[i: j if j != -1 else len(src)]


def _import_diary_module():
    """Import defensivo — `routers.diary` arrastra db_core/vision_agent/
    cron_tasks al import; si el entorno de CI no puede construir esa cadena
    (falta de env vars, etc.), los tests funcionales que dependen de esto
    se saltan con skip explícito. Los tests parser-based (fuente-como-texto)
    de este archivo NO dependen de este import y siguen cubriendo el
    contrato en ese caso."""
    try:
        import routers.diary as diary_mod
        return diary_mod
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"routers.diary no importable en este entorno ({type(e).__name__}: {e})")


# ===========================================================================
# 1. `days_ago` clamps — 0, 7, 8, -1, "basura", None nunca producen un
#    override futuro ni > 7 días atrás.
# ===========================================================================
class TestDaysAgoClamp:
    @pytest.fixture(scope="class")
    def clamp_fn(self):
        diary_mod = _import_diary_module()
        return diary_mod._clamp_diary_days_ago

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0, 0),
            (7, 7),
            (8, 7),       # por encima del tope → clampa al tope, NO rechaza
            (-1, 0),      # negativo (futuro) → hoy
            ("basura", 0),  # garbage no-numérico → hoy
            (None, 0),
        ],
    )
    def test_clamp_values(self, clamp_fn, raw, expected):
        assert clamp_fn(raw) == expected

    @pytest.mark.parametrize("raw", [0, 7, 8, -1, "basura", None, 999, "-50"])
    def test_clamp_result_bounded_0_to_7(self, clamp_fn, raw):
        """Invariante estructural: para CUALQUIER input, el resultado cae
        en [0, 7] — el rango legal del backdate."""
        result = clamp_fn(raw)
        assert isinstance(result, int)
        assert 0 <= result <= 7

    @pytest.mark.parametrize("raw", [0, 7, 8, -1, "basura", None])
    def test_resulting_override_never_future_never_over_7_days(self, clamp_fn, raw):
        """Replica la fórmula real del handler
        (`datetime.now(timezone.utc) - timedelta(days=clamped)`) contra un
        `now` FIJO (no depende de la fecha real de ejecución — evita el
        modo de fallo "verde hoy, rojo mañana" que ya mordió a esta suite
        varias veces) y verifica el contrato de negocio: nunca a futuro,
        nunca más de 7 días atrás."""
        frozen_now = datetime(2026, 7, 20, 15, 30, 0, tzinfo=timezone.utc)
        clamped = clamp_fn(raw)
        override = frozen_now - timedelta(days=clamped)
        assert override <= frozen_now, "el diario nunca registra a futuro"
        assert (frozen_now - override) <= timedelta(days=7), "máximo 7 días atrás"


# ===========================================================================
# 2. `get_consumed_meals_today` expone `id` como columna REAL (no comentario)
#    y la query la usa de verdad.
# ===========================================================================
class TestConsumedMealsIdColumn:
    def test_columns_tuple_includes_id_token(self, db_facts_src: str):
        body = _extract_function_body(db_facts_src, "get_consumed_meals_today")
        m = re.search(r'_COLUMNS\s*=\s*\(([\s\S]*?)\)', body)
        assert m, "`_COLUMNS` no encontrado dentro de get_consumed_meals_today."
        columns_raw = m.group(1)
        # Concatenación adyacente de strings Python ("a, " "b") → un solo string.
        concatenated = re.sub(r'"\s*\n\s*"', '', columns_raw).strip().strip('"')
        tokens = [c.strip() for c in concatenated.split(",") if c.strip()]
        assert "id" in tokens, (
            f"`id` ausente de _COLUMNS (tokens={tokens}). Sin `id`, ningún "
            f"caller (frontend Diario) puede identificar QUÉ fila borrar con "
            f"DELETE /api/diary/consumed/{{meal_id}}."
        )
        # No es un match espurio de substring (ej. dentro de otra palabra).
        assert "healthy_fats" in tokens and "meal_name" in tokens, (
            "sanity: las columnas históricas deben seguir presentes"
        )

    def test_query_actually_uses_columns_variable(self, db_facts_src: str):
        body = _extract_function_body(db_facts_src, "get_consumed_meals_today")
        assert "SELECT {_COLUMNS} FROM consumed_meals" in body, (
            "la query dejó de interpolar `_COLUMNS` — el fix de arriba "
            "sería dead code si esto no está presente."
        )

    def test_other_caller_uses_dict_get_not_positional_unpack(self):
        """agent.py debe seguir leyendo por KEY (`m.get('calories')`, etc.)
        — una columna extra (`id`) es aditiva y segura SOLO si nadie hace
        unpacking posicional de las filas."""
        agent_src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
        # Las dos invocaciones documentadas de get_consumed_meals_today.
        assert agent_src.count("get_consumed_meals_today(") >= 2
        # Debe seguir accediendo vía .get(, nunca desempaquetar tuplas
        # tipo `for a, b, c in consumed_today:`.
        assert not re.search(
            r"for\s+\w+\s*,\s*\w+.*?\s+in\s+consumed_today",
            agent_src,
        ), "positional unpacking de consumed_today detectado — la columna `id` rompería el orden."
        assert "m.get('calories'" in agent_src or 'm.get("calories"' in agent_src


# ===========================================================================
# 3. `delete_consumed_meal` — el DELETE lleva `AND user_id = %s` en el
#    MISMO statement (invariante I2), y retorna falsy en no-op.
# ===========================================================================
class TestDeleteConsumedMealDbLayer:
    def test_delete_sql_carries_and_user_id_same_statement(self, db_facts_src: str):
        body = _extract_function_body(db_facts_src, "delete_consumed_meal")
        m = re.search(r'"(DELETE FROM consumed_meals[^"]*)"', body)
        assert m, "delete_consumed_meal no emite un `DELETE FROM consumed_meals` reconocible."
        stmt = m.group(1)
        assert re.search(r"\bAND\s+user_id\s*=\s*%s", stmt, re.IGNORECASE), (
            "El DELETE debe filtrar `AND user_id = %s` (invariante I2, "
            "CLAUDE.md). Sin este filtro EN EL MISMO statement, cualquier "
            "usuario autenticado que adivine/enumere un `id` ajeno podría "
            "borrar filas de OTRO usuario (IDOR)."
        )
        # `user_id` debe co-existir DESPUÉS del filtro por `id` dentro del
        # MISMO WHERE — no en un comentario ni en un statement separado.
        id_pos = stmt.find("id = %s")
        uid_pos = stmt.find("user_id = %s")
        assert id_pos != -1 and uid_pos != -1 and id_pos < uid_pos, (
            "el filtro `user_id` debe estar en el mismo WHERE que el filtro `id`."
        )

    def test_delete_uses_returning_to_distinguish_noop(self, db_facts_src: str):
        body = _extract_function_body(db_facts_src, "delete_consumed_meal")
        assert "RETURNING id" in body, (
            "sin RETURNING, el helper no puede distinguir 'borré 1 fila' de "
            "'no había ninguna con ese (id, user_id)' → el endpoint no "
            "podría 404 honesto."
        )
        assert "returning=True" in body

    def test_returns_falsy_when_no_row_matched(self, monkeypatch):
        import db_facts
        monkeypatch.setattr(db_facts, "connection_pool", object())  # truthy sentinel
        monkeypatch.setattr(db_facts, "execute_sql_write", lambda *a, **k: [])
        result = db_facts.delete_consumed_meal("user-1", "meal-nonexistent")
        assert not result, "RETURNING vacío (0 filas) debe producir un resultado falsy."

    def test_returns_truthy_when_row_matched(self, monkeypatch):
        import db_facts
        monkeypatch.setattr(db_facts, "connection_pool", object())
        monkeypatch.setattr(db_facts, "execute_sql_write", lambda *a, **k: [{"id": "meal-1"}])
        assert db_facts.delete_consumed_meal("user-1", "meal-1") is True

    def test_passes_meal_id_and_user_id_as_query_params(self, monkeypatch):
        """Verifica que los PARÁMETROS reales pasados a la query coinciden
        con (meal_id, user_id) — no solo que el string SQL los mencione."""
        import db_facts
        captured = {}

        def fake_write(query, params, **kwargs):
            captured["query"] = query
            captured["params"] = params
            return [{"id": "meal-ABC"}]

        monkeypatch.setattr(db_facts, "connection_pool", object())
        monkeypatch.setattr(db_facts, "execute_sql_write", fake_write)
        db_facts.delete_consumed_meal("user-XYZ", "meal-ABC")
        assert captured["params"] == ("meal-ABC", "user-XYZ"), (
            "el orden de los params debe matchear el orden de los "
            "placeholders `WHERE id = %s AND user_id = %s`."
        )

    def test_short_circuits_without_pool(self, monkeypatch):
        import db_facts

        def boom(*a, **k):
            raise AssertionError("no debe invocar execute_sql_write sin connection_pool")

        monkeypatch.setattr(db_facts, "connection_pool", None)
        monkeypatch.setattr(db_facts, "execute_sql_write", boom)
        assert db_facts.delete_consumed_meal("user-1", "meal-1") is False

    def test_swallows_db_exceptions_and_returns_false(self, monkeypatch):
        import db_facts

        def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(db_facts, "connection_pool", object())
        monkeypatch.setattr(db_facts, "execute_sql_write", boom)
        assert db_facts.delete_consumed_meal("user-1", "meal-1") is False


# ===========================================================================
# 4. Endpoint DELETE — registrado con RateLimiter, NUNCA verify_api_quota.
# ===========================================================================
class TestDeleteEndpointQuotaExemption:
    def _handler_block(self, diary_src: str) -> str:
        i = diary_src.find('@router.delete("/consumed/{meal_id}")')
        assert i != -1, "endpoint DELETE /consumed/{meal_id} no encontrado."
        j = diary_src.find("\n@router.", i + 1)
        return diary_src[i: j if j != -1 else len(diary_src)]

    def test_endpoint_uses_ratelimiter_dependency(self, diary_src: str):
        block = self._handler_block(diary_src)
        assert re.search(r"Depends\(\s*_DELETE_CONSUMED_LIMITER\s*\)", block), (
            "el endpoint DELETE debe depender de un RateLimiter dedicado "
            "(mismo patrón que _RESTOCK_LIMITER/_CONSUME_LIMITER)."
        )

    def test_endpoint_does_not_use_verify_api_quota(self, diary_src: str):
        block = self._handler_block(diary_src)
        assert not re.search(r"Depends\(\s*verify_api_quota\s*\)", block), (
            "borrar una fila mal registrada es costo LLM CERO — aplicar el "
            "paywall mensual bloquearía corregir un error justo cuando el "
            "usuario ya agotó su cap. Ver CLAUDE.md 'Historial-quota-exemption'."
        )
        # Defense-in-depth: `verify_api_quota` no debe estar ni IMPORTADO en
        # el router (el módulo lo menciona solo en comentarios explicando
        # POR QUÉ no se usa — eso es intencional, ver P1-MEAL-SCAN-GEMMA).
        assert not re.search(r"^\s*from\s+\S+\s+import\s+[^#\n]*\bverify_api_quota\b", diary_src, re.MULTILINE), (
            "regresión: `verify_api_quota` volvió a importarse en routers/diary.py."
        )
        assert not re.search(r"Depends\(\s*verify_api_quota\s*\)", diary_src), (
            "regresión: `Depends(verify_api_quota)` reapareció en algún endpoint del router."
        )

    def test_limiter_instantiated_with_house_default_20_per_60s(self, diary_src: str):
        assert re.search(
            r"_DELETE_CONSUMED_LIMITER\s*=\s*RateLimiter\(\s*max_calls\s*=\s*20\s*,\s*period_seconds\s*=\s*60\s*\)",
            diary_src,
        ), "`_DELETE_CONSUMED_LIMITER` debe instanciarse 20/60s (house default de esta clase)."

    def test_endpoint_validates_uuid_shape(self, diary_src: str):
        block = self._handler_block(diary_src)
        assert "assert_valid_uuid(meal_id)" in block, (
            "el endpoint debe validar la forma de `meal_id` antes de tocar SQL "
            "(paridad con GET /consumed/{user_id})."
        )


# ===========================================================================
# 5. Endpoint DELETE — wiring funcional: 403/400/404/200 + scope por
#    verified_user_id (nunca por un id inyectado por el cliente).
# ===========================================================================
class TestDeleteEndpointFunctional:
    _VALID_MEAL_ID = "11111111-1111-1111-1111-111111111111"
    _VALID_USER_ID = "22222222-2222-2222-2222-222222222222"

    def test_no_auth_raises_403(self):
        diary_mod = _import_diary_module()
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            diary_mod.api_delete_consumed_meal(meal_id=self._VALID_MEAL_ID, verified_user_id=None)
        assert exc_info.value.status_code == 403

    def test_malformed_meal_id_raises_400(self):
        diary_mod = _import_diary_module()
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            diary_mod.api_delete_consumed_meal(meal_id="not-a-uuid", verified_user_id=self._VALID_USER_ID)
        assert exc_info.value.status_code == 400

    def test_noop_delete_raises_404(self, monkeypatch):
        diary_mod = _import_diary_module()
        from fastapi import HTTPException

        monkeypatch.setattr(diary_mod, "delete_consumed_meal", lambda user_id, meal_id: False)
        with pytest.raises(HTTPException) as exc_info:
            diary_mod.api_delete_consumed_meal(
                meal_id=self._VALID_MEAL_ID, verified_user_id=self._VALID_USER_ID
            )
        assert exc_info.value.status_code == 404

    def test_success_scopes_delete_to_verified_user_not_client_input(self, monkeypatch):
        """La defensa IDOR real: no hay `user_id` en el request del DELETE
        (solo `meal_id` en el path) — el scope viene EXCLUSIVAMENTE de
        `verified_user_id` (resuelto server-side por `get_verified_user_id`
        vía el RateLimiter). Este test falla si algún refactor futuro
        empieza a aceptar un `user_id` del cliente para este endpoint."""
        diary_mod = _import_diary_module()
        captured = {}

        def fake_delete(user_id, meal_id):
            captured["user_id"] = user_id
            captured["meal_id"] = meal_id
            return True

        monkeypatch.setattr(diary_mod, "delete_consumed_meal", fake_delete)
        result = diary_mod.api_delete_consumed_meal(
            meal_id=self._VALID_MEAL_ID, verified_user_id=self._VALID_USER_ID
        )
        assert result.get("success") is True
        assert captured["user_id"] == self._VALID_USER_ID
        assert captured["meal_id"] == self._VALID_MEAL_ID


# ===========================================================================
# 6. POST /consumed — `days_ago` end-to-end: `log_consumed_meal` recibe un
#    `consumed_at_override` acorde al clamp, nunca futuro / nunca >7 días.
# ===========================================================================
class TestPostConsumedDaysAgoWiring:
    def test_field_declared_with_bounds_0_to_7(self, diary_src: str):
        assert re.search(
            r"days_ago:\s*int\s*=\s*Field\(\s*default\s*=\s*0\s*,\s*ge\s*=\s*0\s*,\s*le\s*=\s*7\s*\)",
            diary_src,
        ), "`ConsumedMealRequest.days_ago` debe declarar Field(default=0, ge=0, le=7)."

    def test_handler_passes_consumed_at_override_derived_from_clamp(self, monkeypatch):
        diary_mod = _import_diary_module()
        from fastapi import BackgroundTasks

        captured = {}

        def fake_log(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return True

        monkeypatch.setattr(diary_mod, "log_consumed_meal", fake_log)

        payload = diary_mod.ConsumedMealRequest(
            user_id=self._uid(),
            meal_name="Sancocho de ayer",
            meal_type="almuerzo",
            calories=650,
            protein=35,
            carbs=60,
            healthy_fats=15,
            days_ago=3,
        )
        before = datetime.now(timezone.utc)
        result = diary_mod.api_log_consumed_meal(
            payload=payload, background_tasks=BackgroundTasks(), verified_user_id=payload.user_id
        )
        after = datetime.now(timezone.utc)

        assert result["success"] is True
        assert "consumed_at_override" in captured["kwargs"], (
            "el handler dejó de pasar `consumed_at_override` a log_consumed_meal "
            "— el backdate quedaría sin efecto (siempre NOW())."
        )
        override_dt = datetime.fromisoformat(captured["kwargs"]["consumed_at_override"])
        assert override_dt <= after, "el override nunca debe caer en el futuro."
        # days_ago=3 → el override debe estar ~3 días antes de "ahora" (con
        # margen generoso para la latencia del test, nunca acercándose a 0
        # días ni excediendo el tope de 7).
        delta = before - override_dt
        assert timedelta(days=2, hours=23) <= delta <= timedelta(days=3, hours=1), (
            f"delta={delta} — se esperaba ~3 días (days_ago=3 clampeado)."
        )

    def test_days_ago_zero_keeps_override_near_now(self, monkeypatch):
        diary_mod = _import_diary_module()
        from fastapi import BackgroundTasks

        captured = {}

        def fake_log(*args, **kwargs):
            captured["kwargs"] = kwargs
            return True

        monkeypatch.setattr(diary_mod, "log_consumed_meal", fake_log)

        payload = diary_mod.ConsumedMealRequest(
            user_id=self._uid(),
            meal_name="Desayuno de hoy",
            meal_type="desayuno",
            days_ago=0,
        )
        before = datetime.now(timezone.utc)
        diary_mod.api_log_consumed_meal(
            payload=payload, background_tasks=BackgroundTasks(), verified_user_id=payload.user_id
        )
        after = datetime.now(timezone.utc)
        override_dt = datetime.fromisoformat(captured["kwargs"]["consumed_at_override"])
        assert before - timedelta(seconds=5) <= override_dt <= after + timedelta(seconds=5), (
            "days_ago=0 debe producir un override prácticamente igual a 'ahora'."
        )

    @staticmethod
    def _uid() -> str:
        return "33333333-3333-3333-3333-333333333333"

    def test_pydantic_rejects_out_of_bounds_days_ago(self):
        """El límite HTTP (Field ge/le) rechaza estructuralmente valores
        fuera de [0, 7] con 422 — capa distinta y anterior al clamp interno,
        pero parte del mismo contrato 'nunca más de 7 días atrás'."""
        diary_mod = _import_diary_module()
        from pydantic import ValidationError

        for bad in (8, -1, 100):
            with pytest.raises(ValidationError):
                diary_mod.ConsumedMealRequest(
                    user_id=self._uid(), meal_name="x", meal_type="snack", days_ago=bad
                )


# ===========================================================================
# 7. tools.py + prompts — el chat-agent y el endpoint acuerdan el mismo tope
#    (7 días), no dos contratos divergentes.
# ===========================================================================
class TestChatToolMaxDaysAlignedTo7:
    def test_tools_constant_is_7(self, tools_src: str):
        assert re.search(r"_CONSUMED_MAX_DAYS_AGO\s*=\s*7\b", tools_src), (
            "`tools._CONSUMED_MAX_DAYS_AGO` debe ser 7 — alineado con "
            "`ConsumedMealRequest.days_ago` (routers/diary.py)."
        )
        assert "_CONSUMED_MAX_DAYS_AGO = 3" not in tools_src, (
            "regresión: el tope viejo (3) reapareció — chat y endpoint "
            "volverían a divergir."
        )

    def test_tool_docstring_mentions_7(self, tools_src: str):
        i = tools_src.find("- days_ago: 0 = hoy")
        assert i != -1, "docstring de la tool log_consumed_meal no encontrado."
        window = tools_src[i:i + 300]
        assert "Máximo 7 días atrás" in window

    def test_prompts_bullets_say_max_7(self, chat_agent_prompts_src: str):
        assert chat_agent_prompts_src.count("máx 7") >= 2, (
            "ambos builders (build_tools_instructions inline + stream) deben "
            "decir 'máx 7' — antes decían 'máx 3', divergente del endpoint."
        )
        assert "máx 3" not in chat_agent_prompts_src, (
            "regresión: quedó un bullet con el tope viejo (3)."
        )


# ===========================================================================
# 8. CLAUDE.md — la exención de quota del nuevo endpoint queda documentada
#    EN LAS DOS COPIAS: `backend/CLAUDE.md` (única con git — la que Claude
#    Code auto-carga con cwd=backend/, y la que de verdad viaja en el commit)
#    y la copia espejo de workspace-root (fuera de cualquier repo git, pero
#    mantenida idéntica por convención). El bug real que motivó este test:
#    la primera pasada de P1-DIARY-EDITABLE solo editó la copia de
#    workspace-root — `routers/diary.py:684` quedó apuntando a una fila que
#    NO existía en `backend/CLAUDE.md`, el archivo que de verdad se commitea.
#    Contrato anclado también por test_p1_audit_3_history_quota_exemption.py
#    (que solo cubre las 3 filas originales del Historial).
# ===========================================================================
@pytest.mark.parametrize(
    "claude_md_path",
    [_BACKEND / "CLAUDE.md", _BACKEND.parent / "CLAUDE.md"],
    ids=["backend/CLAUDE.md", "workspace-root/CLAUDE.md"],
)
def test_claude_md_documents_delete_endpoint_exemption(claude_md_path: Path):
    assert claude_md_path.exists(), f"{claude_md_path} no existe."
    claude_md = claude_md_path.read_text(encoding="utf-8")
    assert "Historial-quota-exemption" in claude_md
    assert "/api/diary/consumed/{meal_id}" in claude_md, (
        f"{claude_md_path} no documenta la exención de quota del DELETE — "
        f"sin esto, un futuro mantenedor 've' verify_api_quota faltante y "
        f"lo 'arregla' rompiendo la corrección de errores de logging."
    )
    assert "P1-DIARY-EDITABLE" in claude_md
    assert "_DELETE_CONSUMED_LIMITER" in claude_md
    # No basta con que la fila EXISTA — debe explicar POR QUÉ el paywall
    # sería absurdo aquí, mismo estándar que /restock y /inventory/consume:
    # bloquearía corregir un error propio Y `get_monthly_api_usage` cuenta
    # TODA fila de `api_usage` sin filtrar endpoint (quemaría crédito de
    # planes por borrar una fila).
    assert "get_monthly_api_usage" in claude_md


def test_claude_md_copies_stay_in_sync():
    """Convención del repo: la copia de workspace-root y `backend/CLAUDE.md`
    se mantienen IDÉNTICAS (solo `backend/CLAUDE.md` vive en git; la de
    workspace-root es un espejo de trabajo fuera de cualquier repo). Si
    divergen, alguien editó una sola copia — exactamente el bug que este
    test previene yendo hacia adelante."""
    root_text = (_BACKEND.parent / "CLAUDE.md").read_text(encoding="utf-8")
    backend_text = (_BACKEND / "CLAUDE.md").read_text(encoding="utf-8")
    assert root_text == backend_text, (
        "Las dos copias de CLAUDE.md divergieron. Por convención del repo "
        "se mantienen idénticas — sincronizar ambas en el mismo commit "
        "(la de backend/ es la única que efectivamente se commitea)."
    )
