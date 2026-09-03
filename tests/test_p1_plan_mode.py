"""[P1-PLAN-MODE · 2026-08-11] El interruptor de generación de planes (Fase 2A).

Hay usuarios que quieren la app como contador de macros sin que la IA les genere
planes. Este archivo ancla las decisiones que hacen que APAGAR apague de verdad:

  1. EL GATE DEL PICKUP en LAS DOS ramas. El pickup no hace JOIN con meal_plans: lo
     único que decide si un chunk gasta LLM es su propia fila. El precedente
     `_frozen_at` demuestra que un flag que el pickup no lee no apaga nada. Si
     alguien reescribe una CTE y se lleva el token, el apagado deja de apagar EN ESA
     RAMA y nadie lo nota — por eso este test cuenta tokens, no confía.
  2. `cancelled` y NO `pending_user_action`: el recovery resucita los pausados a las
     12 h (medido en prod — el congelado «apagado» volvía a gastar solo). Y
     `pending_user_action` SÍ está entre los cancelables: dejarlo fuera es
     P1-CHUNK-REBASE-PAUSED con otra ropa.
  3. El ORDEN: bandera primero. Si el proceso muere a mitad, la cola queda viva pero
     INERTE. Al revés, el bg-refill reencola en ≤4 h y el plan resucita solo.
  4. Reanudar restaura el SNAPSHOT, nunca un 'complete' literal (CHECK I8).
  5. El bg-refill filtra por plan_mode con LEFT JOIN (un usuario sin perfil no puede
     desaparecer del cron que garantiza la promesa temporal).
  6. Generar un plan ES el consentimiento: `_postprocess_pipeline_result` re-enciende.
  7. El chat cobra de SU medidor (P1-COACH-METER), no del de planes — y la lista va
     EN NEGATIVO para que un endpoint nuevo quede caro por defecto, no gratis.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_USERDATA = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
_CHAT = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
_AUTH = (_BACKEND / "auth.py").read_text(encoding="utf-8")
_DBPROF = (_BACKEND / "db_profiles.py").read_text(encoding="utf-8")

import plan_mode as pm


# ─────────────────────────── 1. El gate del pickup ───────────────────────────

def test_el_gate_esta_en_LAS_DOS_ramas_del_pickup():
    """El que importa. El token debe aparecer DOS veces dentro de las queries del
    pickup (rama target_plan_id y rama general) y el replace debe estar cableado."""
    assert _CRON.count("__PLAN_MODE_GATE__") >= 2, (
        "el token del gate no está en las dos ramas del pickup: reescribieron una CTE "
        "y el apagado dejó de apagar en esa rama"
    )
    assert 'query = query.replace("__PLAN_MODE_GATE__"' in _CRON, (
        "el token existe pero nadie lo sustituye: el SQL llegaría a Postgres con el "
        "token literal y el pickup entero fallaría"
    )
    assert "P1-PLAN-MODE-PICKUP-GATE" in pm.PICKUP_GATE_SQL
    assert "plan_mode = 'tracking'" in pm.PICKUP_GATE_SQL
    assert "NOT EXISTS" in pm.PICKUP_GATE_SQL


def test_el_gate_es_constante_sin_entrada_de_usuario():
    """El fragmento se concatena a un SQL: tiene que ser una CONSTANTE de módulo.
    Cualquier formateo con datos del usuario aquí sería inyección directa."""
    assert "%s" not in pm.PICKUP_GATE_SQL
    assert "{" not in pm.PICKUP_GATE_SQL


def test_bg_refill_filtra_por_plan_mode_con_LEFT_join():
    i = _CRON.index("def trigger_background_rolling_refill")
    cuerpo = _CRON[i:i + 4000]
    assert "LEFT JOIN user_profiles" in cuerpo, (
        "el bg-refill perdió su filtro de plan_mode: va a ir a buscar cada 4 h "
        "exactamente al usuario que apagó y se fue"
    )
    assert "COALESCE(up.plan_mode, 'plan') <> 'tracking'" in cuerpo, (
        "el filtro no trata «sin perfil» como 'plan': un JOIN interno haría "
        "desaparecer del cron a usuarios sin fila de perfil"
    )


# ─────────────────────────── 2. Estados y orden ───────────────────────────

def test_cancela_los_cinco_estados_y_ninguno_mas():
    """`pending_user_action` DENTRO (el recovery lo resucita si no) y `completed` /
    dead-lettered FUERA (días entregados y forense)."""
    assert set(pm.CANCELLABLE_STATES) == {
        "pending", "processing", "stale", "pending_user_action", "failed",
    }
    src = (_BACKEND / "plan_mode.py").read_text(encoding="utf-8")
    m = re.search(r"SET status = 'cancelled'[\s\S]{0,400}?dead_lettered_at IS NULL", src)
    assert m, "el UPDATE de cancelación perdió el guard de dead_lettered"
    assert "'completed'" not in m.group(0), "la pausa no puede tocar chunks completed"


def test_el_estado_de_pausa_NO_es_pending_user_action():
    """Regresión del freeze: si alguien «simplifica» la pausa al estado del
    congelado, `_recover_pantry_paused_chunks` la resucita a las 12 h y el plan
    vuelve a gastar solo. Está medido en producción, no es teoría."""
    src = (_BACKEND / "plan_mode.py").read_text(encoding="utf-8")
    assert "paused_by_user" in src
    i = src.index("def pause_plan_generation")
    cuerpo = src[i:src.index("def resume_plan_generation")]
    assert "SET status = 'pending_user_action'" not in cuerpo


def test_la_bandera_va_PRIMERO_en_pausar_y_en_reanudar():
    """El orden es la mitad del diseño: morir a mitad de la pausa debe dejar la cola
    inerte (gate ya puesto), nunca un plan que resucita solo."""
    src = (_BACKEND / "plan_mode.py").read_text(encoding="utf-8")
    pausa = src[src.index("def pause_plan_generation"):src.index("def resume_plan_generation")]
    assert pausa.index("plan_mode = 'tracking'") < pausa.index("SET status = 'cancelled'"), (
        "la pausa cancela ANTES de poner la bandera: si muere a mitad, el bg-refill "
        "reencola y el plan resucita"
    )
    reanuda = src[src.index("def resume_plan_generation"):src.index("def ensure_plan_generation_enabled")]
    assert reanuda.index("plan_mode = 'plan'") < reanuda.index("paused_by_user"), (
        "reanudar restaura el plan antes de levantar el gate: los chunks que se "
        "encolen quedan invisibles para el pickup"
    )


def test_reanudar_restaura_el_snapshot_con_guard_I8():
    src = (_BACKEND / "plan_mode.py").read_text(encoding="utf-8")
    reanuda = src[src.index("def resume_plan_generation"):src.index("def ensure_plan_generation_enabled")]
    assert "_paused_prev_generation_status" in reanuda, "reanudar dejó de leer el snapshot"
    assert "jsonb_array_length" in reanuda and "'partial'" in reanuda, (
        "sin el guard I8, reanudar a 'complete' un plan con days=[] viola el CHECK "
        "meal_plans_complete_requires_days"
    )


def test_pausar_libera_los_locks():
    # chunk_user_locks no tiene FK a meal_plans: nadie los limpia por CASCADE.
    src = (_BACKEND / "plan_mode.py").read_text(encoding="utf-8")
    pausa = src[src.index("def pause_plan_generation"):src.index("def resume_plan_generation")]
    assert "DELETE FROM chunk_user_locks WHERE user_id" in pausa


# ─────────────────────────── 3. Los otros tres frenos ───────────────────────────

def test_shift_plan_hace_soft_fail_en_tracking():
    """El Dashboard llama al shift AL MONTAR y el usuario en pausa entra todos los
    días: sin este guard, él mismo reencolaría su plan cada mañana."""
    i = _PLANS.index("def api_shift_plan")
    cuerpo = _PLANS[i:i + 4000]
    assert "plan_generation_paused" in cuerpo, "el shift perdió su guard de modo"
    assert "operation_skipped" in cuerpo, (
        "el guard no es soft-fail: un 4xx pinta rojo en la consola del usuario "
        "cada mañana sin que nada esté roto (P3-SWAP-SOFT-FAIL-200)"
    )


def test_generar_un_plan_ES_el_consentimiento():
    """Sin el re-encendido en `_postprocess_pipeline_result`, un usuario en pausa que
    pulsa «Generar plan» paga su crédito, recibe la semana 1 por SSE y las semanas
    2..N quedan bloqueadas por nuestro propio gate. En silencio."""
    i = _PLANS.index("def _postprocess_pipeline_result")
    cuerpo = _PLANS[i:i + 6000]
    assert "ensure_plan_generation_enabled" in cuerpo, (
        "el punto único por el que pasan /analyze y /analyze/stream dejó de "
        "re-encender el modo al generar"
    )


def test_la_alerta_de_cola_viva_existe_y_se_autoresuelve():
    assert "plan_paused_with_live_queue" in _CRON
    i = _CRON.index("plan_paused_with_live_queue")
    ventana = _CRON[i - 2000:i + 3500]
    assert "resolved_at = NOW()" in ventana, (
        "la alerta no se auto-resuelve: quedaría abierta para siempre tras "
        "arreglarse la condición (modelo Auto implicit)"
    )
    docs = (_BACKEND / "docs" / "system_alerts_resolution_table.md").read_text(encoding="utf-8")
    assert "plan_paused_with_live_queue" in docs, (
        "falta la fila en la tabla de resolución: test_p2_audit_4 fallará"
    )


# ─────────────────────────── 4. Endpoints y medidor coach ───────────────────────────

def test_el_interruptor_no_pasa_por_el_paywall():
    """Aplicar el paywall al botón de APAGAR el gasto es exactamente al revés: un
    usuario topado en 402 que no puede apagar deja al worker gastando."""
    i = _USERDATA.index("def api_put_plan_mode")
    cuerpo = _USERDATA[i - 600:i + 800]
    assert "verify_api_quota" not in cuerpo
    assert "_PLAN_MODE_LIMITER" in cuerpo, "sin RateLimiter: spam de toggles sin freno"


def test_targets_es_fail_closed():
    i = _USERDATA.index("def api_nutrition_targets")
    cuerpo = _USERDATA[i:i + 3500]
    assert "missing_fields" in cuerpo, (
        "la puerta de metas perdió missing_fields: la tarjeta vuelve a pintar "
        "2000/150/200/60 como si fueran metas personales"
    )
    assert '"ok": False' in cuerpo


def test_el_chat_cobra_de_SU_medidor():
    n = _CHAT.count("Depends(verify_coach_quota)")
    assert n == 2, f"el chat tiene {n} endpoints con el medidor coach; deben ser 2"
    assert "Depends(verify_api_quota)" not in _CHAT, (
        "queda un endpoint de chat cobrando del pozo de planes: 10 mensajes/mes "
        "en el tier gratis"
    )


def test_el_medidor_de_generacion_va_en_NEGATIVO():
    """El modo de fallo declarado de P1-COACH-METER: una lista en positivo deja
    gratis por olvido a cada endpoint nuevo. En negativo queda caro por defecto."""
    i = _DBPROF.index("def get_monthly_api_usage")
    cuerpo = _DBPROF[i:i + 2500]
    assert "<> 'llm_chat'" in cuerpo, "la rama de generación dejó de ser negativa"
    assert "= 'llm_chat'" in cuerpo, "la rama coach dejó de filtrar por endpoint"
    assert 'kind: str = "generation"' in cuerpo, (
        "el default dejó de ser generation: todos los callers viejos cambiarían "
        "de significado en silencio"
    )


def test_verify_coach_quota_por_tier():
    assert "verify_coach_quota" in _AUTH
    for knob in ("MEALFIT_COACH_LIMIT_GRATIS", "MEALFIT_COACH_LIMIT_ULTRA"):
        assert knob in _AUTH, f"falta el knob {knob}"
    i = _AUTH.index("def verify_coach_quota")
    assert 'kind="coach"' in _AUTH[i:i + 1200]


# ─────────────────────────── 5. Funcional (mockeado) ───────────────────────────

def test_pausar_es_idempotente_y_reanudar_restaura(monkeypatch):
    """Se capturan las escrituras en orden y se afirma la SECUENCIA completa, con la
    DB simulada — la transacción real ya está anclada arriba por partes."""
    escrituras = []

    def _fake_write(sql, params=None, returning=False):
        # 300 y no 80: el `paused_by_user` del jsonb_build_object cae pasada la
        # posición 80 y truncar antes lo dejaba invisible para las aserciones.
        escrituras.append(" ".join(sql.split())[:300])
        if "RETURNING id" in sql and "cancelled" in sql:
            return [{"id": "c1"}, {"id": "c2"}]
        return [] if returning else 1

    def _fake_query(sql, params=None, **kw):
        return {"paused_days": 2}

    monkeypatch.setattr(pm, "execute_sql_write", _fake_write)
    monkeypatch.setattr(pm, "execute_sql_query", _fake_query)
    monkeypatch.setattr(pm, "PLAN_MODE_SWITCH_ENABLED", True)

    out = pm.pause_plan_generation("u1")
    assert out["plan_mode"] == "tracking"
    assert out["chunks_cancelled"] == 2
    assert escrituras[0].startswith("UPDATE user_profiles"), "la bandera no fue lo primero"
    assert any("cancelled" in e for e in escrituras)
    assert any("chunk_user_locks" in e for e in escrituras)
    assert any("paused_by_user" in e for e in escrituras)

    escrituras.clear()
    out2 = pm.resume_plan_generation("u1")
    assert out2["plan_mode"] == "plan"
    assert out2["paused_days"] == 2
    assert out2["plan_expired"] is False
    assert escrituras[0].startswith("UPDATE user_profiles"), "reanudar tampoco empieza por la bandera"


def test_con_el_switch_apagado_todo_es_noop(monkeypatch):
    monkeypatch.setattr(pm, "PLAN_MODE_SWITCH_ENABLED", False)
    llamadas = []
    monkeypatch.setattr(pm, "execute_sql_write", lambda *a, **k: llamadas.append(a))
    assert pm.pause_plan_generation("u1").get("skipped") == "switch_off"
    assert pm.resume_plan_generation("u1").get("skipped") == "switch_off"
    assert pm.ensure_plan_generation_enabled("u1") is False
    assert llamadas == [], "con el kill switch apagado no puede haber ni una escritura"
