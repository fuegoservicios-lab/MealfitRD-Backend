"""Arnés de la batería de escritura del coach — DRY-RUN, base de datos en SOLO LECTURA.

Método y rúbrica: docs/coach_bateria_2026_09_15.md. Uso:

    python scripts/coach_battery/run_battery.py --label antes [--only A1,B2] [--budget-usd 2]

Garantías (en este orden, antes de importar nada del backend):
  1. `psycopg.Cursor.execute` envuelto: solo pasan lecturas; toda escritura lanza
     DryRunBlockedWrite y queda anotada en el resultado (la red de seguridad).
  2. PostgresSaver → un MemorySaver compartido (el hilo de cada caso vive en RAM).
  3. Las tools que escriben devuelven su mensaje de éxito con el formato REAL, sin tocar nada.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

# El .env del árbol principal (un worktree no lo tiene) y la cuenta cuyos datos se LEEN:
#   MEALFIT_BATTERY_ENV=<ruta al .env>  MEALFIT_BATTERY_USER_ID=<uuid>
_ENV = os.environ.get("MEALFIT_BATTERY_ENV", str(BACKEND / ".env"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ENV)
os.environ["MEALFIT_LLM_COST_TRACKING_ENABLED"] = "1"  # el emit llega a nuestro recolector, no a la base


try:  # [P1-PLAN-LOTE-76] la consola cp1252 de Windows mataba la corrida al imprimir «≠» (48 casos perdidos)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ───────────────────────── 1. Guardia de solo lectura ─────────────────────────
import psycopg  # noqa: E402

BLOCKED: list[str] = []
_WRITE_TOKENS = re.compile(
    r"\b(insert|update|delete|merge|upsert|truncate|create|alter|drop|grant|revoke|copy|vacuum|"
    r"nextval|setval|set_config|pg_advisory\w*|increment_\w+|refresh)\b",
    re.IGNORECASE,
)
_READ_FIRST = {"select", "with", "show", "values", "explain"}
_TX_WORDS = {"begin", "commit", "rollback", "savepoint", "release", "start", "end"}


class DryRunBlockedWrite(RuntimeError):
    pass


def _sql_text(query, cur) -> str:
    if isinstance(query, bytes):
        return query.decode("utf-8", "replace")
    if isinstance(query, str):
        return query
    try:
        return query.as_string(cur)
    except Exception:
        return str(query)


def _strip_sql(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"--[^\n]*", " ", s)
    return s.strip()


def sql_is_read_only(sql: str) -> bool:
    s = _strip_sql(sql)
    if not s:
        return True
    first = s.split(None, 1)[0].lower().rstrip(";")
    if first in _TX_WORDS:
        return True
    if first == "set":
        # `SET LOCAL …` y los timeouts de sesión que pone `configure_sync_conn` al abrir cada
        # conexión del pool (no son datos; bloquearlos marcaría turnos como contaminados).
        return bool(re.match(r"set\s+(local\b|statement_timeout\b|idle_in_transaction_session_timeout\b|lock_timeout\b|search_path\b)", s, re.I))
    if first in _READ_FIRST:
        return not _WRITE_TOKENS.search(s)
    return False


def _guard(orig):
    def execute(self, query, *args, **kwargs):
        sql = _sql_text(query, self)
        if not sql_is_read_only(sql):
            BLOCKED.append(re.sub(r"\s+", " ", _strip_sql(sql))[:220])
            raise DryRunBlockedWrite("DRYRUN_BLOCKED_WRITE")
        return orig(self, query, *args, **kwargs)
    return execute


for _cls in (psycopg.Cursor, psycopg.ServerCursor, psycopg.AsyncCursor):
    for _m in ("execute", "executemany"):
        if hasattr(_cls, _m):
            setattr(_cls, _m, _guard(getattr(_cls, _m)))


def _no_copy(self, *a, **k):
    BLOCKED.append("COPY")
    raise DryRunBlockedWrite("DRYRUN_BLOCKED_WRITE")


psycopg.Cursor.copy = _no_copy

# ───────────────────────── 2. Backend + parches ─────────────────────────
import db_core  # noqa: E402

if db_core.connection_pool is not None:
    db_core.connection_pool.open(wait=True, timeout=90)

import agent  # noqa: E402
import db  # noqa: E402
import db_profiles  # noqa: E402
import prompts.chat_agent as chat_prompts  # noqa: E402
import tools  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

SHARED_SAVER = MemorySaver()
agent.PostgresSaver = lambda *_a, **_k: SHARED_SAVER

# Coste: el mismo cálculo que `llm_usage_events`, sin escribirlo.
COST = {"micros": 0, "calls": 0, "sin_precio": 0, "por_modelo": {}}


def _usage_recorder(**kw):
    micros = db_profiles.compute_llm_cost_micros(
        kw.get("model"), kw.get("input_tokens"), kw.get("output_tokens"), kw.get("cached_tokens") or 0,
    )
    COST["calls"] += 1
    m = kw.get("model") or "?"
    slot = COST["por_modelo"].setdefault(m, {"calls": 0, "micros": 0, "in": 0, "out": 0, "cached": 0})
    slot["calls"] += 1
    slot["in"] += int(kw.get("input_tokens") or 0)
    slot["out"] += int(kw.get("output_tokens") or 0)
    slot["cached"] += int(kw.get("cached_tokens") or 0)
    if micros is None:
        COST["sin_precio"] += 1
    else:
        COST["micros"] += micros
        slot["micros"] += micros


db_profiles.log_llm_usage_event = _usage_recorder
db.log_llm_usage_event = _usage_recorder

# Reloj simulado: martes 15-sep-2026, hora local por caso (RD = UTC-4).
SIM = {"utc": datetime(2026, 9, 15, 16, 40, tzinfo=timezone.utc)}


class _SimDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = SIM["utc"]
        return base.astimezone(tz) if tz else base.replace(tzinfo=None)


chat_prompts.datetime = _SimDatetime


def set_hora(hhmm: str) -> None:
    h, m = (int(x) for x in hhmm.split(":"))
    SIM["utc"] = datetime(2026, 9, 15, h, m, tzinfo=timezone.utc) + timedelta(hours=4)
    SIM["hora"] = h + m / 60.0


# [P1-PLAN-LOTE-132] Las tools leen su propio reloj y su propia fecha: `proponer_comida` deduce la franja de la hora,
# así que sin esto el caso «21:05» corría con la hora real de la máquina.
tools._hora_local_float = lambda _uid=None: SIM.get("hora", 12.67)
tools._local_date_str_for_user = lambda _uid=None: "2026-09-15"

# [P1-PLAN-LOTE-132] Diario de HOY simulado por caso (`"diario": [{meal_type, meal_name, calories, protein, carbs,
# healthy_fats}]`): «le falta proteína a las 9 pm» necesita un día a medias, y el diario real del 15-sep es el que es.
# `None` = se lee el de producción, como siempre. Los días anteriores (date_str ≠ hoy) siguen saliendo de producción.
DIARIO_SIM = {"rows": None}
import db_facts as _db_facts  # noqa: E402

_orig_consumed_today = _db_facts.get_consumed_meals_today


def _consumed_today_sim(user_id, date_str=None, tz_offset_mins=None, *a, **kw):
    if DIARIO_SIM["rows"] is not None and (date_str is None or str(date_str)[:10] == "2026-09-15"):
        return copy.deepcopy(DIARIO_SIM["rows"])
    return _orig_consumed_today(user_id, date_str=date_str, tz_offset_mins=tz_offset_mins, *a, **kw)


_db_facts.get_consumed_meals_today = _consumed_today_sim
agent.get_consumed_meals_today = _consumed_today_sim
if hasattr(db, "get_consumed_meals_today"):
    db.get_consumed_meals_today = _consumed_today_sim


# Persona activa (locale y formulario).
PERSONA = {"locale": None}
_orig_get_user_profile = agent.get_user_profile


def _profile_with_locale(uid):
    p = _orig_get_user_profile(uid)
    if isinstance(p, dict) and PERSONA["locale"]:
        p = dict(p)
        p["locale"] = PERSONA["locale"]
    return p


agent.get_user_profile = _profile_with_locale

# ───────────────────────── 3. Tools: registro + dry-run ─────────────────────────
TOOL_LOG: list[dict] = []
FAKE_DIARY: dict[str, dict] = {}
CTX = {"uid": None, "hp": {}}


def _fake_id() -> str:
    return str(uuid.uuid4())


def _stub_log_consumed_meal(**kw):
    calories = kw.get("calories")
    if calories in (None, 0, "0"):
        return ("⚠️ NO REGISTRADO: faltan las calorías de la comida. Estímalas (o pregúntaselas al "
                "usuario) y vuelve a llamar la herramienta.")
    out, err = tools._validar_macros_comida(
        calories=calories, protein=kw.get("protein"), carbs=kw.get("carbs") or 0,
        healthy_fats=kw.get("healthy_fats") or 0,
    )
    if err:
        return (f"⚠️ NO REGISTRADO: {err} Confirma los valores con el usuario (o estímalos de "
                f"nuevo) y vuelve a llamar la herramienta.")
    mt_ok = tools._resolver_meal_type(kw.get("meal_type")) is not None
    meal_type = tools._normalize_meal_type(kw.get("meal_type"))
    days_ago = tools._clamp_days_ago(kw.get("days_ago") or 0)
    rid = _fake_id()
    FAKE_DIARY[rid] = dict(kw, meal_type=meal_type, days_ago=days_ago)
    cuando = "" if days_ago == 0 else (
        " (con fecha de AYER — no cuenta en las macros de hoy)" if days_ago == 1
        else f" (con fecha de hace {days_ago} días — no cuenta en las macros de hoy)")
    msg = (f"¡Éxito! Se ha registrado el consumo de '{kw.get('meal_name')}' ({calories} kcal, "
           f"{kw.get('protein')}g proteína, {kw.get('carbs') or 0}g carbohidratos, "
           f"{kw.get('healthy_fats') or 0}g grasas saludables) como {meal_type}{cuando} en tu diario.")
    # [P1-PLAN-LOTE-76] la misma nota que la tool real («sigue sin registrar…»), contando los registros en seco de ese día
    try:
        _extra = [{"meal_type": v.get("meal_type")} for v in FAKE_DIARY.values() if v.get("days_ago") == days_ago]
        msg += tools._nota_comidas_sin_registrar(kw.get("user_id") or CTX.get("uid"), days_ago, rows_extra=_extra)
    except Exception:
        pass
    if not mt_ok:
        msg += (f" (Aviso para el asistente: no reconocí el tipo de comida '{kw.get('meal_type')}' y "
                f"quedó como snack; si era desayuno, almuerzo o cena, corrígelo con correct_consumed_meal.)")
    msg += (f" [ID_REGISTRO_DIARIO: {rid} — uso interno tuyo, NO se lo menciones ni se lo leas al "
            f"usuario; guárdalo en tu contexto para poder corregir (`correct_consumed_meal`) o borrar "
            f"ESTA fila exacta si el usuario te dice más adelante en esta conversación que quedó mal "
            f"registrada.]")
    return msg


def _stub_correct_consumed_meal(**kw):
    rid = kw.get("meal_id")
    if rid not in FAKE_DIARY:
        return ("ERROR: no encontré esa fila del diario para corregir (puede que el id no exista, no sea "
                "de este usuario, o no hayas pasado ningún campo a cambiar). NO digas que quedó corregido "
                "— dile la verdad al usuario y pregúntale a cuál comida se refiere.")
    bits = []
    for k in ("meal_name", "meal_type", "days_ago", "calories", "ingredients"):
        if kw.get(k) is not None:
            bits.append("ingredientes" if k == "ingredients" else f"{k} → {kw.get(k)}")
            FAKE_DIARY[rid][k] = kw.get(k)
    detalle = ", ".join(bits) if bits else "los campos indicados"
    return (f"¡Corregido! Actualicé el registro existente ({detalle}) — no se creó ninguna fila nueva. "
            f"[ID_REGISTRO_DIARIO: {rid} — uso interno tuyo, NO se lo menciones ni se lo leas al usuario.]")


_RE_TIENE_CANTIDAD = re.compile(
    r"\d|\b(un|una|uno|dos|tres|cuatro|cinco|media|medio|cart[oó]n|paquete|funda|lata|docena|libra|lb|kg)\b",
    re.IGNORECASE)


def _pantry_rows():
    try:
        from db_inventory import get_raw_user_inventory
        return get_raw_user_inventory(CTX["uid"]) or []
    except Exception:
        return []


def _stub_modify_pantry_inventory(**kw):
    from constants import pantry_names_match
    rows = _pantry_rows()
    add = kw.get("items_to_add") or []
    rem = kw.get("items_to_remove") or []
    dep = kw.get("items_to_deplete") or []
    sin_cantidad = [a for a in add if not _RE_TIENE_CANTIDAD.search(str(a))]
    added = len(add) - len(sin_cantidad)
    no_encontrados, removed, depleted = [], 0, 0
    for lst, kind in ((rem, "rem"), (dep, "dep")):
        for it in lst:
            hit = any(pantry_names_match(str(r.get("ingredient_name") or r.get("name") or ""), str(it)) for r in rows)
            if not hit:
                no_encontrados.append(str(it))
            elif kind == "rem":
                removed += 1
            else:
                depleted += 1
    parts = []
    if added:
        parts.append(f"se agregaron {added} ítem(s)")
    if depleted:
        parts.append(f"se marcaron {depleted} como agotado(s)")
    if removed:
        parts.append(f"se eliminaron {removed} ítem(s)")
    msg = ("¡Despensa actualizada! " + ", ".join(parts) + ".") if parts else \
        "No se modificó la despensa (nada coincidió con lo solicitado)."
    if sin_cantidad:
        msg += (f"\n\n⚠️ Aviso para el asistente: NO agregué {', '.join(sin_cantidad)} porque no traía(n) "
                f"cantidad. Pregúntale al usuario cuánto compró (ej. '2 lb', '1 paquete') y vuelve a "
                f"llamar con la cantidad en el texto. NO digas que se agregó.")
    if no_encontrados:
        msg += (f"\n\nℹ️ Aviso para el asistente: NO encontré en la Nevera: {', '.join(no_encontrados)}. "
                f"No se borró ni se marcó nada por esos. NO digas que lo hiciste; pregúntale cómo "
                f"aparece en su Nevera (check_current_pantry) si hace falta.")
    return msg


def _stub_mark_shopping_list_purchased(**kw):
    n = 0
    try:
        from shopping_calculator import get_shopping_list_delta
        rec = db.get_latest_usable_meal_plan_with_id(CTX["uid"])
        n = len(get_shopping_list_delta(CTX["uid"], (rec or {}).get("plan_data") or {}, is_new_plan=False) or [])
    except Exception:
        n = 0
    if not n:
        return "La lista de compras Delta (ingredientes faltantes) está vacía, no hay nada nuevo que añadir a la despensa."
    excl = kw.get("excluded_items") or []
    msg = f"¡Felicidades! Se han agregado los {max(0, n - len(excl))} ingredientes a tu Nevera Virtual."
    if excl:
        msg += f" (Se excluyeron {len(excl)} ítems que indicaste)."
    return msg


def _stub_log_water_glass(**kw):
    delta = kw.get("count_delta", 1)
    base = 0.0
    try:
        row = db.execute_sql_query(
            "SELECT glasses FROM water_intake_log WHERE user_id = %s AND log_date = %s",
            (CTX["uid"], "2026-09-15"), fetch_one=True)
        base = float((row or {}).get("glasses") or 0)
    except Exception:
        pass
    new = max(0.0, min(50.0, base + float(delta)))
    verb = "sumaron" if float(delta) > 0 else "restaron"
    reached = " ¡Cumplio su meta del dia!" if new >= 8 else ""
    return f"Listo: se {verb} {abs(float(delta)):g} vaso(s). El usuario ahora lleva {new:g} de 8 vasos hoy.{reached}"


def _stub_update_form_field(**kw):
    field, new_value = kw.get("field"), kw.get("new_value", "")
    ok, val = tools._valor_de_campo_para_perfil(field, new_value)
    if not ok:
        return (f"No pude actualizar '{field}': no es un campo que yo pueda editar, o el valor "
                f"'{new_value}' no es válido para ese campo. Dile al usuario que lo cambie desde Configuración.")
    ok2, final, _unidad = tools._valor_canonico_del_formulario(field, val)
    if not ok2:
        return final
    mostrar = final
    if field in ("allergies", "medicalConditions", "dislikes", "struggles"):
        prev = CTX["hp"].get(field) or []
        prev = prev if isinstance(prev, list) else [prev]
        nuevos = [x.strip() for x in str(final).split(",") if x.strip()] if isinstance(final, str) else list(final or [])
        vistos, union = set(), []
        for x in [p for p in prev if str(p).strip() and str(p).strip().lower() not in ("nada", "ninguna", "ninguno")] + nuevos:
            if str(x).lower() not in vistos:
                vistos.add(str(x).lower())
                union.append(str(x))
        mostrar = ", ".join(union)
    return f"¡Éxito! El campo '{field}' ha sido actualizado a '{mostrar}'."


_STUBS = {
    "log_consumed_meal": _stub_log_consumed_meal,
    "correct_consumed_meal": _stub_correct_consumed_meal,
    "modify_pantry_inventory": _stub_modify_pantry_inventory,
    "mark_shopping_list_purchased": _stub_mark_shopping_list_purchased,
    "log_water_glass": _stub_log_water_glass,
    "update_form_field": _stub_update_form_field,
}


def _recording(name, fn):
    def wrapped(*a, **kw):
        args = {k: v for k, v in kw.items() if k not in ("user_id", "callbacks", "run_manager")}
        try:
            res = fn(*a, **kw)
        except Exception as e:  # la tool real lanzó: se registra y se re-lanza (execute_tools lo convierte)
            TOOL_LOG.append({"tool": name, "args": args, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            raise
        TOOL_LOG.append({"tool": name, "args": args, "result": str(res)[:1500], "dry_run": name in _STUBS})
        return res
    return wrapped


_all_tools = {t.name: t for t in list(tools.agent_tools) + [tools.update_form_field]}
for _name, _t in _all_tools.items():
    _impl = _STUBS.get(_name) or _t.func
    object.__setattr__(_t, "func", _recording(_name, _impl))


def _plan_dry_run(*a, **k):
    TOOL_LOG.append({"tool": "plan_mutation", "args": {k2: v for k2, v in k.items() if k2 != "form_data"},
                     "result": "dry-run"})
    return "ERROR: (batería) esta tool de plan no se ejecuta en dry-run."


agent.execute_modify_single_meal = _plan_dry_run
agent.execute_generate_new_plan = _plan_dry_run

# System prompt de cada turno (para el juez).
LAST_SYS = {"prompt": ""}
_orig_llm_msgs = agent._llm_messages_from_state


def _capture_sys(messages, sys_prompt):
    LAST_SYS["prompt"] = sys_prompt
    return _orig_llm_msgs(messages, sys_prompt)


agent._llm_messages_from_state = _capture_sys

# ───────────────────────── 4. Métricas automáticas ─────────────────────────
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
_STOP = {
    "es": {"el", "la", "de", "que", "y", "en", "los", "tu", "te", "hoy", "con", "para", "es", "por", "una", "lo", "del", "más", "tienes"},
    "en": {"the", "and", "you", "your", "is", "for", "to", "of", "with", "today", "have", "it", "that", "what", "dinner"},
    "pt": {"você", "o", "a", "de", "que", "e", "com", "para", "hoje", "seu", "sua", "no", "na", "do", "da", "não", "é"},
    "fr": {"le", "la", "les", "de", "et", "tu", "ton", "ta", "pour", "avec", "aujourd'hui", "est", "des", "un", "une", "vous"},
    "it": {"il", "la", "di", "che", "e", "per", "con", "oggi", "hai", "tuo", "tua", "in", "del", "della", "un", "una", "è"},
}


def guess_lang(text: str) -> str:
    words = re.findall(r"[a-zA-ZÀ-ÿ']+", (text or "").lower())
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOP.items()}
    return max(scores, key=scores.get) if any(scores.values()) else "?"


def auto_metrics(case: dict, final: str, tools_called: list[str]) -> dict:
    exp = case.get("expect") or {}
    words = len(re.findall(r"\w+", final or ""))
    flags = []
    for t in exp.get("tools") or []:
        if t not in tools_called:
            flags.append(f"falta_tool:{t}")
    for t in exp.get("no_tools") or []:
        if t in tools_called:
            flags.append(f"tool_prohibida:{t}")
    lang = guess_lang(final)
    if exp.get("idioma") and lang != exp["idioma"]:
        flags.append(f"idioma:{lang}≠{exp['idioma']}")
    tope = exp.get("max_palabras")
    if tope and words > tope:
        flags.append(f"largo:{words}>{tope}")
    return {"palabras": words, "emojis": len(_EMOJI.findall(final or "")), "idioma": lang, "flags": flags}


# ───────────────────────── 5. Ejecución ─────────────────────────
OWNER = os.environ.get("MEALFIT_BATTERY_USER_ID", "")
# + la autolimpieza de `sintoma_temporal` caducados que `search_user_facts` hace antes de leer
# (P2-PROD-AUDIT-BUNDLE): verificado que el dueño tiene 0 ⇒ bloquearla no cambia la lectura.
_TELEMETRY_RE = re.compile(
    r"^(INSERT INTO (pipeline_metrics|app_kv_store|system_alerts|llm_usage_events)\b"
    r"|DELETE FROM user_facts WHERE metadata @> %s AND created_at < %s)", re.I)


def load_persona(name: str) -> dict:
    prof = _orig_get_user_profile(OWNER) or {}
    hp = copy.deepcopy(prof.get("health_profile") or {})
    plan = db.get_latest_meal_plan(OWNER)
    locale = {"en": "en-US", "pt": "pt-BR", "fr": "fr-FR", "it": "it-IT"}.get(name)
    if name == "alergico":
        hp["allergies"] = ["Maní", "Mariscos"]
        hp["medicalConditions"] = ["Diabetes tipo 2"]
    return {"uid": OWNER, "form_data": hp, "plan": plan, "tier": prof.get("plan_tier") or "gratis", "locale": locale}


def run_turn(sid: str, persona: dict, prompt: str, vision) -> dict:
    TOOL_LOG.clear()
    b0, c0 = len(BLOCKED), COST["micros"]
    events, t0 = [], time.monotonic()
    err = None
    try:
        for raw in agent.chat_with_agent_stream(
            session_id=sid, prompt=prompt, current_plan=copy.deepcopy(persona["plan"]),
            user_id=persona["uid"], form_data=copy.deepcopy(persona["form_data"]),
            local_date="2026-09-15", tz_offset=240, plan_tier=persona["tier"], vision=vision,
        ):
            for line in str(raw).splitlines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except Exception:
                        pass
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
    done = next((e for e in events if e.get("type") == "done"), {}) or {}
    blocked = BLOCKED[b0:]
    return {
        # Telemetría que el chat escribe en producción en cada turno (reset del breaker, duración,
        # alertas): esperada. Cualquier OTRA escritura bloqueada = turno contaminado, a investigar.
        "contaminated": [b for b in blocked if not _TELEMETRY_RE.match(b)],
        "prompt": prompt,
        "vision": vision,
        "shown": "".join(e.get("text", "") for e in events if e.get("type") == "chunk"),
        "final": done.get("response", ""),
        "errors": [e for e in events if e.get("type") == "error"] + ([{"exception": err}] if err else []),
        "tools": copy.deepcopy(TOOL_LOG),
        "updated_fields": done.get("updated_fields") or {},
        "blocked_writes": blocked,
        "cost_usd": round((COST["micros"] - c0) / 1e6, 5),
        "secs": round(time.monotonic() - t0, 1),
        "sys_prompt_chars": len(LAST_SYS["prompt"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--budget-usd", type=float, default=2.0)
    args = ap.parse_args()
    if not OWNER:
        _out("Falta MEALFIT_BATTERY_USER_ID (la cuenta cuyos plan/Nevera/diario se leen).")
        return 2

    battery = json.loads((HERE / "battery.json").read_text(encoding="utf-8"))
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    cases = [c for c in battery["casos"] if not only or c["id"] in only]
    out_dir = HERE / "out" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    personas: dict[str, dict] = {}
    results, sys_saved = [], set()
    for case in cases:
        if COST["micros"] / 1e6 > args.budget_usd:
            _out(f"⛔ presupuesto US${args.budget_usd} superado — corte antes de {case['id']}")
            break
        pname = case["persona"]
        if pname not in personas:
            personas[pname] = load_persona(pname)
        persona = personas[pname]
        PERSONA["locale"] = persona["locale"]
        CTX["uid"], CTX["hp"] = persona["uid"], persona["form_data"]
        FAKE_DIARY.clear()
        set_hora(case.get("hora", "12:40"))
        DIARIO_SIM["rows"] = case.get("diario")   # [P1-PLAN-LOTE-132] None = el diario real de producción
        sid = str(uuid.uuid4())  # la columna es uuid; la sesión no existe en la base (lectura vacía)
        turns = []
        for i, prompt in enumerate(case["turns"]):
            vision = case.get("vision") if i == len(case["turns"]) - 1 else None
            turns.append(run_turn(sid, persona, prompt, vision))
            if pname not in sys_saved:
                (out_dir / f"sysprompt_{pname}.txt").write_text(LAST_SYS["prompt"], encoding="utf-8")
                sys_saved.add(pname)
        last = turns[-1]
        called = [t["tool"] for tr in turns for t in tr["tools"]]
        metrics = auto_metrics(case, last["final"], called)
        results.append({"id": case["id"], "cat": case["cat"], "persona": pname, "hora": case.get("hora"),
                        "expect": case.get("expect"), "turns": turns, "auto": metrics})
        _out(f"{case['id']:>3} {sum(t['secs'] for t in turns):5.1f}s ${sum(t['cost_usd'] for t in turns):.4f} "
             f"tools={called} flags={metrics['flags']} contaminado={sum(len(t['contaminated']) for t in turns)}")

    summary = {"label": args.label, "casos": len(results), "coste_usd": round(COST["micros"] / 1e6, 4),
               "llamadas_llm": COST["calls"], "sin_precio": COST["sin_precio"], "por_modelo": COST["por_modelo"],
               "escrituras_bloqueadas": len(BLOCKED), "bloqueadas_detalle": BLOCKED[:40], "sim": "2026-09-15 (martes), RD UTC-4"}
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    md = [f"# Transcripción — {args.label}", "", "```", json.dumps(summary, ensure_ascii=False, indent=1), "```", ""]
    for r in results:
        md.append(f"## {r['id']} · {r['cat']} · {r['persona']} · {r['hora']}")
        md.append(f"_Esperado:_ {json.dumps(r['expect'], ensure_ascii=False)}")
        for t in r["turns"]:
            md.append(f"\n**Usuario:** {t['prompt']}" + (f"  _(foto: {t['vision']['kind']})_" if t.get("vision") else ""))
            for tc in t["tools"]:
                md.append(f"- 🔧 `{tc['tool']}` {json.dumps(tc.get('args'), ensure_ascii=False)[:300]} → "
                          f"{(tc.get('result') or tc.get('error') or '')[:260]!r}")
            if t["errors"]:
                md.append(f"- ❗ errores: {json.dumps(t['errors'], ensure_ascii=False)[:300]}")
            if t["contaminated"]:
                md.append(f"- 🛑 CONTAMINADO (escritura no-telemetría bloqueada): {t['contaminated']}")
            md.append(f"\n**Coach (final):**\n\n> " + (t["final"] or "(vacío)").replace("\n", "\n> "))
            if t["shown"].strip() != (t["final"] or "").strip():
                md.append(f"\n_(lo que vio en streaming difiere del final)_:\n\n> " + t["shown"].replace("\n", "\n> "))
        md.append(f"\n_Auto:_ {json.dumps(r['auto'], ensure_ascii=False)}\n")
    (out_dir / "transcript.md").write_text("\n".join(md), encoding="utf-8")
    _out(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

