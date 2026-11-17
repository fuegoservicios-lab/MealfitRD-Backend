"""[P1-PLAN-LOTE-1004 · 2026-11-14] (G-04) Evaluador determinista y ledger para la batería del coach.

Provee:
- ``SHORT_PRESET_IDS``: subconjunto corto de 19 casos que cubre registrar, corregir,
  fusionar multi-foto, foto (doble registro, consulta sin registro, dudas, etiqueta),
  clínico/alergias, nevera, plan, idioma y memoria.
- ``evaluar_caso``: veredicto determinista por caso (tools llamadas, args reales,
  fugas internas prohibidas, palabras clave obligatorias/prohibidas, contaminación).
- ``ledger.jsonl`` helpers: registro histórico por fecha, modelo, preset, coste y
  comparación automática ``batería N/M, antes N'/M``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHORT_PRESET_IDS: tuple[str, ...] = (
    "A1",
    "B1",
    "B2",
    "B6",
    "B7",
    "M1",
    "M2",
    "C1",
    "C6",
    "D2",
    "F1",
    "F2",
    "F7",
    "I2",
    "J5",
    "J6",
    "P2",
    "R3",
    "R5",
)

# [P1-PLAN-LOTE-1059 · plan 1031] P-1..P-4: el coach y el progreso por semanas (bloque 🏆, consultar_progreso, nota de
# estrella tras registrar, y que NO hable de estrellas en un turno sin relación).
PROGRESO_PRESET_IDS: tuple[str, ...] = ("W1", "W2", "W3", "W4")

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent / "ledger.jsonl"

LEAK_SUBSTRINGS_DEFAULT: tuple[str, ...] = (
    "[id_registro_diario",
    "id_registro_diario:",
    "traceback (most recent",
    "psycopg.",
    "sqlstate",
    "<function ",
)

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


def _norm_val(val: Any) -> Any:
    if isinstance(val, str):
        return val.strip().lower()
    return val


def _combined_text_from_fields(args: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for f in fields:
        v = args.get(f)
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            parts.append(json.dumps(v, ensure_ascii=False))
        else:
            parts.append(str(v))
    return " ".join(parts).lower()


def _match_args_predicates(args: dict[str, Any], chk: dict[str, Any]) -> str | None:
    """Retorna None si el dict `args` cumple todos los predicados, o razón legible."""
    for field, expected in (chk.get("equals") or {}).items():
        actual = args.get(field)
        if _norm_val(actual) != _norm_val(expected):
            return f"{field}={actual!r}!=esperado:{expected!r}"

    for field, allowed in (chk.get("in_values") or {}).items():
        actual = args.get(field)
        norm_allowed = {_norm_val(a) for a in allowed}
        if _norm_val(actual) not in norm_allowed:
            return f"{field}={actual!r} not in {allowed!r}"

    for field, bounds in (chk.get("numeric_range") or {}).items():
        raw = args.get(field)
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return f"{field}={raw!r} no_numerico"
        lo, hi = float(bounds[0]), float(bounds[1])
        if not (lo <= num <= hi):
            return f"{field}={num} fuera [{lo},{hi}]"

    for rule in chk.get("text_all") or []:
        fields = rule.get("fields") or []
        any_of = [str(s).lower() for s in (rule.get("any_of") or [])]
        haystack = _combined_text_from_fields(args, fields)
        if any_of and not any(tok in haystack for tok in any_of):
            return f"falta_texto({fields}): ninguno de {any_of}"

    for rule in chk.get("text_none") or []:
        fields = rule.get("fields") or []
        any_of = [str(s).lower() for s in (rule.get("any_of") or [])]
        haystack = _combined_text_from_fields(args, fields)
        hit = next((tok for tok in any_of if tok in haystack), None)
        if hit is not None:
            return f"texto_prohibido_args({fields}): {hit!r}"

    return None


def evaluar_caso(case: dict[str, Any], turns: list[dict[str, Any]], final: str | None = None) -> dict[str, Any]:
    """Evalúa determinísticamente la corrida de un caso de batería.

    Args:
        case: Entrada del caso en ``battery.json``.
        turns: Lista de dicts de turno (cada uno con ``tools``, ``final``, ``errors``, ``contaminated``).
        final: Respuesta final opcional (si es None se toma ``turns[-1]["final"]``).

    Returns:
        Dict con ``id``, ``passed``, ``failures``, ``tools_called``, ``warnings``.
    """
    exp = case.get("expect") or {}
    spec = exp.get("assert_spec") or {}
    last = turns[-1] if turns else {}
    final_text = final if final is not None else (last.get("final") or "")
    final_lower = final_text.lower()

    tool_calls: list[dict[str, Any]] = []
    errors: list[Any] = []
    contaminated: list[Any] = []
    for tr in turns:
        for tc in tr.get("tools") or []:
            tool_calls.append(tc)
        errors.extend(tr.get("errors") or [])
        contaminated.extend(tr.get("contaminated") or [])

    tools_called = [str(tc.get("tool") or "") for tc in tool_calls]
    failures: list[str] = []
    warnings: list[str] = []

    if errors:
        failures.append(f"turn_errors:{len(errors)}")
    if contaminated:
        failures.append(f"contaminated_writes:{len(contaminated)}")
    if not final_text.strip() and not errors:
        failures.append("empty_final_response")

    for req_tool in exp.get("tools") or []:
        if req_tool not in tools_called:
            failures.append(f"missing_tool:{req_tool}")

    for banned_tool in exp.get("no_tools") or []:
        if banned_tool in tools_called:
            failures.append(f"forbidden_tool:{banned_tool}")

    if exp.get("registros") is not None:
        n_reg = sum(1 for t in tools_called if t == "log_consumed_meal")
        if n_reg != int(exp["registros"]):
            failures.append(f"registros:{n_reg}!={exp['registros']}")

    for chk in spec.get("args_checks") or []:
        tname = chk["tool"]
        matching = [tc for tc in tool_calls if tc.get("tool") == tname]
        min_c = chk.get("min_count")
        exact_c = chk.get("exact_count")
        if exact_c is not None and len(matching) != int(exact_c):
            failures.append(f"args_count:{tname}={len(matching)}!={exact_c}")
            continue
        if min_c is not None and len(matching) < int(min_c):
            failures.append(f"args_min_count:{tname}={len(matching)}<{min_c}")
            continue
        if not matching:
            if any(k in chk for k in ("equals", "in_values", "numeric_range", "text_all", "text_none")):
                failures.append(f"args_missing_call:{tname}")
            continue

        mode = chk.get("match_mode") or "any"
        if mode == "at_index":
            idx = int(chk.get("index", 0))
            if idx < -len(matching) or idx >= len(matching):
                failures.append(f"args_index_oob:{tname}[{idx}]")
            else:
                reason = _match_args_predicates(matching[idx].get("args") or {}, chk)
                if reason:
                    failures.append(f"args:{tname}[{idx}]:{reason}")
        elif mode == "all":
            for i, tc in enumerate(matching):
                reason = _match_args_predicates(tc.get("args") or {}, chk)
                if reason:
                    failures.append(f"args:{tname}[{i}]:{reason}")
        else:
            reasons = [_match_args_predicates(tc.get("args") or {}, chk) for tc in matching]
            if all(r is not None for r in reasons):
                failures.append(f"args:{tname}:{reasons[0]}")

    for leak in LEAK_SUBSTRINGS_DEFAULT:
        if leak in final_lower:
            failures.append(f"internal_leak:{leak}")

    for banned_phrase in spec.get("must_not_contain_any") or []:
        if str(banned_phrase).lower() in final_lower:
            failures.append(f"forbidden_text:{banned_phrase}")

    for group in spec.get("must_contain_any") or []:
        opts = [str(x).lower() for x in (group if isinstance(group, list) else [group])]
        if opts and not any(opt in final_lower for opt in opts):
            failures.append(f"missing_text_any:{opts}")

    words = len(re.findall(r"\w+", final_text))
    lang_exp = exp.get("idioma")
    if lang_exp and words >= 8:
        detected = guess_lang(final_text)
        if detected != lang_exp:
            failures.append(f"idioma:{detected}!={lang_exp}")

    max_w = exp.get("max_palabras")
    if max_w and words > int(max_w):
        msg = f"largo:{words}>{max_w}"
        if spec.get("strict_max_words"):
            failures.append(msg)
        else:
            warnings.append(msg)

    return {
        "id": case.get("id"),
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "tools_called": tools_called,
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(float(v) for v in values)
    idx = min(len(s) - 1, max(0, int(round((pct / 100.0) * (len(s) - 1)))))
    return round(s[idx], 2)


def summarize_latency_cache(results: list[dict[str, Any]], por_modelo: dict[str, dict[str, Any]]) -> dict[str, Any]:
    turn_secs = [
        float(tr.get("secs") or 0.0)
        for r in results
        for tr in (r.get("turns") or [])
    ]
    total_in = sum(int(slot.get("in") or 0) for slot in (por_modelo or {}).values())
    total_cached = sum(int(slot.get("cached") or 0) for slot in (por_modelo or {}).values())
    hit_pct = round((total_cached * 100.0 / total_in), 2) if total_in > 0 else 0.0
    return {
        "p50_secs": _percentile(turn_secs, 50.0),
        "p90_secs": _percentile(turn_secs, 90.0),
        "input_tokens": total_in,
        "cached_tokens": total_cached,
        "hit_pct": hit_pct,
    }


def load_ledger(path: Path | str = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            entries.append(json.loads(raw))
        except Exception:
            continue
    return entries


def latest_baseline_entry(
    entries: list[dict[str, Any]],
    *,
    preset: str | None = None,
    exclude_label: str | None = None,
) -> dict[str, Any] | None:
    for entry in reversed(entries):
        if exclude_label and entry.get("label") == exclude_label:
            continue
        if preset and entry.get("preset") and entry.get("preset") != preset:
            continue
        return entry
    return None


def build_ledger_entry(
    *,
    label: str,
    preset: str,
    model: str,
    fallback_model: str,
    vision_model: str,
    passed: int,
    total: int,
    cost_usd: float,
    calls: int,
    cache_hit_pct: float,
    p50_secs: float | None,
    p90_secs: float | None,
    failed_ids: list[str],
    git_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "preset": preset,
        "model": model,
        "fallback_model": fallback_model,
        "vision_model": vision_model,
        "passed": int(passed),
        "total": int(total),
        "cost_usd": round(float(cost_usd), 5),
        "calls": int(calls),
        "cache_hit_pct": round(float(cache_hit_pct), 2),
        "p50_secs": p50_secs,
        "p90_secs": p90_secs,
        "failed_ids": list(failed_ids),
        "git_sha": git_sha,
    }


def append_ledger_entry(entry: dict[str, Any], path: Path | str = DEFAULT_LEDGER_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return p


def format_comparison_line(passed: int, total: int, prev_entry: dict[str, Any] | None) -> str:
    if prev_entry is None:
        return f"batería {passed}/{total}, antes sin_previo"
    prev_p = int(prev_entry.get("passed") or 0)
    prev_t = int(prev_entry.get("total") or 0)
    return f"batería {passed}/{total}, antes {prev_p}/{prev_t}"


def regression_detected(
    passed: int,
    total: int,
    prev_entry: dict[str, Any] | None,
    *,
    current_failed_ids: list[str] | None = None,
) -> bool:
    """Devuelve True si hubo regresión respecto al baseline previo."""
    if total <= 0:
        return True
    if prev_entry is None:
        return False
    prev_t = int(prev_entry.get("total") or 0)
    prev_p = int(prev_entry.get("passed") or 0)
    if prev_t > 0 and (passed / total) < (prev_p / prev_t):
        return True
    if prev_t == total and passed < prev_p:
        return True
    if current_failed_ids is not None and "failed_ids" in prev_entry:
        prev_failed = set(prev_entry.get("failed_ids") or [])
        newly_failed = set(current_failed_ids) - prev_failed
        if newly_failed:
            return True
    return False
