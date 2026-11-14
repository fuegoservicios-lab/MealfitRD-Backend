# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1002 · 2026-11-14] Completitud en turnos multi-foto (registro y fusión).

Incidente vivo (2026-11-14 06:17 UTC): el usuario tenía su cena previa (Puré de
plátano con pollo, 600 kcal) y envió DOS fotos —Foto 1: 1 pan de agua mediano
(180 kcal) y Foto 2: 1 vaso de jugo de lechoza (110 kcal)— con el texto
"Esto también fue parte de la cena\\nLechoza · Con azúcar". El modelo solo sumó
el jugo de lechoza (+80 kcal -> 680 kcal) mediante `correct_consumed_meal` y
omitió por completo el pan de agua de la Foto 1.

Tres defensas coordinadas ($0 de IA):
1. `construir_recordatorio_multifoto`: recordatorio explícito en
   `build_vision_context` cuando hay >= 2 fotos de comida.
2. `detectar_bloqueo_multifoto_fusion`: guard determinista en `execute_tools`
   que rechaza UNA vez `correct_consumed_meal` en turno con >= 2 fotos si alguna
   foto no descartada quedó fuera del nombre/ingredientes (o del delta calórico
   cuando el plato previo compartía tokens), devolviendo qué foto faltó y sus kcal.
3. `fusion_explicita_exitosa_este_turno`: evita que `route_tools` dispare
   `nudge_photo_to_log` cuando una fusión explícita con `correct_consumed_meal`
   ya ejecutó con éxito e incluyó todas las fotos.

tooltip-anchor: P1-PLAN-LOTE-1002-MULTIPHOTO-COMPLETENESS
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Optional

MARKER_BLOQUEO_MULTIFOTO = "[P1-PLAN-LOTE-1002 / ERROR_FOTOS_OMITIDAS]"

_PALABRAS_NO_ALIMENTO = frozenset((
    "con de del la las el los un uno una unos unas al en sin por para mas y e o a the and with of com avec et le les du "
    "des di il grande grandes pequeno pequena pequenos pequenas mediano mediana medianos medianas big large small "
    "frito frita fritos fritas fried cocido cocida cocidos cocidas hervido hervida hervidos hervidas guisado guisada "
    "guisados guisadas asado asada asados asadas plancha horno casero casera caseros caseras fresco fresca frescos "
    "frescas blanco blanca blancos blancas rodaja trozo lasca lonja unidad unidade porcion porcione racion racione "
    "plato taza vaso cucharada rebanada pieza gramo slice piece desayuno almuerzo cena merienda snack comida estilo tipo aprox "
    "aproximado aproximada estimacion estimado calorias caloria proteina carbohidrato grasa saludable kcal media medio "
    "mitad entero entera relleno rellena dominicano dominicana tipico tipica servido servida suelto suelta sueltos sueltas "
    "parcialmente mordido mordida ligeramente aproximadamente picado picada picados picadas acompanado acompanada "
    "sobre alrededor poco poca resto adicional extra parte"
).split())

# Palabras secundarias comunes que no deben dar por cubierta por sí solas una foto
# cuando la descripción de la foto contiene un sustantivo principal más específico (p. ej. "pan" en "pan de agua").
_MODIFICADORES_SECUNDARIOS = frozenset(("agua", "sal", "azucar", "aceite", "hielo", "limon"))


def _plano(texto: Any) -> str:
    norm = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in norm if not unicodedata.combining(c))


def _singular_es(w: str) -> str:
    if len(w) > 4 and w.endswith("es") and w[-3] in "nrljdz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def tokens_alimento(texto: Any) -> set[str]:
    """Tokens normalizados de alimentos con singularización básica ES (panes->pan, tostones->toston)."""
    out: set[str] = set()
    for w in re.findall(r"[a-z]+", _plano(texto)):
        w_sing = _singular_es(w)
        if len(w_sing) >= 3 and w_sing not in _PALABRAS_NO_ALIMENTO:
            out.add(w_sing)
    return out


def _tokens_primarios_foto(desc: Any) -> set[str]:
    """Tokens distintivos de una foto (excluye conectores/condimentos si hay alimento principal)."""
    toks = tokens_alimento(desc)
    principales = toks - _MODIFICADORES_SECUNDARIOS
    return principales if principales else toks


def _foto_descartada_en_texto(user_text: str, idx: int, p_tokens: set[str]) -> bool:
    """True solo si el usuario descartó explícitamente esa foto o alimento."""
    norm = _plano(user_text)
    if not norm:
        return False
    if re.search(rf"\b(?:sin|no|omite|descarta|quita|excluye)\b[^.!?\n]{{0,24}}\bfoto\s*{idx}\b", norm):
        return True
    if re.search(rf"\bfoto\s*{idx}\b[^.!?\n]{{0,18}}\bno\b", norm):
        return True
    for tok in p_tokens:
        tok_re = re.escape(tok) + r"(?:es|s)?"
        if re.search(
            rf"\b(?:sin|no me comi|no comi|no inclui|descarta|omite|quita|deja fuera|dejar fuera)\s+(?:el|la|los|las|un|una)?\s*{tok_re}\b",
            norm,
        ):
            return True
        if re.search(rf"\b(?:el|la|los|las)\s+{tok_re}\s+no\b", norm):
            return True
    return False


def construir_recordatorio_multifoto(plate_descriptions: list[str]) -> str:
    """Directiva compacta para `build_vision_context` cuando hay >= 2 fotos de comida/plato."""
    if len(plate_descriptions) < 2:
        return ""
    resumen = ", ".join(
        f"Foto {i}: «{str(desc or '').strip()[:70]}»"
        for i, desc in enumerate(plate_descriptions, start=1)
    )
    n = len(plate_descriptions)
    return (
        f" ⚠️ MULTI-FOTO OBLIGATORIO ({n} fotos de comida: {resumen}): "
        f"TODAS LAS {n} FOTOS deben quedar registradas o sumadas en este turno. "
        "Aunque el usuario solo responda dudas de una de las fotos (ej. «Lechoza · Con azúcar») "
        "o diga «esto también fue parte de…», está PROHIBIDO omitir las otras fotos: incluye cada foto "
        "(su alimento, ingredientes y calorías/macros) al crear registros nuevos o al actualizar con `correct_consumed_meal`, "
        "salvo que el usuario descarte explícitamente alguna."
    )


def _extraer_ingredientes_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val if x]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except Exception:
            return [val]
    return []


def _tokens_en_tool_args(args: dict[str, Any]) -> set[str]:
    ings = _extraer_ingredientes_list(args.get("ingredients"))
    return tokens_alimento(args.get("meal_name")) | tokens_alimento(" ".join(ings))


def fotos_no_cubiertas_en_llamadas(
    plate_photos: list[dict[str, Any]],
    diary_calls_args: list[dict[str, Any]],
    *,
    user_text: str = "",
    orig_meal: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Devuelve las fotos de `plate_photos` que no están cubiertas por el conjunto de llamadas al diario."""
    if len(plate_photos) < 2 or not diary_calls_args:
        return []

    submitted_tokens: set[str] = set()
    for call_args in diary_calls_args:
        if isinstance(call_args, dict):
            submitted_tokens |= _tokens_en_tool_args(call_args)

    orig_tokens: set[str] = set()
    orig_cal = 0.0
    if isinstance(orig_meal, dict):
        orig_ings = _extraer_ingredientes_list(orig_meal.get("ingredients"))
        orig_tokens = tokens_alimento(orig_meal.get("meal_name")) | tokens_alimento(" ".join(orig_ings))
        try:
            orig_cal = float(orig_meal.get("calories") or 0)
        except (TypeError, ValueError):
            orig_cal = 0.0

    activas: list[tuple[int, dict[str, Any], set[str]]] = []
    for idx, p in enumerate(plate_photos, start=1):
        if not isinstance(p, dict):
            continue
        p_toks = _tokens_primarios_foto(p.get("desc"))
        if _foto_descartada_en_texto(user_text, idx, p_toks):
            continue
        activas.append((idx, p, p_toks))

    if len(activas) < 2:
        return []

    expected_photos_cal = sum(float(p.get("kcal") or 0) for _, p, _ in activas)
    total_submitted_cal = 0.0
    for call_args in diary_calls_args:
        if isinstance(call_args, dict):
            try:
                total_submitted_cal += float(call_args.get("calories") or 0)
            except (TypeError, ValueError):
                pass
    delta_cal = total_submitted_cal - orig_cal

    omitidas: list[dict[str, Any]] = []
    for idx, p, p_toks in activas:
        if not p_toks:
            continue
        covered = bool(p_toks & submitted_tokens)
        if not covered:
            omitidas.append({"idx": idx, **p})
            continue
        # Si orig ya compartía todos los tokens primarios de esta foto (p. ej. "pan" añadido sobre cena con pan),
        # exigir coherencia mínima de delta calórico para no tragar omisiones mudas.
        if (
            orig_meal is not None
            and p_toks.issubset(orig_tokens)
            and expected_photos_cal >= 100
            and delta_cal < 0.60 * expected_photos_cal
        ):
            omitidas.append({"idx": idx, **p})

    return omitidas


def _ya_bloqueado_en_este_turno(messages: Iterable[Any]) -> bool:
    """Anti-loop: permite máximo 1 bloqueo multifoto por turno humano."""
    for msg in reversed(list(messages or [])):
        mtype = getattr(msg, "type", None)
        cname = msg.__class__.__name__
        if mtype in ("human", "user") or cname == "HumanMessage":
            break
        if mtype == "tool" or cname == "ToolMessage":
            if MARKER_BLOQUEO_MULTIFOTO in str(getattr(msg, "content", "") or ""):
                return True
    return False


def detectar_bloqueo_multifoto_fusion(
    plate_photos: list[dict[str, Any]],
    tool_args: dict[str, Any],
    diary_calls_in_msg: list[dict[str, Any]],
    messages: list[Any],
    *,
    user_text: str = "",
    orig_meal: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Si un `correct_consumed_meal` (o fusión desde `log_consumed_meal`) en turno multi-foto omite
    alguna foto activa, devuelve el mensaje de bloqueo determinista; None si todo está completo."""
    if not plate_photos or len(plate_photos) < 2:
        return None
    if bool((tool_args or {}).get("delete")):
        return None
    if _ya_bloqueado_en_este_turno(messages):
        return None

    omitidas = fotos_no_cubiertas_en_llamadas(
        plate_photos,
        diary_calls_in_msg or [tool_args],
        user_text=user_text,
        orig_meal=orig_meal,
    )
    if not omitidas:
        return None

    faltan_txt = "; ".join(
        f"Foto {f['idx']}: «{f.get('desc')}»" + (f" (~{int(f['kcal'])} kcal)" if f.get("kcal") else "")
        for f in omitidas
    )
    todas_txt = "; ".join(
        f"Foto {i}: «{p.get('desc')}»" + (f" (~{int(p['kcal'])} kcal)" if p.get("kcal") else "")
        for i, p in enumerate(plate_photos, start=1) if isinstance(p, dict)
    )
    orig_cal = int(float((orig_meal or {}).get("calories") or 0))
    orig_name = str((orig_meal or {}).get("meal_name") or "").strip()
    fotos_cal = sum(int(float(p.get("kcal") or 0)) for p in plate_photos if isinstance(p, dict) and p.get("kcal"))
    meal_id = str((tool_args or {}).get("meal_id") or (orig_meal or {}).get("id") or "").strip()

    sugerencia_total = ""
    if orig_cal > 0 and fotos_cal > 0:
        sugerencia_total = (
            f" Conserva la comida previa («{orig_name}», {orig_cal} kcal) y súmale TODAS las fotos "
            f"(~{orig_cal + fotos_cal} kcal en total aprox., ajustado si eligió opción sin azúcar/etc.)."
        )

    return (
        f"{MARKER_BLOQUEO_MULTIFOTO} FALTAN FOTOS DEL TURNO EN LA COMIDA: "
        f"El usuario subió {len(plate_photos)} fotos ({todas_txt}), pero tu llamada dejó fuera: {faltan_txt}. "
        f"Aunque el usuario haya respondido dudas de una sola foto o dicho que fue parte de su comida, "
        f"NO debes omitir las demás fotos. Llama AHORA a `correct_consumed_meal`"
        + (f" con meal_id='{meal_id}'" if meal_id else "")
        + f" incluyendo TODOS los alimentos, ingredientes, calorías y macros de la comida original más TODAS las fotos activas.{sugerencia_total} "
        "(Nota interna invisible al usuario: corrige la llamada sin disculparte ni mencionar errores internos.)"
    )


def _turn_tool_messages(messages: list[Any]) -> list[str]:
    out: list[str] = []
    for msg in reversed(messages or []):
        mtype = getattr(msg, "type", None)
        cname = msg.__class__.__name__
        if mtype in ("human", "user") or cname == "HumanMessage":
            break
        if mtype == "tool" or cname == "ToolMessage":
            out.append(str(getattr(msg, "content", "") or ""))
    return out


def fusion_explicita_exitosa_este_turno(
    messages: list[Any],
    plate_photos: list[dict[str, Any]],
    *,
    solicitud_explicita: bool,
    correct_called: bool,
    user_text: str = "",
) -> bool:
    """True si en el turno hubo solicitud explícita de fusionar/corregir, `correct_consumed_meal`
    ejecutó con éxito en BD y no dejó fotos sin cubrir."""
    if not solicitud_explicita or not correct_called:
        return False
    tool_msgs = _turn_tool_messages(messages)
    exitoso = any("[ID_REGISTRO_DIARIO:" in m for m in tool_msgs)
    if not exitoso:
        return False
    if len(plate_photos or []) < 2:
        return True
    calls_args: list[dict[str, Any]] = []
    for msg in reversed(messages or []):
        mtype = getattr(msg, "type", None)
        cname = msg.__class__.__name__
        if mtype in ("human", "user") or cname == "HumanMessage":
            break
        for tc in (getattr(msg, "tool_calls", None) or []):
            nombre = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if nombre in ("log_consumed_meal", "correct_consumed_meal"):
                calls_args.append((tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)) or {})
    return not fotos_no_cubiertas_en_llamadas(plate_photos, calls_args, user_text=user_text)
