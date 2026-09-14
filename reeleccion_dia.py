# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-47 · 2026-09-14] Re-elegir, no reescribir.

Tercera prueba RD del dueño (plan d8b10b05, 14-sep, mismo perfil). El plan salió aprobado al primer intento y sin repetir
los planes anteriores, pero ninguno de sus 3 días llegó como lo armó el día determinista: la autocrítica corrigió los
días 1 y 2 con el LLM (yuca en 2 días y una cena de solo queso en ganancia muscular) y la regeneración quirúrgica, el 3
(huevo dos veces). El corrector devuelve el día entero (`SingleDayPlanModel.model_dump()`) y lo pega encima: conservó 6
platos de la biblioteca por el NOMBRE, pero sin `_template_id` ni `_recipe_source`, así que las protecciones de los
lotes 45 y 46 se apagaron sin avisar — el autofix cambió el huevo de la arepa por queso («revuelve queso blanco en una
sartén»), el tiempo por defecto cayó en el paso del agua fría y la batata del guiso bajó de 200 g a 10 g. La corrección
abrió además el mismo defecto que cerraba: casabe en 4 de 12 comidas. Unos 140 de los 257 s se fueron en reescribir.

Dos piezas:

  · `reelegir_en_lugar`: un día determinista que la autocrítica o la regeneración señalan se REARMA con el mismo armador
    (`deterministic_day.build_day_for_skeleton`), pidiéndole evitar lo señalado. Se acepta si las señales verificables
    del día bajan. Si no hay nada verificable ni ningún plato nombrado, el día de biblioteca se CONSERVA: una opinión del
    evaluador no tira una receta congelada. El LLM sólo entra si el rearmado no mejora.
  · `restaurar_procedencia`: cuando el LLM sí reescribe un día, los platos que dejó IGUALES (mismo nombre, misma franja,
    ningún alimento nuevo) vuelven con su procedencia; si vuelven todos, el día vuelve a ser determinista.

Knobs `MEALFIT_CRITIQUE_REPICK_DETERMINISTIC` y `MEALFIT_CRITIQUE_RESTORE_PROVENANCE` (True).
tooltip-anchor: P1-PLAN-LOTE-47-REELEGIR
"""
from __future__ import annotations

import copy
import logging
import re
import unicodedata
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_PREFIJO_NOMBRE = 24      # «bollitos de yuca relleno» basta para reconocer el plato dentro de un texto
_PREFIJO_MIN = 12         # por debajo, un nombre corto («Mangú») casaría con cualquier cosa
# Palabras de una línea de ingrediente que no son el alimento: comparar sin ellas deja pasar el cambio de gramos que
# el corrector hace al re-cuadrar el día, y no deja pasar el cambio de un alimento por otro.
_NO_ALIMENTO = frozenset((
    "de", "del", "la", "el", "los", "las", "con", "y", "en", "al", "para", "sin", "una", "uno", "unos", "unas",
    "taza", "tazas", "cda", "cdas", "cdta", "cdtas", "cucharada", "cucharadas", "cucharadita", "cucharaditas",
    "gramos", "gramo", "unidad", "unidades", "pizca", "rebanada", "rebanadas", "torta", "tortas", "lonja", "lonjas",
    "pedazo", "pedazos", "pequeno", "pequena", "mediano", "mediana", "grande", "gusto", "opcional", "peso", "crudo",
    "cruda", "aprox", "aproximadamente",
))


def _knob(nombre: str) -> bool:
    try:
        from knobs import _env_bool
        return _env_bool(nombre, True)
    except Exception:                                                          # noqa: BLE001
        return True


def enabled() -> bool:
    return _knob("MEALFIT_CRITIQUE_REPICK_DETERMINISTIC")


def procedencia_enabled() -> bool:
    return _knob("MEALFIT_CRITIQUE_RESTORE_PROVENANCE")


def _sa(s) -> str:
    return unicodedata.normalize("NFD", str(s or "").lower()).encode("ascii", "ignore").decode("ascii")


def es_determinista(dia) -> bool:
    return isinstance(dia, dict) and dia.get("_day_source") == "deterministic"


def _dia(days, n) -> Optional[dict]:
    return next((d for d in (days or []) if isinstance(d, dict) and d.get("day") == n), None)


def _franja(m) -> str:
    try:
        from constants import canonical_slot_key
        return str(canonical_slot_key(str((m or {}).get("meal") or "")) or "")
    except Exception:                                                          # noqa: BLE001
        return _sa((m or {}).get("meal"))


def _principal(dia) -> Optional[str]:
    fr = [_franja(m) for m in ((dia or {}).get("meals") or []) if isinstance(m, dict)]
    return next((s for s in ("almuerzo", "cena") if s in fr), None)


def _politica(counts, form_data) -> dict:
    """El mismo filtro de política que la autocrítica aplica a sus contadores de repetición."""
    try:
        import horizon
        fd = form_data or {}
        return horizon.filter_repetition_counts_for_policy(
            dict(counts or {}), fd.get("_plan_policy_effective"), enforced=bool(fd.get("_policy_enforced")))
    except Exception:                                                          # noqa: BLE001
        return dict(counts or {})


def senales(days, n, form_data) -> list:
    """Las señales VERIFICABLES de la autocrítica que tocan al día `n`, con los detectores que ella usa (los de
    `graph_orchestrator`): `[(tipo, detalle)]`. Sin día o sin detectores ⇒ `[]`."""
    dia = _dia(days, n)
    if dia is None:
        return []
    import deterministic_day as dd
    go = dd._go()
    if go is None:
        return []
    fd = form_data or {}
    meals = [m for m in (dia.get("meals") or []) if isinstance(m, dict)]
    out: list = []
    try:
        suyos: set = set()
        for m in meals:
            suyos |= dd._basicos_de(m)
        for lbl in sorted(set(_politica(go._count_staple_repetitions(days), fd)) & suyos):
            out.append(("basico", lbl))
    except Exception:                                                          # noqa: BLE001
        pass
    for tipo, fn in (("cena_debil", lambda: go._detect_gainmuscle_dinner_issues([dia], fd)),
                     ("base_ligera", lambda: go._detect_light_base_repeats([dia])),
                     ("franja", lambda: go._detect_slot_incoherence([dia]))):
        try:
            out.extend((tipo, str(s)) for s in (fn() or []))
        except Exception:                                                      # noqa: BLE001
            pass
    try:
        if go._days_with_same_day_protein_repeat({"days": [dia]}, user_staples=go._user_staple_labels(fd)):
            out.append(("proteina_del_dia", ""))
    except Exception:                                                          # noqa: BLE001
        pass
    try:
        rep = _politica(go.build_variety_report({"days": days}).get("cross_day_dishes") or {}, fd)
        nombres = " ".join(_sa(m.get("name")) for m in meals)
        out.extend(("plato_base", str(t)) for t in sorted(rep) if _sa(t) and _sa(t) in nombres)
    except Exception:                                                          # noqa: BLE001
        pass
    try:
        mono = _politica(go._count_cross_day_heavy_protein_repetition(days), fd)
        suyas: set = set()
        for m in meals:
            suyas |= set(go._protein_gate_labels_in_meal(m))
        out.extend(("monotonia", lbl) for lbl in sorted(set(mono) & suyas))
    except Exception:                                                          # noqa: BLE001
        pass
    return out


def evitar_para(days, n, form_data, textos: Iterable = ()) -> dict:
    """Qué pedirle al armador para el día `n`: las plantillas de las comidas implicadas en sus señales o nombradas en los
    textos (la sugerencia del evaluador, el problema de la marca, los motivos del rechazo), los básicos repetidos y, si
    la señal es de base ligera, la puerta de base ligera encendida para ese día."""
    ev = {"plantillas": set(), "basicos": set(), "bases_ligeras": False}
    dia = _dia(days, n)
    if dia is None:
        return ev
    import deterministic_day as dd
    go = dd._go()
    meals = [m for m in (dia.get("meals") or []) if isinstance(m, dict)]
    principal = _principal(dia)

    def _tids(pred):
        return {str(m.get("_template_id") or "") for m in meals if pred(m)}

    def _proteinas(m):
        try:
            return set(go._protein_gate_labels_in_meal(m)) if go is not None else set()
        except Exception:                                                      # noqa: BLE001
            return set()

    for tipo, det in senales(days, n, form_data):
        if tipo == "basico":
            ev["basicos"].add(det)
            ev["plantillas"] |= _tids(lambda m, d=det: d in dd._basicos_de(m))
        elif tipo == "cena_debil":
            ev["plantillas"] |= _tids(lambda m: _franja(m) == "cena")
        elif tipo == "proteina_del_dia":
            ev["plantillas"] |= _tids(lambda m: _franja(m) != principal and bool(_proteinas(m)))
        elif tipo == "base_ligera":
            ev["bases_ligeras"] = True
            ev["plantillas"] |= _tids(lambda m: _franja(m) == "merienda")
        elif tipo == "franja":
            t = _sa(det)
            if "comparten" in t:
                ev["plantillas"] |= _tids(lambda m: _franja(m) == "cena")
            if "merienda" in t:
                ev["plantillas"] |= _tids(lambda m: _franja(m) == "merienda")
        elif tipo == "plato_base":
            ev["plantillas"] |= _tids(lambda m, d=det: _sa(d) in _sa(m.get("name")))
        elif tipo == "monotonia":
            ev["plantillas"] |= _tids(lambda m, d=det: _franja(m) != principal and d in _proteinas(m))
    texto = " ".join(_sa(t) for t in (textos or ()) if t)
    if texto:
        for m in meals:
            pref = _sa(m.get("name"))[:_PREFIJO_NOMBRE].strip()
            if len(pref) >= _PREFIJO_MIN and pref in texto:
                ev["plantillas"].add(str(m.get("_template_id") or ""))
    ev["plantillas"].discard("")
    return ev


def _firma(dia) -> tuple:
    return tuple(str(m.get("_template_id") or m.get("name") or "") for m in ((dia or {}).get("meals") or [])
                 if isinstance(m, dict))


def reelegir(days, n, *, nutrition, form_data, skeleton_day=None, textos: Iterable = ()) -> tuple:
    """`(día, motivo)`: el día rearmado («reelegido»), el MISMO día («conservado»: nada verificable, o una opinión sin
    alternativa en la biblioteca) o `None` (el rearmado no mejora sus señales: que decida el corrector LLM)."""
    dia = _dia(days, n)
    if not es_determinista(dia):
        return None, "no_determinista"
    antes = senales(days, n, form_data)
    ev = evitar_para(days, n, form_data, textos)
    if not antes and not ev["plantillas"] and not ev["basicos"]:
        return dia, "conservado"
    import deterministic_day as dd
    # Dos intentos: el primero evita sólo lo señalado; si no basta, el segundo libera además la comida principal —la otra
    # mitad de «almuerzo y cena comparten la yuca»: con el almuerzo fijo, ninguna cena limpia cabía en el replay del 14-sep.
    intentos = [ev]
    _princ = {str(m.get("_template_id") or "") for m in (dia.get("meals") or [])
              if isinstance(m, dict) and _franja(m) == _principal(dia)} - {""}
    if antes and _princ and not _princ <= ev["plantillas"]:
        intentos.append(dict(ev, plantillas=set(ev["plantillas"]) | _princ))
    hubo_dia = False
    vistos_despues: list = []
    for ev_i in intentos:
        otros = [copy.deepcopy(d) for d in days if isinstance(d, dict) and d is not dia]
        try:
            nuevo = dd.build_day_for_skeleton(nutrition, form_data, skeleton_day or {}, n, memoria=otros, evitar=ev_i)
        except Exception as e:                                                 # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-47] rearmado del día {n} no-op: {e!r}")
            nuevo = None
        if not nuevo:
            continue
        hubo_dia = True
        nuevo["day"] = n
        cambio = _firma(nuevo) != _firma(dia)
        despues = senales([nuevo if d is dia else d for d in days], n, form_data)
        vistos_despues.append(sorted({t for t, _d in despues}))
        if not antes:
            if cambio and not despues:
                nuevo["_reelegido_por"] = ["texto"]
                return nuevo, "reelegido"
            return dia, "conservado"
        if cambio and len(despues) < len(antes):
            nuevo["_reelegido_por"] = sorted({t for t, _d in antes})
            return nuevo, "reelegido"
    if hubo_dia:
        # [P1-PLAN-LOTE-48] el porqué, en el log: plan 358a2cdf sólo dejó «el rearmado no mejora (sin_mejora)»
        logger.info(f"🔁 [P1-PLAN-LOTE-48] día {n}: el rearmado no bajó sus señales — antes "
                    f"{sorted({t for t, _d in antes})}, después {vistos_despues}")
    return None, ("sin_mejora" if hubo_dia else "sin_dia")


def reelegir_en_lugar(days, nums, *, nutrition, form_data, skeletons=(), textos: Iterable = (),
                      etiqueta: str = "", barrer: bool = False) -> tuple:
    """Rearma, uno a uno, los días deterministas de `nums` y los reemplaza EN `days` (la misma lista). Re-mide tras cada
    cambio: al rehacer el día 2 sin yuca, el día 1 deja de tener la señal y se conserva. Devuelve `(días para el
    corrector LLM, rearmados, conservados)`; los días que no son deterministas pasan tal cual al corrector.

    `barrer`: después, los OTROS días deterministas con señal verificable también se rearman si mejoran (si no, se quedan
    como estaban: nadie los había mandado al corrector). En la autocrítica evita la segunda vuelta: el día 3 del plan
    d8b10b05 (huevo dos veces) no lo nombró el evaluador, lo marcó la verificación posterior y lo rehízo el LLM después
    de aprobado el plan, con otra revisión completa."""
    para_llm, rearmados, conservados = [], [], []
    textos = [str(t) for t in (textos or ()) if t]
    if barrer:
        vistos = set(nums or [])
        _extra = [d.get("day") for d in (days or []) if es_determinista(d) and d.get("day") not in vistos]
    else:
        _extra = []
    for n in list(nums or []):
        dia = _dia(days, n)
        if not es_determinista(dia):
            para_llm.append(n)
            continue
        sk = next((s for s in (skeletons or []) if isinstance(s, dict) and s.get("day") == n), {})
        try:
            nuevo, motivo = reelegir(days, n, nutrition=nutrition, form_data=form_data, skeleton_day=sk, textos=textos)
        except Exception as e:                                                 # noqa: BLE001
            nuevo, motivo = None, f"error:{type(e).__name__}"
        if nuevo is None:
            para_llm.append(n)
            logger.info(f"🔁 [P1-PLAN-LOTE-47] {etiqueta} día {n} determinista: el rearmado no mejora ({motivo}) "
                        f"→ corrector LLM")
        elif nuevo is dia:
            dia.pop("_critique_unresolved", None)
            conservados.append(n)
            logger.info(f"🔁 [P1-PLAN-LOTE-47] {etiqueta} día {n} determinista conservado: la crítica no señala nada "
                        f"verificable en él")
        else:
            for i, d in enumerate(days):
                if d is dia:
                    days[i] = nuevo
                    break
            rearmados.append(n)
            logger.info(f"🔁 [P1-PLAN-LOTE-47] {etiqueta} día {n} re-elegido sin LLM "
                        f"({', '.join(nuevo.get('_reelegido_por') or [])}): "
                        + " | ".join(str(m.get("name"))[:40] for m in (nuevo.get("meals") or []) if isinstance(m, dict)))
    for n in _extra:
        dia = _dia(days, n)
        if not es_determinista(dia) or not senales(days, n, form_data):
            continue
        sk = next((s for s in (skeletons or []) if isinstance(s, dict) and s.get("day") == n), {})
        try:
            nuevo, _motivo = reelegir(days, n, nutrition=nutrition, form_data=form_data, skeleton_day=sk)
        except Exception:                                                      # noqa: BLE001
            nuevo = None
        if nuevo is not None and nuevo is not dia:
            for i, d in enumerate(days):
                if d is dia:
                    days[i] = nuevo
                    break
            rearmados.append(n)
            logger.info(f"🔁 [P1-PLAN-LOTE-47] {etiqueta} día {n} (no nombrado) re-elegido sin LLM "
                        f"({', '.join(nuevo.get('_reelegido_por') or [])})")
    # [P1-PLAN-LOTE-48 · 2026-09-14] Lo que otro rearmado ya arregló no va al corrector. Plan 358a2cdf: el día 3 (nombrado)
    # no mejoraba solo y quedó en cola; el barrido rehízo el día 1 sin yuca, la señal del 3 desapareció, y el 3 igual lo
    # reescribió el LLM (30 s y una cena de LLM). Se re-mide al final. tooltip-anchor: P1-PLAN-LOTE-48-REMEDIR-COLA
    for n in list(para_llm):
        dia = _dia(days, n)
        if es_determinista(dia) and not senales(days, n, form_data):
            para_llm.remove(n)
            dia.pop("_critique_unresolved", None)
            conservados.append(n)
            logger.info(f"🔁 [P1-PLAN-LOTE-48] {etiqueta} día {n} determinista: otro rearmado ya le quitó la señal → "
                        f"se conserva, sin LLM")
    return para_llm, rearmados, conservados


def _alimentos(m) -> frozenset:
    """Las palabras de alimento de la lista de ingredientes, sin cantidades, unidades ni plurales."""
    pal = set()
    for s in ((m or {}).get("ingredients") or []):
        for w in re.findall(r"[a-z]+", _sa(s)):
            if len(w) >= 3 and w not in _NO_ALIMENTO:
                pal.add(w[:-1] if (w.endswith("s") and len(w) > 4) else w)
    return frozenset(pal)


def restaurar_procedencia(original, corregido) -> int:
    """Los platos de biblioteca que el corrector LLM dejó IGUALES vuelven con su procedencia (receta congelada, plantilla,
    factor). Igual = mismo nombre, misma franja y ningún alimento que el original no tuviera. Si vuelven todos, el día
    vuelve a ser determinista. Devuelve cuántos restauró."""
    if not procedencia_enabled() or not isinstance(original, dict) or not isinstance(corregido, dict):
        return 0
    origen = [m for m in (original.get("meals") or []) if isinstance(m, dict)
              and m.get("_recipe_source") == "library" and m.get("_template_id")]
    meals = corregido.get("meals")
    if not origen or not isinstance(meals, list):
        return 0
    n = 0
    for i, m in enumerate(meals):
        if not isinstance(m, dict):
            continue
        o = next((x for x in origen if _sa(x.get("name")) == _sa(m.get("name")) and _franja(x) == _franja(m)), None)
        if o is None or not (_alimentos(m) <= _alimentos(o)):
            continue
        meals[i] = copy.deepcopy(o)
        n += 1
    if n:
        corregido["_procedencia_restaurada"] = n
        if n == len(meals) == len(original.get("meals") or []) and es_determinista(original):
            for k in ("_day_source", "_day_index", "_sodium_mg_est"):
                if k in original:
                    corregido[k] = original[k]
        logger.info(f"🧬 [P1-PLAN-LOTE-47] día {corregido.get('day')}: {n} plato(s) que el corrector dejó igual vuelven "
                    f"con su receta de biblioteca")
    return n
