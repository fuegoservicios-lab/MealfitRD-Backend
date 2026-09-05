"""[P1-NEXT-LEVEL-BATCH · 2026-07-02] Biblioteca de platos — creatividad por RECOMBINACIÓN.

La creatividad del motor dependía 100% del LLM + prompt: los guards (raw-staple, dish-quality)
evitan disparates pero nada COMPONE. Esta biblioteca (data/dish_templates.json, ~85 plantillas
DD curadas con slot/proteína/base/técnica/transform) le da al day-generator un espacio
verificado del cual ELEGIR Y ADAPTAR — los platos transformados que el owner pidió
(panqueques de avena, bollitos de yuca, pastelón, arepitas) se vuelven DATA, no prompt-fe.

Integración: `build_dish_library_context(skeleton_day, day_num)` produce un bloque compacto
por día — muestra determinista (seed=day_num → mismo prompt para el mismo día = prompt-cache
preservado) filtrada por el pool de proteínas asignado por el planner y por slot (los slots
de cada plantilla RESPETAN el SSOT de constants.SLOT_INAPPROPRIATE_FOODS: cero arroz en
cena/desayuno, sopones solo almuerzo). El bloque es INSPIRACIÓN ("elige/adapta o crea uno
equivalente"), no obligación — el LLM conserva libertad creativa.

Fail-open total: sin archivo / JSON corrupto / knob OFF → ''. Knob: MEALFIT_DISH_LIBRARY
(default ON — prompt-aditivo, ~100-150 tokens por día).
tooltip-anchor: P1-NEXT-LEVEL-LIBRARY. Test: test_p1_next_level_batch.py.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re

from knobs import _env_bool, _env_int

logger = logging.getLogger(__name__)

DISH_LIBRARY_ENABLED = _env_bool("MEALFIT_DISH_LIBRARY", True)
DISH_LIBRARY_PER_SLOT = _env_int("MEALFIT_DISH_LIBRARY_PER_SLOT", 2, validator=lambda v: 1 <= v <= 5)
# [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-E) Mínimo diario de platos TRANSFORMADOS pedido al day-gen
# (prompt-side, soft — el LLM conserva libertad; el KPI transform_ratio del dish_quality_report mide
# obediencia). 0 = solo la priorización genérica previa. Clamp [0, 3].
DISH_LIBRARY_TRANSFORM_MIN = _env_int("MEALFIT_DISH_LIBRARY_TRANSFORM_MIN", 1, validator=lambda v: 0 <= v <= 3)

_TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dish_templates.json")
# [P1-DISH-LIBRARY-COUNTRY · 2026-08-21] Caché POR RUTA. Era un único global: con rutas por país,
# el primer archivo cargado en el proceso se le habría servido a todos los países que vinieran
# después — la misma trampa que `_VERIFIED_CATALOG_INSTRUCTION_CACHE` tenía antes de que el país
# entrara en su clave.
_CACHE_BY_PATH: dict = {}

_SLOT_LABELS = {"desayuno": "Desayuno", "almuerzo": "Almuerzo", "cena": "Cena", "merienda": "Merienda"}


def _templates_path_for_country(country=None) -> str:
    """[P1-DISH-LIBRARY-COUNTRY · 2026-08-21] Ruta del `dish_templates*.json` del usuario.

    Dos SSOT ajenos, ninguno reescrito aquí: `country_for_form_data` (la ÚNICA puerta de lectura
    de país del motor, que además aplica el knob maestro) y `_dish_templates_path_for_country`
    (Fase 2 T5-T7, que ya sabe qué archivo tiene cada país y hace fallback a RD). Escribir aquí
    una segunda tabla de rutas sería la 4ª tabla que P1-DIET-CANON-SSOT prohibió.

    Fail-open a la ruta dominicana ante cualquier problema: perder la sección de inspiración
    entera es peor que darla en el idioma equivocado.

    tooltip-anchor: _templates_path_for_country (test_p1_dish_library_country.py)"""
    if not country:
        return _TEMPLATES_PATH
    try:
        from constants import country_for_form_data
        canon = country_for_form_data({"country": country})
        if canon == "DO":
            return _TEMPLATES_PATH
        from graph_orchestrator import _dish_templates_path_for_country
        return _dish_templates_path_for_country(canon) or _TEMPLATES_PATH
    except Exception:
        return _TEMPLATES_PATH


def load_dish_templates(path: str = None) -> list:
    """Carga (una vez por ruta) las plantillas. Fail-open → []."""
    _path = path or _TEMPLATES_PATH
    _hit = _CACHE_BY_PATH.get(_path)
    if _hit is not None:
        return _hit
    try:
        with open(_path, encoding="utf-8") as f:
            data = json.load(f)
        templates = data.get("templates") or []
        loaded = [t for t in templates if isinstance(t, dict) and t.get("name") and t.get("slots")]
        # [P1-ARQ25-F2-PLANPOLICY · 2026-09-02] `template_id` acuñado al cargar (hash estable de
        # biblioteca+base+nombre+técnica, alias para renombres): cero reescritura del JSON.
        try:
            from plan_policy import attach_template_ids, library_key_for_path
            attach_template_ids(loaded, library_key_for_path(_path))
        except Exception as _tid_err:
            logger.warning(f"[P1-ARQ25-F2-PLANPOLICY] template_id no acuñado: {_tid_err}")
    except Exception as _e:
        logger.warning(f"[P1-NEXT-LEVEL-LIBRARY] no se cargaron plantillas: {type(_e).__name__}: {_e}")
        loaded = []
    _CACHE_BY_PATH[_path] = loaded
    return loaded


def _protein_matches_pool(template_protein: str, pool_ascii: str) -> bool:
    """¿La proteína de la plantilla es compatible con el pool asignado por el planner?
    'none'/'mixta'/'queso'/'legumbre'/'huevo' son siempre compatibles (el SSOT del
    day-generator ya permite huevo/queso/legumbres como diversificadores)."""
    p = str(template_protein or "none").lower()
    if p in ("none", "mixta", "queso", "legumbre", "huevo"):
        return True
    return bool(re.search(r"\b" + re.escape(p), pool_ascii))


def sample_templates_for_slot(slot_key: str, pool_ascii: str, k: int, seed: int,
                              avoid_tokens: tuple = (), country=None) -> list:
    """Muestra DETERMINISTA (seed) de hasta k plantillas del slot compatibles con el pool.
    Prioriza transformadas (transform=True) — son la creatividad que los staples no dan.
    [P1-DISH-LIBRARY-COUNTRY · 2026-08-21] `country=None` ⇒ biblioteca dominicana, idéntico a
    antes; país beta ⇒ la suya."""
    cands = []
    for t in load_dish_templates(_templates_path_for_country(country)):
        if slot_key not in (t.get("slots") or []):
            continue
        if not _protein_matches_pool(t.get("protein"), pool_ascii):
            continue
        name_low = str(t.get("name", "")).lower()
        if any(tok and tok in name_low for tok in avoid_tokens):
            continue
        cands.append(t)
    if not cands:
        return []
    rng = random.Random(int(seed) * 1000003 + sum(ord(c) for c in slot_key))
    transformed = [t for t in cands if t.get("transform")]
    plain = [t for t in cands if not t.get("transform")]
    rng.shuffle(transformed)
    rng.shuffle(plain)
    # al menos 1 transformada si existe (la mitad del valor de la biblioteca es el transform)
    picked = (transformed[:max(1, k - 1)] + plain)[:k]
    return picked


def _canon_country_or_do(country=None) -> str:
    """País canonicalizado por la ÚNICA puerta (`country_for_form_data`), 'DO' si falla.

    Extraído de `_inspiration_heading` para que las dos decisiones de país de este módulo
    (el encabezado y el trailer de transformados) salgan de la MISMA derivación: dos copias
    de la misma pregunta es como nacen los espejos que driftan."""
    try:
        from constants import country_for_form_data
        return country_for_form_data({"country": country}) if country else "DO"
    except Exception:
        return "DO"


def _inspiration_heading(country=None, culture_weights=None) -> str:
    """[P1-ARQ25-F7-CULTURE] Con pesos de cocina, el encabezado los nombra («INSPIRACIÓN: COCINA DOMINICANA 70 % ·
    COCINA ESPAÑOLA 30 %»); un solo perfil ⇒ literal histórico byte a byte."""
    if culture_weights:
        try:
            from cultural_profiles import heading_for_weights
            return heading_for_weights(culture_weights)
        except Exception:
            pass
    return _inspiration_heading_legacy(country)


def _inspiration_heading_legacy(country=None) -> str:
    """[P1-DISH-LIBRARY-COUNTRY · 2026-08-21] Encabezado del bloque. DO conserva el literal
    histórico byte a byte; un país beta lee su propio nombre, tomado de
    `COUNTRY_PROFILES[cc]['name_es']` — el MISMO SSOT que usa el juez culinario para su variante,
    no una segunda tabla de gentilicios."""
    try:
        from constants import COUNTRY_PROFILES
        canon = _canon_country_or_do(country)
        if canon != "DO":
            _nm = (COUNTRY_PROFILES.get(canon) or {}).get("name_es")
            if _nm:
                return f"INSPIRACIÓN DE {_nm.upper()}"
    except Exception:
        pass
    return "INSPIRACIÓN DOMINICANA"


def build_dish_library_context(skeleton_day: dict, day_num: int, country=None, culture_weights=None) -> str:
    """Bloque de inspiración por día para el prompt del day-generator. '' si knob OFF /
    sin plantillas compatibles. Determinista por (día, pool) → prompt-cache friendly.

    [P1-DISH-LIBRARY-COUNTRY · 2026-08-21] `country` es el tramo DINÁMICO del prompt: el más
    concreto y el más cercano a la generación. Sin él, la cabecera beta de Fase 1 («los platos
    dominicanos NO son requisito») perdía contra ocho platos dominicanos concretos veinte mil
    caracteres después — el modo de fallo que P1-DIET-BLIND-DIRECTIVES midió: entre una
    declaración general y un ejemplo concreto, el modelo obedece al ejemplo."""
    if not DISH_LIBRARY_ENABLED or not isinstance(skeleton_day, dict):
        return ""
    try:
        from constants import strip_accents
        pool_ascii = strip_accents(", ".join(
            str(x) for x in (skeleton_day.get("protein_pool") or [])).lower())
        meal_types = skeleton_day.get("meal_types") or ["Desayuno", "Almuerzo", "Merienda", "Cena"]
        lines = []
        _seen_slots = set()
        for mt in meal_types:
            slot = strip_accents(str(mt).strip().lower())
            if slot not in _SLOT_LABELS:
                # [P3-DISHLIB-MERIENDA-SLOTS · 2026-07-30] (audit solver+seeder v5) `_SLOT_LABELS`
                # tiene 'merienda' a secas, pero el slot REAL de todo plan clínico es 'Merienda
                # AM'/'PM'/'Nocturna' (lo fuerza `MEAL_TYPES_BY_COUNT[5|6]` antes de que el
                # day-gen lea el esqueleto) ⇒ `continue` y esas comidas se componían SIN
                # biblioteca de inspiración, cayendo a los staples repetidos que la biblioteca
                # existe para evitar. En un plan bariátrico son 3 de 6 slots, y es justo el
                # perfil que más depende de meriendas bien diseñadas.
                # La función hermana de este MISMO módulo (`build_swap_inspiration_context`) ya
                # tenía este fallback: un swap de esa merienda recibía inspiración y la
                # generación no. Paridad entre surfaces del mismo módulo.
                slot = next((k for k in _SLOT_LABELS if k in slot), None)
                if not slot:
                    continue
            if slot in _seen_slots:
                continue   # 3 meriendas → una sola línea de inspiración (evita repetir el bloque)
            _seen_slots.add(slot)
            picks = sample_templates_for_slot(slot, pool_ascii, int(DISH_LIBRARY_PER_SLOT),
                                              int(day_num or 1), country=country)
            if not picks:
                continue
            entries = "; ".join(
                f"{t['name']} ({t.get('technique', 'libre')})" for t in picks
            )
            lines.append(f"   • {_SLOT_LABELS[slot]}: {entries}")
        if not lines:
            return ""
        # [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-E) pedido explícito de mínimo transformado por día
        # (soft): "elige y adapta" era inspiración pura y el LLM podía ignorarla sin costo.
        _tf_min = int(DISH_LIBRARY_TRANSFORM_MIN)
        # [P3-DISHLIB-TRANSFORM-DO-EXAMPLES · 2026-08-23] Este trailer es la ÚLTIMA frase del
        # bloque más concreto del prompt. Medido antes de tocar: los renders de ES/MX/US/PR/CO
        # terminaban TODOS con «panqueques/arepitas/bollitos», y 'arepitas' y 'bollitos' son
        # entradas de `_DO_LEXICON_NEUTRAL` — o sea, léxico DO que el resto del stack beta ya
        # neutraliza, colado en el último renglón que el modelo lee. La variante beta describe la
        # FORMA (masas, tortitas, croquetas, horneados) en vez de recetas concretas: una categoría
        # no pierde nada y no arrastra el nombre de un plato dominicano. NO se neutraliza el bloque
        # entero con `neutralize_do_lexicon`: ese mapa manda `casabe → pan tostado integral` y
        # `Casabe` es una fila VIVA del catálogo. DO conserva su literal byte a byte.
        _tf_examples = ("panqueques/arepitas/bollitos/guiso u horneado con nombre propio"
                        if _canon_country_or_do(country) == "DO"
                        else "masas, tortitas, croquetas, guisos u horneados con nombre propio")
        _tf_line = (
            f"\n   🎯 Incluye HOY al menos {_tf_min} plato(s) TRANSFORMADO(s) ({_tf_examples}) "
            "siempre que encaje con los macros, el horario y las reglas clínicas del día.\n"
        ) if _tf_min > 0 else "\n"
        return (
            f"\n🍽️ {_inspiration_heading(country, culture_weights)} (biblioteca curada — ELIGE Y ADAPTA una, o crea un plato "
            "equivalente en espíritu; ajusta porciones a los macros del día):\n"
            + "\n".join(lines)
            + "\n   💡 Prioriza preparaciones TRANSFORMADAS (masas, guisos, rellenos, horneados) "
              "sobre staples sueltos — un plato con nombre propio se disfruta y se repite."
            + _tf_line
        )
    except Exception as _e:
        logger.debug(f"[P1-NEXT-LEVEL-LIBRARY] contexto no-op: {type(_e).__name__}: {_e}")
        return ""


# [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-F) pool amplio para inspiración en updates: el swap/chat no
# tiene el protein_pool del planner; sin pool, las plantillas proteína-específicas se filtrarían.
_BROAD_POOL_ASCII = "pollo pescado res cerdo pavo atun camarones salmon huevo queso legumbre"


def _diet_safe_pool_ascii(diet_type=None, allergies=None) -> str:
    """[P3-SWAP-INSPIRATION-DIET · 2026-07-31] (audit solver+seeder v6 · F26) Filtra el pool ancho de
    inspiración con el MISMO backstop determinista que juzga el plato resultante.

    `_BROAD_POOL_ASCII` es una constante con pollo/pescado/res/cerdo/…: se ofrecía igual a un vegano
    o a un alérgico, así que el LLM adaptaba una plantilla de carne, `clinical_backstop_for_meal` la
    rechazaba y el swap gastaba un reintento. Si el sesgo persistía los 3 intentos, el usuario recibía
    "no se pudo cambiar el plato" — un fallo de producto inducido por una señal que el propio sistema
    fabricó.

    Import perezoso a propósito: `dish_library` está por debajo de `graph_orchestrator` y no puede
    importarlo a nivel de módulo. Fail-open al pool ancho si el escáner no está disponible: quedarse
    sin inspiración es peor que una inspiración que el backstop ya sabe rechazar.
    tooltip-anchor: P3-SWAP-INSPIRATION-DIET"""
    _toks = _BROAD_POOL_ASCII.split()
    if not diet_type and not allergies:
        return _BROAD_POOL_ASCII
    try:
        from graph_orchestrator import clinical_backstop_for_meal as _cb
        _safe = [t for t in _toks
                 if not _cb({"name": t, "ingredients": [t]},
                            allergies=list(allergies or []), diet_type=diet_type)]
        return " ".join(_safe) if _safe else _BROAD_POOL_ASCII
    except Exception as _e:
        logger.debug(f"[P3-SWAP-INSPIRATION-DIET] filtro no-op: {type(_e).__name__}: {_e}")
        return _BROAD_POOL_ASCII


def build_swap_inspiration_context(meal_type: str, seed: int = 1, avoid_names=None,
                                   *, diet_type=None, allergies=None, country=None) -> str:
    """[P2-AUDIT-V6-BATCH · 2026-07-03] (P2-F) Inspiración compacta de la biblioteca para las
    superficies de UPDATE (swap / chat-modify) — antes solo el day-gen de form-gen la recibía,
    así que un plato actualizado perdía la creatividad por recombinación de las 87 plantillas.
    Soft ('elige y adapta si encaja'), determinista por seed, '' si knob OFF / slot desconocido.
    tooltip-anchor: P2-AUDIT-V6-BATCH (P2-F)"""
    if not DISH_LIBRARY_ENABLED:
        return ""
    try:
        from constants import strip_accents
        slot = strip_accents(str(meal_type or "").strip().lower())
        if slot not in _SLOT_LABELS:
            for k in _SLOT_LABELS:
                if k in slot:
                    slot = k
                    break
        if slot not in _SLOT_LABELS:
            return ""
        avoid = tuple(strip_accents(str(n).lower())[:30] for n in (avoid_names or [])[:10] if str(n).strip())
        picks = sample_templates_for_slot(slot, _diet_safe_pool_ascii(diet_type, allergies),
                                          int(DISH_LIBRARY_PER_SLOT),
                                          int(seed or 1), avoid_tokens=avoid, country=country)
        if not picks:
            return ""
        entries = "; ".join(f"{t['name']} ({t.get('technique', 'libre')})" for t in picks)
        return (
            f"\n    - 🍽️ INSPIRACIÓN ({_SLOT_LABELS[slot]}, biblioteca curada): {entries} — "
            "ELIGE Y ADAPTA una si encaja con los ingredientes disponibles, o crea un plato "
            "equivalente en espíritu. Prefiere preparaciones con nombre propio sobre staples sueltos."
        )
    except Exception as _e:
        logger.debug(f"[P2-AUDIT-V6-BATCH] (P2-F) inspiración de update no-op: {type(_e).__name__}: {_e}")
        return ""
