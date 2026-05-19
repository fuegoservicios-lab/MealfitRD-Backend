"""[P3-HIST-ACTIVE-CHIP · 2026-05-18] Chip "Activo" en el listado del Historial.

Síntoma reportado por usuario:
> "empecemos a trabajar con el historial, quiero que se marque cuando un
>  plan esta activo o cuando ya paso, ejemplo: el plan que ves en la imagen
>  esta activo actualmente"

El listado mostraba todos los planes con el mismo tratamiento visual; un
usuario con varios planes archivados + uno corriendo HOY no podía
distinguirlos sin abrir el modal. El estado temporal (active / past /
future) es ortogonal al estado de generación (complete / partial / failed
/ in_progress) — un plan puede estar "activo + parcial" simultáneamente.

Diseño:
  - Backend: `/history-list` expone `grocery_start_date` + `cycle_start_date`
    top-level (jsonb extract via `->>`). Si el plan no los tiene resueltos
    (cron `_resolve_grocery_start_date` aún no corrió), el frontend
    cae a `plan.created_at`.
  - Frontend: helper `getTemporalStatus(plan)` resuelve el bucket con
    fallback chain start + ventana [start, start + totalDays). Resolución
    por DÍA local (espeja P3-SHIFT-DATEONLY-LOCAL · 2026-05-18) para
    evitar off-by-one por TZ.
  - UI: chip "Activo" verde como PRIMER chip de cardActions (antes que
    calorías y status); card con borde verde sutil + gradiente cuando
    activo. Planes pasados/futuros sin chip (estado por defecto).

Si alguien refactora el feature, este test falla con un mensaje que apunta
al marker original.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_HISTORY_JSX = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.jsx"
).read_text(encoding="utf-8")
_HISTORY_CSS = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.module.css"
).read_text(encoding="utf-8")
_PLANS_ROUTER = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_APP_PY = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


def test_marker_present_in_source():
    """El marker `P3-HIST-ACTIVE-CHIP` DEBE permanecer en el código
    fuente del fix (History.jsx + History.module.css + routers/plans.py)
    como anchor de regresión. Si un refactor borra todos los markers,
    este test falla.

    Nota: NO miramos `_LAST_KNOWN_PFIX` en app.py — ese campo rota a
    cada P-fix nuevo (el último mergeado en HEAD), así que asertarlo
    aquí lo rompería en el siguiente P-fix. El anchor de regresión
    es el comentario en el código que implementa el fix."""
    assert "P3-HIST-ACTIVE-CHIP" in _HISTORY_JSX, (
        "Marker P3-HIST-ACTIVE-CHIP ausente en History.jsx — un "
        "refactor podría borrar el feature sin dejar trazo."
    )
    assert "P3-HIST-ACTIVE-CHIP" in _HISTORY_CSS, (
        "Marker P3-HIST-ACTIVE-CHIP ausente en History.module.css — "
        "los estilos .statusActive/.cardActive perdieron su anchor."
    )
    assert "P3-HIST-ACTIVE-CHIP" in _PLANS_ROUTER, (
        "Marker P3-HIST-ACTIVE-CHIP ausente en routers/plans.py — "
        "los nuevos SELECT/dict keys perdieron su anchor."
    )


def test_backend_exposes_grocery_start_date():
    """El endpoint `/history-list` DEBE proyectar `grocery_start_date` y
    `cycle_start_date` desde plan_data. Sin estos campos, el frontend
    no puede derivar el bucket temporal sin descargar todo el plan_data
    (lo que rompería el ahorro de bandwidth de P1-HIST-AUDIT-4)."""
    assert "plan_data->>'grocery_start_date' AS grocery_start_date" in _PLANS_ROUTER, (
        "SELECT del /history-list no proyecta `grocery_start_date`. "
        "El frontend caerá al fallback `plan.created_at` y el chip "
        "'Activo' será incorrecto para planes cuyo ciclo de compras "
        "empieza en fecha distinta a created_at."
    )
    assert "plan_data->>'cycle_start_date' AS cycle_start_date" in _PLANS_ROUTER, (
        "SELECT del /history-list no proyecta `cycle_start_date`. "
        "Fallback intermedio entre grocery_start_date y created_at "
        "no está disponible."
    )
    # El dict de respuesta también debe exponerlos.
    assert '"grocery_start_date":' in _PLANS_ROUTER, (
        "La response del /history-list no incluye `grocery_start_date` "
        "en el dict de salida — el SELECT lo trae pero no se devuelve "
        "al cliente."
    )
    assert '"cycle_start_date":' in _PLANS_ROUTER, (
        "La response del /history-list no incluye `cycle_start_date` "
        "en el dict de salida."
    )


def test_get_temporal_status_helper_exists():
    """El helper `getTemporalStatus(plan)` es el SSOT del bucket
    temporal en frontend. Tres buckets: active / past / future. Si
    alguien lo renombra o lo elimina, este test falla."""
    assert "const getTemporalStatus = (plan) =>" in _HISTORY_JSX, (
        "Helper `getTemporalStatus` removido o renombrado en History.jsx. "
        "El chip 'Activo' quedará sin source-of-truth y el feature "
        "P3-HIST-ACTIVE-CHIP se rompe silenciosamente."
    )


def test_temporal_status_fallback_chain_preserved():
    """Cuando el plan no trae `grocery_start_date`, el helper debe caer
    a `cycle_start_date` y finalmente a `plan.created_at`. Cualquier
    nivel removido degrada la precisión del chip para planes legacy."""
    # Anclamos a la expresión del fallback chain.
    idx = _HISTORY_JSX.find("const getTemporalStatus = (plan) =>")
    assert idx > 0
    block = _HISTORY_JSX[idx:idx + 3000]
    assert "plan.grocery_start_date" in block, (
        "Fallback chain del start NO empieza por `grocery_start_date` — "
        "es el campo preferido (resuelto por el cron del backend)."
    )
    assert "plan.cycle_start_date" in block or "plan.plan_data" in block, (
        "Fallback intermedio (cycle_start_date / plan_data.*) no presente."
    )
    assert "plan.created_at" in block, (
        "Último fallback `plan.created_at` removido — planes muy "
        "legacy sin metadata de start quedarán sin chip aunque podrían "
        "calificarse por created_at."
    )


def test_temporal_status_parses_date_only_as_local():
    """Espeja P3-SHIFT-DATEONLY-LOCAL · 2026-05-18: strings date-only
    `YYYY-MM-DD` deben parsearse como fecha LOCAL del usuario (cero
    TZ dance). Pre-fix del backend (P3-SHIFT-DATEONLY-LOCAL) producía
    off-by-one en TZ negativas (Santo Domingo -4) — el frontend NO debe
    repetir el bug aquí."""
    idx = _HISTORY_JSX.find("const getTemporalStatus = (plan) =>")
    assert idx > 0
    block = _HISTORY_JSX[idx:idx + 3000]
    # Detection del formato date-only y construcción Local.
    assert re.search(r"\^\\d\{4\}-\\d\{2\}-\\d\{2\}\$", block) is not None or (
        "/^\\d{4}-\\d{2}-\\d{2}$/" in block
    ), (
        "Regex de detección `^\\d{4}-\\d{2}-\\d{2}$` ausente — el helper "
        "no distinguirá date-only strings de timestamps con TZ y "
        "reintroducirá el bug de off-by-one en TZ negativas."
    )
    assert "new Date(y, m - 1, d" in block, (
        "Constructor `new Date(y, m-1, d, ...)` (fecha LOCAL) ausente — "
        "el helper podría estar parseando con `new Date(string)` que "
        "interpreta date-only como UTC midnight (bug TZ)."
    )


def test_active_chip_rendered_first_in_card_actions():
    """El chip 'Activo' debe aparecer ANTES del caloriesBadge y del
    chip de generación (Parcial/Falló/etc). Es la pregunta más
    importante para el usuario ("¿cuál plan estoy comiendo ahora?")
    y va en la posición más prominente del cardActions."""
    actions_idx = _HISTORY_JSX.find('<div className={styles.cardActions}>')
    assert actions_idx > 0, "<div className={styles.cardActions}> no encontrado en History.jsx"
    # Block hasta el siguiente cierre del div (~250 líneas siguientes —
    # cardActions tiene muchos chips condicionales con comentarios extensos,
    # ~3000 chars es la ventana real).
    block = _HISTORY_JSX[actions_idx:actions_idx + 3500]
    active_pos = block.find("styles.statusActive")
    calories_pos = block.find("styles.caloriesBadge")
    assert active_pos > 0, (
        "Chip `styles.statusActive` no se renderiza dentro de cardActions. "
        "Feature P3-HIST-ACTIVE-CHIP quedó sin UI."
    )
    assert calories_pos > 0
    assert active_pos < calories_pos, (
        "El chip 'Activo' debe ir ANTES del caloriesBadge en cardActions. "
        "Si está después, el orden visual rompe el contrato de diseño "
        "P3-HIST-ACTIVE-CHIP (chip temporal primero)."
    )


def test_card_active_class_applied_conditionally():
    """La clase `cardActive` (borde verde + fondo sutil) debe aplicarse
    al wrapper SOLO cuando el plan es temporal-active. Aplicarla siempre
    o nunca rompería el resaltado diferenciado."""
    # Anchor: el className del motion.div del map.
    assert "_isActive ? styles.cardActive" in _HISTORY_JSX, (
        "El motion.div del map NO aplica `styles.cardActive` "
        "condicionalmente. Las cards activas no se distinguen visualmente "
        "del resto."
    )


def test_css_status_active_chip_defined():
    """CSS class `.statusActive` (chip verde) DEBE existir en
    History.module.css. Sin el bloque, el chip no tiene estilo y
    aparece como texto pelado."""
    assert ".statusActive {" in _HISTORY_CSS, (
        ".statusActive ausente de History.module.css — el chip "
        "renderiza sin estilo (texto verde plano sin píldora)."
    )
    # Verde emerald — debe diferenciarse de los otros chips de
    # generación (amber=partial, red=failed, blue=in_progress, gray=unknown).
    # Iteramos TODAS las apariciones de `.statusActive {` porque el
    # selector compuesto inicial (`.statusPartial, .statusFailed, …,
    # .statusActive {`) solo define el shape compartido (padding /
    # border-radius / font); el bloque standalone con la palette
    # viene después. Al menos UNO debe contener el verde emerald.
    found_palette = False
    cursor = 0
    while True:
        idx = _HISTORY_CSS.find(".statusActive {", cursor)
        if idx < 0:
            break
        block = _HISTORY_CSS[idx:idx + 500]
        if "#ECFDF5" in block or "#10B981" in block or "#047857" in block:
            found_palette = True
            break
        cursor = idx + 1
    assert found_palette, (
        ".statusActive no usa palette verde emerald (#ECFDF5/#10B981/#047857). "
        "Si se cambió a otro color, el chip podría confundirse con los "
        "chips de generación (amber=partial, red=failed, blue=in_progress, "
        "gray=unknown)."
    )


def test_css_card_active_modifier_defined():
    """CSS class `.cardActive` (resaltado del wrapper) DEBE existir
    para el borde verde sutil + gradiente solicitado por el usuario."""
    assert ".cardActive {" in _HISTORY_CSS, (
        ".cardActive ausente de History.module.css — el wrapper del "
        "plan activo no se resalta visualmente y solo se nota el chip."
    )
    assert ".cardActive::before" in _HISTORY_CSS, (
        ".cardActive::before (barra lateral verde permanente) ausente. "
        "El acento vertical izquierdo no se fija en verde para el plan "
        "activo — solo aparecerá en hover (mismo comportamiento que cards "
        "no-activas)."
    )
