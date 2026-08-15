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
_DESKTOP_PANEL = (_BACKEND_ROOT.parent / "frontend" / "src" / "components" / "history" / "HistoryDesktopPanel.jsx").read_text(encoding="utf-8")
_MOBILE_PANEL = (_BACKEND_ROOT.parent / "frontend" / "src" / "components" / "history" / "HistoryMobilePanel.jsx").read_text(encoding="utf-8")
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
    """[reapuntado 2026-07-28] El render de cards migró de History.jsx a los paneles
    (HistoryDesktopPanel/HistoryMobilePanel) y el chip 'Activo' se convirtió en un HERO dedicado
    con píldora + punto pulsante — MÁS prominente que el chip-primero del diseño viejo. La
    pregunta que protegía ("¿cuál plan estoy comiendo ahora?") se responde igual o mejor: ambos
    paneles reciben activePlanId y pintan el indicador con el token de acento."""
    # [P1-HIST-PAUSED-BADGE · 2026-08-14] La insignia dejó de ser un literal y pasó a
    # ternario: con el plan en pausa dice «Plan en pausa» en ámbar (sin glow, porque
    # el brillo verde ES la señal de «vivo»), y ACTIVO sigue siendo `var(--secondary)`.
    #
    # La aserción anterior buscaba la cadena exacta `background: "var(--secondary)"` y
    # se puso roja por un cambio correcto — el token no desapareció, se movió dentro de
    # una condición. Es el mismo modo de fallo de siempre: un test pegado a la GRAFÍA
    # convierte una mejora en un rojo, y a un rojo injusto se le responde relajando el
    # test. Aquí se ancla la propiedad: el estado activo sigue pintándose con el token
    # de acento, aparezca solo o como rama de un ternario.
    # Se ancla la RAMA ACTIVA del ternario (`paused ? <pausa> : <activo>`), no la
    # presencia suelta del token: el primer intento buscaba `background: … var(--
    # secondary)` en cualquier parte y una mutación de verificación —cambiar el punto
    # a `#22C55E`— PASÓ, porque el panel de Desktop tiene un
    # `background: "linear-gradient(var(--secondary)…"` decorativo que hacía de
    # señuelo. Un guard que casa con un vecino no vigila a su objetivo.
    rama_activa = re.compile(r'paused\s*\?\s*(?:[^:?]|\([^()]*\))+:\s*"([^"]*)"')
    for nombre, src in (("Desktop", _DESKTOP_PANEL), ("Mobile", _MOBILE_PANEL)):
        assert "activePlanId" in src, f"{nombre}: el panel perdió la noción de plan activo"

        ramas = rama_activa.findall(src)
        assert ramas, (
            f"{nombre}: no encuentro el ternario `paused ? … : …` de la insignia. Si "
            "cambió de forma, reapuntá este test — lo que se protege es que el estado "
            "ACTIVO se pinte con el token de acento."
        )
        assert any("var(--secondary)" in r for r in ramas), (
            f"{nombre}: ninguna rama ACTIVA de la insignia usa `var(--secondary)`; "
            f"encontradas: {ramas}."
        )
        # Y NINGUNA rama activa puede llevar un color a mano. Esta es la mitad que
        # de verdad muerde: con sólo pedir «que alguna use el token», mutar el punto
        # a `#22C55E` seguía pasando porque el `color:` de la píldora conservaba el
        # suyo. Un guard satisfecho por un vecino no vigila a su objetivo.
        # (La rama de PAUSA sí lleva `#FBBF24` a mano y queda fuera a propósito: es
        # el ámbar de espera, sin token propio en el sistema.)
        a_mano = [r for r in ramas if re.fullmatch(r"#[0-9A-Fa-f]{3,8}", r.strip())]
        assert not a_mano, (
            f"{nombre}: rama(s) ACTIVA(s) de la insignia con color a mano: {a_mano}. "
            "Usá un token `var(--…)`: el Historial se renderiza en los cuatro temas y "
            "un hex fijo sólo acierta en uno."
        )

def test_card_active_class_applied_conditionally():
    """[reapuntado 2026-07-28] La clase `cardActive` del monolito murió con la extracción a
    paneles. La CONDICIONALIDAD vive en `normalizePlan`: `active` solo cuando el id coincide
    con activePlanId — ni siempre, ni nunca."""
    assert "raw.id === activePlanId" in _DESKTOP_PANEL, (
        "el panel ya no computa `active` por comparación de id — las cards activas no se "
        "distinguen del resto"
    )

def test_css_status_active_chip_defined():
    """[reapuntado 2026-07-28] La palette emerald hardcodeada (#ECFDF5/#10B981) fue reemplazada
    A PROPÓSITO por tokens (commit "verdes→tokens" · 2026-07-09): el indicador usa
    var(--secondary) con color-mix para fondo/borde. Anclar el hex viejo revertiría esa
    decisión. Se ancla el token."""
    assert _DESKTOP_PANEL.count("var(--secondary)") >= 2, (
        "el indicador de activo perdió el token de acento (¿volvió el hex hardcodeado?)"
    )
    assert "color-mix" in _DESKTOP_PANEL, (
        "el fondo/borde del indicador ya no deriva del token — el chip pierde coherencia de tema"
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
