"""[P3-HIST-FAST-OPEN · 2026-05-18] Optimistic open del modal del Historial.

Síntoma del usuario:
> "quiero que al darle para entrar a ver las especificaciones del historial
>  sea mas fluido ya que siento delay"

Antes: el `onClick` de la card hacía `await _loadPlanDataLazy(plan)` ANTES
de `setSelectedPlan(...)`. La modal no abría hasta que el roundtrip a
Supabase (`select plan_data from meal_plans where id=...`) resolvía —
típicamente 200-500ms en red doméstica, peor en 3G/4G. Click → 300ms de
"nada visible" → modal aparece con contenido completo. Sensación: app
laggy.

Después: el modal abre AL INSTANTE con el summary del listado (calories/
macros/name/created_at top-level del `/history-list`). El plan_data se
carga en paralelo y se enchufa via `setSelectedPlan(prev => ...)` cuando
llega. Mientras tanto, un skeleton con shimmer ocupa el slot del menú
imitando el layout final (3 tabs + 4 meal cards). Swap fluido cuando
llega el data — no hay flash de "vacío → poblado".

Defensas:
  - Race conditions: el `.then` ignora la respuesta si el usuario cerró
    el modal o abrió OTRO plan antes de resolverse (`prev.id !== plan.id`).
  - prefers-reduced-motion: shimmer desactivado para usuarios con
    motion sensitivity (a11y).
  - Plan con plan_data inline (tests legacy / paths que pasan rows
    completos): NO se dispara el fetch, modal abre con data ya completa.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_HISTORY_JSX = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.jsx"
).read_text(encoding="utf-8")
_HISTORY_CSS = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "History.module.css"
).read_text(encoding="utf-8")


def test_marker_present_in_source():
    """Marker P3-HIST-FAST-OPEN permanece en JSX + CSS como anchor de
    regresión. NO miramos `_LAST_KNOWN_PFIX` en app.py — ese campo
    rota a cada P-fix nuevo."""
    assert "P3-HIST-FAST-OPEN" in _HISTORY_JSX, (
        "Marker P3-HIST-FAST-OPEN ausente en History.jsx — un refactor "
        "podría volver al onClick síncrono bloqueante sin dejar trazo."
    )
    assert "P3-HIST-FAST-OPEN" in _HISTORY_CSS, (
        "Marker P3-HIST-FAST-OPEN ausente en History.module.css — los "
        "estilos del skeleton perdieron su anchor."
    )


def test_onclick_is_synchronous_not_async():
    """El handler `onClick` de la card NO debe ser `async` — si lo es,
    React no procesará `setSelectedPlan` hasta que el `await` resuelva,
    revirtiendo a comportamiento bloqueante."""
    # Buscamos el patrón canónico `onClick={() => {` precediendo
    # un bloque que tenga setSelectedPlan + setActiveModalTab.
    # NO debe haber `onClick={async () => {` (forma anti-patrón).
    # Aceptamos `onClick={async` en OTROS handlers (no en la card del
    # listado). Anchor: el setActiveChunkIdx(0) que es único de este
    # handler del listado del Historial.
    idx = _HISTORY_JSX.find("setActiveChunkIdx(0);")
    assert idx > 0, "Anchor `setActiveChunkIdx(0)` no encontrado"
    # Buscamos el `onClick` ANTES de este setActiveChunkIdx (en los
    # 2000 chars previos — el callback es grande).
    block_before = _HISTORY_JSX[max(0, idx - 2000):idx]
    last_onclick = block_before.rfind("onClick={")
    assert last_onclick > 0, "onClick handler del card no encontrado"
    handler_decl = block_before[last_onclick:last_onclick + 60]
    assert "onClick={() =>" in handler_decl, (
        "El handler `onClick` de la card del Historial debería ser "
        "síncrono (`() => {`), no async (`async () => {`). Si es async, "
        "React no flusheará `setSelectedPlan` hasta que el await del "
        "fetch resuelva — el optimistic open queda anulado."
    )


def test_setSelectedPlan_called_before_loadPlanDataLazy():
    """`setSelectedPlan({...plan, plan_data: ...})` debe ejecutarse
    ANTES de `_loadPlanDataLazy(plan).then(...)`. Sin esto, el orden
    se invierte y el modal NO abre hasta que el fetch resuelva
    (bloqueo perceptible)."""
    # Anchors únicos del handler del listado:
    set_idx = _HISTORY_JSX.find("setSelectedPlan({\n                                            ...plan,\n                                            plan_data: _hasInlinePlanData ? plan.plan_data : null,")
    assert set_idx > 0, (
        "Llamada optimista `setSelectedPlan({...plan, plan_data: ...})` "
        "no encontrada en el handler. El fix se perdió."
    )
    load_idx = _HISTORY_JSX.find("_loadPlanDataLazy(plan).then((fullPlanData)")
    assert load_idx > 0, (
        "`_loadPlanDataLazy(plan).then(...)` no encontrado. El fetch "
        "paralelo se eliminó — el modal queda con plan_data=null para "
        "siempre."
    )
    assert set_idx < load_idx, (
        "`setSelectedPlan` (apertura optimista) debe ejecutarse ANTES "
        "de `_loadPlanDataLazy(...).then`. Orden invertido revierte el "
        "comportamiento bloqueante."
    )


def test_race_guard_in_then_callback():
    """El callback `.then` que enchufa `plan_data` DEBE proteger
    contra race conditions: si el usuario cerró el modal o abrió
    OTRO plan antes de resolverse, NO debemos pisar el state."""
    assert "prev.id !== plan.id" in _HISTORY_JSX, (
        "Race guard `prev.id !== plan.id` ausente en el callback "
        ".then. Riesgo: abrir plan A → cerrar → abrir plan B → "
        "tardía del A inyecta plan_data del A sobre el state del B "
        "(meals incorrectos visibles)."
    )


def test_planDataLoading_state_declared():
    """`useState(false)` para `planDataLoading` declarado al top del
    componente. Sin él, no hay forma de gatear el skeleton."""
    assert "const [planDataLoading, setPlanDataLoading] = useState(false);" in _HISTORY_JSX, (
        "State `planDataLoading` ausente. El skeleton no tiene fuente "
        "de verdad para saber cuándo mostrarse."
    )
    # Y debe setearse a true en el fetch path.
    assert "setPlanDataLoading(true)" in _HISTORY_JSX, (
        "setPlanDataLoading(true) ausente del handler — el skeleton "
        "nunca se muestra."
    )
    assert "setPlanDataLoading(false)" in _HISTORY_JSX, (
        "setPlanDataLoading(false) ausente — el skeleton queda "
        "perpetuamente visible después del primer fetch."
    )


def test_skeleton_jsx_rendered_conditionally():
    """El skeleton solo se renderiza cuando (a) está cargando Y (b) no
    hay days en plan_data. Sin la 2da condición, sería redundante con
    el contenido real durante el ms entre setSelectedPlan(full) y el
    setPlanDataLoading(false)."""
    # El gate del JSX combina ambos.
    assert "planDataLoading && !(selectedPlan.plan_data" in _HISTORY_JSX, (
        "Gate del skeleton incorrecto — debe ser `planDataLoading && "
        "!(selectedPlan.plan_data && Array.isArray(...) && length > 0)`. "
        "Si el gate es solo `planDataLoading`, hay flash visual al "
        "completarse el fetch."
    )
    # Y referencia las clases CSS del skeleton.
    assert "styles.menuSkeleton" in _HISTORY_JSX, (
        "Clase `styles.menuSkeleton` no se usa en JSX — el render del "
        "skeleton se eliminó."
    )
    assert "styles.menuSkeletonTab" in _HISTORY_JSX, (
        "Clase `styles.menuSkeletonTab` no se usa en JSX."
    )
    assert "styles.menuSkeletonMeal" in _HISTORY_JSX, (
        "Clase `styles.menuSkeletonMeal` no se usa en JSX."
    )


def test_skeleton_css_classes_defined():
    """Las 9 clases del skeleton DEBEN existir en History.module.css."""
    required = [
        ".menuSkeleton {",
        ".menuSkeletonTabs {",
        ".menuSkeletonTab {",
        ".menuSkeletonTabActive {",
        ".menuSkeletonMeals {",
        ".menuSkeletonMeal {",
        ".menuSkeletonMealIcon {",
        ".menuSkeletonMealText {",
        ".menuSkeletonMealLine {",
        ".menuSkeletonMealKcal {",
    ]
    for sel in required:
        assert sel in _HISTORY_CSS, (
            f"Selector CSS `{sel.rstrip(' {')}` ausente — el skeleton "
            "renderea sin estilo (cajas blancas sin shimmer)."
        )


def test_skeleton_respects_reduced_motion():
    """A11y: usuarios con `prefers-reduced-motion: reduce` no deben ver
    el shimmer. Media query debe desactivar `animation` en los 4
    elementos animados."""
    assert "@media (prefers-reduced-motion: reduce)" in _HISTORY_CSS, (
        "@media prefers-reduced-motion ausente del skeleton. Usuarios "
        "con motion sensitivity verán shimmer no-desactivable."
    )
    # El bloque debe afectar las 4 clases animadas.
    pr_idx = _HISTORY_CSS.find("@media (prefers-reduced-motion: reduce)")
    block = _HISTORY_CSS[pr_idx:pr_idx + 600]
    for cls in ["menuSkeletonTab", "menuSkeletonMealIcon", "menuSkeletonMealLine", "menuSkeletonMealKcal"]:
        assert cls in block, (
            f"@media prefers-reduced-motion no afecta `.{cls}` — "
            "su shimmer seguirá activo en a11y mode."
        )


def test_shimmer_keyframes_defined():
    """`@keyframes menuSkeletonShimmer` DEBE existir para que la
    animación tenga substancia. Sin keyframes, `animation: ...` es
    no-op silencioso."""
    assert "@keyframes menuSkeletonShimmer" in _HISTORY_CSS, (
        "@keyframes menuSkeletonShimmer ausente — la animación `animation: "
        "menuSkeletonShimmer 1.4s ...` referencia un nombre no-existente "
        "y el skeleton queda estático (caja gris fija)."
    )
