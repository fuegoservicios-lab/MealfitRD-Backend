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

import re
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
    """[reapuntado 2026-07-28] El handler migró del onClick inline del monolito a
    `openPlanModal`, que los paneles reciben como prop. La invariante es la misma: SÍNCRONO
    (sin `async`) para que el modal abra en el mismo frame."""
    m = re.search(r"const\s+openPlanModal\s*=\s*(async\s+)?\(", _HISTORY_JSX)
    assert m, "openPlanModal no encontrado en History.jsx"
    assert not m.group(1), (
        "openPlanModal se volvió async — el modal esperaría al fetch para abrir (el bug original)"
    )

def test_setSelectedPlan_called_before_loadPlanDataLazy():
    """[reapuntado 2026-07-28] Mismo contrato, nombres nuevos: dentro de `openPlanModal`, el
    `setSelectedPlan({ ...plan, plan_data: ... })` optimista va ANTES del
    `_loadPlanDataLazy(plan).then(...)` — el usuario ve el modal al instante y los datos llegan
    después."""
    i = _HISTORY_JSX.index("const openPlanModal")
    fin = _HISTORY_JSX.find("};", i)
    cuerpo = _HISTORY_JSX[i:fin]
    a = cuerpo.find("setSelectedPlan({ ...plan")
    b = cuerpo.find("_loadPlanDataLazy(plan)")
    assert a > 0, "la llamada optimista setSelectedPlan({...plan, ...}) desapareció del handler"
    assert b > 0, "el lazy-load desapareció del handler"
    assert a < b, "el optimista debe ir ANTES del lazy-load — si no, vuelve el modal en blanco"

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
    """[reapuntado 2026-07-28] El bloque @media (prefers-reduced-motion) creció a un selector
    compuesto multi-línea (`.menuSkeletonTab,
 .menuSkeletonMealIcon,…`) y el regex de una
    línea dejó de verlo. Se verifica estructuralmente: dentro de ALGÚN bloque reduced-motion
    aparece .menuSkeletonTab."""
    bloques = re.findall(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(.*?)" + chr(10) + r"\}",
                         _HISTORY_CSS, re.DOTALL)
    assert bloques, "History.module.css perdió los bloques prefers-reduced-motion"
    assert any(".menuSkeletonTab" in b for b in bloques), (
        "el shimmer del skeleton no respeta reduced-motion — a11y roto para vestibular"
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
