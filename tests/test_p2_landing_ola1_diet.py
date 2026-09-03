"""[P2-LANDING-OLA1-DIET + P2-LANDING-ENTRY-APP-CODE · 2026-08-14] Lo que el
visitante anónimo descarga ANTES de ver nada.

LOS 181 ICONOS. `manualChunks` metía `lucide-react` entero en `vendor-ui`, y un
vendor chunk NOMBRADO recibe `<link rel=modulepreload>` eager de Vite en TODAS
las rutas — la misma mecánica que P2-NEON-LAZY y P1-PERF-FRAMER-SPLIT ya dejaron
documentada. Medido: el chunk eran 95.846 B y su sourcemap contenía EXACTAMENTE
181 módulos de icono (49.996 B), entre ellos `refrigerator`, `syringe`,
`stethoscope`, `shrimp`, `chef-hat`, `microscope` — pantallas que el apex ni
siquiera puede alcanzar. El chrome eager usa 18; el landing entero, ~25.

Medido antes/después con el mismo método (`npm run build` + gzip -9 sobre lo que
el `<head>` compilado pide de entrada):

    ola 1   196.176 → 180.476 B gzip   (−15.700, −8,0%)
    render  262.767 → 248.840 B gzip   (−13.927, −5,3%)

⚠️ EL COSTE QUE ESTE REPO YA PAGÓ UNA VEZ, y que obliga a medir en vez de asumir:
P2-VENDOR-REACT-CLIENT movió `react-dom/client` a un vendor chunk PRECISAMENTE
porque el entry re-hashea en cada deploy. Los 18 iconos del chrome que ahora caen
en el entry se re-descargan en cada release. El criterio que se fijó antes de
tocar nada era «el entry no puede subir más de 5 kB gz»: subió 3.322 B.

LO QUE **NO** SE HIZO, y por qué. El plan también proponía sacar
`@tanstack/react-query` del entry (26.691 B) cortando la arista
`AssessmentContext` → `quotaCache` → `queryClient`. No se tocó: el test
`queryClient.test.js` **ancla deliberadamente el import ESTÁTICO** de
`clearUserQueryCache` en `AssessmentContext`, con el comentario «si un refactor
lo quita, la fuga PII cross-user vuelve» (P1-3). Convertirlo en dinámico haría
asíncrona la purga de caché del logout, que es justo la operación que cierra esa
clase de fuga. Eso necesita su propia spec, no un paso de una dieta de bytes.

Tooltip-anchor: P2-LANDING-OLA1-DIET
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND = _REPO_ROOT / "frontend"
_VITE = _FRONTEND / "vite.config.js"
_APP = _FRONTEND / "src" / "App.jsx"
_MAIN = _FRONTEND / "src" / "main.jsx"
_DIST = _FRONTEND / "dist"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P2-LANDING-OLA1-DIET] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


def _sin_comentarios(t: str) -> str:
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", t, flags=re.MULTILINE)


def _vendor_chunks() -> dict[str, str]:
    """Los arrays `'vendor-x': [...]` de `manualChunks`, leídos por línea.

    ⚠️ Sin quitar comentarios, y a propósito. El intento anterior los quitaba con
    un `/\\*.*?\\*/`, y `vite.config.js` contiene
    `globPatterns: ['**/*.{js,css,html,ico,png,svg}']`: ese `/*` vive DENTRO de
    una cadena, así que el stripper abría un comentario ahí y se comía medio
    fichero, incluido `manualChunks`. El guard no encontraba nada y fallaba
    contra código correcto.

    Mirar sólo líneas que EMPIEZAN por `'vendor-` esquiva la clase entera: un
    comentario empieza por `//`, nunca por una comilla.
    """
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^\s*'(vendor-[^']+)':\s*(\[[^\]]*\])", _read(_VITE), re.MULTILINE)
    }


def test_lucide_no_vuelve_a_un_vendor_chunk_nombrado():
    chunks = _vendor_chunks()
    assert chunks, "[P2-LANDING-OLA1-DIET] No se encontró ningún `vendor-*` en vite.config.js."
    culpables = [n for n, arr in chunks.items() if "lucide-react" in arr]
    assert not culpables, (
        f"[P2-LANDING-OLA1-DIET] `lucide-react` volvió a un vendor chunk nombrado: {culpables}.\n"
        "Un vendor chunk NOMBRADO recibe `modulepreload` eager en todas las rutas "
        "(P2-NEON-LAZY, P1-PERF-FRAMER-SPLIT), así que eso devuelve los 181 iconos "
        "—incluidos `syringe`, `microscope` y `refrigerator`— al arranque de una "
        "landing que usa ~25."
    )


def test_sonner_sigue_en_el_vendor_eager():
    """Sonner SÍ es eager de verdad: sin `<Toaster/>` no hay capa de avisos."""
    assert "sonner" in _vendor_chunks().get("vendor-ui", ""), (
        "[P2-LANDING-OLA1-DIET] `sonner` salió de `vendor-ui`. Es la única de las "
        "dos que se importa de verdad en el arranque (el `<Toaster/>` de App): "
        "dejarla sin chunk estable la manda al entry, que re-hashea en cada deploy."
    )


def test_el_recovery_de_pipeline_no_se_monta_en_el_apex():
    app = _sin_comentarios(_read(_APP))
    assert "IS_APEX_HOST && (" in app or "!IS_APEX_HOST &&" in app, (
        "[P2-LANDING-ENTRY-APP-CODE] `PendingPipelineRecovery` volvió a montarse "
        "incondicionalmente. Es headless y vigila generaciones pendientes: en el "
        "apex no hay sesión (P3-APEX-NO-SESSION), así que no puede haber nada que "
        "recuperar."
    )
    assert "lazy(() => import('./components/PendingPipelineRecovery'))" in app, (
        "[P2-LANDING-ENTRY-APP-CODE] Volvió el import ESTÁTICO de "
        "`PendingPipelineRecovery`: gatear el render no saca el módulo del bundle, "
        "sólo evita que se ejecute. El peso se va con el `lazy`, no con el `if`."
    )


def test_el_lazy_headless_tiene_su_boundary():
    """Sin `<Suspense>` ancestro, un `lazy` que suspende tumba el render entero."""
    app = _read(_APP)
    i = app.find("<PendingPipelineRecovery />")
    assert i != -1, "[P2-LANDING-ENTRY-APP-CODE] Desapareció `PendingPipelineRecovery`."
    ventana = app[max(0, i - 400): i]
    assert "Suspense" in ventana, (
        "[P2-LANDING-ENTRY-APP-CODE] `PendingPipelineRecovery` es `lazy` pero no "
        "tiene un `<Suspense>` cerca. Este nodo cuelga del árbol de `App`, FUERA "
        "del Suspense de las rutas (que vive en `AnimatedLayout`), así que sin "
        "boundary propio el primer render que suspenda revienta la app."
    )


def test_el_listener_de_push_no_carga_en_el_landing():
    main = _sin_comentarios(_read(_MAIN))
    assert "import('./utils/pushNotifications')" in main, (
        "[P2-LANDING-ENTRY-APP-CODE] `pushNotifications` volvió a importarse de "
        "forma estática. El listener sólo sirve donde puede haber una suscripción "
        "push que rotar, y en el apex nunca la hay."
    )
    assert re.search(r"isMarketingVisit\(\)[\s\S]{0,120}pushNotifications", main), (
        "[P2-LANDING-ENTRY-APP-CODE] La carga de `pushNotifications` dejó de "
        "gatearse por host."
    )


@pytest.mark.skipif(not (_DIST / "index.html").exists(), reason="sin dist/ (corre `npm run build`)")
def test_el_head_compilado_no_precarga_un_chunk_de_iconos():
    """La comprobación contra el artefacto: qué pide de verdad la ola 1."""
    html = _read(_DIST / "index.html")
    # El bloque gateado por host es la ola 2; aquí sólo miramos la ola 1.
    bloque = re.search(r"var esApex.*?\}\)\(\);", html, re.DOTALL)
    ola1 = html.replace(bloque.group(0), "") if bloque else html
    preloads = re.findall(r'(?:src|href)="/(assets/[^"]+\.js)"', ola1)
    gordos = []
    for rel in set(preloads):
        f = _DIST / rel
        if f.exists() and "vendor-ui" in rel and f.stat().st_size > 70_000:
            gordos.append((rel, f.stat().st_size))
    assert not gordos, (
        f"[P2-LANDING-OLA1-DIET] `vendor-ui` volvió a engordar en la ola 1: {gordos}.\n"
        "Antes del recorte eran 95.846 B por llevar los 181 iconos; ahora debería "
        "rondar los 46.000 (sólo sonner y lo que Rollup agrupe con él)."
    )
