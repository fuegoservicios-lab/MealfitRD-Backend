"""[P1-IOS-NATIVE-SHELL · 2026-08-21] Bioboros en la App Store sin comercio dentro de la app.

Apple (guidelines 3.1.1 / 3.1.3(b)) prohíbe que una app venda o ENLACE a compras
externas. El pago es sólo web (PayPal), así que la build nativa (Capacitor) no puede
mostrar ni precios, ni «Mejorar plan», ni PayPal, ni el landing: sólo REFLEJA el tier.

Este test ancla lo que vive FUERA de `frontend/src` (la hermana vitest
`NativeShell.contract.test.jsx` cubre los componentes):

  A. El wrapper existe y declara el bundle ID (irreversible tras la primera subida).
  B. `Info.plist` trae las descripciones de uso de cámara/fotos (sin ellas, rechazo
     automático) y declara cifrado exento (evita la pregunta de export compliance).
  C. `ios/` está fuera de ESLint y de vitest: el job `quality` de Actions aborta en el
     paso 1 si eslint ve el proyecto Xcode (P1-CI-QUALITY-ABORTADO enseñó que un paso
     rojo al principio esconde los ocho siguientes).
  D. El gate de plataforma es UNO (`config/platform.js`) y es FUNCIÓN, no constante
     (la «trampa del const congelado»: evaluado al importar, no se puede mockear).
  E. La clave i18n del botón de Apple vive en los 4 catálogos.

Spec: docs/superpowers/specs/2026-08-21-ios-native-shell-design.md
"""
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONT = _REPO_ROOT / "frontend"
_MARKER = "P1-IOS-NATIVE-SHELL"


def _read(rel: str) -> str:
    return (_FRONT / rel).read_text(encoding="utf-8")


# ── A. Wrapper ──────────────────────────────────────────────────────────────────────────

def test_capacitor_config_declara_bundle_id_y_webdir():
    src = _read("capacitor.config.ts")
    assert _MARKER in src
    assert re.search(r"appId:\s*'com\.bioboros\.app'", src)
    assert re.search(r"appName:\s*'Bioboros'", src)
    assert re.search(r"webDir:\s*'dist'", src)


def test_el_proyecto_xcode_existe_y_esta_commiteable():
    assert (_FRONT / "ios" / "App" / "App.xcodeproj").is_dir()
    assert (_FRONT / "ios" / "App" / "App" / "Info.plist").is_file()
    # Capacitor ignora su copia de `dist` y los config generados: el repo no debe
    # llevar el bundle web dentro de ios/.
    gi = (_FRONT / "ios" / ".gitignore").read_text(encoding="utf-8")
    assert "App/App/public" in gi
    assert "App/App/capacitor.config.json" in gi


def test_los_plugins_nativos_estan_instalados():
    pkg = json.loads(_read("package.json"))
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    for name in ("@capacitor/core", "@capacitor/cli", "@capacitor/ios",
                 "@capacitor/camera", "@capacitor/push-notifications"):
        assert name in deps, name


# ── B. Info.plist ───────────────────────────────────────────────────────────────────────

def test_info_plist_trae_descripciones_de_uso_en_espanol():
    plist = _read("ios/App/App/Info.plist")
    for key in ("NSCameraUsageDescription", "NSPhotoLibraryUsageDescription",
                "NSPhotoLibraryAddUsageDescription"):
        m = re.search(rf"<key>{key}</key>\s*<string>([^<]+)</string>", plist)
        assert m, key
        assert "Bioboros" in m.group(1) and len(m.group(1)) > 30, key
    assert re.search(r"<key>ITSAppUsesNonExemptEncryption</key>\s*<false/>", plist)
    assert re.search(r"<key>CFBundleDevelopmentRegion</key>\s*<string>es</string>", plist)


# ── C. ios/ fuera de lint y tests ───────────────────────────────────────────────────────

def test_eslint_y_vitest_ignoran_el_proyecto_xcode():
    eslint = _read("eslint.config.js")
    m = re.search(r"globalIgnores\(\[([^\]]*)\]\)", eslint)
    assert m and "'ios'" in m.group(1), "ios/ debe estar en globalIgnores de eslint"
    vite = _read("vite.config.js")
    assert "'ios/**'" in vite, "ios/** debe estar en test.exclude de vitest"


# ── D. Un solo gate, función ────────────────────────────────────────────────────────────

def test_el_gate_de_plataforma_es_unico_y_es_funcion():
    src = _read("src/config/platform.js")
    assert _MARKER in src
    assert re.search(r"export function isNativeApp\(\)", src)
    assert re.search(r"export function nativeHidesCommerce\(\)", src)
    assert re.search(r"export function appleSignInEnabled\(\)", src)
    # NO constantes congeladas en ámbito de módulo.
    assert not re.search(r"^export const (isNativeApp|nativeHidesCommerce|appleSignInEnabled)\b", src, re.M)
    # Nadie más le pregunta a Capacitor.
    offenders = []
    for p in (_FRONT / "src").rglob("*"):
        if p.suffix not in (".js", ".jsx", ".ts", ".tsx") or "__tests__" in p.parts:
            continue
        if "isNativePlatform" in p.read_text(encoding="utf-8", errors="replace"):
            offenders.append(p.relative_to(_FRONT / "src").as_posix())
    assert offenders == ["config/platform.js"], offenders


def test_las_seis_superficies_importan_el_gate():
    surfaces = {
        "src/App.jsx": "from './config/platform'",
        "src/components/dashboard/DashboardLayout.jsx": "from '../../config/platform'",
        "src/components/dashboard/PaymentModal.jsx": "from '../../config/platform'",
        "src/pages/Settings.jsx": "from '../config/platform'",
        "src/pages/Login.jsx": "from '../config/platform'",
        "src/components/dashboard/AccountMenu.jsx": "typeof onViewPlans === 'function'",
    }
    for rel, needle in surfaces.items():
        src = _read(rel)
        assert needle in src, rel
        assert _MARKER in src, f"{rel} sin marker {_MARKER}"


# ── E. i18n ─────────────────────────────────────────────────────────────────────────────

def test_la_clave_del_boton_de_apple_vive_en_los_cuatro_catalogos():
    assert "t('Continuar con Apple')" in _read("src/pages/Login.jsx")
    for loc in ("en-US", "fr-FR", "it-IT", "pt-BR"):
        cat = json.loads(_read(f"src/i18n/locales/{loc}.json"))
        assert cat.get("Continuar con Apple"), loc


def test_webview_ios_sin_rubber_band_ni_inset_automatico():
    """[P1-IOS-WEBVIEW-SCROLL \u00b7 2026-08-22] Primer build en el iPhone: \u00abel scroll del login
    sobrepasa lo normal, como si se saliera de la pantalla\u00bb.

    Dos causas que se suman, medidas en el c\u00f3digo:
      - El WebView de Capacitor vive dentro de un UIScrollView NATIVO que rebota
        (rubber-band). `overscroll-behavior-y: none` en `body` no lo frena: eso es CSS
        del documento, y el que rebota es el contenedor nativo. `scrollEnabled: false`
        deja el scroll al contenido web, que s\u00ed obedece al CSS.
      - `contentInset: 'automatic'` suma el alto de la barra de estado al \u00e1rea
        desplazable; el login mide `min-height: 100dvh` y la p\u00e1gina acaba midiendo
        100dvh + inset: al llegar abajo \u00absobra\u00bb justo esa franja. `'never'` deja que el
        CSS gestione el notch con `env(safe-area-inset-*)`, que el login YA hace."""
    src = _read("capacitor.config.ts")
    assert re.search(r"scrollEnabled:\s*false", src), (
        "ios.scrollEnabled debe ser false: el UIScrollView nativo rebota y ning\u00fan CSS lo frena."
    )
    assert re.search(r"contentInset:\s*'never'", src), (
        "ios.contentInset debe ser 'never': 'automatic' suma la barra de estado a un login de 100dvh."
    )


def test_google_oauth_no_se_pinta_en_nativo_hasta_que_exista_deep_link():
    """[P1-IOS-OAUTH-GATE · 2026-08-22] Primer build en el iPhone: «Continuar con
    Google» salía a Safari y moría en «la dirección no es válida» — Neon Auth
    redirige a `capacitor://localhost/dashboard`. Hasta que exista el deep link
    (esquema propio + intercambio de código en backend), el botón NO se pinta en
    nativo. Gate PROPIO en platform.js: cuando llegue el deep link se apaga sólo
    éste, el comercio sigue escondido. Con solo correo en nativo, Apple 4.8
    (Sign in with Apple obligatorio si hay Google) no aplica.
    Espejo backend de NativeShell.contract.test.jsx (el gate de vitest)."""
    platform = _read("src/config/platform.js")
    assert "export function nativeHidesOAuthRedirect()" in platform
    login = _read("src/pages/Login.jsx")
    assert "nativeHidesOAuthRedirect" in login.split("\n", 5)[1], "Login debe importar el gate"
    i = login.index("t('Continuar con Google')")
    assert "!nativeHidesOAuthRedirect() && (" in login[i - 400:i]
