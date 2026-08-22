"""[P1-IOS-CODEMAGIC · 2026-08-22] Fase 2 de P1-IOS-NATIVE-SHELL: el binario nace en
Codemagic y llega a TestFlight sin una Mac.

Parser-based sobre `frontend/codemagic.yaml`, `frontend/package.json` y
`frontend/.env.native`. Ancla las decisiones del diseño aprobado el 2026-08-22, cada
una con la razón que la hace no-obvia:

  1. BUILD WEB PROPIO PARA EL BINARIO. `npm run build` arrastra un `postbuild` para
     nginx (18 HTML por ruta, `.br`, sitemap, precache) que dentro del WebView sobra.
     `build:native` es `vite build` a secas.
  2. `.env.native` con `VITE_API_BASE_URL` ABSOLUTA: en `capacitor://localhost` una
     ruta relativa `/api` apunta a la nada. Y SIN claves PayPal: el gate de plataforma
     ya esconde el comercio; no tener los IDs en el binario es defensa en profundidad
     (Apple 3.1.1).
  3. FIRMA CON API KEY de App Store Connect, no con certificados .p12 a mano: sin Mac
     es el único modo razonable, y Codemagic renueva perfiles sola.
  4. `cap sync` EN LA MAC, siempre. El `Package.swift` commiteado lo generó `cap sync`
     en Windows y trae rutas con BACKSLASH (`..\..\..\node_modules\@capacitor\camera`)
     que macOS no resuelve; y no hay scheme compartido en `xcshareddata`. Confiar en
     lo commiteado tumba el primer build.
  5. DISPARO MANUAL: cada build consume minutos de Mac (500/mes gratis). Un build por
     commit del frontend se los come en una semana.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_FRONT = Path(__file__).resolve().parent.parent.parent / "frontend"
_YAML = _FRONT / "codemagic.yaml"
_PKG = _FRONT / "package.json"
_ENV_NATIVE = _FRONT / ".env.native"
_PACKAGE_SWIFT = _FRONT / "ios" / "App" / "CapApp-SPM" / "Package.swift"


def _yaml() -> str:
    assert _YAML.exists(), "Falta frontend/codemagic.yaml (P1-IOS-CODEMAGIC)."
    return _YAML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. build:native sin postbuild
# ---------------------------------------------------------------------------
def test_build_native_es_vite_a_secas():
    scripts = json.loads(_PKG.read_text(encoding="utf-8"))["scripts"]
    assert "build:native" in scripts, "Falta el script `build:native` en package.json."
    cmd = scripts["build:native"]
    assert "vite build" in cmd, f"`build:native` debe invocar `vite build`, no {cmd!r}."
    for prohibido in ("postbuild", "precomprimir", "build-route-meta", "precache-guard", "build-sitemap"):
        assert prohibido not in cmd, (
            f"`build:native` no debe correr {prohibido}: es tooling de nginx, dentro del WebView sobra."
        )
    assert "--mode native" in cmd, (
        "`build:native` debe pasar `--mode native` para que Vite cargue `.env.native`."
    )


def test_build_native_no_dispara_el_prebuild_de_nginx():
    """npm ejecuta `pre<script>` y `post<script>` por NOMBRE: `prebuild`/`postbuild`
    se enganchan a `build`, NO a `build:native`. Este test fija que nadie añada un
    `prebuild:native`/`postbuild:native` que reintroduzca el tooling de nginx."""
    scripts = json.loads(_PKG.read_text(encoding="utf-8"))["scripts"]
    assert "prebuild:native" not in scripts and "postbuild:native" not in scripts


# ---------------------------------------------------------------------------
# 2. .env.native
# ---------------------------------------------------------------------------
def test_env_native_api_base_absoluta_y_sin_paypal():
    assert _ENV_NATIVE.exists(), "Falta frontend/.env.native."
    txt = _ENV_NATIVE.read_text(encoding="utf-8")
    m = re.search(r"^VITE_API_BASE_URL=(\S+)", txt, flags=re.M)
    assert m, "`.env.native` debe definir VITE_API_BASE_URL (en capacitor:// `/api` relativo apunta a la nada)."
    assert m.group(1).startswith("https://"), f"VITE_API_BASE_URL debe ser absoluta https, no {m.group(1)!r}."
    # Las vars de la pasarela deben estar VACÍAS, no ausentes: Vite carga `.env`
    # (dev, ignorado por git, CON los IDs) antes que `.env.native`, y medido el
    # 2026-08-22 el bundle nativo local los llevaba dentro. Un valor vacío pisa al de
    # dev; una ausencia confiaría en que `.env` no exista en la Mac.
    paypal = re.findall(r"^(VITE_PAYPAL_[A-Z_]+)=(.*)$", txt, flags=re.M)
    assert paypal, "`.env.native` debe VACIAR explícitamente las VITE_PAYPAL_* (pisan al .env de dev)."
    con_valor = [k for k, v in paypal if v.strip()]
    assert not con_valor, (
        f"`.env.native` lleva valores de la pasarela {con_valor}: la app nativa no tiene comercio (Apple 3.1.1)."
    )
    # Y TODAS las que .env.production define, no solo algunas: una que falte hereda el ID de dev.
    prod = (_FRONT / ".env.production").read_text(encoding="utf-8")
    en_prod = set(re.findall(r"^(VITE_PAYPAL_[A-Z_]+)=", prod, flags=re.M))
    faltan = en_prod - {k for k, _ in paypal}
    assert not faltan, f"`.env.native` no vacía {sorted(faltan)}: heredarían el valor del .env de dev."


def test_env_native_conserva_lo_que_la_app_necesita_para_funcionar():
    """Sin estas la app arranca y no hace nada: auth, errores y push."""
    txt = _ENV_NATIVE.read_text(encoding="utf-8")
    for k in ("VITE_NEON_AUTH_URL", "VITE_SENTRY_DSN", "VITE_VAPID_PUBLIC_KEY"):
        assert re.search(rf"^{k}=\S+", txt, flags=re.M), f"`.env.native` debe definir {k}."


# ---------------------------------------------------------------------------
# 3-5. codemagic.yaml
# ---------------------------------------------------------------------------
def test_yaml_firma_con_api_key_no_con_certificados_a_mano():
    y = _yaml()
    assert "app_store_connect" in y, "La firma debe ir por integración App Store Connect (API key)."
    assert "xcode-project use-profiles" in y, "Falta `xcode-project use-profiles` (aplica los perfiles que bajó la API key)."
    assert "app-store-connect fetch-signing-files" in y, "Falta `fetch-signing-files` con `--create`."
    assert "--create" in y, "`fetch-signing-files --create` para que Codemagic genere cert+perfil sin Mac."
    for prohibido in ("certificate_private_key", ".p12", "CM_CERTIFICATE"):
        assert prohibido not in y, f"No subir certificados a mano ({prohibido}): la API key los gestiona."


def test_yaml_sube_a_testflight():
    y = _yaml()
    assert re.search(r"publishing:\s*\n(?:.*\n)*?\s+app_store_connect:", y), "Falta `publishing.app_store_connect`."
    assert "submit_to_testflight: true" in y


def test_yaml_corre_cap_sync_en_la_mac():
    """El Package.swift commiteado tiene backslashes de Windows; hay que regenerarlo."""
    y = _yaml()
    assert re.search(r"npx cap sync ios", y), "Falta `npx cap sync ios` en la Mac."
    idx_sync = y.index("npx cap sync ios")
    idx_build = y.index("xcode-project build-ipa")
    assert idx_sync < idx_build, "`cap sync` debe ir ANTES de `build-ipa`."
    idx_web = y.index("build:native")
    assert idx_web < idx_sync, "`build:native` debe ir ANTES de `cap sync` (sync copia dist/ al proyecto)."


def test_package_swift_commiteado_tiene_backslashes_y_por_eso_se_regenera():
    """Documenta la razón de `cap sync` en CI. Si un día se commitea un Package.swift
    limpio este test deja de aplicar — quitarlo entonces, no antes."""
    txt = _PACKAGE_SWIFT.read_text(encoding="utf-8")
    assert "\\" in txt, (
        "El Package.swift ya no tiene backslashes: la razón de regenerarlo en CI cambió; revisa este test."
    )


def test_yaml_disparo_manual_y_build_number_automatico():
    y = _yaml()
    assert not re.search(r"^\s*triggering:", y, flags=re.M), (
        "Sin `triggering`: el build de iOS se lanza a mano (minutos de Mac limitados)."
    )
    assert "app-store-connect get-latest-testflight-build-number" in y or "get-latest-app-store-build-number" in y, (
        "El build number debe salir de App Store Connect (+1), no de una constante."
    )


def test_yaml_bundle_id_y_plataforma():
    y = _yaml()
    assert "com.bioboros.app" in y, "El bundle id del YAML debe ser el de capacitor.config.ts."
    assert re.search(r"instance_type:\s*mac_mini_m\d", y), "Instancia Mac Apple Silicon."
    assert re.search(r"node:\s*22", y), "Node 22 (el que usa el repo)."
    assert "ios/App/App.xcworkspace" in y or "ios/App/App.xcodeproj" in y
