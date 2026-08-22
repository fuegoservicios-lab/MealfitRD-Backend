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
    # [Build #3, 2026-08-22] «Cannot save Signing Certificates without certificate
    # private key». Para CREAR un certificado, Codemagic necesita una clave RSA con la
    # que generar la solicitud (CSR): la API key de Apple autoriza, no firma. Esa clave
    # va en la env var CERTIFICATE_PRIVATE_KEY (secreta, grupo `default`). Yo la había
    # PROHIBIDO aquí confundiéndola con un .p12 subido a mano: no lo es, es la semilla
    # con la que Codemagic genera los certificados. Lo que sigue prohibido es el .p12.
    for prohibido in (".p12", "CM_CERTIFICATE"):
        assert prohibido not in y, f"No subir certificados a mano ({prohibido}): la API key los gestiona."
    assert "CERTIFICATE_PRIVATE_KEY" in y, (
        "`fetch-signing-files --create` necesita CERTIFICATE_PRIVATE_KEY (clave RSA) para "
        "generar el CSR; sin ella falla con «Cannot save Signing Certificates»."
    )


def test_yaml_no_mezcla_firma_declarativa_con_fetch_create():
    """[Build #1 en la Mac, 2026-08-22] «No matching profiles found for bundle
    identifier com.bioboros.app and distribution type app_store».

    `environment.ios_signing` es la firma AUTOMÁTICA de Codemagic: corre ANTES de
    cualquier script y solo BUSCA perfiles existentes. `fetch-signing-files --create`
    (el paso que los CREA) viene después y nunca llegó a ejecutarse. Son dos
    mecanismos para lo mismo y se pisan: en la primera subida no hay perfil que
    buscar. Se conserva el que crea; el declarativo no puede volver."""
    y = _yaml()
    assert not re.search(r"^\s+ios_signing:", y, flags=re.M), (
        "`environment.ios_signing` busca perfiles antes de los scripts y falla en la "
        "primera subida; la firma va SOLO por `fetch-signing-files --create`."
    )
    assert "app-store-connect fetch-signing-files" in y


def test_script_de_firma_aborta_al_primer_fallo():
    """[Build #3] `fetch-signing-files` falló en la línea 0 y, sin `set -e`, los tres
    comandos siguientes corrieron con las manos vacías y el paso salió en VERDE. El
    fallo real apareció dos pasos después, en Xcode, disfrazado de otra cosa."""
    y = _yaml()
    # Anclado al COMANDO (`app-store-connect fetch-signing-files`), no a la palabra:
    # el comentario del paso también la menciona, antes del `script: |`, y buscando
    # hacia atrás desde ahí el bloque salía vacío.
    i = y.index("app-store-connect fetch-signing-files")
    bloque = y[y.rfind("script: |", 0, i):i]
    assert "set -e" in bloque, "El script de firma debe llevar `set -e`: un fetch fallido no puede salir en verde."


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


def test_yaml_compila_el_xcodeproj_no_un_workspace_que_no_existe():
    """[Build #2 en la Mac, 2026-08-22] `Path "ios/App/App.xcworkspace" does not exist`.

    Capacitor 8 con Swift Package Manager NO genera .xcworkspace (eso es CocoaPods):
    solo existe `ios/App/App.xcodeproj`. Estaba delante en el repo y asumí el layout
    de Pods. `build-ipa` y `use-profiles` van con `--project`, nunca `--workspace`."""
    y = _yaml()
    # Fuera de comentarios: el YAML explica que el workspace NO existe, y esa prosa
    # no puede contar como uso. Se miran solo las líneas de código.
    codigo = "\n".join(l for l in y.splitlines() if not l.lstrip().startswith("#"))
    assert "App.xcworkspace" not in codigo, (
        "No hay workspace: el proyecto usa SPM. `ios/App/App.xcworkspace` no existe."
    )
    assert "--workspace" not in codigo, "`build-ipa`/`use-profiles` no pueden usar `--workspace`."
    # El path va por variable: `XCODE_PROJECT` definida Y usada en build-ipa.
    assert re.search(r'^\s+XCODE_PROJECT:\s*"?ios/App/App\.xcodeproj"?\s*$', codigo, flags=re.M), (
        "Falta `XCODE_PROJECT: ios/App/App.xcodeproj` en environment.vars."
    )
    assert re.search(r'build-ipa[\s\S]{0,120}--project\s+"?\$XCODE_PROJECT', codigo), (
        "`build-ipa` debe ir con `--project \"$XCODE_PROJECT\"`."
    )
    from pathlib import Path as _P
    assert (_FRONT / "ios" / "App" / "App.xcodeproj").is_dir()
    assert not (_FRONT / "ios" / "App" / "App.xcworkspace").exists()


def test_scheme_app_esta_compartido_para_xcodebuild():
    """[Antes del build #3, 2026-08-22] `xcodebuild` desde CLI solo ve schemes
    COMPARTIDOS (`xcshareddata/xcschemes/App.xcscheme`). El proyecto de Capacitor
    no lo tra\u00eda y `cap sync` NO lo genera (mi comentario en el YAML dec\u00eda que s\u00ed:
    era falso, medido en node_modules/@capacitor/cli). Sin \u00e9l, `build-ipa --scheme App`
    falla con \u00abscheme not found\u00bb aunque el proyecto compile en Xcode."""
    scheme = _FRONT / "ios" / "App" / "App.xcodeproj" / "xcshareddata" / "xcschemes" / "App.xcscheme"
    assert scheme.is_file(), f"Falta el scheme compartido: {scheme}"
    xml = scheme.read_text(encoding="utf-8")
    assert 'BuildableName = "App.app"' in xml
    assert 'BlueprintName = "App"' in xml
    # El id del target tiene que ser el del pbxproj, o Xcode lo descarta en silencio.
    pbx = (_FRONT / "ios" / "App" / "App.xcodeproj" / "project.pbxproj").read_text(encoding="utf-8")
    m = re.search(r"^\s+([0-9A-F]{24}) /\* App \*/ = \{\s*\n\s+isa = PBXNativeTarget;", pbx, flags=re.M)
    assert m, "No se encontr\u00f3 el PBXNativeTarget App en el pbxproj."
    assert f'BlueprintIdentifier = "{m.group(1)}"' in xml, (
        f"El scheme apunta a otro target: esperaba BlueprintIdentifier {m.group(1)}."
    )
    # Y no puede estar ignorado por git: viaja en el repo o la Mac no lo tiene.
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", str(scheme)], cwd=str(_FRONT), capture_output=True)
    assert r.returncode != 0, "El scheme est\u00e1 ignorado por git: no llegar\u00eda a la Mac."
