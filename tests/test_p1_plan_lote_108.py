# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-108 · 2026-09-19] Actualizaciones en vivo (OTA) de la app nativa.

El binario de iOS lleva la web empaquetada: cada arreglo exigía un build manual de Codemagic y, ya en la App Store,
la revisión de Apple. Ahora el despliegue del frontend publica además `dist/ota/<id>.zip` + `latest.json` y la app
lo descarga al abrirse y lo aplica en el siguiente arranque en frío. Este test ancla las piezas que, si alguien
las quita, dejan el sistema INERTE o PELIGROSO sin que nada más lo note; el contrato fino (la decisión, el zip, la
paridad productor↔consumidor) vive en `frontend/src/__tests__/lote108.test.js`."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_RAIZ = _BACKEND.parent
_FRONT = _RAIZ / "frontend"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"sin {p.name} al lado")
    return p.read_text(encoding="utf-8")


def _front(rel: str) -> str:
    return _leer(_FRONT / rel)


def test_la_red_de_seguridad_esta_puesta():
    """Sin `readyTimeout` un paquete roto deja la app en blanco para siempre: no hay vuelta atrás automática."""
    cap = _front("capacitor.config.ts")
    m = re.search(r"LiveUpdate:\s*\{\s*readyTimeout:\s*(\d+)", cap)
    assert m and int(m.group(1)) >= 5000
    # autoalojado: nada de Capawesome Cloud ni de su sync automático
    assert "autoUpdateStrategy" not in cap.split("LiveUpdate:")[1].split("}")[0]
    assert not re.search(r"^\s*server\s*:", cap, flags=re.M)


def test_la_app_confirma_tarde_no_recarga_y_no_descarga_de_fuera():
    lu = _front("src/native/liveUpdate.js")
    assert "export const OTA_BASE_URL = 'https://app.bioboros.com/ota/';" in lu
    assert "if (url !== `${OTA_BASE_URL}${bundleId}.zip`) return nada('url_ajena');" in lu
    assert "if (build < minNativeBuild) return nada('binario_antiguo');" in lu
    assert "if (bundleId <= ownId) return nada('al_dia');" in lu
    assert "window.addEventListener('mealfit:app-ready', confirmar, { once: true });" in lu
    assert "await LiveUpdate.downloadBundle({ bundleId, url, checksum });" in lu
    # decisión 1: arranque en frío. `reload()` solo puede aparecer en comentarios.
    codigo = "\n".join(l for l in lu.splitlines() if not l.lstrip().startswith(("//", "*", "/*")))
    assert ".reload(" not in codigo
    assert "if (_iniciado || !isNativeApp() || !otaOwnId()) return;" in lu
    main = _front("src/main.jsx")
    assert main.index("iniciarOtaNativa()\n") < main.index("createRoot(document.getElementById('root')).render(")


def test_el_paquete_conoce_su_id_y_el_despliegue_lo_publica():
    vite = _front("vite.config.js")
    assert "__OTA_BUNDLE_ID__: JSON.stringify(mode === 'native' ? (process.env.MF_OTA_BUNDLE_ID || otaStamp()) : '')" in vite
    script = _front("scripts/build-ota-bundle.mjs")
    assert "MF_OTA_BUNDLE_ID: bundleId" in script and "'--outDir', 'dist-native'" in script
    assert "se reinstalaría en bucle" in script  # la sanidad del id propio
    assert "Apple 3.1.1" in script  # la misma sanidad que el binario
    assert '"build:ota": "node scripts/build-ota-bundle.mjs"' in _front("package.json")
    dep = _leer(_RAIZ / "deploy-mealfit.ps1")
    i_build, i_ota, i_pub = dep.index("npm run build >>/tmp/npm-deploy.log"), dep.index("npm run build:ota"), dep.index("cp -a /opt/mealfit/frontend/dist")
    assert i_build < i_ota < i_pub, "el paquete se construye tras el build web y antes de publicar la release"
    assert '--exclude="frontend/dist-native"' in dep


def test_las_deps_nativas_estan_fotografiadas_y_el_plugin_viaja_en_el_binario():
    cfg = json.loads(_front("ota.config.json"))
    pkg = json.loads(_front("package.json"))
    assert isinstance(cfg["enabled"], bool) and isinstance(cfg["reset"], bool)
    assert isinstance(cfg["minNativeBuild"], int) and cfg["minNativeBuild"] >= 13
    for nombre, version in cfg["nativeDeps"].items():
        assert pkg["dependencies"].get(nombre) == version, f"{nombre}: sube minNativeBuild y actualiza nativeDeps"
    assert "@capawesome/capacitor-live-update" in cfg["nativeDeps"]
    assert "CapawesomeCapacitorLiveUpdate" in _front("ios/App/CapApp-SPM/Package.swift")
    # el plugin usa UserDefaults: Apple exige declararlo en el manifiesto de privacidad de la app
    assert "NSPrivacyAccessedAPICategoryUserDefaults" in _front("ios/App/App/PrivacyInfo.xcprivacy")
    pbx = _front("ios/App/App.xcodeproj/project.pbxproj")
    assert pbx.count("PrivacyInfo.xcprivacy in Resources") == 2 and pbx.count("/* PrivacyInfo.xcprivacy */") >= 3


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 108
    doc = (_BACKEND / "docs" / "ios_ota_live_update.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-108" in doc and "minNativeBuild" in doc and "reset" in doc
