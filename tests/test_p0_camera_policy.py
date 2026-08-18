"""[P0-CAMERA-POLICY · 2026-08-18] `camera=()` rompía el escáner en producción.

QUÉ PASÓ. `snippets/mealfit-security.conf` lo incluyen los DOS dominios y
declaraba `Permissions-Policy: camera=()`. Una allowlist **vacía** no significa
«restringido a los de casa»: desactiva la capacidad para todo el mundo, el propio
origen incluido, y ni siquiera el permiso del usuario la reactiva.

Medido en producción con permiso concedido y cámara simulada:

    getUserMedia({video}) -> NotAllowedError: Permission denied

O sea que «Escanear comida» (`ScanMealModal`) y «Escanear nevera»
(`PantryScanButton`) —los dos pasan por `CameraViewfinder`, que llama a
`getUserMedia`— estaban **rotos**, no limitados. Una función de pago que no podía
funcionar por una cabecera, y que además falla de una forma que parece «el
usuario denegó el permiso»: el modo de fallo se disfraza de decisión del usuario,
que es lo que lo hizo sobrevivir.

EL ARREGLO es un `map $host $pp_camera` en `nginx.conf` (contexto http) que
resuelve `(self)` para los hosts de la app y `()` para todo lo demás. Un solo
snippet, dos comportamientos, cero duplicación que pueda divergir.

ESTE TEST existe porque el fallo era invisible: no rompe el build, no rompe un
test de UI (jsdom no tiene cámara) y en el navegador se parece a un permiso
denegado. Lo único que lo delata es la cabecera.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNIPPET = _REPO_ROOT / "backend" / "infra" / "nginx" / "snippets" / "mealfit-security.conf"


def _texto() -> str:
    if not _SNIPPET.exists():
        pytest.skip(f"{_SNIPPET} no existe en este checkout (repos hermanos)")
    return _SNIPPET.read_text(encoding="utf-8")


def _directiva_permissions_policy(txt: str) -> str:
    """La línea `add_header Permissions-Policy ...`, sin comentarios."""
    for linea in txt.splitlines():
        limpia = linea.strip()
        if limpia.startswith("#"):
            continue
        if "Permissions-Policy" in limpia:
            return limpia
    pytest.fail("no hay ninguna directiva `add_header Permissions-Policy` activa")


def test_camera_no_esta_fijada_a_allowlist_vacia():
    """`camera=()` literal es el bug. Tiene que salir de la variable por host."""
    d = _directiva_permissions_policy(_texto())
    assert "camera=()" not in d, (
        "`camera=()` vuelve a estar fijo en el snippet, y ese snippet lo incluyen "
        "los DOS dominios. Con esto, `getUserMedia` falla en app.bioboros.com con "
        "NotAllowedError aunque el usuario conceda el permiso, y el escáner de "
        "comida y el de la nevera dejan de funcionar. Debe salir de `$pp_camera`."
    )
    assert "camera=$pp_camera" in d, (
        "`camera` debe leerse del `map $host $pp_camera` declarado en nginx.conf: "
        "() por defecto, (self) en los hosts de la app."
    )


def test_microfono_y_geolocalizacion_siguen_cerrados():
    """Abrir la cámara no es excusa para abrir lo demás."""
    d = _directiva_permissions_policy(_texto())
    assert "microphone=()" in d, "el micrófono debe seguir con allowlist vacía"
    assert "geolocation=()" in d, "la geolocalización debe seguir con allowlist vacía"


def test_payment_sigue_acotado_a_paypal():
    """`payment` ya estaba bien; que un cambio de cámara no se lo lleve por delante."""
    d = _directiva_permissions_policy(_texto())
    assert "payment=(self" in d and "paypal.com" in d, (
        "`payment` debe seguir limitado al propio origen y a PayPal"
    )


def test_la_directiva_lleva_always():
    """Sin `always`, la cabecera desaparece en las respuestas de error.

    Un 4xx/5xx sin cabeceras de seguridad es una página del mismo origen servida
    con menos defensas que las demás.
    """
    d = _directiva_permissions_policy(_texto())
    assert d.rstrip().endswith("always;"), (
        "la directiva debe terminar en `always;` para que aplique también a las "
        "respuestas de error"
    )


def test_el_snippet_documenta_por_que_es_por_host():
    """Anclaje: si alguien simplifica el `map`, que encuentre la razón aquí."""
    txt = _texto()
    assert "P0-CAMERA-POLICY" in txt, "falta el marcador del P-fix en el snippet"
    assert re.search(r"map\s+\$host\s+\$pp_camera", txt), (
        "el snippet debe recordar la forma del `map` que vive en nginx.conf; sin "
        "esa pista, quien lea sólo este fichero no sabe de dónde sale el valor"
    )
