"""[P1-COUNTRY-CHECKOUT-BETA-MUDO · 2026-08-23] G27: el checkout cobraba sin decir «beta» ni
una vez, y prometía «todas las tarjetas de República Dominicana» a los cinco países donde el
producto se vende desde el 18-ago.

MEDIDO antes de tocar: `grep -c -i 'beta'` en PaymentModal.jsx → **0**. Y la línea de las
tarjetas viajaba envuelta en `t()`, o sea traducida a los cinco idiomas: un español leía en su
idioma que le procesamos las tarjetas de otro país.

LO QUE SÍ ES NUESTRO Y LO QUE NO. Al usuario beta el propio sistema le entrega MENOS —el
Dashboard le oculta tres paneles porque su país aún no tiene precios de súper—. Cuánto cobrar
por eso es decisión del dueño y este cierre no la toca. Cobrar **sin decirlo** sí es nuestro, y
eso es lo que se arregla.

CERO CLAVES NUEVAS: el copy del aviso ya existía traducido a los cinco idiomas (el PDF lo tenía,
y `P2-DASH-BETA-NOTICE` lo reutilizó para el Dashboard). Se reutiliza literal.

⚠ LA TRAMPA DEL GAP: no reponer el país desde `health_profile.country`. Ese campo es identidad
CULINARIA, no ubicación —lo dice el comentario de `COUNTRY_PROFILES`—, así que inferir de él
dónde vive alguien sería el mismo error con otra cara. La señal correcta es `_pricing_mode` del
plan, que es la MISMA que ya gobierna los paneles ocultos del Dashboard.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend"
_MODAL = _FRONT / "src" / "components" / "dashboard" / "PaymentModal.jsx"
_UPGRADE = _FRONT / "src" / "pages" / "Upgrade.jsx"
_PRICING = _FRONT / "src" / "components" / "home" / "Pricing.jsx"


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _codigo(src: str) -> str:
    """Sin comentarios de línea: la prosa que EXPLICA la regla no puede satisfacerla."""
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith("//"))


# ── la promesa que era falsa para 5 de 6 países ───────────────────────────────

def test_el_checkout_ya_no_promete_las_tarjetas_de_un_pais_concreto():
    codigo = _codigo(_leer(_MODAL))
    assert "República Dominicana" not in codigo, (
        "el checkout vuelve a nombrar un país en la promesa de tarjetas — y viaja traducida a "
        "los cinco idiomas, así que un español la lee en español"
    )


def test_la_propiedad_verdadera_sigue_dicha():
    """Lo que sí es cierto —y es lo que el usuario necesita saber— no necesita país."""
    src = _leer(_MODAL)
    assert "PayPal procesa tarjetas de débito y crédito internacionales." in src
    assert "No necesitas abrir ni tener una cuenta de PayPal." in src


# ── y el silencio sobre la beta ───────────────────────────────────────────────

def test_el_checkout_avisa_de_la_beta_cuando_el_plan_es_beta():
    codigo = _codigo(_leer(_MODAL))
    assert "pricingMode === 'beta_no_prices'" in codigo, (
        "el checkout volvió a ser mudo sobre la beta: se cobra un producto que el propio "
        "Dashboard entrega con tres paneles menos"
    )


def test_reutiliza_el_copy_ya_traducido_y_no_inventa_una_clave():
    """Una clave nueva nacería sin traducir en los cuatro idiomas no-base y el aviso saldría en
    español a un italiano — que es media corrección."""
    src = _leer(_MODAL)
    assert "Tu país está en beta — pronto añadiremos los precios nativos de tu súper a esta lista." in src
    dash = _leer(_FRONT / "src" / "pages" / "Dashboard.jsx")
    assert "Tu país está en beta — pronto añadiremos los precios nativos de tu súper a esta lista." in dash, (
        "el copy dejó de ser compartido con el Dashboard: son dos textos que deben decir lo mismo"
    )


def test_la_senal_es_pricing_mode_y_no_el_pais_del_perfil():
    """⚠ La trampa que el gap señala: `health_profile.country` es identidad CULINARIA, no
    ubicación. La señal correcta es la misma que ya oculta los paneles del Dashboard."""
    codigo = _codigo(_leer(_MODAL))
    assert "health_profile" not in codigo and "healthProfile" not in codigo, (
        "el modal volvió a derivar el país del perfil: ese campo dice qué COCINA alguien, no "
        "dónde vive"
    )


# ── y que llegue de verdad: un prop que nadie pasa es un aviso que no existe ──

def test_los_dos_call_sites_pasan_el_regimen():
    for ruta in (_UPGRADE, _PRICING):
        codigo = _codigo(_leer(ruta))
        assert re.search(r"pricingMode=\{planData\?\._pricing_mode", codigo), (
            f"{ruta.name} monta el modal sin pasarle pricingMode: el aviso no se pintaría nunca"
        )


def test_el_prop_esta_declarado_en_proptypes():
    src = _leer(_MODAL)
    assert "pricingMode: PropTypes.string" in src
