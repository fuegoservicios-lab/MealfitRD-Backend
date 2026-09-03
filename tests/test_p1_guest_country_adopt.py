"""[P1-GUEST-COUNTRY-ADOPT · 2026-08-21] El recorrido que el landing más promociona producía
solo, sin ningún incidente humano detrás, el estado inconsistente que el 18-ago costó un P-fix.

Un español entra por el landing (que anuncia «sin tarjeta para comenzar»), completa el wizard como
invitado, elige España, recibe un plan beta CORRECTO —sin precios, con catálogo español— y se
registra para conservarlo. `api_adopt_guest_plan` contiene **exactamente una escritura**:
`_save_plan_and_track_background`. Guarda el PLAN y descarta el FORMULARIO entero.

El único escritor form→perfil vive dentro de `if actual_user_id:` en `/analyze`, inalcanzable para
un invitado; y `_hydrate_country_from_profile_for_submit` hace no-op explícito para
`user_id == "guest"`. Resultado: `health_profile.country` AUSENTE ⇒ `canonicalize_country(None)`
⇒ 'DO'. A partir de ese segundo:

  · el primer swap le devuelve comida dominicana
  · la renovación del mes siguiente sale dominicana
  · y el plan que conserva sigue marcado `beta_no_prices`, o sea sin precios PARA SIEMPRE
    mientras el motor cree que es dominicano

Es exactamente el estado que P1-COUNTRY-RENEWAL-PROFILE-WINS tuvo que reparar a mano en agosto
—perfil 'DO' con planes ES/US— pero llegando por una vía distinta y sin que nadie se equivoque.

POR QUÉ NO UN VOLCADO DEL PAYLOAD. La tentación es persistir el `formData` completo del invitado.
Sería el CUARTO setter del perfil sin jerarquía, después del PATCH de Configuración, el merge del
submit y la tool del chat — y esa acumulación es la que produjo el incidente que este P-fix imita.
Se persiste una ALLOWLIST, la misma que ya usa la rama corta del wizard (`QTrackingFinish`), y sólo
para claves ausentes: si la cuenta YA declaró país, el invitado no lo pisa.

Cubre:
  A. El país del invitado llega al perfil al adoptar.
  B. La allowlist: lo que no está en ella no se persiste.
  C. Jerarquía: una cuenta que ya declaró país NO se pisa.
  D. Sin `form_data` en el body, conducta byte-idéntica a hoy (el frontend viejo sigue sirviendo).
  E. La adopción no falla si el perfil no se puede escribir (el plan es lo importante).
  F. Parser-based + el frontend manda el formulario.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PATH = _BACKEND_ROOT / "routers" / "plans.py"
_FRONTEND = _BACKEND_ROOT.parent / "frontend" / "src"


@pytest.fixture(scope="module")
def plans():
    from routers import plans as _p
    return _p


# ── A/B/C. El helper de adopción del formulario ─────────────────────────────────────────────────

def test_el_pais_del_invitado_se_persiste_al_adoptar(plans):
    """RED pre-fix: no existía. El invitado eligió España, recibió un plan español y al
    registrarse su perfil quedaba sin país."""
    hp = {}
    plans._adopt_guest_form_into_profile(hp, {"country": "ES"})
    assert hp.get("country") == "ES"


def test_solo_viajan_las_claves_de_la_allowlist(plans):
    """Un volcado del payload sería el CUARTO setter del perfil sin jerarquía. La allowlist es la
    misma que ya usa la rama corta del wizard: si mañana hace falta una clave más, se añade en un
    sitio y con su razón."""
    hp = {}
    plans._adopt_guest_form_into_profile(hp, {
        "country": "MX",
        "allergies": ["mariscos"],
        "tier": "ultra",                 # no es del formulario: privilegio
        "user_id": "otro-usuario",       # identidad: jamás del cliente
        "_pricing_mode": "beta_no_prices",
        "loquesea": 1,
    })
    assert hp.get("country") == "MX"
    assert "tier" not in hp and "user_id" not in hp
    assert "_pricing_mode" not in hp and "loquesea" not in hp


def test_una_cuenta_que_ya_declaro_pais_no_se_pisa(plans):
    """Jerarquía: el invitado RELLENA huecos, no sobrescribe. Si la cuenta ya dijo España y el
    plan de invitado venía con el 'DO' sembrado por `initialFormData`, pisarlo repetiría
    exactamente el incidente de P1-COUNTRY-RENEWAL-PROFILE-WINS — un default sembrado ganándole a
    una elección real."""
    hp = {"country": "ES"}
    plans._adopt_guest_form_into_profile(hp, {"country": "DO"})
    assert hp["country"] == "ES"


def test_un_valor_vacio_no_cuenta_como_declaracion(plans):
    """Cadena vacía, None y lista vacía son «no contestó», no «contestó que nada»."""
    hp = {}
    plans._adopt_guest_form_into_profile(hp, {"country": "", "allergies": []})
    assert "country" not in hp and "allergies" not in hp


def test_un_pais_basura_no_se_persiste_crudo(plans):
    """El body de un invitado no está autenticado por definición. Un país que no canoniza se
    RECHAZA en vez de escribirse y descartarse en silencio — la lección de la tool del chat, que
    escribía `country='España'` y lo dejaba caer a 'DO' sin que el usuario se enterara."""
    hp = {}
    plans._adopt_guest_form_into_profile(hp, {"country": "Reino de Absurdistán"})
    assert "country" not in hp


def test_sin_form_data_no_escribe_nada(plans):
    """Contrato de compatibilidad: el frontend desplegado hoy manda sólo `plan_data`. Sin
    `form_data` el helper debe ser no-op puro — el fix no puede exigir un frontend nuevo para no
    romper la adopción."""
    hp = {"country": "ES"}
    assert plans._adopt_guest_form_into_profile(hp, None) is False
    assert hp == {"country": "ES"}
    assert plans._adopt_guest_form_into_profile(hp, {}) is False


# ── D/E. El endpoint ────────────────────────────────────────────────────────────────────────────

def test_el_endpoint_acepta_form_data_y_no_lo_exige():
    """`plan_data` sigue siendo el único campo obligatorio: si `form_data` fuera requerido, el
    frontend actual dejaría de poder adoptar y el usuario perdería su plan al registrarse — un
    precio inaceptable por un campo de conveniencia."""
    src = _PLANS_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find("def api_adopt_guest_plan(")
    assert i > 0
    _fin = src.find("\n@router", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else i + 9000]
    assert "_adopt_guest_form_into_profile" in cuerpo, (
        "el endpoint sigue descartando el formulario del invitado"
    )
    assert 'get("form_data")' in cuerpo
    assert "P1-GUEST-COUNTRY-ADOPT" in cuerpo


def test_el_fallo_al_escribir_el_perfil_no_tumba_la_adopcion():
    """El plan es lo que el usuario vino a salvar. Si el perfil no se puede escribir, se registra
    y se sigue: perder el plan por no poder guardar una preferencia sería el peor intercambio
    posible."""
    src = _PLANS_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find("_adopt_guest_form_into_profile(", src.find("def api_adopt_guest_plan("))
    assert i > 0
    ventana = src[max(0, i - 400):i + 400]
    assert "try:" in ventana and "except" in ventana, (
        "la adopción del formulario no está aislada: un fallo ahí tumbaría el plan"
    )


# ── F. El frontend manda el formulario ──────────────────────────────────────────────────────────

def test_el_frontend_manda_el_form_data_en_los_dos_call_sites():
    """Los DOS call sites del adopt mandaban `{plan_data}` y nada más. Arreglar sólo el backend
    dejaría el fix INERTE — la forma exacta de feature muerta que este repo ya pagó con el título
    del plan y con el helper de slots."""
    ctx = (_FRONTEND / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8", errors="replace")
    llamadas = [i for i in range(len(ctx)) if ctx.startswith("adopt-guest-plan", i)]
    assert llamadas, "los call sites del adopt desaparecieron"
    for i in llamadas:
        ventana = ctx[i:i + 700]
        assert "form_data" in ventana, (
            "un call site del adopt sigue mandando sólo plan_data: el país del invitado se pierde"
        )
