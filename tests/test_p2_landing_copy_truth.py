"""[P2-LANDING-COPY-TRUTH · 2026-08-14] El sitio público se contradecía consigo
mismo en tres afirmaciones, y la tabla de precios vivía en tres sitios sin guard.

LAS TRES CONTRADICCIONES (medidas el 2026-08-14)

  1. CRÉDITOS. `LegalPages.jsx` prometía «15 créditos» en dos sitios; el backend
     entrega 10 (`auth.py::_TIER_LIMITS` con `MEALFIT_TIER_LIMIT_GRATIS` sin
     definir en ningún `.env`). Roto desde P1-CREDITS-LADDER · 2026-07-31, o sea
     dos semanas en producción dentro del documento que el usuario ACEPTA al
     registrarse.

  2. UN PLAN QUE NO EXISTE. Los Términos ofrecían «Max — USD 449.99/año». Ese
     importe es el del plan que P0-ANNUAL-PLANS-MISCONFIGURED dejó INACTIVE por
     cobrar 449.99 CADA MES, y `ANNUAL_DISABLED_TIERS` lo declara sin anual.
     ⚠️ Por eso el importe NO puede derivarse del objeto `PRICING`: ahí sigue
     vivo `ultra.annual` como valor inerte, y derivarlo de ahí reimportaría el
     fantasma. Se deriva de `ANNUAL_DISABLED_TIERS`, que es quien decide.

  3. UN DIFERENCIADOR FALSO. `/funciones` vendía «memoria a largo plazo y Súper
     Personalización» como exclusivas de los planes pagos. `chat.py` [P1-TIER-PARITY]
     resuelve `is_plus = bool(user_id and user_id != "guest")` — cero lecturas de
     `plan_tier` — y `Pricing.jsx` [P3-PRICING-HONEST-COPY] dice lo contrario en
     la misma web: «Todas las funciones incluidas».

POR QUÉ ES P2 Y NO P1: no hay dinero en juego. El tier gratis no factura y quien
va a pagar pasa por `/precios`, que dice la verdad. Pero es defecto real, público
y con fix de minutos.

LA TABLA TRIPLICADA. `PRICING` era byte a byte idéntico en `Pricing.jsx` y
`Upgrade.jsx`, más una tercera copia en prosa legal. Igual `NAME_BY_TIER` y el
rank (`tierRank` vs `TIER_RANK`: dos nombres para el mismo mapa, que es cómo
empieza toda divergencia). Y ya había divergido de verdad: `getMonthlyEquiv`
tenía dos implementaciones distintas. Con la subida de precio AGENDADA, dejar
tres copias significaba editar tres sitios justo cuando más caro sale fallar.

LA URGENCIA QUE NO CADUCABA. `LAUNCH_OFFER.active` era un booleano a mano y la
fecha una cadena SIN AÑO: cero `Date`, cero comparación, cero test. Pasado el 15
de septiembre el sitio seguiría anunciando una subida que ya ocurrió (o que nunca
ocurre — un dark pattern, como advierte el propio comentario del knob).
⚠️ La zona horaria es load-bearing: `new Date('2026-09-15')` es medianoche UTC =
las 20:00 del día 14 en RD. La oferta moriría a media tarde del día anterior.

Tooltip-anchor: P2-LANDING-COPY-TRUTH
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "frontend" / "src"
_PLANS = _SRC / "config" / "plans.js"
_PRICING = _SRC / "components" / "home" / "Pricing.jsx"
_UPGRADE = _SRC / "pages" / "Upgrade.jsx"
_LEGAL = _SRC / "pages" / "legal" / "LegalPages.jsx"
_FEATURES = _SRC / "pages" / "FeaturesPage.jsx"
_AUTH = _REPO_ROOT / "backend" / "auth.py"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"[P2-LANDING-COPY-TRUTH] No existe {path.relative_to(_REPO_ROOT)}")
    return path.read_text(encoding="utf-8")


def _sin_comentarios(texto: str) -> str:
    """Quita comentarios JS/JSX antes de buscar copy.

    Hace falta porque el comentario que EXPLICA un defecto contiene, por
    necesidad, la cadena del defecto: el comentario de `LegalPages.jsx` que
    documenta por qué se retiraron los «15 créditos» y los «449.99» disparaba
    estos guards contra un fichero ya corregido. Un guard que no distingue el
    código de la prosa que lo explica obliga a escribir comentarios cobardes.
    """
    texto = re.sub(r"/\*.*?\*/", "", texto, flags=re.DOTALL)   # /* … */ y JSX {/* … */}
    return re.sub(r"^\s*//.*$", "", texto, flags=re.MULTILINE)  # // …


# ---------------------------------------------------------------------------
# 1. La tabla de precios y sus vecinos viven en UN sitio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nombre", ["PRICING", "NAME_BY_TIER"])
def test_las_tablas_compartidas_estan_en_el_ssot(nombre):
    plans = _read(_PLANS)
    assert re.search(rf"export const {nombre}\s*=", plans), (
        f"[P2-LANDING-COPY-TRUTH] `{nombre}` no está en `config/plans.js`, que es "
        "el módulo que existe justo para esto (su cabecera lo dice)."
    )


@pytest.mark.parametrize("fichero", [_PRICING, _UPGRADE])
def test_ninguna_pagina_redeclara_la_tabla_de_precios(fichero):
    contenido = _read(fichero)
    assert not re.search(r"^const PRICING\s*=", contenido, re.MULTILINE), (
        f"[P2-LANDING-COPY-TRUTH] `{fichero.name}` vuelve a declarar `PRICING` en local.\n"
        "Eran dos copias byte a byte y ya habían divergido en `getMonthlyEquiv`. "
        "Con la subida de precio agendada, tres copias significan tres sitios que "
        "editar justo cuando más caro sale equivocarse."
    )


def test_el_rank_de_tiers_no_tiene_dos_nombres():
    """`tierRank` y `TIER_RANK` eran el MISMO mapa con dos identificadores."""
    for fichero in (_PRICING, _UPGRADE):
        contenido = _read(fichero)
        assert not re.search(r"(?:const\s+(?:tierRank|TIER_RANK)\s*=\s*\{)", contenido), (
            f"[P2-LANDING-COPY-TRUTH] `{fichero.name}` redeclara el rank de tiers.\n"
            "Dos nombres para el mismo mapa es exactamente cómo empieza una "
            "divergencia que nadie ve."
        )
    assert re.search(r"export const TIER_RANK\s*=", _read(_PLANS)), (
        "[P2-LANDING-COPY-TRUTH] Falta `TIER_RANK` en el SSOT."
    )


# ---------------------------------------------------------------------------
# 2. La oferta de lanzamiento caduca sola
# ---------------------------------------------------------------------------

def test_la_oferta_tiene_fecha_real_no_una_cadena():
    plans = _read(_PLANS)
    assert re.search(r"deadlineISO\s*:\s*['\"]\d{4}-\d{2}-\d{2}['\"]", plans), (
        "[P2-LANDING-COPY-TRUTH] `LAUNCH_OFFER` no tiene `deadlineISO`.\n"
        "`deadlineLabel: '15 de septiembre'` es una cadena SIN AÑO: no hay nada "
        "que comparar contra hoy, así que la urgencia no puede caducar sola."
    )


def test_la_oferta_se_evalua_contra_la_fecha():
    plans = _read(_PLANS)
    assert "isLaunchOfferActive" in plans, (
        "[P2-LANDING-COPY-TRUTH] Falta el predicado que compara la fecha con hoy. "
        "Sin él, `active` sigue siendo un booleano a mano que nadie recordará bajar."
    )


def test_la_caducidad_usa_la_hora_de_rd_no_utc():
    """`new Date('2026-09-15')` es medianoche UTC = 20:00 del día 14 en RD."""
    plans = _read(_PLANS)
    assert "-04:00" in plans or "America/Santo_Domingo" in plans, (
        "[P2-LANDING-COPY-TRUTH] La caducidad no fija la zona horaria de RD.\n"
        "Con UTC la oferta muere a las 20:00 del día ANTERIOR, a media tarde. Es "
        "el mismo «¿día de quién?» que ya se pagó en P1-AGENT-SESSION-DAY."
    )


@pytest.mark.parametrize("fichero", [_PRICING, _UPGRADE])
def test_las_tarjetas_consultan_el_predicado(fichero):
    contenido = _read(fichero)
    assert "isLaunchOfferActive" in contenido, (
        f"[P2-LANDING-COPY-TRUTH] `{fichero.name}` sigue pintando la urgencia con "
        "`LAUNCH_OFFER.active` a secas: el predicado existe pero nadie lo consulta "
        "— el arreglo nace inerte."
    )


# ---------------------------------------------------------------------------
# 3. Las tres afirmaciones públicas dicen la verdad
# ---------------------------------------------------------------------------

def test_lo_legal_no_promete_creditos_que_no_entregamos():
    legal = _sin_comentarios(_read(_LEGAL))
    assert "15 créditos" not in legal, (
        "[P2-LANDING-COPY-TRUTH] Los textos legales vuelven a prometer 15 créditos.\n"
        f"El backend entrega {_creditos_gratis_del_backend()} "
        "(`auth.py::_TIER_LIMITS`, sin override en ningún `.env`). Es una promesa "
        "incumplida dentro del documento que el usuario acepta al registrarse."
    )


def _creditos_gratis_del_backend() -> int:
    m = re.search(r'"gratis"\s*:\s*_env_int\(\s*"MEALFIT_TIER_LIMIT_GRATIS"\s*,\s*(\d+)', _read(_AUTH))
    if not m:
        pytest.fail("[P2-LANDING-COPY-TRUTH] No se pudo leer el default de créditos gratis.")
    return int(m.group(1))


def test_el_ssot_de_creditos_casa_con_el_backend():
    m = re.search(r"TIER_CREDITS\s*=\s*\{\s*gratis:\s*(\d+)", _read(_PLANS))
    assert m, "[P2-LANDING-COPY-TRUTH] No se pudo leer `TIER_CREDITS.gratis`."
    assert int(m.group(1)) == _creditos_gratis_del_backend(), (
        "[P2-LANDING-COPY-TRUTH] El SSOT del frontend y el backend discrepan en los "
        "créditos del plan gratuito."
    )


def test_lo_legal_no_vende_un_plan_anual_que_no_existe():
    legal = _sin_comentarios(_read(_LEGAL))
    assert "449.99" not in legal, (
        "[P2-LANDING-COPY-TRUTH] Los Términos vuelven a ofrecer «Max — USD 449.99/año».\n"
        "Ese plan NO existe: `ANNUAL_DISABLED_TIERS` incluye `ultra` y su id de "
        "PayPal quedó INACTIVE tras P0-ANNUAL-PLANS-MISCONFIGURED, porque cobraba "
        "449.99 CADA MES.\n"
        "⚠️ No lo derives del objeto `PRICING`: ahí `ultra.annual` sigue vivo como "
        "valor inerte y volverías a importar el fantasma."
    )


def test_funciones_no_inventa_un_diferenciador_de_pago():
    """Ninguna frase de `/funciones` puede atar esas dos features a un tier.

    El reclamo aparecía en DOS formulaciones separadas por 200 líneas («los
    planes pagos suman…» y «…varían según tu plan»), así que el patrón busca la
    CLASE —una frase que mencione la feature junto a una marca de tier— y no las
    dos redacciones concretas, que es lo que dejaría entrar a la tercera.
    """
    features = _sin_comentarios(_read(_FEATURES))
    marcas_de_tier = r"(?:planes? pagos?|seg[úu]n tu plan|Plus|Max|de pago)"
    for frase in re.split(r"(?<=\.)\s", features):
        if "memoria a largo plazo" not in frase and "Súper Personalización" not in frase:
            continue
        assert not re.search(marcas_de_tier, frase), (
            "[P2-LANDING-COPY-TRUTH] `/funciones` ata a un tier funciones que TODOS "
            f"tienen: «{frase.strip()[:150]}…»\n"
            "`chat.py` [P1-TIER-PARITY] resuelve `is_plus = bool(user_id and user_id != "
            '"guest")` — cero lecturas de `plan_tier` — y `Pricing.jsx` '
            "[P3-PRICING-HONEST-COPY] dice lo contrario en la misma web.\n"
            "El único diferenciador honesto es el VOLUMEN de créditos."
        )


def test_funciones_afirma_la_paridad_en_positivo():
    """Anti-vacuidad anclada en el copy HONESTO, no en el defectuoso.

    La primera versión de este guard exigía encontrar las frases culpables para
    demostrar que sabía mirar — y entonces el propio arreglo, al hacerlas
    desaparecer, lo tumbaba. *Un guard cuya condición de vida es que el defecto
    siga ahí no se puede satisfacer arreglando el defecto.* Lo que debe seguir
    vivo es la afirmación verdadera.
    """
    features = _sin_comentarios(_read(_FEATURES))
    assert re.search(r"[Tt]odas las funciones", features), (
        "[P2-LANDING-COPY-TRUTH] Desapareció de `/funciones` la afirmación de "
        "paridad («todas las funciones incluidas»). Sin ella la página vuelve a "
        "dejar en el aire qué se compra con un plan pago, que es de donde salió "
        "el diferenciador inventado."
    )
