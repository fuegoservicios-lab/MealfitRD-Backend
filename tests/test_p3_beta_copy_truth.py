"""[P3-BETA-COPY-TRUTH · 2026-08-22] Dos textos del producto afirman cosas que son falsas desde el
flip — y uno de ellos lo era ya antes, para todo el mundo.

────────────────────────────────────────────────────────────────────────────────────────────────
1 · EL BOT DE AYUDA LE PROMETE PRECIOS EN RD$ A UN USUARIO QUE NO LOS TIENE

`prompts/help_bot.py::_PROMPT_BASE` afirma, sin condición ninguna:

  · que la IA genera «un plan … adaptado a la cocina y a los precios de República Dominicana»
  · que el plan incluye «lista de compras con precios estimados en RD$»

Para los cinco países beta las dos son falsas, y no por matiz: el strip de precios de Fase 2 (T7)
deja la lista **sin un solo importe** —verificado: 0 ítems con precio en los 2 planes beta vivos—
y la cocina dejó de forzarse criolla el 18-ago. O sea que el canal oficial de soporte le describe
al usuario un producto que no va a recibir, y cuando abra su lista vacía de precios pensará que
algo se rompió.

POR QUÉ NO SE ARREGLA «PASÁNDOLE EL PAÍS AL BOT». Es la solución que parece obvia y no lo es: el
widget le manda `locale`, y `locale` y `country` son **ejes independientes por decisión
declarada** (un dominicano puede leer la app en inglés; un español, en español). Inferir el país
del idioma sería fabricar el dato — y encima el bot vive también en el landing público, donde no
hay sesión ni perfil que consultar.

Así que el texto se vuelve cierto PARA TODOS: dice lo que hay en RD y dice lo que hay en beta.
Un usuario dominicano lee exactamente lo que leía; uno español lee la verdad en vez de una
promesa.

────────────────────────────────────────────────────────────────────────────────────────────────
2 · LA DESCRIPCIÓN DE `/precios` DICE RD$ Y LA PÁGINA COBRA EN USD

`frontend/src/data/routeMeta.js` describe `/precios` como «Planes y precios de Bioboros … Precios
reales en RD$». Medido en el SSOT (`config/plans.js`): Básico 9.99, Plus 19.99, Max 49.99, todos
**USD** vía PayPal. El propio bot de ayuda lo dice bien en su bloque de planes: «(USD, pago con
PayPal)».

Este no es un defecto del sistema de países: es falso desde antes del flip y para cualquier
visitante, dominicano incluido. Aparece en el snippet de buscadores, o sea en el sitio donde
alguien decide si hace clic — y quien llegue esperando pesos se encuentra dólares.

LO QUE NO SE TOCA, y este fichero lo ancla en positivo: `/supermercado` **sí** dice «precios
reales en RD$» y **es verdad** — es el catálogo del súper dominicano, y sus precios están en
pesos. Corregir el de al lado por barrido sería cambiar un texto cierto por uno peor, que es el
error simétrico y el más fácil de cometer cuando se arregla por grep.

LO QUE TAMPOCO SE TOCA. El `priceCurrency: "DOP"` del JSON-LD de `index.html` describe una `Offer`
de **precio 0** (el plan gratuito), así que no miente sobre ningún cobro; y el copy de marketing
que vende «comida dominicana» es P1-27, una decisión del dueño. Meter cualquiera de los dos aquí
sería colar una decisión de producto dentro de un arreglo factual.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_HELP_BOT = _BACKEND / "prompts" / "help_bot.py"
_ROUTE_META = _BACKEND.parent / "frontend" / "src" / "data" / "routeMeta.js"

#: Los cinco países beta, tal como los ofrece el selector.
_BETA = ("España", "México", "Estados Unidos", "Puerto Rico", "Colombia")


def _leer(p: Path) -> str:
    if not p.is_file():
        pytest.skip(f"{p.name} no está en este árbol")
    return p.read_text(encoding="utf-8", errors="replace")


# ── 1 · El bot de ayuda ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prompt_base() -> str:
    """SÓLO el literal `_PROMPT_BASE`, no el módulo entero: los comentarios de este fichero narran
    los P-fixes anteriores y citan frases del copy viejo. Leer el módulo completo pondría en verde
    al defecto por culpa de la prosa que lo explica — comentario-vence-guard, que esta ola ya ha
    pagado diez veces."""
    src = _leer(_HELP_BOT)
    i = src.index('_PROMPT_BASE = """')
    return src[i:src.index('"""', i + 20)]


def test_no_promete_precios_en_rd_sin_condicion(prompt_base):
    """EL CASO. La frase «precios estimados en RD$» describía la lista de compras de todo el
    mundo; en beta la lista no trae ni un importe."""
    # ⚠️ La alternativa NO puede incluir `en RD\\b`: eso casa con «en RD$», o sea con la propia
    # frase que se quiere prohibir, y el caso pasaba en vacío. El acotador tiene que ser una
    # palabra que sólo aparece cuando alguien SÍ delimitó el alcance.
    for linea in prompt_base.splitlines():
        if "RD$" not in linea:
            continue
        assert re.search(r"beta|Dominicana", linea, re.I), (
            f"esta línea promete RD$ sin acotar a quién le aplica: {linea.strip()!r}"
        )


def test_no_afirma_que_el_plan_se_adapta_a_los_precios_de_rd(prompt_base):
    """La segunda afirmación, más sutil: «adaptado a la cocina y a los precios de República
    Dominicana» describe el motor entero, y dejó de ser cierto el día del flip."""
    assert not re.search(r"adaptado a la cocina y a los precios de Rep[úu]blica Dominicana",
                         prompt_base), (
        "el prompt vuelve a afirmar que el plan se adapta a la cocina y los precios de RD para "
        "todo el mundo. En beta ni la cocina se fuerza criolla ni la lista lleva precios"
    )


def test_declara_que_en_beta_la_lista_no_trae_precios(prompt_base):
    """No basta con borrar la promesa: si el bot no sabe explicar la lista sin precios, mandará al
    usuario a soporte por algo que es el comportamiento esperado."""
    # ⚠️ No basta con que aparezca la palabra «beta»: la mutación quitó la frase que explica la
    # consecuencia y el guard siguió verde, porque «beta» sigue apareciendo en la línea que lista
    # los países. Lo que hay que exigir es el VÍNCULO — una misma frase que ate el estado beta a
    # la ausencia de precios. Un guard que se conforma con la palabra suelta mide vocabulario, no
    # contenido.
    frases = re.split(r"(?<=[.\n])", prompt_base)
    assert any(re.search(r"beta", f, re.I) and re.search(r"sin precios", f, re.I) for f in frases), (
        "el prompt no explica en ninguna frase que en los países beta la lista llega SIN precios. "
        "Sin eso, el bot mandará a soporte a quien pregunte por su lista vacía de importes, que "
        "es el comportamiento esperado"
    )


def test_nombra_los_paises_beta(prompt_base):
    """Que los nombre, porque el usuario pregunta por SU país, no por «los países beta»."""
    faltan = [p for p in _BETA if p not in prompt_base]
    assert not faltan, f"el prompt no nombra estos países del selector: {faltan}"


def test_sigue_diciendo_que_los_planes_se_cobran_en_usd(prompt_base):
    """Ancla en positivo de lo que ya estaba BIEN. El bloque de planes decía «(USD, pago con
    PayPal)» y es cierto; un barrido que «unificara» monedas podría romperlo."""
    assert "USD" in prompt_base, (
        "el prompt dejó de decir que la suscripción se cobra en USD, que es lo único que este "
        "fichero ya tenía correcto sobre dinero"
    )


# ── 2 · La descripción de /precios ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def route_meta() -> str:
    return _leer(_ROUTE_META)


def _descripcion(route_meta: str, ruta: str) -> str:
    """⚠️ Acotado al mapa `DESCRIPTIONS`. `routeMeta.js` tiene DOS mapas indexados por la misma
    ruta —`TITLES` primero, `DESCRIPTIONS` después— y la primera versión de este helper buscaba
    en el fichero entero: devolvía el TÍTULO («Planes y Precios · Bioboros»), que obviamente no
    contiene «RD$», así que el caso que destapa el defecto pasaba en verde sobre el defecto vivo.
    Cuarta vez en esta ola que un test mío mide el sitio equivocado."""
    i = route_meta.index("export const DESCRIPTIONS = {")
    bloque = route_meta[i:]
    m = re.search(rf"'{re.escape(ruta)}':\s*[`'\"](.+?)[`'\"],\n", bloque, re.S)
    assert m, f"no encuentro la descripción de {ruta} en el mapa DESCRIPTIONS de routeMeta.js"
    return m.group(1)


def test_precios_no_dice_que_cobra_en_pesos(route_meta):
    """EL CASO. Los tres planes se cobran en USD (`config/plans.js`), y esta cadena es el snippet
    de buscadores: el sitio donde alguien decide si hace clic."""
    desc = _descripcion(route_meta, "/precios")
    assert "RD$" not in desc, (
        f"la descripción de /precios sigue prometiendo pesos y la página cobra en dólares: {desc!r}"
    )


def test_precios_dice_la_moneda_correcta(route_meta):
    """Borrar la moneda equivocada deja la descripción muda sobre lo único que el visitante quiere
    saber antes de hacer clic."""
    desc = _descripcion(route_meta, "/precios")
    assert "USD" in desc, f"la descripción de /precios no dice en qué moneda se cobra: {desc!r}"


def test_supermercado_conserva_su_rd_que_si_es_cierto(route_meta):
    """El error simétrico, anclado: `/supermercado` es el catálogo del súper DOMINICANO y sus
    precios están de verdad en pesos. Arreglar esto por grep cambiaría un texto cierto por uno
    peor."""
    desc = _descripcion(route_meta, "/supermercado")
    assert "RD$" in desc, (
        "se le quitó el RD$ a /supermercado, que era CORRECTO: es el súper dominicano y sus "
        "precios están en pesos"
    )
