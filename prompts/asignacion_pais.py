# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-748 · 2026-09-28] Las ramas por país del bloque DINÁMICO del día (G24, parte determinista).

`prompts.day_generator.build_day_assignment_context` es el tramo del prompt que el generador lee al final, el más
concreto — el que P1-DIET-BLIND-DIRECTIVES midió que gana a las cabeceras. La auditoría del 28-sep encontró ahí, para
ES/US/MX/PR/CO, cinco restos dominicanos: «en la mesa dominicana…», «casabe» (el catálogo beta lo excluye:
`graph_orchestrator._BETA_CATALOG_DO_EXCLUSIVE_NAMES`), «Salami dominicano», la categoría A del desayuno como
«Tubérculos/plátano» (a una española «plátano» es la BANANA) y el enum crudo «Mangú/Tubérculos» en el brief de los otros
días. Y el system prompt justificaba el «arroz de noche» con la cena dominicana y un gate que en beta sólo avisa.

Reglas de este módulo:
  · La rama DO NO cambia: byte-idéntica, con huellas sha256 en `tests/test_p1_plan_lote_748.py`.
  · Nada de un fragmento por país: lo local sale de DATOS que ya existen (el desayuno típico de
    `cultural_profiles.PROFILES`, la lista de exclusivos DO del catálogo). La profundidad por país (platos propios) es
    la parte de G24 que se valida con IA, no ésta.
  · El gate no cambia: el texto dice lo que el gate HACE.

[ronda 1 de la revisión] TRES países distintos deciden cosas distintas (I16), y el llamador los deriva UNA vez:
  · la COCINA del día (`cultural_country_for_form_data(form_data, day_index=…)`): gentilicio, «arepitas», la
    etiqueta A del desayuno;
  · el MERCADO (`country_for_form_data`): lo que se COMPRA — casabe, «Salami dominicano», «habichuelas»;
  · el GATE (`cultural_country_for_form_data(form_data)`, el `_rpn_country` de `review_plan_node`): con qué dureza
    se rechaza.
Por eso los helpers reciben el booleano ya calculado (o el código ya canónico), no el país crudo: con un país corrupto
`canonicalize_country` avisa una vez por DERIVACIÓN, no una vez por ítem (P2-COUNTRY-HOUSEKEEPING).

tooltip-anchor: P1-PLAN-LOTE-748
"""
from __future__ import annotations

import re

# ─── (b) arroz de noche: el motivo, no la regla ───────────────────────────────────────────────────────────────────────
# El DO es el literal EXACTO de la regla d) CENA de `DAY_GENERATOR_SYSTEM_PROMPT`: `_BETA_FRAGMENT_TABLE` lo sustituye.
# El system prompt se renderiza con la cocina PRINCIPAL (`_day_system_instruction_for_diet`), que es el país del gate.
ARROZ_NOCHE_MOTIVO_DO = "(no se acostumbra en la cena dominicana y el gate lo rechaza)"
ARROZ_NOCHE_MOTIVO_BETA = "(es una regla de horario del plan y el validador de horario lo señala)"


# ─── (a) avena en las comidas fuertes ────────────────────────────────────────────────────────────────────────────────
def cierre_avena(cocina_do: bool, gate_do: bool) -> str:
    """El cierre de la regla «la avena no es almuerzo ni cena».

    La COCINA del día decide el gentilicio y «arepitas» (léxico DO: `constants._DO_LEXICON_NEUTRAL`); el GATE decide la
    consecuencia. [ronda 1] Nunca «sólo lo señala»: aunque la regla de horario beta sea blanda, este patrón lo rechaza
    el revisor cultural con severidad `high` también en beta (16 rechazos en 4 días en el journal, P1-OATS-NOT-A-DINNER).
    Con gate DO además reintenta el validador de horario ⇒ «el plan se rechaza», el literal de siempre."""
    lista = "tortitas, arepitas, bowls salados" if cocina_do else "tortitas, bowls salados"
    mesa = "en la mesa dominicana " if cocina_do else ""
    consecuencia = "el plan se rechaza" if gate_do else "el revisor lo rechaza"
    return (f"Nada de {lista} ni «avena al caldo» en las comidas fuertes: {mesa}eso no es un almuerzo ni una cena, "
            f"y {consecuencia}.")


AVENA_CIERRE_DO = cierre_avena(True, True)
AVENA_CIERRE_BETA = cierre_avena(False, False)

# ─── (a) Salami dominicano en las proteínas prohibidas ──────────────────────────────────────────────────────────────
# Sólo la ETIQUETA que ve el modelo; la clave y el emparejamiento contra el pool no cambian. Es un nombre de PRODUCTO
# del catálogo DO ⇒ lo decide el mercado.
_ETIQUETAS_PROTEINA_BETA = {"Salami dominicano": "Salami"}

# ─── (c) la categoría A del desayuno ────────────────────────────────────────────────────────────────────────────────
ENUM_DESAYUNO_A = "Mangú/Tubérculos"            # valor del esquema (schemas.py `breakfast_category`): NO se toca
_ETIQUETA_A_BETA_CORTA = "Desayuno típico local"
# [ronda 1] Cuando del desayuno típico no queda nada propio (US y PR: sus ítems son TODOS base de otra categoría; o la
# alergia/dieta vetó el resto). Tubérculo y NO «cereal/tubérculo»: `desayuno_por_alergia.reasignar` usa la A como
# REFUGIO del alérgico al gluten (gluten+huevo ⇒ Mangú/Tubérculos, Batido/Bowl, Mangú/Tubérculos), el vocabulario del
# gluten no veta «cereal» y «cereal» es la base de la B. «Tubérculo» es además lo común a las otras tres definiciones de
# la A (planner beta «tubérculo o plátano», §15a beta «cereal/tubérculo», regla 9 «Mangú/tubérculos»), sin el «plátano»
# que una española lee como banana.
ETIQUETA_A_BETA_RESPALDO = "Base de tubérculo local"
# El aviso de las otras cuatro categorías: en DO «NO uses mangú/tubérculos…»; en beta la A ya no es un tubérculo, y
# «NO uses tubérculo/plátano» le vetaba la banana del desayuno a quien lee «plátano» como banana.
AVISO_DESAYUNO_BETA = "NO la cambies por la categoría de otro día"

# [ronda 1] La A debe ser DISTINTA de las otras cuatro (schemas.py: «DEBE ser diferente para cada día»): fuera lo que ya
# es la base de B (Avena/Cereales), C (Pan/Tostadas), D (Batido/Bowl, el yogur) y E (Revoltillo/Tortilla, el huevo).
# «tortilla» NO está a propósito: el único perfil que la lista para el desayuno es MX, donde es la de MAÍZ, no la de
# huevo de la E — y el dato lo dice («tortilla de maíz», ronda 2).
_BASES_DE_OTRAS_CATEGORIAS = ("avena", "cereal", "granola", "pan", "tostada", "yogur", "yogurt", "batido", "bowl",
                              "huevo", "revoltillo")
# §15a beta: «DESAYUNO … PROHIBIDO: … sopas sustanciosas» (el «caldo» del desayuno colombiano).
_NO_ES_DESAYUNO = ("caldo", "sopa")
# [ronda 2] La fruta va en TODO desayuno (regla 9: «base sólida + proteína + fruta») y es la base de la D (Batido/Bowl):
# no es la identidad de la A. España quedaba en «Desayuno típico local (fruta)», que choca con las dos cosas ⇒ respaldo.
_NO_ES_BASE_PROPIA = ("fruta",)

# [ronda 2] La CLASE de un ítem, para la puerta de alergias. El vocabulario de alergias no expande la clase a sus
# miembros (`_allergen_pool_item_banned('frijoles', ['legumbres'])` es False, y el escáner tampoco ve la legumbre en
# «frijoles refritos»): la A se lo imponía a quien declaró «legumbres». Los MIEMBROS salen de la tupla con la que el
# motor ya reconoce una legumbre (`graph_orchestrator._LEGUME_PROTEIN_HINT`); éstos son los dos nombres con que se
# DECLARA la clase. La causa de fondo (el vocabulario) es un lote propio, junto con el de frutos secos.
_CLASE_LEGUMBRE = ("legumbres", "leguminosas")
_LEGUMBRES_RESPALDO = ("habichuela", "frijol", "lenteja", "garbanzo", "gandul", "guandul", "arveja", "guisante")

_EXCLUSIVOS_DO_RESPALDO = ("casabe",)


def es_do(country) -> bool:
    """La MISMA puerta que usa `build_day_assignment_context` (`canonicalize_country`, None ⇒ DO). Para UNA derivación;
    dentro de un render se llama una vez y se pasa el booleano."""
    try:
        from constants import canonicalize_country
        return canonicalize_country(country) == "DO"
    except Exception:                                                          # noqa: BLE001
        return True


def _norm(s) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(s or "").casefold())
    except Exception:                                                          # noqa: BLE001
        return str(s or "").casefold()


def _contiene(item, terminos) -> bool:
    """¿El ítem nombra alguno de los términos? Palabra completa, con plural (`pan` ≠ `panqueque`)."""
    t = _norm(item)
    return any(re.search(rf"\b{re.escape(_norm(x))}(?:s|es)?\b", t) for x in terminos)


def _exclusivos_do() -> tuple:
    """Los alimentos que el catálogo beta NO ofrece, en minúsculas. SSOT: la lista del catálogo cerrado."""
    try:
        import graph_orchestrator as _go
        nombres = tuple(str(n).casefold() for n in _go._BETA_CATALOG_DO_EXCLUSIVE_NAMES)
        return nombres or _EXCLUSIVOS_DO_RESPALDO
    except Exception:                                                          # noqa: BLE001
        return _EXCLUSIVOS_DO_RESPALDO


def sin_exclusivos_do(items, mercado_do: bool) -> list:
    """Sugerencias sin lo que el catálogo beta no vende. Mercado DO ⇒ la lista tal cual (mismo contenido, mismo orden).

    `mercado_do` es el del MERCADO (`country_for_form_data`), no el de la cocina del día: un usuario DO con un día de
    cocina española sigue comprando en su catálogo DO, que sí vende casabe."""
    items = list(items or [])
    if mercado_do:
        return items
    excl = _exclusivos_do()
    return [i for i in items
            if not any(re.search(rf"\b{re.escape(e)}\b", str(i).casefold()) for e in excl)]


def etiqueta_proteina(label: str, mercado_do: bool) -> str:
    if mercado_do:
        return label
    return _ETIQUETAS_PROTEINA_BETA.get(label, label)


def legumbres_del_mercado(texto, mercado: str):
    """«habichuelas/lentejas/garbanzos» (`constants.diet_protein_suggestions`) con la palabra del MERCADO.

    DO ⇒ tal cual. Beta ⇒ «frijoles», salvo que «habichuelas» sea palabra del propio mercado (está en los staples de su
    perfil: Puerto Rico). Dato, no un texto por país. `mercado` es un código YA canónico."""
    if not texto or es_do(mercado):
        return texto
    try:
        from cultural_profiles import PROFILES, profile_for_market
        staples = (PROFILES.get(profile_for_market(mercado)) or {}).get("staples") or []
        if any(_contiene(s, ("habichuela",)) for s in staples):
            return texto
    except Exception:                                                          # noqa: BLE001
        pass
    return texto.replace("habichuelas/", "frijoles/")


def _desayuno_tipico(country) -> list:
    try:
        from cultural_profiles import PROFILES, profile_for_market
        perfil = PROFILES.get(profile_for_market(country)) or {}
        return [str(x).strip() for x in ((perfil.get("slot_affinity") or {}).get("desayuno") or []) if str(x).strip()]
    except Exception:                                                          # noqa: BLE001
        return []


def _nombres_para_la_puerta(item) -> list:
    """[ronda 2] Los nombres con que la puerta de alergias pregunta por un ítem de la A: el ítem, la BASE con que el
    catálogo lo compra (`constants.GLOBAL_REVERSE_MAP`: «arepa» ⇒ «harina de maíz precocida») y, si es una legumbre,
    el nombre de la CLASE. La puerta (`_allergen_pool_item_banned`) sólo casa el término declarado DENTRO del texto:
    «arepa» no contiene «maíz» y «frijoles» no contiene «legumbres». El dato ya viene calificado («arepa de maíz»,
    `cultural_profiles`); esto es la defensa para el que no lo venga."""
    nombres = [item]
    try:
        from constants import GLOBAL_REVERSE_MAP
        for clave in (str(item).strip().casefold(), _norm(item).strip()):
            base = GLOBAL_REVERSE_MAP.get(clave)
            if base and base not in nombres:
                nombres.append(base)
    except Exception:                                                          # noqa: BLE001
        pass
    try:
        import graph_orchestrator as _go
        miembros = tuple(_go._LEGUME_PROTEIN_HINT) or _LEGUMBRES_RESPALDO
    except Exception:                                                          # noqa: BLE001
        miembros = _LEGUMBRES_RESPALDO
    if _contiene(item, miembros):
        nombres.extend(_CLASE_LEGUMBRE)
    return nombres


def _vetados_por_dieta(dieta) -> tuple:
    """Lo que la dieta prohíbe, con el vocabulario del ESCÁNER de dieta (`graph_orchestrator._scan_diet_violations`),
    no con una lista nueva. «caldo» se suma para veg*: la línea dura vegana prohíbe «caldos de origen animal»."""
    try:
        from constants import canonicalize_diet_type
        canon = canonicalize_diet_type(dieta) if dieta else None
    except Exception:                                                          # noqa: BLE001
        canon = None
    if canon not in ("vegan", "vegetarian", "pescatarian"):
        return ()
    try:
        import graph_orchestrator as _go
        carne = tuple(_go._DIET_FLESH_TERMS)
        mar = tuple(_go._DIET_SEAFOOD_TERMS)
        huevo, lacteo = tuple(_go._DIET_EGG_TERMS), tuple(_go._DIET_DAIRY_TERMS)
        solo_vegano = tuple(__import__("vocabulario_dieta").SOLO_VEGANO)
    except Exception:                                                          # noqa: BLE001
        carne, mar = ("pollo", "res", "cerdo", "jamon", "salami", "carne"), ("pescado", "atun", "marisco")
        huevo, lacteo, solo_vegano = ("huevo",), ("leche", "queso", "yogur", "yogurt"), ("miel",)
    if canon == "pescatarian":
        return carne
    if canon == "vegetarian":
        return carne + mar + ("caldo",)
    return carne + mar + huevo + lacteo + solo_vegano + ("caldo",)


def etiqueta_desayuno(categoria, pais_cocina, detalle: bool = True, vetado=None, dieta=None) -> str:
    """Etiqueta que VE el modelo para la categoría de desayuno asignada.

    DO ⇒ la categoría tal cual. Beta ⇒ sólo la categoría A cambia: «Desayuno típico local (…)» con el desayuno típico del
    perfil de cocina del país (`cultural_profiles.PROFILES[...]['slot_affinity']`), SIN
      · lo que ya es la base de otra categoría (la A tiene que ser distinta de las otras cuatro) y la fruta, que va en
        todo desayuno,
      · las sopas (§15a beta),
      · lo que `vetado` (la puerta de alergias y rechazos de `day_generator._vetado`) o la `dieta` prohíben: la A es
        el refugio de `desayuno_por_alergia.reasignar`, y un «⚠️ OBLIGATORIO … (avena, huevo, pan)» a un alérgico al
        gluten y al huevo reabre el «ALÉRGENO DETECTADO» del lote 227. [ronda 2] La puerta pregunta también por la base
        del catálogo y la clase (`_nombres_para_la_puerta`): «arepa» a un alérgico al maíz, «frijoles» a quien declaró
        «legumbres», y ni el escáner ni el backstop lo marcan después.
    Si no queda nada ⇒ `ETIQUETA_A_BETA_RESPALDO` (también en el brief). `detalle=False` (el brief de los OTROS días) da
    sólo el nombre, sin la lista. `pais_cocina` es un código YA canónico: el de la cocina de ESE día."""
    if categoria != ENUM_DESAYUNO_A or es_do(pais_cocina):
        return categoria
    prohibidos = _BASES_DE_OTRAS_CATEGORIAS + _NO_ES_DESAYUNO + _NO_ES_BASE_PROPIA + _vetados_por_dieta(dieta)
    items = [i for i in _desayuno_tipico(pais_cocina) if not _contiene(i, prohibidos)]
    if vetado is not None:
        items = [i for i in items if not any(vetado(n) for n in _nombres_para_la_puerta(i))]
    if not items:
        return ETIQUETA_A_BETA_RESPALDO
    return f"{_ETIQUETA_A_BETA_CORTA} ({', '.join(items)})" if detalle else _ETIQUETA_A_BETA_CORTA
