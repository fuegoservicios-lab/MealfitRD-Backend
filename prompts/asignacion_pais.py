# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-748 · 2026-09-28] Las ramas por país del bloque DINÁMICO del día (G24, parte determinista).

`prompts.day_generator.build_day_assignment_context` es el tramo del prompt que el generador lee al final, el más
concreto — el que P1-DIET-BLIND-DIRECTIVES midió que gana a las cabeceras. La auditoría del 28-sep encontró ahí, para
ES/US/MX/PR/CO, cinco restos dominicanos: «en la mesa dominicana…», «casabe» (el catálogo beta lo excluye:
`graph_orchestrator._BETA_CATALOG_DO_EXCLUSIVE_NAMES`), «Salami dominicano», la categoría A del desayuno como
«Tubérculos/plátano» (a una española «plátano» es la BANANA) y el enum crudo «Mangú/Tubérculos» en el brief de los otros
días. Y el system prompt justificaba el «arroz de noche» con la cena dominicana y un gate que en beta sólo avisa.

Reglas de este módulo:
  · La rama DO NO pasa por aquí: el llamador conserva su literal y sólo consulta `es_do`. Byte-idéntico, con huellas
    sha256 en `tests/test_p1_plan_lote_748.py`.
  · Nada de un fragmento por país: lo local sale de DATOS que ya existen (el desayuno típico de
    `cultural_profiles.PROFILES`, la lista de exclusivos DO del catálogo). La profundidad por país (platos propios) es
    la parte de G24 que se valida con IA, no ésta.
  · El gate no cambia: el motivo del arroz de noche en beta dice lo que hace la regla blanda (señalar), no lo que no hace.

tooltip-anchor: P1-PLAN-LOTE-748
"""
from __future__ import annotations

# ─── (b) arroz de noche: el motivo, no la regla ───────────────────────────────────────────────────────────────────────
# El DO es el literal EXACTO de la regla d) CENA de `DAY_GENERATOR_SYSTEM_PROMPT`: `_BETA_FRAGMENT_TABLE` lo sustituye.
ARROZ_NOCHE_MOTIVO_DO = "(no se acostumbra en la cena dominicana y el gate lo rechaza)"
ARROZ_NOCHE_MOTIVO_BETA = "(es una regla de horario del plan y el validador de horario lo señala)"

# ─── (a) avena en las comidas fuertes ────────────────────────────────────────────────────────────────────────────────
# Sin «arepitas» (léxico DO: `constants._DO_LEXICON_NEUTRAL`) y sin «el plan se rechaza»: en beta la regla de horario
# de la cena es blanda y entrega con aviso desde el intento 1 (`_slot_appropriateness_advisory_decision`).
AVENA_CIERRE_BETA = ("Nada de tortitas, bowls salados ni «avena al caldo» en las comidas fuertes: eso se lee como un "
                     "desayuno fuera de hora, no como un almuerzo ni una cena, y el validador de horario lo señala.")

# ─── (a) Salami dominicano en las proteínas prohibidas ──────────────────────────────────────────────────────────────
# Sólo la ETIQUETA que ve el modelo; la clave y el emparejamiento contra el pool no cambian.
_ETIQUETAS_PROTEINA_BETA = {"Salami dominicano": "Salami"}

# ─── (c) la categoría A del desayuno ────────────────────────────────────────────────────────────────────────────────
ENUM_DESAYUNO_A = "Mangú/Tubérculos"            # valor del esquema (schemas.py `breakfast_category`): NO se toca
ETIQUETA_A_BETA_RESPALDO = "Tubérculos/plátano (preparación local)"   # la de antes, si el perfil no trae desayuno
_ETIQUETA_A_BETA_CORTA = "Desayuno típico local"
# El aviso de las otras cuatro categorías: en DO «NO uses mangú/tubérculos…»; en beta la A ya no es un tubérculo, y
# «NO uses tubérculo/plátano» le vetaba la banana del desayuno a quien lee «plátano» como banana.
AVISO_DESAYUNO_BETA = "NO la cambies por la categoría de otro día"

_EXCLUSIVOS_DO_RESPALDO = ("casabe",)


def es_do(country) -> bool:
    """La MISMA puerta que ya usa `build_day_assignment_context` (`canonicalize_country`, None ⇒ DO)."""
    try:
        from constants import canonicalize_country
        return canonicalize_country(country) == "DO"
    except Exception:                                                          # noqa: BLE001
        return True


def _exclusivos_do() -> tuple:
    """Los alimentos que el catálogo beta NO ofrece, en minúsculas. SSOT: la lista del catálogo cerrado."""
    try:
        import graph_orchestrator as _go
        nombres = tuple(str(n).casefold() for n in _go._BETA_CATALOG_DO_EXCLUSIVE_NAMES)
        return nombres or _EXCLUSIVOS_DO_RESPALDO
    except Exception:                                                          # noqa: BLE001
        return _EXCLUSIVOS_DO_RESPALDO


def sin_exclusivos_do(items, country) -> list:
    """Sugerencias sin lo que el catálogo beta no vende. DO ⇒ la lista tal cual (mismo contenido, mismo orden)."""
    items = list(items or [])
    if es_do(country):
        return items
    import re
    excl = _exclusivos_do()
    return [i for i in items
            if not any(re.search(rf"\b{re.escape(e)}\b", str(i).casefold()) for e in excl)]


def etiqueta_proteina(label: str, country) -> str:
    if es_do(country):
        return label
    return _ETIQUETAS_PROTEINA_BETA.get(label, label)


def _desayuno_tipico(country) -> list:
    try:
        from constants import canonicalize_country
        from cultural_profiles import PROFILES, profile_for_market
        perfil = PROFILES.get(profile_for_market(canonicalize_country(country))) or {}
        return [str(x).strip() for x in ((perfil.get("slot_affinity") or {}).get("desayuno") or []) if str(x).strip()]
    except Exception:                                                          # noqa: BLE001
        return []


def etiqueta_desayuno(categoria, country, detalle: bool = True) -> str:
    """Etiqueta que VE el modelo para la categoría de desayuno asignada.

    DO ⇒ la categoría tal cual. Beta ⇒ sólo la categoría A cambia: «Desayuno típico local (tostada, huevo, yogur,
    fruta)» con el desayuno típico del perfil de cocina del país (`cultural_profiles.PROFILES[...]['slot_affinity']`).
    `detalle=False` (el brief de los OTROS días) da sólo el nombre, sin la lista: la lista del día ajeno solapa con las
    categorías propias (huevo, tostada) y leída como «ya lo usa otro día» le quitaría al día su propia base.
    Sin desayuno en el perfil ⇒ la etiqueta neutra de antes."""
    if es_do(country) or categoria != ENUM_DESAYUNO_A:
        return categoria
    items = _desayuno_tipico(country)
    if not items:
        return ETIQUETA_A_BETA_RESPALDO
    return f"{_ETIQUETA_A_BETA_CORTA} ({', '.join(items)})" if detalle else _ETIQUETA_A_BETA_CORTA
