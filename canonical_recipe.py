# backend/canonical_recipe.py
"""[P1-ARQ30-F4-CANONICAL · 2026-09-06] `IngredientLine`: la línea de ingrediente como DATO.

Primera rebanada de `ARQ30-P1-01`. El gap propone una autoridad única sobre la composición
—`IngredientLine` / `RecipeVariant` / `PlanRevision`— de la que se deriven texto y compras. Eso es
una migración de las que el propio encargo describe como **expand → verificación/canary → promoción
→ retirada**, y dice, literalmente, que no se borre la vía anterior en el mismo paso que transfiere
su autoridad.

Esto es el `expand`, y **nada más**: una representación de SOLO LECTURA que no cambia ni una línea
de lo que se entrega hoy. Nadie escribe planes a través de ella todavía.

## Lo que NO hace, a propósito

**No parsea de cero.** La máquina ya existe y es intrincada: `shopping_calculator._parse_quantity`
resuelve fracciones, pistas de gramos y rendimiento, y `_calculate_yield_multiplier` tiene cuatro
reglas cuyo efecto depende de qué llamador pregunte (el agregador de compras y la Nevera necesitan
respuestas distintas para la MISMA línea, y eso está razonado en `P1-2` y `P2-PDF-1`). Un quinto
parser aquí sería la deriva que este repo ya pagó con `canonicalize_diet_type` y con
`pantry_names_match`. Este módulo **compone**: pregunta y registra.

**No decide gramos.** Guarda lo que la autoridad respondió, junto con QUIÉN respondió. Cuando
`ARQ30-P1-04` unifique la autoridad de cantidades, cambia el proveedor y no el contrato.

**No inventa el estado.** `seco/cocido/escurrido/crudo` sale del MISMO vocabulario que ya mueve el
rendimiento; si el texto no lo dice, el estado es `desconocido` — que **no es lo mismo que crudo**.
La distinción entre ausente y valor es la invariante que este repo ya pagó dos veces (`int(x or -1)`
con `attempts=0`, y el nutriente ausente leído como cero).

## Alcance: compras y macros. **El texto del usuario NO.**

El gap proponía derivar también el texto de esta representación. Se midió primero, sobre **11.073
líneas de 96 planes vivos**, y el número decidió lo contrario:

    roundtrip exacto ....... 0,5 %   (50 de 11.073)
    roundtrip equivalente .. 17,8 %  (1.973)
    cantidad conservada .... 96,3 %  (10.667)

Las cantidades sobreviven; **la prosa no**. Lo que se pierde no es formato:

    3 huevos                  →  3 unidad de Huevo
    45 g de melón en cubos    →  45 g de Melón        ← «en cubos» desaparece
    3 dientes de ajo          →  3 diente de Ajo
    1¼ cucharadas de cilantro →  1.25 cda de Cilantro

El CORTE es información culinaria de la que dependen los pasos de la receta; el plural natural y el
tamaño («mediana») son lo que hace que una receta se lea como escrita por alguien. Derivar el texto
de aquí degradaría visiblemente todas las recetas.

**Decisión del dueño (2026-09-06): la representación sirve para COMPRAS y MACROS.** El texto sigue
siendo del LLM y de la tubería que ya lo escribe. Es también el alcance donde el 96,3 % dice que la
representación sí aguanta.

Un modelo que no aspira a reproducir la prosa no necesita cargar con ella — y decir esto por escrito
evita que alguien, dentro de seis meses, «complete» el roundtrip creícndolo una tarea pendiente.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

ESTADO_DESCONOCIDO = "desconocido"

#: Vocabulario de estado. Sale del mismo sitio que el rendimiento
#: (`shopping_calculator._calculate_yield_multiplier`): si una palabra no mueve el yield, no es un
#: estado a efectos del motor. No es una tabla nueva — es la lectura de la que ya manda.
_ESTADOS = (
    ("escurrido", ("escurrid", "drenad", "colad")),
    ("cocido", ("cocid", "cocinad", "hervid", "asad", "salteado", "guisad", "horneado", "a la plancha")),
    ("seco", ("seco", "seca", "secos", "secas", "crudo de paquete", "sin cocinar")),
    ("crudo", ("crudo", "cruda", "crudos", "crudas", "fresco", "fresca")),
)

_RE_ESPACIOS = re.compile(r"\s+")
#: Una cifra con unidad de masa/volumen al final del NOMBRE es la pista que el parser ya movió
#: a `qty`; dejarla dentro parte la identidad del alimento en dos.
_RE_GRAM_HINT = re.compile(r"[,(]?\s*[\u2248~]?\s*\d+(?:[.,]\d+)?\s*(?:gramos|gramo|gr|kg|ml|g|l)\b\s*\)?\s*$",
                           re.IGNORECASE)


@dataclass(frozen=True)
class IngredientLine:
    """Una línea de ingrediente, con su procedencia. `None` significa *no se pudo determinar*."""

    raw: str
    name: str
    ingredient_id: str
    qty: Optional[float] = None
    unit: Optional[str] = None
    grams: Optional[float] = None
    state: str = ESTADO_DESCONOCIDO
    #: Quién respondió a los gramos. Cuando ARQ30-P1-04 unifique la autoridad, cambia esto.
    grams_source: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _estado_de(texto: str) -> str:
    from constants import strip_accents
    t = strip_accents(str(texto or "").lower())
    for estado, marcas in _ESTADOS:
        if any(m in t for m in marcas):
            return estado
    return ESTADO_DESCONOCIDO


def parse_line(raw: Any, nutrition_db: Any = None) -> Optional[IngredientLine]:
    """Una línea de texto → `IngredientLine`, preguntando a las autoridades que ya existen.

    Devuelve `None` si la línea no es un ingrediente con cantidad («Sal al gusto»): eso no es un
    fallo, es una línea sin cantidad, y tratarla como cero gramos sería inventar.

    `nutrition_db` es una instancia de `IngredientNutritionDB` OPCIONAL. Se pasa desde fuera a
    propósito: construirla por línea costaría un lookup de catálogo cada vez, y quien recorre un plan
    entero debe pasar la misma. Sin ella, `grams` queda en `None` cuando la unidad no es de masa — y
    `None` significa «no se pudo determinar», nunca cero."""
    texto = str(raw or "").strip()
    if not texto:
        return None
    try:
        from shopping_calculator import _parse_quantity
        # `apply_yield_multiplier=False`: aquí se registra lo que el TEXTO dice, no lo que habría que
        # comprar. Aplicar el rendimiento es una decisión del agregador, y depende de quién pregunte.
        qty, unit, name = _parse_quantity(texto, apply_yield_multiplier=False)
        # «1 batata mediana cocida, 250 g»: la pista de gramos se colaba DENTRO del nombre
        # (`ingredient_id` salía `batata_250_g`). `_reconcile_qty_with_gram_hint` existe justo para
        # esto y ya lo usa el agregador — no hay que reimplementarlo aquí.
        try:
            from shopping_calculator import _reconcile_qty_with_gram_hint
            _q2, _u2 = _reconcile_qty_with_gram_hint(texto, qty, unit)
            if _q2 is not None:
                qty, unit = _q2, _u2
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"[ARQ30-P1-01] no se pudo parsear {texto[:50]!r}: {e!r}")
        return None

    name = _RE_ESPACIOS.sub(" ", str(name or "").strip())
    # La pista de gramos se queda DENTRO del nombre: «batata mediana cocida, 250 g» produce el id
    # `batata_250_g`. `_reconcile_qty_with_gram_hint` corrige la CANTIDAD pero no limpia el texto, así
    # que dos líneas del mismo alimento con pistas distintas dan ids distintos. Esto no es un parser
    # nuevo: es quitar del nombre lo que el parser ya extrajo a `qty`.
    name = _RE_GRAM_HINT.sub("", name).strip(" ,;·-")
    name = _RE_ESPACIOS.sub(" ", name).strip()
    if not name:
        return None

    try:
        from plan_policy import ingredient_id_for
        iid = ingredient_id_for(name)
    except Exception:
        iid = ""

    unit_canon, gramos, fuente = None, None, None
    try:
        from canonical_units import canonicalize_unit, to_base_amount
        unit_canon = canonicalize_unit(unit) if unit else None
        if qty is not None and unit_canon:
            base, base_unit = to_base_amount(qty, unit_canon)
            if base is not None and str(base_unit or "").lower() in ("g", "gramo", "gramos"):
                gramos, fuente = float(base), "canonical_units.to_base_amount"
    except Exception as e:
        logger.debug(f"[ARQ30-P1-01] unidad no canonicalizable en {texto[:50]!r}: {e!r}")

    # Solo el 19,9 % de las lineas vivas dan gramos por la unidad sola: una taza de espinacas, tres
    # huevos o dos rebanadas de pan necesitan la DENSIDAD del catalogo. Esa conversion ya tiene
    # duenno —`nutrition_db.IngredientNutritionDB.to_grams`— y ademas lleva dentro la leccion de
    # `P1-UNKNOWN-UNIT-NOT-WHOLE` (una unidad desconocida NO es una unidad entera del alimento: para
    # una hierba eso es el mazo, y de ahi salian los 415 g de cebollin). Preguntar es lo correcto;
    # reimplementarlo aqui seria el quinto conversor.
    if gramos is None and qty is not None and nutrition_db is not None:
        try:
            info = nutrition_db.lookup(name)
            if info is not None:
                g2 = nutrition_db.to_grams(float(qty), unit_canon or unit or "", info)
                if g2 is not None and g2 > 0:
                    gramos, fuente = float(g2), "nutrition_db.to_grams"
        except Exception as e:
            logger.debug(f"[ARQ30-P1-01] densidad no disponible para {name[:40]!r}: {e!r}")

    return IngredientLine(
        raw=texto, name=name, ingredient_id=iid,
        qty=(float(qty) if qty is not None else None),
        unit=unit_canon or (str(unit) if unit else None),
        grams=gramos, state=_estado_de(texto), grams_source=fuente,
    )


def render_line(line: IngredientLine) -> str:
    """Texto DE DIAGNÓSTICO a partir de una `IngredientLine`. **No es la línea del usuario.**

    ⚠️ Nada de producción debe escribir con esto. Medido sobre 11.073 líneas vivas, el roundtrip
    reproduce el original en el 0,5 % de los casos y el 17,8 % ya normalizado: se pierde el corte
    («en cubos»), el plural natural («3 huevos» → «3 unidad de Huevo») y el tamaño («mediana»). Por
    eso el alcance quedó acotado a compras y macros — ver la cabecera del módulo.

    Existe para poder LEER una `IngredientLine` en una sonda o en un log, y para que
    `scripts/canonical_roundtrip.py` pueda seguir midiendo esa distancia: si algún día sube, será
    porque el modelo ganó los campos que hoy le faltan.

    El espacio entre cifra y unidad es load-bearing igualmente: `P1-CLOSER-LINE-SPANISH`."""
    if line is None:
        return ""
    if line.qty is None or not line.unit:
        return line.name
    cant = f"{line.qty:g}"
    # El estado NO se re-añade si el nombre ya lo trae. `_parse_quantity` deja el calificativo dentro
    # del nombre («pasta integral seca»), así que anexarlo producía «pasta integral seca seco» — el
    # render inventaba una palabra que el texto original no tenía. Se compara sin acentos porque el
    # texto vivo mezcla «cocida» y «cocido» y son el mismo estado.
    estado = ""
    if line.state != ESTADO_DESCONOCIDO:
        try:
            from constants import strip_accents
            marcas = dict(_ESTADOS)[line.state]
            if not any(m in strip_accents(line.name.lower()) for m in marcas):
                estado = f" {line.state}"
        except Exception:
            estado = f" {line.state}"
    return f"{cant} {line.unit} de {line.name}{estado}"


def parse_meal(meal: dict, nutrition_db: Any = None) -> list:
    """Las líneas de una comida. Las que no traen cantidad se descartan, no se rellenan con ceros."""
    salida = []
    for raw in ((meal or {}).get("ingredients") or []):
        linea = parse_line(raw, nutrition_db)
        if linea is not None:
            salida.append(linea)
    return salida


def shopping_view(line: IngredientLine) -> dict:
    """Lo que compras y macros necesitan de una línea, y nada más.

    Este es el contrato del alcance acordado. `None` en `grams` significa que la cantidad no es
    convertible a masa desde el texto (una taza de espinacas necesita la densidad del catálogo), NO
    que sean cero gramos — quien consuma esto tiene que preguntar, no asumir.
    """
    if line is None:
        return {}
    return {"ingredient_id": line.ingredient_id, "name": line.name,
            "qty": line.qty, "unit": line.unit, "grams": line.grams,
            "state": line.state, "grams_source": line.grams_source}


__all__ = ["IngredientLine", "parse_line", "render_line", "parse_meal", "shopping_view",
           "ESTADO_DESCONOCIDO"]
