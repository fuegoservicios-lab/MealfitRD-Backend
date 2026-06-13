"""[P2-MDDA-NUTRITION-DB · 2026-06-13] Capa de lookup de macros nutricionales —
cimiento del lado DETERMINISTA del "cerebro dividido" (MDDA).

El LLM elige alimentos y crea recetas (creatividad). Este módulo + `portion_solver`
computan los macros REALES desde `master_ingredients.{kcal,protein,carbs,fats}_per_100g`
(poblados desde USDA por `scripts/populate_nutrition_db.py`) en vez de que el LLM los
adivine. Aquí va solo el LOOKUP (nombre→macros + conversión a gramos); el solver de
porciones vive en `portion_solver.py`.

Diseño:
  - `IngredientNutritionDB(rows=None)` — sin `rows` carga `get_master_ingredients()`
    (cache TTL del shopping_calculator); con `rows` inyectados es 100% offline-testable.
  - `lookup(nombre_llm)` resuelve el string del LLM al row canónico vía matching
    exacto→alias→regex (mismas tiers que `normalize_name` tiers 1-4, SIN el fallback
    semántico/embeddings — queremos lookup barato y determinista; los ingredientes del
    plan vienen de nuestro propio catálogo, así que exact/alias/regex los cubre).
  - Degradación grácil: sin match o sin macros poblados → `None`. El caller decide
    (el solver deja ese ingrediente tal cual, no inventa números).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from canonical_units import to_base_amount, canonicalize_unit


def _strip_accents(s: str) -> str:
    try:
        from constants import strip_accents
        return strip_accents(s)
    except Exception:  # pragma: no cover - fallback defensivo si constants cambia
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                       if not unicodedata.combining(c))


def _num(x) -> Optional[float]:
    """Castea Decimal/str/int a float; None/no-numérico → None."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ── Parsing/rescaling de ingredientes-string del plan ("0.5 taza de avena (50g)") ──
_FRACTION_MAP = {"½": "0.5", "¼": "0.25", "¾": "0.75", "⅓": "0.3333",
                 "⅔": "0.6667", "⅕": "0.2", "⅛": "0.125"}
# Hint de peso/volumen del LLM: "(50g)", "(240 ml)", "(140gr)".
_GRAM_HINT_RE = re.compile(r"\((\d+(?:[.,]\d+)?)\s*(g|gr|gramos|ml|mililitros)\b[^)]*\)", re.I)
# Cantidad líder al inicio del string: "0.5", "1/2", "150", "1,5".
_LEAD_QTY_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)\s*")
_UNIT_TOKEN_RE = re.compile(r"^\s*([a-záéíóúñ]+)\b", re.I)


def _frac_to_float(tok: str) -> Optional[float]:
    tok = tok.strip().replace(",", ".")
    if "/" in tok:
        try:
            a, b = tok.split("/")
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(tok)
    except ValueError:
        return None


def _fmt_num(x: float) -> str:
    """Formatea un número escalado: entero si es casi entero, si no 2 decimales."""
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _normalize_unicode_fractions(s: str) -> str:
    for k, v in _FRACTION_MAP.items():
        if s.startswith(k):
            return v + " " + s[len(k):].lstrip()
    return s


def _strip_qty_prefix(s: str) -> str:
    """Quita cantidad líder + unidad + 'de' para dejar el nombre (best-effort)."""
    s2 = _normalize_unicode_fractions(str(s).strip())
    s2 = _LEAD_QTY_RE.sub("", s2, count=1)
    from canonical_units import canonicalize_unit
    mu = _UNIT_TOKEN_RE.match(s2)
    if mu and canonicalize_unit(mu.group(1)):
        s2 = s2[mu.end():].lstrip()
    s2 = re.sub(r"^(de|del)\s+", "", s2, flags=re.I)
    return s2.strip()


def _split_qty_unit_name(s: str):
    """(string del plan) → (qty: float, unit: str, name: str). Lightweight, offline.
    qty 0.0 si no hay número líder (e.g. 'Sal al gusto')."""
    from canonical_units import canonicalize_unit
    raw = _normalize_unicode_fractions(str(s).strip())
    mq = _LEAD_QTY_RE.match(raw)
    if not mq:
        return 0.0, "unidad", _strip_qty_prefix(s)
    qty = _frac_to_float(mq.group(1)) or 0.0
    rest = raw[mq.end():]
    mu = _UNIT_TOKEN_RE.match(rest)
    unit = "unidad"
    if mu and canonicalize_unit(mu.group(1)):
        unit = mu.group(1).lower()
        rest = rest[mu.end():].lstrip()
    rest = re.sub(r"^(de|del)\s+", "", rest, flags=re.I)
    name = _GRAM_HINT_RE.sub("", rest).strip()
    return qty, unit, name


def rescale_ingredient_string(s: str, factor: float) -> str:
    """Re-escala la cantidad líder Y el hint "(NNg/ml)" de un ingrediente-string por
    `factor`, preservando el resto del texto. Items sin cantidad numérica ('Sal al
    gusto') quedan intactos. Mantiene recipe↔shopping↔display consistentes (ambos
    lados parsean la misma cantidad líder)."""
    if not s or abs(factor - 1.0) < 1e-6:
        return s
    out = _normalize_unicode_fractions(str(s).strip())
    mq = _LEAD_QTY_RE.match(out)
    if not mq:
        return s  # sin cantidad líder → no tocar (al gusto, etc.)
    lead = _frac_to_float(mq.group(1))
    if lead is None or lead <= 0:
        return s
    new_lead = _fmt_num(lead * factor)
    out = out[:mq.start(1)] + new_lead + out[mq.end(1):]

    def _scale_hint(m):
        val = float(m.group(1).replace(",", "."))
        return m.group(0).replace(m.group(1), _fmt_num(val * factor), 1)

    out = _GRAM_HINT_RE.sub(_scale_hint, out)
    return out


@dataclass
class NutritionInfo:
    """Macros por-100g de un ingrediente canónico (más metadata de conversión)."""
    name: str                       # nombre canónico de master_ingredients
    kcal: float                     # por 100g (Atwater 4/4/9)
    protein: float
    carbs: float
    fats: float
    fiber: Optional[float] = None
    sodium_mg: Optional[float] = None
    source: Optional[str] = None    # usda | manual | off | faoinfoods
    fdc_id: Optional[int] = None
    is_dominican: bool = False
    density_g_per_unit: Optional[float] = None
    density_g_per_cup: Optional[float] = None
    container_weight_g: Optional[float] = None


class IngredientNutritionDB:
    def __init__(self, rows: Optional[list] = None):
        self._injected = rows is not None
        self._rows = rows
        self._by_name: dict = {}
        self._aliases: list = []  # [(alias_stripped_lower, row)] orden desc por longitud
        if rows is not None:
            self._build_index(rows)

    # ---- carga / índice -------------------------------------------------
    def _ensure_loaded(self):
        if self._rows is None:
            try:
                from shopping_calculator import get_master_ingredients
                self._rows = get_master_ingredients() or []
            except Exception:
                self._rows = []
            self._build_index(self._rows)

    def _build_index(self, rows: list):
        self._by_name = {}
        aliases = []
        for row in rows or []:
            name = row.get("name")
            if not name:
                continue
            self._by_name[name] = row
            aliases.append((_strip_accents(str(name).strip().lower()), row))
            for alias in (row.get("aliases") or []):
                aliases.append((_strip_accents(str(alias).strip().lower()), row))
        # más largos primero: evita que 'platano' se trague 'platano maduro'
        aliases.sort(key=lambda x: len(x[0]), reverse=True)
        self._aliases = aliases

    # ---- matching nombre→row (tiers 1-4, sin semántico) -----------------
    def _match_row(self, raw_name: str) -> Optional[dict]:
        self._ensure_loaded()
        if not raw_name:
            return None
        n = re.sub(r"\(.*?\)", "", str(raw_name).lower()).strip()
        n_stripped = _strip_accents(n)
        # Tier 1: match exacto sobre el texto crudo
        for alias_stripped, row in self._aliases:
            if n_stripped == alias_stripped:
                return row
        # Tier 2: alias como palabra dentro del texto (word-boundary)
        for alias_stripped, row in self._aliases:
            if alias_stripped and re.search(r"\b" + re.escape(alias_stripped) + r"\b", n_stripped):
                return row
        return None

    def lookup(self, raw_name: str) -> Optional[NutritionInfo]:
        """Resuelve string del LLM → NutritionInfo, o None si no hay match o
        el row no tiene macros poblados (kcal_per_100g IS NULL)."""
        row = self._match_row(raw_name)
        if not row:
            return None
        kcal = _num(row.get("kcal_per_100g"))
        if kcal is None:
            return None  # sin macros poblados → degradar (no inventar)
        return NutritionInfo(
            name=row.get("name"),
            kcal=kcal,
            protein=_num(row.get("protein_g_per_100g")) or 0.0,
            carbs=_num(row.get("carbs_g_per_100g")) or 0.0,
            fats=_num(row.get("fats_g_per_100g")) or 0.0,
            fiber=_num(row.get("fiber_g_per_100g")),
            sodium_mg=_num(row.get("sodium_mg_per_100g")),
            source=row.get("nutrition_source"),
            fdc_id=row.get("fdc_id"),
            is_dominican=bool(row.get("is_dominican_cultivar")),
            density_g_per_unit=_num(row.get("density_g_per_unit")),
            density_g_per_cup=_num(row.get("density_g_per_cup")),
            container_weight_g=_num(row.get("container_weight_g")),
        )

    # ---- conversión a gramos -------------------------------------------
    def to_grams(self, qty, unit, info: NutritionInfo) -> Optional[float]:
        """(qty, unit) → gramos comestibles usando densidades del master.
        Retorna None cuando no se puede convertir con confianza (container sin
        peso, unidad discreta sin densidad) — el caller deja el ingrediente tal cual."""
        q = _num(qty)
        if q is None or q <= 0:
            return None
        base_qty, base_unit = to_base_amount(q, unit)
        if base_unit == "g":
            return base_qty
        if base_unit == "ml":
            # volumen → g requiere densidad del alimento (g/ml).
            if info.density_g_per_cup:
                return base_qty * (info.density_g_per_cup / 240.0)
            return None  # sin densidad volumétrica conocida → no adivinar
        # No convertible por to_base_amount (unidad/diente/rebanada/paquete/…)
        canonical = canonicalize_unit(unit) or "unidad"
        if canonical in ("unidad", "rebanada", "hoja", "diente"):
            if info.density_g_per_unit:
                return q * info.density_g_per_unit
            return None
        if canonical in ("paquete", "caja", "bolsa", "lata", "pote", "botella"):
            if info.container_weight_g:
                return q * info.container_weight_g
            return None
        return None

    # ---- soporte para ingredientes-string del plan (F3) ----------------
    def grams_from_ingredient_string(self, s: str) -> Optional[float]:
        """Gramos comestibles de un ingrediente-string del plan ("0.5 taza de
        avena (50g)"). Prioriza el hint "(NNg)" del LLM (su propia conversión, lo
        más confiable); si es "(NNml)" usa densidad volumétrica; sin hint, parsea
        cantidad+unidad y usa `to_grams`. None si no se resuelve."""
        s = str(s)
        m = _GRAM_HINT_RE.search(s)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2).lower()
            if unit.startswith("g"):  # g / gr / gramos
                return val
            info = self.lookup(_strip_qty_prefix(s))  # ml → densidad del alimento
            if info and info.density_g_per_cup:
                return val * (info.density_g_per_cup / 240.0)
            return val  # ~1 g/ml (agua/leche); error pequeño para aceite ~0.92
        qty, unit, name = _split_qty_unit_name(s)
        info = self.lookup(name)
        if not info:
            return None
        return self.to_grams(qty, unit, info)

    def macros_from_ingredient_string(self, s: str) -> Optional[dict]:
        """(string del plan) → {name, grams, kcal, protein, carbs, fats, …} o None."""
        name = _split_qty_unit_name(s)[2]
        info = self.lookup(name)
        if not info:
            return None
        grams = self.grams_from_ingredient_string(s)
        if grams is None:
            return None
        f = grams / 100.0
        return {
            "name": info.name, "grams": round(grams, 2),
            "kcal": round(info.kcal * f, 1), "protein": round(info.protein * f, 2),
            "carbs": round(info.carbs * f, 2), "fats": round(info.fats * f, 2),
        }

    def macros_for_line(self, qty, unit, raw_name: str) -> Optional[dict]:
        """Atajo: (qty, unit, nombre) → {grams, kcal, protein, carbs, fats, …}
        o None si no se resuelve nombre o gramos."""
        info = self.lookup(raw_name)
        if not info:
            return None
        grams = self.to_grams(qty, unit, info)
        if grams is None:
            return None
        f = grams / 100.0
        return {
            "name": info.name,
            "grams": round(grams, 2),
            "kcal": round(info.kcal * f, 1),
            "protein": round(info.protein * f, 2),
            "carbs": round(info.carbs * f, 2),
            "fats": round(info.fats * f, 2),
            "fiber": round((info.fiber or 0.0) * f, 2),
            "sodium_mg": round((info.sodium_mg or 0.0) * f, 1),
            "source": info.source,
        }
