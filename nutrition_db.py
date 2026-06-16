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

import os
import re
from dataclasses import dataclass
from typing import Optional

from canonical_units import to_base_amount, canonicalize_unit

# [P4-UNIFIED-RESOLVER · 2026-06-14] Cuando los tiers baratos (exacto/alias) de este módulo fallan,
# delega al resolver canónico `shopping_calculator.normalize_name` (regex clean_n + fuzzy difflib +
# semántico Cohere v4) → un SOLO resolver para shopping Y nutrición. Mata el "0 silencioso" #1 (el
# ingrediente no resolvía al catálogo → aportaba 0 macros). Knob de rollback sin redeploy.
_UNIFIED_RESOLVER_ENABLED = (
    os.environ.get("MEALFIT_NUTRITION_UNIFIED_RESOLVER", "true").strip().lower()
    not in ("0", "false", "no", "off"))


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
# Hint de peso/volumen del LLM: "(50g)", "(240 ml)", "(140gr)". [P1-RESOLVER-COVERAGE · 2026-06-16]
# `[^)]*?` permite texto ANTES del número dentro del paréntesis ("(wrap, 60g)", "(aprox. 50g)",
# "(cocido, 150g)") — el número+unidad ya no tiene que ir pegado a "(". Usa el peso que el LLM mismo
# declaró (no adivina). Medido: cerraba lonjas tipo "tortilla de harina de trigo (wrap, 60g)".
_GRAM_HINT_RE = re.compile(r"\([^)]*?(\d+(?:[.,]\d+)?)\s*(g|gr|gramos|ml|mililitros)\b[^)]*\)", re.I)
# Hint SOLO en gramos (no ml). El peso explícito en g es la conversión más confiable del LLM → se
# prefiere SIEMPRE sobre un hint en ml en el mismo paréntesis ("(60 ml leche, 200g)" → 200g, no densidad).
_GRAM_ONLY_HINT_RE = re.compile(r"\([^)]*?(\d+(?:[.,]\d+)?)\s*(g|gr|gramos)\b[^)]*\)", re.I)
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
    out = str(s)
    # Número mixto: "1¼" → "1.25", "2½" → "2.5" (dígitos seguidos de fracción unicode).
    for k, v in _FRACTION_MAP.items():
        if k in out:
            out = re.sub(r"(\d+)\s*" + re.escape(k),
                         lambda m, _v=v: _fmt_num(int(m.group(1)) + float(_v)) + " ", out)
    # Fracción unicode standalone al inicio: "½ taza" → "0.5 taza".
    for k, v in _FRACTION_MAP.items():
        if out.startswith(k):
            out = v + " " + out[len(k):].lstrip()
            break
    return re.sub(r"\s{2,}", " ", out)  # colapsa dobles espacios del reemplazo


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


# ── Cuantización de porciones a unidades de cocina medibles (P3-PORTION-QUANTIZE) ──
_CUP_UNITS = ("taza", "tazas")
_SPOON_UNITS = ("cucharada", "cucharadas", "cucharadita", "cucharaditas",
                "cda", "cdta", "cdas", "cdtas")
_GRAM_UNITS = ("g", "gr", "gramo", "gramos", "ml", "mililitro", "mililitros",
               "kg", "litro", "litros", "l")
# Fracciones medibles permitidas por tipo (snap a la más cercana; 1.0 acarrea al entero
# siguiente). Tazas: ¼/⅓/½/⅔/¾ (existen tazas medidoras de esos tamaños). Cucharas: ¼/½/¾.
# Discretos (huevo/papa/rebanada): solo ½ o entero. Incluyen 0 para detectar el carry/floor.
_THIRD, _TWOTHIRD = 1.0 / 3.0, 2.0 / 3.0
_CUP_FRACS = (0.0, 0.25, _THIRD, 0.5, _TWOTHIRD, 0.75, 1.0)
_SPOON_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
_COUNT_FRACS = (0.0, 0.5, 1.0)


def _detect_kind(raw: str) -> str:
    """Tipo de unidad líder del ingrediente-string: cup/spoon/gram/count(discreto)."""
    s = _strip_accents(_normalize_unicode_fractions(str(raw).strip()).lower())
    m = _LEAD_QTY_RE.match(s)
    rest = s[m.end():] if m else s
    wm = re.match(r"\s*([a-z]+)", rest)
    w = wm.group(1) if wm else ""
    if w in _CUP_UNITS:
        return "cup"
    if w in _SPOON_UNITS or w.startswith("cucharad"):
        return "spoon"
    if w in _GRAM_UNITS:
        return "gram"
    return "count"


def _snap_qty(qty: float, allowed) -> float:
    """Snap `qty` a entero+fracción-permitida más cercana (1.0 acarrea). Nunca 0:
    una cantidad positiva minúscula sube al mínimo incremento positivo permitido."""
    whole = int(qty)
    frac = qty - whole
    best = min(allowed, key=lambda a: abs(a - frac))
    val = whole + best  # best=1.0 acarrea al entero siguiente de forma natural
    min_pos = min(a for a in allowed if a > 0)
    if val < min_pos:
        val = min_pos
    return round(val, 4)


def _integerize_gram_hint(s: str) -> str:
    """Redondea el hint "(NNg/ml)" a entero para que la referencia de báscula sea limpia."""
    def repl(m):
        val = float(m.group(1).replace(",", "."))
        return m.group(0).replace(m.group(1), str(int(round(val))), 1)
    return _GRAM_HINT_RE.sub(repl, s)


def quantize_ingredient_string(s: str):
    """[P3-PORTION-QUANTIZE · 2026-06-13] Redondea la cantidad líder de un ingrediente-
    string a un incremento medible en cocina (¼ taza, ¼ cda, ½ unidad discreta, 5 g) y
    escala el hint de gramos de forma coherente; integeriza el hint como referencia de
    báscula. Retorna (nuevo_string, factor) con factor = nueva_qty/qty_original (1.0 si
    no cambió) — el caller ajusta los macros del meal por el delta del aporte de ESE
    ingrediente. 'al gusto'/'opcional' → quita el número espurio sin tocar macros.

    Cierra el hallazgo de la auditoría clínica: '0.66 huevos', '3.87 papas', '0.53 taza',
    '3.74 rebanadas' no son pesables/medibles → matan la adherencia. Determinista y
    macro-consistente (vía delta). Anchor: P3-PORTION-QUANTIZE."""
    raw = str(s)
    norm = _normalize_unicode_fractions(raw.strip())
    m = _LEAD_QTY_RE.match(norm)
    if not m:
        return raw, 1.0
    qty = _frac_to_float(m.group(1))
    if qty is None or qty <= 0:
        return raw, 1.0
    low = _strip_accents(raw.lower())
    # 'al gusto'/'opcional': la cantidad es espuria → quítala, deja el texto legible.
    if "al gusto" in low or "a gusto" in low or "opcional" in low:
        cleaned = _LEAD_QTY_RE.sub("", norm, count=1).strip()
        if cleaned:
            cleaned = cleaned[:1].upper() + cleaned[1:]
            return cleaned, 1.0
        return raw, 1.0
    kind = _detect_kind(raw)
    if kind == "gram":
        new_qty = round(qty / 5.0) * 5.0  # gramos/ml → múltiplo de 5
        if new_qty <= 0:  # <2.5 g: ya es pesable en báscula de precisión, no inflar
            return _integerize_gram_hint(raw), 1.0
    elif kind == "cup":
        new_qty = _snap_qty(qty, _CUP_FRACS)
    elif kind == "spoon":
        new_qty = _snap_qty(qty, _SPOON_FRACS)
    else:  # count (discreto: huevo/papa/rebanada/fruta)
        new_qty = _snap_qty(qty, _COUNT_FRACS)
    if abs(new_qty - qty) < 1e-4:
        return _integerize_gram_hint(raw), 1.0
    factor = new_qty / qty
    return _integerize_gram_hint(rescale_ingredient_string(raw, factor)), factor


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
    # [P3-MICRONUTRIENTS] panel clínico por-100g (None si la fila USDA no lo reporta).
    vit_d_mcg: Optional[float] = None
    calcium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    b12_mcg: Optional[float] = None
    sugars_g: Optional[float] = None
    potassium_mg: Optional[float] = None
    # [P4-UNIFIED-RESOLVER] columnas nuevas para DASH (Mg) + dislipidemia (satfat/colesterol) + ERC (P).
    magnesium_mg: Optional[float] = None
    phosphorus_mg: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None


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

    # ---- matching nombre→row (tiers baratos + delegación unificada) -----
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
        # Tier 3 [P4-UNIFIED-RESOLVER]: delega al resolver canónico (regex clean_n + fuzzy + Cohere
        # semántico) que resuelve lo que los tiers baratos no. Solo con catálogo real (NO en rows
        # inyectados de test, que usan un catálogo distinto al de normalize_name → determinismo offline).
        if _UNIFIED_RESOLVER_ENABLED and not self._injected:
            try:
                from shopping_calculator import normalize_name
                canon = normalize_name(raw_name)
                if canon:
                    row = self._by_name.get(canon)
                    if row is None:  # normalize_name compara stripped; reintenta por nombre normalizado
                        cs = _strip_accents(str(canon).strip().lower())
                        for alias_stripped, r in self._aliases:
                            if alias_stripped == cs:
                                row = r
                                break
                    if row is not None:
                        return row
            except Exception:
                pass
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
            vit_d_mcg=_num(row.get("vitamin_d_mcg_per_100g")),
            calcium_mg=_num(row.get("calcium_mg_per_100g")),
            iron_mg=_num(row.get("iron_mg_per_100g")),
            b12_mcg=_num(row.get("vitamin_b12_mcg_per_100g")),
            sugars_g=_num(row.get("sugars_g_per_100g")),
            potassium_mg=_num(row.get("potassium_mg_per_100g")),
            magnesium_mg=_num(row.get("magnesium_mg_per_100g")),
            phosphorus_mg=_num(row.get("phosphorus_mg_per_100g")),
            saturated_fat_g=_num(row.get("saturated_fat_g_per_100g")),
            cholesterol_mg=_num(row.get("cholesterol_mg_per_100g")),
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
        mg = _GRAM_ONLY_HINT_RE.search(s)
        if mg:  # peso explícito en g gana siempre (incluso si hay un hint en ml en el mismo paréntesis)
            return float(mg.group(1).replace(",", "."))
        m = _GRAM_HINT_RE.search(s)  # sin hint en g → si hay ml, convertir por densidad
        if m:
            val = float(m.group(1).replace(",", "."))
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

    def micros_from_ingredient_string(self, s: str) -> Optional[dict]:
        """[P3-MICRONUTRIENTS] (string del plan) → panel de micros del aporte de ESE
        ingrediente {grams, fiber, sodium_mg, vit_d_mcg, calcium_mg, iron_mg, b12_mcg,
        sugars_g, potassium_mg} o None si no resuelve nombre/gramos. Cada micro es None si
        el ingrediente no lo reporta (manual/USDA sin dato) → el validador no lo contabiliza."""
        name = _split_qty_unit_name(s)[2]
        info = self.lookup(name)
        if not info:
            return None
        grams = self.grams_from_ingredient_string(s)
        if grams is None:
            return None
        f = grams / 100.0

        def _sc(x):
            return round(x * f, 3) if x is not None else None

        return {
            "grams": round(grams, 1),
            "fiber": _sc(info.fiber),
            "sodium_mg": _sc(info.sodium_mg),
            "vit_d_mcg": _sc(info.vit_d_mcg),
            "calcium_mg": _sc(info.calcium_mg),
            "iron_mg": _sc(info.iron_mg),
            "b12_mcg": _sc(info.b12_mcg),
            "sugars_g": _sc(info.sugars_g),
            "potassium_mg": _sc(info.potassium_mg),
            "magnesium_mg": _sc(info.magnesium_mg),
            "phosphorus_mg": _sc(info.phosphorus_mg),
            "saturated_fat_g": _sc(info.saturated_fat_g),
            "cholesterol_mg": _sc(info.cholesterol_mg),
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
