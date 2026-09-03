# [P1-URGENT-LIST-CANONICAL · 2026-08-09] La lista del owner (f380821a) tenía 104 ítems y 33
# eran LÍNEAS DE RECETA crudas («95 g de mango en cubos», «1 cdta de pimentón») coladas como
# pseudo-productos: la inyección de «🚨 Compra Urgente» metía `_pantry_supplement_required`
# VERBATIM sin pasar por el agregador. Y cada comida del Dashboard mostraba una pared roja de
# 27 faltantes idénticos (hígado y mejillones «faltando» en un bowl de avena) porque el chunk
# worker estampaba la UNIÓN del chunk en cada meal. Tres invariantes:
#   1. los urgentes entran a la lista CANONICALIZADOS (agregador SSOT, duplicados fundidos);
#   2. cada meal recibe SOLO sus propios faltantes (matching normalizado contra sus líneas);
#   3. fail-open: si el agregador falla, inyección cruda (mejor crudo que ausente — seguridad).
import os
import sys


def _cuerpo_js(src: str, decl: str) -> str:
    """Cuerpo de una función JS emparejando llaves desde su declaración.

    [2026-08-14] Sustituye a las ventanas de N caracteres: el bloque crece con cada
    comentario legítimo y el guard acaba fallando por unas decenas de bytes, acusando
    a producción de haber borrado algo que sigue ahí."""
    i = src.index(decl)
    a = src.index("{", i)
    prof = 0
    for k in range(a, len(src)):
        if src[k] == "{":
            prof += 1
        elif src[k] == "}":
            prof -= 1
            if prof == 0:
                return src[i:k + 1]
    raise AssertionError(f"no se cerró el cuerpo de {decl!r}")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

_SC = open(os.path.join(os.path.dirname(__file__), "..", "shopping_calculator.py"),
           encoding="utf-8").read()
_CT = open(os.path.join(os.path.dirname(__file__), "..", "cron_tasks.py"),
           encoding="utf-8").read()

# `import cron_tasks` arrastra la cadena agent→ChatGLM que el venv-test rompe. Mismo
# contrato contenido que test_p1_swap_macro_repair: stub + import fresco en setup_module,
# RESTAURACIÓN del mundo exacto en teardown (lección tree9: purgar sin restaurar contaminó
# ~600 veredictos de la suite).
import importlib  # noqa: E402

import llm_provider as _lp  # noqa: E402


class _StubLLM:
    def __init__(self, *a, **k):
        pass

    def bind_tools(self, *a, **k):
        return self

    def with_structured_output(self, *a, **k):
        return self


ct = None
_saved = {}


def setup_module(module):
    global ct
    _saved["cron_tasks"] = sys.modules.get("cron_tasks")
    _saved["agent"] = sys.modules.get("agent")
    _saved["llm"] = getattr(_lp, "ChatGLM", None)
    _lp.ChatGLM = _StubLLM
    sys.modules.pop("cron_tasks", None)
    ct = importlib.import_module("cron_tasks")
    module.ct = ct
    assert hasattr(ct, "_meal_scoped_missing"), "import fresco sin el helper — revisar entorno"


def teardown_module(module):
    if _saved.get("llm") is not None:
        _lp.ChatGLM = _saved["llm"]
    for _m in ("cron_tasks", "agent"):
        if _saved.get(_m) is not None:
            sys.modules[_m] = _saved[_m]
        elif _m == "cron_tasks":
            sys.modules.pop(_m, None)


def test_inyeccion_pasa_por_el_agregador():
    # Estructural: el bloque de urgentes invoca aggregate_and_deduct_shopping_list ANTES de
    # inyectar, con fail-open a la inyección cruda.
    i = _SC.index("P1-URGENT-LIST-CANONICAL")
    win = _SC[i:i + 4000]
    assert "aggregate_and_deduct_shopping_list(" in win, (
        "los urgentes deben canonicalizarse por el agregador SSOT — verbatim infló la lista "
        "del owner a 104 ítems con 33 pseudo-productos")
    assert "_raw_urgent" in win and "fail-open" in win.lower() or "_urgent_entries is None" in win, (
        "sin fail-open, un fallo del agregador borraría compras de SEGURIDAD del modo flexible")


def test_scoping_por_comida_existe_y_se_usa():
    assert hasattr(ct, "_meal_scoped_missing")
    i = _CT.index("_meal_scoped_missing(")
    # el call site del chunk worker usa el scoping (no la unión entera)
    j = _CT.index("_meal_scoped_missing(", i + 10)
    win = _CT[max(0, j - 1500):j]
    assert "_missing_ingredients" in _CT[j:j + 400], (
        "el chunk worker debe estampar el resultado SCOPED, no missing_list entero")


def test_scoping_funcional():
    missing = ["95 g de mango en cubos", "105 g de hígado de res en tiras",
               "1 cdta de pimentón", "35g de sardinas en lata"]
    bowl = {"name": "Bowl de avena con mango",
            "ingredients": ["40 g de avena", "95 g de mango en cubos", "1 cdta de chía"]}
    own = ct._meal_scoped_missing(bowl, missing)
    assert own == ["95 g de mango en cubos"], (
        f"el bowl solo carece de SU mango — no del hígado de otra cena (got: {own})")
    locrio = {"name": "Locrio de sardinas",
              "ingredients": ["60 g de arroz", "35 g de sardinas en lata", "1 cdta de pimentón"]}
    own2 = ct._meal_scoped_missing(locrio, missing)
    assert "35g de sardinas en lata" in own2, "espacio distinto (35g vs 35 g) debe matchear igual"
    assert "1 cdta de pimentón" in own2
    assert "95 g de mango en cubos" not in own2


def test_scoping_fail_safe():
    assert ct._meal_scoped_missing({}, ["x"]) == []
    assert ct._meal_scoped_missing({"ingredients": None}, ["x"]) == []
    assert ct._meal_scoped_missing(None, ["x"]) == []


def test_restock_nace_categorizado():
    # Las 49 filas del restock del owner nacieron con category NULL → todas caían en la
    # pestaña Alacena mientras el header contaba 49 y «Nevera» decía vacía. El INSERT del
    # inventario debe llevar la categoría del master (ya resuelto en la misma función).
    di = open(os.path.join(os.path.dirname(__file__), "..", "db_inventory.py"),
              encoding="utf-8").read()
    i = di.index("INSERT INTO user_inventory")
    win = di[i:i + 900]
    assert "category" in win.split("VALUES")[0], "el INSERT debe incluir la columna category"
    assert "category = COALESCE(EXCLUDED.category, user_inventory.category)" in win, (
        "on-conflict no debe borrar una categoría existente con NULL")
    assert "master_category" in di[i:i + 1400], "la categoría viene del master ya resuelto"


def test_badge_rojo_se_evalua_en_vivo():
    # La caja «Compra Urgente Requerida» era una FOTO de generación: el owner compró la lista
    # entera y los avisos seguían. El badge debe filtrar contra la Nevera VIVA
    # (filterStillMissing: subconjunto por token, jamás substring — «sal» no absuelve «salsa»).
    dj = open(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src",
                           "pages", "Dashboard.jsx"), encoding="utf-8").read()
    assert "const filterStillMissing" in dj or "filterStillMissing =" in dj
    i = dj.index("PANTRY UNSAFE BADGE")
    # [2026-08-14] La ventana era de 1.200 caracteres fijos. P1-URGENT-FLASH-UNKNOWN
    # (13-ago) metió dentro del bloque los TRES estados (cargando / fetch caído / array)
    # con su comentario, y los marcadores se fueron al 1.595-1.969: el test dijo «el
    # badge dejó de evaluar en vivo» sobre un badge que lo hace más fino que antes.
    # El límite pasa a ser el BLOQUE: desde el ancla hasta el cierre del IIFE que lo
    # renderiza (`})()}` al final del `&& (() => {`). No envejece con el contenido.
    _fin = dj.find("})()}", i)
    assert _fin > i, "no se encontró el cierre del IIFE del badge"
    win = dj[i:_fin]
    assert "filterStillMissing(" in win, (
        "el badge debe evaluar los faltantes contra el inventario VIVO, no la foto de generación")
    assert "_still.length === 0" in win and "return null" in win, (
        "Nevera cubre todo → sin caja roja")
    # [2026-08-14] Era `dj[j:j+1400]` desde `const _missingNormTokens` y el `every(...)`
    # quedó en el 1.453: falló por 53 caracteres cuando P1-URGENT-FLASH-UNKNOWN documentó
    # los tres estados dentro de la función. El matching por tokens NUNCA se movió.
    # Ahora el límite es el CUERPO de `filterStillMissing`, emparejando llaves — exacto y
    # sin número mágico que caduque con el próximo comentario.
    cuerpo = _cuerpo_js(dj, "const filterStillMissing")
    assert "_missingNormTokens" in cuerpo, "filterStillMissing dejó de tokenizar el nombre"
    assert "every(t => foodTokens.has(t))" in cuerpo, (
        "matching por SUBCONJUNTO de tokens completos — substring reintroduciría la 15ª clase")


def test_frontend_ceil_de_empaques():
    dj = open(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src",
                           "pages", "Dashboard.jsx"), encoding="utf-8").read()
    i = dj.index("_WEIGHT_UNITS")
    win = dj[i:i + 900]
    assert "Math.ceil" in win, (
        "las unidades CONTABLES deben redondear a empaque entero — «0.87 funda» no es comprable")
    assert "'lb'" in win, "el peso conserva la fracción (0,43 lb es legítimo en carnicería)"
    j = dj.index("market_qty_numeric: shopQty")
    assert j > 0, "el espejo numérico debe alinearse al qty ceileado (resolveShopQty lo prefiere)"
