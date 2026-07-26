"""[P1-SEASONING-WORD-BOUNDARY · 2026-07-26] `"sal"` ⊂ `"salsa"` ⊂ `"salmón"` ⊂ `"salami"`.

Los dos guards que consultan `_QTY_GUARD_SEASONING_SKIP` decidían "esto es un sazonador" con
`any(_seas in bare_low ...)`. En español eso se traga alimentos enteros — la misma clase de bug que
`"pollo"` ⊂ `"repollo"` (2026-07-24), con otra palabra.

## Los 5 alimentos que el catálogo real (204) perdía

    sal  →  Salami · Salmón · Salsa de soya · Salsa de tomate
    ajo  →  Ajonjolí

## Las dos consecuencias, que NO son la misma

`_ensure_ingredient_quantities` inyecta una porción cuando un alimento verificado llega sin gramos,
y exime a los sazonadores porque su macro es ≈0. Un "Salmón" sin cantidad se eximía ⇒ **contaba ≈0
en macros en silencio**, que es literalmente el agujero que P2-QTY-PRESENCE-GUARD existe para
cerrar. La subcadena lo reabría para 5 alimentos, dos de ellos proteínas.

`_ensure_ingredients_used_in_recipe` exige que cada ingrediente aparezca en algún paso, y exime a
los sazonadores porque uno sazona sin que la receta lo diga. **Caso vivo, plan 2b3be84e** («Filete
de pescado blanco Salteados al Wok»): la lista trae `1 cda de salsa de soya`, los pasos usan
vinagre y no la mencionan nunca. Nadie lo detectó porque `"sal"` ⊂ `"salsa"`. El usuario la compra,
le suma ~900 mg de sodio a la cuenta del día, y la receta no le dice qué hacer con ella.

## Por qué este fix sí y los otros defectos de la misma auditoría no

Medí el delta sobre **1.955 líneas de ingrediente de 9 planes ANTES de tocar el código**: cambia
exactamente 6 líneas y produce **cero** regresiones. Los otros candidatos de esa auditoría se
cayeron al medirlos — el "21% de ingredientes sin usar" era mi propio matcher sin manejar plurales
(el de producción reporta 0/100), la leche no estaba corta, y el ceviche que "cocina el pescado" lo
blanquea 1-2 min a propósito y lo explica en el paso 5 por seguridad alimentaria.
"""
import pytest

import graph_orchestrator as go


# ───────────── 1. los 5 alimentos dejan de ser "sazón" ─────────────

@pytest.mark.parametrize("alimento", [
    "salsa de soya", "salsa de tomate", "salmon", "salami", "ajonjoli",
    "salchicha", "salchichon",              # mismo prefijo, no están en el catálogo hoy
])
def test_un_alimento_no_es_un_sazonador(alimento):
    assert go._is_seasoning_name(alimento) is False


def test_el_caso_vivo_del_wok():
    """Plan 2b3be84e: `1 cda de salsa de soya` listada y jamás usada en los pasos."""
    assert go._is_seasoning_name("salsa de soya") is False


# ───────────── 2. la mitad que importa: los sazonadores SIGUEN exentos ─────────────

@pytest.mark.parametrize("sazon", [
    "sal", "sal marina", "sal y pimienta",
    "ajo", "dientes de ajo", "ajo en polvo", "ajo machacado",
    "cebolla", "cebolla morada", "cebollin",
    "pimienta", "pimienta negra",
    "limon", "jugo de limon", "limones",
    "aji", "aji cubanela", "ajies morrones",
    "oregano", "oregano seco", "cilantro", "cilantro fresco", "perejil",
    "comino", "laurel", "hojas de laurel", "vinagre", "vinagre de manzana",
    "jengibre", "especias", "apio", "cubito", "cubito de pollo",
    "curcuma", "curcuma en polvo", "achiote",
    "sazon", "sazon completo",
    "sazonador",   # el ÚNICO que el límite de palabra habría perdido → entrada propia
])
def test_los_sazonadores_reales_siguen_exentos(sazon):
    assert go._is_seasoning_name(sazon) is True


def test_plural_cubierto():
    for s in ("ajos", "limones", "cebollas", "ajies", "especias"):
        assert go._is_seasoning_name(s) is True, s


# ───────────── 3. la clase de bug, no sólo la instancia ─────────────

def test_ningun_alimento_del_catalogo_queda_clasificado_como_sazon():
    """El test que habría atrapado esto en su día: cruzar la lista de sazonadores contra los
    nombres de alimento reales, en vez de confiar en que la subcadena "se ve bien"."""
    ALIMENTOS = [
        "Salami", "Salmón", "Salsa de soya", "Salsa de tomate", "Ajonjolí",
        "Repollo", "Berenjena", "Salchicha", "Aguacate", "Ajonjolí tostado",
    ]
    from constants import strip_accents as sa
    malos = [a for a in ALIMENTOS if go._is_seasoning_name(sa(a.lower()))]
    assert not malos, f"alimentos tratados como sazón: {malos}"


def test_no_es_subcadena():
    """Ancla explícita: si alguien vuelve al `in`, esto falla."""
    assert "sal" in "salsa de soya"                       # la subcadena SÍ está…
    assert go._is_seasoning_name("salsa de soya") is False  # …y aun así no es sazón


# ───────────── 4. knob de rollback ─────────────

def test_knob_revierte_al_comportamiento_previo(monkeypatch):
    monkeypatch.setattr(go, "SEASONING_WORD_BOUNDARY", False)
    assert go._is_seasoning_name("salsa de soya") is True   # subcadena otra vez
    assert go._is_seasoning_name("sal") is True


def test_fail_safe_con_entrada_vacia():
    assert go._is_seasoning_name("") is False
    assert go._is_seasoning_name(None) is False


# ───────────── 5. los dos callsites usan el helper ─────────────

def test_ambos_guards_pasan_por_el_helper():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_is_seasoning_name(bare_low)") == 2, \
        "los dos guards (qty-presence y reverse-coherence) deben compartir el criterio"
    assert "any(_seas in bare_low for _seas in _QTY_GUARD_SEASONING_SKIP)" not in \
        src[src.index("def _ensure_ingredient_quantities"):], \
        "quedó un callsite con la comparación por subcadena"


def test_sazonador_esta_en_la_lista():
    assert "sazonador" in go._QTY_GUARD_SEASONING_SKIP


# ───────────── 6. [P1-QTY-GUARD-SPICE-GAP · 2026-07-26] las dos listas de "esto es sazón" ─────────

@pytest.mark.parametrize("especia", ["canela", "vainilla", "pimenton", "albahaca", "tomillo",
                                    "curry", "mostaza", "polvo de hornear"])
def test_las_especias_que_solo_conocia_la_otra_lista(especia):
    """`_SHRINK_FLOOR_EXEMPT_TOKENS` (34 entradas) ya sabía que canela/vainilla/mostaza/polvo de
    hornear son sazón; `_QTY_GUARD_SEASONING_SKIP` no. Dos vocabularios para el MISMO concepto con
    contenido distinto — la clase de bug que más costó el 2026-07-26."""
    assert go._is_seasoning_name(especia) is True


def test_las_dos_listas_no_se_contradicen():
    """Cualquier token que una lista considere sazón, la otra no debe tratar como alimento. Ancla la
    CLASE: si alguien añade una especia a una sola de las dos, esto falla."""
    from constants import strip_accents as sa
    faltan = [t for t in go._SHRINK_FLOOR_EXEMPT_TOKENS
              # 'salsa'/'rallado'/'semillas'/'caldo' son modificadores de alimento, no sazón en sí:
              # el piso los exime por otra razón (no son macro-bearing servibles).
              if t not in ("salsa", "rallado", "semillas", "caldo", "azucar", "cacao", "miel",
                           "manteca", "mantequilla", "margarina", "mayonesa", "parmesano",
                           "aceite", "chia", "linaza", "levadura", "sofrito", "ajonjoli")
              and not go._is_seasoning_name(sa(t))]
    assert not faltan, f"sazón que el qty-guard trataría como comida: {faltan}"


def test_miel_y_cacao_siguen_siendo_alimento():
    """Sus porciones reales son de tamaño alimento (1 cda de miel = 21 g, 64 kcal). Eximirlas
    esconderia macro de verdad."""
    assert go._is_seasoning_name("miel") is False
    assert go._is_seasoning_name("cacao en polvo") is False
