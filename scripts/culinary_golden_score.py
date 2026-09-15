# -*- coding: utf-8 -*-
"""[P0-CULINARY-GOLDEN · 2026-09-06] Precisión y recall de cada capa contra las etiquetas humanas.

Lee `docs/culinary_golden_set.json` YA ETIQUETADO y calcula, por capa, cuánto acierta y cuánto se le
escapa. **Este es el único número con el que se decide si V5 escala a `block`** — no la tasa del
juez, que es un LLM opinando sobre sí mismo.

    python scripts/culinary_golden_score.py                 # binario: ¿la capa marcó una comida que tiene defecto?
    python scripts/culinary_golden_score.py --json
    python scripts/culinary_golden_score.py --estricto      # [C1] por hallazgo: ¿marcó EL defecto que la persona vio?
    python scripts/culinary_golden_score.py --estricto --anotaciones docs/culinary_golden_anotaciones_B.json
    python scripts/culinary_golden_score.py --particiones   # [C1] folds por linaje (plan), sin parientes cruzados
    python scripts/culinary_golden_score.py --estricto --anotaciones docs/culinary_golden_anotaciones_angelo.json --maquina 2026-09-15
    python scripts/culinary_golden_score.py --estricto --anotaciones docs/culinary_golden_anotaciones_angelo.json \
        --comparar-maquina 2026-09-15 --json                 # [lote 60] antes/después del refresco: la línea base estricta
    python scripts/culinary_golden_score.py --estricto --anotaciones docs/culinary_golden_anotaciones_angelo.json \
        --desde 2026-09-15 --comparar-maquina 2026-09-15-lote62   # [lote 62] contra la línea base del lote 38
    python scripts/culinary_golden_score.py --anotaciones docs/culinary_golden_anotaciones_angelo.json \
        --juez-por-codigo maquina_juez_2026-09-15 [--observacion paso_incoherente,...]   # [lote 63] el juez por código

## Cómo se corrige el sesgo del muestreo

La muestra es estratificada a propósito: 25 comidas «sin hallazgo» de un universo de 919 y 15 «ambas»
de un universo de 23. Contar los aciertos en bruto daría una precisión inventada, porque los estratos
raros están sobre-representados.

Cada caso pesa `disponibles_en_su_estrato / muestreados_en_su_estrato`, así que las cifras se leen
como si fueran de la población. **Se publican las dos**: la cruda (lo que se contó) y la ponderada
(lo que significa). Un solo número aquí escondería el sesgo en vez de corregirlo.

## [P1-PLAN-LOTE-22 · 2026-09-12] (C1 · CUL-P0-02) El marcador estricto

El binario responde «¿la máquina marcó una comida que la persona también marcó?». Con 68 de 80 comidas
etiquetadas `defecto`, eso deja pasar un acierto por casualidad: la máquina acusa al aceite y la persona
vio que la quinoa no se cocina — cuenta como TP. El estricto adjudica **hallazgo a hallazgo**:

  · cada defecto humano trae `clase` (de `RUBRICA`), `severidad`, `evidencia` localizada y, si aplica, `alimento`;
  · un hallazgo de la máquina cuenta como TP sólo si su clase corresponde a un defecto humano de ESA comida
    (y, si el defecto nombra `alimento` y el hallazgo nombra alguno, que sea ése — lote 60, abajo); si no, es un FP
    localizado y el
    defecto humano queda como FN — «un error distinto produce FP y deja FN del esperado»;
  · los hallazgos duplicados de la máquina se cuentan UNA vez (no multiplican TP);
  · cero división devuelve `null`, nunca 0 ni 100.

El estricto necesita anotaciones con rúbrica. Las 80 etiquetas del 2026-09-07 son binarias (`veredicto_humano` +
`nota_humana`): valen para el binario, y para el estricto cuentan como anotador «dueño» con `defectos`
DESCONOCIDOS — el resultado sale **incompleto (exit 4)** hasta que exista la anotación con rúbrica. No se
rellena desde la nota con el modelo: las etiquetas pendientes no se sustituyen por una respuesta del agente.

Acuerdo y adjudicación: con dos anotadores independientes se publica el acuerdo binario (kappa de Cohen) y las
discrepancias caso a caso; el veredicto que puntúa es la `adjudicacion` cuando existe y, si no, la anotación única.
Un caso con dos anotaciones en desacuerdo y sin adjudicar queda `pendiente_adjudicacion` y no puntúa.

Intervalos: bootstrap por CONGLOMERADO (el plan es la unidad independiente; las comidas del mismo plan no lo son),
percentiles 2,5 y 97,5 sobre 1.000 remuestreos con semilla fija.

Particiones por linaje: `--particiones` reparte los PLANES (no las comidas) en k folds por su hash — ningún plan
queda a ambos lados, así que un umbral ajustado en un fold no se evalúa sobre hermanas del mismo plan.

## [P1-PLAN-LOTE-60 · 2026-09-15] (lote 38 del plan · C1 cierre) El alimento sólo restringe si el hallazgo nombra uno

Con la anotación del dueño (80/80, 2026-09-13) el estricto daba 0 aciertos POR CONSTRUCCIÓN: el `alimento` del defecto
se exigía como SUBCADENA del texto de la máquina, el dueño lo rellenó en casi todos («Carne de res magra», «Casabe y
queso fresco.») y los textos de V4 no nombran alimento — «V4: ingrediente declara 85 g, pasos declaran 140 g» es
exactamente el defecto del caso `0108f857ae` («Lista: 85 g de res; paso 1: porción de 140 g») y salía FN + FP. Ahora:

  · el hallazgo NOMBRA un alimento si alguna de sus palabras está en la lista de ingredientes de ESA comida (sin
    cantidades, unidades, estados ni colores: `_GENERICAS`); sin lista —fixtures— cualquier palabra no genérica cuenta;
  · si lo nombra, el defecto se empareja sólo si CUALQUIER palabra de ≥ 4 letras de su `alimento` (o una de las cortas
    de comida: ajo, ají, res, pan, sal, uva) aparece en el hallazgo, sin acentos y en singular; si no nombra ninguno,
    se empareja sólo por código;
  · si el hallazgo DECLARA el alimento que acusa —las columnas refrescadas terminan en `(alimento: X)`, el `food` de la
    violación, o `(componente: X)` en el juez; el builder del 09-06 los tiraba— se compara con ése y no con el texto:
    un V7e cuyo detalle cita el paso («…corta la ciruela…») acusa al casabe, no a la ciruela;
  · cada acierto dice `emparejado_por: codigo | codigo+alimento` y cada capa publica cuántos de cada;
  · «defecto» sin ningún defecto de la rúbrica no es `completo`: queda `sin_rubrica` (no puntúa) y el informe lo nombra.

Se midieron tres formas de decidir «nombra un alimento» (palabras entre comillas, comillas validadas contra la lista,
palabras de la lista en cualquier sitio): las tres dan los mismos aciertos sobre la anotación del dueño y se eligió la
última porque es la única que ve un alimento sin comillas (el texto libre del juez, el paso que cita V5). Sin ningún
filtro el juez ganaba dos aciertos que no lo son — `098d23388f` («Queso Cottage» frente al merey sin tostar del dueño) y
`0ee6d0a81c` (la nota de seguridad del queso frente a la masa que ningún paso prepara) —: el filtro los descarta a
propósito. *Un veredicto vale lo que valga el instrumento que lo produjo.*

`--maquina FECHA` puntúa con las columnas refrescadas (`maquina_<capa>_FECHA`, `scripts/culinary_golden_refresh.py`);
una capa sin columna de esa fecha usa la del 09-06 y el informe lo dice. `--comparar-maquina FECHA` saca antes y
después a la vez: es la forma de la línea base estricta (`docs/culinary_baseline_estricto_2026-09-15.md`).
tooltip-anchor: P1-PLAN-LOTE-60-ADJUDICADOR

## Lo que NO calcula

Una «nota de calidad culinaria» de 1 a 10. Precisión y recall son propiedades del DETECTOR; la
calidad del plan es otra cosa y necesitaría un criterio de gravedad que hoy nadie ha definido.
Fabricar una nota a partir de estos números sería darle a una opinión la cara de una medición.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"
#: [P1-PLAN-LOTE-60] Las columnas de la máquina del 2026-09-06. Un refresco escribe `maquina_<capa>_<fecha>` AL LADO.
_COLUMNAS_BASE = ("maquina_determinista", "maquina_juez")

MINIMO_ETIQUETAS = 20

_DEFECTO = {"defecto", "malo", "mal", "si", "sí"}
_OK = {"ok", "bien", "correcto", "no"}

#: [C1] Rúbrica: clase humana → códigos de la máquina que la cubren. Las clases sin código son defectos que hoy
#: ninguna capa mecaniza (aparecen como FN del sistema entero cuando la persona las anota).
RUBRICA = {
    "verbo_alimento": {"V1"},
    "estado_imposible": {"V2"},
    "ingrediente_huerfano": {"V3"},
    "cantidad_inconsistente": {"V4", "V6", "V7e"},
    "usa_lo_que_no_esta": {"V5"},
    "lista_de_mas": {"V7a"},
    "duplicado_incompatible": {"V7b"},
    "seco_sin_coccion": {"V7c"},
    "masa_sobrante": {"V7d"},
    "tiempo_oculto": {"V8a"},           # [P1-PLAN-LOTE-26]
    "equipo_no_disponible": {"V8b"},    # [P1-PLAN-LOTE-26]
    "estructura_del_plato": {"V9"},      # [P1-PLAN-LOTE-27]
    "combo_absurdo": {"combo_absurdo"},
    "tecnica_impropia": {"tecnica_impropia"},
    "paso_incoherente": {"paso_incoherente"},
    "slot_inapropiado": {"slot_inapropiado"},
    "nombre_no_corresponde": {"nombre_no_corresponde"},
    "rendimiento_vs_unidades": set(),
    "coccion_faltante": {"V7f"},        # [P1-PLAN-LOTE-62]
    "otro": set(),
}
SEVERIDADES = ("minor", "high")
_CODIGOS_DET = {"V1", "V2", "V3", "V4", "V5", "V6", "V7a", "V7b", "V7c", "V7d", "V7e", "V7f", "V8a", "V8b", "V9"}

#: [P1-PLAN-LOTE-63 · 2026-09-15] (lote 40 del plan · C5/C6) Lo que cada código del JUEZ puede describir además de su clase
#: homónima, para la adjudicación «por sustancia»: el juez tiene 5 códigos y la rúbrica reparte los defectos de pasos y
#: lista entre clases que asigna al determinista, así que dice `paso_incoherente` donde el dueño dice
#: `cantidad_inconsistente` y la precisión estricta le da 0 por construcción. SÓLO en el marcador — el instrumento puede
#: ser generoso, el juez no cambia de voz — y siempre con el filtro de alimento del adjudicador (el mismo alimento que el
#: defecto del dueño: eso es lo que ata `slot_inapropiado` a «el arroz crudo que se incorpora»).
#: tooltip-anchor: P1-PLAN-LOTE-63-SUSTANCIA
SUSTANCIA_JUEZ = {
    "paso_incoherente": {"cantidad_inconsistente", "usa_lo_que_no_esta", "ingrediente_huerfano", "seco_sin_coccion"},
    "tecnica_impropia": {"seco_sin_coccion", "coccion_faltante", "verbo_alimento", "paso_incoherente"},
    "nombre_no_corresponde": {"usa_lo_que_no_esta"},
    "slot_inapropiado": {"seco_sin_coccion"},
    "combo_absurdo": set(),
}


def rubrica_por_sustancia() -> dict:
    """[P1-PLAN-LOTE-63] La RUBRICA con cada código del juez añadido a las clases que su sustancia describe."""
    return {cl: set(cods) | {k for k, cls in SUSTANCIA_JUEZ.items() if cl in cls} for cl, cods in RUBRICA.items()}
BOOTSTRAP_N = 1000
BOOTSTRAP_SEMILLA = 20260912


def _sin_acentos(s) -> str:
    t = unicodedata.normalize("NFD", str(s or ""))
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn").lower()


#: [P1-PLAN-LOTE-60 · 2026-09-15] Palabras que acompañan a un alimento sin serlo (cantidades, unidades, estados, colores)
#: y las de las plantillas de los hallazgos: no cuentan para decidir si un hallazgo nombra un alimento ni para emparejarlo.
#: Sin acentos, como las compara `_palabras_de_alimento`.
_GENERICAS = frozenset("""
    taza tazas cucharada cucharadas cucharadita cucharaditas cdas cdta cdtas gramo gramos kilo kilos libra libras onza
    onzas litro litros unidad unidades porcion porciones rebanada rebanadas lonja lonjas pedazo pedazos pizca pizcas
    chorrito punado ramita ramitas hoja hojas diente dientes rodaja rodajas trozo trozos
    mediano mediana medianos medianas grande grandes pequeno pequena pequenos pequenas entero entera enteros enteras
    cocido cocida cocidos cocidas crudo cruda crudos crudas seco seca secos secas fresco fresca frescos frescas picado
    picada picados picadas rallado rallada rallados ralladas molido molida light bajo baja grasa integral integrales
    blanco blanca blancos blancas negro negra negros negras rojo roja rojos rojas verde verdes maduro madura maduros
    maduras magro magra magros magras dominicano dominicana natural criollo criolla tostado tostada tostados tostadas
    para sobre entre como pero desde hasta cada todo toda todos todas mismo misma este esta estos estas otro otra otros
    otras solo sola gusto
    lista paso pasos receta plato platos ingrediente ingredientes declara declaran declarado declarada declarados listado
    ningun ninguna menciona aplica listo comer viene compra usan trae pide piden aparece incompatibles habla hablan
    siempre singular espera minutos horas tiempo persona tenerlo evidencia preparacion nombre montaje mise place toque
    fuego sirve servir alimento alimentos componente
""".split())
#: Alimentos de tres letras que sí cuentan (el mínimo general es 4): «res» de «Carne de res magra», «ajo», «pan»…
_ALIMENTOS_CORTOS = frozenset({"ajo", "aji", "res", "pan", "sal", "uva"})


def _singular(w: str) -> str:
    """Singular de andar por casa, suficiente para comparar nombres de alimentos: nueces → nuez, limones → limon,
    tomates → tomate, rábanos → rabano. Se aplica a los DOS lados, así que un error simétrico no rompe el emparejamiento."""
    if w.endswith("ces") and len(w) > 5:
        return w[:-3] + "z"
    if w.endswith("es") and len(w) > 5 and w[-3] in "lnrdzj":
        return w[:-2]
    if w.endswith("s") and len(w) > 4:
        return w[:-1]
    return w


def _palabras_de_alimento(texto) -> set:
    """[P1-PLAN-LOTE-60] Palabras sin acentos, en minúscula y en singular, de ≥ 4 letras (o de las cortas de comida), sin
    las genéricas. Es la unidad de comparación del adjudicador: «Rábanos.» y «rabano» son la misma palabra."""
    out = set()
    for w in re.findall(r"[a-z]+", _sin_acentos(texto)):
        if (len(w) >= 4 or w in _ALIMENTOS_CORTOS) and w not in _GENERICAS:
            out.add(_singular(w))
    return out


def vocabulario_de_comida(caso) -> set:
    """[P1-PLAN-LOTE-60] Las palabras de alimento de la lista de ingredientes de la comida: con ellas se decide si un
    hallazgo NOMBRA un alimento. Vacío si el caso no trae lista (fixtures sintéticos)."""
    voc = set()
    for linea in caso.get("ingredientes") or []:
        voc |= _palabras_de_alimento(linea)
    return voc


def _cuerpo_del_hallazgo(texto: str) -> str:
    """El texto sin el código (`"V4: ..."` → `"..."`): «paso_incoherente» no es un alimento."""
    t = str(texto or "")
    return t.split(":", 1)[1] if ":" in t else t


#: [P1-PLAN-LOTE-60] El alimento que el hallazgo DECLARA acusar, al final del texto de una columna refrescada.
_EXPLICITO = re.compile(r"\((?:alimento|componente): ([^()]+)\)\s*$")


def _alimento_explicito(texto: str) -> set:
    """[P1-PLAN-LOTE-60] Las palabras del alimento que el hallazgo declara acusar (`(alimento: X)` de capa 1,
    `(componente: X)` del juez). Las columnas del 09-06 no lo traen: `set()`, y manda el texto."""
    m = _EXPLICITO.search(str(texto or ""))
    return _palabras_de_alimento(m.group(1)) if m else set()


def _nombra_alimento(texto: str, vocabulario) -> bool:
    """[P1-PLAN-LOTE-60] ¿El hallazgo nombra algún alimento? Si declara el que acusa, sí. Si no, con lista de
    ingredientes: alguna de sus palabras está en ella; sin lista, cualquier palabra no genérica cuenta (restringir de
    más en un fixture es mejor que emparejar dos alimentos distintos)."""
    if _alimento_explicito(texto):
        return True
    palabras = _palabras_de_alimento(_cuerpo_del_hallazgo(texto))
    return bool(palabras & vocabulario) if vocabulario else bool(palabras)


def _verdad(caso) -> bool | None:
    v = str(caso.get("veredicto_humano") or "").strip().lower()
    if v in _DEFECTO:
        return True
    if v in _OK:
        return False
    return None                     # vacío o «dudoso»: NO se fuerza a binario


def _r(a, b):
    """Porcentaje redondeado; `None` con denominador 0 — cero división devuelve null, nunca un número."""
    return round(100.0 * a / (a + b), 1) if (a + b) else None


def _unidad(caso, i):
    """La unidad independiente: el plan. Sin plan (fixtures sintéticos), cada caso es su propia unidad."""
    return str(caso.get("plan") or f"caso-{i}")


def _bootstrap(casos, conteo_fn, n=BOOTSTRAP_N, semilla=BOOTSTRAP_SEMILLA):
    """IC 95 % por conglomerado (plan) de precisión y recall crudos. `conteo_fn(caso) -> (tp, fp, fn)`.
    `None` con menos de 2 conglomerados: un intervalo sobre una sola unidad no es un intervalo."""
    grupos = collections.defaultdict(list)
    for i, c in enumerate(casos):
        grupos[_unidad(c, i)].append(c)
    claves = sorted(grupos)
    if len(claves) < 2:
        return None
    rng = random.Random(semilla)
    ps, rs = [], []
    for _ in range(n):
        tp = fp = fn = 0
        for _k in range(len(claves)):
            for c in grupos[claves[rng.randrange(len(claves))]]:
                a, b, d = conteo_fn(c)
                tp += a; fp += b; fn += d
        p, r = _r(tp, fp), _r(tp, fn)
        if p is not None:
            ps.append(p)
        if r is not None:
            rs.append(r)

    def _pct(xs):
        if len(xs) < 20:
            return None
        xs = sorted(xs)
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]]
    return {"precision": _pct(ps), "recall": _pct(rs), "conglomerados": len(claves), "remuestreos": n}


def puntuar(d: dict, columnas: dict | None = None) -> dict:
    """El marcador BINARIO (por comida): ¿la capa marcó una comida que la persona marcó?

    [P1-PLAN-LOTE-60] `columnas` = `{"determinista": "maquina_determinista_<fecha>", ...}` puntúa una columna refrescada;
    un caso sin esa columna no se juzgó con ella y queda fuera de la capa (`sin_columna`)."""
    casos = d.get("casos") or []
    disp = d.get("disponibles_por_estrato") or {}
    muestreados = collections.Counter(c.get("estrato") for c in casos)

    etiquetados = [c for c in casos if _verdad(c) is not None]
    dudosos = sum(1 for c in casos
                  if str(c.get("veredicto_humano") or "").strip().lower() == "dudoso")
    sin_etiquetar = len(casos) - len(etiquetados) - dudosos

    def peso(c):
        e = c.get("estrato")
        m = muestreados.get(e) or 1
        return (disp.get(e) or m) / m

    salida = {"casos": len(casos), "etiquetados": len(etiquetados),
              "dudosos": dudosos, "sin_etiquetar": sin_etiquetar, "capas": {}}

    columnas = columnas or {}
    for capa, clave in (("determinista", "maquina_determinista"), ("juez", "maquina_juez")):
        clave = columnas.get(capa) or clave
        de_la_capa = etiquetados if clave in _COLUMNAS_BASE else [c for c in etiquetados if clave in c]
        tp = fp = fn = tn = 0.0
        tp_n = fp_n = fn_n = tn_n = 0
        for c in de_la_capa:
            marco = bool(c.get(clave))
            real = _verdad(c)
            w = peso(c)
            if marco and real:
                tp += w; tp_n += 1
            elif marco and not real:
                fp += w; fp_n += 1
            elif not marco and real:
                fn += w; fn_n += 1
            else:
                tn += w; tn_n += 1

        def _conteo(c, clave=clave):
            marco, real = bool(c.get(clave)), _verdad(c)
            return (1 if marco and real else 0, 1 if marco and not real else 0, 1 if (not marco) and real else 0)
        salida["capas"][capa] = {
            "crudo": {"tp": tp_n, "fp": fp_n, "fn": fn_n, "tn": tn_n,
                      "precision": _r(tp_n, fp_n), "recall": _r(tp_n, fn_n)},
            "ponderado": {"precision": _r(tp, fp), "recall": _r(tp, fn)},
            "ic95_crudo": _bootstrap(de_la_capa, _conteo),   # [C1] por conglomerado (plan)
            "columna": clave, "sin_columna": len(etiquetados) - len(de_la_capa),
        }
    return salida


# ── [C1] anotaciones con rúbrica, acuerdo y adjudicación ──────────────────────────────────────

def _codigo(texto: str) -> str:
    """`"V4: ..."` → `V4`; `"paso_incoherente: ..."` → `paso_incoherente`."""
    return str(texto or "").split(":", 1)[0].strip()


#: [P1-PLAN-LOTE-28 · 2026-09-12] (CUL-P1-06) Un hallazgo del juez marcado `[dudosa]` es OBSERVACIÓN: no cuenta como
#: falso positivo ni como acierto salvo `--con-dudosas`. El estado incierto tiene que poder decirse sin pagar precisión.
INCLUIR_DUDOSAS = False
MARCA_DUDOSA = "[dudosa]"
#: [P1-PLAN-LOTE-63] Códigos del juez que cuentan como `[dudosa]` aunque la columna no lo diga: simula
#: `MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES` sobre una columna escrita antes del post-proceso (`--observacion`).
OBSERVACION: frozenset = frozenset()


def _es_dudosa(texto: str, observacion=None) -> bool:
    obs = OBSERVACION if observacion is None else observacion
    return MARCA_DUDOSA in texto or _codigo(texto).replace(MARCA_DUDOSA, "").strip() in obs


def _hallazgos_maquina(caso, clave, observacion=None) -> list:
    """Hallazgos de la máquina deduplicados (mismo texto = mismo hallazgo): los duplicados no multiplican TP.
    Las `[dudosa]` del juez se excluyen salvo `INCLUIR_DUDOSAS`; también los códigos en observación (lote 63)."""
    vistos, out = set(), []
    for t in caso.get(clave) or []:
        k = str(t).strip()
        if _es_dudosa(k, observacion) and not INCLUIR_DUDOSAS:
            continue
        if k and k not in vistos:
            vistos.add(k)
            out.append({"codigo": _codigo(k).replace(MARCA_DUDOSA, "").strip(), "texto": k})
    return out


def contar_dudosas(d: dict, clave: str = "maquina_juez") -> int:
    return sum(1 for c in (d.get("casos") or []) for t in (c.get(clave) or []) if _es_dudosa(str(t)))


def excluir_casos(d: dict, ids) -> dict:
    """[P1-PLAN-LOTE-28] (CUL-P1-06) Los ejemplos de DESARROLLO (los que se leyeron para diseñar reglas) no pueden ser
    holdout: se sacan por `id` antes de puntuar, y el informe dice cuántos."""
    ids = {str(x) for x in (ids or [])}
    if not ids:
        return d
    out = dict(d)
    out["casos"] = [c for c in (d.get("casos") or []) if str(c.get("id")) not in ids]
    out["excluidos_dev"] = len(d.get("casos") or []) - len(out["casos"])
    return out


def _anotaciones_de(caso, externas: dict) -> list:
    """Las anotaciones de un caso: las embebidas (`anotaciones`), las externas por `id` y, como legado, la etiqueta
    binaria del 2026-09-07 (anotador «dueño», `defectos` DESCONOCIDOS = None)."""
    out = []
    for a in caso.get("anotaciones") or []:
        if isinstance(a, dict):
            out.append(dict(a))
    for a in externas.get(str(caso.get("id")), []):
        out.append(dict(a))
    if not out and str(caso.get("veredicto_humano") or "").strip():
        out.append({"anotador": "dueño (2026-09-07, binaria)", "veredicto": caso.get("veredicto_humano"),
                    "defectos": None, "nota": caso.get("nota_humana")})
    return out


def _veredicto_bin(v) -> bool | None:
    return _verdad({"veredicto_humano": v})


def _defectos_validos(defectos):
    """Sólo cuentan defectos con clase de la RUBRICA; el resto se reporta como `clase_desconocida`."""
    ok, raros = [], []
    for df in defectos or []:
        if isinstance(df, dict) and df.get("clase") in RUBRICA:
            ok.append(df)
        else:
            raros.append(df)
    return ok, raros


def _sin_rubrica_por_defecto_vacio(por, raros) -> dict:
    """[P1-PLAN-LOTE-60] «defecto» sin ningún defecto de la RUBRICA: no hay contra qué emparejar, así que no puntúa —ni
    como FP de todo lo que marcó la máquina ni como acierto de nada—. El informe lo nombra para completarlo en la hoja."""
    return {"estado": "sin_rubrica", "motivo": "defecto_sin_defectos", "defectos": [], "veredicto": True, "por": por,
            "raros": raros}


def _verdad_estricta(caso, externas: dict) -> dict:
    """Qué se toma como verdad para el estricto: `adjudicacion` > anotación única > acuerdo entre varias.
    Estados: `completo`, `sin_rubrica` (sólo binaria, o «defecto» sin ningún defecto de la rúbrica — lote 60),
    `pendiente_adjudicacion`, `sin_anotar`, `dudoso`."""
    adj = caso.get("adjudicacion") or externas.get(("adjudicacion", str(caso.get("id"))))
    if isinstance(adj, dict) and adj.get("veredicto") is not None:
        vb = _veredicto_bin(adj.get("veredicto"))
        if vb is None:
            return {"estado": "dudoso", "defectos": [], "veredicto": None, "por": "adjudicacion"}
        defs, raros = _defectos_validos(adj.get("defectos") or [])
        if vb and not defs:
            return _sin_rubrica_por_defecto_vacio("adjudicacion", raros)
        return {"estado": "completo", "defectos": defs if vb else [], "veredicto": vb, "por": "adjudicacion", "raros": raros}
    anots = _anotaciones_de(caso, externas)
    con_rubrica = [a for a in anots if isinstance(a.get("defectos"), list)]
    if not anots:
        return {"estado": "sin_anotar", "defectos": [], "veredicto": None, "por": None}
    if not con_rubrica:
        return {"estado": "sin_rubrica", "defectos": [], "veredicto": _veredicto_bin(anots[0].get("veredicto")), "por": None}
    if len(con_rubrica) == 1:
        a = con_rubrica[0]
        vb = _veredicto_bin(a.get("veredicto"))
        if vb is None:
            return {"estado": "dudoso", "defectos": [], "veredicto": None, "por": a.get("anotador")}
        defs, raros = _defectos_validos(a.get("defectos"))
        if vb and not defs:
            return _sin_rubrica_por_defecto_vacio(a.get("anotador"), raros)
        return {"estado": "completo", "defectos": defs if vb else [], "veredicto": vb, "por": a.get("anotador"), "raros": raros}
    # dos o más con rúbrica: coinciden en veredicto Y en el conjunto de clases ⇒ completo; si no, pendiente
    vbs = {_veredicto_bin(a.get("veredicto")) for a in con_rubrica}
    clases = [frozenset(df.get("clase") for df in _defectos_validos(a.get("defectos"))[0]) for a in con_rubrica]
    if len(vbs) == 1 and None not in vbs and len(set(clases)) == 1:
        a = con_rubrica[0]
        defs, raros = _defectos_validos(a.get("defectos"))
        if vbs == {True} and not defs:
            return _sin_rubrica_por_defecto_vacio("acuerdo:" + "+".join(str(x.get("anotador")) for x in con_rubrica), raros)
        return {"estado": "completo", "defectos": defs if vbs == {True} else [], "veredicto": vbs.pop(),
                "por": "acuerdo:" + "+".join(str(x.get("anotador")) for x in con_rubrica), "raros": raros}
    return {"estado": "pendiente_adjudicacion", "defectos": [], "veredicto": None, "por": None}


def _adjudicar_hallazgos(maquina: list, defectos: list, codigos_capa: set,
                         vocabulario: set | None = None, rubrica: dict | None = None) -> tuple[int, int, int, list]:
    """Emparejamiento hallazgo↔defecto, cada uno como mucho una vez. Devuelve (tp, fp, fn, detalle).

    [P1-PLAN-LOTE-60 · 2026-09-15] El `alimento` del defecto sólo restringe cuando el hallazgo NOMBRA algún alimento
    (`_nombra_alimento` con el `vocabulario` de la lista de ingredientes de la comida): entonces basta con que CUALQUIER
    palabra del `alimento` aparezca en el que el hallazgo DECLARA acusar (`(alimento: X)` / `(componente: X)`, columnas
    refrescadas) o, si no declara ninguno, en su texto. Si no nombra ninguno (V4: «ingrediente declara 85 g, pasos declaran
    140 g») se empareja sólo por código. Antes se exigía el `alimento` entero como subcadena: 0 aciertos con la
    anotación del dueño. Cada acierto dice `emparejado_por`. tooltip-anchor: P1-PLAN-LOTE-60-ADJUDICADOR"""
    usados = set()
    tp = 0
    detalle = []
    for df in defectos:
        codigos = (rubrica or RUBRICA).get(df.get("clase"), set()) & codigos_capa   # [P1-PLAN-LOTE-63] o la «por sustancia»
        alimento = _palabras_de_alimento(df.get("alimento") or "")
        elegido, por = None, None
        for i, h in enumerate(maquina):
            if i in usados or h["codigo"] not in codigos:
                continue
            if alimento and _nombra_alimento(h["texto"], vocabulario):
                acusado = _alimento_explicito(h["texto"]) or _palabras_de_alimento(_cuerpo_del_hallazgo(h["texto"]))
                if not alimento & acusado:
                    continue
                por = "codigo+alimento"
            else:
                por = "codigo"
            elegido = i
            break
        if elegido is not None:
            usados.add(elegido)
            tp += 1
            detalle.append({"defecto": df.get("clase"), "hallazgo": maquina[elegido]["texto"][:80], "resultado": "tp",
                            "emparejado_por": por})
        elif codigos:
            detalle.append({"defecto": df.get("clase"), "hallazgo": None, "resultado": "fn"})
        else:
            detalle.append({"defecto": df.get("clase"), "hallazgo": None, "resultado": "fn_no_mecanizable"})
    fp = len([i for i in range(len(maquina)) if i not in usados])
    for i, h in enumerate(maquina):
        if i not in usados:
            detalle.append({"defecto": None, "hallazgo": h["texto"][:80], "resultado": "fp"})
    fn = sum(1 for x in detalle if x["resultado"] in ("fn", "fn_no_mecanizable"))
    return tp, fp, fn, detalle


def puntuar_estricto(d: dict, externas: dict | None = None, columnas: dict | None = None) -> dict:
    """El marcador ESTRICTO (por hallazgo). `externas`: anotaciones cargadas de ficheros, por `id` de caso.

    [P1-PLAN-LOTE-60] `columnas` = `{"determinista": "maquina_determinista_<fecha>", "juez": ...}` puntúa columnas
    refrescadas; un caso sin esa columna queda fuera de la capa (`sin_columna`). Cada capa publica `emparejados`
    (`codigo` / `codigo+alimento`) y la salida nombra los «defecto» sin defectos (`sin_rubrica_casos`)."""
    externas = externas or {}
    casos = d.get("casos") or []
    disp = d.get("disponibles_por_estrato") or {}
    muestreados = collections.Counter(c.get("estrato") for c in casos)

    def peso(c):
        e = c.get("estrato")
        m = muestreados.get(e) or 1
        return (disp.get(e) or m) / m

    estados = collections.Counter()
    verdades = []
    for c in casos:
        v = _verdad_estricta(c, externas)
        estados[v["estado"]] += 1
        verdades.append((c, v))
    completos = [(c, v) for c, v in verdades if v["estado"] == "completo"]

    salida = {"casos": len(casos), "completos": len(completos), "estados": dict(estados),
              "minimo": MINIMO_ETIQUETAS, "completo": len(completos) >= MINIMO_ETIQUETAS,
              "promocion_habilitada": False, "capas": {}, "por_clase": {}, "no_mecanizable": collections.Counter(),
              "detalle": []}
    salida["sin_rubrica_casos"] = [c.get("id") for c, v in verdades if v.get("motivo") == "defecto_sin_defectos"]
    por_clase = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    # Los defectos que ninguna capa mecaniza se cuentan UNA vez, como FN del sistema entero, no de cada capa.
    for c, v in completos:
        for df in v["defectos"]:
            if not RUBRICA.get(df.get("clase")):
                salida["no_mecanizable"][df.get("clase")] += 1
                salida["detalle"].append({"caso": c.get("id"), "capa": None, "defecto": df.get("clase"),
                                          "hallazgo": None, "resultado": "fn_no_mecanizable"})
    for capa, clave, codigos in (("determinista", "maquina_determinista", _CODIGOS_DET),
                                 ("juez", "maquina_juez", set().union(*[RUBRICA[k] for k in RUBRICA]) - _CODIGOS_DET)):
        clave = (columnas or {}).get(capa) or clave
        de_la_capa = completos if clave in _COLUMNAS_BASE else [(c, v) for c, v in completos if clave in c]
        tp_n = fp_n = fn_n = 0
        tp_w = fp_w = fn_w = 0.0
        conteos = {}
        emparejados = collections.Counter()
        for c, v in de_la_capa:
            maquina = _hallazgos_maquina(c, clave)
            # a cada capa sólo se le exigen los defectos de SU competencia (los que algún código suyo cubre)
            defectos = [df for df in v["defectos"] if RUBRICA.get(df.get("clase"), set()) & codigos]
            tp, fp, fn, det = _adjudicar_hallazgos(maquina, defectos, codigos, vocabulario_de_comida(c))
            w = peso(c)
            tp_n += tp; fp_n += fp; fn_n += fn
            tp_w += tp * w; fp_w += fp * w; fn_w += fn * w
            conteos[id(c)] = (tp, fp, fn)
            for x in det:
                if x["resultado"] == "tp":
                    por_clase[x["defecto"]]["tp"] += 1
                    emparejados[x.get("emparejado_por")] += 1
                elif x["resultado"] == "fn":
                    por_clase[x["defecto"]]["fn"] += 1
                elif x["resultado"] == "fn_no_mecanizable":
                    salida["no_mecanizable"][x["defecto"]] += 1
                salida["detalle"].append({"caso": c.get("id"), "capa": capa, **x})
        salida["capas"][capa] = {
            "crudo": {"tp": tp_n, "fp": fp_n, "fn": fn_n, "precision": _r(tp_n, fp_n), "recall": _r(tp_n, fn_n)},
            "ponderado": {"precision": _r(tp_w, fp_w), "recall": _r(tp_w, fn_w)},
            "ic95_crudo": _bootstrap([c for c, _ in de_la_capa], lambda c, k=conteos: k.get(id(c), (0, 0, 0))),
            "columna": clave, "sin_columna": len(completos) - len(de_la_capa), "emparejados": dict(emparejados),
        }
    salida["columnas"] = {k: v["columna"] for k, v in salida["capas"].items()}
    salida["por_clase"] = {k: {**v, "precision": _r(v["tp"], v["fp"]), "recall": _r(v["tp"], v["fn"])}
                           for k, v in sorted(por_clase.items())}
    salida["no_mecanizable"] = dict(salida["no_mecanizable"])
    salida["acuerdo"] = acuerdo(d, externas)
    salida["promocion_habilitada"] = bool(salida["completo"] and estados.get("pendiente_adjudicacion", 0) == 0
                                          and (salida["acuerdo"] or {}).get("anotadores", 0) >= 2)
    return salida


def tabla_juez_por_codigo(d: dict, externas: dict | None = None, columna: str = "maquina_juez",
                          observacion=frozenset()) -> dict:
    """[P1-PLAN-LOTE-63] El juez POR CÓDIGO contra la verdad estricta. Por código: `n` (hallazgos seguros), `dudosas` (las
    de la columna más las de `observacion`), `tp`/`fp` con el adjudicador estricto y `sustancia` = aciertos con la rúbrica
    por sustancia (`SUSTANCIA_JUEZ`, mismo alimento, emparejamiento 1:1 igual que el estricto)."""
    externas = externas or {}
    codigos = set().union(*[RUBRICA[k] for k in RUBRICA]) - _CODIGOS_DET
    rub = rubrica_por_sustancia()
    por = collections.defaultdict(collections.Counter)
    for c in d.get("casos") or []:
        v = _verdad_estricta(c, externas)
        if v["estado"] != "completo" or (columna not in _COLUMNAS_BASE and columna not in c):
            continue
        for t in c.get(columna) or []:
            if _es_dudosa(str(t), observacion):
                por[_codigo(str(t)).replace(MARCA_DUDOSA, "").strip()]["dudosas"] += 1
        seguras = _hallazgos_maquina(c, columna, observacion) if not INCLUIR_DUDOSAS else [
            h for h in _hallazgos_maquina(c, columna, observacion) if not _es_dudosa(h["texto"], observacion)]
        voc = vocabulario_de_comida(c)
        for h in seguras:
            por[h["codigo"]]["n"] += 1
        for clave, rubrica in (("tp", None), ("sustancia", rub)):
            r = rubrica or RUBRICA
            defectos = [df for df in v["defectos"] if r.get(df.get("clase"), set()) & codigos]
            _, _, _, det = _adjudicar_hallazgos(seguras, defectos, codigos, voc, rubrica=rubrica)
            for x in det:
                if x["resultado"] == "tp":
                    por[_codigo(x["hallazgo"]).replace(MARCA_DUDOSA, "").strip()][clave] += 1
    filas = {}
    for cod in sorted(por, key=lambda k: (-por[k]["n"], k)):
        x = por[cod]
        filas[cod] = {"n": x["n"], "dudosas": x["dudosas"], "tp": x["tp"], "fp": x["n"] - x["tp"],
                      "sustancia": x["sustancia"], "precision": _r(x["tp"], x["n"] - x["tp"]),
                      "precision_sustancia": _r(x["sustancia"], x["n"] - x["sustancia"])}
    tot = {k: sum(f[k] for f in filas.values()) for k in ("n", "dudosas", "tp", "fp", "sustancia")}
    tot["precision"] = _r(tot["tp"], tot["fp"])
    tot["precision_sustancia"] = _r(tot["sustancia"], tot["n"] - tot["sustancia"])
    return {"columna": columna, "observacion": sorted(observacion), "codigos": filas, "total": tot}


def codigos_en_observacion(tabla: dict, umbral: float = 25.0, n_min: int = 4) -> list:
    """[P1-PLAN-LOTE-63] El criterio del plan: precisión ESTRICTA < `umbral` % con n ≥ `n_min`."""
    return sorted(c for c, f in tabla["codigos"].items()
                  if f["n"] >= n_min and (f["precision"] is None or f["precision"] < umbral))


def render_juez_por_codigo(t: dict) -> str:
    out = [f"[juez por código] columna {t['columna']} · observación {', '.join(t['observacion']) or '—'}",
           "  código                    n  dudosas  tp  fp  sustancia   precisión estricta · por sustancia"]
    for cod, f in list(t["codigos"].items()) + [("TOTAL", t["total"])]:
        out.append(f"  {cod:24s} {f['n']:3d} {f['dudosas']:8d} {f['tp']:3d} {f['fp']:3d} {f['sustancia']:10d}   "
                   f"{f['precision']} · {f['precision_sustancia']}")
    return "\n".join(out)


def acuerdo(d: dict, externas: dict | None = None) -> dict | None:
    """Acuerdo entre anotadores independientes: kappa de Cohen sobre el veredicto binario en los casos que los
    DOS anotaron, y las discrepancias caso a caso (los «casos materiales» que hay que adjudicar). `None` sin dos."""
    externas = externas or {}
    por_anotador = collections.defaultdict(dict)
    for c in d.get("casos") or []:
        for a in _anotaciones_de(c, externas):
            vb = _veredicto_bin(a.get("veredicto"))
            por_anotador[str(a.get("anotador"))][str(c.get("id"))] = vb
    nombres = sorted(por_anotador)
    if len(nombres) < 2:
        return {"anotadores": len(nombres), "kappa": None, "comunes": 0, "discrepancias": [], "nombres": nombres}
    a, b = nombres[0], nombres[1]
    comunes = [k for k in por_anotador[a] if k in por_anotador[b]
               and por_anotador[a][k] is not None and por_anotador[b][k] is not None]
    n = len(comunes)
    if n == 0:
        return {"anotadores": len(nombres), "kappa": None, "comunes": 0, "discrepancias": [], "nombres": nombres}
    acu = sum(1 for k in comunes if por_anotador[a][k] == por_anotador[b][k])
    pa = sum(1 for k in comunes if por_anotador[a][k]) / n
    pb = sum(1 for k in comunes if por_anotador[b][k]) / n
    po = acu / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if pe == 1 else round((po - pe) / (1 - pe), 3)
    disc = [k for k in comunes if por_anotador[a][k] != por_anotador[b][k]]
    return {"anotadores": len(nombres), "nombres": nombres, "comunes": n, "acuerdo_observado": round(po, 3),
            "kappa": kappa, "discrepancias": disc}


def particiones_por_linaje(d: dict, k: int = 2) -> dict:
    """Folds por PLAN: `sha256(plan)` decide el fold, todas las comidas de un plan caen en el mismo. Devuelve los
    folds, los planes por fold y `cruzados` (planes en más de un fold: debe ser 0 por construcción)."""
    folds = collections.defaultdict(list)
    plan_fold = {}
    for i, c in enumerate(d.get("casos") or []):
        u = _unidad(c, i)
        f = int(hashlib.sha256(u.encode("utf-8")).hexdigest(), 16) % max(1, k)
        folds[f].append(str(c.get("id")))
        plan_fold.setdefault(u, set()).add(f)
    cruzados = [u for u, fs in plan_fold.items() if len(fs) > 1]
    return {"k": k, "folds": {str(f): ids for f, ids in sorted(folds.items())},
            "planes_por_fold": {str(f): sum(1 for u, fs in plan_fold.items() if f in fs) for f in sorted(folds)},
            "cruzados": cruzados}


def columnas_de(d: dict, fecha: str | None, respaldo: dict | None = None) -> tuple[dict, list]:
    """[P1-PLAN-LOTE-60] Las columnas de la máquina de una fecha: `maquina_<capa>_<fecha>` si ALGÚN caso la trae; si no,
    la del 09-06, y una nota que lo dice (sin `--con-juez` el refresco no escribe la del juez).

    [P1-PLAN-LOTE-62] `respaldo` (las columnas de `--desde`) manda sobre la del 09-06: una capa que el lote no re-corrió
    se compara consigo misma, no con la de hace nueve días — si no, la tabla muestra un «cambio» que es de columna."""
    out, notas = {}, []
    for capa in ("determinista", "juez"):
        k = f"maquina_{capa}_{fecha}"
        if fecha and any(k in c for c in d.get("casos") or []):
            out[capa] = k
        elif respaldo and respaldo.get(capa) and respaldo[capa] != f"maquina_{capa}":
            out[capa] = respaldo[capa]
            notas.append(f"{capa}: sin columna del {fecha}; se usa `{respaldo[capa]}` (la de --desde)")
        else:
            out[capa] = f"maquina_{capa}"
            if fecha:
                notas.append(f"{capa}: sin columna del {fecha}; se usa `maquina_{capa}` (2026-09-06)")
    return out, notas


def cargar_anotaciones(paths: list) -> dict:
    """Ficheros de anotación externos, uno por anotador:
        {"anotador": "B", "casos": {"<id>": {"veredicto": "ok|defecto|dudoso",
                                             "defectos": [{"clase", "severidad", "evidencia", "alimento"?}]}}}
    y, opcionalmente, {"adjudicacion": {"<id>": {"veredicto", "defectos", "por", "nota"}}}."""
    out = collections.defaultdict(list)
    for p in paths or []:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        anotador = str(doc.get("anotador") or Path(p).stem)
        for cid, a in (doc.get("casos") or {}).items():
            if isinstance(a, dict):
                out[str(cid)].append({"anotador": anotador, **a})
        for cid, a in (doc.get("adjudicacion") or {}).items():
            out[("adjudicacion", str(cid))] = a
    return out


def render(r: dict) -> str:
    o = [f"casos {r['casos']} · etiquetados {r['etiquetados']} · dudosos {r['dudosos']} · "
         f"sin etiquetar {r['sin_etiquetar']}", ""]
    if r["etiquetados"] < MINIMO_ETIQUETAS:
        o += ["  ⛔ Con menos de 20 casos etiquetados estas cifras no significan nada.", ""]
    o.append("  capa            precision      recall        (crudo -> ponderado)          IC95 crudo (por plan)")
    for capa, v in r["capas"].items():
        c, p, ic = v["crudo"], v["ponderado"], v.get("ic95_crudo") or {}
        o.append(f"  {capa:14s}  {str(c['precision']):>5s} -> {str(p['precision']):<6s} "
                 f"{str(c['recall']):>5s} -> {str(p['recall']):<6s}"
                 f"  (tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']})"
                 f"  p={ic.get('precision')} r={ic.get('recall')}")
    o += ["", "  El ponderado corrige el sesgo del muestreo estratificado; el crudo dice lo que se",
          "  conto de verdad. Se publican los dos a proposito.",
          "", "  `dudoso` no cuenta en ninguna direccion: forzarlo a binario contaminaria la medida.",
          "", "  Binario = «marco una comida con defecto». Para saber si marco EL defecto: --estricto."]
    return "\n".join(o)


def render_estricto(r: dict) -> str:
    o = [f"[estricto] casos {r['casos']} · con rubrica adjudicable {r['completos']} (minimo {r['minimo']}) · "
         f"estados {r['estados']}", ""]
    if not r["completo"]:
        o += ["  ⛔ INCOMPLETO: con menos de 20 casos anotados con rubrica estas cifras no significan nada.",
              "     Las etiquetas del 2026-09-07 son binarias (ok/defecto + nota): hace falta la anotacion con clase,",
              "     severidad y evidencia por defecto — `scripts/culinary_golden_sample.py --plantilla` la prepara.", ""]
    o.append("  capa            precision      recall        (crudo -> ponderado)   IC95 crudo")
    for capa, v in r["capas"].items():
        c, p, ic = v["crudo"], v["ponderado"], v.get("ic95_crudo") or {}
        o.append(f"  {capa:14s}  {str(c['precision']):>5s} -> {str(p['precision']):<6s} "
                 f"{str(c['recall']):>5s} -> {str(p['recall']):<6s}  (tp={c['tp']} fp={c['fp']} fn={c['fn']})"
                 f"  p={ic.get('precision')} r={ic.get('recall')}")
        if v.get("emparejados") or v.get("columna") not in (None,) + _COLUMNAS_BASE or v.get("sin_columna"):
            o.append(f"  {'':14s}  columna {v.get('columna')} · emparejados {v.get('emparejados') or {}}"
                     + (f" · casos sin la columna {v['sin_columna']}" if v.get("sin_columna") else ""))
    if r["por_clase"]:
        o.append("  por clase (tp/fp/fn · precision · recall):")
        for k, v in r["por_clase"].items():
            o.append(f"    {k:26s} {v['tp']}/{v['fp']}/{v['fn']} · {v['precision']} · {v['recall']}")
    if r["no_mecanizable"]:
        o.append(f"  defectos que ninguna capa mecaniza (FN del sistema entero): {r['no_mecanizable']}")
    if r.get("sin_rubrica_casos"):
        o.append("  «defecto» sin ningun defecto de la rubrica (no puntuan; completarlos en la hoja): "
                 + ", ".join(str(x) for x in r["sin_rubrica_casos"]))
    a = r.get("acuerdo") or {}
    o.append(f"  acuerdo: anotadores={a.get('anotadores')} comunes={a.get('comunes')} kappa={a.get('kappa')} "
             f"discrepancias={len(a.get('discrepancias') or [])}")
    o.append(f"  promocion habilitada: {r['promocion_habilitada']} (exige >= {MINIMO_ETIQUETAS} completos, 0 pendientes "
             f"de adjudicar y 2 anotadores)")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--estricto", action="store_true", help="[C1] por hallazgo, con rubrica; exit 4 si incompleto")
    ap.add_argument("--anotaciones", action="append", default=[], help="[C1] fichero(s) de anotacion externos")
    ap.add_argument("--particiones", type=int, default=0, help="[C1] k folds por linaje (plan)")
    ap.add_argument("--con-dudosas", action="store_true", help="[P1-PLAN-LOTE-28] cuenta tambien los hallazgos [dudosa] del juez")
    ap.add_argument("--excluir-dev", help="[P1-PLAN-LOTE-28] JSON con ids de casos de desarrollo que NO son holdout")
    ap.add_argument("--maquina", help="[P1-PLAN-LOTE-60] puntua con las columnas maquina_<capa>_<FECHA> del refresco")
    ap.add_argument("--comparar-maquina", help="[P1-PLAN-LOTE-60] con --estricto: antes (09-06) y despues (<FECHA>) a la vez")
    ap.add_argument("--desde", help="[P1-PLAN-LOTE-62] con --comparar-maquina: el ANTES es la columna de esta fecha, no la del 09-06")
    ap.add_argument("--juez-por-codigo", metavar="COLUMNA", help="[P1-PLAN-LOTE-63] el juez por codigo (estricta y por sustancia)")
    ap.add_argument("--observacion", default="", help="[P1-PLAN-LOTE-63] codigos del juez que cuentan como [dudosa] (simula el knob "
                                                      "MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES sobre una columna ya escrita)")
    a = ap.parse_args()
    global INCLUIR_DUDOSAS, OBSERVACION
    INCLUIR_DUDOSAS = bool(a.con_dudosas)
    OBSERVACION = frozenset(x.strip() for x in (a.observacion or "").split(",") if x.strip())
    if not GOLDEN.exists():
        print(f"no existe {GOLDEN.name}: crealo con scripts/culinary_golden_sample.py")
        return 1
    d = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if a.excluir_dev:
        d = excluir_casos(d, json.loads(Path(a.excluir_dev).read_text(encoding="utf-8")))
    if a.particiones:
        part = particiones_por_linaje(d, a.particiones)
        print(json.dumps(part, ensure_ascii=False, indent=2) if a.json else
              f"particiones k={part['k']} · planes por fold {part['planes_por_fold']} · cruzados {len(part['cruzados'])}")
        return 0 if not part["cruzados"] else 3
    columnas, notas = columnas_de(d, a.maquina)
    if a.juez_por_codigo:
        t = tabla_juez_por_codigo(d, cargar_anotaciones(a.anotaciones), a.juez_por_codigo, OBSERVACION)
        t["en_observacion_por_el_criterio"] = codigos_en_observacion(t)
        print(json.dumps(t, ensure_ascii=False, indent=2) if a.json
              else render_juez_por_codigo(t) + f"\n  criterio del plan (estricta < 25 %, n >= 4): {t['en_observacion_por_el_criterio']}")
        return 0
    if a.estricto and a.comparar_maquina:
        # [P1-PLAN-LOTE-60] la línea base estricta: el MISMO adjudicador sobre las columnas del 09-06 y las refrescadas
        ext = cargar_anotaciones(a.anotaciones)
        antes_cols, notas_antes = columnas_de(d, a.desde) if a.desde else (None, [])
        despues_cols, notas = columnas_de(d, a.comparar_maquina, antes_cols)
        out = {"anotaciones": [Path(p).name for p in a.anotaciones], "notas": notas_antes + notas,
               "antes": puntuar_estricto(d, ext, antes_cols), "despues": puntuar_estricto(d, ext, despues_cols)}
        for k in ("antes", "despues"):
            out[k]["dudosas_excluidas"] = 0 if INCLUIR_DUDOSAS else contar_dudosas(d, out[k]["columnas"]["juez"])
            out[k]["excluidos_dev"] = d.get("excluidos_dev", 0)
        if a.json:
            print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        else:
            print("\n".join([f"== ANTES · {out['antes']['columnas']} ==", render_estricto(out["antes"]), "",
                             f"== DESPUES · {out['despues']['columnas']} ==", *[f"  nota: {n}" for n in notas],
                             render_estricto(out["despues"])]))
        return 0 if (out["antes"]["completo"] and out["despues"]["completo"]) else 4
    if a.estricto:
        r = puntuar_estricto(d, cargar_anotaciones(a.anotaciones), columnas)
        r["dudosas_excluidas"] = 0 if INCLUIR_DUDOSAS else contar_dudosas(d, r["columnas"]["juez"])
        r["excluidos_dev"] = d.get("excluidos_dev", 0)
        r["notas_columnas"] = notas
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str) if a.json
              else "\n".join([*[f"  nota: {n}" for n in notas], render_estricto(r)]))
        return 0 if r["completo"] else 4
    r = puntuar(d, columnas)
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r))
    # [P1-SCORE-INCOMPLETE-EXIT · 2026-09-07] Un experimento SIN etiquetas suficientes salia con
    # codigo 0 y metricas `null`: para CI y para cualquier consumidor eso es indistinguible de
    # "medido y correcto". El aviso en pantalla solo lo ve una persona que ademas lo lea.
    # Exit 4 = incompleto (no es un fallo del programa, es la ausencia de la referencia humana).
    if r["etiquetados"] < MINIMO_ETIQUETAS:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
