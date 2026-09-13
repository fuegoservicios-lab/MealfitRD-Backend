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
    (y, si el defecto nombra `alimento`, el texto de la máquina lo menciona); si no, es un FP localizado y el
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
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
GOLDEN = _BACKEND / "docs" / "culinary_golden_set.json"

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
    "coccion_faltante": set(),
    "otro": set(),
}
SEVERIDADES = ("minor", "high")
_CODIGOS_DET = {"V1", "V2", "V3", "V4", "V5", "V6", "V7a", "V7b", "V7c", "V7d", "V7e", "V8a", "V8b", "V9"}
BOOTSTRAP_N = 1000
BOOTSTRAP_SEMILLA = 20260912


def _sin_acentos(s) -> str:
    t = unicodedata.normalize("NFD", str(s or ""))
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn").lower()


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


def puntuar(d: dict) -> dict:
    """El marcador BINARIO (por comida): ¿la capa marcó una comida que la persona marcó?"""
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

    for capa, clave in (("determinista", "maquina_determinista"), ("juez", "maquina_juez")):
        tp = fp = fn = tn = 0.0
        tp_n = fp_n = fn_n = tn_n = 0
        for c in etiquetados:
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
            "ic95_crudo": _bootstrap(etiquetados, _conteo),   # [C1] por conglomerado (plan)
        }
    return salida


# ── [C1] anotaciones con rúbrica, acuerdo y adjudicación ──────────────────────────────────────

def _codigo(texto: str) -> str:
    """`"V4: ..."` → `V4`; `"paso_incoherente: ..."` → `paso_incoherente`."""
    return str(texto or "").split(":", 1)[0].strip()


def _hallazgos_maquina(caso, clave) -> list:
    """Hallazgos de la máquina deduplicados (mismo texto = mismo hallazgo): los duplicados no multiplican TP."""
    vistos, out = set(), []
    for t in caso.get(clave) or []:
        k = str(t).strip()
        if k and k not in vistos:
            vistos.add(k)
            out.append({"codigo": _codigo(k), "texto": k})
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


def _verdad_estricta(caso, externas: dict) -> dict:
    """Qué se toma como verdad para el estricto: `adjudicacion` > anotación única > acuerdo entre varias.
    Estados: `completo`, `sin_rubrica` (sólo binaria), `pendiente_adjudicacion`, `sin_anotar`, `dudoso`."""
    adj = caso.get("adjudicacion") or externas.get(("adjudicacion", str(caso.get("id"))))
    if isinstance(adj, dict) and adj.get("veredicto") is not None:
        vb = _veredicto_bin(adj.get("veredicto"))
        if vb is None:
            return {"estado": "dudoso", "defectos": [], "veredicto": None, "por": "adjudicacion"}
        defs, raros = _defectos_validos(adj.get("defectos") or [])
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
        return {"estado": "completo", "defectos": defs if vb else [], "veredicto": vb, "por": a.get("anotador"), "raros": raros}
    # dos o más con rúbrica: coinciden en veredicto Y en el conjunto de clases ⇒ completo; si no, pendiente
    vbs = {_veredicto_bin(a.get("veredicto")) for a in con_rubrica}
    clases = [frozenset(df.get("clase") for df in _defectos_validos(a.get("defectos"))[0]) for a in con_rubrica]
    if len(vbs) == 1 and None not in vbs and len(set(clases)) == 1:
        a = con_rubrica[0]
        defs, raros = _defectos_validos(a.get("defectos"))
        return {"estado": "completo", "defectos": defs if vbs == {True} else [], "veredicto": vbs.pop(),
                "por": "acuerdo:" + "+".join(str(x.get("anotador")) for x in con_rubrica), "raros": raros}
    return {"estado": "pendiente_adjudicacion", "defectos": [], "veredicto": None, "por": None}


def _adjudicar_hallazgos(maquina: list, defectos: list, codigos_capa: set) -> tuple[int, int, int, list]:
    """Emparejamiento hallazgo↔defecto, cada uno como mucho una vez. Devuelve (tp, fp, fn, detalle)."""
    usados = set()
    tp = 0
    detalle = []
    for df in defectos:
        codigos = RUBRICA.get(df.get("clase"), set()) & codigos_capa
        alimento = _sin_acentos(df.get("alimento") or "").strip()
        elegido = None
        for i, h in enumerate(maquina):
            if i in usados or h["codigo"] not in codigos:
                continue
            if alimento and alimento not in _sin_acentos(h["texto"]):
                continue
            elegido = i
            break
        if elegido is not None:
            usados.add(elegido)
            tp += 1
            detalle.append({"defecto": df.get("clase"), "hallazgo": maquina[elegido]["texto"][:80], "resultado": "tp"})
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


def puntuar_estricto(d: dict, externas: dict | None = None) -> dict:
    """El marcador ESTRICTO (por hallazgo). `externas`: anotaciones cargadas de ficheros, por `id` de caso."""
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
        tp_n = fp_n = fn_n = 0
        tp_w = fp_w = fn_w = 0.0
        conteos = {}
        for c, v in completos:
            maquina = _hallazgos_maquina(c, clave)
            # a cada capa sólo se le exigen los defectos de SU competencia (los que algún código suyo cubre)
            defectos = [df for df in v["defectos"] if RUBRICA.get(df.get("clase"), set()) & codigos]
            tp, fp, fn, det = _adjudicar_hallazgos(maquina, defectos, codigos)
            w = peso(c)
            tp_n += tp; fp_n += fp; fn_n += fn
            tp_w += tp * w; fp_w += fp * w; fn_w += fn * w
            conteos[id(c)] = (tp, fp, fn)
            for x in det:
                if x["resultado"] == "tp":
                    por_clase[x["defecto"]]["tp"] += 1
                elif x["resultado"] == "fn":
                    por_clase[x["defecto"]]["fn"] += 1
                elif x["resultado"] == "fn_no_mecanizable":
                    salida["no_mecanizable"][x["defecto"]] += 1
                salida["detalle"].append({"caso": c.get("id"), "capa": capa, **x})
        salida["capas"][capa] = {
            "crudo": {"tp": tp_n, "fp": fp_n, "fn": fn_n, "precision": _r(tp_n, fp_n), "recall": _r(tp_n, fn_n)},
            "ponderado": {"precision": _r(tp_w, fp_w), "recall": _r(tp_w, fn_w)},
            "ic95_crudo": _bootstrap([c for c, _ in completos], lambda c, k=conteos: k.get(id(c), (0, 0, 0))),
        }
    salida["por_clase"] = {k: {**v, "precision": _r(v["tp"], v["fp"]), "recall": _r(v["tp"], v["fn"])}
                           for k, v in sorted(por_clase.items())}
    salida["no_mecanizable"] = dict(salida["no_mecanizable"])
    salida["acuerdo"] = acuerdo(d, externas)
    salida["promocion_habilitada"] = bool(salida["completo"] and estados.get("pendiente_adjudicacion", 0) == 0
                                          and (salida["acuerdo"] or {}).get("anotadores", 0) >= 2)
    return salida


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
    if r["por_clase"]:
        o.append("  por clase (tp/fp/fn · precision · recall):")
        for k, v in r["por_clase"].items():
            o.append(f"    {k:26s} {v['tp']}/{v['fp']}/{v['fn']} · {v['precision']} · {v['recall']}")
    if r["no_mecanizable"]:
        o.append(f"  defectos que ninguna capa mecaniza (FN del sistema entero): {r['no_mecanizable']}")
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
    a = ap.parse_args()
    if not GOLDEN.exists():
        print(f"no existe {GOLDEN.name}: crealo con scripts/culinary_golden_sample.py")
        return 1
    d = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if a.particiones:
        part = particiones_por_linaje(d, a.particiones)
        print(json.dumps(part, ensure_ascii=False, indent=2) if a.json else
              f"particiones k={part['k']} · planes por fold {part['planes_por_fold']} · cruzados {len(part['cruzados'])}")
        return 0 if not part["cruzados"] else 3
    if a.estricto:
        r = puntuar_estricto(d, cargar_anotaciones(a.anotaciones))
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str) if a.json else render_estricto(r))
        return 0 if r["completo"] else 4
    r = puntuar(d)
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
