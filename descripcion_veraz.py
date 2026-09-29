# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-854 · 2026-09-29] La ficha dice lo que el plato ES: ni lo que el prompt pidió ni lo que el plato fue.

## De dónde sale

La validación beta G24 (29-sep, 6 planes reales con P1-PLAN-LOTE-815 desplegado) leyó cada comida entera. En los 6
países la descripción (`desc`) hacía dos cosas que el usuario no debería ver:

1. **Repetía el lenguaje del prompt**, la justificación de la regla en vez del plato: «Una preparación transformada con
   identidad propia, distinta al almuerzo del día», «un desayuno de categoría Avena/Cereales», «Sin avena y sin repetir
   la base del desayuno», «Base de tubérculo local: …», «ese toque fresco que pide el tema del día».
2. **Nombraba lo que el plato ya no lleva**, porque la cola de reparación cambia ingredientes después de que alguien
   escribió la ficha: ES D2 Cena «pechuga de pollo» con pavo en la lista; MX D2 Merienda «manzana crujiente» con
   lechosa; PR D3 Merienda «sin lácteos… mango» con lechosa y cottage; CO D2 Desayuno «cocida en leche» con agua.

Medido sobre 9 755 fichas de 780 planes guardados (baterías + producción): 539 dicen «identidad propia», 591 comparan
con otra comida («distinta al almuerzo»), 284 «no/sin repetir», 117 citan la categoría del desayuno. Los nombres de los
platos están sanos (0 de 9 755), así que el limpiador del nombre sólo existe como red.

## Parte 1 — metalenguaje: una lista CERRADA, cada familia con su frase de origen

No hay heurística de «suena a prompt». `FAMILIAS_DEL_PROMPT` enumera las familias; cada una cita la frase literal del
prompt de la que sale (el test comprueba que esa frase sigue en el código de los prompts: si alguien la borra del
prompt, la familia deja de tener razón de existir y el test lo dice). Tres formas de quitar, de menos a más:

- **etiqueta** delante de «:» («Cena con identidad propia: pechuga…» → «Pechuga…»);
- **modificador** dentro de la frase («un desayuno de categoría Avena/Cereales que…» → «un desayuno que…»);
- **cláusula** de comparación o justificación («, distinta al almuerzo», «Sin avena y sin repetir la base del
  desayuno.»), que se quita entera hasta el signo de puntuación o la «y» que la cierra.

Una frase que tras limpiar queda en un muñón genérico de ≤3 palabras («Una cena ligera.», «Plato criollo.») se retira;
una ficha que quedaría con menos de 4 palabras NO se toca: dejarla con metalenguaje es malo, dejarla vacía es peor.

Replay sin IA (29-sep): G24 21 de 60 fichas beta cambian y DO 0 de 12 (hash idéntico); las 9 755 fichas guardadas con
DO forzado cambiarían 1 807 (1 006 textos únicos), leídos 145 cambios de ingredientes y ~160 de metalenguaje; 0
anomalías del detector (doble signo, conector colgando, minúscula tras punto, restos del prompt).

## Parte 2 — la ficha frente a los ingredientes: la maquinaria que ya existía, en la cola

`graph_orchestrator._desc_food_honesty_pass` (P1-DESC-FOOD-HONESTY) ya tenía la escalera —cambiar por el de la misma
familia con su artículo, retirar la mención si cierra la cláusula, y si nada es seguro dejarla—. No se reescribe: se
reutiliza su vocabulario, sus subgrupos y sus ayudantes de género. Lo que le faltaba, medido en G24:

- corría DENTRO de `finalize_plan_data_coherence`, antes de la cola que sigue cambiando ingredientes;
- su regla del token siguiente sólo admitía su lista blanca de adjetivos aunque el cambio no toque el género
  («pechuga de pollo **salteada**», «pollo **desmenuzado**», «manzana **crujiente**» quedaban sin arreglar);
- su ventana de negación (24 caracteres) cruzaba el «que»: en «Merienda sin lácteos que combina mango» el mango
  contaba como negado;
- la presencia se decidía con expresiones propias; aquí la deciden los resolvedores SSOT de nombres
  (`normalize_ingredient_for_tracking`, `pantry_names_match` con los alias del catálogo y tokens completos, nunca
  subcadenas: «res» ⊄ «fresco»).

Además: «sin lácteos» cuando el plato lleva cottage es falso y se quita; «cocida en leche» sin leche y con agua pasa a
«cocida en agua».

## Alcance

Sólo planes beta (`_country` ≠ DO por `constants.canonicalize_country`): DO es el control de la validación beta y no
cambia NADA. `MEALFIT_DESCRIPTION_TRUTH_DO` (False) lo abre cuando el dueño lo decida; el defecto en DO existe (lo
medido arriba es sobre todo DO). Knob maestro `MEALFIT_DESCRIPTION_TRUTH` (True). Sólo `desc` (y el nombre, como red);
nunca los ingredientes ni los pasos: los nombres de alimentos del catálogo son identificadores del motor.

tooltip-anchor: P1-PLAN-LOTE-854-FICHA-VERAZ
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional

logger = logging.getLogger(__name__)

_L = r"[^\W\d_]"                    # una letra (incluye acentos y ñ)
_PAL = _L + r"+"                     # una palabra
_COMIDA = r"(?:almuerzos?|desayunos?|cenas?|meriendas?)"


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DESCRIPTION_TRUTH", True)
    except Exception:
        return True


def incluye_do() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DESCRIPTION_TRUTH_DO", False)
    except Exception:
        return False


# ============================================================ Parte 1: el metalenguaje del prompt
#: La lista cerrada. `origen` es la frase LITERAL del prompt (o del código que lo arma) de la que sale la familia;
#: `tipo` dice cómo se quita. tooltip-anchor: FAMILIAS_DEL_PROMPT (test_p1_plan_lote_854.py)
FAMILIAS_DEL_PROMPT: tuple = (
    {"id": "comparacion_con_otra_comida", "tipo": "clausula",
     "origen": "tubérculo distinto al del almuerzo",
     "rx": r"\b(?:(?:con\s+|de\s+)?(?:una?\s+)?(?:base|preparaci[oó]n|perfil|alternativa|opci[oó]n|textura|t[eé]cnica|"
           r"forma|manera|prote[ií]na|guarnici[oó]n|carbohidrato|cena|merienda)\s+"
           r"(?:" + _PAL + r"\s+){0,3}?)?distint[ao]s?\s+(?:a|al|de|del)\b(?:\s+" + _PAL + r"){0,6}?\s+"
           r"(?:" + _COMIDA + r"|d[ií]as?|plan|semana|comidas?)\b"},
    {"id": "base_distinta", "tipo": "clausula",
     "origen": "una base distinta DE TU categoría",
     "rx": r"\b(?:(?:con\s+)?(?:una?\s+)?)?base\s+(?:de\s+" + _PAL + r"(?:\s+" + _PAL + r")?\s+)?distint[ao]s?\b"},
    {"id": "mas_ligera_que", "tipo": "clausula",
     "origen": "más ligera que el almuerzo",
     "rx": r"\b(?:y\s+)?m[aá]s\s+(?:liger|suave|livian)" + _L + r"*\s+que\s+(?:el|la)\s+" + _COMIDA + r"\b"},
    {"id": "no_repetir", "tipo": "clausula",
     "origen": "PROHIBIDO repetir la PROTEÍNA PRINCIPAL",
     "rx": r"\b(?:(?:para\s+)?no\s+repetir|sin\s+repetir|evitando\s+repetir|(?:rompe|romper|evita|evitar)\s+la\s+"
           r"repetici[oó]n)\b|\bsin\s+(?:" + _PAL + r"\s+){1,4}?(?:ni\s+(?:" + _PAL + r"\s+){1,2}?)?repetid[oa]s?\b"},
    {"id": "base_repetida_entre_dias", "tipo": "clausula",
     "origen": "NUNCA repitas la misma categoría base",
     "rx": r"\b(?:una?\s+)?opci[oó]n\s+(?:de\s+" + _PAL + r"\s+)?sin\s+(?:avena|yogur|yogurt|cereales?)\b"
           r"|\b(?:una?\s+)?alternativa\b(?:\s+" + _PAL + r"){0,3}?\s+(?:a|al|a\s+la|del?)\s+(?:(?:la|el|los|las)\s+)?"
           + _COMIDA + r"\b"},
    {"id": "variar", "tipo": "clausula",
     "origen": "tanto mejor variar",
     "rx": r"\bpara\s+variar\b"},
    {"id": "staples", "tipo": "clausula",
     "origen": "staples servidos",
     "rx": r"\bstaples?\b"},
    {"id": "categoria_asignada", "tipo": "clausula",
     "origen": "CATEGORÍA DE DESAYUNO ASIGNADA",
     "rx": r"\b(?:(?:tal\s+)?como\s+pide|cumple|respeta)\s+(?:con\s+)?(?:la\s+)?categor[ií]a\b"
           r"|\b(?:(?:con|de)\s+(?:la\s+)?)?categor[ií]a\s+(?:de\s+" + _PAL + r"(?:\s*/\s*" + _PAL + r")?\s+)?asignada\b"},
    {"id": "tecnica_asignada", "tipo": "clausula",
     "origen": "tu técnica asignada es la identidad",
     "rx": r"\b(?:(?:con|de)\s+(?:la\s+)?)?t[eé]cnica\s+asignada\b"},
    {"id": "identidad_propia", "tipo": "modificador",
     "origen": "preparación REAL con identidad propia",
     "rx": r"\s*(?:\by\s+)?\b(?:con|de)\s+(?:una?\s+)?identidad\s+propia\b|\bidentidad\s+propia\b"},
    {"id": "nombre_propio", "tipo": "modificador",
     "origen": "un plato con nombre propio se disfruta",
     "rx": r"\s*(?:\by\s+)?\bcon\s+nombre\s+propio\b"},
    {"id": "transformada", "tipo": "modificador",
     "origen": "PREPARACIONES TRANSFORMADAS",
     "rx": r"\s+transformad[ao]s?\b(?!\s+(?:en|a)\b)(?:\s*,(?=\s)|\s+y(?=\s))?"},
    {"id": "categoria_del_desayuno", "tipo": "modificador",
     "origen": "CATEGORÍA DE DESAYUNO ASIGNADA",
     "rx": r"\s*\b(?:de\s+(?:la\s+)?|la\s+)?categor[ií]a\s+(?:de\s+)?(?:mang[uú]|tub[eé]rculos?|avena|cereales?|pan|"
           r"tostadas?|batido|bowl|revoltillo|tortilla)(?:\s*/\s*" + _PAL + r")?(?:\s+asignada)?\b"},
    {"id": "etiqueta_beta_del_desayuno", "tipo": "modificador",
     "origen": "Base de tubérculo local",
     "rx": r"\bbase\s+de\s+tub[eé]rculo\s+local\b"},
    {"id": "concepto_tematico", "tipo": "modificador",
     "origen": "CONCEPTO TEMÁTICO",
     "rx": r"\s*\b(?:que|como|seg[uú]n)\s+(?:lo\s+)?(?:pide\s+)?el\s+(?:tema|concepto)(?:\s+tem[aá]tico)?\s+del\s+d[ií]a\b"},
)

_RX = {f["id"]: re.compile(f["rx"], re.IGNORECASE) for f in FAMILIAS_DEL_PROMPT}
_CLAUSULAS = [(f["id"], _RX[f["id"]]) for f in FAMILIAS_DEL_PROMPT if f["tipo"] == "clausula"]
_MODIFICADORES = [(f["id"], _RX[f["id"]]) for f in FAMILIAS_DEL_PROMPT if f["tipo"] == "modificador"]
_REF_COMIDA = re.compile(r"\b(?:del|al|de\s+la|de\s+las|de\s+los)\s+(?:resto\s+del?\s+|otros\s+|otras\s+)?"
                         r"(?:" + _COMIDA + r"|d[ií]as?|plan|semana)\b", re.IGNORECASE)
_COLA_SUELTA = re.compile(r"(?:\s+|^)(?:y|e|o|con|de|del|a|al|para|que|una?|el|la|los|las|es|sin|ni|pero)$",
                          re.IGNORECASE)
_CABEZA_SUELTA = re.compile(r"^(?:y|e|pero)\s+", re.IGNORECASE)
#: Verbos cuyo objeto era la comparación quitada («ofrece un perfil distinto al…», «aporta una alternativa al…»).
_VERBO_COLGANTE = re.compile(r"(?:^|\s+)(?:ofrece|aporta|tiene|lleva|brinda|presenta|mantiene|usa|es|queda|resulta|"
                             r"cambia|var[ií]a|rompe|evita|sustituye|reemplaza|"
                             r"ofreciendo|aportando|manteniendo)$", re.IGNORECASE)
#: «Cena de identidad propia y alto valor muscular» → «Cena de alto valor muscular»: la preposición es del resto.
_PROPIO_Y = re.compile(r"\b(con|de)\s+(?:una?\s+)?(?:identidad|nombre)\s+propi[ao]\s+y\s+", re.IGNORECASE)
#: Una oración que empieza por un sustantivo genérico de comida: con ≤3 palabras tras limpiar no dice nada del plato.
_GENERICA = re.compile(r"^(?:(?:una?|la|el)\s+)?(?:cena|merienda|desayuno|almuerzo|comida|preparaci[oó]n|opci[oó]n|"
                       r"plato|snack|pausa|bocado)\b", re.IGNORECASE)
#: Lo que queda al final de una oración tras quitar su comparación: «…con aguacate: una cena.»
_MUNON_FINAL = re.compile(r"^(?:una?|la|el)\s+(?:cena|merienda|desayuno|almuerzo|comida|preparaci[oó]n|opci[oó]n|"
                          r"plato)(?:\s+distint[ao])?$", re.IGNORECASE)


def _tiene_meta(texto: str) -> bool:
    return any(rx.search(texto) for _i, rx in _CLAUSULAS) or any(rx.search(texto) for _i, rx in _MODIFICADORES)


def _cap(s: str) -> str:
    return (s[0].upper() + s[1:]) if s and s[0].islower() else s


def _podar(s: str) -> str:
    """Espacios, conectores colgando y signos huérfanos que deja una poda."""
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\s+([,;:])", r"\1", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)            # «sustancioso,, para» → «sustancioso, para»
    for _ in range(4):
        s2 = _COLA_SUELTA.sub("", s).strip().rstrip(",;:").strip()
        s2 = _CABEZA_SUELTA.sub("", s2).strip()
        if s2 == s:
            break
        s = s2
    return s


def _limpiar_parte(parte: str) -> tuple[str, bool]:
    """Una «parte» es un tramo sin comas ni «y» que la corten. Devuelve (texto, ¿se quitó por cláusula?)."""
    for _id, rx in _CLAUSULAS:
        m = rx.search(parte)
        if not m:
            continue
        izquierda = _podar(parte[:m.start()])
        # «…y ofrece un perfil distinto al bowl del almuerzo»: el verbo se queda sin objeto → fuera con él.
        izquierda = _podar(_VERBO_COLGANTE.sub("", izquierda))
        # «sin frutas tropicales para variar…»: la ausencia ES la justificación → fuera entera.
        if not izquierda or re.match(r"^(?:sin|ni)\b", izquierda, re.IGNORECASE):
            return "", True
        return izquierda, True
    return parte, False


def _cerrar_enumeracion(izq: str) -> str:
    """Quitado el último miembro («cálido, dulce y con nombre propio»), la coma de antes pasa a «y»: «cálido y dulce»."""
    m = re.search(r",\s+(" + _PAL + r"(?:\s+" + _PAL + r"){0,2})\s*$", izq)
    return (izq[:m.start()] + " y " + m.group(1)) if m else izq


def _limpiar_segmento(seg: str) -> tuple[str, bool, bool]:
    """Un segmento va entre [,;:—]. Se trocea por « y » sólo si lleva metalenguaje (las listas de alimentos quedan).
    Devuelve (texto, ¿hubo cláusula?, ¿se quitó el último miembro tras una «y»?)."""
    if not any(rx.search(seg) for _i, rx in _CLAUSULAS):
        return seg, False, False
    trozos = re.split(r"(\s+[ye]\s+)", seg)
    partes, seps = trozos[0::2], [""] + trozos[1::2]
    salida: list = []                      # [(sep, texto, quitada_por_clausula)]
    hubo_clausula = proposito = False
    for sep, parte in zip(seps, partes):
        if proposito or (hubo_clausula and _REF_COMIDA.search(parte)):
            continue                        # «sin repetir el pollo | y el arroz del almuerzo»: sigue la comparación
        texto, por_clausula = _limpiar_parte(parte)
        # «para variar los acompañamientos | y moderar la sal»: el propósito quitado se lleva sus coordinadas
        proposito = por_clausula and bool(re.search(r"\bpara\s+(?:no\s+)?(?:repetir|variar)\b", parte, re.I))
        if por_clausula:
            hubo_clausula = True
            if not texto and salida and re.match(r"^(?:sin|ni)\b", salida[-1][1], re.IGNORECASE):
                salida.pop()                # «Sin avena | y sin repetir la base…»: la ausencia era la justificación
        if texto:
            salida.append((sep, texto, por_clausula))
    if not salida:
        return "", True, False
    quitada_tras_y = len(salida) == 1 and len(partes) > 1 and salida[0][1] == partes[0]
    out = salida[0][1]
    for sep, texto, _q in salida[1:]:
        out += sep + texto
    return _podar(out), hubo_clausula, quitada_tras_y


def _limpiar_oracion(oracion: str) -> str:
    s = oracion.strip()
    m_fin = re.search(r"[.!?…]+$", s)
    fin = m_fin.group(0) if m_fin else ""
    cuerpo = s[:m_fin.start()] if m_fin else s
    if not _tiene_meta(cuerpo):
        return oracion.strip()
    # 1) etiqueta delante de los dos puntos
    m = re.match(r"^([^:]{2,90}):\s+(\S.*)$", cuerpo)
    if m and _tiene_meta(m.group(1)) and len(m.group(2).split()) >= 3:
        # la etiqueta se limpia como una oración; si queda en muñón («Cena con identidad propia» → «Cena») se va
        etiqueta = re.sub(r"[.!?…]+$", "", _limpiar_oracion(m.group(1)))
        cuerpo = (etiqueta + ": " + m.group(2)) if etiqueta else _cap(m.group(2))
    # 2) modificadores, sobre la oración entera (una coma tras «transformada» es suya, no del segmento)
    cuerpo = _PROPIO_Y.sub(lambda mm: mm.group(1) + " ", cuerpo)
    for _id, rx in _MODIFICADORES:
        for _ in range(4):
            mm = rx.search(cuerpo)
            if not mm:
                break
            izq, der = cuerpo[:mm.start()], cuerpo[mm.end():]
            if re.match(r"^\s*y\s", mm.group(0)) and not re.match(r"^\s*\w", der):
                izq = _cerrar_enumeracion(izq)      # «cálido, dulce y con nombre propio» → «cálido y dulce»
            elif der.startswith(",") and len(re.split(r"[,.;:]", izq)[-1].split()) <= 3 \
                    and not any(rx.match(der[1:].lstrip()) for _i, rx in _CLAUSULAS):
                der = der[1:]                       # «una cena con identidad propia, cálida» → «una cena cálida»
            cuerpo = izq + der
    cuerpo = _cap(_podar(cuerpo))
    # 3) cláusulas, segmento a segmento
    trozos = re.split(r"(\s*[,;:]\s+|\s+—\s+)", cuerpo)
    segs, seps = trozos[0::2], [""] + trozos[1::2]
    salida: list = []
    anterior_quitado = False
    for sep, seg in zip(seps, segs):
        texto, por_clausula, quitada_tras_y = _limpiar_segmento(seg)
        if por_clausula and not texto and salida and re.match(r"^(?:sin|ni)\b", salida[-1][1], re.IGNORECASE) \
                and len(salida[-1][1].split()) <= 6:
            salida.pop()                    # «Sin avena ni yogurt, | para no repetir bases del día»
        if texto and quitada_tras_y and salida and sep.strip() == "," and len(texto.split()) <= 3:
            sep = " y "                     # «una cena vegetal, sabrosa y distinta al…» → «una cena vegetal y sabrosa»
        if texto and anterior_quitado and salida and sep.strip() == "," and re.match(r"^que\b", texto):
            sep = " "                       # «un plato de cuchara, distinto del almuerzo, que cierra» → «… cuchara que»
        if texto:
            salida.append((sep, texto))
        anterior_quitado = not texto
        ultimo_recortado = bool(texto) and por_clausula
    if not salida:
        return ""
    if len(salida) > 1 and (anterior_quitado or ultimo_recortado) and _MUNON_FINAL.match(salida[-1][1].strip()):
        salida.pop()                        # «…con aguacate: una cena[, sin arroz y más ligera que el almuerzo].»
    out = salida[0][1]
    for sep, texto in salida[1:]:
        out += sep + texto
    out = _cap(_podar(out))
    if len(out.split()) <= 1 or (len(out.split()) <= 3 and _GENERICA.match(out)):
        return ""                           # «Una cena ligera.», «Una preparación.»: lo que queda es un muñón
    return out + (fin or ".")


def limpiar_metalenguaje(texto: Any, *, minimo_palabras: int = 4) -> Any:
    """Quita de un texto las frases del prompt (lista cerrada). Idempotente. Si quedaría casi vacío, NO toca nada."""
    if not isinstance(texto, str) or not texto.strip():
        return texto
    base = unicodedata.normalize("NFC", texto)
    if not _tiene_meta(base):
        return texto
    oraciones = re.split(r"(?<=[.!?…])\s+(?=\S)", base.strip())
    limpias = [o for o in (_limpiar_oracion(x) for x in oraciones) if o]
    nuevo = " ".join(limpias).strip()
    if len(nuevo.split()) < minimo_palabras:
        return texto
    return nuevo if nuevo != base.strip() else texto


# ============================================================ Parte 2: la ficha frente a los ingredientes
def _go():
    import graph_orchestrator as go   # import perezoso: graph_orchestrator importa db_plans, que nos llama
    return go


def _sa(s: str) -> str:
    from constants import strip_accents
    return strip_accents(s)


def _tokens(s: str) -> list:
    from constants import canonical_pantry_key
    return canonical_pantry_key(s).split()


def _variantes(t: str) -> set:
    from constants import _pantry_token_variants
    return _pantry_token_variants(t) or {t}


def presente(alimento: str, ingredientes: list) -> bool:
    """¿El plato lleva `alimento`? Sólo resolvedores SSOT y tokens COMPLETOS (nunca subcadenas)."""
    from constants import normalize_ingredient_for_tracking as _nz, pantry_names_match as _pm
    fn = _nz(alimento)
    ft = _tokens(alimento)
    ftn = _tokens(fn) if fn else []
    for ing in ingredientes or []:
        s = str(ing or "")
        inn = _nz(s)
        if fn and inn and (fn == inn or _pm(fn, inn)):
            return True
        for mios in (ft, ftn):
            if not mios:
                continue
            for pool in (_tokens(s), _tokens(inn) if inn else []):
                if pool and all(any(_variantes(a) & _variantes(b) for b in pool) for a in mios):
                    return True
    return False


#: Familias ESTRECHAS para el cambio de palabra: el subgrupo «carne» de P1-DESC-FOOD-HONESTY junta huevo, atún, pollo y
#: cerdo; aquí sólo se cambia dentro de la familia estrecha (pollo↔pavo, sí; atún↔pollo, no).
_FAMILIA = {"pollo": "ave", "pavo": "ave", "cerdo": "carne_roja", "chivo": "carne_roja",
            "atun": "mar", "mero": "mar", "salmon": "mar", "camaron": "mar"}
#: Con el MISMO género y familia estrecha, el adjetivo que sigue concuerda igual: «pechuga de pollo salteada» →
#: «pechuga de pavo salteada». Lo que sigue NO puede ser un calificativo que cambie de producto.
_CALIFICATIVOS = {"en", "de", "integral", "integrales", "verde", "verdes", "rojo", "roja", "rojos", "rojas",
                  "amarillo", "amarilla", "ahumado", "ahumada", "enlatado", "enlatada", "seco", "seca", "secos",
                  "secas", "deshidratado", "deshidratada", "light", "entero", "entera"}
_RELAJADAS = {"ave", "carne_roja", "dulce"}
_NEGACION = re.compile(r"\b(?:sin|ni|no|nada\s+de|libre\s+de|en\s+vez\s+de|en\s+lugar\s+de|sustituy\w+|reemplaz\w+)\b"
                       r"(?:\s+\S+){0,3}\s*$", re.IGNORECASE)
_OTRA_COMIDA = re.compile(r"^\s+(?:de\s+(?:la|los|las)|del)\s+(?:" + _COMIDA + r"|resto|otros?|otras?)\b", re.IGNORECASE)


def _familia(clave: str) -> Optional[str]:
    if clave in _FAMILIA:
        return _FAMILIA[clave]
    if _go()._DESC_SWAP_SUBGROUP.get(clave) == "dulce":
        return "dulce"
    return None


def _esta(clave: str, ingredientes: list, ings_sa: str) -> bool:
    """Presente por el SSOT o por la presencia de P1-DESC-FOOD-HONESTY (unión: ante la duda, se da por presente)."""
    v = _go()._DESC_FOOD_VOCAB[clave]
    return presente(v["lbl"], ingredientes) or bool(re.search(v["pres"], ings_sa))


def _afirmada(texto: str, i: int, j: int) -> bool:
    """¿La mención afirma que el plato lo lleva? Negación sólo dentro de SU cláusula (el «que» la corta)."""
    clausula = re.split(r"[,.;:!?()]|\bque\b", texto[:i])[-1]
    if _NEGACION.search(clausula):
        return False
    return not _OTRA_COMIDA.match(texto[j:])


# --- afirmaciones de ausencia: «sin lácteos» con cottage en la lista es falso
_LACTEOS = {"yogurt", "yogur", "queso", "requeson", "ricotta", "cottage", "mozzarella", "kefir"}
_LECHE_NO_LACTEA = {"coco", "almendra", "almendras", "soya", "soja", "avena", "arroz"}


def _lleva_lacteo(ingredientes: list, *, solo_leche: bool = False) -> bool:
    for ing in ingredientes or []:
        toks = _tokens(str(ing or ""))
        for k, t in enumerate(toks):
            if t == "leche" and not (k + 2 < len(toks) and toks[k + 1] == "de" and toks[k + 2] in _LECHE_NO_LACTEA):
                return True
            if not solo_leche and (_variantes(t) & _LACTEOS):
                return True
    return False


#: Sólo la afirmación SOLA: «sin lácteos fermentados» (el queso fresco no fermenta) o «sin lácteos ni azúcar añadida»
#: dicen otra cosa, y recortarlas dejaba «fermentados.» o «añadida, …» (replay sobre 1 055 fichas).
_AUSENCIA = re.compile(r"\bsin\s+(l[aá]cteos?|leche|yogurt?|quesos?|avena|arroz|huevos?|pollo)\b"
                       r"(?=\s*(?:[,.;:!?]|$)|\s+(?:y|e|que)\s)", re.IGNORECASE)


def _coser(antes: str, despues: str) -> str:
    """Une lo que queda a los dos lados de un tramo quitado sin dejar «y:», «:,» ni «..»."""
    a, d = antes.rstrip(), despues.lstrip()
    if re.search(r"(?:^|\s)(?:y|e)$", a):
        a = _cerrar_enumeracion(re.sub(r"\s*\b(?:y|e)$", "", a).rstrip())   # «cálido, dulce y [sin lácteos],» → «cálido y dulce,»
    elif re.match(r"^(?:y|e)\s", d):
        d = re.sub(r"^(?:y|e)\s+", "", d)                         # «[sin yogur] y fácil de llevar» → «fácil de…»
    elif a and a[-1].isalpha() and d[:1] == ",":
        d = d[1:].lstrip()                                        # «merienda [sin lácteos], crujiente» → «merienda crujiente»
    if (not a or a[-1] in ".!?:;,") and d[:1] in ",;":
        d = d[1:].lstrip()                                        # «:, sin avena» → «: sin avena»
    if a[-1:] in ",;" and d[:1] in ".:!?":
        a = a[:-1]
    if (not a or a[-1] in ".!?") and d[:1] in ".!?":
        d = d[1:].lstrip()                                        # la oración era sólo «Sin lácteos.»
    if (not a or a[-1] in ".!?") and d[:1] in ":;,":
        d = re.sub(r"^[:;,][^.!?]*[.!?]?\s*", "", d)              # «Sin lácteos: pura fruta.» → la oración entera
    if not a or a[-1] in ".!?":
        d = _cap(d)
    return (a + ("" if not d or d[0] in ".,;:!?" else " ") + d).strip()


def _ausencia_falsa(palabra: str, ingredientes: list) -> bool:
    p = _sa(palabra.lower())
    if p.startswith("lacteo"):
        return _lleva_lacteo(ingredientes)
    if p == "leche":
        return _lleva_lacteo(ingredientes, solo_leche=True)
    return presente(p.rstrip("s") if p.endswith("os") or p == "huevos" else p, ingredientes)


def _quitar_ausencias_falsas(desc: str, ingredientes: list) -> tuple[str, int]:
    n = 0
    for _ in range(3):
        m = next((mm for mm in _AUSENCIA.finditer(desc) if _ausencia_falsa(mm.group(1), ingredientes)), None)
        if not m:
            break
        nuevo = _cap(re.sub(r"\s{2,}", " ", _coser(desc[:m.start()], desc[m.end():])).strip())
        if nuevo == desc:
            break
        desc, n = nuevo, n + 1
    return desc, n


# --- el líquido de cocción: «cocida en leche» cuando se cuece en agua
_EN_LECHE = re.compile(r"\b((?:cocid|hervid|hech|preparad|cocinad)[ao]s?\s+en\s+)leche"
                       r"(?:\s+(?:descremada|desnatada|entera|semidescremada|light))?\b(?!\s+de\b)", re.IGNORECASE)


def _liquido_de_coccion(desc: str, ingredientes: list) -> tuple[str, int]:
    if not _EN_LECHE.search(desc) or _lleva_lacteo(ingredientes, solo_leche=True) or not presente("agua", ingredientes):
        return desc, 0
    return _EN_LECHE.sub(lambda m: m.group(1) + "agua", desc), 1


# --- el alimento nombrado que el plato ya no lleva
def _siguiente(resto: str) -> tuple[str, Optional[re.Match]]:
    m = re.match(r"\s+(?:(?:bien|muy)\s+)?(" + _PAL + r")", resto)
    return ((m.group(1).lower() if m else ""), m)


def _cambio_seguro(clave: str, mate: str, resto: str) -> tuple[bool, Optional[str]]:
    """(¿seguro?, adjetivo a re-concordar o None)."""
    go = _go()
    v, tv = go._DESC_FOOD_VOCAB[clave], go._DESC_FOOD_VOCAB[mate]
    nx, m_nx = _siguiente(resto)
    if not nx or nx in go._DESC_NEXT_CONNECTORS and nx not in _CALIFICATIVOS:
        return True, None
    if v["gen"] == tv["gen"] and _familia(clave) in _RELAJADAS:
        return (nx not in _CALIFICATIVOS), None
    if go._DESC_ADJ_OK_RX.match(nx):
        return True, (m_nx.group(1) if (m_nx and v["gen"] != tv["gen"]) else None)
    return False, None


def _alimentos_fantasma(desc: str, ingredientes: list) -> tuple[str, int]:
    go = _go()
    ings_sa = _sa(" | ".join(str(x) for x in ingredientes).lower())
    presentes = {k for k in go._DESC_FOOD_VOCAB if _esta(k, ingredientes, ings_sa)}
    n = 0
    for clave, v in go._DESC_FOOD_VOCAB.items():
        if clave in presentes:
            continue
        low = _sa(desc.lower())
        if len(low) != len(desc):
            return desc, n                   # sin correspondencia de posiciones no se edita
        m = next((mm for mm in re.finditer(r"\b(?:" + v["pat"] + r")\b", low) if _afirmada(low, mm.start(), mm.end())),
                 None)
        if not m:
            continue
        i, j = m.start(), m.end()
        fam = _familia(clave)
        mates = [k for k in presentes if k != clave and fam and _familia(k) == fam
                 and not re.search(r"\b(?:" + go._DESC_FOOD_VOCAB[k]["pat"] + r")\b", low)]
        if len(mates) == 1:
            mate = mates[0]
            ok, adj = _cambio_seguro(clave, mate, desc[j:])
            if ok:
                tv = go._DESC_FOOD_VOCAB[mate]
                lbl = go._desc_pluralize_lbl(tv["lbl"]) if m.group(0).endswith("s") and not tv["lbl"].endswith("s") \
                    else tv["lbl"]
                if desc[i:i + 1].isupper():
                    lbl = _cap(lbl)
                pre = go._desc_swap_gender_articles(desc[:i], tv["gen"]) if tv["gen"] != v["gen"] else desc[:i]
                resto = desc[j:]
                if adj:
                    resto = resto.replace(adj, go._desc_regender_adj(adj, tv["gen"]), 1)
                desc, n = pre + lbl + resto, n + 1
                continue
        # retirada: sólo si la mención CIERRA su cláusula y el tramo no se lleva otro alimento que sí está
        rx_rm = re.compile(r"(?:,\s*)?\b(?:y|e|con|junto\s+a)\s+(?:(?!\b(?:y|e|con)\b)[^,.;:!?¡¿]){0,40}?\b(?:"
                           + v["pat"] + r")\b(?=\s*[.;:!?)]|\s*$)", re.IGNORECASE)
        mr = rx_rm.search(low)
        if mr and not any(re.search(r"\b(?:" + go._DESC_FOOD_VOCAB[k]["pat"] + r")\b", mr.group(0)) for k in presentes):
            n2 = _podar_puntuacion(desc[:mr.start()] + desc[mr.end():])
            if len(n2) >= 25 and not re.search(r"\b(?:" + v["pat"] + r")\b", _sa(n2.lower())):
                desc, n = n2, n + 1
                continue
        logger.info(f"🎭 [P1-PLAN-LOTE-854] ficha nombra «{clave}» y el plato no lo lleva: sin arreglo seguro")
    return desc, n


def _podar_puntuacion(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s)
    return re.sub(r"\s+([,.;:!?])", r"\1", s).strip()


def alinear_con_ingredientes(meal: Any) -> int:
    """Pone la ficha de acuerdo con la lista de ingredientes final. Devuelve cuántos arreglos hizo. Fail-open."""
    if not isinstance(meal, dict):
        return 0
    desc, ings = meal.get("desc"), meal.get("ingredients")
    if not (isinstance(desc, str) and desc.strip() and isinstance(ings, list) and ings):
        return 0
    nuevo = unicodedata.normalize("NFC", desc)
    total = 0
    for paso in (_quitar_ausencias_falsas, _liquido_de_coccion, _alimentos_fantasma):
        try:
            nuevo, k = paso(nuevo, ings)
            total += k
        except Exception as e:                                        # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-854] {paso.__name__} no-op: {e!r}")
    if total and nuevo != desc:
        meal["desc"] = nuevo
        return total
    return 0


# ============================================================ el plan
def aplica_a(plan_data: Any) -> bool:
    """Beta sí; DO (y el plan sin país, que es anterior al sistema de países) sólo con el knob propio."""
    if not activo() or not isinstance(plan_data, dict):
        return False
    raw = plan_data.get("_country")
    if not isinstance(raw, str) or not raw.strip():
        return incluye_do()
    from constants import canonicalize_country
    return canonicalize_country(raw) != "DO" or incluye_do()


def aplicar_plan(plan_data: Any) -> int:
    """Limpia el metalenguaje y alinea cada ficha con sus ingredientes. Devuelve cuántas fichas cambió. Fail-open."""
    if not aplica_a(plan_data):
        return 0
    fichas = meta = verdad = 0
    try:
        for d in plan_data.get("days") or []:
            for m in (d or {}).get("meals") or [] if isinstance(d, dict) else []:
                if not isinstance(m, dict):
                    continue
                antes = (m.get("name"), m.get("desc"))
                try:
                    nombre = limpiar_metalenguaje(m.get("name"), minimo_palabras=2)
                    if nombre != m.get("name"):
                        m["name"] = nombre
                    desc = limpiar_metalenguaje(m.get("desc"))
                    if desc != m.get("desc"):
                        m["desc"] = desc
                        meta += 1
                    verdad += 1 if alinear_con_ingredientes(m) else 0
                except Exception as e:                               # noqa: BLE001
                    logger.debug(f"[P1-PLAN-LOTE-854] comida no-op: {e!r}")
                if (m.get("name"), m.get("desc")) != antes:
                    fichas += 1
    except Exception as e:                                           # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-854] plan no-op: {e!r}")
    if fichas:
        plan_data["_description_truth"] = {"fichas": fichas, "metalenguaje": meta, "ingredientes": verdad}
        logger.info(f"🧾 [P1-PLAN-LOTE-854] {fichas} ficha(s) corregidas: {meta} sin metalenguaje del prompt, "
                    f"{verdad} alineadas con sus ingredientes")
    return fichas


__all__ = ["FAMILIAS_DEL_PROMPT", "limpiar_metalenguaje", "presente", "alinear_con_ingredientes", "aplica_a",
           "aplicar_plan"]
