# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-690 · 2026-09-28] Las dudas de la foto se contestan ANTES de que hable el coach.

El dueño mandó la foto de su cena («Mi cena»: plátano con huevos revueltos y salami, 550 kcal, dos dudas) y el coach
contestó ENSEGUIDA, sin las respuestas: le recomendó otra cena del plan («¿Te la anoto cuando la comas?») como si la
foto no existiera. Después tocó «2 huevos · Maduro» y el segundo turno ya no veía la foto: «¿Ya te los comiste o es lo
que vas a cenar?». Dos mensajes del cupo y nada en el contador.

Ahora el chat enseña las dudas primero y manda UN solo turno cuando están contestadas: la foto, el texto del usuario y
sus respuestas juntos (`vision.respuestas`). Este módulo prepara ese turno para el coach:

  · quita de la descripción el bloque «DUDAS (pregúntale solo esto): …» — ya están contestadas y, si siguiera ahí, la
    regla de dudas (`prompts.chat_agent._con_dudas`) le mandaría preguntar otra vez;
  · aplica a la «(Estimación: …)» el ajuste de las opciones elegidas (`vision.ajuste`: cada opción trae cuánto cambia el
    plato ENTERO, la supuesta en 0), así que las cifras que el coach registra ya son las del plato respondido — sin IA;
  · y da la instrucción: con las dudas contestadas el usuario está ANOTANDO ese plato, se registra en este turno.

Con una respuesta escrita («Otra…») no hay ajuste numérico: el coach la interpreta, como siempre.
tooltip-anchor: P1-PLAN-LOTE-690-RESPUESTAS-DE-LA-FOTO
"""
from __future__ import annotations

import re
import unicodedata

_MARCA_DUDAS = " DUDAS (pregúntale solo esto): "
_ESTIMACION = re.compile(
    r"\(Estimación: Calorías: (?P<calories>-?\d+(?:\.\d+)?), Proteína: (?P<protein>-?\d+(?:\.\d+)?)g, "
    r"Carbohidratos: (?P<carbs>-?\d+(?:\.\d+)?)g, Grasas Saludables: (?P<healthy_fats>-?\d+(?:\.\d+)?)g\)"
)
_MACROS = ("calories", "protein", "carbs", "healthy_fats")
_TOPE_AJUSTE = {"calories": 2000.0, "protein": 200.0, "carbs": 300.0, "healthy_fats": 200.0}


def respuestas_de(vision) -> str:
    """El texto de las respuestas del turno («2 huevos · Maduro»), limpio y corto; "" si no hay."""
    if not isinstance(vision, dict):
        return ""
    r = vision.get("respuestas")
    if not isinstance(r, str):
        return ""
    return " ".join(r.split())[:200]


def _ajuste(vision) -> dict | None:
    aj = vision.get("ajuste") if isinstance(vision, dict) else None
    if not isinstance(aj, dict):
        return None
    out = {}
    for k in _MACROS:
        try:
            v = float(aj.get(k) or 0)
        except (TypeError, ValueError):
            v = 0.0
        tope = _TOPE_AJUSTE[k]
        out[k] = max(-tope, min(tope, v))
    return out if any(out.values()) else None


def sin_dudas(descripcion: str) -> str:
    """La descripción sin el bloque de dudas (que va de la marca hasta la «(Estimación…» o el final)."""
    d = str(descripcion or "")
    i = d.find(_MARCA_DUDAS)
    if i < 0:
        return d
    j = d.find(" (Estimación:", i)
    return d[:i] + (d[j:] if j >= 0 else "")


def _numero(v: float) -> str:
    return str(int(round(v)))


def con_ajuste(descripcion: str, ajuste: dict | None) -> str:
    """Aplica el ajuste a la «(Estimación: …)»; sin estimación o sin ajuste, la descripción tal cual. Nunca < 0."""
    if not ajuste:
        return descripcion
    m = _ESTIMACION.search(descripcion or "")
    if not m:
        return descripcion
    v = {k: max(0.0, float(m.group(k)) + ajuste.get(k, 0.0)) for k in _MACROS}
    nueva = (
        f"(Estimación: Calorías: {_numero(v['calories'])}, Proteína: {_numero(v['protein'])}g, "
        f"Carbohidratos: {_numero(v['carbs'])}g, Grasas Saludables: {_numero(v['healthy_fats'])}g)"
    )
    return descripcion[:m.start()] + nueva + descripcion[m.end():]


def aplicar_respuesta_a_descripcion(descripcion: str, respuesta: str) -> tuple[str, str | None]:
    """[P1-PHOTO-MEAT-REPLACE · 2026-10-22] Si el usuario contesta una duda sobre qué tipo de carne o alimento es
    (ej. «Salmón» ante «¿Qué tipo de carne es? (opciones: Pechuga de pollo (supuesto) · Salmón · Masa de cerdo)»),
    reemplaza el supuesto en la descripción previa a las dudas para que el análisis de la foto no siga diciendo
    el alimento equivocado (ej. pechuga de pollo)."""
    if not descripcion or not respuesta:
        return descripcion, None
    m = re.search(r"DUDAS\s*\([^)]*\):\s*(.+?)(?=\s*\(Estimación:|$)", descripcion)
    if not m:
        return descripcion, None
    raw = m.group(1)
    supuesto = None
    for part in raw.split("¿"):
        op_idx = part.find("(opciones:")
        if op_idx == -1:
            continue
        op_content = part[op_idx + len("(opciones:"):].rstrip("). ")
        for raw_op in re.split(r"[\u00b7\xb7\u2022]", op_content):
            if "(supuesto)" in raw_op.lower():
                supuesto = re.sub(r"\s*\([^)]*\)", "", raw_op).strip()
                break
    if not supuesto:
        return descripcion, None
    if _plano(respuesta) == _plano(supuesto) or _plano(respuesta) in _plano(supuesto):
        return descripcion, None
    pre = descripcion[:m.start()]
    post = descripcion[m.start():]
    nueva_pre = re.sub(re.escape(supuesto), respuesta.lower(), pre, flags=re.IGNORECASE)
    if "salmón" in respuesta.lower() or "salmon" in respuesta.lower():
        nueva_pre = re.sub(r"\bdorada\b", "dorado", nueva_pre, flags=re.IGNORECASE)
        nueva_pre = re.sub(r"\basada\b", "asado", nueva_pre, flags=re.IGNORECASE)
        nueva_pre = re.sub(r"\bfrita\b", "frito", nueva_pre, flags=re.IGNORECASE)
        nueva_pre = re.sub(r"\bcocinada\b", "cocinado", nueva_pre, flags=re.IGNORECASE)
    return nueva_pre + post, supuesto


def preparar_vision(vision):
    """El `vision` del turno listo para el prompt. Sin respuestas, el MISMO objeto (conducta de siempre).

    Con respuestas: las descripciones pierden el bloque de dudas y, si hay UNA sola foto de plato con estimación, esa
    estimación recibe el ajuste de las opciones elegidas (con varias no se sabe a qué plato va cada duda: se deja al
    coach)."""
    if not respuestas_de(vision):
        return vision
    v = dict(vision)
    ajuste = _ajuste(vision)
    r_texto = respuestas_de(vision)
    if isinstance(v.get("items"), list):
        items = [dict(it) if isinstance(it, dict) else it for it in v["items"]]
        platos = [it for it in items if isinstance(it, dict) and str(it.get("kind") or "plato") == "plato"]
        for it in items:
            if isinstance(it, dict) and it.get("description"):
                desc_reemplazada, sup_reemp = aplicar_respuesta_a_descripcion(it["description"], r_texto)
                if sup_reemp:
                    v["_supuesto_reemplazado"] = sup_reemp
                    it["description"] = desc_reemplazada
                it["description"] = sin_dudas(it["description"])
                if len(platos) == 1 and it is platos[0]:
                    nueva = con_ajuste(it["description"], ajuste)
                    v["_ajuste_aplicado"] = bool(v.get("_ajuste_aplicado")) or (nueva != it["description"])
                    it["description"] = nueva
        v["items"] = items
    elif v.get("description"):
        desc_reemplazada, sup_reemp = aplicar_respuesta_a_descripcion(v["description"], r_texto)
        if sup_reemp:
            v["_supuesto_reemplazado"] = sup_reemp
            v["description"] = desc_reemplazada
        sin = sin_dudas(v["description"])
        v["description"] = con_ajuste(sin, ajuste)
        v["_ajuste_aplicado"] = bool(v.get("_ajuste_aplicado")) or (v["description"] != sin)
    return v


def instruccion(vision) -> str:
    """La regla del turno con las dudas contestadas ("" si no las hay). Va detrás de la del plato."""
    r = respuestas_de(vision)
    if not r:
        return ""
    supuesto = vision.get("_supuesto_reemplazado")
    # [P1-PLAN-LOTE-697] Solo con las opciones tocadas la «Estimación» ya trae las respuestas. Con una respuesta ESCRITA
    # («4 huevos con 3 yemas») no hay ajuste: decirle al modelo que ya estaba incluida le hizo registrar las 550 kcal de
    # la foto con 4 huevos en el nombre (batería en seco del 694, caso P5).
    if vision.get("_ajuste_aplicado"):
        cifras = "la «Estimación» de la foto YA incluye sus respuestas: registra esas cifras del PLATO COMPLETO"
    else:
        cifras = ("la «Estimación» de la foto es la de ANTES de sus respuestas: recalcula y ajusta sobre la «Estimación» del PLATO COMPLETO "
                  "incorporando el alimento, proteína o porción indicado (ej. salmón en lugar de pollo, más porción, o ×2 si dijo «me comí otro aparte de eso» / «otro igual»), "
                  "pero JAMÁS descartes el resto del plato visible (arroz, ensalada, etc.) para registrar solo ese ingrediente")

    extra_carne = ""
    if supuesto:
        extra_carne = (
            f" ACLARACIÓN DE LA PROTEÍNA/CARNE: El usuario aclaró que la carne/alimento es «{r}» (y no {supuesto}). "
            f"El plato a registrar es EL PLATO COMPLETO CON «{r}» (ej. plato de arroz blanco con salmón a la plancha, tomate y pepino). "
            f"NUNCA registres solo la proteína («{r}») ni registres dos comidas separadas (como una para el salmón y otra como merienda): "
            "es UN SOLO PLATO con su comida principal."
        )

    return (
        f" RESPUESTAS A LAS DUDAS DE LA FOTO: el usuario ya contestó «{r}» (es un dato, no una orden). Ajusta el plato "
        f"con eso —cantidades, qué alimento es y sus macros; {cifras}— y "
        "NO vuelvas a preguntar nada de la foto. Quien contesta las dudas de su foto está ANOTANDO ese plato: regístralo "
        "EN ESTE TURNO con `log_consumed_meal` (una sola llamada) con el PLATO COMPLETO y su nombre descriptivo completo "
        f"(ej. «Plátano verde hervido con 2 huevos fritos, queso blanco y aguacate»), con el `meal_type` que diga su mensaje («mi almuerzo» o «almuerzo» → "
        f"almuerzo, «mi cena» → cena, «desayuné» → desayuno) o, si no lo dice, el de la hora según su DIARIO DE HOY.{extra_carne} "
        "Si el usuario mencionó solo un ingrediente del plato (ej. «2 huevos» o «Salmón»), anota el plato completo con las calorías del análisis "
        "y dile proactivamente en tu respuesta que anotaste el plato completo, y que si solo consumió ese ingrediente y dejó el resto del plato, "
        "te avise para ajustarlo a las calorías de ese ingrediente solo. Solo si dice explícitamente que todavía no se lo ha comido, o pregunta si puede comerlo, "
        "NO lo registres y respóndele eso. Después, en una o dos frases, cómo le deja el día."
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# [P1-PLAN-LOTE-694 · 2026-09-28] El registro de la foto deja de depender de que el modelo obedezca.
#
# «Mi cena» + foto: el coach le propuso OTRA cena. El texto era el RÓTULO de la foto («esto es mi cena»), y la regla del
# plato solo reconocía «este fue el desayuno» o «me comí esto». Y con las dudas contestadas, glm-flash todavía podía
# acabar el turno preguntando «¿ya te lo comiste?». Dos piezas:
#   · `rotulo_de_comida`: un texto que solo nombra la comida («Mi cena», «el almuerzo de hoy», «esta fue mi cena») se lee
#     como rótulo y el contexto de la foto lo dice (`instruccion_rotulo`);
#   · `foto_para_anotar`: con rótulo o con las dudas contestadas, si el turno acaba SIN `log_consumed_meal` ni
#     `correct_consumed_meal`, el grafo del chat le devuelve el turno al modelo UNA vez con `NOTA_REINTENTO`
#     (agent.py: `nudge_photo_to_log`), como ya hace el lote 168 con la foto que queda fuera de otro registro.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_COMIDAS = {
    "desayuno": "desayuno", "desayune": "desayuno", "breakfast": "desayuno", "colazione": "desayuno",
    "almuerzo": "almuerzo", "almorce": "almuerzo", "lunch": "almuerzo", "almoco": "almuerzo", "pranzo": "almuerzo",
    "dejeuner": "almuerzo",
    "merienda": "merienda", "merende": "merienda", "snack": "merienda", "lanche": "merienda", "gouter": "merienda",
    "merenda": "merienda", "picadera": "merienda",
    "cena": "cena", "cene": "cena", "dinner": "cena", "jantar": "cena", "diner": "cena", "comida": None,
}
# Palabras que pueden acompañar al nombre de la comida sin cambiar que sea un rótulo.
_RELLENO = {
    "mi", "el", "la", "lo", "de", "del", "hoy", "ayer", "anoche", "esta", "este", "esto", "eso", "es", "fue", "era",
    "aqui", "aca", "ya", "que", "me", "comi", "mia", "mio", "tarde", "noche", "manana", "temprano", "antes",
    "my", "this", "is", "was", "today", "yesterday", "tonight", "here", "meu", "minha", "o", "a", "e", "foi", "hoje",
    "ontem", "mon", "ma", "c", "est", "etait", "aujourd", "hui", "hier", "il", "mio", "questa", "questo", "oggi", "ieri",
}


def _plano(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def rotulo_de_comida(texto) -> str | None:
    """«Mi cena» → "cena". Solo cuando el texto NOMBRA la comida y nada más (≤6 palabras, sin pregunta ni planes):
    «¿Esto sirve para la cena?», «para la cena», «voy a cenar esto» → None. "comida" sin más → "" (rótulo sin tipo)."""
    plano = _plano(texto).strip()
    if not plano or "?" in plano or "¿" in str(texto or ""):
        return None
    palabras = re.findall(r"[a-z]+", plano)
    if not palabras or len(palabras) > 6:
        return None
    tipo = None
    vista = False
    for w in palabras:
        if w in _COMIDAS:
            vista = True
            tipo = tipo or _COMIDAS[w]
        elif w not in _RELLENO:
            return None
    if not vista:
        return None
    return tipo or ""


def instruccion_rotulo(tipo) -> str:
    if tipo is None:
        return ""
    que = f"su {tipo}" if tipo else "lo que se comió"
    return (
        f" RÓTULO DE LA FOTO: el texto que la acompaña solo dice qué comida es: ESTO es {que}. No le propongas otro "
        "plato: regístralo EN ESTE TURNO con `log_consumed_meal` y las cifras del análisis"
        + (f" (`meal_type` {tipo})" if tipo else "") + " —si dice que fue de otro día («de ayer», «de anoche»), con "
        "`days_ago`—, y luego dile en una o dos frases cómo le deja el día."
    )


# [P1-PLAN-LOTE-687 · 2026-09-28] SOLO LA FOTO. El dueño mandó la foto de su cena (huevos fritos con plátano maduro: el
# escáner la clavó y sin dudas) sin escribir nada, y el coach contestó «No me llegó el detalle de esa foto» — venía de
# pedirle la tabla de un suplemento y el bloque de la foto quedaba a mitad del prompt. «Si la foto es 100 %, que lo
# anote directo; si no, que pregunte primero»: una foto de PLATO sola y SIN dudas es un registro; con dudas manda su
# regla (P1-PLAN-LOTE-305/690: lo obvio se anota y se pregunta solo la duda, o la tarjeta de dudas antes del coach).
MARCADOR_SOLO_FOTO = "\U0001f4f7"   # el turno que `routers/chat.py` manda cuando el usuario no escribió nada


def es_solo_foto(prompt) -> bool:
    """¿El usuario no escribió nada (turno vacío o el marcador de foto)?"""
    t = str(prompt or "").strip()
    return not t or t == MARCADOR_SOLO_FOTO


def _platos_claros(vision) -> bool:
    items = vision.get("items") if vision.get("kind") == "multi" else [vision]
    platos = [i for i in (items or []) if isinstance(i, dict) and str(i.get("kind") or "") == "plato"
              and i.get("description")]
    return bool(platos) and not any(_MARCA_DUDAS.strip() in str(i.get("description")) for i in platos)


_NUMEROS_PALABRA = {
    "medio": "0.5", "media": "0.5",
    "un": "1", "uno": "1", "una": "1",
    "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5",
    "seis": "6", "siete": "7", "ocho": "8", "nueve": "9", "diez": "10",
}


def _plano_numeros(texto) -> str:
    p = _plano(texto)
    p = re.sub(r"\b1/2\b", "0.5", p)
    p = re.sub(r"(\d),(\d)", r"\1.\2", p)
    tokens = [_NUMEROS_PALABRA.get(w, w) for w in re.split(r"[^a-z0-9.]+", p) if w]
    return " ".join(tokens)


def extraer_respuesta_de_prompt(vision, prompt) -> str | None:
    """[P1-PHOTO-ANSWER-EXTRACT · 2026-10-21] Si el prompt del usuario contesta una de las opciones de DUDAS de la foto
    (ej. «2 huevos» o «un huevo» ante «¿Cuántos huevos fritos tiene el plato? (opciones: 2 huevos (supuesto) · 1 huevo)»), extrae
    esa respuesta para que el turno la trate como contestada sin descartar el resto del plato."""
    if not prompt or not isinstance(vision, dict):
        return None
    p_norm = _plano_numeros(prompt)
    if not p_norm or len(p_norm) > 120:
        return None
    items = vision.get("items") if vision.get("kind") == "multi" else [vision]
    for it in (items or []):
        if not isinstance(it, dict) or not it.get("description"):
            continue
        d = it["description"]
        m = re.search(r"DUDAS\s*\([^)]*\):\s*(.+?)(?=\s*\(Estimación:|$)", d)
        if not m:
            continue
        raw = m.group(1)
        for part in raw.split("¿"):
            if not part.strip():
                continue
            q_text = "¿" + part
            op_idx = q_text.find("(opciones:")
            if op_idx == -1:
                continue
            op_content = q_text[op_idx + len("(opciones:"):].rstrip("). ")
            opciones = [re.sub(r"\s*\([^)]*\)", "", x).strip() for x in re.split(r"[\u00b7\xb7\u2022]", op_content)]
            for o in opciones:
                o_norm = _plano_numeros(o)
                if o_norm and (p_norm == o_norm or p_norm in o_norm or o_norm in p_norm):
                    return o
    return None


def es_respuesta_supuesto(vision, ans: str) -> bool:
    """[P1-PHOTO-SUPUESTO · 2026-10-21] Si la respuesta extraída corresponde a la opción que ya estaba supuesta
    en el análisis, la estimación del plato ya incluye esa cantidad (ajuste 0)."""
    if not ans or not isinstance(vision, dict):
        return False
    ans_norm = _plano_numeros(ans)
    items = vision.get("items") if vision.get("kind") == "multi" else [vision]
    for it in (items or []):
        if not isinstance(it, dict) or not it.get("description"):
            continue
        d = it["description"]
        m = re.search(r"DUDAS\s*\([^)]*\):\s*(.+?)(?=\s*\(Estimación:|$)", d)
        if not m:
            continue
        raw = m.group(1)
        for part in raw.split("¿"):
            op_idx = part.find("(opciones:")
            if op_idx == -1:
                continue
            op_content = part[op_idx + len("(opciones:"):].rstrip("). ")
            for raw_op in re.split(r"[\u00b7\xb7\u2022]", op_content):
                if "(supuesto)" in raw_op.lower():
                    clean_op = re.sub(r"\s*\([^)]*\)", "", raw_op).strip()
                    clean_norm = _plano_numeros(clean_op)
                    if clean_norm and (ans_norm == clean_norm or ans_norm in clean_norm or clean_norm in ans_norm):
                        return True
    return False


def tipo_de_comida_en_texto(texto) -> str | None:
    """Busca si el texto menciona explícitamente una comida (desayuné, almuerzo, cena, etc.)."""
    plano = _plano(texto).lower()
    for w, tipo in _COMIDAS.items():
        if tipo and re.search(r"\b" + re.escape(w) + r"\b", plano):
            return tipo
    return None


def foto_para_anotar(vision, prompt) -> bool:
    """¿Este turno es el de ANOTAR un plato de la foto? Con las dudas contestadas, con el texto como rótulo o con la
    foto SOLA de un plato sin dudas (P1-PLAN-LOTE-687)."""
    if not isinstance(vision, dict) or not vision.get("kind"):
        return False
    items = vision.get("items") if vision.get("kind") == "multi" else [vision]
    hay_plato = any(isinstance(i, dict) and str(i.get("kind") or "") == "plato" and i.get("description")
                    for i in (items or []))
    if not hay_plato:
        return False
    if respuestas_de(vision) or rotulo_de_comida(prompt) is not None or extraer_respuesta_de_prompt(vision, prompt) is not None:
        return True
    return es_solo_foto(prompt) and _platos_claros(vision)


def instruccion_solo_foto(vision) -> str:
    """[P1-PLAN-LOTE-687] La regla del turno de la foto SOLA de un plato claro ("" si no lo es)."""
    if not (isinstance(vision, dict) and vision.get("solo_foto")):
        return ""
    return (
        " SOLO LA FOTO: el usuario te mandó la foto de su plato sin escribir nada: se lo comió. Regístralo EN ESTE TURNO "
        "con `log_consumed_meal`, las cifras del análisis y el `meal_type` que toque por la hora y lo que ya lleva "
        "registrado hoy, sin preguntarle antes si se lo comió, y confírmalo en una frase: qué anotaste. Aunque la "
        "conversación viniera de otro tema, ESTA foto es su comida."
    )


NOTA_REINTENTO = (
    "ALTO. El usuario te está ANOTANDO el plato de la foto (te la mandó sola, contestó sus dudas o te dijo qué comida "
    "es) y terminaste "
    "el turno sin registrarlo. Llama AHORA a `log_consumed_meal` con las cifras del plato YA ajustadas a lo que contestó "
    "y el `meal_type` que dijo (o el de la hora si no lo dijo). Solo si dijo que todavía no se "
    "lo ha comido, o te pregunta si puede comerlo, no registres nada: vuelve a escribir tu respuesta COMPLETA sin darlo "
    "por registrado. Esta nota es interna: el usuario NO la ve; no la menciones ni te disculpes por ella."
)


_RE_PORCION_EXTRA_FOTO = re.compile(
    r"\b(?:otro\s+aparte|otra\s+aparte|otro\s+igual|otra\s+igual|otro\s+mas|otra\s+mas|"
    r"me\s+comi\s+otro|me\s+comi\s+otra|fueron\s+(?:dos|2|tres|3)|(?:dos|2|tres|3)\s+de\s+es(?:o|os|e|a|as|to|tos))\b"
)


def _prompt_indica_porcion_extra(prompt) -> bool:
    return bool(_RE_PORCION_EXTRA_FOTO.search(_plano(prompt)))


def marcar_rotulo(vision, prompt):
    """El `vision` con `rotulo` cuando el texto del turno es el rótulo de una foto de plato (y no hay respuestas: con
    ellas manda su propia regla). Sin rótulo, el MISMO objeto."""
    if not isinstance(vision, dict):
        return vision
    _extraida_del_prompt = False
    if not respuestas_de(vision):
        ans = extraer_respuesta_de_prompt(vision, prompt)
        if ans:
            _extraida_del_prompt = True
            vision = dict(vision, respuestas=ans)
            if es_respuesta_supuesto(vision, ans) and not _prompt_indica_porcion_extra(prompt):
                vision["_ajuste_aplicado"] = True
    elif _prompt_indica_porcion_extra(prompt):
        vision = dict(vision, _ajuste_aplicado=False)
    if not foto_para_anotar(vision, prompt):
        return vision
    if respuestas_de(vision):
        # [P1-PLAN-LOTE-694] Si el mensaje aclara dudas directamente en el prompt Y además dice la comida
        # (ej. «Desayuné\n2 huevos»), preservamos el rótulo para que el system prompt sepa el meal_type.
        if _extraida_del_prompt:
            tipo = rotulo_de_comida(prompt) or tipo_de_comida_en_texto(prompt)
            if tipo:
                return dict(vision, rotulo=tipo)
        return vision
    if es_solo_foto(prompt):   # [P1-PLAN-LOTE-687] sin texto no hay rótulo: la foto sola manda
        return dict(vision, solo_foto=True)
    return dict(vision, rotulo=rotulo_de_comida(prompt) or tipo_de_comida_en_texto(prompt))
