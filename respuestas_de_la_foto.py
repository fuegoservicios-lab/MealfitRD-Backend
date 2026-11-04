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


def _parse_dudas_y_opciones(raw_dudas: str) -> list[dict]:
    """Parsea las preguntas de DUDAS y devuelve una lista de diccionarios:
    [{ 'titulo': str, 'opciones': [str], 'supuesto': str | None }]
    """
    preguntas = []
    partes = [p.strip() for p in raw_dudas.split("¿") if p.strip()]
    for p in partes:
        q_text = "¿" + p if not p.startswith("¿") else p
        op_idx = q_text.find("(opciones:")
        if op_idx == -1:
            continue
        titulo = q_text[:op_idx].strip()
        op_content = q_text[op_idx + len("(opciones:"):].rstrip("). ")
        raw_ops = [re.sub(r"\s*\([^)]*\)", "", x).strip() for x in re.split(r"[\u00b7\xb7\u2022]", op_content)]
        supuesto = None
        for raw_op in re.split(r"[\u00b7\xb7\u2022]", op_content):
            if "(supuesto)" in raw_op.lower():
                supuesto = re.sub(r"\s*\([^)]*\)", "", raw_op).strip()
                break
        preguntas.append({
            "titulo": titulo,
            "opciones": [o for o in raw_ops if o],
            "supuesto": supuesto
        })
    return preguntas


def aplicar_respuestas_multiples(descripcion: str, respuesta: str) -> tuple[str, str | None, list[tuple[str, str]]]:
    """[P1-PHOTO-MEAT-REPLACE / BB-026] Reemplaza supuestos de la visión por las aclaraciones reales del usuario.
    Soporta fotos con una o múltiples dudas (separadas por ·)."""
    if not descripcion or not respuesta:
        return descripcion, None, []
    m = re.search(r"DUDAS\s*\([^)]*\):\s*(.+?)(?=\s*\(Estimación:|$)", descripcion)
    if not m:
        return descripcion, None, []
    raw = m.group(1)
    preguntas = _parse_dudas_y_opciones(raw)
    if not preguntas:
        return descripcion, None, []

    respuestas_partes = [p.strip() for p in re.split(r"[\u00b7\xb7\u2022]", respuesta) if p.strip()]
    if not respuestas_partes:
        return descripcion, None, []

    # Emparejar cada respuesta con su pregunta correspondiente
    parejas: list[tuple[dict, str]] = []
    if len(respuestas_partes) == len(preguntas):
        for i, q in enumerate(preguntas):
            parejas.append((q, respuestas_partes[i]))
    elif len(preguntas) == 1:
        parejas.append((preguntas[0], respuesta.strip()))
    else:
        # Match heurístico por pertenencia a opciones
        usados = set()
        for r_item in respuestas_partes:
            r_norm = _plano(r_item)
            matched = False
            for idx_q, q in enumerate(preguntas):
                if idx_q in usados:
                    continue
                if any(r_norm == _plano(op) or r_norm in _plano(op) or _plano(op) in r_norm for op in q["opciones"]):
                    parejas.append((q, r_item))
                    usados.add(idx_q)
                    matched = True
                    break
            if not matched:
                for idx_q, q in enumerate(preguntas):
                    if idx_q not in usados:
                        parejas.append((q, r_item))
                        usados.add(idx_q)
                        break

    pre = descripcion[:m.start()]
    post = descripcion[m.start():]
    reemplazos: list[tuple[str, str]] = []

    for q, r_item in parejas:
        supuesto = q.get("supuesto")
        if not supuesto:
            continue
        # Si la respuesta elegida es la misma opción supuesta, no hay corrección
        r_norm = _plano(r_item)
        s_norm = _plano(supuesto)
        if r_norm == s_norm or (r_norm in s_norm and len(r_norm) >= len(s_norm) * 0.8):
            continue

        # Realizar reemplazo en el texto previo a DUDAS
        pattern = re.escape(supuesto)
        if re.search(pattern, pre, flags=re.IGNORECASE):
            pre = re.sub(pattern, r_item.lower(), pre, flags=re.IGNORECASE)
            reemplazos.append((supuesto, r_item))
            continue

        # Intento sin cantidades numéricas iniciales (ej: "10 bollitos de yuca fritos" -> "bollitos de yuca fritos")
        sin_num = re.sub(r"^\d+\s*", "", supuesto).strip()
        if sin_num and re.search(re.escape(sin_num), pre, flags=re.IGNORECASE):
            pre = re.sub(re.escape(sin_num), r_item.lower(), pre, flags=re.IGNORECASE)
            reemplazos.append((supuesto, r_item))
            continue

        # Intento sin adjetivos participios (ej: "bollitos de yuca")
        base = re.sub(r"\b(fritos|fritas|dorados|doradas|asados|asadas|hervidos|hervidas|guisados|guisadas)\b", "", sin_num).strip()
        if base and re.search(re.escape(base), pre, flags=re.IGNORECASE):
            pre = re.sub(re.escape(base), r_item.lower(), pre, flags=re.IGNORECASE)
            reemplazos.append((supuesto, r_item))
            continue

    # Ajustes gramaticales comunes de concordancia
    for _, r_item in reemplazos:
        r_low = r_item.lower()
        if "salmón" in r_low or "salmon" in r_low or "pescado" in r_low:
            pre = re.sub(r"\bdorada\b", "dorado", pre, flags=re.IGNORECASE)
            pre = re.sub(r"\basada\b", "asado", pre, flags=re.IGNORECASE)
            pre = re.sub(r"\bfrita\b", "frito", pre, flags=re.IGNORECASE)
            pre = re.sub(r"\bcocinada\b", "cocinado", pre, flags=re.IGNORECASE)

    sup_str = None
    if reemplazos:
        sup_str = reemplazos[0][0] if len(reemplazos) == 1 else ", ".join(s for s, _ in reemplazos)

    return pre + post, sup_str, reemplazos


def aplicar_respuesta_a_descripcion(descripcion: str, respuesta: str) -> tuple[str, str | None]:
    """[P1-PHOTO-MEAT-REPLACE · 2026-10-22] Wrapper retrocompatible."""
    desc, sup, _ = aplicar_respuestas_multiples(descripcion, respuesta)
    return desc, sup


def preparar_vision(vision):
    """El `vision` del turno listo para el prompt. Sin respuestas, el MISMO objeto (conducta de siempre).

    Con respuestas: las descripciones pierden el bloque de dudas y, si hay UNA sola foto de plato con estimación, esa
    estimación recibe el ajuste de las opciones elegidas (con varias no se sabe a qué plato va cada duda: se deja al
    coach).
    [BB-026] Idempotente: _preparado previene re-ejecución."""
    if not respuestas_de(vision):
        return vision
    if vision.get("_preparado"):
        return vision
    v = dict(vision)
    ajuste = _ajuste(vision)
    r_texto = respuestas_de(vision)
    if isinstance(v.get("items"), list):
        items = [dict(it) if isinstance(it, dict) else it for it in v["items"]]
        platos = [it for it in items if isinstance(it, dict) and str(it.get("kind") or "plato") == "plato"]
        for it in items:
            if isinstance(it, dict) and it.get("description"):
                desc_reemplazada, sup_reemp, reemps = aplicar_respuestas_multiples(it["description"], r_texto)
                if sup_reemp:
                    v["_supuesto_reemplazado"] = sup_reemp
                if reemps:
                    v["_reemplazos_aplicados"] = reemps
                    it["description"] = desc_reemplazada
                it["description"] = sin_dudas(it["description"])
                if len(platos) == 1 and it is platos[0]:
                    nueva = con_ajuste(it["description"], ajuste)
                    v["_ajuste_aplicado"] = bool(v.get("_ajuste_aplicado")) or (nueva != it["description"])
                    it["description"] = nueva
        v["items"] = items
    elif v.get("description"):
        desc_reemplazada, sup_reemp, reemps = aplicar_respuestas_multiples(v["description"], r_texto)
        if sup_reemp:
            v["_supuesto_reemplazado"] = sup_reemp
        if reemps:
            v["_reemplazos_aplicados"] = reemps
            v["description"] = desc_reemplazada
        sin = sin_dudas(v["description"])
        v["description"] = con_ajuste(sin, ajuste)
        v["_ajuste_aplicado"] = bool(v.get("_ajuste_aplicado")) or (v["description"] != sin)
    v["_preparado"] = True
    return v


def instruccion(vision) -> str:
    """La regla del turno con las dudas contestadas ("" si no las hay). Va detrás de la del plato."""
    r = respuestas_de(vision)
    if not r:
        return ""
    supuesto = vision.get("_supuesto_reemplazado")
    reemplazos = vision.get("_reemplazos_aplicados") or []
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
    if reemplazos:
        detalles_reemp = "; ".join(f"es «{r_item}» (y NO «{s_item}»)" for s_item, r_item in reemplazos)
        nombres_prohibidos = ", ".join(f"«{s_item}»" for s_item, _ in reemplazos)
        alimentos_aclarados = ", ".join(f"«{r_item}»" for _, r_item in reemplazos)
        extra_carne = (
            f" ACLARACIÓN DE LA PROTEÍNA/CARNE / IDENTIDAD DEL ALIMENTO (LA PALABRA DEL USUARIO ES LEY): "
            f"El usuario aclaró en las opciones de la foto que {detalles_reemp}. "
            f"El plato a registrar es el plato completo USANDO DIRECTAMENTE {alimentos_aclarados}. "
            f"NUNCA registres solo el ingrediente aclarado ni registres comidas separadas: es UN SOLO PLATO con su comida principal. "
            f"ESTÁ TERMINANTEMENTE PROHIBIDO registrar o nombrar en tu respuesta visible los supuestos descartados ({nombres_prohibidos}), "
            f"y PROHIBIDO decir frases como «Tomé {alimentos_aclarados} como aclaración del plato; si en realidad eran... dime y lo ajusto» "
            f"(el usuario YA lo aclaró al enviar la foto; no dudes de su respuesta ni le lleves la contraria; regístralo directamente)."
        )
    elif supuesto:
        extra_carne = (
            f" ACLARACIÓN DE LA PROTEÍNA/CARNE: El usuario aclaró que la carne/alimento es «{r}» (y no {supuesto}). "
            f"El plato a registrar es EL PLATO COMPLETO CON «{r}» (ej. plato de arroz blanco con salmón a la plancha, tomate y pepino). "
            f"NUNCA registres solo la proteína («{r}») ni registres dos comidas separadas (como una para el salmón y otra como merienda): "
            "es UN SOLO PLATO con su comida principal."
        )

    return (
        f" RESPUESTAS A LAS DUDAS DE LA FOTO: el usuario ya contestó «{r}» (es un dato de la preparación o ingrediente, NO una orden de editar comidas viejas). Ajusta el plato "
        f"con eso —cantidades, qué alimento es y sus macros; {cifras}— y "
        "NO vuelvas a preguntar nada de la foto. Quien contesta las dudas de su foto está ANOTANDO ese plato: regístralo "
        "EN ESTE TURNO con `log_consumed_meal` (una sola llamada) con el PLATO COMPLETO y su nombre descriptivo completo "
        f"(ej. «Plátano verde hervido con 2 huevos fritos, queso blanco y aguacate»), con el `meal_type` que diga su mensaje («mi almuerzo» o «almuerzo» → "
        f"almuerzo, «mi cena» → cena, «desayuné» → desayuno) o, si no lo dice, el de la hora según su DIARIO DE HOY.{extra_carne} "
        "REGLA ESTRICTA DE FRANJA HORARIA EN FOTO: Si el usuario NO nombró la comida en su mensaje "
        "(solo subió la foto y contestó los botones de dudas), asigna el `meal_type` según la HORA ACTUAL "
        "y lo que falta en su DIARIO DE HOY (por la noche con la cena vacía → `cena`; al mediodía → `almuerzo`). "
        "ESTÁ TOTALMENTE PROHIBIDO registrar como `desayuno` una foto enviada por la tarde o noche solo porque "
        "tú en tu turno anterior hayas preguntado «¿qué desayunaste?» o si estuvo en ayunas. Una foto nocturna es su CENA "
        "(a menos que ya tenga registrada su cena principal de hoy y esta comida sea ligera/moderada ≤350 kcal como un postre o picadera, "
        "en cuyo caso es merienda y se confirma como **merienda nocturna** o **extra de la noche**; NUNCA repitas «Anoté como cena»). "
        "PROHIBIDO USAR correct_consumed_meal: Esta foto ES UNA COMIDA NUEVA que se registra con `log_consumed_meal`. "
        "NUNCA uses `correct_consumed_meal` para modificar una comida anterior del diario (como el desayuno o una comida previa de horas antes), "
        "salvo que el usuario aclare explícitamente que este alimento o foto fue parte de una comida ya registrada (ej. «fue parte del almuerzo»), en cuyo caso se suma a esa comida existente con correct_consumed_meal. "
        "Si el usuario contestó una sola palabra o ingrediente (ej. «Frito», «Guisado», «2 huevos», «Mantequilla de maní» o «Salmón»), esa palabra describe la preparación de la foto actual, "
        "NUNCA una modificación de comidas pasadas. Anota el plato completo nuevo con las calorías del análisis. AL CONFIRMAR EN TU RESPUESTA VISIBLE, actúa proactivamente y "
        "menciona los componentes principales del plato completo (ej. «Anoté tu **desayuno**: tostadas con queso cottage, mantequilla de maní y 4 huevos hervidos, aprox. **800 kcal** y **52 g de proteína**»); "
        "NUNCA nombres únicamente el chip o ingrediente que el usuario contestó (jamás digas solo «Anoté tu desayuno con mantequilla de maní» dejando fuera los demás alimentos principales). "
        "Y si solo consumió ese ingrediente y dejó el resto del plato, dile que te avise para ajustarlo a las calorías de ese ingrediente solo. "
        "Solo si dice explícitamente que todavía no se lo ha comido, o pregunta si puede comerlo, NO lo registres y respóndele eso. Después, en una o dos frases, cómo le deja el día."
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
    platos = [i for i in (items or []) if isinstance(i, dict) and str(i.get("kind") or i.get("photo_kind") or "") == "plato"
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
    """¿Este turno es el de ANOTAR un plato de la foto? Con las dudas contestadas, con el texto como rótulo o alimentos,
    o con la foto de un plato servido (P1-PLAN-LOTE-687/694)."""
    if not isinstance(vision, dict) or not vision.get("kind"):
        return False
    items = vision.get("items") if vision.get("kind") == "multi" else [vision]
    hay_plato = any(isinstance(i, dict) and str(i.get("kind") or "") == "plato" and i.get("description")
                    for i in (items or []))
    if not hay_plato:
        return False
    
    t = str(prompt or "").strip()
    if "?" in t or "¿" in t:
        return False
    p_norm = _plano(t)
    if any(q in p_norm for q in ("puedo comer", "debo comer", "cuantas calorias", "cuanto engorda", "que tiene", "es saludable", "sirve para", "receta de")):
        return False
    
    # Si el usuario mandó SOLO la foto (sin texto) y el análisis tiene dudas, manda la regla de dudas (no registrar aún)
    if es_solo_foto(prompt) and not _platos_claros(vision):
        return False

    return True


def instruccion_solo_foto(vision) -> str:
    """[P1-PLAN-LOTE-687] La regla del turno de la foto SOLA de un plato claro ("" si no lo es)."""
    if not (isinstance(vision, dict) and vision.get("solo_foto")):
        return ""
    return (
        " SOLO LA FOTO: el usuario te mandó la foto de su plato sin escribir nada: se lo comió. Regístralo EN ESTE TURNO "
        "con `log_consumed_meal`, las cifras del análisis y el `meal_type` que toque por la hora y lo que ya lleva "
        "registrado hoy (de noche con cena previa de hoy, una comida moderada ≤350 kcal es merienda nocturna), sin preguntarle antes si se lo comió, y confírmalo en una frase: qué anotaste. Aunque la "
        "conversación viniera de otro tema, ESTA foto es su comida. NUNCA le asignes 'desayuno' de noche o tarde solo porque tu pregunta anterior hablase de desayuno."
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
