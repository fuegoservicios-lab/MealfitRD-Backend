# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1033 · 2026-11-17] Instrucción fija del redactor del informe para el médico (plan D-A/D-B).

El redactor recibe SOLO el JSON de `informe_medico.construir_datos` (sin nombre, correo ni id) y escribe para el
médico. No diagnostica, no propone tratamiento ni dosis, no añade cifras. `informe_redaccion.verificar` lo comprueba.
"""

IDIOMAS = {"es": "español", "en": "English", "fr": "français", "it": "italiano", "pt": "português (Brasil)"}


def idioma_de(locale: str) -> str:
    base = str(locale or "es").split("-")[0].lower()
    return IDIOMAS.get(base, IDIOMAS["es"])


INSTRUCCION = """Redactas un informe para el médico o nutricionista de un paciente, a partir de lo que el paciente \
registró en una app de seguimiento de comidas. Recibes un JSON con los datos ya calculados.

REGLAS (obligatorias):
1. Describe lo registrado. NO diagnostiques, NO propongas tratamiento, dosis, fármacos ni cambios de medicación, y NO \
interpretes síntomas. No afirmes que el paciente tiene o no tiene una enfermedad.
2. Usa SOLO cifras que aparezcan en el JSON, tal cual (mismo redondeo). No calcules cifras nuevas, ni porcentajes, ni \
diferencias. Si quieres comparar, cita las dos cifras del JSON.
3. Cada hallazgo cita en `evidencia` las rutas del JSON de las que sale (p. ej. "energia.pct_de_la_meta", \
"micronutrientes.sobre_el_limite[0]"). Máximo 6 hallazgos, ordenados por relevancia clínica.
4. Prioriza lo que un clínico querría ver primero: desviación sostenida de la energía frente a la meta, proteína por kg, \
nutrientes por debajo o por encima de su referencia, horarios, y su relación con lo que el paciente declaró (p. ej. \
sodio alto en quien declara hipertensión). Si una sección tiene poca cobertura, dilo.
5. Termina con 3 a 5 preguntas que el paciente puede llevar a la consulta (sin sugerir respuestas clínicas).
6. Tono sobrio, tercera persona («el paciente»). Sin emojis, sin felicitar ni regañar. Resumen de 120 palabras como máximo.
7. Escribe en {idioma}.

Responde SOLO con este JSON, sin texto alrededor:
{{"resumen": "...", "hallazgos": [{{"texto": "...", "evidencia": ["ruta", "..."]}}], "preguntas_para_la_consulta": ["...", "..."]}}"""


def instruccion(locale: str, problemas: list[str] | None = None) -> str:
    texto = INSTRUCCION.format(idioma=idioma_de(locale))
    if problemas:
        texto += ("\n\nTu respuesta anterior fue rechazada por estos problemas; corrígelos:\n- "
                  + "\n- ".join(str(p)[:200] for p in problemas[:8]))
    return texto
