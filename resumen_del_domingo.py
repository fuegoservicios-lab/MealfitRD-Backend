# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1108 · 2026-11-18] El resumen del domingo, contado por el coach en el chat.

El dueño eligió «Resumen del domingo: el coach cuenta cómo cerró tu semana y una cosa concreta para la siguiente».
Sobre el aviso del lote 1059 (`aviso_semanal`, a quien lo encendió en Ajustes): además del push, un mensaje del coach
en el chat del día con SUS cifras de la semana cerrada (`progreso_semanas`: días en meta, por debajo, por encima, sin
registrar, proteína media) y UNA sola cosa para la siguiente, elegida por regla (no por modelo):

  1. menos de 5 días registrados → registrar (sin registro no hay qué ajustar);
  2. proteína media < 85 % de su meta → cuánta le faltó al día;
  3. más días por encima que por debajo (2+) → un día más dentro;
  4. 2+ días por debajo → una merienda con proteína;
  5. si no, repetir.

Logs: nada de cifras (son datos de salud). Nunca lanza.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_T = {
    "es-DO": {
        "cierre": "Tu Semana {p} cerró con {d} de 7 días en meta{e}.",
        "desglose": " {b} por debajo, {s} por encima y {x} sin registrar.",
        "racha": " Llevas {r} días seguidos registrando.",
        "puntos": " Sumaste {pts} pts.",
        "para": " Para la Semana {n}:",
        "registro": " registra al menos 5 días, aunque sea solo lo principal: sin registro no hay cómo ajustar tu plan.",
        "proteina": " sube la proteína: te faltaron unos {g} g al día (un huevo y un yogur extra lo cubren).",
        "sobre": " busca quedar dentro de tu meta un día más que esta semana; si te pasas, que sea por poco.",
        "bajo": " los días que te quedes corto, añade una merienda con proteína (yogur, huevos o queso).",
        "bien": " repite lo que hiciste, que va muy bien.",
    },
    "en-US": {
        "cierre": "Your Week {p} closed with {d} of 7 days on goal{e}.",
        "desglose": " {b} under, {s} over and {x} not logged.",
        "racha": " You're on a {r}-day logging streak.",
        "puntos": " You earned {pts} pts.",
        "para": " For Week {n}:",
        "registro": " log at least 5 days, even just the main things: without logs there's nothing to adjust.",
        "proteina": " raise your protein: you were about {g} g short per day (an extra egg and a yogurt cover it).",
        "sobre": " try to stay within your goal one more day than this week; if you go over, keep it small.",
        "bajo": " on days you fall short, add a snack with protein (yogurt, eggs or cheese).",
        "bien": " repeat what you did, it's going really well.",
    },
    "pt-BR": {
        "cierre": "Sua Semana {p} fechou com {d} de 7 dias na meta{e}.",
        "desglose": " {b} abaixo, {s} acima e {x} sem registro.",
        "racha": " Você está há {r} dias seguidos registrando.",
        "puntos": " Você somou {pts} pts.",
        "para": " Para a Semana {n}:",
        "registro": " registre pelo menos 5 dias, mesmo que só o principal: sem registro não há como ajustar.",
        "proteina": " aumente a proteína: faltaram uns {g} g por dia (um ovo e um iogurte a mais resolvem).",
        "sobre": " tente ficar dentro da meta um dia a mais que nesta semana; se passar, que seja por pouco.",
        "bajo": " nos dias em que ficar abaixo, inclua um lanche com proteína (iogurte, ovos ou queijo).",
        "bien": " repita o que fez, está indo muito bem.",
    },
    "fr-FR": {
        "cierre": "Votre semaine {p} s’est terminée avec {d} jours sur 7 dans l’objectif{e}.",
        "desglose": " {b} en dessous, {s} au-dessus et {x} sans saisie.",
        "racha": " Vous enchaînez {r} jours de saisie.",
        "puntos": " Vous avez gagné {pts} pts.",
        "para": " Pour la semaine {n} :",
        "registro": " notez au moins 5 jours, même l’essentiel seulement : sans saisie, rien à ajuster.",
        "proteina": " augmentez les protéines : il vous en manquait environ {g} g par jour (un œuf et un yaourt de "
                    "plus suffisent).",
        "sobre": " essayez de rester dans l’objectif un jour de plus que cette semaine ; si vous dépassez, de peu.",
        "bajo": " les jours où vous êtes en dessous, ajoutez une collation protéinée (yaourt, œufs ou fromage).",
        "bien": " continuez comme ça, ça se passe très bien.",
    },
    "it-IT": {
        "cierre": "La tua settimana {p} si è chiusa con {d} giorni su 7 in obiettivo{e}.",
        "desglose": " {b} sotto, {s} sopra e {x} senza registrazione.",
        "racha": " Sei a {r} giorni di fila di registrazione.",
        "puntos": " Hai totalizzato {pts} pts.",
        "para": " Per la settimana {n}:",
        "registro": " registra almeno 5 giorni, anche solo il principale: senza registrazioni non c’è nulla da regolare.",
        "proteina": " aumenta le proteine: te ne mancavano circa {g} g al giorno (un uovo e uno yogurt in più bastano).",
        "sobre": " prova a restare nell’obiettivo un giorno in più di questa settimana; se sfori, che sia di poco.",
        "bajo": " nei giorni in cui resti sotto, aggiungi uno spuntino proteico (yogurt, uova o formaggio).",
        "bien": " ripeti quello che hai fatto, sta andando molto bene.",
    },
}


def consejo(semana: dict, meta_proteina: Optional[float]) -> tuple[str, dict]:
    """(clave del consejo, datos para su texto). Puro."""
    if int(semana.get("dias_registrados") or 0) < 5:
        return "registro", {}
    prot = semana.get("proteina_media")
    if meta_proteina and prot is not None and float(prot) < 0.85 * float(meta_proteina):
        return "proteina", {"g": int(round((float(meta_proteina) - float(prot)) / 5.0) * 5) or 5}
    sobre, bajo = int(semana.get("dias_sobre") or 0), int(semana.get("dias_bajo") or 0)
    if sobre >= 2 and sobre >= bajo:
        return "sobre", {}
    if bajo >= 2:
        return "bajo", {}
    return "bien", {}


def mensaje(locale: Optional[str], semana: Optional[dict], racha: int = 0,
            meta_proteina: Optional[float] = None) -> Optional[str]:
    """El mensaje del coach para la semana CERRADA `semana` (fila de `progreso_semanas`), o None. Puro."""
    if not semana or not semana.get("numero"):
        return None
    t = _T.get(str(locale or ""), _T["es-DO"])
    d = int(semana.get("dias_en_meta") or 0)
    e = int(semana.get("estrellas") or 0)
    reg = int(semana.get("dias_registrados") or 0)
    txt = t["cierre"].format(p=semana["numero"], d=d, e=(" " + "★" * e) if e else "")
    txt += t["desglose"].format(b=int(semana.get("dias_bajo") or 0), s=int(semana.get("dias_sobre") or 0),
                                x=max(0, 7 - reg))
    if semana.get("puntos"):
        txt += t["puntos"].format(pts=int(semana["puntos"]))
    if racha >= 2:
        txt += t["racha"].format(r=racha)
    clave, datos = consejo(semana, meta_proteina)
    return txt + t["para"].format(n=int(semana["numero"]) + 1) + t[clave].format(**datos)


def meta_de_proteina(uid: str, dia: str, off: int) -> Optional[float]:
    try:
        from diary_targets import resolve_targets_by_day
        m = (resolve_targets_by_day(uid, [dia], off).get(dia) or {}).get("macros") or {}
        v = m.get("protein") or m.get("protein_g") or m.get("proteina")
        return float(v) if v else None
    except Exception:  # noqa: BLE001
        return None


def escribir_en_el_chat(uid: str, locale: Optional[str], off: int, off_reloj: int, ahora) -> bool:
    """Deja el mensaje del domingo en el chat del día. `off` = el del juego (con la hora de corte del diario, como
    `aviso_semanal`); `off_reloj` = el del reloj, para elegir el chat de hoy. True si lo escribió. Nunca lanza."""
    try:
        import progreso_semanas as ps
        res = ps.resumen(uid, off, con_semanas=True, con_eventos=False) or {}
        cerradas = [s for s in res.get("semanas") or [] if s.get("cerrada")]
        if not cerradas:
            return False
        semana = cerradas[-1]
        txt = mensaje(locale, semana, int(res.get("racha") or 0),
                      meta_de_proteina(uid, str(semana.get("termina") or res.get("hoy") or ""), off))
        if not txt:
            return False
        from aviso_peso import _sesion_del_dia
        sesion = _sesion_del_dia(uid, ahora, off_reloj)
        if not sesion:
            return False
        from db_chat import save_message
        save_message(sesion, "model", txt)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[LOTE-1108] resumen del domingo sin chat: {type(exc).__name__}")
        return False
