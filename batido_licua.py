# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-862 · 2026-09-29] Un batido siempre dice que se licúa lo que lleva.

Batería real MX de 6d (29-sep, D2): «Batido cremoso de guineo, manzana y leche de soya con queso cottage» salió con
«Mise en place: pela ½ guineo… mide 600 ml de leche de soya…», «💪 Agrega queso cottage a la licuadora y licúa hasta
integrar» y «Montaje: sirve frío en un vaso alto»: el guineo, la manzana, la leche y la avena nunca entran en la
licuadora. El pase de los platos sin cocción (P1-NOCOOK-TDF-STRIP) borra el «El Toque de Fuego» que dice «sin cocción»…
con la instrucción de licuar dentro; y el paso de licuadora obligatorio (P1-BLEND-STEP-REQUIRED) veía «licúa» en el paso
del cerrador y se daba por cumplido. Corpus: 7 batidos así, todos de planes RD.

Aquí, al final de la cadena (contrato), un plato cuyo NOMBRE dice que es un batido (`batido_nombre.es_batido`) y en el
que ningún paso propio (ni nota ni «💪» del cerrador) licúa, bate o tritura recibe «El Toque de Fuego: coloca los
ingredientes… en la licuadora y licúa…» ANTES del paso del cerrador (así el «💪 Agrega… a la licuadora» se lee después)
o del Montaje. Si el Montaje pone algo por encima, eso queda fuera de la licuadora. Knob `MEALFIT_BATIDO_LICUA` (True).
tooltip-anchor: P1-PLAN-LOTE-862
"""
from __future__ import annotations

import re
import unicodedata

#: el VERBO de licuar (no «el batido» ni «el licuado» del Montaje: «sirve el batido en un vaso» no licúa nada)
_LICUA_RE = re.compile(r"\b(?:licu(?:a|ar|alo|ala|alos|alas|ando|e|en)|licuadora|bat(?:e|ir|iendo|elo|ela|elos|elas)|"
                       r"tritur(?:a|ar|alo|ala|ando)|proces(?:a|ar|alo|ala|ando)|mixer|batidora)\b")
_NOTA_RE = re.compile(r"^\s*(?:⚠|🤰|⚕|🌱|🛡|💡|💪|nota\b)", re.IGNORECASE)
_ENCIMA_RE = re.compile(r"\b(?:espolvorea\w*|corona\w*|por\s+encima|decora\w*|encima)\b")


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_BATIDO_LICUA", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def asegurar(meal) -> bool:
    """Inserta el paso de licuar si el batido no lo tiene. True si lo insertó; jamás lanza."""
    try:
        if not on() or not isinstance(meal, dict):
            return False
        rec = meal.get("recipe")
        if not isinstance(rec, list) or not rec:
            return False
        if not __import__("batido_nombre").es_batido(_sa(meal.get("name"))):
            return False
        propios = [p for p in rec if isinstance(p, str) and not _NOTA_RE.search(p)]
        if any(_LICUA_RE.search(_sa(p)) for p in propios):
            return False
        montaje = next((p for p in propios if _sa(p).lstrip().startswith("montaje")), "")
        que = ("los ingredientes del batido (menos lo que va por encima al servir)" if _ENCIMA_RE.search(_sa(montaje))
               else "todos los ingredientes")
        paso = f"El Toque de Fuego: coloca {que} en la licuadora y licúa a alta velocidad hasta obtener una mezcla homogénea."
        i = next((k for k, p in enumerate(rec) if isinstance(p, str) and p.lstrip().startswith("💪")
                  and "licuadora" in _sa(p)), None)
        if i is None:
            i = next((k for k, p in enumerate(rec) if isinstance(p, str) and _sa(p).lstrip().startswith("montaje")),
                     len(rec))
        meal["recipe"] = rec[:i] + [paso] + rec[i:]
        meal.pop("_display", None)
        return True
    except Exception:                                                          # noqa: BLE001
        return False
