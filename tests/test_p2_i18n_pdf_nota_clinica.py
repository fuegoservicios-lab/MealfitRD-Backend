"""[P2-I18N-PDF-NOTA-CLINICA · 2026-08-22] La advertencia clínica del PDF salía con el
TITULAR traducido y el CUERPO en español.

El recuadro del PDF ya envolvía su titular en `t()` («⚕️ Consulta a tu profesional de
salud»), y debajo interpolaba `escapeHtml(String(_rpr.note))` crudo: la frase que dice POR
QUÉ hay que consultar --y que en la rama renal es la más crítica del documento-- entraba en
español dentro de un documento francés. Ahora `glossClinicalNote` glosa los fragmentos fijos
al renderizar, sin tocar el dato persistido.

═══════════════════════════════════════════════════════════════════════════════
POR QUÉ ESTE GUARD MIRA AL BACKEND Y NO AL CATÁLOGO
═══════════════════════════════════════════════════════════════════════════════

`npm run i18n:check` ya garantiza que toda clave viva tenga traducción en los 4 idiomas. Lo
que NO puede ver es que la clave siga siendo el texto que el backend emite: para el gate,
`t('frase que ya nadie produce')` es una clave viva y perfectamente traducida.

O sea que el modo de fallo real de esta pieza es exactamente el que el motor de i18n tiene
por diseño --la clave ES el texto español, así que cambiar el copy huérfana su traducción en
silencio-- pero DESPLAZADO: el copy no vive en el fichero que lo traduce, vive en
`graph_orchestrator.py`. Cambiar ahí una coma deja el glosador entero inerte y los cuatro
catálogos en verde al 100%.

Por eso el ancla es la paridad BIDIRECCIONAL:

  · Todo fragmento fijo que el backend concatena para `requires_professional_review.note`
    tiene que ser reconocible por `clinicalNoteGloss.js`.  ← el backend cambió y nadie lo dijo
  · Toda clave declarada en `clinicalNoteGloss.js` tiene que corresponder a algo que el
    backend siga emitiendo.                                ← quedó una clave fósil traducida ×4

Se probó por MUTACIÓN antes de escribirlo: cambiar «nefrólogo» por «nefrologo» en
`graph_orchestrator.py` pone rojo el primer sentido, y borrar un fragmento de
`clinicalNoteGloss.js` pone rojo el otro.

tooltip-anchor: P2-I18N-PDF-NOTA-CLINICA
"""
from __future__ import annotations

import ast
import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_ORQ = _BACKEND / "graph_orchestrator.py"
_GLOSADOR = _ROOT / "frontend" / "src" / "utils" / "clinicalNoteGloss.js"
_DASHBOARD = _ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"

_MARKER = "P2-I18N-PDF-NOTA-CLINICA"

# [P1-I18N-NOTA-MEDICAMENTOS-EN-ESPANOL · 2026-08-23] Esto era una TUPLA A MANO, y por eso
# se quedó corta: la rama de MEDICAMENTOS —la que avisa de interacción fármaco-alimento
# (warfarina/INR, levotiroxina, potasio)— usa `_med_note`, que no estaba en la lista. El
# guard existía justamente para impedir «si alguien cambia el copy clínico, el glosador queda
# inerte» y no podía ver la séptima rama, así que un francés con medicamento declarado
# recibía el recuadro con el titular en francés y el cuerpo en español; y si además era
# renal, el párrafo cambiaba de idioma a mitad de frase.
#
# Ahora se DERIVAN del código: cualquier variable que el backend meta en `"note"` entra bajo
# el guard sin que nadie tenga que acordarse. La tupla de abajo queda sólo como suelo mínimo
# —si la derivación se rompe, seguimos cubriendo lo que ya cubríamos— y su desajuste con lo
# derivado lo denuncia `test_la_derivacion_cubre_al_menos_las_variables_conocidas`.
_VARIABLES_MINIMAS = ("_note", "_cap_txt", "_comorbid_txt", "_lc_note", "_mn_note", "_pg_note")


def _variables_de_la_nota(src: str) -> set:
    """Los nombres que el backend asigna a `requires_professional_review["note"]`.

    Dos formas, las dos vivas en `graph_orchestrator.py`:
        "note": _med_note,
        _existing["note"] = ((_existing.get("note") or "") + " " + _pg_note).strip()
    """
    nombres = set(_VARIABLES_MINIMAS)
    nombres |= set(re.findall(r'"note"\s*:\s*(_[A-Za-z0-9_]+)', src))
    for linea in re.findall(r'\["note"\]\s*=\s*(.+)', src):
        nombres |= set(re.findall(r'\b(_[A-Za-z0-9_]*(?:note|txt))\b', linea))
    return nombres

# Un literal cuenta como PROSA (y por tanto es traducible) si lleva al menos un espacio y
# 6 caracteres. Descarta de un plumazo las claves de `dict.get('protein_g')` y el separador
# `", ".join(...)`, sin necesitar una lista negra que se quedaría atrás.
_MIN_LONGITUD = 6


def _fuente(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"no existe {p} (¿repo hermano sin clonar?)")
    return io.open(p, encoding="utf-8").read()


def _fragmentos_del_backend() -> set[str]:
    """Los trozos de PROSA FIJA que el backend concatena en cada rama de la nota.

    `ast` ya funde los literales adyacentes de una concatenación implícita, así que cada
    bloque de varias líneas sale como UNA constante -- que es justo la granularidad con la
    que el glosador tiene que reconocerlo.
    """
    src = _fuente(_ORQ)
    arbol = ast.parse(src)

    fragmentos: set[str] = set()
    variables = _variables_de_la_nota(src)

    def _recoger(valor) -> None:
        for hijo in ast.walk(valor):
            if isinstance(hijo, ast.Constant) and isinstance(hijo.value, str):
                texto = hijo.value
                if len(texto) >= _MIN_LONGITUD and " " in texto:
                    fragmentos.add(texto)

    def _es_disclaimer(objetivo) -> bool:
        # [P1-I18N-SERVER-COPY-GANA-DISCLAIMER · 2026-08-23] `plan["_review_disclaimer"] = ("…")`:
        # el glosador traduce TAMBIÉN el disclaimer, y el backend lo asigna por subíndice, no
        # por variable. Sin esto, sus 6 claves salían como «fósiles que el backend no emite».
        return (isinstance(objetivo, ast.Subscript) and isinstance(objetivo.slice, ast.Constant)
                and objetivo.slice.value == "_review_disclaimer")

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign):
            nombres = {t.id for t in nodo.targets if isinstance(t, ast.Name)}
            if (nombres & variables) or any(_es_disclaimer(t) for t in nodo.targets):
                _recoger(nodo.value)
        elif isinstance(nodo, ast.Dict):
            for k, v in zip(nodo.keys, nodo.values):
                if isinstance(k, ast.Constant) and k.value == "_review_disclaimer" and v is not None:
                    _recoger(v)
    return fragmentos


def _claves_del_glosador() -> set[str]:
    """Las claves españolas declaradas en `clinicalNoteGloss.js` (`t('…')` / `i18nKey('…')`)."""
    src = _fuente(_GLOSADOR)
    # [P2-REVIEW-ISSUES-CLARO · 2026-09-02] El mismo archivo aloja desde entonces el glosador de
    # las OBSERVACIONES del revisor (`_REVIEW_ISSUE_COPIES`, «Día N, franja: copy corto»), cuyo
    # backend es `_REVIEW_ISSUE_HUMANIZE_MAP`, no la nota clínica: sus claves las cuida su propio
    # test. Aquí sólo cuentan las de la nota (`_FRAGMENTOS`, tope de proteína, disclaimers).
    src = src.split("const _REVIEW_ISSUE_COPIES")[0]
    # Mismo criterio TEXTUAL que usa `i18n-check.mjs`: el literal escrito dentro de la
    # llamada. Ninguna de estas cadenas lleva comilla simple, así que no hay que tratar
    # escapes -- y si alguien añade una que sí, este test lo verá como clave ausente.
    # Sólo CÓDIGO: un `t('…')` citado en un comentario (`// … || t('…')`) no es una clave —
    # comentario-vence-guard, y la prosa que explica el fallback la disparaba.
    codigo = "\n".join(l for l in src.splitlines()
                       if not l.strip().startswith("//") and not l.strip().startswith("*"))
    return set(re.findall(r"(?:\bt|\bi18nKey)\(\s*'([^'\\]+)'", codigo))


def _sin_placeholders(clave: str) -> list[str]:
    """Parte una clave-plantilla en sus trozos estáticos: `a{x}b` -> ['a', 'b']."""
    return [t for t in re.split(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", clave) if t]


def test_todo_fragmento_del_backend_lo_reconoce_el_glosador() -> None:
    """Si alguien cambia el copy clínico en el backend, el glosador queda inerte."""
    fragmentos = _fragmentos_del_backend()
    assert fragmentos, (
        f"no extraje ni un fragmento de {_ORQ.name}: se renombraron las variables "
        f"{sorted(_variables_de_la_nota(_fuente(_ORQ)))} o cambió la forma de componer la "
        f"nota. Con el extractor vacío este "
        f"guard aprueba TODO. [{_MARKER}]"
    )

    claves = _claves_del_glosador()
    assert claves, f"no extraje ni una clave de {_GLOSADOR.name} [{_MARKER}]"

    # COBERTURA POR SUSTRACCIÓN, no contención. `fragmento in "\n".join(claves)` diría que
    # sí a un fragmento que ninguna clave cubre entera y --peor-- a uno que una clave cubre
    # sólo en parte: el glosador sustituye por coincidencia LITERAL, así que media clave no
    # traduce nada. Restando cada clave del fragmento, lo que sobra es exactamente lo que se
    # quedaría en español. (Así se cazó que la red de seguridad renal cierra su frase SIN el
    # espacio final que sí lleva la rama de assemble, y que redacta la comorbilidad distinto.)
    #
    # Las claves-PLANTILLA se restan además por trozos: el backend emite `_cap_txt` como
    # f-string, así que sus constantes son los pedazos de ALREDEDOR de las dos cifras
    # (`' … a ~'`, `'g/día (≈'`, `' g/kg) …'`) y jamás casarían contra la clave entera, que
    # en su lugar lleva `{proteina}` y `{gkg}`. Los trozos se restan DESPUÉS de las claves
    # completas, para no relajar el criterio de los fragmentos fijos.
    ordenadas = sorted(claves, key=len, reverse=True)
    trozos = sorted(
        {t for k in claves if "{" in k for t in _sin_placeholders(k)}, key=len, reverse=True
    )

    def _resto(fragmento: str) -> str:
        queda = fragmento
        for k in ordenadas:
            queda = queda.replace(k, "")
        for t in trozos:
            queda = queda.replace(t, "")
        return queda

    huerfanos = sorted(f for f in fragmentos if _resto(f).strip())
    assert not huerfanos, (
        f"{len(huerfanos)} fragmento(s) de la nota clínica que el backend emite y que "
        f"`clinicalNoteGloss.js` NO sabe traducir. El glosador queda inerte para esa rama y "
        f"el usuario recibe la advertencia en español dentro de un PDF en su idioma — con "
        f"`i18n:check` en verde al 100%, porque para él la clave vieja sigue viva y "
        f"traducida.\n\nAñádelos a `_FRAGMENTOS` y traduce en los 4 catálogos:\n"
        + "\n".join(
            f"  · sin cubrir {_resto(f).strip()!r}\n    (dentro de {f!r})" for f in huerfanos
        )
        + f"\n[{_MARKER}]"
    )


def test_toda_clave_del_glosador_sigue_viva_en_el_backend() -> None:
    """El sentido contrario: una clave fósil, traducida ×4, que ya nadie produce."""
    src_backend = _fuente(_ORQ)
    # El backend parte sus literales por líneas, así que se compara contra el texto de las
    # constantes ya fundidas por `ast`, no contra el fichero en crudo.
    unido = "\n".join(sorted(_fragmentos_del_backend()))

    fosiles = []
    for clave in sorted(_claves_del_glosador()):
        trozos = _sin_placeholders(clave)
        if not all(t in unido for t in trozos):
            fosiles.append(clave)

    assert not fosiles, (
        f"{len(fosiles)} clave(s) de `clinicalNoteGloss.js` que el backend ya no emite. "
        f"Siguen consumiendo traducción en los 4 idiomas y `i18n:check` las da por buenas "
        f"(para él son claves vivas). Bórralas del glosador y de los catálogos:\n"
        + "\n".join(f"  · {c!r}" for c in fosiles)
        + f"\n[{_MARKER}]"
    )
    assert "requires_professional_review" in src_backend, (
        f"desapareció `requires_professional_review` del orquestador: si la nota ya no se "
        f"produce, este glosador entero sobra. [{_MARKER}]"
    )


def test_el_pdf_glosa_la_nota_en_vez_de_interpolarla_cruda() -> None:
    """El defecto original, en una línea: `escapeHtml(String(_rpr.note))`."""
    src = _fuente(_DASHBOARD)
    assert "glossClinicalNote(String(_rpr.note), t)" in src, (
        f"el PDF volvió a interpolar la nota clínica sin glosar. El titular del recuadro sí "
        f"pasa por `t()`, así que el resultado es un recuadro con la cabecera en el idioma "
        f"del usuario y la advertencia --lo único que de verdad hay que leer-- en español. "
        f"[{_MARKER}]"
    )
    assert re.search(r"escapeHtml\(\s*glossClinicalNote", src), (
        f"la nota glosada dejó de pasar por `escapeHtml`. La nota lleva nombres de "
        f"condición influenciados por el formulario: es una interpolación a `innerHTML`. "
        f"[{_MARKER}]"
    )


def test_el_glosador_no_traduce_los_nombres_de_las_condiciones() -> None:
    """La frontera dura: se traduce lo que el usuario LEE, no lo que el motor IDENTIFICA.

    Los nombres de condición van dentro de la nota, entre paréntesis, y el backend los compara
    por igualdad de string para decidir el gate clínico. Traducirlos rompería esa comparación
    igual que traducir «Pollo» rompe `pantry_names_match`.
    """
    claves = _claves_del_glosador()
    # La clave del preámbulo termina en el paréntesis ABIERTO: la lista de condiciones queda
    # fuera de toda sustitución posible, por construcción.
    preambulo = [c for c in claves if c.rstrip().endswith("(")]
    assert preambulo, (
        f"la clave del preámbulo dejó de terminar en `(`. Si alguien la extendió para "
        f"abarcar también la lista de condiciones, esos nombres pasan a ser traducibles — y "
        f"son identificadores del motor, no copy. [{_MARKER}]"
    )
