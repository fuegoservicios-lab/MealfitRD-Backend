"""[P2-I18N-MEMOS-CONGELADOS · 2026-08-21] Un `useMemo` que traduce dentro devuelve
texto CONGELADO en el idioma anterior.

LA MECÁNICA, que es contraintuitiva y por eso hace falta anclarla:

`useT()` **no** devuelve una función nueva por locale. Devuelve la función de MÓDULO
(`i18n/index.js::t`), cuya identidad no cambia jamás entre renders — el hook solo
suscribe el componente al cambio de idioma. Así que ponerla en las deps de un memo es
un **no-op** para un cambio de idioma: React compara `t === t` y conserva el valor
cacheado. Las etiquetas se quedan en el idioma anterior hasta que algo *no relacionado*
invalide el memo, o hasta recargar la página.

Y como `es-DO` no tiene catálogo (es el fallback), en español **parece correcto**. El
bug solo existe para quien cambió de idioma, que es justamente quien no lo va a
reportar en español.

LA ASIMETRÍA CON `useCallback`, que es la razón de que este guard mire SOLO `useMemo`:
un `useCallback` no cachea un valor, cachea la función. Su cuerpo corre en cada
invocación y `t()` lee `_catalog` —estado de módulo— en ese instante. Un handler que
lanza `toast(t('Guardado'))` habla siempre el idioma vivo. Extender este guard a
`useCallback` daría 19 falsos positivos medidos (2026-08-21) y lo apagaría.

LAS DOS SALIDAS, y por qué el fix elige la segunda:

  1. `locale` en las deps. Funciona, pero `exhaustive-deps` lo declara *innecesario*
     (el cuerpo no nombra `locale`), el aviso se reporta en la línea del `useMemo` y la
     directiva de escape acaba huérfana. Cuesta un warning permanente contra el techo.
  2. Sacar el rotulado FUERA del memo. Lo caro es agrupar N sesiones por fecha o
     calcular el delta contra el inventario; rotular son tres cadenas por render. Fuera
     del memo siguen SIEMPRE al idioma vivo y el linter no tiene nada que decir.

El guard acepta las dos: ancla la INVARIANTE (un memo no devuelve texto traducido sin
depender del idioma), no la implementación.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "frontend" / "src"

_MARKER = "P2-I18N-MEMOS-CONGELADOS"

# `t('…')` y `tn(n, '…', '…')`. Son DOS patrones, no uno con `n?` opcional: el primer
# argumento de `tn` es el CONTADOR, no una cadena, así que exigir comilla tras el
# paréntesis deja invisible toda llamada plural. Lo cazó la mutación de control de este
# mismo fichero, que es exactamente para lo que está.
#
# El `tn(` no necesita exigir comilla porque el nombre ya es inequívoco; el negative
# lookbehind basta para descartar `btn(`, `Btn(` y `_tn(`, que existen en el árbol.
_LLAMADA = re.compile(r"(?<![\w.])(?:t\(\s*['\"]|tn\()")

# `locale`, `_dashLocale`, `_locale`… Case-insensitive Y sin `\b` por delante: el
# prefijo con guion bajo o el camelCase son word chars y `\b` no casaría.
_DEP_IDIOMA = re.compile(r"locale", re.IGNORECASE)

# Un exento necesita su razón; una whitelist sin motivo es indistinguible de un olvido.
_EXENTOS: dict[str, str] = {}


def _ficheros() -> list[Path]:
    if not _SRC.exists():
        pytest.skip(f"{_SRC} no existe en este checkout (repos hermanos)")
    return sorted(_SRC.rglob("*.jsx")) + sorted(_SRC.rglob("*.js"))


def _memos(fuente: str):
    """(offset, cuerpo, deps) de cada `useMemo(...)`, con balanceo real de paréntesis.

    Contar llaves o partir por comas no sirve: los cuerpos llevan objetos, arrays,
    template literals y llamadas anidadas. El balanceo es la única forma de saber dónde
    acaba la llamada, y el último `[` de nivel superior es dónde empiezan las deps.
    """
    for m in re.finditer(r"\buseMemo\(", fuente):
        i = m.end() - 1
        prof = 0
        while i < len(fuente):
            if fuente[i] == "(":
                prof += 1
            elif fuente[i] == ")":
                prof -= 1
                if prof == 0:
                    break
            i += 1
        if i >= len(fuente):
            continue
        interior = fuente[m.end():i]
        corte = interior.rfind("[")
        cuerpo = interior[:corte] if corte != -1 else interior
        deps = interior[corte:] if corte != -1 else ""
        yield m.start(), cuerpo, deps


def _hallazgos() -> list[tuple[str, int, int]]:
    out = []
    for p in _ficheros():
        rel = p.relative_to(_SRC).as_posix()
        if rel in _EXENTOS:
            continue
        s = p.read_text(encoding="utf-8")
        if "useMemo" not in s:
            continue
        for off, cuerpo, deps in _memos(s):
            n = len(_LLAMADA.findall(cuerpo))
            if n and not _DEP_IDIOMA.search(deps):
                out.append((rel, s[:off].count("\n") + 1, n))
    return out


def test_ningun_memo_traduce_sin_depender_del_idioma() -> None:
    malos = _hallazgos()
    assert not malos, (
        "Estos `useMemo` llaman a `t()`/`tn()` y no dependen del idioma, así que su "
        "texto se congela en el idioma anterior al cambiar de idioma sin recargar "
        "(y en es-DO parece correcto):\n  "
        + "\n  ".join(f"{f}:{ln} — {n} llamada(s)" for f, ln, n in malos)
        + f"\n\nDos salidas: sacar el rotulado FUERA del memo (preferida — el cálculo "
        f"caro se queda dentro, la traducción sale) o meter `locale` en las deps con "
        f"una directiva de escape para `exhaustive-deps`. [{_MARKER}]"
    )


def test_el_detector_ve_un_memo_congelado_de_verdad() -> None:
    """MUTACIÓN DE CONTROL. Sin esto, un fallo del parser (una llamada que ya no case,
    un balanceo que se coma el cuerpo) deja el test en verde PASANDO EN VACÍO —
    exactamente el modo de fallo que ya se pagó en `P1-CULINARY-METADATA-BETA`.
    """
    fuente = """
    const x = useMemo(() => {
        const opciones = { a: t('semanal'), b: tn(n, '{n} día', '{n} días') };
        return Object.entries(opciones).map(([k, v]) => ({ k, v }));
    }, [n]);
    """
    encontrados = [(cuerpo, deps) for _, cuerpo, deps in _memos(fuente)]
    assert len(encontrados) == 1, f"el parser no aisló el memo: {encontrados}"
    cuerpo, deps = encontrados[0]
    assert len(_LLAMADA.findall(cuerpo)) == 2, "no vio las dos llamadas del cuerpo"
    assert not _DEP_IDIOMA.search(deps), "creyó ver el idioma en unas deps que no lo traen"


def test_el_detector_acepta_las_dos_salidas() -> None:
    """La invariante es «no devuelve texto traducido sin depender del idioma», no una
    forma concreta de conseguirlo. Las dos formas legítimas tienen que pasar."""
    con_locale = "const x = useMemo(() => ({ a: t('semanal') }), [n, _dashLocale]);"
    assert not [1 for _, c, d in _memos(con_locale)
                if _LLAMADA.findall(c) and not _DEP_IDIOMA.search(d)], (
        "`_dashLocale` en las deps es una salida válida y el guard la rechazó "
        "(¿un `\\b` delante de `locale`? el guion bajo también es word char)"
    )

    fuera = """
    const x = useMemo(() => grupos(sesiones), [sesiones]);
    const rotulado = x.map((g) => ({ ...g, label: t('Hoy') }));
    """
    assert not [1 for _, c, d in _memos(fuera)
                if _LLAMADA.findall(c) and not _DEP_IDIOMA.search(d)], (
        "el rotulado FUERA del memo es la salida preferida y el guard la rechazó "
        "(¿el balanceo se comió texto de después del `useMemo`?)"
    )


def test_los_dos_memos_del_pfix_siguen_arreglados() -> None:
    """Los dos sitios concretos que este P-fix cerró, por nombre.

    Un guard genérico se satisface si alguien borra el memo; estos dos anclan que el
    cálculo caro sigue ahí y que lo que salió fuera es el rotulado.
    """
    for rel, memo, rotulo in (
        ("pages/AgentPage.jsx", "groupedSessions", "ETIQUETA_GRUPO"),
        ("pages/Dashboard.jsx", "restockPreview", "_restockDurationLabel"),
    ):
        p = _SRC / rel
        if not p.exists():
            pytest.skip(f"{rel} no existe en este checkout")
        s = p.read_text(encoding="utf-8")
        assert f"const {memo} = useMemo(" in s, (
            f"{rel}: desapareció el memo `{memo}`. El cálculo caro tiene que seguir "
            f"memoizado; lo que salió fuera es solo la traducción. [{_MARKER}]"
        )
        assert rotulo in s, (
            f"{rel}: falta `{rotulo}`, el rotulado que se sacó del memo. Si volvió "
            f"dentro, las etiquetas vuelven a congelarse. [{_MARKER}]"
        )
