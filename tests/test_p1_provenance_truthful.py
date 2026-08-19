"""[P1-PROVENANCE-TRUTHFUL · 2026-08-19] Un `fdc_id` deja de poder mentir.

`fdc_id` no es una nota al pie: es una AFIRMACIÓN — «esta fila ES ese alimento de USDA».
Tras el de-proxy español (`P1-BEDCA-DEPROXY-ES`) y el yogur (`P1-YOGURT-NATURAL`)
quedaban 16 grupos con el id compartido por 36 filas, y como mucho una de cada grupo
podía estar diciendo la verdad.

LA REGLA: conserva el id la fila cuya **identidad** y cuyos **valores** siguen
coincidiendo con la fila real de USDA — descripción consultada a la API, una por grupo.
Las demás pasan a `fdc_id = NULL`, `nutrition_source = 'manual'` y
`nutrition_source_ref = 'usda:<id> (proxy: <descripción>)'`. La traza no se pierde: deja
de presentarse como fuente. Un dato aproximado ETIQUETADO como aproximado es honesto; el
mismo dato con un `fdc_id` es una fuente falsa.

DOS GRUPOS QUEDAN FUERA A PROPÓSITO (`174220` Mejillones/Vieira, `175202` Habichuelas
blancas/Judías blancas): la API no devolvió su descripción por el límite de `DEMO_KEY`
(30 req/hora). Sin saber qué alimento es realmente el id, decidir quién lo conserva sería
adivinar — y este P-fix existe justamente para que el catálogo deje de afirmar lo que no
sabe. Este test **ancla ese número**: si alguien los cierra, tiene que bajarlo aquí, y si
alguien añade un grupo compartido nuevo, sube y falla.

LO QUE NO HACE: no borra ni fusiona filas. `Requesón`/`Queso ricotta` y `Judías
blancas`/`Habichuelas blancas` son el mismo alimento con dos nombres, pero el catálogo se
resuelve **por cadena, no por id**: fusionarlas rompería cualquier plan, `user_inventory`
o `supermarket_products.master_food_name` que las referencie por nombre.

tooltip-anchor: P1-PROVENANCE-TRUTHFUL
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_MIG = "p1_provenance_truthful_2026_08_19.sql"

#: Grupos que la migración deja intactos por falta de descripción verificada.
_SIN_VERIFICAR_ESPERADOS = 2


def _sql(root: bool = False) -> str:
    base = _ROOT if root else _BACKEND
    return io.open(base / "migrations" / _MIG, encoding="utf-8").read()


def test_migracion_en_los_dos_dirs_ssot_y_byte_identica():
    a = (_BACKEND / "migrations" / _MIG).read_bytes()
    b = (_ROOT / "migrations" / _MIG).read_bytes()
    assert a and a == b


def test_toda_fila_que_pierde_el_id_deja_escrito_de_donde_salio():
    """Vaciar el `fdc_id` sin dejar rastro convertiría un proxy documentado en un
    número huérfano: peor que el problema original."""
    bloques = re.findall(r"UPDATE public\.master_ingredients SET(.*?);", _sql(), re.S)
    assert bloques, "no hay ningún UPDATE"
    for b in bloques:
        assert "fdc_id = NULL" in b, f"UPDATE que no limpia el fdc_id: {b[:100]}"
        assert re.search(r"nutrition_source_ref = 'usda:\d+ \(proxy: ", b), (
            f"UPDATE que vacía el id sin dejar la referencia: {b[:120]}")
        assert "nutrition_source = 'manual'" in b, (
            f"una fila sin fdc_id no puede seguir declarándose 'usda': {b[:100]}")


def test_es_idempotente():
    for b in re.findall(r"UPDATE public\.master_ingredients SET(.*?);", _sql(), re.S):
        assert "fdc_id IS NOT NULL" in b, (
            "sin el filtro `fdc_id IS NOT NULL`, re-ejecutar reescribiría filas ya limpias")


def test_cada_grupo_declara_la_descripcion_REAL_de_usda():
    """La decisión de quién conserva el id se justifica con lo que USDA dice que es ese
    alimento, no con una corazonada. Si el comentario no trae la descripción, la próxima
    persona no puede auditar la decisión."""
    cabeceras = re.findall(r"-- fdc (\d+) = USDA «([^»]*)»", _sql())
    assert len(cabeceras) >= 12, f"solo {len(cabeceras)} grupos documentados"
    for fid, desc in cabeceras:
        assert desc and desc != "?", f"fdc {fid} sin descripción real"
        assert "conserva:" in _sql(), "falta declarar quién conserva el id"


def test_los_grupos_no_verificados_estan_declarados_y_contados():
    """El límite de la migración es parte del artefacto. Si alguien cierra esos dos
    grupos debe bajar el número aquí; si aparece un grupo compartido nuevo, sube y este
    test lo caza."""
    sql = _sql()
    assert "FUERA DE ESTA MIGRACION" in sql, (
        "la migración debe declarar qué grupos NO toca y por qué")
    declarados = re.findall(r"^--   fdc (\d+): ", sql, re.M)
    assert len(declarados) == _SIN_VERIFICAR_ESPERADOS, (
        f"la migración declara {len(declarados)} grupos sin verificar, "
        f"el contrato dice {_SIN_VERIFICAR_ESPERADOS}")
    # El sanity SQL debe tolerar exactamente esos, ni uno más.
    m = re.search(r"IF _dup > (\d+) THEN", sql)
    assert m and int(m.group(1)) == _SIN_VERIFICAR_ESPERADOS, (
        "el sanity de duplicados debe tolerar solo los grupos declarados sin verificar")


def test_no_borra_ni_fusiona_filas():
    """Guard duro: una limpieza de PROCEDENCIA que borre filas se lleva por delante
    referencias por nombre en planes, inventarios y supermarket_products."""
    sql = _sql()
    assert not re.search(r"\bDELETE\s+FROM\b", sql, re.I), "esta migración no borra filas"
    assert not re.search(r"\bDROP\s+TABLE\b", sql, re.I)
    assert "no borra ni fusiona" in sql, (
        "la decisión de NO fusionar los sinónimos debe quedar escrita: es lo que impide "
        "que alguien lo 'complete' luego sin ver el coste")


def test_explica_por_que_los_sinonimos_no_se_fusionan():
    sql = _sql()
    for pista in ("Requeson", "Judias blancas", "master_food_name"):
        assert pista in sql, f"falta la justificación que menciona {pista}"


# ══════════════ fix round 2: la corrección tenía su propia procedencia mal ══════════

_FIX2 = "p1_provenance_truthful_fix_round_2.sql"


def _sql2(root: bool = False) -> str:
    base = _ROOT if root else _BACKEND
    return io.open(base / "migrations" / _FIX2, encoding="utf-8").read()


def test_fix2_en_los_dos_dirs_ssot_y_byte_identica():
    a = (_BACKEND / "migrations" / _FIX2).read_bytes()
    b = (_ROOT / "migrations" / _FIX2).read_bytes()
    assert a and a == b


def test_fix2_purga_los_sentinels_de_error():
    """DEFECTO 1 de la ronda 1: el generador descartaba descripciones VACÍAS
    (`if not desc`), pero cuando la API devolvía 429 el caché guardaba la cadena
    `'SIN RESPUESTA (429)'` — perfectamente no-vacía. Cinco filas quedaron en
    producción con un mensaje de error donde va la fuente.

    Un guard que filtra por «vacío» no filtra sentinels: hay que filtrar por «es una
    descripción válida»."""
    sql = _sql2()
    assert "SIN RESPUESTA" in sql, "el sanity debe cazar el sentinel exacto que se coló"
    assert re.search(r"RAISE EXCEPTION '\[P1-PROVENANCE-TRUTHFUL fix2\][^']*sentinel", sql), (
        "falta el sanity que aborta si un sentinel vuelve a colarse")
    for basura in ("HTTP 4", "HTTP 5"):
        assert basura in sql, f"el sanity no cubre {basura}"


def test_fix2_distingue_proxy_de_valores_propios():
    """DEFECTO 2: la ronda 1 llamó «proxy» a filas cuyos valores son PROPIOS.

    `Tilapia` (96 kcal, colesterol 50) no es un proxy de camarón (85 y 161): sus
    números son suyos y lo único falso era el id. Degradar un dato bueno es el mismo
    error que ascender uno malo, al revés."""
    sql = _sql2()
    assert "id previo; valores propios" in sql
    assert "Tilapia" in sql and "Nueces mixtas" in sql, (
        "las filas DIFERENCIADO deben re-etiquetarse una a una")
    # Y ninguna de ellas puede seguir marcada como proxy en este archivo.
    for bloque in re.findall(r"UPDATE public\.master_ingredients SET(.*?);", sql, re.S):
        if "Tilapia" in bloque or "Nueces mixtas" in bloque or "Chinola" in bloque:
            assert "proxy:" not in bloque, f"una fila de valores propios sigue como proxy: {bloque[:110]}"


def test_fix2_no_afirma_lo_que_no_pudo_verificar():
    """173443 nunca respondió. Ambas filas comparten valores, así que una es proxy de
    la otra — pero sin la descripción no se sabe cuál, y afirmarlo sería volver a
    inventar."""
    sql = _sql2()
    assert "desc sin verificar" in sql
    assert "Crema mexicana" in sql and "Suero costeño" in sql


def test_fix2_solo_admite_tres_etiquetas_canonicas():
    """Sin este sanity, la próxima ronda inventa una cuarta forma de decir lo mismo y
    la columna deja de ser consultable."""
    sql = _sql2()
    assert "3 etiquetas canonicas" in sql or "etiquetas canonicas" in sql
    assert r"^usda:[0-9]+ \((proxy: .+|id previo; (valores propios|desc sin verificar))\)$" in sql, (
        "el sanity debe anclar el formato exacto de las tres etiquetas")
