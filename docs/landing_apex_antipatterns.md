# Reglas anti-refactor del landing y el apex

[P3-CLAUDEMD-CAP · movido desde CLAUDE.md 2026-08-16] Estas ocho reglas vivían bajo
el encabezado «El path degradado necesita su propio backstop», que trata del motor de
planes. No tienen nada que ver con él: llegaron ahí por deriva, párrafo a párrafo,
hasta sumar 4.634 bytes bajo un título que las contradice. Un lector que busque cómo
se protege la generación degradada encontraba ocho párrafos sobre bundles de
JavaScript, y quien buscara las reglas del landing no las buscaría ahí jamás.

Se mueven **verbatim**: cada una es una advertencia de las que dicen «la corrección
obvia es la equivocada», y comprimirlas las volvería crípticas justo donde importa.
CLAUDE.md conserva el puntero.

Ninguna está anclada por un test que lea CLAUDE.md (verificado 2026-08-16 sobre los
14 tests de landing/apex/supermercado: cero referencias). Los guards de conducta viven
en sus propios ficheros y siguen intactos.

---

[P1-LANDING-SW-DEFER · 2026-08-14] El SW se registraba con `immediate: true`: el install (73 entradas ≈ 988 KiB en el apex) arrancaba al evaluar el entry, o sea compitiendo con el chunk del hero. Ahora espera a `load`, y los `apple-touch-icon*` salen del precache — los pide el SO al INSTALAR y no se renderizan nunca. **Fuera del PRECACHE, NO borrados**: `manifest.json` los referencia y BRAND-FAVICON-B los declara fallback de root. Test test_p1_landing_sw_defer.py.


[P1-LANDING-OBS-PAPER] Replay de Sentry y autocapture de PostHog no corren en el apex. La política es SSOT en `utils/observabilityScope.js` y corta **por HOST, no por ruta** (`/precios` existe también en app.*, donde sí hay sesión que depurar). **`VITE_SENTRY_REPLAYS_SESSION_RATE=0` NO ahorra un byte** — regula la ingesta y el chunk se descarga igual; sólo saltarse el import lo hace. El opt-out de analítica pasa además a cookie de `.bioboros.com`: `localStorage` es por ORIGEN, así que el interruptor de Configuración era invisible para el landing. PostHog declarado en Privacidad §7/§8/§12/§13. Tests test_p1_landing_obs_paper.py + ObservabilityScope.p1_landing_obs_paper.test.js.


[P1-LANDING-HEAD-PRELOAD] El chunk del landing se precarga desde un bloque **gateado por host** (plugin `bioboros-landing-head`, `scripts/landingHead.mjs`): hay UN solo index.html para dos hosts, así que un `<link>` fijo le daría 226 kB de landing eager a app.* — lo que P3-APP-SUBDOMAIN-BUILD-SEP quitó de ahí. Los nombres salen del bundle porque llevan hash: escritos a mano caducan y fallan en **silencio** (un preload a un 404 no rompe nada, sólo deja de servir). El preconnect de Neon Auth deja de ser incondicional: el apex no contacta ese origen (P3-APEX-NO-SESSION). Test test_p1_landing_head_preload.py.


[P2-LANDING-COPY-TRUTH] `PRICING`, `NAME_BY_TIER` y `TIER_RANK` viven SOLO en `config/plans.js` (estaban byte a byte en Pricing.jsx y Upgrade.jsx, y ya habían divergido en `getMonthlyEquiv`). **El anual de un tier se decide con `ANNUAL_DISABLED_TIERS`, NUNCA mirando si `PRICING[tier].annual` existe**: ahí sobrevive `ultra.annual` como dato inerte y preguntarle a él resucita el «Max anual 449.99», que es el plan que P0-ANNUAL-PLANS-MISCONFIGURED dejó INACTIVE por cobrar esa cifra cada MES. La urgencia caduca sola vía `isLaunchOfferActive()` con offset `-04:00` (en UTC moriría a las 20:00 de la víspera). Test test_p2_landing_copy_truth.py.


[P2-LANDING-SITEMAP-SSOT] `public/sitemap.xml` se GENERA (`scripts/build-sitemap.mjs`, en `prebuild`) desde `paperSurface.js` + `news.js`. No lo edites a mano. Excluye `/cookies` y `/login` (redirigen) y las noticias con `href` (apuntan a otra página: su slug sería un duplicado). Test test_p2_landing_routes_ssot.py.


[P2-LANDING-OLA1-DIET] `lucide-react` fuera de `manualChunks`: un vendor chunk NOMBRADO recibe `modulepreload` eager en todas las rutas y arrastraba **181 iconos** (el landing usa ~25). Ola 1 medida 196.176 → 180.476 B gzip. Contrapartida honesta: el precache pasa de 67 a 125 entradas con los bytes planos (muchos ficheros diminutos comprimen peor) — aceptable porque el install ya corre tras `load`. Test test_p2_landing_ola1_diet.py.


[P1-APEX-ENTRY-DIET · 2026-08-14] `@sentry/*` se importa **sólo** desde `utils/sentryBoot.js`. Era el **37,2% del entry síncrono** (427.010 B) y había CINCO puertas (main, los 2 error boundaries, `analytics.js`, AgentPage): bastaba una abierta para devolverlo entero, porque los boundaries son eager. Los call sites van por `utils/observability.js`, que no tiene @sentry en su grafo, ENCOLA lo previo al init y **arranca el SDK ante el primer error** sin esperar al idle. **Diferir sin encolar no es optimizar, es quedarse ciego al arranque.** Entry 86,5→53,6 kB gz.


[P1-APEX-PRECACHE-BLIND · 2026-08-14] El precache excluía 237 KiB gz que el apex tiene PROHIBIDO ejecutar (replay, SDK de auth, markdown): el filtro miraba NOMBRES DE PÁGINA y esos son chunks `index-<hash>` anónimos — no se equivocaba, no podía verlos. Ahora se clasifica por **marcador de paquete** (`precacheAudience.mjs`), no por dominancia: `@sentry/core` diluye el chunk de replay y una regla de volumen NO lo atrapa. **No quites `manifestTransforms` de vite.config.js** — `globIgnores` no puede casar nombres hasheados. Guard de peso en `postbuild`. 721,7→485,4 KiB gz. Tests test_p1_apex_precache_blind.py, test_p1_ci_gate_passable.py.


Plan de producción del landing (25 gaps, 0 P0, 3 P1): [`docs/superpowers/specs/2026-08-14-landing-produccion-design.md`](docs/superpowers/specs/2026-08-14-landing-produccion-design.md).

