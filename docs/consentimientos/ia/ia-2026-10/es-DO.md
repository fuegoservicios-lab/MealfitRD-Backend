# Permiso para la IA de terceros · `ia-2026-10` · es-DO

[P1-PLAN-LOTE-844 · ronda 1] Texto EXACTO de la hoja «Tus datos y la IA» que la app muestra en `es-DO` para la versión
`ia-2026-10`. Es lo que la persona acepta (art. 7.1 del RGPD: el consentimiento tiene que poder demostrarse): la fila de
`user_consents` guarda la versión y el `text_sha256` de este texto.

- Versión: `ia-2026-10`
- Idioma: `es-DO`
- SHA-256 (`text_sha256`): `830abb4b08c5108f67001f2e406ba345ef5882e7efda98c208369aafe6d7ba14`
- Fuente en la app: `frontend/src/consent/textoDeLaHoja.js` (los bloques, por `t()`) y el catálogo del idioma
  (`frontend/src/i18n/locales/<locale>.json`; `es-DO` es el texto base, sin catálogo). `textoPlanoDeLaHoja` une los
  bloques con un salto de línea: ese texto, en UTF-8, es el que se firma.
- Guardián: `frontend/src/__tests__/lote844.textos.test.js` fija el SHA-256 de cada idioma y comprueba que este fichero
  coincide con el catálogo. Cambiar el texto obliga a decidir si se sube la versión (y a escribir `ia-AAAA-MM/`).

El texto va entre las dos marcas; los dos saltos de línea que lo separan de ellas no forman parte de él.

<!-- texto:inicio -->
Tus datos y la IA
Para crear tu plan, responderte en el coach y analizar tus fotos, Bioboros envía algunos de tus datos a proveedores de inteligencia artificial externos. No los usamos para publicidad ni los vendemos.
DeepSeek
Hangzhou DeepSeek, República Popular China
Recibe tu perfil de salud (edad, peso, condiciones, medicamentos, alergias, embarazo, restricción religiosa de dieta), tus preferencias, tu nombre, tus mensajes con el coach y lo que anotas en el diario.
Para generar partes de tu plan, responderte en el coach y estimar lo que anotas.
OpenAI
EE. UU.
Recibe tu perfil de salud, tus preferencias y, cuando hace falta, parte de la conversación.
Para generar días de tu plan, revisar que sea seguro y servir de respaldo.
Google Gemini
EE. UU.
Recibe las fotos que escaneas o envías al chat, lo que escribes para aclararlas y tu país; en el modo voz, el texto que lee en voz alta.
Para reconocer los alimentos y generar la voz.
Cohere
Servidores principalmente en EE. UU.
Recibe un resumen de tu perfil (objetivo, alergias, condiciones), las notas que el coach recuerda de ti y la descripción de tus comidas.
Para que el coach encuentre lo relevante.
Transferencia a China
DeepSeek trata tus datos en la República Popular China. La Unión Europea no reconoce a China un nivel de protección adecuado y no tenemos firmadas cláusulas contractuales con DeepSeek: tus datos podrían quedar al alcance de las autoridades de ese país y te sería más difícil ejercer allí tus derechos.
Acepto que Bioboros use mis datos de salud para crear mi plan y darme el coach, y que los envíe a los proveedores de IA indicados.
Acepto que mis datos se envíen a DeepSeek, en China, conociendo los riesgos descritos.
Opcional
Ayúdanos a mejorar: analítica de uso, sin datos de salud.
Bioboros no sustituye a tu médico ni a tu nutricionista. Consúltales antes de cambiar tu alimentación, sobre todo si tienes una condición médica, tomas medicamentos, estás embarazada o en lactancia, o te operaron de cirugía bariátrica.
Puedes retirar tu permiso cuando quieras en Configuración → Privacidad → IA de terceros. Retirarlo detiene los envíos nuevos.
Política de Privacidad
Uso de IA
<!-- texto:fin -->
