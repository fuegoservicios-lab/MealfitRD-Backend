# Permiso para la IA de terceros · `ia-2026-10` · it-IT

[P1-PLAN-LOTE-844 · ronda 1] Texto EXACTO de la hoja «Tus datos y la IA» que la app muestra en `it-IT` para la versión
`ia-2026-10`. Es lo que la persona acepta (art. 7.1 del RGPD: el consentimiento tiene que poder demostrarse): la fila de
`user_consents` guarda la versión y el `text_sha256` de este texto.

- Versión: `ia-2026-10`
- Idioma: `it-IT`
- SHA-256 (`text_sha256`): `6572aa3c131eca2e909f0269837deee79dc8e68bd4f6af527c357af090a8c091`
- Fuente en la app: `frontend/src/consent/textoDeLaHoja.js` (los bloques, por `t()`) y el catálogo del idioma
  (`frontend/src/i18n/locales/<locale>.json`; `es-DO` es el texto base, sin catálogo). `textoPlanoDeLaHoja` une los
  bloques con un salto de línea: ese texto, en UTF-8, es el que se firma.
- Guardián: `frontend/src/__tests__/lote844.textos.test.js` fija el SHA-256 de cada idioma y comprueba que este fichero
  coincide con el catálogo. Cambiar el texto obliga a decidir si se sube la versión (y a escribir `ia-AAAA-MM/`).

El texto va entre las dos marcas; los dos saltos de línea que lo separan de ellas no forman parte de él.

<!-- texto:inicio -->
I tuoi dati e l'IA
Per creare il tuo piano, risponderti nel coach e analizzare le tue foto, Bioboros invia alcuni dei tuoi dati a fornitori esterni di intelligenza artificiale. Non li usiamo per la pubblicità e non li vendiamo.
DeepSeek
Hangzhou DeepSeek, Repubblica Popolare Cinese
Riceve il tuo profilo di salute (età, peso, patologie, farmaci, allergie, gravidanza, restrizione alimentare religiosa), le tue preferenze, il tuo nome, i tuoi messaggi con il coach e ciò che annoti nel diario.
Per generare parti del tuo piano, risponderti nel coach e stimare ciò che annoti.
OpenAI
USA
Riceve il tuo profilo di salute, le tue preferenze e, quando serve, parte della conversazione.
Per generare giorni del tuo piano, verificare che sia sicuro e fare da riserva.
Google Gemini
USA
Riceve le foto che scansioni o invii nella chat, ciò che scrivi per chiarirle e il tuo paese; in modalità vocale, il testo che legge ad alta voce.
Per riconoscere gli alimenti e generare la voce.
Cohere
Server principalmente negli USA
Riceve un riepilogo del tuo profilo (obiettivo, allergie, patologie), le note che il coach ricorda di te e la descrizione dei tuoi pasti.
Perché il coach trovi ciò che è rilevante.
Trasferimento in Cina
DeepSeek tratta i tuoi dati nella Repubblica Popolare Cinese. L'Unione europea non riconosce alla Cina un livello di protezione adeguato e non abbiamo firmato clausole contrattuali con DeepSeek: i tuoi dati potrebbero essere accessibili alle autorità di quel paese e ti sarebbe più difficile esercitare lì i tuoi diritti.
Accetto che Bioboros usi i miei dati di salute per creare il mio piano e offrirmi il coach, e che li invii ai fornitori di IA indicati.
Accetto che i miei dati siano inviati a DeepSeek, in Cina, consapevole dei rischi descritti.
Facoltativo
Aiutaci a migliorare: statistiche d'uso, senza dati di salute.
Bioboros non sostituisce il tuo medico né il tuo nutrizionista. Consultali prima di cambiare la tua alimentazione, soprattutto se hai una patologia, prendi farmaci, sei incinta o allatti, o hai subito un intervento di chirurgia bariatrica.
Puoi ritirare il tuo consenso quando vuoi in Impostazioni → Privacy → IA di terze parti. Ritirarlo interrompe i nuovi invii.
Informativa sulla privacy
Uso dell'IA
<!-- texto:fin -->
