# 🎧 Instant Translator

Traduzione audio **multicanale in tempo reale** per eventi live (conferenze,
piazze, sale multilingue, chiese, teatri). Un backend di **regia** prende i
flussi audio da un mixer digitale (Behringer **XR18** via USB) o da una scheda
multicanale, li separa per canale e per ciascuno esegue **ASR → traduzione →
TTS**, distribuendo a ogni ascoltatore **sottotitoli** e **audio parlato** nella
lingua scelta dal proprio telefono.

Tutto **offline/locale**: faster-whisper (ASR) · NLLB-200 (traduzione) · Piper (TTS).

---

## Come funziona

```
Mixer XR18 (USB, 48 kHz) ──► demux canali ──► ricampionamento 16 kHz ──► [per canale]
   VAD adattivo (segmenta voce) ──► ASR faster-whisper con commit incrementale
        │                          (manda in onda i pezzi già stabili, senza
        │                           aspettare la fine della frase)
        └─► testo sorgente ──► Traduzione NLLB-200 ──► [solo lingue trasmesse/ascoltate]
                                      │
                          ┌───────────┴───────────┐
                          ▼                        ▼
                   Sottotitoli (testo)        TTS Piper (audio)
                          │                        │
                          └──── HLS / WebSocket ───┘
                                      │
                        nginx / CDN ──► browser ascoltatori
```

Scelte pensate per **GPU singola + tanti ascoltatori**:

- **ASR una sola volta per canale** (non per ascoltatore).
- **Traduzione e TTS solo per le lingue trasmesse o con ascoltatori attivi**.
- **Parziali generati solo se qualcuno li guarda** (costano un'inferenza ASR
  sull'intero buffer: in broadcast puro non servono a nessuno).
- **Audio TTS generato una volta per (canale, lingua)** e riusato da HLS e WS.

### Componenti

| Modulo | Ruolo |
|---|---|
| `app/audio/` | enumerazione device, cattura multicanale, ricampionamento, demux, meter |
| `app/pipeline/` | motori ASR/MT/TTS (mock + reali), VAD adattivo e **orchestratore** |
| `app/hls/` | encoder ffmpeg per (canale × lingua), pacer timeline, watchdog, sottotitoli |
| `app/state.py` | **Hub** pub/sub thread-safe per il fan-out WebSocket |
| `app/api/` | API REST regia + ascoltatori, WebSocket, HLS, `/healthz` |
| `app/web/` | frontend: landing ascoltatore + dashboard di regia |
| `scripts/audio_probe.py` | diagnostica scheda audio e **mappatura dei canali del mixer** |
| `deploy/nginx.conf` | consegna a migliaia di ascoltatori (cache + static da disco) |

---

## Avvio rapido (modalità DEMO, senza GPU né modelli)

La **modalità mock** fa girare *tutto* il sistema con sorgenti e traduzioni
simulate e audio a tono: perfetta per provare interfaccia e flusso end-to-end.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (su Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy config.example.yaml config.yaml    # Linux/Mac: cp ...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Poi apri:

- **Ascoltatore** → <http://localhost:8000/>  (scegli canale → lingua → audio)
- **Regia** → <http://localhost:8000/admin>

La regia chiede un **token**: se non lo fissi in `config.yaml` (`admin.token`)
viene generato all'avvio e scritto nel log:

```
WARNING  instanttranslator.security: token regia generato per questa sessione: 3fK9-xQ2vTm
```

> Nota: l'audio nel browser parte solo dopo un tocco (policy autoplay): premi
> **“Attiva audio”** nella vista live.

---

## Modalità reale (faster-whisper + NLLB + Piper)

Richiede una **GPU NVIDIA** (ASR + traduzione su GPU via CTranslate2; Piper TTS
su CPU). Verificato su Windows 11 + Python 3.12 + RTX 5070.

```bash
pip install -r requirements-ml.txt

# torch serve SOLO per convertire NLLB in CTranslate2 (una tantum, non a runtime)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Scarica/converte i modelli (NLLB in CTranslate2 + voci Piper)
python scripts/download_models.py --languages it en es fr de
```

In `config.yaml` imposta `engine.mode: real` e configura i canali (vedi la
sezione sull'XR18 qui sotto). Poi riavvia.

> All'avvio in modalità reale parte un **warmup** in background che pre-compila i
> kernel CUDA: la *prima* inferenza su una GPU nuova può costare ~10s, dopo il
> warmup è immediata.

Se manca la voce Piper di una lingua, quella lingua **non viene offerta** al
pubblico (meglio che uno stream muto) e il log lo dice all'avvio; `/healthz`
riporta `missing_voices`.

### Performance misurate (RTX 5070, 12 GB)

| Motore | Latenza a regime |
|---|---|
| Whisper `large-v3` (GPU) | RTF ~0.09 → ~11 canali in real-time |
| Whisper `small` (GPU) | RTF ~0.03 → ~35× real-time |
| NLLB-200 600M (GPU) | ~110 ms per frase, con cache |
| Piper TTS (CPU) | RTF ~0.9 (real-time) |

Con molti canali simultanei valuta un modello Whisper più piccolo
(`medium`/`small`) per aumentare il margine.

---

## 🎛️ Ingresso audio dal Behringer XR18

L'XR18 è un'interfaccia USB **18 in / 18 out a 48 kHz fissi**. Tre cose da
sapere prima di collegarlo, perché determinano quanti canali puoi tradurre.

### 1. Quanti canali vedi su Windows

Il driver X-AIR per Windows espone gli ingressi come device **WDM/WASAPI**:
coppie (`IN 1-2`, `IN 3-4`, …) e un device **`IN 1-8` da 8 canali**. Tutti i 18
canali passano **solo via ASIO**, e il PortAudio incluso in `sounddevice` è
compilato **senza ASIO**:

```bash
python scripts/audio_probe.py      # elenca device, canali e host API
```

Se nell'elenco l'host API `ASIO` non c'è, il massimo su Windows con questo
stack è **8 canali contemporanei** (device `IN 1-8`). Opzioni:

| Vincolo | Cosa fare |
|---|---|
| ≤ 8 microfoni da tradurre | Instrada i mic che ti interessano sulle **USB sends 1-8** dell'XR18 (X-AIR Edit → *Setup → Routing → USB Sends*) e usa il device `IN 1-8`. |
| Più di 8 microfoni | Sposta la cattura su **Linux**: l'XR18 è USB Audio Class 2.0, ALSA espone tutti i 18 canali senza driver, e PortAudio li apre a 48 kHz. |
| Vuoi ASIO su Windows | Ricompila PortAudio con l'ASIO SDK e reinstalla `sounddevice` sul tuo build (nessuna wheel pubblica lo include). |

### 2. Da dove prendere il segnale sul mixer

Per l'ASR conta la **pulizia** del segnale, non il mix artistico:

- **Tap point per canale** (X-AIR Edit → *Channel → Config → USB send tap*):
  usa **post-EQ / pre-fader**. Post-EQ perché il filtro passa-alto e l'EQ
  ripuliscono rombo e sibilo; pre-fader perché i movimenti di fader del
  fonico non devono spegnere la traduzione.
- **Gate/expander attivo** su ogni canale (soglia ~-40 dB): in una piazza ogni
  microfono aperto sente il PA e gli altri mic (*bleed*). Il gate è la prima
  difesa contro trascrizioni doppie e frasi inventate.
- **Canale “Originale” (FLOOR)**: mandaci un **bus o il main LR**, non un
  singolo microfono, così chi sceglie “🎤 Originale” sente l'evento come in
  piazza. Si configura per canale con `floor_channel_index`.
- **Clock a 48 kHz** e mixer come master. L'app apre la scheda al suo rate
  nativo e ricampiona a 16 kHz per Whisper: **non** impostare `audio.samplerate`
  a 48000 (è il rate di elaborazione, non quello della scheda).

### 3. Mappare i `channel_index`

`channel_index` è la **colonna dentro lo stream** (parte da 0), non il numero
del canale sul mixer: `IN 1-8` → indici 0…7.

```bash
python scripts/audio_probe.py --device "IN 1-8" -s 60
```

Parla in un microfono e guarda quale barra si muove: quello è il tuo indice.
Lo stesso meter è nella **regia** (colonna “Livello”), con la soglia del VAD:
prima dell'evento serve a verificare che ogni canale riceva davvero segnale.

### 4. Esempio di configurazione

```yaml
audio:
  samplerate: 16000            # elaborazione (Whisper); la scheda resta a 48 kHz
  blocksize: 1600              # 100 ms
  default_input_device: "IN 1-8"   # per nome: gli indici USB si spostano tra i riavvii
  host_api: WASAPI             # WASAPI/WDM-KS lavorano al rate nativo; MME ricampiona

channels:
  - id: palco
    name: "Palco"
    description: "Relatore principale"
    source_language: it
    channel_index: 0           # USB send 1
    gain_db: 0                 # ritocco digitale se il mic è debole
    floor_channel_index: 7     # USB send 8 = main LR per lo stream "Originale"
    broadcast_languages: [en, es, fr]
  - id: ospite
    name: "Ospite"
    source_language: en
    channel_index: 1           # USB send 2
    broadcast_languages: [it]
```

> Gli indici PortAudio cambiano quando cambia l'ordine dei device USB: indica il
> device **per nome** (`"IN 1-8"`) e non ti si sposta la mattina dell'evento.

---

## 📡 Consegna a 2500 ascoltatori in piazza

Per grandi platee la consegna PCM-su-WebSocket non regge (banda e numero di
connessioni). Con **HLS** ogni canale trasmette le sue lingue come stream
segmentato **cacheabile**: l'origine genera ogni stream una sola volta,
nginx/CDN fa il fan-out a tutti.

```yaml
hls:
  enabled: true
  segment_time: 1.0        # segmenti corti = meno latenza
  list_size: 10
  delete_threshold: 30     # segmenti tenuti oltre la playlist (telefoni lenti)
  audio_codec: aac         # massima compatibilità (iOS incluso)
  audio_bitrate: 64k
  floor: true              # stream "Originale" senza ASR/MT/TTS
delivery:
  ws_audio_with_hls: false # l'audience va su HLS, non su PCM/WebSocket
  subtitle_poll_seconds: 3
  max_ws_listeners: 200
```

Richiede **ffmpeg** nel PATH. La landing sceglie automaticamente HLS per le
lingue trasmesse e usa **hls.js** (vendorizzato, funziona offline); su
Safari/iOS usa l'HLS nativo. I sottotitoli si sincronizzano all'audio via
`EXT-X-PROGRAM-DATE-TIME` (`hls.playingDate` su hls.js, `getStartDate()` su
Safari), quindi restano allineati anche con buffer diversi tra telefoni.

### Dimensionamento

| Voce | Conto |
|---|---|
| Banda audio | 64 kbps × 2500 = **~160 Mbps** in uscita |
| Richieste playlist | 1/s per ascoltatore → **cache 1 s** su nginx, 1 req/s all'origine |
| Richieste sottotitoli | 1 ogni 3 s → ~830 req/s su nginx, 1 req/s all'origine |
| Segmenti `.ts` | immutabili → cache lunga, un solo MISS per segmento |
| GPU | 1 ASR per canale + 1 traduzione per (canale × lingua) per enunciato |
| CPU | 1 TTS + 1 ffmpeg per (canale × lingua) + 1 per il FLOOR |

**Regole pratiche non negoziabili per una piazza:**

1. **Il processo Python non serve il pubblico.** Statici e file HLS li serve
   **nginx da disco** (vedi [`deploy/nginx.conf`](deploy/nginx.conf)): il
   processo che fa ASR/TTS in tempo reale non deve competere con 2500 telefoni.
2. **nginx su Linux.** Su Windows nginx usa `select()` e regge ~1024 connessioni
   per worker. L'origine con la GPU può restare su Windows, la cache no.
3. **Verifica l'uplink.** 160 Mbps sostenuti. Se il pubblico è su LTE/5G o su
   WiFi pubblica, metti una **CDN** davanti a nginx.
4. **WiFi dedicata solo se progettata.** 2500 client su access point non
   dimensionati non funziona: o rete cellulare (con CDN), o un impianto WiFi
   progettato per la densità (molti AP, 5 GHz, band steering).
5. **HTTPS + QR code.** Un dominio corto e un QR sui volantini/schermi; su HTTP
   alcuni browser mostrano avvisi che in piazza si traducono in supporto.
6. **`broadcast_languages` minime.** Ogni lingua in più = 1 TTS + 1 ffmpeg per
   canale, sempre attivi. Offri solo le lingue che servono davvero.

### Latenza attesa (onesta)

Con la **traduzione incrementale** attiva (default) il testo non aspetta la
fine della frase: appena un pezzo di trascrizione si stabilizza, viene tradotto,
sintetizzato e mandato in onda. Il flusso è continuo, come farebbe un
interprete simultaneo.

| Percorso | Ritardo |
|---|---|
| Testo committato (dal parlato) | **~1,5-2,5 s** |
| Sottotitoli WebSocket (regia/monitor) | ~2 s |
| Voce tradotta su HLS | **~5-7 s**, continua |
| Stream “Originale” su HLS | ~3-4 s |

Misurato su un discorso di 11 s (tre frasi): il primo testo utile esce a
**+4,2 s** con la traduzione incrementale, contro **+12,0 s** aspettando la fine
del parlato — e i pezzi successivi arrivano mentre il relatore continua.

Chi ascolta la traduzione con l'audio di sala nelle orecchie sentirà comunque
due cose sfasate: **consiglia gli auricolari** e imposta le aspettative.
Per ridurre ancora: `delivery.hls_live_sync` più basso (meno margine di rete),
`streaming.min_commit_words` più basso (pezzi più corti, traduzione peggiore),
`tts.length_scale: 0.95`, Whisper `medium` invece di `large-v3`.

---

## Resilienza / fault tolerance

- **Watchdog ffmpeg**: ogni stream HLS è sorvegliato; se ffmpeg muore viene
  rilanciato con backoff. I segmenti riprendono con un **prefisso nuovo** e una
  media sequence crescente, così le cache non servono audio vecchio.
- **Stream FLOOR** (`hls.floor: true`): audio originale in HLS senza
  ASR/MT/TTS. Sopravvive a un crash di GPU o modelli: l'audience sente almeno
  la lingua sorgente.
- **Tetto sul backlog TTS** (`hls.max_backlog`): la voce tradotta è spesso più
  lunga dell'originale; senza tetto il ritardo crescerebbe per tutto l'evento.
  Oltre la soglia si scarta l'audio più vecchio (uno stacco ora è meglio di due
  minuti di ritardo alla fine). Visibile in `/healthz` come `backlog_s`.
- **Riconfigurazione a caldo**: avviare/fermare un canale non chiude lo stream
  della scheda audio: gli altri canali non hanno buchi.
- **Client che si ricuce**: la landing riconnette HLS e WebSocket con backoff,
  recupera gli stalli del player, riprende dopo lo schermo bloccato e tiene lo
  schermo acceso (dove supportato).
- **`/healthz`**: stato per-componente (GPU, cattura + freschezza dei blocchi,
  stream HLS, backlog, voci mancanti). **200** se sano/degradato, **503** se
  critico (consegna giù) → usabile per il failover su un nodo di standby.
- **Persistenza**: i canali creati/modificati in regia finiscono in
  `data/channels.json` e tornano dopo un riavvio (il file ha la precedenza su
  `config.yaml`; cancellalo per ripartire dal seme YAML).

Topologia consigliata per ~2500: **2 nodi GPU (primario + hot standby)** con lo
stesso audio, **2 cache nginx** dietro un VIP/keepalived, auto-restart con
systemd/Docker, UPS e rete cablata.

---

## Sicurezza

- La landing è pubblica; **la regia no**: tutte le `/api/admin/*` e il monitor
  WebSocket richiedono il token (`X-Admin-Token`, o `?token=` per il WS).
  Fissalo in `config.yaml` (`admin.token`) o via `IT_ADMIN__TOKEN`.
- In `deploy/nginx.conf` la regia è ristretta anche per **IP** alla subnet di
  regia: in piazza il link gira, e un `POST /stop` a caso durante l'evento non
  è un rischio teorico.
- Le variabili d'ambiente `IT_*` sovrascrivono il YAML
  (`IT_ENGINE__MODE=real`, `IT_ADMIN__TOKEN=…`, `IT_HLS__ENABLED=true`).

---

## Traduzione incrementale (non aspettare la fine della frase)

Mentre il relatore parla, l'audio accumulato viene ri-trascritto ogni secondo.
La parte di testo che si è **stabilizzata** — stesse parole nella stessa
posizione in due passate consecutive, e con abbastanza audio dopo da escludere
che Whisper la stia ancora rivedendo — viene *committata*: tradotta,
sintetizzata e mandata in onda subito. L'audio committato esce dal buffer, così
le passate successive restano corte.

Due dettagli che fanno la differenza tra "funziona" e "funziona bene":

- **Il taglio dell'audio cade in una pausa tra due parole.** I tempi di Whisper
  sono approssimati: tagliando a metà parola, quella dopo si spezza (“Stasera”
  → “Stas” + “sera”) o sparisce del tutto.
- **Si preferisce chiudere su una punteggiatura.** Tradurre mezza frase dà
  risultati peggiori: il modello non vede il seguito. Senza punteggiatura si
  taglia comunque dopo `min_commit_words` parole, altrimenti si aspetterebbe la
  fine del discorso.

```yaml
engine:
  streaming:
    enabled: true
    interval: 1.0            # ogni quanto ri-trascrivere mentre si parla
    stability_margin: 0.5    # audio che deve seguire una parola perché sia definitiva
    min_commit_words: 6      # più basso = meno ritardo, traduzione più a pezzi
    max_pending_seconds: 4.0 # oltre, manda in onda comunque quel che è stabile
    min_cut_gap: 0.08        # pausa minima dove può cadere il taglio
```

**Costo GPU**: ogni canale che sta parlando esegue un'inferenza al secondo su un
buffer di pochi secondi. Con `large-v3` (RTF ~0.09) è circa il 40-50% della GPU
per canale attivo: con più canali che parlano insieme alza `interval` (1.5-2 s)
o passa a `medium`. Con `enabled: false` si torna al comportamento precedente
(una trascrizione a fine frase, meno GPU, molto più ritardo).

## 🗣️ Voce della traduzione simile a quella del microfono

Di default (`tts.match_speaker`) la voce sintetica viene avvicinata a quella del
relatore in tre passi, senza modelli aggiuntivi:

1. **Scelta della voce.** Tra le voci installate per la lingua target si prende
   quella con il registro più vicino: in pratica uomo → voce maschile, donna →
   voce femminile, senza configurare nulla. Serve però che ce ne sia più di una:

   ```bash
   python scripts/download_models.py --pairs --languages it en es fr de
   ```

2. **Altezza.** Il residuo si corregge sintetizzando più lento e ricampionando
   (sposta il pitch a durata invariata). Lo spostamento è limitato a
   `max_pitch_shift` (±25%): una voce tirata fuori dal suo registro suona
   artificiale, e una voce che non somiglia è meglio di una voce ridicola.

3. **Durata.** La traduzione dura circa quanto l'originale (±20%). Oltre alla
   somiglianza, evita che la voce tradotta accumuli ritardo per tutta la serata:
   le lingue non hanno la stessa lunghezza.

L'altezza del parlante si stima dall'audio del microfono (F0 mediana dei tratti
sonori) e si consolida su più enunciati: la voce non cambia a ogni frase. In
regia la colonna “Livello” mostra il valore rilevato (es. `-28 dB · 118 Hz`);
`?` significa che non ci sono ancora abbastanza dati e si usa la voce di default.

```yaml
engine:
  tts:
    match_speaker: true
    max_pitch_shift: 0.25   # ±25% di altezza (circa ±4 semitoni)
    match_duration: true
    max_rate_shift: 0.2     # ±20% di velocità
```

**Cosa questo NON è**: la voce resta quella del modello Piper. È il grado di
somiglianza ottenibile a costo ~zero — la differenza tra “una voce a caso” e
“una voce plausibile per quella persona” — non la voce del relatore.

### Se serve la voce vera del relatore (clonazione)

Si può fare, ma è un'altra categoria di intervento. Due strade:

| | **TTS zero-shot (es. XTTS-v2)** | **Conversione di voce (es. RVC)** |
|---|---|---|
| Come | 6-10 s di riferimento del relatore → sintetizza nella sua voce | modello addestrato sul relatore, converte l'uscita di Piper |
| Preparazione | una clip al soundcheck | minuti di registrazione + addestramento (una volta) |
| Dipendenze | **torch a runtime** (~2,5 GB) + ~2 GB di VRAM | torch + modello per relatore |
| GPU | inferenza sostanziosa, in **competizione** con Whisper e la traduzione incrementale sulla stessa GPU | più leggera, ma sempre in coda sulla GPU |
| Licenza | XTTS-v2 è sotto **CPML: non commerciale** — da verificare per un evento pubblico | dipende dal modello |
| Qualità | timbro riconoscibile, prosodia più piatta dell'originale | timbro molto vicino, richiede audio pulito |

Va misurato sul posto se la GPU regge entrambe le cose: oggi Whisper +
traduzione incrementale usano già il 40-50% della 5070 per canale attivo.
Con 2 nodi (uno per ASR/MT, uno per il TTS) il problema non c'è.

Serve inoltre il **consenso di chi parla**: la sua voce che dice frasi che non
ha pronunciato è un dato personale, e l'evento è pubblico.

## VAD e qualità della trascrizione

In una piazza il livello di fondo non è mai zero e cambia durante l'evento
(PA, applausi, bleed). Il VAD quindi:

- **segue il rumore di fondo** del singolo canale (`vad.mode: adaptive`), con
  soglia a `noise_margin_db` sopra il fondo e isteresi in chiusura;
- tiene un **pre-roll** (`vad.preroll_seconds`) così la prima sillaba non si
  perde;
- si **ricalibra** se il gate resta aperto per un'intera finestra
  `max_utterance_seconds` (segno che il fondo è stabilmente sopra soglia).

Sul testo, i segmenti con alta probabilità di non-parlato o bassa confidenza
vengono scartati, insieme alle tipiche **allucinazioni** di Whisper
(“sottotitoli a cura di…”, “grazie per aver guardato”, annotazioni tipo
`[Musica]`) e alle ripetizioni identiche consecutive. Meglio nessun sottotitolo
che una frase inventata su 2500 telefoni.

Tarature utili in `config.yaml`:

```yaml
engine:
  vad:
    noise_margin_db: 10      # alza se apre sul rumore, abbassa se non apre
    silence_seconds: 0.7     # pausa che chiude la frase (più bassa = più reattivo)
    max_utterance_seconds: 12
    partials: true           # parziali per il monitor di regia
  asr:
    no_speech_threshold: 0.6 # alza per essere più permissivo
    logprob_threshold: -1.0
```

---

## Lingue

Definite in `app/languages.py` con i codici per ciascun motore (Whisper / NLLB /
Piper). Aggiungerne una la rende disponibile sia come lingua sorgente sia come
target. `target_languages` in `config.yaml` limita le lingue scelibili
dall'ascoltatore; `broadcast_languages` per canale limita quelle trasmesse in HLS.

---

## Modalità di consegna a confronto

| | **Bassa latenza (WebSocket)** | **Scala (HLS)** |
|---|---|---|
| Latenza | ~ molto bassa (sub-secondo + pipeline) | qualche secondo (buffer HLS) |
| Ascoltatori | decine (tetto `max_ws_listeners`) | **migliaia** (dietro CDN/nginx) |
| Trasporto | PCM su WebSocket + AudioWorklet | segmenti AAC cacheabili + hls.js |
| Quando | regia, interpreti, monitor, sale piccole | grandi eventi |

Con HLS attivo l'audio PCM su WebSocket è **disabilitato** per default
(`delivery.ws_audio_with_hls: false`): 350 kbps e una connessione persistente
per telefono non sono un'opzione in piazza.

---

## Checklist giorno-evento

**Il giorno prima**

- [ ] `python scripts/audio_probe.py --device "IN 1-8"` → tutti i mic si vedono
- [ ] Gate e tap point (post-EQ/pre-fader) impostati su ogni canale dell'XR18
- [ ] `python scripts/download_models.py --languages …` → nessuna voce mancante
      (`/healthz` → `missing_voices: []`)
- [ ] `admin.token` fissato in `config.yaml`; regia limitata per IP in nginx
- [ ] Prova completa con un telefono Android **e** un iPhone (HLS nativo)
- [ ] Cartella `hls.output_dir` esclusa dall'antivirus (scrive ~1 file/s per stream)

**Un'ora prima**

- [ ] Canali avviati, meter di livello che si muovono su ogni mic
- [ ] `/healthz` → `status: ok`, `capture.stalled: false`, `restarts: 0`
- [ ] QR code / link corto provati dalla rete del pubblico
- [ ] Misura la banda in uscita con qualche client reale
- [ ] UPS collegato, rete cablata, nodo di standby acceso

**Durante**

- [ ] Monitor di regia aperto sul canale principale (verifica il testo)
- [ ] `/healthz` sotto controllo: `backlog_s` in crescita = TTS in ritardo
      (riduci le lingue o `tts.length_scale`)
- [ ] `overflows` in crescita = macchina satura (Whisper più piccolo)

---

## Limiti noti

- **8 canali** simultanei dall'XR18 su Windows (limite del driver WDM/WASAPI;
  vedi sopra per le alternative).
- La voce tradotta resta indietro di ~5-7 s: è simultanea nel flusso, non
  istantanea.
- Tradurre a pezzi costa qualità: il modello non vede il seguito della frase.
  Su lingue con il verbo in fondo (tedesco) si nota di più.
- Una sola GPU: i canali si serializzano sull'inferenza, e la traduzione
  incrementale moltiplica le inferenze per i canali attivi.
- Il conteggio degli ascoltatori in regia riguarda solo i WebSocket: gli
  ascoltatori HLS si contano dai log di nginx.
