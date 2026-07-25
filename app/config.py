"""Configurazione applicazione.

Caricata da ``config.yaml`` (se presente) con fallback ai default. Le
variabili d'ambiente con prefisso ``IT_`` possono sovrascrivere i campi
top-level (es. ``IT_ENGINE__MODE=real``, ``IT_ADMIN__TOKEN=segreto``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import logging

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .naming import normalize_id

log = logging.getLogger("instanttranslator.config")

EngineMode = Literal["mock", "real"]
Device = Literal["cuda", "cpu", "auto"]


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AudioConfig(BaseModel):
    # Sample rate di *elaborazione* (Whisper lavora a 16 kHz). NON è il rate a
    # cui viene aperta la scheda: i mixer digitali (Behringer XR18 & co.) girano
    # a 48 kHz fissi, quindi apriamo al rate nativo e ricampioniamo noi.
    samplerate: int = 16000
    blocksize: int = 1600            # 100 ms a 16 kHz (frame di elaborazione)
    # device di input di default: indice PortAudio *oppure* parte del nome
    # (es. "XR18"). Il nome è più robusto: gli indici cambiano se cambia
    # l'ordine dei device USB tra un riavvio e l'altro.
    default_input_device: int | str | None = None
    # Rate a cui aprire la scheda. None = usa il rate nativo dichiarato dal
    # device (per l'XR18: 48000). Forzalo solo se sai cosa stai facendo.
    device_samplerate: int | None = None
    # Host API preferita quando il device è indicato per nome (una scheda
    # compare una volta per host API): "WASAPI", "WDM-KS", "ASIO", "MME"…
    host_api: str | None = "WASAPI"
    # Buffer PortAudio: "high" = più robusto ai glitch (quello che vogliamo in
    # un evento live, la latenza di cattura è irrilevante rispetto alla pipeline).
    latency: float | str = "high"
    # WASAPI in modalità esclusiva: accesso diretto alla scheda, spesso
    # necessario per ottenere *tutti* i canali di un'interfaccia multicanale.
    wasapi_exclusive: bool = False


class VadConfig(BaseModel):
    """Segmentazione del parlato (quando inizia/finisce un enunciato).

    In una piazza il livello di fondo non è mai zero: PA, applausi, bleed tra
    microfoni aperti. Una soglia fissa o non apre mai (mic a basso guadagno) o
    non chiude mai (fondo alto) → per default la soglia *segue il rumore di
    fondo* del singolo canale.
    """

    mode: Literal["adaptive", "fixed"] = "adaptive"
    energy_threshold: float = 0.012      # soglia in mode=fixed
    noise_margin_db: float = 10.0        # quanto sopra il fondo per aprire
    hysteresis_db: float = 4.0           # quanto sotto la soglia per chiudere
    absolute_floor: float = 0.004        # rms: sotto è silenzio comunque
    noise_halflife: float = 3.0          # s: reattività della stima di fondo
    silence_seconds: float = 0.7         # pausa che chiude l'enunciato
    min_utterance_seconds: float = 0.4
    max_utterance_seconds: float = 12.0
    preroll_seconds: float = 0.3         # audio tenuto prima dell'attacco
    partial_interval: float = 1.5        # cadenza dei parziali
    partials: bool = True                # parziali (solo se qualcuno li guarda)


class StreamingConfig(BaseModel):
    """Traduzione incrementale: non aspettare la fine della frase.

    Mentre il relatore parla, l'audio accumulato viene ri-trascritto a cadenza
    regolare e la parte di testo che si è **stabilizzata** (identica in due
    passate consecutive e con audio successivo che non la modifica più) viene
    *committata*: tradotta, sintetizzata e mandata in onda subito, senza
    aspettare il punto fermo. L'audio committato esce dal buffer, così le
    passate successive restano corte e la GPU non esplode.

    È il compromesso classico dell'interpretariato simultaneo: si guadagnano
    diversi secondi, si perde un po' di qualità perché la traduzione lavora su
    frammenti senza conoscere il seguito della frase.
    """

    enabled: bool = True
    interval: float = 1.0            # ogni quanto ri-trascrivere mentre si parla
    # Una parola si considera definitiva se dopo di lei c'è già questo tanto di
    # audio: significa che Whisper non la sta più rivedendo.
    stability_margin: float = 0.5
    # Sotto questo numero di parole si aspetta ancora (a meno che il pezzo
    # finisca con una punteggiatura): tradurre frammenti troppo corti dà
    # risultati peggiori, perché il modello non vede il seguito della frase.
    min_commit_words: int = 6
    max_pending_seconds: float = 4.0  # oltre, committa comunque quel che c'è
    # Pausa minima tra due parole perché il taglio dell'audio possa cadere lì.
    # Tagliare dove non c'è pausa spezza la parola successiva ("Stasera" ->
    # "Stas" + "sera") perché i tempi di Whisper sono approssimati.
    min_cut_gap: float = 0.08
    # Quanto arretrare il taglio quando non c'è nessuna parola dopo (fine del
    # buffer): evita di mangiare l'attacco di quella che deve ancora arrivare.
    trim_back_seconds: float = 0.25


class ASRConfig(BaseModel):
    model: str = "large-v3"          # tiny/base/small/medium/large-v3
    device: Device = "auto"
    compute_type: str = "float16"    # float16/int8_float16/int8 su GPU
    beam_size: int = 1               # 1 = greedy, più veloce per il real-time
    # Filtri anti-allucinazione: su audio non-parlato Whisper "inventa" frasi
    # (tipicamente ringraziamenti e sigle da sottotitoli). Scartiamo i segmenti
    # con alta probabilità di non-parlato o bassa confidenza.
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    drop_repeats: bool = True        # scarta la ripetizione identica consecutiva


class TranslateConfig(BaseModel):
    # Cartella del modello NLLB convertito in CTranslate2 (vedi download_models.py)
    model_dir: str = "models/nllb-200-distilled-600M-ct2"
    tokenizer: str = "facebook/nllb-200-distilled-600M"
    device: Device = "auto"
    compute_type: str = "float16"
    beam_size: int = 2
    max_decoding_length: int = 256


class TTSConfig(BaseModel):
    voices_dir: str = "models/piper"     # cartella con i .onnx + .onnx.json
    # Piper su CPU è già real-time e tiene la GPU libera per ASR/MT.
    # use_cuda=True richiede onnxruntime-gpu installato.
    use_cuda: bool = False
    samplerate: int = 22050              # tipico per le voci Piper medium
    length_scale: float = 1.0            # >1 più lento, <1 più veloce

    # Avvicina la voce sintetica a quella del microfono: sceglie tra le voci
    # installate quella con il registro più vicino e corregge il residuo di
    # altezza. Non è clonazione (per quella serve un modello zero-shot), ma è
    # la differenza tra "una voce a caso" e "una voce plausibile".
    match_speaker: bool = True
    # Quanto si può spostare l'altezza (0.25 = ±25%, circa ±4 semitoni).
    # Oltre, una voce tirata fuori dal suo registro suona artificiale.
    max_pitch_shift: float = 0.25
    # Fa durare la traduzione quanto l'originale: oltre alla somiglianza,
    # impedisce che la voce tradotta accumuli ritardo per tutta la serata.
    match_duration: bool = True
    max_rate_shift: float = 0.2          # ±20% sulla velocità del parlato


class EngineConfig(BaseModel):
    mode: EngineMode = "mock"
    asr: ASRConfig = Field(default_factory=ASRConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


class HlsConfig(BaseModel):
    """Consegna scalabile (CDN-ready) per grandi platee.

    Quando attivo, ogni canale in onda trasmette in HLS le sue
    ``broadcast_languages`` (o ``target_languages`` globali). I file vengono
    serviti come statici e messi in cache da nginx/CDN: l'origine genera ogni
    stream una sola volta, indipendentemente dal numero di ascoltatori.
    """

    enabled: bool = False
    ffmpeg: str = "ffmpeg"               # path o comando in PATH
    output_dir: str = "data/hls"
    segment_time: float = 1.0            # durata segmenti (s); più corto = meno latenza
    list_size: int = 10                  # segmenti nella playlist
    # Segmenti tenuti su disco *oltre* la playlist: un telefono con rete lenta
    # può chiedere un segmento già uscito dalla playlist. Con 1 s di segmento,
    # 30 = ~30 s di grazia. Costa pochi MB.
    delete_threshold: int = 30
    audio_codec: str = "aac"             # aac = massima compatibilità (iOS incluso)
    audio_bitrate: str = "64k"
    output_samplerate: int = 44100       # 44100 per compatibilità player
    pacer_tick: float = 0.1              # granularità di scrittura verso ffmpeg
    # Il TTS può produrre più audio del tempo reale (la voce tradotta è spesso
    # più lunga dell'originale): senza un tetto il ritardo cresce per tutto
    # l'evento. Oltre questo backlog si scarta l'audio più vecchio.
    max_backlog: float = 4.0
    subtitle_window: float = 30.0        # secondi di cue mantenuti in subs.json
    subtitle_offset: float = 0.0         # ritocco fine sync sottotitoli (s)
    # Stream "FLOOR": audio originale del canale in HLS, senza ASR/MT/TTS.
    # Sopravvive a un crash di GPU/modelli: fallback di ultima istanza.
    floor: bool = True


class DeliveryConfig(BaseModel):
    """Come arriva l'audio agli ascoltatori (e quanto carico regge l'origine)."""

    # Audio PCM su WebSocket: ~350 kbps *per ascoltatore* e una connessione
    # persistente sul processo Python. Ottimo per la regia e le sale piccole,
    # da non usare per una piazza: con HLS attivo è disabilitato di default.
    ws_audio: bool = True
    ws_audio_with_hls: bool = False
    # Tetto di sicurezza sulle connessioni WebSocket accettate dall'origine.
    max_ws_listeners: int = 200
    # Cadenza con cui il telefono scarica i sottotitoli in modalità HLS. I cue
    # hanno tempi assoluti e arrivano in anticipo sull'audio: 3 s non si vedono.
    subtitle_poll_seconds: float = 3.0
    # Quanto indietro rispetto al live si posiziona il player HLS. Più basso =
    # meno ritardo ma meno margine su una rete che perde colpi. Con segmenti da
    # 1 s, 2.0 è un buon compromesso in piazza.
    hls_live_sync: float = 2.0
    # Lingue mostrate all'ascoltatore quando HLS è attivo:
    #   "all"       = tutte le target. Quelle senza audio in onda vengono
    #                 offerte **con i soli sottotitoli**, che costano solo una
    #                 traduzione (niente TTS né ffmpeg) e si servono dalla
    #                 stessa cache dei sottotitoli: scalano come l'audio.
    #   "broadcast" = solo quelle con audio in onda.
    audience_languages: Literal["all", "broadcast"] = "all"
    # Secondi di cache dichiarati su /api/info (nginx/CDN lo rispettano).
    info_cache_seconds: int = 5


class AdminConfig(BaseModel):
    """Accesso alla regia. In una piazza il link gira: la regia va protetta."""

    require_auth: bool = True
    # None = token casuale generato all'avvio e scritto nel log.
    token: str | None = None


class ChannelConfig(BaseModel):
    """Configurazione iniziale di un canale (modificabile a runtime via regia)."""

    id: str
    name: str = "Canale"
    description: str = ""
    source_language: str = "it"      # lingua parlata sul canale
    # indice device PortAudio o parte del nome (None = default globale)
    input_device: int | str | None = None
    channel_index: int = 0           # indice del canale dentro lo stream multicanale
    # Guadagno digitale applicato in cattura: comodo per allineare un mic
    # debole senza toccare il mixer. 0 = nessuna modifica.
    gain_db: float = 0.0
    # Canale usato per lo stream "Originale" (FLOOR). None = lo stesso dell'ASR.
    # Su un mixer conviene mandarci un bus/mix (es. il main LR), non un mic solo.
    floor_channel_index: int | None = None
    enabled: bool = True
    # Lingue trasmesse in HLS per questo canale (None = usa target_languages globali).
    broadcast_languages: list[str] | None = None

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        """L'id finisce in URL, path su disco e chiavi di cache: va normalizzato."""
        clean = normalize_id(value)
        if clean != value:
            log.warning(
                "id canale '%s' normalizzato in '%s' (serve per URL HLS e cache)",
                value, clean,
            )
        return clean


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    hls: HlsConfig = Field(default_factory=HlsConfig)
    delivery: DeliveryConfig = Field(default_factory=DeliveryConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)

    # Cartella di stato scrivibile (canali salvati dalla regia, ecc.).
    state_dir: str = "data"

    # Lingue target abilitate (codici dal registro languages.py). Vuoto = tutte.
    target_languages: list[str] = Field(
        default_factory=lambda: ["it", "en", "es", "fr", "de"]
    )

    # Canali predefiniti all'avvio.
    channels: list[ChannelConfig] = Field(default_factory=list)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings,
        file_secret_settings,
    ):
        # L'ambiente vince sul YAML: così lo stesso config.yaml gira su nodo
        # primario e standby cambiando solo una variabile
        # (IT_ENGINE__MODE, IT_ADMIN__TOKEN, IT_HLS__ENABLED…).
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    # -- helper ---------------------------------------------------------------
    @property
    def ws_audio_allowed(self) -> bool:
        """Se l'audio PCM su WebSocket è servibile agli ascoltatori."""
        if not self.delivery.ws_audio:
            return False
        if self.hls.enabled and not self.delivery.ws_audio_with_hls:
            return False
        return True


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Carica la config da YAML con fallback ai default + override da env."""
    path = Path(path)
    data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    return AppConfig(**data)
