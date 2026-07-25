import { fetchJSON, wsURL, el, toast } from './common.js';

// Client ascoltatore. Gira su telefoni qualsiasi, su una rete di piazza che
// perde colpi: tutto ciò che si connette deve saper tornare su da solo, e il
// consumo (CPU/batteria) deve restare basso per un evento di ore.

const state = {
  info: null,
  channel: null,
  lang: null,
  mode: 'ws',          // 'ws' (bassa latenza) | 'hls' (scala)
  // WebSocket
  ws: null, wsRetry: 0, wsTimer: null,
  audioCtx: null, node: null, gain: null,
  // HLS
  hls: null, nativeHls: false, audioEl: null, hlsRetry: 0, hlsTimer: null,
  subTimer: null, syncTimer: null, watchdog: null,
  hlsCues: [], lastProgress: 0, lastTime: -1,
  renderKey: '',
  // comune
  audioEnabled: false,
  samplerate: 22050,
  pollSeconds: 3,
  liveSync: 2,
  wakeLock: null,
  finals: [], partial: null,
};

const MAX_LINES = 8;          // righe di sottotitolo tenute a schermo
const RETRY_MS = [1000, 2000, 4000, 8000, 15000];
const retryDelay = (n) => RETRY_MS[Math.min(n, RETRY_MS.length - 1)];

const views = {
  channels: document.getElementById('view-channels'),
  languages: document.getElementById('view-languages'),
  live: document.getElementById('view-live'),
};
function show(view) {
  for (const [k, n] of Object.entries(views)) n.classList.toggle('hidden', k !== view);
}

// ---------------------------------------------------------------- caricamento
async function load() {
  try {
    state.info = await fetchJSON('/api/info');
  } catch (err) {
    toast('Errore nel caricamento: ' + err.message, true);
    setTimeout(load, 3000);
    return;
  }
  state.samplerate = state.info.samplerate || 22050;
  state.pollSeconds = state.info.subtitle_poll_seconds || 3;
  state.liveSync = state.info.hls_live_sync || 2;
  const banner = document.getElementById('banner');
  if (state.info.engine_mode === 'mock') {
    banner.textContent = '⚙️ Modalità DEMO: sorgenti e traduzioni simulate, audio a tono di prova.';
    banner.classList.remove('hidden');
    banner.classList.add('mock');
  }
  renderChannels();
}

function renderChannels() {
  const wrap = document.getElementById('channels');
  wrap.innerHTML = '';
  const channels = state.info.channels || [];
  document.getElementById('no-channels').classList.toggle('hidden', channels.length > 0);
  for (const ch of channels) {
    const src = langInfo(ch.source_language);
    const status = ch.running
      ? el('span', { class: 'badge live' }, [el('span', { class: 'dot' }), 'In onda'])
      : el('span', { class: 'badge off' }, [el('span', { class: 'dot' }), 'Non attivo']);
    wrap.appendChild(el('div', { class: 'card', onclick: () => selectChannel(ch) }, [
      el('div', { class: 'flag' }, src ? src.flag : '🎙️'),
      el('div', { class: 'name' }, ch.name),
      el('div', { class: 'desc' }, ch.description || '—'),
      el('div', { class: 'meta' }, [
        status,
        el('span', { class: 'badge' }, `Originale: ${src ? src.name : ch.source_language}`),
      ]),
    ]));
  }
  show('channels');
}

function channelLanguages(ch) {
  // L'origine dice quali lingue sono realmente servibili su questo canale.
  return ch.languages && ch.languages.length ? ch.languages : (state.info.target_languages || []);
}

function selectChannel(ch) {
  state.channel = ch;
  document.getElementById('lang-intro').textContent =
    `${ch.name} · lingua originale: ${(langInfo(ch.source_language) || {}).name || ch.source_language}`;
  const wrap = document.getElementById('languages');
  wrap.innerHTML = '';

  // Stream FLOOR: audio originale del canale (fallback, sopravvive a guasti GPU).
  if (ch.hls_floor) {
    wrap.appendChild(el('div', { class: 'lang-card', onclick: () => selectLanguage(FLOOR_LANG_INFO) }, [
      el('span', { class: 'flag' }, FLOOR_LANG_INFO.flag),
      el('div', {}, [
        el('div', { class: 'label' }, FLOOR_LANG_INFO.name),
        el('div', { class: 'sub' }, 'senza traduzione'),
      ]),
    ]));
  }

  const langs = channelLanguages(ch);
  for (const l of langs) {
    // Diciamo subito cosa si ottiene: scoprire dopo il tocco che una lingua
    // non ha la voce è il tipo di sorpresa che genera code all'infopoint.
    const solotesto = l.mode === 'text';
    wrap.appendChild(el('div', { class: 'lang-card', onclick: () => selectLanguage(l) }, [
      el('span', { class: 'flag' }, l.flag),
      el('div', {}, [
        el('div', { class: 'label' }, l.name),
        el('div', { class: 'sub' }, solotesto ? '📝 solo sottotitoli' : '🔊 voce + sottotitoli'),
      ]),
    ]));
  }
  if (langs.length === 0 && !ch.hls_floor) {
    wrap.appendChild(el('p', { class: 'banner' }, 'Nessuna lingua disponibile su questo canale.'));
  }
  show('languages');
}

function selectLanguage(l) {
  state.lang = l;
  // Il server dice come va servita questa lingua: audio in onda (hls),
  // bassa latenza (ws) o soli sottotitoli (text).
  const hlsLangs = state.channel.hls_languages || [];
  if (l.code === FLOOR_LANG_INFO.code) state.mode = 'hls';
  else if (l.mode) state.mode = l.mode;
  else state.mode = (state.info.hls_enabled && hlsLangs.includes(l.code)) ? 'hls' : 'ws';

  document.getElementById('live-title').textContent = state.channel.name;
  document.getElementById('live-sub').textContent =
    `${l.flag} ${l.name} · originale ${(langInfo(state.channel.source_language) || {}).name || ''}`;
  const modeBadge = document.getElementById('live-mode');
  modeBadge.textContent = { hls: '📡 Voce tradotta', ws: '⚡ Bassa latenza',
                            text: '📝 Solo sottotitoli' }[state.mode] || '';

  state.finals = []; state.partial = null; state.hlsCues = [];
  state.hlsRetry = 0; state.wsRetry = 0; state.renderKey = '';
  resetAudioUI();
  renderSubtitles();
  show('live');
  requestWakeLock();

  if (state.mode === 'hls') connectHLS();
  else if (state.mode === 'text') connectText();
  else connectWS();
}

// Solo sottotitoli: nessun audio da scaricare, solo il JSON cacheabile dei
// cue. È il livello che permette di offrire tutte le lingue senza un TTS e un
// encoder per ciascuna.
function connectText() {
  const btn = document.getElementById('audio-toggle');
  btn.classList.add('hidden');
  document.getElementById('audio-hint').textContent =
    'Questa lingua è disponibile con i sottotitoli.';
  setStatus(true, 'In onda');
  state.subTimer = setInterval(pollSubtitles, Math.max(state.pollSeconds, 1) * 1000);
  state.syncTimer = setInterval(syncHLSSubtitles, 250);
  pollSubtitles();
}

function resetAudioUI() {
  state.audioEnabled = false;
  const btn = document.getElementById('audio-toggle');
  btn.textContent = '▶︎ Attiva audio';
  btn.classList.remove('hidden');
  document.getElementById('audio-hint').textContent = "L'audio richiede un tocco per partire.";
}

// ====================================================================== HLS ==
function hlsURL() {
  return `/hls/${encodeURIComponent(state.channel.id)}/${encodeURIComponent(state.lang.code)}/audio.m3u8`;
}

function connectHLS() {
  const url = hlsURL();
  const audio = document.getElementById('hls-audio');
  state.audioEl = audio;
  audio.volume = document.getElementById('volume').value / 100;
  state.lastProgress = Date.now();
  state.lastTime = -1;

  if (window.Hls && window.Hls.isSupported()) {
    const hls = new window.Hls({
      lowLatencyMode: true,
      liveSyncDuration: state.liveSync,
      liveMaxLatencyDuration: state.liveSync * 4,
      // Se il player scivola indietro (buffering, schermo bloccato) recupera
      // accelerando impercettibilmente, invece di accumulare ritardo per ore.
      maxLiveSyncPlaybackRate: 1.1,
      backBufferLength: 30,
      manifestLoadingMaxRetry: 6,
      levelLoadingMaxRetry: 6,
      fragLoadingMaxRetry: 6,
    });
    state.hls = hls;
    state.nativeHls = false;
    hls.loadSource(url);
    hls.attachMedia(audio);
    hls.on(window.Hls.Events.ERROR, (_e, data) => {
      if (!data.fatal) return;
      const T = window.Hls.ErrorTypes;
      if (data.type === T.MEDIA_ERROR) {
        // Buffer inconsistente: hls.js sa rimettersi in piedi da solo.
        setStatus(false, 'Recupero audio…');
        try { hls.recoverMediaError(); return; } catch (_) { /* cade nel retry */ }
      }
      scheduleHlsRetry(data.type === T.NETWORK_ERROR ? 'Rete instabile…' : 'Riconnessione…');
    });
    hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
      state.hlsRetry = 0;
      setStatus(true, 'In onda');
      if (state.audioEnabled) audio.play().catch(() => {});
    });
  } else if (audio.canPlayType('application/vnd.apple.mpegurl')) {
    // Safari/iOS: HLS nativo (nessun MSE). Il sync dei sottotitoli usa
    // getStartDate(), che Safari popola da EXT-X-PROGRAM-DATE-TIME.
    state.nativeHls = true;
    audio.src = url;
    audio.onerror = () => scheduleHlsRetry('Riconnessione…');
    audio.onplaying = () => { state.hlsRetry = 0; setStatus(true, 'In onda'); };
    audio.load();
    setStatus(true, 'In onda');
  } else {
    setStatus(false, 'HLS non supportato');
    toast('Il browser non supporta HLS', true);
    return;
  }

  state.subTimer = setInterval(pollSubtitles, Math.max(state.pollSeconds, 1) * 1000);
  state.syncTimer = setInterval(syncHLSSubtitles, 250);
  state.watchdog = setInterval(watchHLS, 3000);
  pollSubtitles();
}

function scheduleHlsRetry(label) {
  if (state.hlsTimer) return;               // un tentativo alla volta
  const delay = retryDelay(state.hlsRetry++);
  setStatus(false, label);
  state.hlsTimer = setTimeout(() => {
    state.hlsTimer = null;
    if (state.mode !== 'hls') return;
    teardownHLS();
    connectHLS();
    if (state.audioEnabled && state.audioEl) state.audioEl.play().catch(() => {});
  }, delay);
}

function watchHLS() {
  const audio = state.audioEl;
  if (!audio || !state.audioEnabled || audio.paused) return;
  // La riproduzione non avanza da troppo tempo: stream fermo o rete caduta.
  if (audio.currentTime !== state.lastTime) {
    state.lastTime = audio.currentTime;
    state.lastProgress = Date.now();
    return;
  }
  if (Date.now() - state.lastProgress > 10000) scheduleHlsRetry('Stream fermo, riprovo…');
}

async function pollSubtitles() {
  if (document.hidden) return;              // schermo spento: non consumare rete
  try {
    const data = await fetchJSON(
      `/api/hls/${encodeURIComponent(state.channel.id)}/${encodeURIComponent(state.lang.code)}/subs.json`);
    state.hlsCues = data.cues || [];
  } catch (_) { /* transitorio */ }
}

function playheadEpoch() {
  // Senza audio attivo (telefono muto, o chi vuole solo leggere) non c'è un
  // playhead: mostriamo i cue appena il loro istante d'onda arriva, così la
  // pagina resta utile come display di soli sottotitoli.
  if (!state.audioEnabled) return Date.now() / 1000;
  if (state.hls && state.hls.playingDate) return state.hls.playingDate.getTime() / 1000;
  const audio = state.audioEl;
  if (state.nativeHls && audio && !audio.paused) {
    // Safari: orario assoluto d'inizio dello stream + posizione corrente.
    if (typeof audio.getStartDate === 'function') {
      const start = audio.getStartDate();
      const ms = start ? start.getTime() : NaN;
      if (!Number.isNaN(ms)) return ms / 1000 + audio.currentTime;
    }
    return Date.now() / 1000 - 8;           // ultima spiaggia: stima del buffer
  }
  return null;
}

function syncHLSSubtitles() {
  const now = playheadEpoch();
  if (now == null) return;
  const box = document.getElementById('subtitles');
  const aired = state.hlsCues
    .filter((c) => c.start <= now + 0.15)
    .sort((a, b) => a.start - b.start)
    .slice(-MAX_LINES);
  if (aired.length === 0) {
    if (state.renderKey !== 'empty') {
      state.renderKey = 'empty';
      box.innerHTML = '<p class="empty">In attesa del parlato…</p>';
    }
    return;
  }
  // Ridisegna solo quando cambia davvero qualcosa: con 5 refresh al secondo
  // per ore, ricostruire il DOM ogni volta si sente sulla batteria.
  const active = aired[aired.length - 1];
  const key = aired.map((c) => c.seq).join(',') + '|' + (active.end >= now ? '1' : '0');
  if (key === state.renderKey) return;
  state.renderKey = key;
  box.innerHTML = '';
  for (const c of aired) {
    const isActive = c.start <= now && now <= c.end;
    box.appendChild(el('p', { class: isActive ? 'partial' : 'final' }, c.text));
  }
  box.scrollTop = box.scrollHeight;
}

function teardownHLS() {
  for (const t of ['subTimer', 'syncTimer', 'watchdog', 'hlsTimer']) {
    if (state[t]) { clearInterval(state[t]); clearTimeout(state[t]); state[t] = null; }
  }
  if (state.hls) { try { state.hls.destroy(); } catch (_) {} state.hls = null; }
  if (state.audioEl) {
    const a = state.audioEl;
    a.onerror = null; a.onplaying = null;
    try { a.pause(); a.removeAttribute('src'); a.load(); } catch (_) {}
  }
  state.nativeHls = false;
}

// ================================================================ WebSocket ==
function connectWS() {
  let ws;
  try {
    ws = new WebSocket(wsURL(
      `/api/listen/${encodeURIComponent(state.channel.id)}/${encodeURIComponent(state.lang.code)}`));
  } catch (_) {
    scheduleWsRetry();
    return;
  }
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  ws.onopen = () => { state.wsRetry = 0; };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') handleWSMessage(JSON.parse(ev.data));
    else if (state.audioEnabled && state.node) state.node.port.postMessage(ev.data, [ev.data]);
  };
  ws.onclose = (ev) => {
    state.ws = null;
    if (ev.code === 4429) {                 // tetto ascoltatori sull'origine
      setStatus(false, 'Servizio pieno');
      toast('Troppi ascoltatori in bassa latenza: riprova tra poco', true);
      return;
    }
    if (ev.code === 4404) { setStatus(false, 'Canale non disponibile'); return; }
    setStatus(false, 'Riconnessione…');
    scheduleWsRetry();
  };
  ws.onerror = () => setStatus(false, 'Errore connessione');
}

function scheduleWsRetry() {
  if (state.wsTimer || state.mode !== 'ws') return;
  state.wsTimer = setTimeout(() => {
    state.wsTimer = null;
    if (state.mode === 'ws') connectWS();
  }, retryDelay(state.wsRetry++));
}

function handleWSMessage(msg) {
  if (msg.type === 'hello') {
    state.samplerate = msg.samplerate || state.samplerate;
    setStatus(msg.running, msg.running ? 'In onda' : 'Canale non attivo');
    if (msg.audio === false) {
      document.getElementById('audio-hint').textContent =
        'Su questo canale sono disponibili i sottotitoli.';
      document.getElementById('audio-toggle').classList.add('hidden');
    }
  } else if (msg.type === 'subtitle') {
    if (msg.final) {
      state.finals.push({ seq: msg.seq, text: msg.text });
      while (state.finals.length > MAX_LINES) state.finals.shift();
      if (state.partial && state.partial.seq === msg.seq) state.partial = null;
    } else {
      state.partial = { seq: msg.seq, text: msg.text };
    }
    renderSubtitles();
  }
}

function teardownWS() {
  if (state.wsTimer) { clearTimeout(state.wsTimer); state.wsTimer = null; }
  if (state.ws) { state.ws.onclose = null; state.ws.close(); state.ws = null; }
  if (state.audioCtx) { try { state.audioCtx.close(); } catch (_) {} state.audioCtx = null; }
  state.node = null; state.gain = null;
}

function renderSubtitles() {
  const box = document.getElementById('subtitles');
  if (state.finals.length === 0 && !state.partial) {
    box.innerHTML = '<p class="empty">In attesa del parlato…</p>';
    return;
  }
  box.innerHTML = '';
  for (const f of state.finals) box.appendChild(el('p', { class: 'final' }, f.text));
  if (state.partial) box.appendChild(el('p', { class: 'partial' }, state.partial.text));
  box.scrollTop = box.scrollHeight;
}

// ===================================================================== audio ==
async function enableAudio() {
  if (state.audioEnabled) return;
  if (state.mode === 'hls') {
    try {
      state.audioEl.muted = false;
      await state.audioEl.play();
      state.audioEnabled = true;
      document.getElementById('audio-toggle').textContent = '🔊 Audio attivo';
      document.getElementById('audio-hint').textContent = 'Audio attivo (HLS).';
      setupMediaSession();
    } catch (err) { toast('Audio non disponibile: ' + err.message, true); }
    return;
  }
  // WS: AudioContext + worklet PCM.
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    let ctx;
    try { ctx = new Ctx({ sampleRate: state.samplerate }); } catch (_) { ctx = new Ctx(); }
    await ctx.audioWorklet.addModule('/static/js/pcm-player.js');
    const node = new AudioWorkletNode(ctx, 'pcm-player', { numberOfInputs: 0, outputChannelCount: [1] });
    const gain = ctx.createGain();
    gain.gain.value = document.getElementById('volume').value / 100;
    node.connect(gain).connect(ctx.destination);
    await ctx.resume();
    state.audioCtx = ctx; state.node = node; state.gain = gain; state.audioEnabled = true;
    document.getElementById('audio-toggle').textContent = '🔊 Audio attivo';
    document.getElementById('audio-hint').textContent = 'Audio attivo.';
  } catch (err) { toast('Audio non disponibile: ' + err.message, true); }
}

function setupMediaSession() {
  // Così sul lock screen si vede cosa si sta ascoltando (e i tasti funzionano).
  if (!('mediaSession' in navigator) || !window.MediaMetadata) return;
  try {
    navigator.mediaSession.metadata = new window.MediaMetadata({
      title: `${state.lang.flag} ${state.lang.name}`,
      artist: state.channel.name,
      album: 'Traduzione live',
    });
  } catch (_) { /* opzionale */ }
}

async function requestWakeLock() {
  // Leggere sottotitoli con lo schermo che si spegne ogni 30 s è inutilizzabile.
  if (!('wakeLock' in navigator) || state.wakeLock) return;
  try {
    state.wakeLock = await navigator.wakeLock.request('screen');
    state.wakeLock.addEventListener('release', () => { state.wakeLock = null; });
  } catch (_) { /* non supportato: pazienza */ }
}

function releaseWakeLock() {
  if (state.wakeLock) { try { state.wakeLock.release(); } catch (_) {} state.wakeLock = null; }
}

function teardown() {
  teardownWS();
  teardownHLS();
  releaseWakeLock();
  resetAudioUI();
}

function setStatus(live, label) {
  const badge = document.getElementById('live-status');
  badge.className = 'badge ' + (live ? 'live' : 'off');
  badge.lastElementChild.textContent = label;
}

// ------------------------------------------------------------------- controlli
document.getElementById('back-to-channels').onclick = () => { teardown(); show('channels'); };
document.getElementById('back-from-live').onclick = () => { teardown(); selectChannel(state.channel); };
document.getElementById('audio-toggle').onclick = enableAudio;
document.getElementById('volume').oninput = (e) => {
  const v = e.target.value / 100;
  if (state.gain) state.gain.gain.value = v;
  if (state.audioEl) state.audioEl.volume = v;
};

// Ritorno da schermo bloccato / app in background: riprendi tutto.
document.addEventListener('visibilitychange', () => {
  if (document.hidden || !views.live || views.live.classList.contains('hidden')) return;
  requestWakeLock();
  if (state.mode === 'hls') {
    pollSubtitles();
    if (state.audioEnabled && state.audioEl && state.audioEl.paused) {
      state.audioEl.play().catch(() => {});
    }
    // Rientrando, il player è indietro rispetto al live: riallinealo.
    if (state.hls && state.hls.liveSyncPosition) {
      const target = state.hls.liveSyncPosition;
      if (state.audioEl && Math.abs(state.audioEl.currentTime - target) > 5) {
        try { state.audioEl.currentTime = target; } catch (_) {}
      }
    }
  } else if (!state.ws) {
    connectWS();
  }
});

const FLOOR_LANG_INFO = { code: 'orig', name: '🎤 Originale', english_name: 'Original audio', flag: '🎤' };

function langInfo(code) {
  if (code === 'orig') return FLOOR_LANG_INFO;
  const all = [];
  for (const ch of (state.info.channels || [])) all.push(...(ch.languages || []));
  all.push(...(state.info.target_languages || []));
  return all.find((l) => l.code === code) || ALL_LANG[code];
}
const ALL_LANG = {
  it: { code: 'it', name: 'Italiano', flag: '🇮🇹' }, en: { code: 'en', name: 'English', flag: '🇬🇧' },
  es: { code: 'es', name: 'Español', flag: '🇪🇸' }, fr: { code: 'fr', name: 'Français', flag: '🇫🇷' },
  de: { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
};

window.addEventListener('beforeunload', teardown);
load();
