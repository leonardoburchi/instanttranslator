import { fetchJSON, wsURL, el, toast } from './common.js';

// Regia. Protetta da token: viene chiesto una volta e tenuto in localStorage.

const TOKEN_KEY = 'it_admin_token';

const state = {
  info: null,
  devices: [],
  languages: [],
  channels: [],
  monitorWs: null,
  monitorId: null,
  token: localStorage.getItem(TOKEN_KEY) || '',
};

// ------------------------------------------------------------------ auth utils
function authHeaders(extra = {}) {
  return state.token ? { ...extra, 'X-Admin-Token': state.token } : extra;
}

async function api(url, opts = {}) {
  const o = { ...opts, headers: authHeaders(opts.headers || {}) };
  try {
    return await fetchJSON(url, o);
  } catch (err) {
    if (/401|token/i.test(err.message)) {
      askToken('Token regia non valido.');
      throw new Error('non autorizzato');
    }
    throw err;
  }
}

function askToken(message) {
  const box = document.getElementById('auth-gate');
  document.getElementById('auth-msg').textContent = message || '';
  box.classList.remove('hidden');
  document.getElementById('main-ui').classList.add('hidden');
}

document.getElementById('auth-btn').onclick = () => {
  const value = document.getElementById('auth-token').value.trim();
  if (!value) return;
  state.token = value;
  localStorage.setItem(TOKEN_KEY, value);
  document.getElementById('auth-gate').classList.add('hidden');
  document.getElementById('main-ui').classList.remove('hidden');
  load();
};

// ---------------------------------------------------------------- caricamento
async function load() {
  try {
    [state.info, state.devices] = await Promise.all([
      api('/api/admin/info'),
      api('/api/admin/devices'),
    ]);
  } catch (err) {
    if (err.message !== 'non autorizzato') toast('Errore caricamento: ' + err.message, true);
    return;
  }
  state.languages = state.info.languages;

  const banner = document.getElementById('banner');
  if (state.info.engine_mode === 'mock') {
    banner.textContent = '⚙️ Modalità DEMO (engine.mode = mock). I device audio reali non vengono usati.';
    banner.classList.remove('hidden');
    banner.classList.add('mock');
  }
  const apis = (state.info.audio && state.info.audio.host_apis) || [];
  document.getElementById('audio-note').textContent = apis.length
    ? `Host API disponibili: ${apis.join(', ')} · elaborazione a ${state.info.audio.samplerate} Hz`
    : '';

  fillSelect(document.getElementById('new-lang'), languageOptions());
  fillSelect(document.getElementById('new-device'), deviceOptions());

  // Mostra i controlli HLS solo se la consegna scalabile è attiva.
  if (state.info.hls_enabled) {
    document.getElementById('th-hls').classList.remove('hidden');
    for (const f of document.querySelectorAll('.hls-field')) f.classList.remove('hidden');
  }

  await refreshChannels();
  setInterval(refreshStatus, 3000);
  setInterval(refreshLevels, 500);
}

function parseLangCodes(value) {
  const codes = (value || '').split(',').map((s) => s.trim()).filter(Boolean);
  return codes.length ? codes : null;
}

function languageOptions() {
  return state.languages.map((l) => ({ value: l.code, label: `${l.flag} ${l.english_name}` }));
}
function deviceOptions() {
  const opts = [{ value: '', label: 'Default di sistema' }];
  for (const d of state.devices) {
    opts.push({ value: String(d.index), label: `[${d.index}] ${d.name} (${d.max_input_channels}ch · ${d.hostapi})` });
  }
  return opts;
}
function fillSelect(sel, options, selected) {
  sel.innerHTML = '';
  for (const o of options) {
    const opt = el('option', { value: o.value }, o.label);
    if (String(o.value) === String(selected)) opt.selected = true;
    sel.appendChild(opt);
  }
}

// ------------------------------------------------------------------- tabella
async function refreshChannels() {
  state.channels = await api('/api/admin/channels');
  renderTable();
}

function renderTable() {
  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  for (const ch of state.channels) tbody.appendChild(renderRow(ch));
}

function renderRow(ch) {
  const nameI = el('input', { type: 'text', value: ch.name });
  const descI = el('input', { type: 'text', value: ch.description });
  const langS = el('select');
  fillSelect(langS, languageOptions(), ch.source_language);
  const devS = el('select');
  fillSelect(devS, deviceOptions(), ch.input_device === null ? '' : ch.input_device);
  const idxI = el('input', { type: 'number', min: '0', value: ch.channel_index });
  const gainI = el('input', { type: 'number', step: '1', value: ch.gain_db, style: 'width:64px' });
  const hlsI = el('input', {
    type: 'text', placeholder: 'tutte',
    value: (ch.broadcast_languages || []).join(','),
  });

  const status = el('span', { class: 'badge ' + (ch.running ? 'live' : 'off') }, [
    el('span', { class: 'dot' }), ch.running ? 'In onda' : 'Fermo',
  ]);
  const listeners = el('span', {}, String(ch.listeners));
  // Meter: barra + valore in dBFS, aggiornata ogni 500 ms.
  const meterBar = el('div', { class: 'meter-fill' });
  const meterText = el('span', { class: 'meter-text' }, '—');
  const meter = el('div', { class: 'meter' }, [meterBar, meterText]);

  const startStop = el('button', {
    class: 'btn small ' + (ch.running ? 'danger' : 'primary'),
    onclick: () => toggleChannel(ch.id, !ch.running),
  }, ch.running ? 'Stop' : 'Start');

  const save = el('button', {
    class: 'btn small',
    onclick: async () => {
      try {
        await api(`/api/admin/channels/${encodeURIComponent(ch.id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: nameI.value,
            description: descI.value,
            source_language: langS.value,
            input_device: devS.value === '' ? null : parseInt(devS.value, 10),
            channel_index: parseInt(idxI.value, 10) || 0,
            gain_db: parseFloat(gainI.value) || 0,
            broadcast_languages: parseLangCodes(hlsI.value),
          }),
        });
        toast('Canale salvato');
        await refreshChannels();
      } catch (err) { toast('Errore: ' + err.message, true); }
    },
  }, 'Salva');

  const monitor = el('button', {
    class: 'btn small', onclick: () => startMonitor(ch.id, ch.name),
  }, 'Monitor');

  const del = el('button', {
    class: 'btn small danger',
    onclick: async () => {
      if (!confirm(`Eliminare il canale "${ch.name}"?`)) return;
      try {
        await api(`/api/admin/channels/${encodeURIComponent(ch.id)}`, { method: 'DELETE' });
        await refreshChannels();
      } catch (err) { toast('Errore: ' + err.message, true); }
    },
  }, 'Elimina');

  const cells = [
    el('td', {}, status),
    el('td', {}, el('code', {}, ch.id)),
    el('td', {}, nameI),
    el('td', {}, descI),
    el('td', {}, langS),
    el('td', {}, devS),
    el('td', {}, idxI),
    el('td', {}, gainI),
    el('td', { class: 'meter-cell' }, meter),
  ];
  if (state.info.hls_enabled) cells.push(el('td', {}, hlsI));
  cells.push(el('td', { class: 'listeners' }, listeners));
  cells.push(el('td', {}, el('div', { class: 'row-actions' }, [startStop, save, monitor, del])));
  return el('tr', { 'data-id': ch.id }, cells);
}

// Aggiornamento leggero di stato/ascoltatori senza toccare gli input in modifica.
async function refreshStatus() {
  let channels;
  try { channels = await api('/api/admin/channels'); } catch (_) { return; }
  const byId = Object.fromEntries(channels.map((c) => [c.id, c]));
  // Se sono cambiati i canali (aggiunti/rimossi), re-render completo.
  if (channels.length !== state.channels.length) {
    state.channels = channels;
    renderTable();
    return;
  }
  state.channels = channels;
  for (const tr of document.querySelectorAll('#rows tr')) {
    const ch = byId[tr.dataset.id];
    if (!ch) continue;
    const status = tr.querySelector('td .badge');
    status.className = 'badge ' + (ch.running ? 'live' : 'off');
    status.lastChild.textContent = ch.running ? 'In onda' : 'Fermo';
    tr.querySelector('.listeners').textContent = String(ch.listeners);
    const btn = tr.querySelector('.row-actions button');
    btn.textContent = ch.running ? 'Stop' : 'Start';
    btn.className = 'btn small ' + (ch.running ? 'danger' : 'primary');
    btn.onclick = () => toggleChannel(ch.id, !ch.running);
  }
}

// I livelli servono a mappare gli ingressi del mixer: parla in un microfono e
// guarda quale riga si muove. Senza questo, trovare il channel_index giusto su
// 18 canali è indovinare.
async function refreshLevels() {
  if (document.hidden) return;
  let data;
  try { data = await api('/api/admin/levels'); } catch (_) { return; }
  const levels = data.levels || {};
  for (const tr of document.querySelectorAll('#rows tr')) {
    const lv = levels[tr.dataset.id];
    const fill = tr.querySelector('.meter-fill');
    const text = tr.querySelector('.meter-text');
    if (!fill || !text) continue;
    if (!lv || lv.rms_dbfs == null) {
      fill.style.width = '0%';
      fill.className = 'meter-fill';
      text.textContent = '—';
      continue;
    }
    // -60 dBFS -> 0%, 0 dBFS -> 100%
    const pct = Math.max(0, Math.min(100, ((lv.rms_dbfs + 60) / 60) * 100));
    fill.style.width = pct.toFixed(0) + '%';
    fill.className = 'meter-fill' + (lv.speaking ? ' speaking' : '') + (lv.clipped ? ' clip' : '');
    const parts = [`${lv.rms_dbfs} dB`];
    // Altezza rilevata della voce: è quella su cui il TTS sceglie e adatta la
    // voce della traduzione. Se resta vuota, non c'è ancora parlato a sufficienza.
    if (lv.f0_hz) parts.push(`${Math.round(lv.f0_hz)} Hz${lv.ready ? '' : '?'}`);
    else if (lv.threshold_dbfs != null) parts.push(`soglia ${lv.threshold_dbfs}`);
    if (lv.clipped) parts.push('CLIP');
    text.textContent = parts.join(' · ');
  }
}

async function toggleChannel(id, start) {
  try {
    await api(`/api/admin/channels/${encodeURIComponent(id)}/${start ? 'start' : 'stop'}`,
              { method: 'POST' });
    await refreshStatus();
  } catch (err) { toast('Errore: ' + err.message, true); }
}

// --------------------------------------------------------------- nuovo canale
document.getElementById('add-btn').onclick = async () => {
  const id = document.getElementById('new-id').value.trim();
  if (!id) { toast('Inserisci un ID', true); return; }
  const dev = document.getElementById('new-device').value;
  try {
    await api('/api/admin/channels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id,
        name: document.getElementById('new-name').value || 'Canale',
        description: document.getElementById('new-desc').value,
        source_language: document.getElementById('new-lang').value,
        input_device: dev === '' ? null : parseInt(dev, 10),
        channel_index: parseInt(document.getElementById('new-index').value, 10) || 0,
        broadcast_languages: parseLangCodes(document.getElementById('new-hls').value),
      }),
    });
    document.getElementById('new-id').value = '';
    document.getElementById('new-name').value = '';
    document.getElementById('new-desc').value = '';
    toast('Canale creato');
    await refreshChannels();
  } catch (err) { toast('Errore: ' + err.message, true); }
};

// -------------------------------------------------------------------- monitor
function startMonitor(id, name) {
  if (state.monitorWs) { state.monitorWs.close(); state.monitorWs = null; }
  state.monitorId = id;
  document.getElementById('monitor-label').textContent = `· ${name}`;
  const box = document.getElementById('monitor');
  box.innerHTML = '<div class="line">Connesso, in attesa del parlato…</div>';

  const qs = state.token ? `?token=${encodeURIComponent(state.token)}` : '';
  const ws = new WebSocket(wsURL(`/api/admin/monitor/${encodeURIComponent(id)}${qs}`));
  state.monitorWs = ws;
  let lastSeq = null;
  let lastLine = null;
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== 'transcript') return;
    if (msg.seq !== lastSeq) {
      lastLine = el('div', { class: 'line' });
      box.appendChild(lastLine);
      lastSeq = msg.seq;
    }
    lastLine.textContent = `[${msg.seq}] ${msg.text}`;
    lastLine.classList.toggle('final', msg.final);
    box.scrollTop = box.scrollHeight;
    while (box.children.length > 80) box.removeChild(box.firstChild);
  };
  ws.onclose = (ev) => {
    if (ev.code === 4401) toast('Monitor: token regia non valido', true);
  };
}

window.addEventListener('beforeunload', () => { if (state.monitorWs) state.monitorWs.close(); });

if (state.token) load();
else askToken('Inserisci il token della regia (lo trovi nel log di avvio del server).');
