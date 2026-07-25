// AudioWorklet: riproduce un flusso continuo di PCM int16 mono.
// Il thread principale invia ArrayBuffer (int16 LE); qui li convertiamo in
// float e li accodiamo. process() svuota la coda campione per campione,
// emettendo silenzio quando è vuota (gap tra un enunciato e l'altro).
class PCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];      // array di Float32Array
    this.readIndex = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d === 'flush') { this.queue = []; this.readIndex = 0; return; }
      const int16 = new Int16Array(d);
      const f = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
      this.queue.push(f);
    };
  }

  process(inputs, outputs) {
    const channel = outputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      if (this.queue.length === 0) {
        channel[i] = 0;
        continue;
      }
      const cur = this.queue[0];
      channel[i] = cur[this.readIndex++];
      if (this.readIndex >= cur.length) {
        this.queue.shift();
        this.readIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-player', PCMPlayer);
