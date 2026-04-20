/*
 * ESP32 + INMP441  ->  WiFi live PCM streaming
 *
 * Browser (http://<ESP32_IP>/):
 *   - Live waveform + FFT spectrum
 *   - Monitor (play through speakers — beware feedback!)
 *   - Record → downloads a 16-bit 16 kHz mono WAV to your computer
 *
 * Raw binary stream at ws://<ESP32_IP>/ws  (Int16 little-endian PCM, 16 kHz mono)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <driver/i2s.h>

// ---------------- USER CONFIG ----------------
static const char* WIFI_SSID     = "Verizon_GDKKH3";
static const char* WIFI_PASSWORD = "hin6saw9dusty";
// ---------------------------------------------

#define I2S_WS   25
#define I2S_SCK  33
#define I2S_SD   32

#define I2S_SAMPLE_RATE  32000  // 32 kHz -> BCLK ~2 MHz (INMP441 needs >= 1 MHz)
#define I2S_BUFFER_SIZE  1024   // samples per I2S read (= one WebSocket frame)

AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

static const char INDEX_HTML[] PROGMEM = R"HTML(
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>ESP32 INMP441 Live</title>
<style>
  body{font-family:sans-serif;margin:20px;background:#111;color:#eee;}
  h2{margin:0 0 10px 0;}
  button{font-size:15px;padding:8px 14px;margin:4px 6px 4px 0;border:none;border-radius:4px;cursor:pointer;font-weight:600;}
  .mon{background:#3b7;color:white;} .rec{background:#d44;color:white;}
  .active{outline:3px solid #ffd04f;}
  #status{color:#aaa;font-size:13px;margin-left:10px;}
  canvas{background:#1a1a1a;border-radius:6px;display:block;margin:6px 0;}
  .lbl{color:#888;font-size:13px;margin-top:10px;}
</style></head><body>
<h2>INMP441 Live Stream</h2>
<div>
  <button id="monBtn"  class="mon">&#128263; Start Monitoring</button>
  <button id="recBtn"  class="rec">&#9679; Start Recording</button>
  <span id="status">connecting...</span>
</div>
<div class="lbl">Level &mdash; peak: <span id="peak">0</span> &nbsp; RMS: <span id="rms">0</span>
  &nbsp;|&nbsp; Waveform scale:
  <label><input type="radio" name="scl" value="auto" checked>auto</label>
  <label><input type="radio" name="scl" value="full">full (&plusmn;32k)</label>
</div>
<canvas id="wave" width="880" height="140"></canvas>
<div class="lbl">Spectrum (FFT 1024 pt, 0 &ndash; 16 kHz, Hann window, log scale)</div>
<canvas id="fft"  width="880" height="220"></canvas>

<script>
const SR = 32000;
const statusEl = document.getElementById('status');
const monBtn   = document.getElementById('monBtn');
const recBtn   = document.getElementById('recBtn');

// ---- WebSocket ----
const ws = new WebSocket('ws://' + location.host + '/ws');
ws.binaryType = 'arraybuffer';
ws.onopen  = () => statusEl.textContent = 'connected (' + SR + ' Hz, 16-bit mono)';
ws.onclose = () => statusEl.textContent = 'disconnected';
ws.onerror = () => statusEl.textContent = 'error';

// ---- Audio graph (playback only; FFT is computed manually from raw PCM) ----
let audioCtx = null, gainNode = null, playhead = 0;
let monitoring = false;
function ensureAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  gainNode = audioCtx.createGain();
  gainNode.gain.value = 0;
  gainNode.connect(audioCtx.destination);
  playhead = audioCtx.currentTime;
}

// ---- Minimal in-place radix-2 FFT (Cooley-Tukey) ----
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const t1=re[i]; re[i]=re[j]; re[j]=t1; const t2=im[i]; im[i]=im[j]; im[j]=t2; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const tr = cr * re[i+k+half] - ci * im[i+k+half];
        const ti = cr * im[i+k+half] + ci * re[i+k+half];
        re[i+k+half] = re[i+k] - tr;
        im[i+k+half] = im[i+k] - ti;
        re[i+k]      = re[i+k] + tr;
        im[i+k]      = im[i+k] + ti;
        const cr2 = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = cr2;
      }
    }
  }
}
// Hann window (length 1024)
const WIN_N = 1024;
const hann = new Float32Array(WIN_N);
for (let i = 0; i < WIN_N; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WIN_N - 1));
const fftRe = new Float32Array(WIN_N);
const fftIm = new Float32Array(WIN_N);
const spec  = new Float32Array(WIN_N / 2);  // smoothed magnitude bins

// ---- Recording buffer ----
let recording = false;
let recChunks = [];
let recSamples = 0;
function startRec() { recChunks = []; recSamples = 0; recording = true; }
function stopRec()  {
  recording = false;
  if (recChunks.length === 0) return;
  const pcm = new Int16Array(recSamples);
  let off = 0;
  for (const c of recChunks) { pcm.set(c, off); off += c.length; }
  recChunks = [];
  downloadWav(pcm, SR);
}
function downloadWav(int16, sr) {
  const byteLen = int16.length * 2;
  const buf = new ArrayBuffer(44 + byteLen);
  const v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, 'RIFF'); v.setUint32(4, 36 + byteLen, true); wstr(8, 'WAVE');
  wstr(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  wstr(36, 'data'); v.setUint32(40, byteLen, true);
  new Int16Array(buf, 44).set(int16);
  const blob = new Blob([buf], { type: 'audio/wav' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'recording_' + new Date().toISOString().replace(/[:.]/g, '-') + '.wav';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---- Level meter ----
const peakEl = document.getElementById('peak');
const rmsEl  = document.getElementById('rms');

// ---- Incoming audio ----
ws.onmessage = (e) => {
  if (typeof e.data === 'string') return;
  const int16 = new Int16Array(e.data);

  if (recording) {
    recChunks.push(new Int16Array(int16));
    recSamples += int16.length;
  }

  // Level meter
  let peak = 0, sumsq = 0;
  for (let i = 0; i < int16.length; i++) {
    const a = Math.abs(int16[i]);
    if (a > peak) peak = a;
    sumsq += int16[i] * int16[i];
  }
  const rms = Math.sqrt(sumsq / int16.length) | 0;
  peakEl.textContent = peak;
  rmsEl.textContent  = rms;

  drawWave(int16, peak);
  computeFFT(int16);

  // Playback (only active if user enabled monitoring)
  if (audioCtx && monitoring) {
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
    const buf = audioCtx.createBuffer(1, f32.length, SR);
    buf.copyToChannel(f32, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(gainNode);
    const now = audioCtx.currentTime;
    if (playhead < now + 0.02) playhead = now + 0.02;
    src.start(playhead);
    playhead += f32.length / SR;
  }
};

function computeFFT(int16) {
  const n = WIN_N;
  const len = Math.min(int16.length, n);
  for (let i = 0; i < len; i++) { fftRe[i] = (int16[i] / 32768) * hann[i]; fftIm[i] = 0; }
  for (let i = len; i < n; i++) { fftRe[i] = 0; fftIm[i] = 0; }
  fft(fftRe, fftIm);
  const halfN = n >> 1;
  const alpha = 0.6; // smoothing
  for (let i = 0; i < halfN; i++) {
    const mag = Math.sqrt(fftRe[i]*fftRe[i] + fftIm[i]*fftIm[i]);
    spec[i] = alpha * spec[i] + (1 - alpha) * mag;
  }
}

// ---- UI ----
monBtn.onclick = async () => {
  ensureAudio();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  monitoring = !monitoring;
  gainNode.gain.value = monitoring ? 1 : 0;
  monBtn.textContent = monitoring ? '\u{1F507} Stop Monitoring' : '\u{1F508} Start Monitoring';
  monBtn.classList.toggle('active', monitoring);
};
recBtn.onclick = async () => {
  ensureAudio();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  if (!recording) {
    startRec();
    recBtn.textContent = '\u25A0 Stop & Download';
    recBtn.classList.add('active');
  } else {
    stopRec();
    recBtn.textContent = '\u25CF Start Recording';
    recBtn.classList.remove('active');
  }
};

// ---- Drawing ----
const waveCv = document.getElementById('wave'), waveCtx = waveCv.getContext('2d');
function currentScaleMode() {
  const r = document.querySelector('input[name="scl"]:checked');
  return r ? r.value : 'auto';
}
function drawWave(int16, peak) {
  const w = waveCv.width, h = waveCv.height;
  waveCtx.fillStyle = '#1a1a1a'; waveCtx.fillRect(0, 0, w, h);
  // center line
  waveCtx.strokeStyle = '#333'; waveCtx.beginPath();
  waveCtx.moveTo(0, h/2); waveCtx.lineTo(w, h/2); waveCtx.stroke();

  const mode = currentScaleMode();
  const denom = (mode === 'full') ? 32768 : Math.max(512, peak * 1.2);

  waveCtx.strokeStyle = '#4fc3f7'; waveCtx.lineWidth = 1.2; waveCtx.beginPath();
  for (let i = 0; i < int16.length; i++) {
    const x = i * w / int16.length;
    const y = h / 2 - (int16[i] / denom) * (h / 2);
    i ? waveCtx.lineTo(x, y) : waveCtx.moveTo(x, y);
  }
  waveCtx.stroke();

  // scale label
  waveCtx.fillStyle = '#666'; waveCtx.font = '11px sans-serif';
  waveCtx.fillText('y-scale: \u00B1' + Math.round(denom), 8, 14);
}
const fftCv = document.getElementById('fft'), fftCtx = fftCv.getContext('2d');
function drawFFT() {
  requestAnimationFrame(drawFFT);
  const w = fftCv.width, h = fftCv.height;
  fftCtx.fillStyle = '#1a1a1a'; fftCtx.fillRect(0, 0, w, h);
  const bins = spec.length;  // 512 bins covering 0..Nyquist (0..8 kHz)
  const bw = w / bins;
  // log scale: dB-like, clamped to [-60, 0] dB
  const norm = 200;          // rough reference magnitude for 0 dB
  for (let i = 0; i < bins; i++) {
    const mag = spec[i];
    const db  = 20 * Math.log10(Math.max(mag, 1e-4) / norm);
    const v   = Math.max(0, Math.min(1, (db + 60) / 60));  // -60..0 dB -> 0..1
    fftCtx.fillStyle = 'hsl(' + Math.round(220 - v * 220) + ',85%,' + Math.round(30 + v * 40) + '%)';
    fftCtx.fillRect(i * bw, h - v * h, Math.max(1, bw), v * h);
  }
  // axis ticks (2 kHz lines, Nyquist = SR/2)
  fftCtx.fillStyle = '#888'; fftCtx.font = '10px sans-serif';
  const nyq = SR / 2;
  for (let khz = 2; khz < nyq / 1000; khz += 2) {
    const x = (khz * 1000 / nyq) * w;
    fftCtx.fillRect(x, 0, 1, 4);
    fftCtx.fillText(khz + 'k', x + 2, 12);
  }
}
drawFFT();
</script>
</body></html>
)HTML";

void setupI2S() {
  // APLL + more DMA buffers + zeroing avoids periodic bit-slip glitches
  // commonly seen with INMP441 on ESP32 at 16-bit resolution.
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = I2S_BUFFER_SIZE,
    .use_apll = true,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num  = I2S_WS,
    .data_out_num = -1,
    .data_in_num  = I2S_SD
  };
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);

  // Discard the first few frames — DMA ring can contain junk right after install.
  int32_t junk[I2S_BUFFER_SIZE];
  size_t br;
  for (int k = 0; k < 4; k++) i2s_read(I2S_NUM_0, junk, sizeof(junk), &br, 100 / portTICK_PERIOD_MS);
}

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400); Serial.print('.');
  }
  Serial.println();
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\nINMP441 + WiFi PCM streaming");

  setupI2S();
  setupWiFi();

  server.addHandler(&ws);
  server.on("/", HTTP_GET, [](AsyncWebServerRequest* r) {
    r->send_P(200, "text/html", INDEX_HTML);
  });
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  static int32_t raw[I2S_BUFFER_SIZE];
  static int16_t pcm[I2S_BUFFER_SIZE];
  size_t bytes_read = 0;

  i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes_read, portMAX_DELAY);
  size_t n = bytes_read / sizeof(int32_t);

  // --- Diagnostic: once per second ---
  static uint32_t last_dbg = 0;
  if (millis() - last_dbg > 1000) {
    int64_t sum = 0;
    int32_t mx = INT32_MIN, mn = INT32_MAX;
    int zeros = 0, changes = 0;
    for (size_t i = 0; i < n; i++) {
      sum += raw[i];
      if (raw[i] > mx) mx = raw[i];
      if (raw[i] < mn) mn = raw[i];
      if (raw[i] == 0) zeros++;
      if (i > 0 && raw[i] != raw[i-1]) changes++;
    }
    int32_t mean = n > 0 ? (int32_t)(sum / (int64_t)n) : 0;
    // "changes" close to 1023 => real dynamic data; close to 0 => static junk.
    Serial.printf("changes=%d/1023  zeros=%d/1024  mean=%d  min=%d  max=%d\n",
                  changes, zeros, mean, mn, mx);
    Serial.printf("  [0..7]:     %d %d %d %d %d %d %d %d\n",
                  raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
    Serial.printf("  [512..519]: %d %d %d %d %d %d %d %d\n",
                  raw[512], raw[513], raw[514], raw[515], raw[516], raw[517], raw[518], raw[519]);
    last_dbg = millis();
  }

  // INMP441 outputs 24-bit data left-aligned in bits 31:8 of a 32-bit word.
  // >>14 keeps the top 18 bits, giving ~4x digital gain before clipping to int16.
  for (size_t i = 0; i < n; i++) {
    int32_t v = raw[i] >> 14;
    if (v > 32767) v = 32767;
    else if (v < -32768) v = -32768;
    pcm[i] = (int16_t)v;
  }

  // --- Glitch de-spike (median-of-3 on near-full-scale isolated samples) ---
  // If a sample is near ±FS but its neighbors are small, it's almost certainly
  // an I2S bit-slip artifact — replace with the mean of its neighbors.
  const int16_t SPIKE_THR = 30000;   // "near full scale"
  const int16_t NEIGH_THR = 8000;    // "neighbors much smaller"
  for (size_t i = 1; i + 1 < n; i++) {
    int16_t s = pcm[i], l = pcm[i-1], r = pcm[i+1];
    if ((s > SPIKE_THR || s < -SPIKE_THR) &&
        abs(l) < NEIGH_THR && abs(r) < NEIGH_THR) {
      pcm[i] = (int16_t)(((int32_t)l + (int32_t)r) / 2);
    }
  }

  if (ws.count() > 0) {
    ws.binaryAll((uint8_t*)pcm, n * sizeof(int16_t));
  }
  ws.cleanupClients();
}
