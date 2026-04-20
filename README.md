# HVAC Fan Anomaly Detection — Acoustic Sensing with ESP32

A sensor-based system for detecting abnormal HVAC fan states (filter blockage, blade imbalance) using low-cost acoustic sensing, embedded firmware, and a machine learning inference pipeline.

---

## Checkpoints

### Checkpoint 1 (Milestone 1) — System Proposal

**Deliverables:**
- Block diagram of the physical system with network connections and component annotations
- Component catalog (see table below)
- 10-minute whiteboard pitch detailing what the system does and how it is designed

**What the system does:**

This system detects abnormal operating conditions of an HVAC fan using acoustic sensing. A microphone captures the sound produced by the fan under three conditions — normal operation, blocked air filter, and blade imbalance — across multiple voltage levels (4V, 8V, 12V) and acoustic environments (quiet, noisy). Audio features are extracted and passed through a pre-trained autoencoder. Reconstruction error (MSE) is used as an anomaly score: normal sounds are reconstructed well (low MSE), while abnormal sounds produce high reconstruction error and are flagged as anomalies.

**System Design — Component Catalog:**

| Component | Role |
|-----------|------|
| HVAC Fan | Target system under monitoring |
| INMP441 | MEMS digital microphone (I2S) — primary acoustic sensor on ESP32 |
| ESP32 | Microcontroller / sensor node — reads I2S audio, outputs to Serial or SD card |
| Razer Seiren V3 Mini (USB) | Backup USB microphone used for dataset collection (see Checkpoint 2) |
| SD Card Module *(planned)* | Local WAV file storage on the sensor node (SPI) |
| MPU6050 *(planned)* | Vibration / acceleration sensor — second sensor for the node (I2C) |
| Laptop (Edge) | Runs Python pipeline: recording, feature extraction, model inference |
| Pre-trained Autoencoder | ML model — reconstruction-based anomaly detection |

**Block Diagram:**
```
┌─────────────┐     I2S      ┌──────────────┐    SPI (planned)   ┌──────────┐
│   INMP441   │─────────────▶│    ESP32     │───────────────────▶│ SD Card  │
│  Microphone │              │ Sensor Node  │                     │ Storage  │
└─────────────┘              │              │    Serial (current) └──────────┘
                             │              │───────────────────▶ Serial Monitor
┌─────────────┐     I2C      │              │
│   MPU6050   │─────────────▶│  (planned)   │
│  Vibration  │              └──────────────┘
└─────────────┘
                                    │
                              (data transfer)
                                    ▼
                        ┌───────────────────────┐
                        │   Laptop  (Edge)       │
                        │  Python Pipeline       │
                        │  - record_audio.py     │
                        │  - feature extraction  │
                        │  - autoencoder model   │
                        │  - MSE threshold       │
                        └───────────────────────┘
                                    │
                                    ▼
                        Normal / Blocked / Imbalance
```

---

### Checkpoint 2 (Milestone 2) — Sensor Node

**Deliverables:**
- Sensor node working (full or partial)
- Collecting measurements and running
- Source versioned and submitted to GitHub
- 10–15 minute whiteboard and demo
- Design docs updated to current state

**Status: Partial — data collection pipeline validated, ESP32 firmware running**

#### Design Intent vs. Actual Implementation

The original design called for the ESP32 to capture audio via INMP441 over I2S and write WAV files directly to an SD card for offline transfer. However, the SD card module was not available at this stage, which blocked the full embedded recording pipeline.

**Workaround adopted for Checkpoint 2:**

The Razer Seiren V3 Mini USB microphone was used as a direct replacement for data collection. Audio was recorded from the HVAC fan setup via Python (`record_audio.py`) into a structured dataset on the laptop. This allowed the full ML inference pipeline to be validated ahead of the embedded pipeline being completed.

#### Current Data Pipeline

```
                    ┌─────────────────────────────────────┐
                    │         INTENDED PIPELINE           │
                    │                                     │
  HVAC Fan Sound    │   INMP441 ──I2S──▶ ESP32            │
                    │                      │              │
                    │               SD Card (planned)     │
                    │                      │              │
                    │               WAV files ──▶ Laptop  │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │     CURRENT (CHECKPOINT 2)          │
                    │                                     │
  HVAC Fan Sound ──▶│  Razer Seiren V3 Mini (USB)         │
                    │          │                          │
                    │    record_audio.py                  │
                    │          │                          │
                    │   WAV files (16 kHz, mono, 10 s)    │
                    │          │                          │
                    │  baseline_runner_custom.py          │
                    │    (feature extraction + model)     │
                    │          │                          │
                    │  MSE Score ──▶ Threshold ──▶ Label  │
                    └─────────────────────────────────────┘
```

#### ESP32 Firmware (ESP32_INMP41.ino)
The firmware is functional. It configures the I2S interface to read from the INMP441 at 16 kHz and streams block-level amplitude values to the Serial Plotter for real-time monitoring. SD card writing will be integrated in the next phase.

**INMP441 Pin Mapping:**
| Signal | ESP32 GPIO |
|--------|-----------|
| WS (LRCL) | GPIO 25 |
| SCK (BCLK) | GPIO 33 |
| SD (DIN) | GPIO 32 |

#### Dataset Collected
- **180 WAV files** — 16 kHz, mono, 10 seconds each
- 3 conditions × 3 voltages × 2 environments × 10 runs

| Condition | Description |
|-----------|-------------|
| `normal` | Fan operating normally |
| `blocked` | Air filter blocked (abnormal) |
| `imbalance` | Fan blade imbalance (abnormal) |

#### ML Inference Results

| Class | Typical MSE Range |
|-------|-------------------|
| Normal | 1 – 10 |
| Imbalance | 20 – 50 |
| Blocked | 50 – 87 |

| Threshold | Value | Source |
|-----------|-------|--------|
| Original | 7.01 | MIMII Dataset baseline paper |
| Optimized (current data) | 58.19 | Best F1-score on collected dataset |

---

### Checkpoint 3 (Milestone 3) — Backend — ML Inference Engine

**Status: Completed (backend track, 3 model stages)**

Instead of the originally-planned InfluxDB+Grafana pipeline,
I delivered a ML-centric backend: three progressively stronger
inference models on the collected 180-sample dataset.

| Stage | Approach | Binary AUC | 3-class Acc | Notes |
|---|---|---|---|---|
| Stage 0 | Frozen MIMII id_04 AE | 0.769 | 95% (RF on z) | severe domain shift |
| Stage 1 | Fine-tune + LeakyReLU | 0.997 | 95% (RF on z) | recovers src-domain threshold |
| Stage 2 | Domain-Adversarial MTL | 1.000 | **100% (end-to-end)** | joint rec+cls+adv loss |

**Backend deliverables:**
- `scripts/baseline_runner_custom.py` — Stage 0 inference
- `scripts/finetune_and_evaluate.py` — Stage 1 fine-tune
- `scripts/stage2_domain_adversarial.py` — Stage 2 DANN-MTL
- `analysis/` — full evaluation CSVs + 20+ figures

**Known limitations (Plan to address in Checkpoint 4):**
- ESP32 + INMP441 hardware path still has signal distortion
  (suspected unsoldered connections) — USB mic used as proxy

---

### Checkpoint 4 (Milestone 4) — Full System Integration *(Planned)*

- End-to-end system demo
- ESP32 sensor node → Edge → Backend → Dashboard
- Add second sensor (MPU6050 vibration)
- Final design doc update

---

## Repository Structure

```
Sensor Project/
│
├── README.md                          # This file
│
├── scripts/                           # All source code
│   ├── ESP32_INMP41.ino               # ESP32 firmware — I2S audio capture, real-time serial output
│   ├── record_audio.py                # Dataset recording utility (USB mic → structured WAV files)
│   ├── baseline_runner_custom.py      # Model inference: feature extraction + autoencoder evaluation
│   ├── analysis_visualization.py      # Generate analysis plots from evaluation results
│   ├── checkpth.py                    # Inspect PyTorch model checkpoint structure
│   └── model/                         # Pre-trained autoencoder weights
│       ├── baseline_fan_id_00.pth
│       ├── baseline_fan_id_02.pth
│       ├── baseline_fan_id_04.pth
│       └── baseline_fan_id_06.pth
│
├── raw_audio/                         # Audio dataset (180 WAV files, 16 kHz, mono, 10 s each)
│   ├── normal/                        # Normal fan operation
│   │   ├── 4V/ ├── quiet/  └── noise/
│   │   ├── 8V/ ├── quiet/  └── noise/
│   │   └── 12V/├── quiet/  └── noise/
│   ├── blocked/                       # Abnormal: air filter blocked
│   │   ├── 4V/ ├── quiet/  └── noise/
│   │   ├── 8V/ ├── quiet/  └── noise/
│   │   └── 12V/├── quiet/  └── noise/
│   └── imbalance/                     # Abnormal: fan blade imbalance
│       ├── 4V/ ├── quiet/  └── noise/
│       ├── 8V/ ├── quiet/  └── noise/
│       └── 12V/├── quiet/  └── noise/
│
├── analysis/                          # Inference results and visualizations
│   ├── baseline_eval_id_06.csv        # Per-file MSE scores, true labels, and predictions
│   ├── baseline_mse_results_id04.csv  # Raw MSE results for model id_04
│   ├── baseline_mse_results_id_06.csv # Raw MSE results for model id_06
│   └── figures/                       # Generated plots (PNG, 300 DPI)
│       ├── mse_distribution.png       # MSE histogram: normal vs abnormal
│       ├── mse_boxplot.png            # MSE boxplot by true label
│       ├── mse_by_condition.png       # MSE by fault condition
│       ├── mse_by_voltage.png         # MSE by fan voltage (4V / 8V / 12V)
│       ├── mse_noise_vs_quiet.png     # MSE: quiet vs noisy environment
│       └── threshold_comparison.png   # Original vs optimized threshold
│
└── metadata/
    └── recording_log.csv              # Recording metadata: filename, condition, voltage, noise, device
```

---

## ML Pipeline

### Feature Extraction
Each 10-second WAV file → sequence of **320-dimensional feature vectors**:
- Mel-spectrogram: 64 mel bands, FFT size 1024, hop length 512
- Log-amplitude scaling
- Sliding window of 5 consecutive frames → one 320-dim vector per window

### Model Architecture
```
Encoder:  320 → 64 → 64 → 8   (ReLU)
Decoder:  8   → 64 → 64 → 320 (ReLU)
```

---

## Getting Started

### Requirements
```bash
pip install torch librosa numpy pandas matplotlib seaborn sounddevice soundfile
```

### Run Inference
```bash
cd scripts
python baseline_runner_custom.py
# Results → analysis/baseline_eval_id_06.csv
```

### Generate Plots
```bash
cd scripts
python analysis_visualization.py
# Figures → analysis/figures/
```

### Record New Audio
Edit device/path config at the top of `record_audio.py`, then:
```bash
cd scripts
python record_audio.py
```
