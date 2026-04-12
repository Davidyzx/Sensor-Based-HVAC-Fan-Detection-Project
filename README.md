# HVAC Fan Anomaly Detection — Acoustic Sensing with ESP32

A sensor-based system for detecting abnormal HVAC fan states (filter blockage, blade imbalance) using low-cost acoustic sensing, embedded firmware, and a machine learning inference pipeline.

---

## System Overview

```
HVAC Fan
   │
   ▼
INMP441 (I2S Microphone)
   │
   ▼
ESP32 Microcontroller
   │  ├── Serial output → Real-time amplitude monitoring
   │  └── [Planned] SD card → WAV file storage
   │
   ▼
Python (Laptop / Edge)
   │  ├── record_audio.py     → Dataset collection
   │  └── baseline_runner_custom.py → Feature extraction + inference
   │
   ▼
Pre-trained Autoencoder Model
   │
   ▼
Anomaly Score (MSE) → Threshold → Normal / Abnormal
```

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
│   │   ├── 4V/
│   │   │   ├── quiet/                 # Recordings in quiet environment (10 files)
│   │   │   └── noise/                 # Recordings with background noise (10 files)
│   │   ├── 8V/
│   │   └── 12V/
│   ├── blocked/                       # Abnormal: air filter blocked
│   │   ├── 4V/  ├── quiet/  └── noise/
│   │   ├── 8V/  ├── quiet/  └── noise/
│   │   └── 12V/ ├── quiet/  └── noise/
│   └── imbalance/                     # Abnormal: fan blade imbalance
│       ├── 4V/  ├── quiet/  └── noise/
│       ├── 8V/  ├── quiet/  └── noise/
│       └── 12V/ ├── quiet/  └── noise/
│
├── analysis/                          # Inference results and visualizations
│   ├── baseline_eval_id_06.csv        # Per-file MSE scores, true labels, and predictions
│   ├── baseline_mse_results_id04.csv  # Raw MSE results for model id_04
│   ├── baseline_mse_results_id_06.csv # Raw MSE results for model id_06
│   └── figures/                       # Generated plots (PNG, 300 DPI)
│       ├── mse_distribution.png       # MSE histogram: normal vs abnormal
│       ├── mse_boxplot.png            # MSE boxplot by true label
│       ├── mse_by_condition.png       # MSE by fault condition (normal/blocked/imbalance)
│       ├── mse_by_voltage.png         # MSE by fan voltage (4V/8V/12V)
│       ├── mse_noise_vs_quiet.png     # MSE: quiet vs noisy environment
│       └── threshold_comparison.png   # Original threshold vs optimized threshold
│
└── metadata/
    └── recording_log.csv              # Recording metadata: filename, condition, voltage, noise, device info
```

---

## Component Descriptions

| Component | Description |
|-----------|-------------|
| `ESP32_INMP41.ino` | Firmware for ESP32. Configures I2S interface to read from INMP441 microphone at 16 kHz. Computes block-level amplitude and outputs to Serial Plotter for real-time monitoring. |
| `record_audio.py` | Automated recording script. Captures 10-second mono WAV clips at 16 kHz from USB microphone, organizes files by condition/voltage/noise, and logs metadata to CSV. |
| `baseline_runner_custom.py` | Core ML inference script. Extracts 320-dim log-mel-spectrogram feature vectors (64 bands × 5-frame window) from each WAV file, passes through pre-trained autoencoder, computes MSE reconstruction error, applies threshold to classify normal vs abnormal. |
| `analysis_visualization.py` | Reads `baseline_eval_id_06.csv` and generates 6 publication-quality figures covering MSE distributions, per-condition breakdowns, voltage effects, noise impact, and threshold comparison. |
| `checkpth.py` | Utility script to inspect the keys and data types inside a `.pth` model checkpoint. |
| `model/*.pth` | Pre-trained autoencoder weights. `id_06` is the primary evaluated model. Original training follows the MIMII Dataset baseline (input→64→64→8→64→64→output). |
| `raw_audio/` | Structured audio dataset. Three top-level conditions × three voltages × two noise environments × 10 runs = 180 files total. Naming format: `{condition}_{voltage}_{noise}_run{##}.wav`. |
| `analysis/*.csv` | Tabular evaluation results. Each row corresponds to one audio file with columns: filename, condition, voltage, noise, true_label (0=normal, 1=abnormal), mse, pred_original, pred_best. |
| `metadata/recording_log.csv` | Tracks all recorded files with device info, sample rate, and recording notes. |

---

## ML Pipeline

### Feature Extraction
Each 10-second WAV file is converted to a sequence of **320-dimensional feature vectors**:
- Mel-spectrogram: 64 mel bands, FFT size 1024, hop length 512
- Log-amplitude scaling
- Sliding window of 5 consecutive frames → one 320-dim vector per window

### Model Architecture
Fully-connected autoencoder (reconstruction-based anomaly detection):
```
Encoder:  320 → 64 → 64 → 8   (ReLU activations)
Decoder:  8   → 64 → 64 → 320 (ReLU activations)
```

### Anomaly Scoring
Mean Squared Error (MSE) between input and reconstructed output.
A higher MSE indicates a larger deviation from the normal acoustic pattern learned during training.

| Class | Typical MSE Range |
|-------|-------------------|
| Normal | 1 – 10 |
| Imbalance (abnormal) | 20 – 50 |
| Blocked (abnormal) | 50 – 87 |

### Thresholds
| Threshold | Value | Source |
|-----------|-------|--------|
| Original | 7.01 | MIMII Dataset baseline paper |
| Optimized (current data) | 58.19 | Best F1-score on collected dataset |

---

## Hardware Setup

| Component | Role | Interface |
|-----------|------|-----------|
| ESP32 | Microcontroller / sensor node | — |
| INMP441 | MEMS digital microphone (acoustic sensor) | I2S |
| [Planned] MPU6050 | Vibration / acceleration sensor | I2C |
| [Planned] SD Card Module | Local WAV storage on sensor node | SPI |

**INMP441 Pin Mapping (ESP32):**
| Signal | ESP32 GPIO |
|--------|-----------|
| WS (LRCL) | GPIO 25 |
| SCK (BCLK) | GPIO 33 |
| SD (DIN) | GPIO 32 |

---

## Getting Started

### Requirements
```
Python >= 3.8
torch
librosa
numpy
pandas
matplotlib
seaborn
sounddevice
soundfile
```

Install dependencies:
```bash
pip install torch librosa numpy pandas matplotlib seaborn sounddevice soundfile
```

### Run Inference on Existing Dataset
```bash
cd scripts
python baseline_runner_custom.py
# Results saved to: analysis/baseline_eval_id_06.csv
```

### Generate Analysis Plots
```bash
cd scripts
python analysis_visualization.py
# Figures saved to: analysis/figures/
```

### Record New Audio Samples
Edit the configuration section at the top of `record_audio.py`, then:
```bash
cd scripts
python record_audio.py
```

---

## Project Status

| Checkpoint | Status |
|------------|--------|
| Checkpoint 1: System Proposal | In progress |
| Checkpoint 2: Sensor Node | In progress — firmware working, dataset collected, ML inference complete |
| Checkpoint 3: Edge Component + Backend | Planned |
| Checkpoint 4: Full System Integration | Planned |
