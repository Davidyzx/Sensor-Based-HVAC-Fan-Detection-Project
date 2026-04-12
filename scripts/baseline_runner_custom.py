import re
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)


# =========================
# 0. 配置区
# =========================
DATASET_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\raw_audio")
OUTPUT_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\analysis")

# 现在先跑 id_06
MODEL_NAME = "id_06"
MODEL_PATH = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\scripts\model\baseline_fan_id_06.pth（副本）")

# 原论文/原数据上找到的 threshold
ORIGINAL_THRESHOLD = 7.010286

RAW_OUTPUT_CSV = OUTPUT_DIR / f"baseline_mse_results_{MODEL_NAME}.csv"
EVAL_OUTPUT_CSV = OUTPUT_DIR / f"baseline_eval_{MODEL_NAME}.csv"

INPUT_DIM = 320


# =========================
# 1. 特征提取（沿用原 baseline）
# =========================
def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    y, sr = librosa.load(file_name, sr=None, mono=True)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + np.finfo(float).eps)

    vectorarray_size = log_mel_spectrogram.shape[1] - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, n_mels * frames), float)

    dims = n_mels * frames
    vectorarray = np.zeros((vectorarray_size, dims), float)

    for t in range(frames):
        vectorarray[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + vectorarray_size].T

    return vectorarray


# =========================
# 2. 模型结构（沿用原 baseline）
# =========================
class MIMII_Baseline_AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# =========================
# 3. 加载模型
# =========================
def load_trained_model(model_path: Path, device):
    model = MIMII_Baseline_AE(input_dim=INPUT_DIM).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# =========================
# 4. 收集所有音频
# 目录结构应为 raw_audio/condition/voltage/noise/*.wav
# =========================
def collect_audio_files(dataset_dir: Path):
    wav_files = sorted(dataset_dir.glob("*/*/*/*.wav"))
    return wav_files


# =========================
# 5. 解析文件名
# 示例: normal_4V_quiet_run01.wav
# =========================
def parse_audio_metadata(audio_path: Path):
    filename = audio_path.name

    pattern = r"^(normal|blocked|imbalance)_(4V|8V|12V)_(quiet|noise)_run(\d+)\.wav$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")

    condition, voltage, noise, run = match.groups()

    true_label = 0 if condition == "normal" else 1

    return {
        "filename": filename,
        "filepath": str(audio_path),
        "condition": condition,
        "voltage": voltage,
        "noise": noise,
        "run": int(run),
        "true_label": true_label,
    }


# =========================
# 6. 单条音频推理：输出 MSE
# =========================
def infer_one_file(model, audio_path: Path, device):
    data = file_to_vector_array(str(audio_path))

    if data.shape[0] == 0:
        return np.nan

    data_tensor = torch.FloatTensor(data).to(device)

    with torch.no_grad():
        reconstructed = model(data_tensor)
        mse_per_frame = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
        file_error = torch.mean(mse_per_frame).cpu().item()

    return file_error


# =========================
# 7. 给定阈值评估
# =========================
def evaluate_with_threshold(df, threshold, name=""):
    df_eval = df.copy()
    df_eval["pred_label"] = (df_eval["mse"] > threshold).astype(int)

    y_true = df_eval["true_label"].values
    y_pred = df_eval["pred_label"].values
    y_score = df_eval["mse"].values

    metrics_dict = {
        "name": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    print(f"\n===== {name} =====")
    print(f"Threshold : {metrics_dict['threshold']:.6f}")
    print(f"Accuracy  : {metrics_dict['accuracy']:.4f}")
    print(f"Precision : {metrics_dict['precision']:.4f}")
    print(f"Recall    : {metrics_dict['recall']:.4f}")
    print(f"F1-score  : {metrics_dict['f1']:.4f}")
    print(f"AUC       : {metrics_dict['auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics_dict["confusion_matrix"])

    return df_eval, metrics_dict


# =========================
# 8. 在当前数据上自动找最佳 F1 阈值
# =========================
def find_best_f1_threshold(df):
    y_true = df["true_label"].values
    y_score = df["mse"].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve 返回的 precision/recall 比 thresholds 长1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nBest F1 Threshold found on your data: {best_threshold:.6f}")
    print(f"Best F1 Score on your data: {f1_scores[best_idx]:.4f}")

    return best_threshold


# =========================
# 9. 主流程
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    audio_files = collect_audio_files(DATASET_DIR)
    print(f"Found {len(audio_files)} audio files.")

    if len(audio_files) == 0:
        raise ValueError("No audio files found. Please check DATASET_DIR.")

    model = load_trained_model(MODEL_PATH, device)
    print(f"Loaded model from: {MODEL_PATH}")

    results = []

    for audio_path in tqdm(audio_files, desc=f"Running inference ({MODEL_NAME})"):
        meta = parse_audio_metadata(audio_path)
        mse = infer_one_file(model, audio_path, device)
        meta["mse"] = mse
        results.append(meta)

    df = pd.DataFrame(results)

    # 保存原始 MSE
    df.to_csv(RAW_OUTPUT_CSV, index=False)
    print(f"\nSaved raw MSE results to: {RAW_OUTPUT_CSV}")

    # 不依赖阈值的整体可分性
    overall_auc = roc_auc_score(df["true_label"], df["mse"])
    print(f"\nOverall AUC (threshold-free): {overall_auc:.4f}")

    # 1) 原模型阈值
    df_original, metrics_original = evaluate_with_threshold(
        df, ORIGINAL_THRESHOLD, name=f"Original Threshold ({MODEL_NAME})"
    )

    # 2) 当前数据自动寻找最佳阈值
    best_threshold = find_best_f1_threshold(df)
    df_best, metrics_best = evaluate_with_threshold(
        df, best_threshold, name=f"Best Threshold on Your Data ({MODEL_NAME})"
    )

    # 合并到一个表里保存
    df_final = df.copy()
    df_final["pred_original"] = (df_final["mse"] > ORIGINAL_THRESHOLD).astype(int)
    df_final["pred_best"] = (df_final["mse"] > best_threshold).astype(int)

    df_final.to_csv(EVAL_OUTPUT_CSV, index=False)
    print(f"\nSaved evaluation results to: {EVAL_OUTPUT_CSV}")

    # 打印一个简短对比总结
    print("\n================ FINAL SUMMARY ================")
    print(f"Model               : {MODEL_NAME}")
    print(f"Original threshold  : {ORIGINAL_THRESHOLD:.6f}")
    print(f"Best threshold      : {best_threshold:.6f}")
    print(f"AUC                 : {overall_auc:.4f}")
    print(f"Original F1         : {metrics_original['f1']:.4f}")
    print(f"Best-threshold F1   : {metrics_best['f1']:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()