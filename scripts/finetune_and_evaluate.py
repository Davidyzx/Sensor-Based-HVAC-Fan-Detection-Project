"""
阶段一：改结构 + 微调 + 全面评估
Stage 1: Architecture modification + fine-tuning + comprehensive evaluation

本脚本完成以下步骤 / This script performs the following steps:
  1. 定义改进的模型结构（瓶颈层 ReLU → LeakyReLU）
     Define improved model architecture (bottleneck ReLU → LeakyReLU)
  2. 加载 id_04 预训练权重并迁移到新结构
     Load id_04 pretrained weights and transfer to new architecture
  3. 用 60 个 normal 样本微调（48 train / 12 val）
     Fine-tune on 60 normal samples (48 train / 12 val)
  4. 微调完成后执行全面评估：
     After fine-tuning, run comprehensive evaluation:
       a. MSE 异常检测（AUC, F1）→ 对比 baseline 0.769
          MSE anomaly detection (AUC, F1) → compare with baseline 0.769
       b. 提取 latent → 检查激活维度数量
          Extract latent → check number of active dimensions
       c. t-SNE 可视化
          t-SNE visualization
       d. 多分类（RF / SVM / KNN）→ 对比 baseline 95%
          Multi-class classification → compare with baseline 95%

依赖 / Dependencies: torch, librosa, sklearn, matplotlib, seaborn, tqdm
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import librosa
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# 0. 配置区 / Configuration
# ============================================================
DATASET_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\raw_audio")
OUTPUT_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\analysis")
MODEL_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\scripts\model")
FIGURE_DIR = OUTPUT_DIR / "figures"

# 预训练模型（迁移效果最好的 id_04）
# Pretrained model (best transfer performance: id_04)
PRETRAINED_PATH = MODEL_DIR / "baseline_fan_id_04.pth（副本）"

# 微调后模型保存路径 / Fine-tuned model save path
FINETUNED_PATH = MODEL_DIR / "finetuned_leakyrelu_id_04.pth"

INPUT_DIM = 320
LATENT_DIM = 8

# --- 训练超参数 / Training hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
# LeakyReLU 负斜率 / LeakyReLU negative slope
LEAKY_SLOPE = 0.01
# 早停耐心值 / Early stopping patience
PATIENCE = 15
# 训练/验证划分比 / Train/val split ratio
VAL_RATIO = 0.2  # 60 normal → 48 train + 12 val

# 随机种子 / Random seed for reproducibility
SEED = 42

# t-SNE 参数 / t-SNE parameters
TSNE_PERPLEXITY = 30

# 交叉验证折数 / Cross-validation folds
CV_FOLDS = 5

# 标签后缀（用于区分输出文件）/ Label suffix for output files
TAG = "finetuned"


# ============================================================
# 1. 特征提取（沿用原 baseline，不做任何修改）
#    Feature extraction (same as baseline, no modifications)
# ============================================================
def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """
    将音频文件转为帧级 320 维特征向量（同 baseline）。
    Convert audio file to frame-level 320-dim feature vectors (same as baseline).
    """
    y, sr = librosa.load(file_name, sr=None, mono=True)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power,
    )
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + np.finfo(float).eps)

    vectorarray_size = log_mel_spectrogram.shape[1] - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, n_mels * frames), float)

    dims = n_mels * frames
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray


# ============================================================
# 2. 模型定义
#    Model definitions
# ============================================================

class MIMII_Baseline_AE(nn.Module):
    """
    原始 baseline 模型（全 ReLU），仅用于加载预训练权重。
    Original baseline model (all ReLU), used only for loading pretrained weights.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ImprovedAE(nn.Module):
    """
    改进的自编码器：瓶颈层使用 LeakyReLU 替代 ReLU。
    Improved autoencoder: bottleneck activation changed from ReLU to LeakyReLU.

    改动说明 / Changes from baseline:
      - encoder 最后一层: ReLU → LeakyReLU（解决 dead neuron 问题）
        encoder last layer: ReLU → LeakyReLU (fixes dead neuron problem)
      - decoder 保持不变 / decoder unchanged
      - 所有 Linear 层维度不变 / all Linear layer dimensions unchanged

    原始结构 / Original:
      Encoder: 320 → 64(ReLU) → 64(ReLU) → 8(ReLU)       ← 6 维死亡
      Decoder: 8 → 64(ReLU) → 64(ReLU) → 320

    改进结构 / Improved:
      Encoder: 320 → 64(ReLU) → 64(ReLU) → 8(LeakyReLU)  ← 允许负值通过
      Decoder: 8 → 64(ReLU) → 64(ReLU) → 320
    """

    def __init__(self, input_dim, leaky_slope=0.01):
        super().__init__()
        # 编码器：前两层保持 ReLU，最后一层改为 LeakyReLU
        # Encoder: keep ReLU for first two layers, change last to LeakyReLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=leaky_slope),  # 关键改动 / key change
        )
        # 解码器：保持不变 / Decoder: unchanged
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)    # 潜在表示 / latent representation
        out = self.decoder(z)  # 重建输出 / reconstructed output
        return out


# ============================================================
# 3. 权重迁移：从原始 ReLU 模型加载权重到 LeakyReLU 模型
#    Weight transfer: load pretrained ReLU weights into LeakyReLU model
# ============================================================
def transfer_weights(pretrained_path: Path, device):
    """
    加载预训练的 ReLU 模型权重，迁移到 LeakyReLU 模型。
    Load pretrained ReLU model weights and transfer to LeakyReLU model.

    由于只改了激活函数（无参数），所有 Linear 层的权重可以直接复制。
    Since only the activation function changed (no parameters), all Linear
    layer weights can be directly copied.
    """
    # 先加载到原始架构 / Load into original architecture first
    old_model = MIMII_Baseline_AE(input_dim=INPUT_DIM).to(device)
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        old_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        old_model.load_state_dict(checkpoint)

    # 创建新架构 / Create new architecture
    new_model = ImprovedAE(input_dim=INPUT_DIM, leaky_slope=LEAKY_SLOPE).to(device)

    # 逐层复制 Linear 权重（跳过激活函数，激活函数没有参数）
    # Copy Linear weights layer by layer (skip activations, they have no parameters)
    new_model.load_state_dict(old_model.state_dict())

    print(f"[INFO] 权重迁移完成 / Weight transfer complete")
    print(f"  源模型 / Source: MIMII_Baseline_AE (ReLU)")
    print(f"  目标模型 / Target: ImprovedAE (LeakyReLU at bottleneck)")
    print(f"  所有 Linear 层权重已复制 / All Linear layer weights copied")

    return new_model


# ============================================================
# 4. 数据准备：收集 normal 样本并划分 train/val
#    Data preparation: collect normal samples and split train/val
# ============================================================
def prepare_training_data(dataset_dir: Path, val_ratio=0.2, seed=42):
    """
    只收集 normal 条件下的音频，提取帧级特征，划分训练/验证集。
    Collect only normal-condition audio, extract frame-level features, split train/val.

    无监督范式：训练只用 normal，blocked/imbalance 完全不参与训练。
    Unsupervised paradigm: train on normal only; blocked/imbalance never seen during training.

    返回 / Returns:
      train_data: np.ndarray, shape (num_train_frames, 320)
      val_data:   np.ndarray, shape (num_val_frames, 320)
      normal_files: list of Path, 所有 normal 文件路径 / all normal file paths
    """
    # 收集所有 normal 文件 / Collect all normal files
    normal_files = sorted(dataset_dir.glob("normal/*/*/*.wav"))
    print(f"Found {len(normal_files)} normal audio files.")

    # 固定随机种子并打乱 / Fix seed and shuffle
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(normal_files))

    # 划分 / Split
    n_val = int(len(normal_files) * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    print(f"Train files: {len(train_indices)}, Val files: {len(val_indices)}")

    # 提取特征 / Extract features
    train_vectors = []
    for idx in tqdm(train_indices, desc="Extracting train features"):
        vectors = file_to_vector_array(str(normal_files[idx]))
        if vectors.shape[0] > 0:
            train_vectors.append(vectors)

    val_vectors = []
    for idx in tqdm(val_indices, desc="Extracting val features"):
        vectors = file_to_vector_array(str(normal_files[idx]))
        if vectors.shape[0] > 0:
            val_vectors.append(vectors)

    train_data = np.concatenate(train_vectors, axis=0)
    val_data = np.concatenate(val_vectors, axis=0)

    print(f"Train vectors: {train_data.shape}  (约 {len(train_indices)} files × ~300 frames)")
    print(f"Val vectors:   {val_data.shape}  (约 {len(val_indices)} files × ~300 frames)")

    return train_data, val_data, normal_files


# ============================================================
# 5. 训练循环（含早停）
#    Training loop (with early stopping)
# ============================================================
def train_model(model, train_data, val_data, device):
    """
    用 MSE 重建损失微调自编码器。
    Fine-tune autoencoder with MSE reconstruction loss.

    训练策略 / Training strategy:
      - 优化器: Adam / Optimizer: Adam
      - 损失: MSE（和原论文一致）/ Loss: MSE (same as original paper)
      - 早停: 验证集损失连续 PATIENCE 个 epoch 不下降则停止
        Early stopping: stop if val loss doesn't improve for PATIENCE epochs
      - 保存验证损失最低的模型权重
        Save model weights with lowest validation loss
    """
    # 构建 DataLoader / Build DataLoaders
    train_tensor = torch.FloatTensor(train_data).to(device)
    val_tensor = torch.FloatTensor(val_data).to(device)

    train_dataset = TensorDataset(train_tensor, train_tensor)  # 自编码器：输入=目标 / AE: input=target
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 优化器和损失函数 / Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 早停相关变量 / Early stopping variables
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # 训练历史 / Training history
    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'='*60}")
    print(f"  开始训练 / Starting training")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Early stopping patience: {PATIENCE}")
    print(f"{'='*60}\n")

    model.train()

    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 / Training phase ---
        model.train()
        train_losses = []

        for batch_x, batch_target in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- 验证阶段 / Validation phase ---
        model.eval()
        with torch.no_grad():
            val_reconstructed = model(val_tensor)
            val_loss = criterion(val_reconstructed, val_tensor).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        # --- 早停检查 / Early stopping check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
            marker = " *"  # 标记最优 / mark best
        else:
            patience_counter += 1
            marker = ""

        # 每 10 个 epoch 打印一次 / Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or marker:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss: {avg_train_loss:.6f}  "
                  f"val_loss: {val_loss:.6f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\n[早停 / Early stopping] 验证损失 {PATIENCE} 个 epoch 未改善，停止训练。")
            print(f"[Early stopping] Val loss did not improve for {PATIENCE} epochs.")
            break

    # 恢复最优权重 / Restore best weights
    model.load_state_dict(best_state)
    model.eval()
    print(f"\n最优验证损失 / Best val loss: {best_val_loss:.6f} (epoch with *)")

    return model, history


# ============================================================
# 6. 绘制训练曲线 / Plot training curves
# ============================================================
def plot_training_curves(history):
    """
    绘制训练/验证损失曲线。
    Plot train/val loss curves.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=1.5)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Fine-tuning Training Curves (LeakyReLU)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"training_curves_{TAG}.png", dpi=300)
    plt.show()
    print(f"Saved: training_curves_{TAG}.png")


# ============================================================
# 7. 解析文件名 / Parse filename metadata
# ============================================================
def parse_audio_metadata(audio_path: Path):
    """同 baseline / Same as baseline."""
    filename = audio_path.name
    pattern = r"^(normal|blocked|imbalance)_(4V|8V|12V)_(quiet|noise)_run(\d+)\.wav$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")
    condition, voltage, noise, run = match.groups()
    true_label = 0 if condition == "normal" else 1
    return {
        "filename": filename, "filepath": str(audio_path),
        "condition": condition, "voltage": voltage, "noise": noise,
        "run": int(run), "true_label": true_label,
    }


# ============================================================
# 8. MSE 异常检测评估（和 baseline 完全一致的逻辑）
#    MSE anomaly detection evaluation (identical logic to baseline)
# ============================================================
def evaluate_anomaly_detection(model, audio_files, device):
    """
    对所有 180 个文件计算 MSE，做二分类异常检测评估。
    Compute MSE for all 180 files and evaluate binary anomaly detection.

    返回 / Returns:
      df: DataFrame with per-file MSE and metadata
      metrics: dict with AUC, best_threshold, best_f1, etc.
    """
    results = []
    for audio_path in tqdm(audio_files, desc=f"MSE inference ({TAG})"):
        meta = parse_audio_metadata(audio_path)

        data = file_to_vector_array(str(audio_path))
        if data.shape[0] == 0:
            meta["mse"] = np.nan
        else:
            data_tensor = torch.FloatTensor(data).to(device)
            with torch.no_grad():
                reconstructed = model(data_tensor)
                mse_per_frame = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
                meta["mse"] = torch.mean(mse_per_frame).cpu().item()

        results.append(meta)

    df = pd.DataFrame(results)

    # 阈值无关 AUC / Threshold-free AUC
    auc = roc_auc_score(df["true_label"], df["mse"])

    # 最优 F1 阈值 / Best F1 threshold
    precision_arr, recall_arr, thresholds = precision_recall_curve(
        df["true_label"], df["mse"]
    )
    f1_scores = 2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # 用最优阈值评估 / Evaluate with best threshold
    df["pred_label"] = (df["mse"] > best_threshold).astype(int)
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    metrics = {
        "auc": auc,
        "best_threshold": best_threshold,
        "best_f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mse_mean_normal": df.loc[df["true_label"] == 0, "mse"].mean(),
        "mse_mean_abnormal": df.loc[df["true_label"] == 1, "mse"].mean(),
    }

    return df, metrics


# ============================================================
# 9. Latent 特征提取 / Latent feature extraction
# ============================================================
def extract_latent_features(model, audio_files, device):
    """
    提取每个文件的 latent 均值和标准差。
    Extract per-file latent mean and std.
    """
    records = []
    for audio_path in tqdm(audio_files, desc=f"Extracting latent ({TAG})"):
        meta = parse_audio_metadata(audio_path)
        data = file_to_vector_array(str(audio_path))

        if data.shape[0] == 0:
            for i in range(LATENT_DIM):
                meta[f"z_mean_{i}"] = np.nan
                meta[f"z_std_{i}"] = np.nan
        else:
            data_tensor = torch.FloatTensor(data).to(device)
            with torch.no_grad():
                latent = model.encoder(data_tensor)  # shape: (num_frames, 8)
            latent_np = latent.cpu().numpy()
            for i in range(LATENT_DIM):
                meta[f"z_mean_{i}"] = latent_np[:, i].mean()
                meta[f"z_std_{i}"] = latent_np[:, i].std()

        records.append(meta)

    return pd.DataFrame(records)


# ============================================================
# 10. Latent 维度活跃度分析 / Latent dimension activity analysis
# ============================================================
def analyze_active_dimensions(df_latent):
    """
    统计有多少个 latent 维度是"活跃的"（均值的方差 > 阈值）。
    Count how many latent dimensions are "active" (variance of means > threshold).

    一个维度如果对所有样本输出几乎相同的值（或全 0），就是"死"的。
    A dimension is "dead" if it outputs nearly the same value (or all zeros) for all samples.
    """
    mean_cols = [f"z_mean_{i}" for i in range(LATENT_DIM)]
    print(f"\n{'='*60}")
    print(f"  Latent 维度活跃度分析 / Latent Dimension Activity Analysis")
    print(f"{'='*60}")

    active_dims = 0
    for i, col in enumerate(mean_cols):
        values = df_latent[col].values
        dim_mean = np.mean(values)
        dim_std = np.std(values)
        dim_range = np.max(values) - np.min(values)

        # 判断活跃条件：范围 > 1.0（避免全零或几乎恒定的维度）
        # Active condition: range > 1.0 (avoids all-zero or near-constant dims)
        is_active = dim_range > 1.0
        status = "ACTIVE" if is_active else "DEAD"
        if is_active:
            active_dims += 1

        print(f"  z{i}: mean={dim_mean:8.3f}, std={dim_std:7.3f}, "
              f"range={dim_range:8.3f} → [{status}]")

    print(f"\n  活跃维度 / Active dimensions: {active_dims}/{LATENT_DIM}")
    print(f"  (Baseline ReLU 模型只有 2/8 活跃 / Baseline ReLU model had only 2/8 active)")

    return active_dims


# ============================================================
# 11. t-SNE 可视化 / t-SNE visualization
# ============================================================
def plot_tsne(latent_matrix, labels, title, filename, palette=None):
    """同 latent_analysis.py / Same as latent_analysis.py."""
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED,
                init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(latent_matrix)

    df_plot = pd.DataFrame({
        "t-SNE 1": embedding[:, 0],
        "t-SNE 2": embedding[:, 1],
        "label": labels,
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x="t-SNE 1", y="t-SNE 2", hue="label",
                    palette=palette, s=60, alpha=0.8, edgecolor="white", linewidth=0.5)
    plt.title(title)
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


# ============================================================
# 12. 逐维度 boxplot / Per-dimension boxplot
# ============================================================
def plot_latent_dimensions(df_latent):
    mean_cols = [f"z_mean_{i}" for i in range(LATENT_DIM)]
    df_long = df_latent[["condition"] + mean_cols].melt(
        id_vars="condition", var_name="dimension", value_name="value"
    )
    df_long["dimension"] = df_long["dimension"].str.replace("z_mean_", "z")

    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df_long, x="dimension", y="value", hue="condition",
                hue_order=["normal", "blocked", "imbalance"],
                palette={"normal": "#55A868", "blocked": "#DD8452", "imbalance": "#C44E52"})
    plt.title(f"Latent Dimensions by Condition ({TAG})")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Activation Value")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"latent_dimensions_by_condition_{TAG}.png", dpi=300)
    plt.show()
    print(f"Saved: latent_dimensions_by_condition_{TAG}.png")


# ============================================================
# 13. 多分类实验 / Multi-class classification
# ============================================================
def run_classification(X, y):
    """
    在 latent features 上跑三个分类器的 5-fold CV。
    Run 5-fold CV with three classifiers on latent features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    classifiers = {
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    print(f"\n{'='*60}")
    print(f"  多分类实验 / Multi-class Classification ({CV_FOLDS}-fold CV)")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Features: {X_scaled.shape[1]}-dim latent mean")
    print(f"  Samples: {X_scaled.shape[0]}")
    print(f"{'='*60}")

    results = []
    for name, clf in classifiers.items():
        cv_results = cross_validate(clf, X_scaled, y_encoded, cv=cv,
                                    scoring=scoring, return_train_score=False)
        row = {
            "classifier": name,
            "accuracy": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "f1_macro": cv_results["test_f1_macro"].mean(),
            "f1_macro_std": cv_results["test_f1_macro"].std(),
            "precision_macro": cv_results["test_precision_macro"].mean(),
            "recall_macro": cv_results["test_recall_macro"].mean(),
        }
        results.append(row)
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {row['accuracy']:.4f} +/- {row['accuracy_std']:.4f}")
        print(f"  F1 macro : {row['f1_macro']:.4f} +/- {row['f1_macro_std']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# 14. 打印 baseline 对比总结 / Print baseline comparison summary
# ============================================================
def print_comparison(metrics, clf_results, active_dims):
    """
    将微调结果与 baseline 冻结模型结果进行对比。
    Compare fine-tuned results with frozen baseline model results.
    """
    # Baseline 参考值（来自之前的实验）
    # Baseline reference values (from previous experiments)
    baseline_auc = 0.7693
    baseline_f1_binary = 0.8458
    baseline_rf_acc = 0.9500
    baseline_active_dims = 2

    best_clf = clf_results.loc[clf_results["accuracy"].idxmax()]

    print(f"\n{'='*70}")
    print(f"  BASELINE vs FINE-TUNED COMPARISON / 基线 vs 微调对比")
    print(f"{'='*70}")
    print(f"{'指标 / Metric':<35} {'Baseline (冻结)':<18} {'Fine-tuned':<18} {'变化 / Change'}")
    print(f"{'-'*70}")

    # AUC
    auc_delta = metrics["auc"] - baseline_auc
    print(f"{'AUC (binary)':<35} {baseline_auc:<18.4f} {metrics['auc']:<18.4f} {auc_delta:+.4f}")

    # F1
    f1_delta = metrics["best_f1"] - baseline_f1_binary
    print(f"{'Best F1 (binary)':<35} {baseline_f1_binary:<18.4f} {metrics['best_f1']:<18.4f} {f1_delta:+.4f}")

    # Threshold
    print(f"{'Best Threshold':<35} {'38.18':<18} {metrics['best_threshold']:<18.4f}")

    # Active dims
    print(f"{'Active latent dims':<35} {str(baseline_active_dims)+'/8':<18} {str(active_dims)+'/8':<18}")

    # 3-class accuracy
    acc_delta = best_clf["accuracy"] - baseline_rf_acc
    print(f"{'Best 3-class accuracy (CV)':<35} {baseline_rf_acc:<18.4f} {best_clf['accuracy']:<18.4f} {acc_delta:+.4f}")
    print(f"{'Best classifier':<35} {'Random Forest':<18} {best_clf['classifier']}")

    print(f"{'='*70}")


# ============================================================
# 15. 主流程 / Main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================================
    # Step 1: 迁移权重 / Transfer weights
    # ========================================
    print(f"\n[Step 1] 加载预训练权重并迁移到 LeakyReLU 架构...")
    print(f"[Step 1] Loading pretrained weights and transferring to LeakyReLU architecture...")
    model = transfer_weights(PRETRAINED_PATH, device)

    # ========================================
    # Step 2: 准备训练数据 / Prepare training data
    # ========================================
    print(f"\n[Step 2] 准备训练数据（仅 normal 样本）...")
    print(f"[Step 2] Preparing training data (normal samples only)...")
    train_data, val_data, normal_files = prepare_training_data(
        DATASET_DIR, val_ratio=VAL_RATIO, seed=SEED
    )

    # ========================================
    # Step 3: 微调训练 / Fine-tune training
    # ========================================
    print(f"\n[Step 3] 开始微调训练...")
    print(f"[Step 3] Starting fine-tuning...")
    model, history = train_model(model, train_data, val_data, device)

    # 保存微调后的模型 / Save fine-tuned model
    torch.save(model.state_dict(), FINETUNED_PATH)
    print(f"模型已保存 / Model saved to: {FINETUNED_PATH}")

    # 绘制训练曲线 / Plot training curves
    plot_training_curves(history)

    # ========================================
    # Step 4: MSE 异常检测评估 / MSE anomaly detection evaluation
    # ========================================
    print(f"\n[Step 4] MSE 异常检测评估（全部 180 个文件）...")
    print(f"[Step 4] MSE anomaly detection evaluation (all 180 files)...")
    all_audio = sorted(DATASET_DIR.glob("*/*/*/*.wav"))
    print(f"Found {len(all_audio)} audio files.")

    df_mse, metrics = evaluate_anomaly_detection(model, all_audio, device)

    # 保存 MSE 结果 / Save MSE results
    mse_csv = OUTPUT_DIR / f"baseline_mse_results_{TAG}.csv"
    df_mse.to_csv(mse_csv, index=False)
    print(f"Saved MSE results to: {mse_csv}")

    print(f"\n--- MSE 异常检测结果 / MSE Anomaly Detection Results ---")
    print(f"  AUC            : {metrics['auc']:.4f}")
    print(f"  Best Threshold : {metrics['best_threshold']:.4f}")
    print(f"  Best F1        : {metrics['best_f1']:.4f}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  MSE normal     : {metrics['mse_mean_normal']:.4f}")
    print(f"  MSE abnormal   : {metrics['mse_mean_abnormal']:.4f}")

    # ========================================
    # Step 5: Latent 特征提取与分析 / Latent feature extraction & analysis
    # ========================================
    print(f"\n[Step 5] 提取 latent 特征并分析...")
    print(f"[Step 5] Extracting latent features and analyzing...")
    df_latent = extract_latent_features(model, all_audio, device)

    # 保存 / Save
    latent_csv = OUTPUT_DIR / f"latent_features_{TAG}.csv"
    df_latent.to_csv(latent_csv, index=False)
    print(f"Saved latent features to: {latent_csv}")

    # 维度活跃度分析 / Dimension activity analysis
    active_dims = analyze_active_dimensions(df_latent)

    # 逐维度 boxplot / Per-dimension boxplot
    plot_latent_dimensions(df_latent)

    # ========================================
    # Step 6: t-SNE 可视化 / t-SNE visualization
    # ========================================
    print(f"\n[Step 6] t-SNE 可视化...")
    print(f"[Step 6] t-SNE visualization...")
    mean_cols = [f"z_mean_{i}" for i in range(LATENT_DIM)]
    latent_matrix = df_latent[mean_cols].values

    plot_tsne(latent_matrix, df_latent["condition"].values,
              f"t-SNE by Condition ({TAG})",
              f"tsne_by_condition_{TAG}.png",
              {"normal": "#55A868", "blocked": "#DD8452", "imbalance": "#C44E52"})

    plot_tsne(latent_matrix, df_latent["voltage"].values,
              f"t-SNE by Voltage ({TAG})",
              f"tsne_by_voltage_{TAG}.png",
              {"4V": "#4C72B0", "8V": "#DD8452", "12V": "#C44E52"})

    plot_tsne(latent_matrix, df_latent["noise"].values,
              f"t-SNE by Noise ({TAG})",
              f"tsne_by_noise_{TAG}.png",
              {"quiet": "#4C72B0", "noise": "#C44E52"})

    # ========================================
    # Step 7: 多分类实验 / Multi-class classification
    # ========================================
    print(f"\n[Step 7] 多分类实验...")
    print(f"[Step 7] Multi-class classification...")
    clf_results = run_classification(latent_matrix, df_latent["condition"].values)

    clf_csv = OUTPUT_DIR / f"classification_results_{TAG}.csv"
    clf_results.to_csv(clf_csv, index=False)
    print(f"Saved classification results to: {clf_csv}")

    # ========================================
    # Step 8: 对比总结 / Comparison summary
    # ========================================
    print_comparison(metrics, clf_results, active_dims)

    print(f"\n{'='*60}")
    print(f"  全部完成 / All done!")
    print(f"  输出目录 / Output directory: {OUTPUT_DIR}")
    print(f"  图片目录 / Figure directory: {FIGURE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
