import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# 路径配置
# =========================
DATA_PATH = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\analysis\baseline_eval_id_06.csv")
OUTPUT_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset\analysis\figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 读取数据
# =========================
df = pd.read_csv(DATA_PATH)

print(df.head())

# =========================
# 1️⃣ MSE 分布（normal vs abnormal）
# =========================
plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="mse", hue="true_label", bins=30, kde=True)

plt.title("MSE Distribution (Normal vs Abnormal)")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend(title="Label", labels=["Normal", "Abnormal"])

plt.savefig(OUTPUT_DIR / "mse_distribution.png", dpi=300)
plt.show()


# =========================
# 2️⃣ Boxplot（normal vs abnormal）
# =========================
plt.figure(figsize=(6, 5))

sns.boxplot(data=df, x="true_label", y="mse")

plt.title("MSE Boxplot (Normal vs Abnormal)")
plt.xlabel("Label (0=Normal, 1=Abnormal)")
plt.ylabel("MSE")

plt.savefig(OUTPUT_DIR / "mse_boxplot.png", dpi=300)
plt.show()


# =========================
# 3️⃣ 按 condition 分组
# =========================
plt.figure(figsize=(10, 6))

sns.boxplot(data=df, x="condition", y="mse")

plt.title("MSE by Condition")
plt.xlabel("Condition")
plt.ylabel("MSE")

plt.savefig(OUTPUT_DIR / "mse_by_condition.png", dpi=300)
plt.show()


# =========================
# 4️⃣ 按 voltage 分组
# =========================
plt.figure(figsize=(10, 6))

sns.boxplot(data=df, x="voltage", y="mse")

plt.title("MSE by Voltage")
plt.xlabel("Voltage")
plt.ylabel("MSE")

plt.savefig(OUTPUT_DIR / "mse_by_voltage.png", dpi=300)
plt.show()


# =========================
# 5️⃣ 噪声 vs 无噪声
# =========================
plt.figure(figsize=(6, 5))

sns.boxplot(data=df, x="noise", y="mse")

plt.title("MSE: Noise vs Quiet")
plt.xlabel("Environment")
plt.ylabel("MSE")

plt.savefig(OUTPUT_DIR / "mse_noise_vs_quiet.png", dpi=300)
plt.show()


# =========================
# 6️⃣ 原 threshold vs 新 threshold 可视化
# =========================
original_threshold = 7.010286
best_threshold = df["mse"].quantile(0.5)  # 只是辅助线（或你可以手动填 58）

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="mse", hue="true_label", bins=30, kde=True)

plt.axvline(original_threshold, color='red', linestyle='--', label='Original Threshold')
plt.axvline(58.194927, color='green', linestyle='--', label='Best Threshold')

plt.title("Threshold Comparison")
plt.xlabel("MSE")
plt.legend()

plt.savefig(OUTPUT_DIR / "threshold_comparison.png", dpi=300)
plt.show()