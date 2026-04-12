import time
from pathlib import Path

import sounddevice as sd
import pandas as pd
from scipy.io.wavfile import write

# =========================
# Basic configuration
# =========================
BASE_DIR = Path(r"C:\Users\14259\Desktop\hvac_fan_dataset")
RAW_DIR = BASE_DIR / "raw_audio"
META_DIR = BASE_DIR / "metadata"
META_FILE = META_DIR / "recording_log.csv"

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 10  # seconds

# 固定你的麦克风
DEVICE_ID = 1
DEVICE_NAME = "Razer Seiren V3 Mini"

# 每条录音前倒计时
COUNTDOWN_SEC = 3

# 每条录完后的缓冲时间（秒）
REST_BETWEEN_RUNS = 1


# =========================
# Helper functions
# =========================
def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def ensure_metadata_file():
    ensure_directories()
    if not META_FILE.exists():
        df = pd.DataFrame(columns=[
            "filename",
            "filepath",
            "condition",
            "voltage",
            "noise",
            "run",
            "duration_sec",
            "sample_rate",
            "channels",
            "device_id",
            "device_name",
            "notes"
        ])
        df.to_csv(META_FILE, index=False)


def validate_inputs(condition, voltage, noise, run):
    valid_conditions = ["normal", "blocked", "imbalance"]
    valid_voltages = ["4V", "8V", "12V"]
    valid_noise = ["quiet", "noise"]

    if condition not in valid_conditions:
        raise ValueError(f"condition must be one of {valid_conditions}")
    if voltage not in valid_voltages:
        raise ValueError(f"voltage must be one of {valid_voltages}")
    if noise not in valid_noise:
        raise ValueError(f"noise must be one of {valid_noise}")
    if not isinstance(run, int) or run < 1:
        raise ValueError("run must be a positive integer")


def make_output_folder(condition, voltage, noise):
    folder = RAW_DIR / condition / voltage / noise
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def append_metadata(row_dict):
    df = pd.DataFrame([row_dict])
    df.to_csv(META_FILE, mode="a", header=False, index=False)


def countdown(seconds=3):
    for i in range(seconds, 0, -1):
        print(i)
        time.sleep(1)


def record_once(condition, voltage, noise, run, duration=DURATION, notes=""):
    validate_inputs(condition, voltage, noise, run)
    ensure_metadata_file()

    sd.default.device = (DEVICE_ID, None)

    output_folder = make_output_folder(condition, voltage, noise)
    filename = f"{condition}_{voltage}_{noise}_run{run:02d}.wav"
    filepath = output_folder / filename

    print("\n" + "=" * 60)
    print(f"Device:    [{DEVICE_ID}] {DEVICE_NAME}")
    print(f"Condition: {condition}")
    print(f"Voltage:   {voltage}")
    print(f"Noise:     {noise}")
    print(f"Run:       {run}")
    print(f"Duration:  {duration} sec")
    print(f"Save to:   {filepath}")
    print("=" * 60)

    print("Recording starts in:")
    countdown(COUNTDOWN_SEC)
    print("Recording...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )
    sd.wait()

    write(str(filepath), SAMPLE_RATE, audio)

    append_metadata({
        "filename": filename,
        "filepath": str(filepath),
        "condition": condition,
        "voltage": voltage,
        "noise": noise,
        "run": run,
        "duration_sec": duration,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "device_id": DEVICE_ID,
        "device_name": DEVICE_NAME,
        "notes": notes
    })

    print(f"Saved: {filename}")


def record_batch(condition, voltage, noise, start_run=1, num_runs=10, duration=DURATION,
                 notes="", confirm_mode="once"):
    """
    confirm_mode options:
    - "each": 每条录音前都按一次 Enter
    - "once": 这组开始前按一次 Enter，后面自动连续录
    - "none": 完全自动开始，不需要按 Enter
    """
    print("\n" + "#" * 70)
    print("BATCH RECORDING START")
    print(f"Condition      : {condition}")
    print(f"Voltage        : {voltage}")
    print(f"Noise          : {noise}")
    print(f"Start run      : {start_run}")
    print(f"Number of runs : {num_runs}")
    print(f"Duration/run   : {duration} sec")
    print(f"Confirm mode   : {confirm_mode}")
    print("#" * 70 + "\n")

    if confirm_mode == "once":
        input("Press Enter once to start this whole batch...")
    elif confirm_mode == "none":
        print("Batch will start automatically...")
        time.sleep(1)

    for i in range(num_runs):
        run = start_run + i

        if confirm_mode == "each":
            input(f"\nPress Enter to start run {run}...")
        elif confirm_mode in ["once", "none"]:
            print(f"\nStarting run {run}...")

        record_once(
            condition=condition,
            voltage=voltage,
            noise=noise,
            run=run,
            duration=duration,
            notes=notes
        )

        if i < num_runs - 1:
            print(f"Waiting {REST_BETWEEN_RUNS} second(s) before next run...")
            time.sleep(REST_BETWEEN_RUNS)

    print("\nBatch recording completed.")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # ===== 这里改参数就行 =====
    condition = "imbalance"      # normal / blocked / imbalance
    voltage = "12V"            # 4V / 8V / 12V
    noise = "noise"           # quiet / noise

    start_run = 1
    num_runs = 10
    duration = 10
    notes = ""

    # "each" = 每条都按回车
    # "once" = 整组只按一次回车
    # "none" = 完全自动开始
    confirm_mode = "once"
    # =======================

    record_batch(
        condition=condition,
        voltage=voltage,
        noise=noise,
        start_run=start_run,
        num_runs=num_runs,
        duration=duration,
        notes=notes,
        confirm_mode=confirm_mode
    )