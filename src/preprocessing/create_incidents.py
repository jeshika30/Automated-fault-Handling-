import os
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/synthetic_syslogs.csv"
PROCESSED_DIR = "data/processed"
INCIDENTS_PATH = os.path.join(PROCESSED_DIR, "incidents.csv")


def floor_to_5min(ts):
    """Floor timestamp to 5-minute boundary."""
    minute = (ts.minute // 5) * 5
    return ts.replace(minute=minute, second=0, microsecond=0)


def create_incidents():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"[INFO] Reading raw logs from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by device + timestamp
    df = df.sort_values(["device", "timestamp"]).reset_index(drop=True)

    # Compute 5-minute window_start per log line
    df["window_start"] = df["timestamp"].apply(floor_to_5min)

    incidents = []
    incident_id = 0
    min_logs_per_window = 2  # drop very small windows

    # Group by device + 5-minute window
    for (device, window_start), group in df.groupby(["device", "window_start"]):
        if len(group) < min_logs_per_window:
            continue  # skip tiny windows

        window_end = group["timestamp"].max()
        messages = group["message"].tolist()
        text = " ".join(messages)

        # Majority label in this window (this allows mixed logs but picks dominant fault)
        label = group["fault_class"].value_counts().idxmax()

        incidents.append(
            {
                "incident_id": incident_id,
                "device": device,
                "window_start": window_start,
                "window_end": window_end,
                "text": text,
                "fault_class": label,
            }
        )
        incident_id += 1

    incidents_df = pd.DataFrame(incidents)
    print(f"[INFO] Created {len(incidents_df)} incidents")

    if incidents_df.empty:
        print("[ERROR] No incidents created – consider lowering min_logs_per_window.")
        incidents_df.to_csv(INCIDENTS_PATH, index=False)
        return

    # Save full incidents table
    incidents_df.to_csv(INCIDENTS_PATH, index=False)
    print(f"[OK] Saved incidents to {INCIDENTS_PATH}")

    # Train/val/test splits
    train_df, temp_df = train_test_split(
        incidents_df,
        test_size=0.3,
        random_state=42,
        stratify=incidents_df["fault_class"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["fault_class"],
    )

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"[OK] Saved train/val/test splits in {PROCESSED_DIR}")
    print(f"    train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")


if __name__ == "__main__":
    create_incidents()