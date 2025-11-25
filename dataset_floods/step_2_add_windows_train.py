import json
import random
import subprocess
from pathlib import Path

# ---------------------------------------
# n = 10       # random subset
n = None       # random subset
seed = 42
# ---------------------------------------

with open("sen1floods11_data/splits/windows_train.json") as f:
    windows = json.load(f)

if n is not None:
    random.seed(seed)
    windows = random.sample(windows, n)
    print(f"Selected {len(windows)} windows (seed={seed})")
else:
    print(f"Using all {len(windows)} windows")

root = Path("./dataset_floods")

for idx, w in enumerate(windows, start=1):
    name = w["name"]
    minx, miny, maxx, maxy = w["bbox"]
    dt = w["datetime"]

    start = dt
    end = dt.replace("00:00:00", "23:59:59")

    from datetime import datetime, timedelta

    # Convert 'dt' to a datetime object (handles Z or no Z)
    dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))

    # Expand time range ±3 days
    start = (dt_obj - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (dt_obj + timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")

    bbox_str = f"{minx},{miny},{maxx},{maxy}"

    print(f"\n[{idx}/{len(windows)}] {name}")

    cmd = [
        "rslearn", "dataset", "add_windows",
        f"--root={root}",
        "--group=default",           
        f"--name={name}",
        f"--box={bbox_str}",
        f"--start={start}",
        f"--end={end}",
        "--src_crs=EPSG:4326",
        "--resolution=10",
        "--utm"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("  SUCCESS")
    else:
        print("  FAILED")
        print("  STDERR:", result.stderr.strip())
        continue

    # Patch metadata to include split=default
    meta_file = root / "windows" / "default" / name / "metadata.json"

    if meta_file.exists():
        with meta_file.open("r") as f:
            meta = json.load(f)

        if "options" not in meta:
            meta["options"] = {}

        meta["options"]["split"] = "default"

        with meta_file.open("w") as f:
            json.dump(meta, f)

        print("Patched metadata.json with split=default")
    else:
        print("WARNING: metadata.json not found for", name)
