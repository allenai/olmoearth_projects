"""
Adds test windows to rslearn dataset under the 'predict' group.
Expands each chip's timestamp by ±3 days for Sentinel imagery matching.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

with open("sen1floods11_data/splits/windows_test.json") as f:
    windows = json.load(f)

print(f"Processing {len(windows)} test windows")

root = Path("./dataset_floods")

for idx, w in enumerate(windows, start=1):
    name = w["name"]
    minx, miny, maxx, maxy = w["bbox"]
    dt = w["datetime"]

    dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    start = (dt_obj - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (dt_obj + timedelta(days=3)).strftime("%Y-%m-%dT23:59:59Z")

    bbox_str = f"{minx},{miny},{maxx},{maxy}"

    print(f"\n[{idx}/{len(windows)}] {name}")

    cmd = [
        "rslearn", "dataset", "add_windows",
        f"--root={root}",
        "--group=predict",
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
        print("   SUCCESS")
    else:
        print("   FAILED")
        print("  STDERR:", result.stderr.strip())