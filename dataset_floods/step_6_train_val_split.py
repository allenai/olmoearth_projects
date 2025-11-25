"""
Splits windows into train/val sets using deterministic hash-based assignment.
Roughly 12.5% of data goes to validation (hex digits 0-1 out of 0-f).
"""

from rslearn.dataset import Dataset
from upath import UPath
import hashlib
import tqdm

ds = Dataset(UPath("./dataset_floods"))
windows = ds.load_windows(workers=16)

for w in tqdm.tqdm(windows):
    hexid = hashlib.sha256(w.name.encode()).hexdigest()[0]
    w.options["split"] = "val" if hexid in ["0","1"] else "train"
    w.save()