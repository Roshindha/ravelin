"""B4: generate the TUNE (180 = 60 train + 20 val per familiar family... 60+20 x 3 = 240 records) and LAB-DESIGN (40) sets.

Writes data/v2/design_sets.json. Touches only streams T and L; stream F (pool / FINAL) is never generated or loaded.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import json

from breachlens import layouts as Y

if __name__ == "__main__":
    d = Y.build_design_sets()
    Y.DESIGN_SETS_PATH.write_text(json.dumps(d, indent=1) + "\n")
    print(d["counts"], d["rejections"], d["stream_use"], sep="\n")
