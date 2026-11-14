"""B6: build and freeze the 300-layout pool (data/v2/pool_v1.json + pool_v1.sha256, read-only) from seed stream F.

Prints ONLY layout counts per family/split and rejection counts per rule (and redraw counts). No FINAL aggregate of any score, baseline
or Oracle result exists in this process's output. Refuses to overwrite an existing pool. Usage: python scripts_b6_build_pool.py
"""
import json
import os
import stat
import sys
import time
from collections import Counter

from breachlens import pool as P

if __name__ == "__main__":
    if P.POOL_PATH.exists():
        sys.exit(f"{P.POOL_PATH} exists and is frozen; refusing to rebuild")
    t0 = time.time()
    doc = P.build_pool()
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    P.POOL_PATH.write_text(text)
    sha = P.pool_sha256()
    P.POOL_SHA_PATH.write_text(f"{sha}  pool_v1.json\n")
    os.chmod(P.POOL_PATH, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(P.POOL_SHA_PATH, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    cnt = Counter((r["family"], r["split"]) for r in doc["records"])
    print("layout counts per family / split:")
    for (fam, sp), n in sorted(cnt.items()):
        print(f"  {fam:12s} {sp:15s} {n}")
    print("total", len(doc["records"]))
    print("rejection counts per rule (and D4 redraws) per walk:")
    for k, w in doc["walk"].items():
        print(f"  {k:18s} candidates={w['candidates']:4d} accepted={w['accepted']:3d} structural={w['structural']} "
              f"d4_redraws={w['redraws_d4_duplicate']} single_type_trivial={w['single_type_trivial']} too_hard={w['too_hard']} "
              f"stat_rejected={w['stat_rejected']}")
    print("sha256", sha, f"({time.time() - t0:.0f}s)")
