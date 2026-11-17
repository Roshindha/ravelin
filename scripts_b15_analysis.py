"""B15 analysis: every table, interval and comparison of docs/SPEC_V2.md row (gg), computed from the RAW FILES ONLY.

    .venv/bin/python scripts_b15_analysis.py --session results/b15 --out results/b15/tables
    .venv/bin/python scripts_b15_analysis.py --session <temp dir> --out <dir> --label REHEARSAL

Inputs: <session>/manifest.jsonl and the raw cell files it lists (each verified against its sha256 and row count), plus two seen-data files for the
generalisation gap (`results/b14/seen_layout_rows.csv`, sha256 pinned in row (gg); `data/v2/baselines/b8_layout_rows.csv`, DEV-train baselines).
No simulator call, no model call, no pool or ledger access: this module imports only numpy, the standard library and `scripts_b15_cells`
(a test asserts that torch, the simulator and the pool are not imported).

Statistics (row (gg) (d)):
  * estimand = mean over the group's layouts of the mean over the condition's 10 seeds of the per-layout value (baselines: one value per layout);
  * PRIMARY interval = hierarchical bootstrap: resample seeds with replacement (per condition), then layouts; 10 000 resamples; percentile 95 %;
    numpy default_rng(8080) started afresh for every interval; per resample the draw order is the seed indices (arm A, then arm B) and then ONE
    vector of layout indices; statistic = mean over resampled layouts of the mean over resampled seeds;
  * paired difference: ONE layout vector shared by both arms; seed indices independent per arm, SHARED when both arms are the same trained models
    (TTA vs no TTA, planner vs its own learner); a learner vs a deterministic baseline resamples seeds for the learner only;
  * SECONDARY interval (section 7 wording): resample the learners only, layouts fixed, same rng; it never decides whether a comparison holds;
  * a primary comparison HOLDS iff its paired-difference primary 95 % CI excludes 0 (lower bound > 0 for 'X beats Y'); everything else is exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

import scripts_b15_cells as C

ROOT = Path(__file__).resolve().parent
SEEN_ROWS = ROOT / "results" / "b14" / "seen_layout_rows.csv"
B8_ROWS = ROOT / "data" / "v2" / "baselines" / "b8_layout_rows.csv"
NRES, BSEED = 10_000, 8080
METRICS = ("breach", "score")                          # primary metrics; d / n_bought / n_deployed / army are descriptive
LEARNER_BASES = ("ppo_plain", "ppo_d4", "ddqn")
CONTROL_ORDER = ("ras", "rps", "rpsfull")


class AnalysisError(Exception):
    pass


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------- bootstrap core
@lru_cache(maxsize=None)
def _draws(sizes: tuple, layouts: tuple, R: int, seed: int):
    """The resample indices, drawn exactly in the pinned order from a fresh default_rng(seed): per resample, the seed vectors (one per entry of
    `sizes`, in order), then one layout vector per entry of `layouts` (len 0 = layouts fixed). Returns (seed count matrices (R, S), layout indices (R, n))."""
    rng = np.random.default_rng(seed)
    sa = [np.empty((R, s), dtype=np.int64) for s in sizes]
    la = [np.empty((R, n), dtype=np.int64) for n in layouts]
    for b in range(R):
        for k, s in enumerate(sizes):
            sa[k][b] = rng.integers(0, s, size=s)
        for k, n in enumerate(layouts):
            la[k][b] = rng.integers(0, n, size=n)
    counts = [(a[:, :, None] == np.arange(a.shape[1])[None, None, :]).sum(1).astype(np.float64) for a in sa]
    return counts, la


def _pct(x: np.ndarray) -> tuple:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def interval(arms: list, signs: list, cols: np.ndarray, seed_groups: list | None = None, level: str = "hier", R: int = NRES, seed: int = BSEED) -> dict:
    """arms = [(M, det)], M shaped (seeds, layouts-of-the-whole-session); `cols` = the group's layout columns.
    seed_groups[i] = id of the seed draw arm i uses (equal ids share one draw; None for a deterministic arm). Returns mean, lo95, hi95."""
    cols = np.asarray(cols)
    n = len(cols)
    if seed_groups is None:
        seed_groups, k = [], 0
        for _, det in arms:
            seed_groups.append(None if det else k)
            k += 0 if det else 1
    order = list(OrderedDict.fromkeys(g for g in seed_groups if g is not None))          # draw order: first use
    sizes = []
    for g in order:
        s = {arms[i][0].shape[0] for i in range(len(arms)) if seed_groups[i] == g}
        if len(s) != 1:
            raise AnalysisError("arms sharing a seed draw need the same number of seeds")
        sizes.append(s.pop())
    per_layout = sum(sg * M[:, cols].mean(axis=0) for (M, _), sg in zip(arms, signs))            # (n,) seed-averaged difference per layout
    mean = float(per_layout.mean())
    if not sizes and level == "seeds":
        return {"mean": mean, "lo95": mean, "hi95": mean}                                       # nothing is random
    counts, la = _draws(tuple(sizes), (n,) if level == "hier" else (), R, seed)
    D = np.zeros((R, n))
    for (M, det), sg, g in zip(arms, signs, seed_groups):
        D += sg * (M[0, cols][None, :] if det else counts[order.index(g)] @ M[:, cols] / M.shape[0])
    stat = np.take_along_axis(D, la[0], axis=1).mean(axis=1) if level == "hier" else D.mean(axis=1)
    lo, hi = _pct(stat)
    return {"mean": mean, "lo95": lo, "hi95": hi}


def interval_two_sets(M1: np.ndarray, cols1: np.ndarray, M2: np.ndarray, cols2: np.ndarray, det: bool, level: str = "hier", R: int = NRES, seed: int = BSEED) -> dict:
    """mean over set 1 minus mean over set 2 for the SAME learners (one shared seed draw), the two layout sets resampled independently
    (draw order: seeds, set-1 layouts, set-2 layouts). level 'seeds': layouts fixed."""
    cols1, cols2 = np.asarray(cols1), np.asarray(cols2)
    m1, m2 = M1[:, cols1].mean(axis=0), M2[:, cols2].mean(axis=0)
    mean = float(m1.mean() - m2.mean())
    if det and level == "seeds":
        return {"mean": mean, "lo95": mean, "hi95": mean}
    sizes = () if det else (M1.shape[0],)
    if level == "hier":
        counts, la = _draws(sizes, (len(cols1), len(cols2)), R, seed)
    else:
        counts, la = _draws(sizes, (), R, seed)
    A1 = M1[0, cols1][None, :] if det else counts[0] @ M1[:, cols1] / M1.shape[0]
    A2 = M2[0, cols2][None, :] if det else counts[0] @ M2[:, cols2] / M2.shape[0]
    if level == "hier":
        A1, A2 = np.broadcast_to(A1, (R, len(cols1))), np.broadcast_to(A2, (R, len(cols2)))
        stat = np.take_along_axis(A1, la[0], 1).mean(1) - np.take_along_axis(A2, la[1], 1).mean(1)
    else:
        stat = A1.mean(1) - A2.mean(1)
    lo, hi = _pct(stat)
    return {"mean": mean, "lo95": lo, "hi95": hi}


# ---------------------------------------------------------------------------------------------------------------------- reading the raw files
def read_session(session: Path, allow_partial: bool = False) -> dict:
    session = Path(session)
    man = OrderedDict()
    for ln in (session / "manifest.jsonl").read_text().splitlines():
        if ln.strip():
            e = json.loads(ln)
            if e["cell"] in man:
                raise AnalysisError(f"duplicate manifest line for {e['cell']}")
            man[e["cell"]] = e
    cells = C.all_cells()
    if not allow_partial and [c["name"] for c in cells] != list(man):
        raise AnalysisError("the manifest does not list every cell exactly once in the pinned order")
    rows, files = [], []
    for name, e in man.items():
        p = session / e["path"]
        if not p.exists() or sha256_file(p) != e["sha256"]:
            raise AnalysisError(f"{p} is missing or does not hash to its manifest line")
        with open(p, newline="") as f:
            rs = list(csv.DictReader(f))
        if len(rs) != e["rows"]:
            raise AnalysisError(f"{p}: {len(rs)} rows, manifest says {e['rows']}")
        rows += rs
        files.append({"cell": name, "sha256": e["sha256"], "rows": e["rows"], "commit": e["commit"]})
    sessions = sorted({r["session_id"] for r in rows})
    if len(sessions) != 1:
        raise AnalysisError(f"raw rows carry {len(sessions)} session ids: {sessions}")
    return {"rows": rows, "files": files, "session_id": sessions[0], "commits": sorted({e["commit"] for e in man.values()})}


class Data:
    """Per-condition matrices (seeds x layouts) of every metric, built from the raw rows."""

    METRIC_KEYS = ("breach", "score", "d", "n_bought", "n_deployed", *C.ARMY_COLS)

    def __init__(self, rows: list):
        first = [r for r in rows if r["cell"] == rows[0]["cell"]]
        self.layouts = OrderedDict()                                                      # id -> (group, family)
        for r in first:
            self.layouts[r["id"]] = (r["group"], r["family"])
        self.col = {i: k for k, i in enumerate(self.layouts)}
        L = len(self.layouts)
        cond_seeds: dict = {}
        for r in rows:
            cid = C.cond_id(r["condition"], r["N"] if r["N"] != "" else None)
            cond_seeds.setdefault(cid, set()).add(int(r["learner_seed"]) if r["learner_seed"] != "" else 0)
        self.seeds = {c: sorted(s) for c, s in cond_seeds.items()}
        self.val = {c: {m: np.full((len(s), L), np.nan) for m in self.METRIC_KEYS} for c, s in self.seeds.items()}
        self.army_ok = {c: True for c in self.seeds}
        for r in rows:
            cid = C.cond_id(r["condition"], r["N"] if r["N"] != "" else None)
            si = self.seeds[cid].index(int(r["learner_seed"]) if r["learner_seed"] != "" else 0)
            li = self.col[r["id"]]
            if self.layouts[r["id"]] != (r["group"], r["family"]):
                raise AnalysisError(f"layout {r['id']} changes group / family between cells")
            for m in self.METRIC_KEYS:
                if r[m] == "":
                    if m in C.ARMY_COLS:
                        self.army_ok[cid] = False
                        continue
                    raise AnalysisError(f"blank {m} in {r['cell']}")
                self.val[cid][m][si, li] = float(r[m])
        for c, mm in self.val.items():
            for m, a in mm.items():
                if np.isnan(a).any() and not (m in C.ARMY_COLS and not self.army_ok[c]):
                    raise AnalysisError(f"condition {c} has missing {m} values (incomplete cells)")
        grp = np.array([g for g, _ in self.layouts.values()])
        fam = np.array([f for _, f in self.layouts.values()])
        self.groups = OrderedDict()
        for g in C.GROUPS:
            self.groups[g] = np.flatnonzero(grp == g)
        self.groups["POOLED (descriptive)"] = np.arange(L)
        for g in C.GROUPS:
            for f in OrderedDict.fromkeys(fam[grp == g]):
                self.groups[f"{g} / {f}"] = np.flatnonzero((grp == g) & (fam == f))

    def arm(self, cid: str, metric: str) -> tuple:
        return self.val[cid][metric], cid in C.DETERMINISTIC


# ---------------------------------------------------------------------------------------------------------------------- tables
def condition_rows(D: Data) -> list:
    out = []
    for gname, cols in D.groups.items():
        for cid in D.val:
            row = {"group": gname, "n_layouts": len(cols), "condition": cid, "label": C.cond_label(cid), "n_seeds": len(D.seeds[cid])}
            for m in METRICS:
                M, det = D.arm(cid, m)
                h = interval([(M, det)], [1], cols, level="hier")
                s = interval([(M, det)], [1], cols, level="seeds")
                row.update({f"{m}": h["mean"], f"{m}_lo95": h["lo95"], f"{m}_hi95": h["hi95"], f"{m}_learner_only_lo95": s["lo95"], f"{m}_learner_only_hi95": s["hi95"]})
            for m in ("d", "n_bought", "n_deployed"):
                row[f"{m}_mean"] = float(D.val[cid][m][:, cols].mean())
            out.append(row)
    return out


def army_rows(D: Data) -> list:
    out = []
    for gname, cols in D.groups.items():
        for cid in D.val:
            if not D.army_ok[cid]:
                continue
            row = {"group": gname, "condition": cid, "label": C.cond_label(cid), "n_layouts": len(cols)}
            for a in C.ARMY_COLS:
                row[a] = float(D.val[cid][a][:, cols].mean())
            row["n_bought"], row["n_deployed"] = float(D.val[cid]["n_bought"][:, cols].mean()), float(D.val[cid]["n_deployed"][:, cols].mean())
            out.append(row)
    return out


def strongest_controls(D: Data) -> tuple:
    """For each (group, N): the control with the highest mean TRUE SCORE among RAS, RPS, RPS-full (ties -> the earlier of that order)."""
    choice, rows = {}, []
    for g in C.GROUPS:
        for n in C.N_VALUES:
            means = {m: float(D.val[C.cond_id(m, n)]["score"][:, D.groups[g]].mean()) for m in CONTROL_ORDER}
            best = max(CONTROL_ORDER, key=lambda m: (means[m], -CONTROL_ORDER.index(m)))
            choice[(g, n)] = best
            rows.append({"group": g, "N": n, **{f"{m}_score": means[m] for m in CONTROL_ORDER},
                         **{f"{m}_breach": float(D.val[C.cond_id(m, n)]["breach"][:, D.groups[g]].mean()) for m in CONTROL_ORDER}, "strongest": best})
    return choice, rows


def pair(D: Data, gname: str, metric: str, a: str, b: str, shared: bool) -> dict:
    """Paired difference a - b on one group: primary (hierarchical) and secondary (learner-only) intervals."""
    cols = D.groups[gname]
    Ma, deta = D.arm(a, metric)
    Mb, detb = D.arm(b, metric)
    sg = [None if deta else 0, None if detb else (0 if shared else 1)]        # shared: the same trained models, ONE seed draw; else independent draws (A first)
    h = interval([(Ma, deta), (Mb, detb)], [1, -1], cols, sg, "hier")
    s = interval([(Ma, deta), (Mb, detb)], [1, -1], cols, sg, "seeds")
    return {"diff": h["mean"], "lo95": h["lo95"], "hi95": h["hi95"], "learner_only_lo95": s["lo95"], "learner_only_hi95": s["hi95"],
            "mean_a": float(Ma[:, cols].mean()), "mean_b": float(Mb[:, cols].mean())}


def comparison_rows(D: Data, choice: dict) -> tuple:
    prim, expl = [], []

    def add(dst, cid, name, a, b, shared, status, groups=C.GROUPS):
        for g in groups:
            for m in METRICS:
                r = pair(D, g, m, a, b, shared)
                dst.append({"id": cid, "status": status, "comparison": name, "A": a, "B": b, "group": g, "metric": m, "shared_seeds": int(shared), **r,
                            "holds": int(r["lo95"] > 0) if status == "PRIMARY" else "", "verdict": ("holds" if r["lo95"] > 0 else "not established") if status == "PRIMARY" else "exploratory"})
    for c in LEARNER_BASES:
        add(prim, "P1", f"{C.LABEL[c]} beats BestSingle", c, "bestsingle", False, "PRIMARY")
    for c in LEARNER_BASES:
        add(prim, "P2", f"{C.LABEL[c]} beats Greedy", c, "greedy", False, "PRIMARY")
    add(prim, "P3", "PPO+D4 beats PPO plain", "ppo_d4", "ppo_plain", False, "PRIMARY")
    for pl in C.PLANNERS:
        for n in C.N_VALUES:
            for g in C.GROUPS:                                                                # the strongest control is chosen per (group, N): its own add() call
                ctrl = choice[(g, n)]
                add(prim, "P4", f"Planner on {C.LABEL[pl]} N={n} beats the strongest equal-N control ({C.LABEL[ctrl]}-{n})", f"planner_{pl}_{n}", C.cond_id(ctrl, n), False, "PRIMARY", (g,))
    for c in LEARNER_BASES:
        add(prim, "P5", f"{C.LABEL[c]} + TTA beats {C.LABEL[c]}", c + "_tta", c, True, "PRIMARY")
    for pl in C.PLANNERS:
        for n in C.N_VALUES:
            add(expl, "E1", f"Planner on {C.LABEL[pl]} N={n} vs its own learner", f"planner_{pl}_{n}", pl, True, "EXPLORATORY")
        for n1, n2 in ((32, 8), (128, 32)):
            add(expl, "E2", f"Planner on {C.LABEL[pl]} N={n1} vs N={n2}", f"planner_{pl}_{n1}", f"planner_{pl}_{n2}", True, "EXPLORATORY")
    add(expl, "E3", "DDQN vs PPO plain", "ddqn", "ppo_plain", False, "EXPLORATORY")
    add(expl, "E3", "DDQN vs PPO+D4", "ddqn", "ppo_d4", False, "EXPLORATORY")
    return prim, expl


def transfer_rows(D: Data) -> list:
    """E4 (exploratory): FINAL-transfer minus FINAL-familiar per condition (two independent layout sets, shared seeds)."""
    out = []
    for cid in (*LEARNER_BASES, "greedy", "bestsingle", "random", "oracle"):
        for m in METRICS:
            M, det = D.arm(cid, m)
            h = interval_two_sets(M, D.groups["FINAL-transfer"], M, D.groups["FINAL-familiar"], det, "hier")
            s = interval_two_sets(M, D.groups["FINAL-transfer"], M, D.groups["FINAL-familiar"], det, "seeds")
            out.append({"id": "E4", "status": "EXPLORATORY", "condition": cid, "label": C.cond_label(cid), "metric": m, "transfer_minus_familiar": h["mean"],
                        "lo95": h["lo95"], "hi95": h["hi95"], "learner_only_lo95": s["lo95"], "learner_only_hi95": s["hi95"]})
    return out


def gap_rows(D: Data, seen_path: Path, b8_path: Path) -> list:
    """Descriptive: mean DEV-train in-sample score (150 layouts) minus FINAL-familiar score per learner condition, hierarchical CI with the two layout
    sets resampled independently (one shared seed draw); the baselines' own DEV-train minus FINAL-familiar change next to it. Not a test."""
    seen = list(csv.DictReader(open(seen_path)))
    ids = list(OrderedDict.fromkeys(r["id"] for r in seen if r["set"] == "DEV-train"))
    out = []
    fam_cols = D.groups["FINAL-familiar"]
    for cid in LEARNER_BASES:
        rs = [r for r in seen if r["set"] == "DEV-train" and r["cond"] == cid]
        seeds = sorted({int(r["seed"]) for r in rs})
        M1 = np.full((len(seeds), len(ids)), np.nan)
        for r in rs:
            M1[seeds.index(int(r["seed"])), ids.index(r["id"])] = float(r["score"])
        if np.isnan(M1).any() or len(seeds) != len(D.seeds[cid]):
            raise AnalysisError(f"seen rows for {cid} are incomplete")
        M2 = D.val[cid]["score"]
        h = interval_two_sets(M1, np.arange(len(ids)), M2, fam_cols, False, "hier")
        s = interval_two_sets(M1, np.arange(len(ids)), M2, fam_cols, False, "seeds")
        out.append({"kind": "learner", "condition": cid, "label": C.cond_label(cid), "dev_train_score": float(M1.mean()), "final_familiar_score": float(M2[:, fam_cols].mean()),
                    "gap": h["mean"], "lo95": h["lo95"], "hi95": h["hi95"], "learner_only_lo95": s["lo95"], "learner_only_hi95": s["hi95"]})
    b8 = list(csv.DictReader(open(b8_path)))
    for pol, cid in (("Random", "random"), ("Greedy", "greedy"), ("BestSingle", "bestsingle"), ("Oracle", "oracle")):
        rs = [r for r in b8 if r["set"] == "DEV-train" and r["policy"] == pol]
        by = {r["id"]: float(r["score"]) for r in rs}
        M1 = np.array([[by[i] for i in ids]])
        M2 = D.val[cid]["score"]
        h = interval_two_sets(M1, np.arange(len(ids)), M2, fam_cols, True, "hier")
        out.append({"kind": "baseline (difficulty reference)", "condition": cid, "label": C.cond_label(cid), "dev_train_score": float(M1.mean()), "final_familiar_score": float(M2[:, fam_cols].mean()),
                    "gap": h["mean"], "lo95": h["lo95"], "hi95": h["hi95"], "learner_only_lo95": "", "learner_only_hi95": ""})
    return out


# ---------------------------------------------------------------------------------------------------------------------- output
def write_csv(path: Path, rows: list, banner: str) -> None:
    keys = list(OrderedDict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        f.write(f"# {banner}\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fnum(x, k=4) -> str:
    return f"{x:.{k}f}" if isinstance(x, float) else str(x)


def md_table(rows: list, cols: list, heads: list | None = None) -> str:
    out = ["| " + " | ".join(heads or cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fnum(r[c]) if c in r else "" for c in cols) + " |")
    return "\n".join(out)


def ci(r: dict, m: str) -> str:
    return f"{r[m]:.3f} [{r[m + '_lo95']:.3f}, {r[m + '_hi95']:.3f}]"


def analyze(session: Path, out_dir: Path, label: str | None = None, seen_path: Path = SEEN_ROWS, b8_path: Path = B8_ROWS, allow_partial: bool = False) -> dict:
    session, out_dir = Path(session), Path(out_dir)
    S = read_session(session, allow_partial)
    D = Data(S["rows"])
    banner = (f"{label} -- NOT RESULTS. " if label else "") + f"session {S['session_id']}; commits {', '.join(c[:7] for c in S['commits'])}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = condition_rows(D)
    choice, ctrl_rows = strongest_controls(D)
    prim, expl = comparison_rows(D, choice)
    trans = transfer_rows(D)
    gap = gap_rows(D, seen_path, b8_path)
    army = army_rows(D)
    write_csv(out_dir / "conditions.csv", cond, banner)
    write_csv(out_dir / "comparisons_primary.csv", prim, banner)
    write_csv(out_dir / "comparisons_exploratory.csv", expl + trans, banner)
    write_csv(out_dir / "strongest_control.csv", ctrl_rows, banner)
    write_csv(out_dir / "generalisation_gap.csv", gap, banner)
    write_csv(out_dir / "army_composition.csv", army, banner)
    inputs = {"raw_files": S["files"], "seen_layout_rows_sha256": sha256_file(seen_path), "b8_layout_rows_sha256": sha256_file(b8_path),
              "bootstrap": {"n_resamples": NRES, "rng": f"numpy default_rng({BSEED}) fresh per interval", "ci": "percentile 2.5 / 97.5"}, "session_id": S["session_id"], "commits": S["commits"]}
    (out_dir / "analysis_inputs.json").write_text(json.dumps(inputs, indent=1, sort_keys=True))
    L = [f"# B15 tables {'(' + label + ', never results)' if label else '(FINAL, single session)'}", "", f"> {banner}", "",
         "Primary interval: hierarchical bootstrap (seeds, then layouts; 10 000 resamples; percentile 95 %; rng 8080 per interval). Learner-only interval: SECONDARY, never decides 'holds'. "
         "A primary comparison holds only if its paired-difference primary CI excludes 0. Everything outside P1-P5 is exploratory (no multiplicity correction).", ""]
    L += ["## Primary comparisons (P1-P5): 16 x 2 groups x 2 metrics", "",
          md_table(prim, ["id", "comparison", "group", "metric", "mean_a", "mean_b", "diff", "lo95", "hi95", "learner_only_lo95", "learner_only_hi95", "verdict"],
                   ["id", "comparison", "group", "metric", "A", "B", "A - B", "lo95", "hi95", "learner-only lo95", "learner-only hi95", "verdict"]), ""]
    L += ["## Strongest equal-N control (rule: highest mean true score at that group and N; all three shown)", "",
          md_table(ctrl_rows, ["group", "N", *[f"{m}_score" for m in CONTROL_ORDER], *[f"{m}_breach" for m in CONTROL_ORDER], "strongest"]), ""]
    for g in (*C.GROUPS, "POOLED (descriptive)"):
        rs = [r for r in cond if r["group"] == g]
        L += [f"## Conditions: {g} ({rs[0]['n_layouts']} layouts)", "",
              "| condition | seeds | breach rate [95% CI] | breach learner-only CI | mean score [95% CI] | score learner-only CI | mean D | bought | deployed |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['label']} | {r['n_seeds']} | {ci(r, 'breach')} | [{r['breach_learner_only_lo95']:.3f}, {r['breach_learner_only_hi95']:.3f}] | {ci(r, 'score')} | "
                     f"[{r['score_learner_only_lo95']:.3f}, {r['score_learner_only_hi95']:.3f}] | {r['d_mean']:.3f} | {r['n_bought_mean']:.1f} | {r['n_deployed_mean']:.1f} |")
        L.append("")
    L += ["## Per family (every cell is also reported per family)", ""]
    for g in [k for k in D.groups if " / " in k]:
        rs = [r for r in cond if r["group"] == g]
        L += [f"### {g} ({rs[0]['n_layouts']} layouts)", "", "| condition | breach rate [95% CI] | mean score [95% CI] |", "|---|---|---|"]
        L += [f"| {r['label']} | {ci(r, 'breach')} | {ci(r, 'score')} |" for r in rs] + [""]
    L += ["## Generalisation gap (descriptive, not a test): DEV-train in-sample score minus FINAL-familiar score", "",
          md_table(gap, ["kind", "label", "dev_train_score", "final_familiar_score", "gap", "lo95", "hi95", "learner_only_lo95", "learner_only_hi95"]), ""]
    L += ["## Exploratory comparisons (no multiplicity correction)", "", md_table(expl, ["id", "comparison", "group", "metric", "diff", "lo95", "hi95", "learner_only_lo95", "learner_only_hi95"]), "",
          "### E4: FINAL-transfer minus FINAL-familiar", "", md_table(trans, ["label", "metric", "transfer_minus_familiar", "lo95", "hi95", "learner_only_lo95", "learner_only_hi95"]), ""]
    L += ["## Army composition (mean units bought per type; Random and Oracle carry no army)", "",
          md_table([r for r in army if r["group"] in (*C.GROUPS, "POOLED (descriptive)")], ["group", "label", *C.ARMY_COLS, "n_bought", "n_deployed"]), ""]
    (out_dir / "tables.md").write_text("\n".join(L) + "\n")
    return {"conditions": cond, "primary": prim, "exploratory": expl + trans, "gap": gap, "strongest": ctrl_rows, "inputs": inputs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=str(ROOT / "results" / "b15"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help="e.g. REHEARSAL: stamped on every table")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.session) / "tables"
    r = analyze(Path(a.session), out, a.label)
    n_hold = sum(1 for x in r["primary"] if x["holds"] == 1)
    print(f"{len(r['primary'])} primary rows ({n_hold} hold), tables in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
