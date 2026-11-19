"""B14 grid launcher / job queue / status (SPEC 11 B14; Gate changelog rows (dd) and (ee)). Written in B14a, NOT started by B14a.

    .venv/bin/python scripts/b14_grid.py plan                  # preflight + the 30 commands + disk and time estimate; starts nothing
    .venv/bin/python scripts/b14_grid.py launch                # preflight, then start the queue detached under caffeinate; run it again after a reboot = resume
    .venv/bin/python scripts/b14_grid.py status [--watch]      # per-job progress, slots, ETA
    .venv/bin/python scripts/b14_grid.py stop                  # SIGTERM the scheduler and the trainers (all resumable)

Grid (frozen in data/v2/grid_config_*.json, row (ee)): PPO plain x10, PPO+D4 x10, DDQN (train_every 8) x10, seeds 0-9, shaping on, T = 1 500 000, validation on
DEV-validation every 25k, final model at T (nothing selected). Queue: 6 slots, PPO first (seed ascending, plain then D4), then DDQN; a freed slot takes the next
job in that order. Every job is `... --resume`: it skips a finished run, continues from the latest checkpoint, or starts fresh. A reboot / sleep / kill loses at
most the steps since the last checkpoint; `launch` again continues.

Safety: refuses unless data/v2/gates.json g7.pass is true and both report hashes match their files; rows (bb), (cc), (dd), (ee), (ff) and every config sha are in the
SPEC; the configs equal the code constants; the tracked tree is clean; no foreign trainer / pilot / second scheduler is running; the FINAL ledger has no eval
entry; no stats / generator override is set; code under rl/ breachlens/ has not changed since the first launch (unless --accept-code-change); free disk is at
least the remaining expected peak + 10 GB headroom (row (ff)). One scheduler (flock) and a process-table scan before every start: a running job is never started twice (orphans of a
dead scheduler are adopted, not duplicated). At the end of every job the layout manifest is checked (breachlens.manifest: FINAL / transfer / LAB-DESIGN ids or
streams fail it), the evaluated layout ids must be inside the manifest, and checkpoint retention (latest 2 + final) is verified; results in logs_b14/<run>.result.json.

The real mode takes no overrides. `--smoke` (dry run) needs --models-dir / --logs-dir outside models_v2 / logs_b14 and prefixes every run name with `smoke_`.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scripts import b13_pilot as B

ROOT = B.ROOT
DATA = ROOT / "data" / "v2"
GATES = DATA / "gates.json"
STEP_BUDGET = DATA / "step_budget.json"
CONFIG_FILES = {"ppo_plain": "grid_config_ppo_plain.json", "ppo_d4": "grid_config_ppo_d4.json", "ddqn": "grid_config_ddqn.json"}
QUEUE_FILE = "grid_queue.json"
ROWS = ("(bb) **B13d item 0", "(cc) **B13d items", "(dd) **B14a item 0", "(ee) **B14a item 2", "(ff) **B14b")
REAL_LOGS = ROOT / "logs_b14"
TRAINER = {"ppo": "rl.train_ppo_v2", "ddqn": "rl.train_ddqn_v2"}
FINAL_FILE = {"ppo": "final.zip", "ddqn": "final.pt"}
CODE_PATHS = ["rl", "breachlens", "scripts/b14_grid.py", "scripts/b13_pilot.py",
              *[f"data/v2/{f}" for f in (*CONFIG_FILES.values(), QUEUE_FILE)]]
CKPT_KEEP = 2
# Measured on the pilots (bytes). PPO checkpoint dir 9 643 204 (D4 9 643 204 at 250k). DDQN checkpoint dir, B14b format (compressed buffer, row (ff)), measured at
# 250k with a full 50 000-transition buffer: nets.pt 8 694 389 + state.pkl 1 039 331 + buffer 1 477 636 = 11 211 356 (was 378 791 591). Final models (kept for every run):
# PPO final.zip 8.1 MB, DDQN final.pt 2.17 MB. Per-run files other than checkpoint and final model (csv / jsonl / identity): PPO 0.6 MB, DDQN 0.3 MB. All rounded up.
SIZES = {"ppo_ckpt": 9_700_000, "ppo_final": 8_500_000, "ppo_other": 1_500_000, "ddqn_ckpt": 12_000_000, "ddqn_final": 2_500_000, "ddqn_other": 1_500_000,
         "ddqn_buffer_ref": 50_000, "ddqn_ckpt_fixed": 10_500_000}
DISK_HEADROOM = 10_000_000_000       # row (ff): free >= remaining expected peak + 10 GB (macOS swap), replacing the 1.5 x rule
# Rates (steps/s per job) used for the wall-time ESTIMATE only. PPO: row (cc) r = 184.3 at P=6 (shaped); last wave of 2 at ~330. DDQN train_every 8: pilot 123.2
# (4 processes, then 2; includes validation); at P=6 scaled from B11 (43.2 at P=6, train_every 4) x the measured f8/f4 pilot ratio 123.2 / 73.6 = 1.67.
RATES = {"ppo_p6": 184.3, "ppo_p2": 330.0, "ddqn_p6": 72.0, "ddqn_p4": 105.0, "ddqn_pilot": 123.2}


# ------------------------------------------------------------------------------------------------ configs and jobs
def load_config(cond: str) -> dict:
    return json.loads((DATA / CONFIG_FILES[cond]).read_text())["config"]


def load_queue() -> dict:
    return json.loads((DATA / QUEUE_FILE).read_text())["config"]


def config_shas() -> dict:
    out = {c: B.config_sha(load_config(c)) for c in CONFIG_FILES}
    out["queue"] = B.config_sha(load_queue())
    return out


def configs_vs_code(cfgs: dict, queue: dict) -> list:
    """Problems if a frozen file differs from the code constants (PPO_CONFIG, DDQN_CONFIG with train_every 8) or from the grid design."""
    from rl import ddqn as D
    from rl import train_ppo_v2 as TP
    out = []
    for c, cfg in cfgs.items():
        if cfg["condition"] != c:
            out.append(f"{c}: condition field is {cfg['condition']!r}")
        if (cfg["train_set"], cfg["val_set"]) != ("DEV-train", "DEV-validation"):
            out.append(f"{c}: sets {cfg['train_set']} / {cfg['val_set']} are not DEV-train / DEV-validation")
        if cfg["timesteps"] != 1_500_000 or cfg["curriculum"]["phase_a_steps"] != cfg["timesteps"] // 3 or cfg["curriculum"]["mode"] != "on":
            out.append(f"{c}: timesteps / curriculum differ from T = 1.5M, phase A = T / 3")
        if cfg["seeds"] != list(range(10)) or cfg["val_freq"] != 25_000 or cfg["checkpoint_freq"] != 50_000 or cfg["checkpoint_keep"] != CKPT_KEEP:
            out.append(f"{c}: seeds / val_freq / checkpoint settings differ from the design")
        if not str(cfg["shaping"]).startswith("on") or cfg["torch_threads"] != 1:
            out.append(f"{c}: shaping not on or threads != 1")
    if cfgs["ppo_plain"]["ppo"] != json.loads(json.dumps(TP.PPO_CONFIG)) or cfgs["ppo_d4"]["ppo"] != json.loads(json.dumps(TP.PPO_CONFIG)):
        out.append("a PPO config differs from rl.train_ppo_v2.PPO_CONFIG")
    if (cfgs["ppo_plain"]["aug"], cfgs["ppo_d4"]["aug"], cfgs["ddqn"]["aug"]) != ("none", "d4", "none"):
        out.append("aug fields are not none / d4 / none")
    if cfgs["ddqn"]["ddqn"] != json.loads(json.dumps(dict(D.DDQN_CONFIG, train_every=8))):
        out.append("the DDQN config differs from rl.ddqn.DDQN_CONFIG with train_every 8")
    if queue["slots"] != 6 or queue["retention"]["resumable_checkpoints_kept"] != CKPT_KEEP:
        out.append("queue file: slots != 6 or retention != 2")
    return out


@dataclass(frozen=True)
class Job:
    name: str
    cond: str
    seed: int

    @property
    def algo(self) -> str:
        return "ddqn" if self.cond == "ddqn" else "ppo"


def job_list(cfgs: dict, seeds=None, prefix: str = "") -> list:
    """Queue order (row ee): PPO first, seed ascending, plain then D4 within a seed; then DDQN, seed ascending."""
    seeds = list(cfgs["ppo_plain"]["seeds"] if seeds is None else seeds)
    jobs = [Job(prefix + cfgs[c]["run_names"].format(seed=s), c, s) for s in seeds for c in ("ppo_plain", "ppo_d4")]
    return jobs + [Job(prefix + cfgs["ddqn"]["run_names"].format(seed=s), "ddqn", s) for s in seeds]


@dataclass(frozen=True)
class Overrides:
    """Smoke-only: tiny steps / frequencies / hyperparameters. The real mode is Overrides() (all None / empty)."""
    steps: int | None = None
    val_freq: int | None = None
    checkpoint_freq: int | None = None
    speed_freq: int | None = None
    hp: tuple = ()          # (("n_steps", 50), ...)

    def any(self) -> bool:
        return bool(self.steps or self.val_freq or self.checkpoint_freq or self.speed_freq or self.hp)


def target_steps(cfgs: dict, ov: Overrides) -> int:
    return int(ov.steps or cfgs["ppo_plain"]["timesteps"])


def build_argv(job: Job, cfgs: dict, ov: Overrides = Overrides()) -> list:
    """argv of one job. Every hyperparameter is passed explicitly from the frozen config; `--resume` makes it restartable."""
    cfg = cfgs[job.cond]
    steps = target_steps(cfgs, ov)
    phase_a = steps // 3 if ov.steps else cfg["curriculum"]["phase_a_steps"]
    hp = dict(cfg["ddqn" if job.algo == "ddqn" else "ppo"])
    for k, v in ov.hp:
        if k in hp:
            hp[k] = v
    a = [sys.executable, "-m", TRAINER[job.algo], "--train-set", cfg["train_set"], "--seed", str(job.seed), "--run-name", job.name,
         "--timesteps", str(steps), "--val-freq", str(ov.val_freq or cfg["val_freq"]), "--checkpoint-freq", str(ov.checkpoint_freq or cfg["checkpoint_freq"]),
         "--curriculum", cfg["curriculum"]["mode"], "--phase-a-steps", str(phase_a), "--shaping", "on", "--threads", str(cfg["torch_threads"]), "--resume"]
    if job.algo == "ppo":
        a += ["--aug", cfg["aug"], "--speed-freq", str(ov.speed_freq or cfg["speed_freq"])]
    for k, v in hp.items():
        if k != "net_arch":
            a += ["--" + k.replace("_", "-"), str(v)]
    return a


# ------------------------------------------------------------------------------------------------ process table
def _is_shell(tok) -> bool:
    return os.path.basename(tok[1]) in ("zsh", "bash", "sh", "grep", "pgrep")


def running_jobs(ps: str, names: set, own_pid: int | None = None) -> dict:
    """{run name: pid} of trainer processes of THIS grid (a `--run-name` in `names` on a rl.train_* command line)."""
    out = {}
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 2 or not tok[0].isdigit() or int(tok[0]) == own_pid or _is_shell(tok):
            continue
        if "--run-name" in tok and any(t in tok for t in TRAINER.values()):
            n = tok[tok.index("--run-name") + 1]
            if n in names:
                out[n] = int(tok[0])
    return out


def foreign_processes(ps: str, names: set, own_pid: int | None = None) -> list:
    """Trainers of other runs, any pilot launcher, or another copy of this script's scheduler (`run`) that is not ours."""
    out = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 2 or not tok[0].isdigit() or int(tok[0]) == own_pid or _is_shell(tok):
            continue
        if any(t in tok for t in TRAINER.values()):
            n = tok[tok.index("--run-name") + 1] if "--run-name" in tok else "?"
            if n not in names:
                out.append(ln.strip())
        elif any(t.endswith(("scripts/b13_pilot.py", "scripts/b13_ddqn_pilot.py")) for t in tok):
            out.append(ln.strip())
    return out


class SchedulerLock:
    """Exclusive flock on <logs>/queue.lock; held for the life of the scheduler process (released by the OS if it dies)."""

    def __init__(self, logs: Path) -> None:
        logs.mkdir(parents=True, exist_ok=True)
        self.path, self.fd = logs / "queue.lock", None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self.fd = fd
        return True

    def held_by_other(self) -> int | None:
        """pid of the live scheduler, or None (probe without keeping the lock)."""
        if not self.path.exists():
            return None
        fd = os.open(self.path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        except OSError:
            try:
                return int(self.path.read_text().strip() or 0) or -1
            except ValueError:
                return -1
        finally:
            os.close(fd)


# ------------------------------------------------------------------------------------------------ disk
def ckpt_bytes(algo: str, buffer_size: int | None = None) -> int:
    if algo == "ppo":
        return SIZES["ppo_ckpt"]
    b = SIZES["ddqn_buffer_ref"] if buffer_size is None else buffer_size
    per = (SIZES["ddqn_ckpt"] - SIZES["ddqn_ckpt_fixed"]) / SIZES["ddqn_buffer_ref"]
    return int(SIZES["ddqn_ckpt_fixed"] + per * b)


def peak_bytes(n: dict, slots: int = 6, ddqn_buffer: int | None = None) -> dict:
    """Upper bound on grid disk use, per algorithm: with N runs of an algorithm, F finished (each keeps 2 checkpoints) and R running (up to 3 checkpoint dirs
    each while a new one is being written: 2 kept + the new one), F + R <= N, R <= slots  =>  max (2F + 3R) = 2N + R checkpoint dirs, R = min(slots, N); plus the
    final model and other files of all N runs. PPO and DDQN peaks are SUMMED (conservative: they overlap only in the queue's tail)."""
    out = {}
    for algo, key in (("ppo", "ppo"), ("ddqn", "ddqn")):
        N = n.get(algo, 0)
        R = min(slots, N)
        ck = ckpt_bytes(algo, ddqn_buffer if algo == "ddqn" else None)
        out[algo] = (2 * N + R) * ck + N * (SIZES[f"{key}_other"] + SIZES[f"{key}_final"])
    out["total"] = out["ppo"] + out["ddqn"]
    return out


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) if Path(path).exists() else 0


def disk_problem(total_peak: int, used_by_grid: int, free: int) -> str | None:
    need = max(0, total_peak - used_by_grid)
    if free < need + DISK_HEADROOM:
        return (f"free disk {free / 1e9:.1f} GB < remaining expected peak {need / 1e9:.1f} GB + {DISK_HEADROOM / 1e9:.0f} GB swap headroom "
                f"(= {(need + DISK_HEADROOM) / 1e9:.1f} GB; expected peak {total_peak / 1e9:.1f} GB, already used by the grid {used_by_grid / 1e9:.1f} GB)")
    return None


# ------------------------------------------------------------------------------------------------ time estimate (labelled estimate, not a measurement)
def time_estimate(n_ppo: int = 20, n_ddqn: int = 10, T: int = 1_500_000, slots: int = 6) -> dict:
    ppo_waves = [min(slots, n_ppo - i) for i in range(0, n_ppo, slots)]
    ppo_h = sum(T / (RATES["ppo_p6"] if w >= 3 else RATES["ppo_p2"]) for w in ppo_waves) / 3600
    d_waves = [min(slots, n_ddqn - i) for i in range(0, n_ddqn, slots)]
    ddqn_h = sum(T / (RATES["ddqn_p6"] if w >= 5 else RATES["ddqn_p4"]) for w in d_waves) / 3600
    low = n_ddqn * T / (slots * RATES["ddqn_pilot"]) / 3600
    return {"ppo_h": ppo_h, "ddqn_h": ddqn_h, "total_h": ppo_h + ddqn_h, "total_low_h": ppo_h + low, "total_high_h": ppo_h + ddqn_h * 1.25}


# ------------------------------------------------------------------------------------------------ preflight
def gate_problems(spec_text: str) -> list:
    out = []
    try:
        g = json.loads(GATES.read_text())["g7"]
    except Exception as e:
        return [f"data/v2/gates.json has no readable g7 entry ({e})"]
    if g.get("pass") is not True:
        out.append("data/v2/gates.json g7.pass is not true")
    if not g.get("shaping_adopted_for_grid"):
        out.append("g7.shaping_adopted_for_grid is not true")
    for rel, want in g.get("reports", {}).items():
        f = ROOT / rel
        if not f.exists():
            out.append(f"g7 report {rel} is missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != want:
            out.append(f"g7 report {rel} does not match the hash in gates.json")
    if not g.get("reports"):
        out.append("g7 entry lists no report hashes")
    return out


def resolve_commit(commit: str) -> str:
    """Translate a commit id recorded before the 2026-11-19 history rewrite (trailer edit only) to its current id via docs/commit_map.tsv."""
    mp = ROOT / "docs" / "commit_map.tsv"
    if mp.exists() and commit:
        for line in mp.read_text().splitlines():
            if line.startswith("#") or "\t" not in line:
                continue
            old, new = line.split("\t")
            if old.startswith(commit):
                return new
    return commit


def code_changed_since(commit: str) -> list:
    commit = resolve_commit(commit)
    r = subprocess.run(["git", "diff", "--name-only", commit, "HEAD", "--", *CODE_PATHS], cwd=ROOT, capture_output=True, text=True)
    return [x for x in r.stdout.split() if x] if r.returncode == 0 else [f"git diff failed for {commit}"]


def head_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_state(logs: Path) -> dict:
    p = logs / "grid_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(logs: Path, st: dict) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    tmp = logs / "grid_state.json.tmp"
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    os.replace(tmp, logs / "grid_state.json")


def preflight(models: Path, logs: Path, jobs: list, ov: Overrides, smoke: bool, accept_code_change: bool = False, own_pid: int | None = None,
              check_disk: bool = True) -> list:
    own_pid = os.getpid() if own_pid is None else own_pid
    problems = []
    d = B.dirty_tracked()
    if d:
        problems.append(f"git tree has {len(d)} modified tracked file(s): {d[:3]}")
    spec = B.SPEC.read_text()
    problems += gate_problems(spec)
    miss = [r for r in ROWS if r not in spec]
    if miss:
        problems.append(f"Gate changelog rows missing: {miss}")
    shas = config_shas()
    for k, s in shas.items():
        if s not in spec:
            problems.append(f"sha256 of the {k} config ({s[:12]}...) is not recorded in row (ee)")
    cfgs = {c: load_config(c) for c in CONFIG_FILES}
    problems += configs_vs_code(cfgs, load_queue())
    try:
        sb = json.loads(STEP_BUDGET.read_text())
        if sb.get("T") != cfgs["ppo_plain"]["timesteps"] or sb.get("stop_for_user_decision"):
            problems.append(f"data/v2/step_budget.json T={sb.get('T')} differs from the configs or says stop")
    except Exception as e:
        problems.append(f"data/v2/step_budget.json unreadable ({e})")
    if "a6d5bee3178bc8154eeec9d6213e26bf36c7af618b9b67cc47aa7f7cf825bd54" not in spec or not (ROOT / "results/b13/ddqn_pilot_report.txt").exists():
        problems.append("the DDQN pilot report (row dd) is missing or its sha is not recorded")
    names = {j.name for j in jobs}
    foreign = foreign_processes(B.ps_text(), names, own_pid)
    if foreign:
        problems.append(f"a foreign trainer / pilot process is running: {foreign[:2]}")
    n = B.eval_entries()
    if n:
        problems.append(f"the FINAL ledger has {n} eval entr{'y' if n == 1 else 'ies'}")
    bad = [k for k in B.FORBIDDEN_ENV if k in os.environ]
    if bad:
        problems.append(f"environment override set: {bad}")
    if smoke:
        if Path(models).resolve() == B.REAL_MODELS.resolve() or Path(logs).resolve() == REAL_LOGS.resolve():
            problems.append("--smoke must not use models_v2 / logs_b14")
    elif ov.any() or Path(models).resolve() != B.REAL_MODELS.resolve() or Path(logs).resolve() != REAL_LOGS.resolve():
        problems.append("the real grid takes no overrides (steps, frequencies, hyperparameters, directories); use --smoke")
    first = load_state(logs).get("first_launch")
    if first and not accept_code_change:
        ch = code_changed_since(first["commit"])
        if ch:
            problems.append(f"code changed since the first launch ({first['commit'][:7]}): {ch[:4]}; pass --accept-code-change to continue anyway")
    if check_disk:
        n_by = {"ppo": sum(j.algo == "ppo" for j in jobs), "ddqn": sum(j.algo == "ddqn" for j in jobs)}
        buf = dict(ov.hp).get("buffer_size")
        pk = peak_bytes(n_by, load_queue()["slots"], buf)
        used = sum(dir_bytes(Path(models) / j.name) for j in jobs)
        Path(models).mkdir(parents=True, exist_ok=True)
        pr = disk_problem(pk["total"], used, shutil.disk_usage(models).free)
        if pr:
            problems.append(pr)
    return problems


# ------------------------------------------------------------------------------------------------ run directory inspection
def run_dir(models: Path, job: Job) -> Path:
    return Path(models) / job.name


def _last_csv_t(path: Path, col: str) -> int:
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            return 0
        head = lines[0].split(",")
        return int(float(lines[-1].split(",")[head.index(col)]))
    except Exception:
        return 0


def latest_ckpts(d: Path) -> list:
    out = []
    for c in (d / "checkpoints").glob("ckpt_*") if (d / "checkpoints").exists() else []:
        if c.is_dir() and not c.name.endswith(".tmp"):
            try:
                out.append(int(c.name.split("_")[1]))
            except ValueError:
                pass
    return sorted(out)


def progress_t(models: Path, job: Job) -> int:
    d = run_dir(models, job)
    ident = d / "model_identity.json"
    if (d / FINAL_FILE[job.algo]).exists() and ident.exists():
        try:
            return int(json.loads(ident.read_text()).get("timesteps_trained", 0))
        except Exception:
            pass
    cands = [_last_csv_t(d / "val_eval.csv", "timestep"), _last_csv_t(d / "speed.csv", "timestep"), _last_csv_t(d / "progress.csv", "timestep")]
    cands += latest_ckpts(d)[-1:]
    return max(cands)


def last_val(models: Path, job: Job):
    p = run_dir(models, job) / "val_eval.csv"
    try:
        rows = [r for r in csv.DictReader(p.open()) if r["group"] == "ALL"]
        return (int(rows[-1]["timestep"]), float(rows[-1]["score_mean"])) if rows else None
    except Exception:
        return None


def check_job_outputs(models: Path, job: Job, target: int) -> dict:
    """End-of-job checker: (1) manifest.json passes breachlens.manifest.check_manifest (FINAL / transfer / LAB-DESIGN ids or streams, overlaps, exact sets);
    (2) every layout id the validator evaluated is inside the manifest; (3) the model at T exists and identity says finished at T; (4) retention: at most
    CKPT_KEEP resumable checkpoints, no leftover .tmp, the final model present."""
    from breachlens import manifest as MF
    d = run_dir(models, job)
    res = {"manifest_ok": False, "manifest_error": None, "val_ids_in_manifest": None, "final_ok": False, "retention_ok": False, "n_ckpt": None}
    try:
        m = MF.check_manifest_file(d / "manifest.json")
        res["manifest_ok"] = True
        res["n_train_ids"], res["n_val_ids"] = len(m["train_ids"]), len(m["val_ids"])
        vl = d / "val_layouts.jsonl"
        ids = {json.loads(ln)["id"] for ln in vl.read_text().splitlines() if ln.strip()} if vl.exists() else set()
        res["val_ids_in_manifest"] = ids <= set(m["val_ids"]) and bool(ids)
    except Exception as e:                                         # ManifestError, missing file, bad json: all are failures
        res["manifest_error"] = f"{type(e).__name__}: {e}"
    try:
        ident = json.loads((d / "model_identity.json").read_text())
        res["final_ok"] = (d / FINAL_FILE[job.algo]).exists() and ident.get("status") == "finished" and int(ident.get("timesteps_trained", -1)) == target
        res["commit"], res["git_dirty"], res["resumes"] = ident.get("git_commit"), ident.get("git_dirty"), len(ident.get("resumes", []))
    except Exception as e:
        res["final_error"] = f"{type(e).__name__}: {e}"
    ck = (d / "checkpoints")
    tmp = [c.name for c in ck.glob("*.tmp")] if ck.exists() else []
    res["n_ckpt"] = len(latest_ckpts(d))
    res["retention_ok"] = res["n_ckpt"] <= CKPT_KEEP and not tmp
    res["ok"] = bool(res["manifest_ok"] and res["val_ids_in_manifest"] and res["final_ok"] and res["retention_ok"])
    return res


def clean_tmp(models: Path, job: Job) -> None:
    ck = run_dir(models, job) / "checkpoints"
    if ck.exists():
        for c in ck.glob("*.tmp"):
            shutil.rmtree(c, ignore_errors=True)


def job_state(models: Path, logs: Path, job: Job, running: dict, failed: set) -> str:
    d = run_dir(models, job)
    rj = Path(logs) / f"{job.name}.result.json"
    if (d / FINAL_FILE[job.algo]).exists():
        if rj.exists():
            try:
                return "finished" if json.loads(rj.read_text()).get("checks", {}).get("ok") else "CHECK-FAILED"
            except Exception:
                return "CHECK-FAILED"
        return "finished (unchecked)"
    if job.name in running:
        return "running"
    if job.name in failed:
        return "FAILED"
    if d.exists() and (latest_ckpts(d) or (d / "model_identity.json").exists()):
        return "interrupted"
    return "queued"


# ------------------------------------------------------------------------------------------------ scheduler
class Scheduler:
    def __init__(self, models: Path, logs: Path, cfgs: dict, jobs: list, ov: Overrides, slots: int = 6, poll: float = 15.0, max_restarts: int = 3,
                 argv_fn=None, ps_fn=None) -> None:
        self.models, self.logs, self.cfgs, self.jobs, self.ov = Path(models), Path(logs), cfgs, jobs, ov
        self.slots, self.poll, self.max_restarts = slots, poll, max_restarts
        self.argv_fn = argv_fn or (lambda j: build_argv(j, cfgs, ov))
        self.ps_fn = ps_fn or B.ps_text
        self.target = target_steps(cfgs, ov)
        self.procs: dict = {}          # name -> Popen (our own children)
        self.attempts: dict = {}
        self.failed: set = set()
        self.names = {j.name for j in jobs}
        self.stop = False
        self.adopted: set = set()

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)

    # -- state file
    def _state(self) -> dict:
        return load_state(self.logs)

    def mark_segment(self, job: Job) -> None:
        st = self._state()
        st.setdefault("segments", {})[job.name] = {"start_time": time.time(), "start_t": progress_t(self.models, job)}
        st["attempts"] = self.attempts
        save_state(self.logs, st)

    # -- one job
    def start(self, job: Job) -> bool:
        if job.name in running_jobs(self.ps_fn(), self.names, os.getpid()):             # never a second copy
            self.log(f"[skip-start] {job.name} is already running")
            return False
        clean_tmp(self.models, job)
        argv = self.argv_fn(job)
        n = self.attempts.get(job.name, 0)
        lf = open(self.logs / f"{job.name}.log", "ab")
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start (attempt {n + 1}, resumes from checkpoint t={(latest_ckpts(run_dir(self.models, job)) or [0])[-1]}): {' '.join(argv)}\n".encode())
        lf.flush()
        env = dict(os.environ, BREACHLENS_MODELS_DIR=str(self.models), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        self.mark_segment(job)
        self.procs[job.name] = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        self.log(f"[start] {job.name} pid {self.procs[job.name].pid} (resumes from checkpoint t={(latest_ckpts(run_dir(self.models, job)) or [0])[-1]})")
        return True

    def finish(self, job: Job, code) -> None:
        clean_tmp(self.models, job)
        d = run_dir(self.models, job)
        if (d / FINAL_FILE[job.algo]).exists():
            checks = check_job_outputs(self.models, job, self.target)
            res = {"name": job.name, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "exit_code": code, "checks": checks}
            (self.logs / f"{job.name}.result.json").write_text(json.dumps(res, indent=1))
            self.log(f"[done] {job.name} exit {code}; checks ok={checks['ok']} manifest_ok={checks['manifest_ok']} retention_ok={checks['retention_ok']} (n_ckpt {checks['n_ckpt']})")
            return
        self.attempts[job.name] = self.attempts.get(job.name, 0) + 1
        self.log(f"[exit] {job.name} ended with code {code} and no final model; attempt {self.attempts[job.name]} of {self.max_restarts}")
        if self.attempts[job.name] >= self.max_restarts:
            self.failed.add(job.name)
            self.log(f"[FAILED] {job.name}: restart limit reached; see {self.logs / (job.name + '.log')}")
        st = self._state()
        st["attempts"], st["failed"] = self.attempts, sorted(self.failed)
        save_state(self.logs, st)

    # -- loop
    def tick(self) -> bool:
        """One scheduling pass; True while there is still work (running or startable jobs)."""
        by_name = {j.name: j for j in self.jobs}
        for name, p in list(self.procs.items()):
            code = p.poll()
            if code is not None:
                del self.procs[name]
                self.finish(by_name[name], code)
        running = running_jobs(self.ps_fn(), self.names, os.getpid())
        for name in list(running):
            if name not in self.procs and name in by_name:
                self.adopt(by_name[name])
        adopted = {n for n in self.adopted if n not in running}                          # orphans that have just ended
        for n in adopted:
            self.adopted.discard(n)
            self.finish(by_name[n], "?")
        running = running_jobs(self.ps_fn(), self.names, os.getpid())
        pending = [j for j in self.jobs if job_state(self.models, self.logs, j, running, self.failed) in ("queued", "interrupted") and j.name not in self.procs]
        if self.stop:
            return bool(running) or bool(self.procs)
        free = self.slots - len(running)
        for job in pending:
            if free <= 0:
                break
            if self.start(job):
                free -= 1
        running = running_jobs(self.ps_fn(), self.names, os.getpid())
        return bool(self.procs) or bool(running) or any(job_state(self.models, self.logs, j, running, self.failed) in ("queued", "interrupted") for j in self.jobs)

    def adopt(self, job: Job) -> None:
        if job.name not in self.adopted:
            self.adopted.add(job.name)
            self.log(f"[adopt] {job.name} is running without us (orphan of an earlier scheduler); it keeps its slot and is not started again")

    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        self.log(f"scheduler pid {os.getpid()}: {len(self.jobs)} jobs, {self.slots} slots, target {self.target} steps; models {self.models}")
        for j in self.jobs:                                                                  # finished before a crash / reboot but never checked
            if (run_dir(self.models, j) / FINAL_FILE[j.algo]).exists() and not (self.logs / f"{j.name}.result.json").exists():
                self.finish(j, "found finished")
        while True:
            more = self.tick()
            if self.stop:
                self.log("SIGTERM: scheduler exits; running trainers keep going (resumable; `launch` adopts them)")
                return 143
            if not more:
                break
            time.sleep(self.poll)
        running = {}
        states = {j.name: job_state(self.models, self.logs, j, running, self.failed) for j in self.jobs}
        bad = {n: s for n, s in states.items() if s != "finished"}
        self.log(f"queue empty: {len(states) - len(bad)} finished and checked, {len(bad)} not: {bad}")
        return 0 if not bad else 1


# ------------------------------------------------------------------------------------------------ status
def fmt_h(s) -> str:
    return "-" if s is None or s != s else f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"


def status_rows(models: Path, logs: Path, jobs: list, target: int, ps: str) -> list:
    st = load_state(logs)
    failed = set(st.get("failed", []))
    running = running_jobs(ps, {j.name for j in jobs})
    rows, now = [], time.time()
    for j in jobs:
        state = job_state(models, logs, j, running, failed)
        t = progress_t(models, j)
        seg = st.get("segments", {}).get(j.name)
        rate = eta = None
        if state == "running" and seg and now > seg["start_time"] and t > seg["start_t"]:
            rate = (t - seg["start_t"]) / (now - seg["start_time"])
            eta = (target - t) / rate if rate > 0 else None
        lv = last_val(models, j)
        res = None
        try:
            res = json.loads((Path(logs) / f"{j.name}.result.json").read_text())
        except Exception:
            pass
        ident = {}
        try:
            ident = json.loads((run_dir(models, j) / "model_identity.json").read_text())
        except Exception:
            pass
        rows.append({"job": j.name, "algo": j.algo, "state": state, "t": t, "target": target, "pct": 100.0 * t / target, "rate": rate, "eta_s": eta,
                     "val_t": lv[0] if lv else None, "val": lv[1] if lv else None, "resumes": len(ident.get("resumes", [])),
                     "attempts": st.get("attempts", {}).get(j.name, 0), "check": (res or {}).get("checks", {}).get("ok")})
    return rows


def render_status(rows: list, slots: int, lock_pid) -> str:
    out = [f"{'job':<24}{'state':<20}{'step':>10}{'%':>7}{'val@t':>8}{'val':>8}{'steps/s':>9}{'ETA':>9}{'resumes':>8}{'check':>7}"]
    for r in rows:
        out.append(f"{r['job']:<24}{r['state']:<20}{r['t']:>10}{r['pct']:>6.1f}%{(str(r['val_t'] // 1000) + 'k') if r['val_t'] is not None else '-':>8}"
                   f"{('%.4f' % r['val']) if r['val'] is not None else '-':>8}{('%.0f' % r['rate']) if r['rate'] else '-':>9}{fmt_h(r['eta_s']):>9}{r['resumes']:>8}"
                   f"{('ok' if r['check'] else 'FAIL') if r['check'] is not None else '-':>7}")
    cnt: dict = {}
    for r in rows:
        cnt[r["state"]] = cnt.get(r["state"], 0) + 1
    tot, done = sum(r["target"] for r in rows), sum(r["t"] for r in rows)
    rates = {a: [r["rate"] for r in rows if r["algo"] == a and r["rate"]] for a in ("ppo", "ddqn")}
    prior = {"ppo": RATES["ppo_p6"], "ddqn": RATES["ddqn_p6"]}
    rem_s = sum((r["target"] - r["t"]) / ((sum(rates[r["algo"]]) / len(rates[r["algo"]])) if rates[r["algo"]] else prior[r["algo"]]) for r in rows) / slots
    n_run = cnt.get("running", 0)
    out.append(f"scheduler: {'running, pid ' + str(lock_pid) if lock_pid else 'NOT running (launch again to resume)'};  slots in use {n_run}/{slots};  "
               f"states: {', '.join(f'{k} {v}' for k, v in sorted(cnt.items()))}")
    out.append(f"overall: {100.0 * done / tot:.1f}% of {tot:,} steps;  rough ETA for the whole queue {fmt_h(rem_s)} "
               f"({'observed rates' if any(rates.values()) else 'prior rates'} / {slots} slots)")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ CLI
def parse_overrides(a) -> Overrides:
    hp = []
    for kv in a.hp or []:
        k, v = kv.split("=", 1)
        hp.append((k, float(v) if "." in v or "e" in v.lower() else int(v)))
    return Overrides(a.steps, a.val_freq, a.checkpoint_freq, a.speed_freq, tuple(hp))


def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["plan", "launch", "run", "status", "stop"])
    p.add_argument("--smoke", action="store_true", help="dry run: tiny scale, needs --models-dir and --logs-dir outside the real ones, run names get a smoke_ prefix")
    p.add_argument("--models-dir", default=str(B.REAL_MODELS))
    p.add_argument("--logs-dir", default=str(REAL_LOGS))
    p.add_argument("--steps", type=int)
    p.add_argument("--val-freq", type=int)
    p.add_argument("--checkpoint-freq", type=int)
    p.add_argument("--speed-freq", type=int)
    p.add_argument("--hp", action="append", metavar="KEY=VALUE", help="smoke only: override a hyperparameter of the algorithm (e.g. n_steps=50, learning_starts=500)")
    p.add_argument("--seeds", default=None, help="smoke only: e.g. 0,1")
    p.add_argument("--poll", type=float, default=15.0, help="scheduler poll interval, seconds")
    p.add_argument("--slots", type=int, default=None, help="smoke only")
    p.add_argument("--wait", action="store_true", help="launch: stay in the foreground until the queue is empty")
    p.add_argument("--commands", action="store_true", help="plan: print the 30 commands")
    p.add_argument("--watch", action="store_true", help="status: refresh every 30 s")
    p.add_argument("--accept-code-change", action="store_true", help="continue a grid although rl/ or breachlens/ changed since its first launch (recorded)")
    return p


def setup(a):
    cfgs = {c: load_config(c) for c in CONFIG_FILES}
    ov = parse_overrides(a)
    models, logs = Path(a.models_dir).resolve(), Path(a.logs_dir).resolve()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
    if (seeds or a.slots) and not a.smoke:
        raise SystemExit("REFUSED: --seeds / --slots exist only with --smoke")
    jobs = job_list(cfgs, seeds, prefix="smoke_" if a.smoke else "")
    slots = a.slots or load_queue()["slots"]
    return cfgs, ov, models, logs, jobs, slots


def cmd_plan(a) -> int:
    cfgs, ov, models, logs, jobs, slots = setup(a)
    T = target_steps(cfgs, ov)
    n_by = {"ppo": sum(j.algo == "ppo" for j in jobs), "ddqn": sum(j.algo == "ddqn" for j in jobs)}
    pk = peak_bytes(n_by, slots, dict(ov.hp).get("buffer_size"))
    free = shutil.disk_usage(models if Path(models).exists() else ROOT).free
    te = time_estimate(n_by["ppo"], n_by["ddqn"], T, slots)
    print(f"grid: {len(jobs)} jobs ({n_by['ppo']} PPO, {n_by['ddqn']} DDQN), T = {T:,}, {slots} slots; models {models}; logs {logs}")
    print("config sha256: " + ", ".join(f"{k} {v}" for k, v in config_shas().items()))
    if a.commands:
        for j in jobs:
            print(f"BREACHLENS_MODELS_DIR={models} " + " ".join(build_argv(j, cfgs, ov)) + f"  >> {logs}/{j.name}.log")
    print(f"expected peak disk (upper bound, checkpoints kept = {CKPT_KEEP} + one in flight): PPO {pk['ppo'] / 1e9:.2f} GB + DDQN {pk['ddqn'] / 1e9:.2f} GB = {pk['total'] / 1e9:.2f} GB; "
          f"required free (peak + {DISK_HEADROOM / 1e9:.0f} GB headroom) {(pk['total'] + DISK_HEADROOM) / 1e9:.2f} GB; free now {free / 1e9:.2f} GB")
    print(f"wall-time ESTIMATE: PPO {te['ppo_h']:.1f} h + DDQN {te['ddqn_h']:.1f} h = {te['total_h']:.1f} h (range {te['total_low_h']:.1f}-{te['total_high_h']:.1f} h)")
    pf = preflight(models, logs, jobs, ov, a.smoke, a.accept_code_change)
    print("PREFLIGHT: " + ("OK, `launch` would start" if not pf else "REFUSED\n  " + "\n  ".join(pf)))
    return 0 if not pf else 3


def scheduler_cmd(a) -> list:
    c = [sys.executable, str(Path(__file__).resolve()), "run", "--models-dir", a.models_dir, "--logs-dir", a.logs_dir, "--poll", str(a.poll)]
    for flag, v in (("--steps", a.steps), ("--val-freq", a.val_freq), ("--checkpoint-freq", a.checkpoint_freq), ("--speed-freq", a.speed_freq), ("--seeds", a.seeds),
                    ("--slots", a.slots)):
        if v is not None:
            c += [flag, str(v)]
    for kv in a.hp or []:
        c += ["--hp", kv]
    if a.smoke:
        c.append("--smoke")
    if a.accept_code_change:
        c.append("--accept-code-change")
    return c


def cmd_launch(a) -> int:
    cfgs, ov, models, logs, jobs, slots = setup(a)
    pid = SchedulerLock(logs).held_by_other()
    if pid:
        print(f"a scheduler is already running (pid {pid}); nothing started. Use `status`.")
        return 0
    pf = preflight(models, logs, jobs, ov, a.smoke, a.accept_code_change)
    if pf:
        print("REFUSED:\n  " + "\n  ".join(pf), file=sys.stderr)
        return 3
    logs.mkdir(parents=True, exist_ok=True)
    cmd = ["caffeinate", "-dimsu", *scheduler_cmd(a)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(logs / "scheduler.log", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    if pr.poll() is not None:
        print(f"the scheduler exited at once with code {pr.returncode}; see {logs / 'scheduler.log'}", file=sys.stderr)
        return 1
    print(f"queue started (wrapper pid {pr.pid}); log {logs / 'scheduler.log'}; `status` shows progress; run `launch` again after a reboot to resume.")
    if a.wait:
        return pr.wait()
    return 0


def cmd_run(a) -> int:
    cfgs, ov, models, logs, jobs, slots = setup(a)
    lock = SchedulerLock(logs)
    if not lock.acquire():
        print("another scheduler holds the lock; exiting", file=sys.stderr)
        return 4
    pf = preflight(models, logs, jobs, ov, a.smoke, a.accept_code_change)
    if pf:
        print("REFUSED:\n  " + "\n  ".join(pf), file=sys.stderr)
        return 3
    st = load_state(logs)
    now = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": head_commit(), "pid": os.getpid(), "config_sha256": config_shas(), "accept_code_change": a.accept_code_change}
    st.setdefault("first_launch", now)
    st.setdefault("launches", []).append(now)
    st["target_steps"], st["smoke"], st["n_jobs"] = target_steps(cfgs, ov), a.smoke, len(jobs)
    st["failed"], st["attempts"] = [], {}                                            # a fresh launch gives every job a fresh restart budget
    save_state(logs, st)
    return Scheduler(models, logs, cfgs, jobs, ov, slots=slots, poll=a.poll).run()


def cmd_status(a) -> int:
    cfgs, ov, models, logs, jobs, slots = setup(a)
    while True:
        st = load_state(logs)
        rows = status_rows(models, logs, jobs, st.get("target_steps") or target_steps(cfgs, ov), B.ps_text())
        print(render_status(rows, slots, SchedulerLock(logs).held_by_other()), flush=True)
        if not a.watch:
            return 0
        print()
        time.sleep(30)


def cmd_stop(a) -> int:
    cfgs, ov, models, logs, jobs, slots = setup(a)
    pid = SchedulerLock(logs).held_by_other()
    if pid and pid > 0:
        os.kill(pid, signal.SIGTERM)
        print(f"SIGTERM to the scheduler (pid {pid})")
    run = running_jobs(B.ps_text(), {j.name for j in jobs})
    for n, p in run.items():
        os.kill(p, signal.SIGTERM)
    print(f"SIGTERM to {len(run)} trainer(s); all are resumable from their latest checkpoint (`launch` continues)")
    return 0


def main(argv=None) -> int:
    a = make_parser().parse_args(argv)
    return {"plan": cmd_plan, "launch": cmd_launch, "run": cmd_run, "status": cmd_status, "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
