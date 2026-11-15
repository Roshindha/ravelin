"""B13 G7 pilot launcher (prepared in B13a; NOT started by B13a). Starts 3 plain + 3 D4 PPO runs as separate 1-thread processes under caffeinate.

    .venv/bin/python scripts_b13_pilot.py                    # B13b: the real pilot, frozen config only (data/v2/g7_pilot_config.json)
    .venv/bin/python scripts_b13_pilot.py --extend           # only if scripts_g7_report.py says EXTEND (changelog (u))
    .venv/bin/python scripts_b13_pilot.py --print-only       # print the 6 commands, start nothing
    .venv/bin/python scripts_b13_pilot.py --shaped           # B13c: the SPEC 7.4 shaped-reward pilot (3 plain seeds 9201-9203, frozen shaped config, NO extension)
    .venv/bin/python scripts_b13_pilot.py --smoke --models-dir /tmp/x --logs-dir /tmp/y --steps 5000 --val-freq 2500 --checkpoint-freq 2500 --speed-freq 2500 \\
        --plain-seeds 9001,9002 --d4-seeds '' --wait         # dry run: tiny, outside the repo, never a G7 result

--shaped additionally refuses unless Gate changelog rows (y) and (z) of B13c are in the SPEC and the sha256 of data/v2/g7_shaped_pilot_config.json matches the one recorded in row (z).
Refuses to start when: the tracked git tree is dirty; a changelog row (r)-(w) or the frozen-config hash is missing / wrong; any pilot or rl.train_ppo_v2
process is already running; the FINAL ledger has an eval entry; BREACHLENS_STATS_PATH or BREACHLENS_GEN_PARAMS is set. The real mode takes no overrides:
steps, seeds and frequencies come from the frozen config; --smoke allows overrides but only with a models dir outside models_v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "docs" / "SPEC_V2.md"
FROZEN = ROOT / "data" / "v2" / "g7_pilot_config.json"
FROZEN_SHAPED = ROOT / "data" / "v2" / "g7_shaped_pilot_config.json"
LEDGER = ROOT / "data" / "v2" / "final_access.log"
REAL_MODELS = ROOT / "models_v2"
ROWS = ("(r) **B13a", "(s) **B13a", "(t) **B13a", "(u) **B13a", "(v) **B13a", "(w) **B13a")
ROWS_SHAPED = ROWS + ("(y) **B13c", "(z) **B13c")
FORBIDDEN_ENV = ("BREACHLENS_STATS_PATH", "BREACHLENS_GEN_PARAMS")


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen(shaped: bool = False) -> dict:
    return json.loads((FROZEN_SHAPED if shaped else FROZEN).read_text())["config"]


# ------------------------------------------------------------------------------------------------ preflight (each check is a function so tests can drive it)
def dirty_tracked() -> list:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def missing_item0(spec_text: str, cfg_sha: str, shaped: bool = False) -> list:
    miss = [r for r in (ROWS_SHAPED if shaped else ROWS) if r not in spec_text]
    if cfg_sha not in spec_text:
        miss.append(f"{'shaped-' if shaped else ''}frozen-config sha256 {cfg_sha[:12]}... not recorded in the changelog")
    return miss


def running_copies(ps_text: str, own_pid: int) -> list:
    """Process lines (pid command) of a running trainer or launcher. Shell wrappers (zsh -c '...text...') are not processes of ours and are skipped."""
    out = []
    for ln in ps_text.splitlines():
        tok = ln.split()
        if len(tok) < 2 or not tok[0].isdigit() or int(tok[0]) == own_pid or os.path.basename(tok[1]) in ("zsh", "bash", "sh", "grep", "pgrep"):
            continue
        if "rl.train_ppo_v2" in tok or any(t.endswith("scripts_b13_pilot.py") for t in tok):
            out.append(ln.strip())
    return out


def ps_text() -> str:
    return subprocess.run(["ps", "-eo", "pid,command"], capture_output=True, text=True, check=True).stdout


def eval_entries(ledger: Path | None = None) -> int:
    ledger = LEDGER if ledger is None else ledger
    if not ledger.exists():
        return 0
    return sum(1 for ln in ledger.read_text().splitlines() if ln.strip() and json.loads(ln).get("purpose") == "eval")


def preflight(own_pid: int | None = None, shaped: bool = False) -> list:
    own_pid = os.getpid() if own_pid is None else own_pid
    problems = []
    d = dirty_tracked()
    if d:
        problems.append(f"git tree has {len(d)} modified tracked file(s): {d[:3]}")
    cfg = frozen(shaped)
    miss = missing_item0(SPEC.read_text(), config_sha(cfg), shaped)
    if miss:
        problems.append(f"item 0 incomplete: {miss}")
    if shaped and not str(cfg.get("shaping", "")).startswith("on"):
        problems.append("the shaped config does not switch shaping on")
    if not shaped and cfg.get("shaping") != "off":
        problems.append("the plain G7 config must have shaping off")
    run = running_copies(ps_text(), own_pid)
    if run:
        problems.append(f"a pilot / training process is already running: {run[:2]}")
    n = eval_entries()
    if n:
        problems.append(f"the FINAL ledger has {n} eval entr{'y' if n == 1 else 'ies'}")
    bad = [k for k in FORBIDDEN_ENV if k in os.environ]
    if bad:
        problems.append(f"environment override set: {bad}")
    return problems


# ------------------------------------------------------------------------------------------------ commands
def build_commands(cfg: dict, *, steps: int, val_freq: int, checkpoint_freq: int, speed_freq: int, plain_seeds: list, d4_seeds: list, extend_to: int | None = None) -> list:
    """[(run_name, argv)] for the 6 runs. Every PPO hyperparameter is passed explicitly from the frozen config."""
    hp = cfg["ppo"]
    out = []
    for aug, seeds, tmpl in (("none", plain_seeds, cfg["run_names"]["plain"]), ("d4", d4_seeds, cfg["run_names"].get("d4"))):
        for s in seeds:
            name = tmpl.format(seed=s)
            a = ["caffeinate", "-dimsu", sys.executable, "-m", "rl.train_ppo_v2", "--train-set", cfg["train_set"], "--aug", aug, "--seed", str(s),
                 "--run-name", name, "--val-freq", str(val_freq), "--checkpoint-freq", str(checkpoint_freq), "--speed-freq", str(speed_freq),
                 "--threads", str(cfg["torch_threads"]), "--resume"]
            for k, v in hp.items():
                if k != "net_arch":
                    a += ["--" + k.replace("_", "-"), str(v)]
            if str(cfg.get("shaping", "off")).startswith("on"):
                a += ["--shaping", "on"]
            if extend_to is None:
                a += ["--timesteps", str(steps), "--curriculum", cfg["curriculum"]["mode"], "--phase-a-steps", str(steps // 3)]
            else:
                a += ["--extend-to", str(extend_to)]
            out.append((name, a))
    return out


def main(argv=None) -> int:
    shaped = "--shaped" in (sys.argv[1:] if argv is None else argv)
    cfg = frozen(shaped)
    p = argparse.ArgumentParser()
    p.add_argument("--shaped", action="store_true", help="B13c: the SPEC 7.4 shaped-reward pilot (Gate changelog (z)); no extension exists for it")
    p.add_argument("--extend", action="store_true", help=f"continue the finished runs to {cfg.get('extension_timesteps')} (only after the G7 report says EXTEND; never for --shaped)")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--wait", action="store_true", help="stay in the foreground until all runs end")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--models-dir", default=str(REAL_MODELS))
    p.add_argument("--logs-dir", default=str(ROOT / "logs_b13"))
    p.add_argument("--steps", type=int, default=cfg["timesteps"])
    p.add_argument("--val-freq", type=int, default=cfg["val_freq"])
    p.add_argument("--checkpoint-freq", type=int, default=cfg["checkpoint_freq"])
    p.add_argument("--speed-freq", type=int, default=10_000)
    p.add_argument("--plain-seeds", default=",".join(map(str, cfg["plain_seeds"])))
    p.add_argument("--d4-seeds", default=",".join(map(str, cfg["d4_seeds"])))
    a = p.parse_args(argv)
    if a.shaped and a.extend:
        print("REFUSED: the shaped pilot gets no extension (Gate changelog (z), SPEC 3.4 G7)", file=sys.stderr)
        return 3
    plain = [int(x) for x in a.plain_seeds.split(",") if x]
    d4 = [int(x) for x in a.d4_seeds.split(",") if x]
    models = Path(a.models_dir).resolve()
    default_args = (a.steps == cfg["timesteps"] and a.val_freq == cfg["val_freq"] and a.checkpoint_freq == cfg["checkpoint_freq"] and plain == cfg["plain_seeds"]
                    and d4 == cfg["d4_seeds"] and models == REAL_MODELS.resolve())
    if not a.smoke and not default_args:
        print("REFUSED: the real pilot takes no overrides (frozen config); use --smoke with a models dir outside models_v2 for a dry run", file=sys.stderr)
        return 3
    if a.smoke and models == REAL_MODELS.resolve():
        print("REFUSED: --smoke must not write into models_v2", file=sys.stderr)
        return 3
    if frozen(shaped)["ppo"] != _ppo_config() or frozen(False)["ppo"] != _ppo_config():
        print("REFUSED: a frozen pilot config differs from rl.train_ppo_v2.PPO_CONFIG", file=sys.stderr)
        return 3
    problems = preflight(shaped=shaped)
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3
    cmds = build_commands(cfg, steps=a.steps, val_freq=a.val_freq, checkpoint_freq=a.checkpoint_freq, speed_freq=a.speed_freq, plain_seeds=plain, d4_seeds=d4,
                          extend_to=cfg["extension_timesteps"] if a.extend else None)
    if a.print_only:
        for name, c in cmds:
            print(f"BREACHLENS_MODELS_DIR={models} " + " ".join(c) + f"  > {a.logs_dir}/{name}.log 2>&1")
        return 0
    logs = Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BREACHLENS_MODELS_DIR=str(models), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    procs, pids = [], {}
    for name, c in cmds:
        f = open(logs / f"{name}.log", "ab")
        pr = subprocess.Popen(c, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, pr))
        pids[name] = pr.pid
    (logs / "pids.json").write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "launcher_pid": os.getpid(), "models_dir": str(models), "pids": pids}, indent=1))
    print(f"started {len(procs)} runs; logs in {logs}; models in {models}", flush=True)
    if a.wait:
        codes = {n: pr.wait() for n, pr in procs}
        print("exit codes:", codes)
        return 0 if all(c == 0 for c in codes.values()) else 1
    return 0


def _ppo_config() -> dict:
    from rl.train_ppo_v2 import PPO_CONFIG
    return json.loads(json.dumps(PPO_CONFIG))


if __name__ == "__main__":
    sys.exit(main())
