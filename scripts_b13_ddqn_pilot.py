"""B13e DDQN train_freq pilot launcher (rows (p) + (cc); prepared, NOT started by B13e). Non-gating. 4 shaped runs as separate 1-thread processes under caffeinate:
train_every (row p's `train_freq`) 4 and 8 x seeds 9301 / 9302, 300k steps, TUNE-train -> TUNE-val every 25k, every other hyperparameter = rl/ddqn.py DDQN_CONFIG.

    .venv/bin/python scripts_b13_ddqn_pilot.py                 # the real pilot, frozen config only (data/v2/ddqn_pilot_config.json)
    .venv/bin/python scripts_b13_ddqn_pilot.py --print-only     # print the 4 commands, start nothing
    .venv/bin/python scripts_b13_ddqn_pilot.py --smoke --models-dir /tmp/x --logs-dir /tmp/y --steps 4000 --val-freq 1000 --checkpoint-freq 2000 \\
        --learning-starts 500 --wait                            # dry run: tiny, outside models_v2, never a pilot result

Refuses when: rows (p) / (cc) are missing from the SPEC; the sha256 of the config file differs from the one recorded in row (cc) (or the file differs from
rl/ddqn.py); the tracked git tree is dirty; any pilot / rl.train_ddqn_v2 / rl.train_ppo_v2 process is running; the FINAL ledger has an eval entry;
BREACHLENS_STATS_PATH / BREACHLENS_GEN_PARAMS is set. The real mode takes no overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import scripts_b13_pilot as B

ROOT = B.ROOT
CONFIG = ROOT / "data" / "v2" / "ddqn_pilot_config.json"
ROWS = ("(p) **B12 item 0", "(cc) **B13d")


def frozen() -> dict:
    return json.loads(CONFIG.read_text())["config"]


def config_matches_code(cfg: dict) -> list:
    from rl import ddqn as D
    want = {k: v for k, v in json.loads(json.dumps(D.DDQN_CONFIG)).items() if k != "train_every"}
    out = []
    if cfg["ddqn"] != want:
        out.append("config 'ddqn' block differs from rl.ddqn.DDQN_CONFIG (minus train_every)")
    if cfg["arms"]["f4"]["train_every"] != D.DDQN_CONFIG["train_every"] or cfg["arms"]["f8"]["train_every"] != 2 * D.DDQN_CONFIG["train_every"]:
        out.append("arms are not train_every 4 / 8")
    return out


def missing_rows(spec_text: str, cfg_sha: str) -> list:
    miss = [r for r in ROWS if r not in spec_text]
    if cfg_sha not in spec_text:
        miss.append(f"ddqn-pilot config sha256 {cfg_sha[:12]}... not recorded in row (cc)")
    return miss


def running_copies(ps: str, own_pid: int) -> list:
    out = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 2 or not tok[0].isdigit() or int(tok[0]) == own_pid or os.path.basename(tok[1]) in ("zsh", "bash", "sh", "grep", "pgrep"):
            continue
        if "rl.train_ddqn_v2" in tok or "rl.train_ppo_v2" in tok or any(t.endswith(("scripts_b13_ddqn_pilot.py", "scripts_b13_pilot.py")) for t in tok):
            out.append(ln.strip())
    return out


def preflight(own_pid: int | None = None) -> list:
    own_pid = os.getpid() if own_pid is None else own_pid
    problems = []
    d = B.dirty_tracked()
    if d:
        problems.append(f"git tree has {len(d)} modified tracked file(s): {d[:3]}")
    cfg = frozen()
    miss = missing_rows(B.SPEC.read_text(), B.config_sha(cfg))
    if miss:
        problems.append(f"rows incomplete: {miss}")
    problems += config_matches_code(cfg)
    if cfg.get("shaping", "").split()[0] != "on":
        problems.append("the config does not switch shaping on")
    run = running_copies(B.ps_text(), own_pid)
    if run:
        problems.append(f"a trainer / pilot process is already running: {run[:2]}")
    n = B.eval_entries()
    if n:
        problems.append(f"the FINAL ledger has {n} eval entr{'y' if n == 1 else 'ies'}")
    bad = [k for k in B.FORBIDDEN_ENV if k in os.environ]
    if bad:
        problems.append(f"environment override set: {bad}")
    return problems


def build_commands(cfg: dict, *, steps: int, val_freq: int, checkpoint_freq: int, learning_starts: int, seeds: list) -> list:
    out = []
    for arm in ("f4", "f8"):
        for s in seeds:
            name = cfg["run_names"].format(freq=arm[1:], seed=s)
            hp = dict(cfg["ddqn"], train_every=cfg["arms"][arm]["train_every"], learning_starts=learning_starts)
            a = ["caffeinate", "-dimsu", sys.executable, "-m", "rl.train_ddqn_v2", "--train-set", cfg["train_set"], "--seed", str(s), "--run-name", name,
                 "--timesteps", str(steps), "--val-freq", str(val_freq), "--checkpoint-freq", str(checkpoint_freq), "--curriculum", cfg["curriculum"]["mode"],
                 "--phase-a-steps", str(steps // 3), "--shaping", "on", "--threads", str(cfg["torch_threads"]), "--resume"]
            for k, v in hp.items():
                if k != "net_arch":
                    a += ["--" + k.replace("_", "-"), str(v)]
            out.append((name, a))
    return out


def main(argv=None) -> int:
    cfg = frozen()
    p = argparse.ArgumentParser()
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--models-dir", default=str(B.REAL_MODELS))
    p.add_argument("--logs-dir", default=str(ROOT / "logs_b13"))
    p.add_argument("--steps", type=int, default=cfg["timesteps"])
    p.add_argument("--val-freq", type=int, default=cfg["val_freq"])
    p.add_argument("--checkpoint-freq", type=int, default=cfg["checkpoint_freq"])
    p.add_argument("--learning-starts", type=int, default=cfg["ddqn"]["learning_starts"])
    p.add_argument("--seeds", default=",".join(map(str, cfg["seeds"])))
    a = p.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x]
    models = Path(a.models_dir).resolve()
    default_args = (a.steps == cfg["timesteps"] and a.val_freq == cfg["val_freq"] and a.checkpoint_freq == cfg["checkpoint_freq"]
                    and a.learning_starts == cfg["ddqn"]["learning_starts"] and seeds == cfg["seeds"] and models == B.REAL_MODELS.resolve())
    if not a.smoke and not default_args:
        print("REFUSED: the real DDQN pilot takes no overrides (frozen config); use --smoke with a models dir outside models_v2 for a dry run", file=sys.stderr)
        return 3
    if a.smoke and models == B.REAL_MODELS.resolve():
        print("REFUSED: --smoke must not write into models_v2", file=sys.stderr)
        return 3
    problems = preflight()
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3
    cmds = build_commands(cfg, steps=a.steps, val_freq=a.val_freq, checkpoint_freq=a.checkpoint_freq, learning_starts=a.learning_starts, seeds=seeds)
    if a.print_only:
        for name, c in cmds:
            print(f"BREACHLENS_MODELS_DIR={models} " + " ".join(c) + f"  > {a.logs_dir}/{name}.log 2>&1")
        return 0
    logs = Path(a.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BREACHLENS_MODELS_DIR=str(models), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    procs, pids = [], {}
    for name, c in cmds:
        pr = subprocess.Popen(c, cwd=ROOT, env=env, stdout=open(logs / f"{name}.log", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, pr))
        pids[name] = pr.pid
    (logs / "ddqn_pids.json").write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "launcher_pid": os.getpid(), "models_dir": str(models), "pids": pids}, indent=1))
    print(f"started {len(procs)} runs; logs in {logs}; models in {models}", flush=True)
    if a.wait:
        codes = {n: pr.wait() for n, pr in procs}
        print("exit codes:", codes)
        return 0 if all(c == 0 for c in codes.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
