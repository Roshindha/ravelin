"""B17: human-vs-agents duel core (docs/SPEC_V2.md, Gate changelog row (kk)). Headless engine + text CLI. A DEMO, not a result.

The human plays one DEV-validation layout through the SAME ``BreachEnv`` the agents play (same observation grid, same action masks, same
budget); nothing the agents do not see is shown. Opponents (three seed-0 learners with TTA, BestSingle, Oracle) are precomputed into the committed
``data/v2/duel_cache.json`` by ``scripts/build_duel_cache.py`` and stay hidden until the human's episode has ended (battle resolved).
Human runs go only to the gitignored local file ``results_v2/duel_human_runs.jsonl`` and are never reported as results.

    .venv/bin/python -m breachlens.duel                       # first DEV-validation layout
    .venv/bin/python -m breachlens.duel --layout spine_pods-300000056
    .venv/bin/python -m breachlens.duel --list

Commands: ``buy <unit>``, ``deploy <unit> <tile>``, ``end`` (END_BUY), ``wait`` (start the battle), ``show``, ``undo`` (BUY phase only), ``help``, ``quit``.
Unit names: scout, sniper, bulwark, blaster, mender. Tiles: border indices 0..51 (clockwise from the top-left corner), printed by ``show``.
Importing this module never imports torch / stable-baselines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

from . import pool as P
from .actions import BORDER, decode_action, encode_buy, encode_deploy
from .armies import unit_cost
from .constants import BUDGET, END_BUY, N_BORDER, N_UNIT_TYPES, UNIT_TYPES, WAIT
from .env import BreachEnv, IllegalActionError
from .score import result_score

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "v2" / "duel_cache.json"
HUMAN_RUNS_PATH = ROOT / "results_v2" / "duel_human_runs.jsonl"
DUEL_SPLIT = "DEV-validation"
CACHE_VERSION = 1
CONTESTANTS = ("ppo_plain_tta", "ppo_d4_tta", "ddqn_tta", "best_single", "oracle")
LABELS = {"human": "You (human)", "ppo_plain_tta": "PPO plain + TTA (seed 0)", "ppo_d4_tta": "PPO+D4 + TTA (seed 0)", "ddqn_tta": "DDQN + TTA (seed 0)",
          "best_single": "BestSingle (Sniper army, scripted)", "oracle": "Oracle (search baseline, not the optimum)"}
DEMO_NOTE = ("This is a demo, not a result: one layout, one human run, opponents precomputed on DEV-validation layouts, which were held out from training and only monitored during it (never used to select models). "
             "Oracle is a fixed-budget search baseline (about 400 simulations), NOT the optimum.")


class DuelError(ValueError):
    """An illegal or malformed human action; the message says what was wrong and what is legal."""


class DuelNotFinished(RuntimeError):
    """Opponent outcomes are withheld until the human's episode has ended."""


def reset_seed(layout_id: str) -> int:
    return zlib.crc32(layout_id.encode())              # the B15 reset seed, shared by the human and the learners


# ---------------------------------------------------------------------------------------------------------------------------- cache
def _check_duel_id(layout_id: str) -> None:
    P.assert_dev_only([layout_id], "validation")       # FINAL / transfer / LAB-DESIGN ids raise PermissionError
    from . import evaluator as EV
    if layout_id not in set(EV.dev_layout_ids(DUEL_SPLIT, entry_point="breachlens.duel")):
        raise PermissionError(f"layout {layout_id!r} is not a {DUEL_SPLIT} layout; the duel uses {DUEL_SPLIT} only")


def load_cache(path: Path = CACHE_PATH) -> dict:
    c = json.loads(Path(path).read_text())
    if c.get("version") != CACHE_VERSION or c.get("meta", {}).get("split") != DUEL_SPLIT:
        raise ValueError(f"{path}: not a duel cache of version {CACHE_VERSION} for {DUEL_SPLIT}")
    return c


def cache_lookup(cache: dict, layout_id: str) -> dict:
    """{contestant: row} for one layout. A non-DEV-validation id is refused (PermissionError) even if someone put it in the file."""
    _check_duel_id(layout_id)
    if layout_id not in cache["layouts"]:
        raise PermissionError(f"layout {layout_id!r} has no entry in the duel cache")
    return cache["layouts"][layout_id]["contestants"]


def outcome_row(contestant: str, result, army: list, owned: list, bought: int) -> dict:
    """One comparison row. `army` = [(unit type, border tile)] in deployment order; `owned` = bought but not deployed."""
    counts = np.bincount([t for t, _ in army], minlength=N_UNIT_TYPES) + np.asarray(owned, dtype=int)     # units bought per type (as B15)
    return {"contestant": contestant, "score": float(result_score(result)), "d": float(result.d_num / max(result.d_den, 1)),
            "breach": bool(result.core_destroyed), "n_bought": int(bought), "n_deployed": len(army),
            "bought": {u: int(counts[i]) for i, u in enumerate(UNIT_TYPES)}, "deploy": [[UNIT_TYPES[int(t)], int(b)] for t, b in army]}


# ---------------------------------------------------------------------------------------------------------------------------- engine
def parse_unit(name) -> int:
    if isinstance(name, (int, np.integer)) and 0 <= int(name) < N_UNIT_TYPES:
        return int(name)
    s = str(name).strip().lower()
    if s in UNIT_TYPES:
        return UNIT_TYPES.index(s)
    raise DuelError(f"unknown unit {name!r}; choose one of: {', '.join(UNIT_TYPES)}")


class Duel:
    """One human episode on one DEV-validation layout. Opponents are not loaded and not reachable before the episode has ended."""

    def __init__(self, layout_id: str | None = None, index: int | None = 0, *, cache_path: Path = CACHE_PATH, human_runs_path: Path | None = HUMAN_RUNS_PATH):
        from . import evaluator as EV
        self.rec, self.layout = EV.load_dev_layout(DUEL_SPLIT, None if layout_id else index, layout_id, entry_point="breachlens.duel")
        self.layout_id, self.family = self.rec["id"], self.rec["family"]
        self._cache_path, self._runs_path = Path(cache_path), (Path(human_runs_path) if human_runs_path else None)
        self.env = BreachEnv(layouts=[self.layout], role="train")        # the same env, observation and masks the agents get
        self.actions: list[int] = []
        self._reset()

    def _reset(self) -> None:
        self.obs, _ = self.env.reset(seed=reset_seed(self.layout_id), options={"layout_index": 0})

    # ----- state
    @property
    def finished(self) -> bool:
        return bool(self.env.done)

    @property
    def phase(self) -> str:
        return ("BUY", "DEPLOY", "BATTLE")[self.env.phase]

    def legal_actions(self) -> list[int]:
        return [int(a) for a in np.flatnonzero(self.env.action_masks())] if not self.finished else []

    def describe_legal(self) -> list[str]:
        env, out = self.env, []
        if self.finished:
            return out
        m = env.action_masks()
        if env.phase == 0:
            buys = [f"{UNIT_TYPES[u]} (cost {unit_cost(u)})" for u in range(N_UNIT_TYPES) if m[encode_buy(u)]]
            out.append("buy: " + (", ".join(buys) if buys else "nothing affordable (or unit cap reached)"))
            out.append("end: stop buying and start deploying" if m[END_BUY] else "end: not yet legal (buy at least one unit)")
        else:
            for u in range(N_UNIT_TYPES):
                if env.owned[u] > 0:
                    tiles = [b for b in range(N_BORDER) if m[encode_deploy(u, b)]]
                    out.append(f"deploy {UNIT_TYPES[u]} (x{env.owned[u]}) on a free tile: {_ranges(tiles)}")
            out.append("wait: end deployment and start the battle (undeployed units are forfeited)")
        return out

    def render(self) -> str:
        """Text view built from the observation grid the agents get. Cells: C core, 1-3 building (value), D defender, # wall, u deployed unit,
        .. empty interior; free border tiles show their border index."""
        g, env = self.obs["grid"], self.env
        from .constants import GRID
        from .env import BORDER_CELL
        tile_at = {BORDER_CELL[b]: b for b in range(N_BORDER)}
        lines = ["     " + "".join(f"{c:>3d}" for c in range(GRID))]
        for r in range(GRID):
            cells = []
            for c in range(GRID):
                i = r * GRID + c
                if g[5, r, c] > 0:
                    s = "C"
                elif g[1, r, c] > 0:
                    s = str(int(round(float(g[4, r, c]) * 3)) or 1)
                elif g[2, r, c] > 0:
                    s = "D"
                elif g[3, r, c] > 0:
                    s = "#"
                elif g[0, r, c] > 0:
                    s = "u"
                elif i in tile_at:
                    s = f"{tile_at[i]}"
                else:
                    s = "."
                cells.append(f"{s:>3s}")
            lines.append(f"{r:>3d}  " + "".join(cells))
        legend = "legend: C core | 1-3 building (value) | D defender | # wall | u your unit | number = border tile index | . empty"
        owned = ", ".join(f"{UNIT_TYPES[u]} x{n}" for u, n in enumerate(env.owned) if n) or "none"
        deployed = ", ".join(f"{UNIT_TYPES[t]}@{b}" for t, b in env.army) or "none"
        head = [f"layout {self.layout_id} [{self.family}]  ({DUEL_SPLIT})", f"phase {self.phase}  gold {env.gold}/{BUDGET}  bought {env.bought}  "
                f"owned (to deploy): {owned}  deployed: {deployed}"]
        return "\n".join(head + lines + [legend] + ["legal now:"] + ["  " + s for s in self.describe_legal()])

    # ----- actions
    def _explain_illegal(self, a: int) -> str:
        env = self.env
        kind, u, b = decode_action(a)
        if env.done:
            return "the episode is over"
        if kind == "buy":
            if env.phase != 0:
                return f"cannot buy during the {self.phase} phase (buying ended with END_BUY)"
            if env.bought >= _unit_cap():
                return f"unit cap reached ({_unit_cap()} units)"
            return f"cannot afford {UNIT_TYPES[u]}: costs {unit_cost(u)}, you have {env.gold} gold"
        if kind == "end_buy":
            return "cannot end buying in the DEPLOY phase" if env.phase != 0 else "buy at least one unit before ending the buy phase"
        if kind == "wait":
            return "cannot start the battle during the BUY phase; use 'end' first" if env.phase == 0 else "not legal now"
        if env.phase == 0:
            return "cannot deploy during the BUY phase; use 'end' first"
        if env.owned[u] <= 0:
            return f"you have no {UNIT_TYPES[u]} left to deploy"
        if not env._st.border_free[b]:
            return f"border tile {b} {tuple(BORDER[b])} is a wall tile"
        return f"border tile {b} {tuple(BORDER[b])} is already occupied"

    def act(self, action: int) -> dict:
        if self.finished:
            raise DuelError("the episode is over; start a new duel")
        a = int(action)
        if not 0 <= a < 267 or not self.env.action_masks()[a]:
            raise DuelError("illegal action: " + (self._explain_illegal(a) if 0 <= a < 267 else f"index {a} out of range"))
        try:
            self.obs, _, done, _, info = self.env.step(a)
        except IllegalActionError as e:                                                           # cannot happen after the mask check
            raise DuelError(str(e)) from e
        self.actions.append(a)
        if done:
            self._log_human_run()
        return info

    def buy(self, unit) -> dict:
        return self.act(encode_buy(parse_unit(unit)))

    def deploy(self, unit, tile) -> dict:
        try:
            b = int(tile)
        except (TypeError, ValueError):
            raise DuelError(f"tile must be an integer border index 0..{N_BORDER - 1}, got {tile!r}") from None
        if not 0 <= b < N_BORDER:
            raise DuelError(f"border tile {b} is out of range 0..{N_BORDER - 1}")
        return self.act(encode_deploy(parse_unit(unit), b))

    def end_buy(self) -> dict:
        return self.act(END_BUY)

    def wait(self) -> dict:
        return self.act(WAIT)

    def undo(self) -> int:
        """Undo the last BUY (BUY phase only, before END_BUY): reset with the same seed and replay the remaining buys through the same env.
        The env has no undo of its own and is not modified. Returns the undone action."""
        if self.finished or self.env.phase != 0:
            raise DuelError("undo is only possible during the BUY phase, before 'end'")
        if not self.actions:
            raise DuelError("nothing to undo")
        last, rest = self.actions[-1], self.actions[:-1]
        self._reset()
        self.actions = []
        for a in rest:
            self.obs, _, _, _, _ = self.env.step(a)
            self.actions.append(a)
        return last

    # ----- results
    def human_row(self) -> dict:
        if not self.finished:
            raise DuelNotFinished("your episode is not over yet; opponent outcomes stay hidden until the battle has been resolved")
        e = self.env
        return outcome_row("human", e.result, e.army, e.owned, e.bought)

    def reveal(self) -> list[dict]:
        """Comparison rows (you first, then the five contestants) with labels. Raises DuelNotFinished until the human's episode has ended; the
        cache file is not even opened before that."""
        if not self.finished:
            raise DuelNotFinished("opponent outcomes are hidden until your episode has ended (battle resolved)")
        opp = cache_lookup(load_cache(self._cache_path), self.layout_id)
        rows = [self.human_row()] + [dict(opp[k], contestant=k) for k in CONTESTANTS]
        for r in rows:
            r["label"] = LABELS[r["contestant"]]
        return rows

    def _log_human_run(self) -> None:
        if self._runs_path is None:
            return
        self._runs_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"kind": "human demo run, never a result", "layout_id": self.layout_id, "split": DUEL_SPLIT, "actions": self.actions, "row": self.human_row(),
               "commit": _git("rev-parse", "HEAD")}
        with open(self._runs_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def _unit_cap() -> int:
    from .config import CFG as _C
    return _C.unit_cap


def _git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _ranges(xs: list[int]) -> str:
    if not xs:
        return "none"
    out, s, p = [], xs[0], xs[0]
    for x in xs[1:] + [None]:
        if x is not None and x == p + 1:
            p = x
            continue
        out.append(str(s) if s == p else f"{s}-{p}")
        if x is not None:
            s = p = x
    return ", ".join(out)


# ---------------------------------------------------------------------------------------------------------------------------- table
def format_table(rows: list[dict]) -> str:
    w = max(len(r["label"]) for r in rows)
    out = [f"{'contestant':<{w}}  {'score':>6}  {'D':>6}  breach  bought  deployed  army (bought per type)"]
    for r in rows:
        army = " ".join(f"{u[:3]}={n}" for u, n in r["bought"].items() if n)
        out.append(f"{r['label']:<{w}}  {r['score']:6.3f}  {r['d']:6.3f}  {'yes' if r['breach'] else 'no ':<6}  {r['n_bought']:>6}  {r['n_deployed']:>8}  {army}")
    out.append("")
    for r in rows:
        out.append(f"deploy tiles, {r['label']}: " + (", ".join(f"{u}@{b}" for u, b in r["deploy"]) or "none"))
    out += ["", "Oracle = search baseline, not the optimum.", DEMO_NOTE]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------------- CLI
HELP = ("commands: buy <unit> | deploy <unit> <tile> | end | wait | show | undo (BUY phase only) | help | quit\n"
        f"units: {', '.join(UNIT_TYPES)}; tiles: border index 0..{N_BORDER - 1}")


def execute(duel: Duel, line: str) -> str | None:
    """Run one CLI line; returns text to print (None = quit). DuelError is caught by the caller."""
    parts = line.strip().lower().split()
    if not parts:
        return ""
    cmd, args = parts[0], parts[1:]
    if cmd in ("quit", "exit", "q"):
        return None
    if cmd in ("help", "?"):
        return HELP
    if cmd == "show":
        return duel.render()
    if cmd == "buy":
        if len(args) != 1:
            raise DuelError("usage: buy <unit>")
        duel.buy(args[0])
        return f"bought {args[0]}; gold {duel.env.gold}"
    if cmd == "deploy":
        if len(args) != 2:
            raise DuelError("usage: deploy <unit> <tile>")
        duel.deploy(args[0], args[1])
        return f"deployed {args[0]} on tile {args[1]}" + ("" if not duel.finished else " (all units out: battle resolved)")
    if cmd in ("end", "end_buy"):
        duel.end_buy()
        return "buying ended; now deploy your units (or 'wait' to fight with nothing)"
    if cmd == "wait":
        duel.wait()
        return "battle resolved"
    if cmd == "undo":
        a = duel.undo()
        return f"undid: {decode_action(a)[0]} {UNIT_TYPES[decode_action(a)[1]]}"
    raise DuelError(f"unknown command {cmd!r}; type 'help'")


def run_cli(duel: Duel, lines, out=print) -> list[dict] | None:
    out(f"BreachLens duel, a demo. {DEMO_NOTE}\n")
    out(HELP + "\n")
    out(duel.render())
    for line in lines:
        try:
            res = execute(duel, line)
        except DuelError as e:
            out(f"! {e}")
            continue
        if res is None:
            out("bye (no result recorded)")
            return None
        if res:
            out(res)
        if duel.finished:
            break
    if not duel.finished:
        return None
    rows = duel.reveal()
    out("\n=== Comparison (same layout, same reset seed) ===\n" + format_table(rows))
    return rows


def _stdin_lines():
    interactive = sys.stdin.isatty()
    while True:
        try:
            line = input("duel> " if interactive else "")
        except EOFError:
            return
        yield line


def main(argv: list[str] | None = None) -> None:
    from . import evaluator as EV
    p = argparse.ArgumentParser(description="Human-vs-agents duel on a DEV-validation layout (text only; a demo).")
    p.add_argument("--layout", default=None, help="DEV-validation layout id")
    p.add_argument("--index", type=int, default=0, help="position in DEV-validation (default 0); ignored with --layout")
    p.add_argument("--list", action="store_true", help="list the DEV-validation layouts and exit")
    p.add_argument("--script", default=None, help="read commands from this file instead of stdin")
    p.add_argument("--no-log", action="store_true", help="do not write the local human-run file")
    a = p.parse_args(argv)
    if a.list:
        for i, lid in enumerate(EV.dev_layout_ids(DUEL_SPLIT, entry_point="breachlens.duel")):
            print(f"{i:>2d}  {lid}  [{lid.rsplit('-', 1)[0]}]")
        return
    duel = Duel(a.layout, a.index, human_runs_path=None if a.no_log else HUMAN_RUNS_PATH)
    lines = Path(a.script).read_text().splitlines() if a.script else _stdin_lines()
    run_cli(duel, lines)


if __name__ == "__main__":
    main()
