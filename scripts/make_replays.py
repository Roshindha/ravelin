"""B16b: self-contained HTML replays with the "Why this move?" panel (docs/SPEC_V2.md, Gate changelog row (jj)). No training, no new results.

    .venv/bin/python scripts/make_replays.py                  # writes docs/replays/ (18 replays + index.html)
    .venv/bin/python scripts/make_replays.py --out /some/dir

Replay set (row (jj) (a)): the FIRST DEV-validation record of each familiar family (pool order), x 3 seed-0 display arms x {plain, TTA} = 18.
Every number shown is read from the frozen checkpoint through ``rl.explain`` (``explain`` / ``_choose`` = the B15 choosers) and from the simulator;
the generator evaluates nothing else and writes no timestamp, so two runs give byte-identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breachlens import pool as P                                   # noqa: E402
from breachlens.config import sim_version                          # noqa: E402
from breachlens.constants import GRID, N_BORDER, UNIT_TYPES        # noqa: E402
from breachlens.env import BreachEnv                               # noqa: E402
from breachlens.score import SCORE_VERSION, score_value            # noqa: E402
from breachlens.sim import simulate                                # noqa: E402
from rl import explain as X                                        # noqa: E402

OUT_DIR = ROOT / "docs" / "replays"
SPLIT = "DEV-validation"
FAMILIES = ("spine_pods", "archipelago", "bunker_nest")
ARMS = ("ppo_plain", "ppo_d4", "ddqn")
VIEWS = (False, True)                                              # plain, TTA
TOTAL_CAP = 15 * 1024 * 1024
FILE_CAP = int(1.5 * 1024 * 1024)
RATES = (1, 2, 3, 4, 6, 8, 12)                                     # battle-frame subsampling: smallest rate that meets the per-file cap
SPLIT_LABEL = "DEV-validation, held out from training"
CODE_PATHS = ("scripts/make_replays.py", "rl", "breachlens")


# ----------------------------------------------------------------------------------------------------------------------------- selection
def replay_set(split: str = SPLIT) -> list[tuple]:
    """[(family, arm, tta)] x layouts fixed by row (jj): the first record of each family in the DEV-validation pool order. Refuses other splits."""
    if split not in X.SPLIT_ROLE:
        raise PermissionError(f"only DEV familiar layouts may be loaded; {split!r} is refused")
    if split != SPLIT:
        raise PermissionError(f"the replay set of row (jj) is {SPLIT} only; {split!r} is refused")
    return [(fam, arm, tta) for fam in FAMILIES for arm in ARMS for tta in VIEWS]


def first_layouts(split: str = SPLIT) -> dict:
    """family -> (position in the split, record, layout): the first record of each family, loaded through the role-guarded DEV loader."""
    replay_set(split)
    recs = P.load_pool_records(split, "validation", entry_point="scripts.make_replays")
    out = {}
    for i, r in enumerate(recs):
        if r["family"] in FAMILIES and r["family"] not in out:
            rec, layout = X.load_dev_layout(split, i)                      # the guarded loader (evaluator.load_dev_layout)
            assert rec["id"] == r["id"]
            out[r["family"]] = (i, rec, layout)
    assert set(out) == set(FAMILIES)
    return out


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("log", "-1", "--format=%H", "--", *CODE_PATHS)
    dirty = bool(git("status", "--porcelain", "--", *CODE_PATHS))
    return {"commit": commit + ("+dirty" if dirty else ""), "sim_version": sim_version(), "score_version": SCORE_VERSION}


# ----------------------------------------------------------------------------------------------------------------------------- one episode
def _r(x, n=4):
    return None if x is None else round(float(x), n)


def _struct_table(P_) -> list:
    """[kind, row, col, value-or-type] per structure sid, in the simulator's order (buildings, core, defenders, walls)."""
    from breachlens.sim import COL, ROW
    kinds = ["b"] * P_.n_buildings + ["c"] + ["d"] * P_.n_def + ["w"] * P_.n_walls
    out = []
    for s, k in enumerate(kinds):
        c = P_.s_cell[s]
        extra = P_.s_value[s] if k == "b" else (P_.cfg.defenders[P_.s_dtype[s]].name if k == "d" and hasattr(P_.cfg.defenders[P_.s_dtype[s]], "name") else "")
        out.append([k, ROW[c], COL[c], extra])
    return out


def _permille(hp, mx):
    return int(round(1000 * hp / mx)) if mx else 0


def run_episode(model, algo: str, arm: str, run: str, rec: dict, layout: dict, tta: bool) -> dict:
    """Play the learner's deterministic B15 episode, explaining every decision step; then record the battle. Plain data."""
    env = BreachEnv(layouts=[layout], role="train")
    obs, _ = env.reset(seed=zlib.crc32(rec["id"].encode()), options={"layout_index": 0})
    choose = X._choose(model, algo, tta)
    steps = []
    while not env.done:
        mask = env.action_masks()
        a = int(choose(obs, mask))
        ex = X.explain(model, algo, obs, mask, tta=tta)
        assert ex["greedy_action"] == a, (arm, tta, rec["id"], len(steps))      # fidelity: the explanation's argmax IS the eval-path action
        h = ex["heatmap"]
        steps.append({
            "phase": ["BUY", "DEPLOY", "BATTLE"][env.phase], "gold": int(env.gold), "bought": int(env.bought), "owned": [int(x) for x in env.owned],
            "army": [[int(t), int(b)] for t, b in env.army], "action": a, "label": X.action_label(a), "n_legal": ex["n_legal"],
            "top": [{"a": t["action"], "label": t["label"], "p": _r(t["probability"]), "q": _r(t["q"])} for t in ex["top"]],
            "p_chosen": _r(ex["distribution"]["probs"][a]), "heat": [[_r(x) for x in row] for row in h["heat"]], "heat_tile": [_r(x) for x in h["heat_tile"]],
            "heat_unit": [_r(x) for x in h["heat_unit"]], "heat_total": _r(h["heat_total"]), "value": _r(ex["value"]["estimate"])})
        obs, _, _, _, info = env.step(a)
    res = env.result
    outcome = {"score": score_value(res.d_num, res.d_den, res.core_destroyed), "d_num": int(res.d_num), "d_den": int(res.d_den),
               "d": res.d_num / max(res.d_den, 1), "core": bool(res.core_destroyed), "ticks": int(res.ticks), "reason": res.reason,
               "n_units": len(env.army), "steps": len(steps), "info_score": info["score"]}
    assert outcome["score"] == info["score"]
    return {"steps": steps, "outcome": outcome, "battle": _battle(env), "structs": _struct_table(env._st.P)}


def _battle(env) -> dict:
    """Per-tick unit state (recorded re-run of the same battle) and per-tick structure hp (prefix re-runs, validated against the trace)."""
    Pp = env._st.P
    full = simulate(Pp, env.army, record=True)
    assert (full.d_num, full.d_den, full.core_destroyed, full.ticks) == (env.result.d_num, env.result.d_den, env.result.core_destroyed, env.result.ticks)
    umax = [Pp.cfg.units[t].hp for t, _ in env.army]
    smax = Pp.s_hp0
    frames = [{"t": 0, "u": [[None, 1000, 0] for _ in env.army], "s": [1000] * len(smax)}]
    from breachlens.actions import BORDER
    from breachlens.sim import cell_of
    frames[0]["u"] = [[cell_of(BORDER[b]), 1000, 0] for _, b in env.army]
    for t in range(1, full.ticks + 1):
        pre = simulate(Pp, env.army, max_ticks=t)
        assert [h for _, h, _ in full.trace[t - 1]] == list(pre.unit_hp), "prefix run diverges from the recorded trace"
        frames.append({"t": t, "u": [[c, _permille(h, umax[i]) if st >= 0 else 0, st] for i, (c, h, st) in enumerate(full.trace[t - 1])],
                       "s": [_permille(h, smax[s]) for s, h in enumerate(pre.struct_hp)]})
    assert list(pre.struct_hp) == list(env.result.struct_hp)
    return {"frames": frames, "units": [[int(t), int(b)] for t, b in env.army]}


def subsample(frames: list, rate: int) -> list:
    keep = [f for f in frames if f["t"] % rate == 0]
    if keep[-1]["t"] != frames[-1]["t"]:
        keep.append(frames[-1])
    return keep


# ----------------------------------------------------------------------------------------------------------------------------- html
CSS = """
:root{--bg:#12151c;--panel:#1b2030;--line:#2e3650;--fg:#e6e9f2;--mut:#9aa3bd;--acc:#ffb347;--ok:#6fd08c;--bad:#ff6b6b;--hot:#ffd34d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header,main,footer{max-width:1180px;margin:0 auto;padding:12px 16px}h1{font-size:20px;margin:4px 0}h2{font-size:15px;margin:0 0 6px;color:var(--mut);font-weight:600}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}.badge{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:2px 8px}
.badge b{color:var(--acc)}.outcome{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0}
.yes{color:var(--ok)}.no{color:var(--bad)}.wrap{display:flex;flex-wrap:wrap;gap:16px}.col{flex:1 1 360px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:12px}
canvas{display:block;max-width:100%;height:auto;background:#0d1017;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
tr.chosen td{background:#3a2d12;color:var(--hot);font-weight:600}.bar{height:8px;background:var(--acc);border-radius:2px;min-width:1px}
.mut{color:var(--mut)}.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}button,select{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
button[aria-pressed=true]{border-color:var(--acc);color:var(--acc)}input[type=range]{flex:1 1 300px}details{margin:6px 0}summary{cursor:pointer;color:var(--mut)}
code{background:#0d1017;padding:1px 4px;border-radius:3px}a{color:var(--acc)}.legend{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--mut)}
.grad{width:120px;height:10px;background:linear-gradient(90deg,#1b2030,#7a3d00,#ff9d00,#ffe566);border:1px solid var(--line)}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const UT = D.unit_types, N = D.grid, CELL = 40, W = N * CELL;
const UCOL = ['#4fc3f7', '#ba68c8', '#81c784', '#ff8a65', '#fff176'], UAB = ['Sc', 'Sn', 'Bu', 'Bl', 'Me'];
const BT = D.border;                                  // border tile -> [row, col]
const nDec = D.steps.length, nFr = D.battle.frames.length, total = nDec + nFr;
const $ = id => document.getElementById(id);
let pos = 0, heatMode = 'chosen', timer = null;
const cv = $('grid'), cx = cv.getContext('2d');
cv.width = W; cv.height = W;
function fmt(x, n) { return x === null || x === undefined ? 'n/a' : x.toFixed(n === undefined ? 4 : n); }
function heatColor(f) {                               // 0..1 -> dark..yellow (sequential)
  const s = [[27, 32, 48], [122, 61, 0], [255, 157, 0], [255, 229, 102]], p = Math.min(Math.max(f, 0), 1) * 3, i = Math.min(Math.floor(p), 2), t = p - i;
  return 'rgb(' + [0, 1, 2].map(k => Math.round(s[i][k] + (s[i + 1][k] - s[i][k]) * t)).join(',') + ')';
}
function drawStruct(s, hp) {
  const [k, r, c, ex] = s, x = c * CELL, y = r * CELL, dead = hp <= 0;
  cx.globalAlpha = dead ? 0.25 : 1;
  if (k === 'w') { cx.fillStyle = '#6b7388'; cx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4); }
  else if (k === 'b') { cx.fillStyle = '#4f7fbf'; cx.fillRect(x + 4, y + 4, CELL - 8, CELL - 8); cx.fillStyle = '#fff'; cx.font = 'bold 14px sans-serif'; cx.textAlign = 'center'; cx.fillText(String(ex), x + CELL / 2, y + CELL / 2 + 5); }
  else if (k === 'c') { cx.fillStyle = '#e8c547'; cx.beginPath(); cx.moveTo(x + CELL / 2, y + 3); cx.lineTo(x + CELL - 3, y + CELL / 2); cx.lineTo(x + CELL / 2, y + CELL - 3); cx.lineTo(x + 3, y + CELL / 2); cx.closePath(); cx.fill(); cx.fillStyle = '#000'; cx.font = 'bold 12px sans-serif'; cx.textAlign = 'center'; cx.fillText('C', x + CELL / 2, y + CELL / 2 + 4); }
  else { cx.fillStyle = '#d9534f'; cx.beginPath(); cx.arc(x + CELL / 2, y + CELL / 2, CELL / 2 - 5, 0, 7); cx.fill(); cx.fillStyle = '#fff'; cx.font = 'bold 12px sans-serif'; cx.textAlign = 'center'; cx.fillText(String(ex).charAt(0).toUpperCase(), x + CELL / 2, y + CELL / 2 + 4); }
  cx.globalAlpha = 1;
  if (k !== 'w' && hp < 1000 && !dead) { cx.fillStyle = '#000'; cx.fillRect(x + 4, y + CELL - 7, CELL - 8, 4); cx.fillStyle = '#6fd08c'; cx.fillRect(x + 4, y + CELL - 7, (CELL - 8) * hp / 1000, 4); }
  if (dead && k !== 'w') { cx.strokeStyle = '#ff6b6b'; cx.lineWidth = 2; cx.beginPath(); cx.moveTo(x + 6, y + 6); cx.lineTo(x + CELL - 6, y + CELL - 6); cx.moveTo(x + CELL - 6, y + 6); cx.lineTo(x + 6, y + CELL - 6); cx.stroke(); }
}
function drawUnit(t, x, y, hp, st) {
  if (hp <= 0) return;
  cx.fillStyle = UCOL[t]; cx.beginPath(); cx.arc(x + CELL / 2, y + CELL / 2, 13, 0, 7); cx.fill();
  if (st === 2) { cx.strokeStyle = '#fff'; cx.lineWidth = 2; cx.stroke(); }
  cx.fillStyle = '#000'; cx.font = 'bold 11px sans-serif'; cx.textAlign = 'center'; cx.fillText(UAB[t], x + CELL / 2, y + CELL / 2 + 4);
  cx.fillStyle = '#000'; cx.fillRect(x + 6, y + 2, CELL - 12, 3); cx.fillStyle = '#6fd08c'; cx.fillRect(x + 6, y + 2, (CELL - 12) * hp / 1000, 3);
}
function heatFor(s) {                                  // returns {vals: per border tile, label}
  const hv = s.heat_total;
  if (heatMode === 'total') return {vals: s.heat_tile, label: 'per-tile total over all unit types'};
  let u = -1;
  const m = s.action - D.deploy_base;
  if (m >= 0 && m < 5 * D.n_border) u = Math.floor(m / D.n_border);
  let note = 'unit type of the chosen action: ' + UT[Math.max(u, 0)];
  if (u < 0) { u = s.heat_unit.indexOf(Math.max.apply(null, s.heat_unit)); note = 'no deploy chosen at this step; showing the unit type with the most deploy probability (' + UT[u] + ')'; }
  return {vals: s.heat[u], label: note};
}
function draw() {
  cx.clearRect(0, 0, W, W);
  cx.fillStyle = '#0d1017'; cx.fillRect(0, 0, W, W);
  cx.strokeStyle = '#1f2638'; cx.lineWidth = 1;
  for (let i = 0; i <= N; i++) { cx.beginPath(); cx.moveTo(i * CELL, 0); cx.lineTo(i * CELL, W); cx.moveTo(0, i * CELL); cx.lineTo(W, i * CELL); cx.stroke(); }
  const dec = pos < nDec;
  let hinfo = '';
  if (dec) {
    const s = D.steps[pos], h = heatFor(s), mx = Math.max.apply(null, h.vals.concat([1e-12]));
    if (s.heat_total > 0) {
      h.vals.forEach((v, b) => { const [r, c] = BT[b]; cx.fillStyle = heatColor(v / mx); cx.globalAlpha = v > 0 ? 0.9 : 0.15; cx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2); cx.globalAlpha = 1; });
      hinfo = h.label + '; colour scale 0 to ' + fmt(mx) + ' (max over border tiles); total deploy probability ' + fmt(s.heat_total);
    } else hinfo = 'no deploy action is legal at this step, so the heatmap is empty (total deploy probability 0)';
    D.structs.forEach(st => drawStruct(st, 1000));
    s.army.forEach(([t, b]) => { const [r, c] = BT[b]; drawUnit(t, c * CELL, r * CELL, 1000, 0); });
    const m = s.action - D.deploy_base;
    if (m >= 0 && m < 5 * D.n_border) { const b = m % D.n_border, [r, c] = BT[b]; cx.strokeStyle = '#fff'; cx.lineWidth = 3; cx.strokeRect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4); }
  } else {
    const f = D.battle.frames[pos - nDec];
    D.structs.forEach((st, i) => drawStruct(st, f.s[i]));
    f.u.forEach((u, i) => { const [t] = D.battle.units[i], c = u[0]; drawUnit(t, (c % N) * CELL, Math.floor(c / N) * CELL, u[1], u[2]); });
  }
  $('heatinfo').textContent = hinfo;
}
function panel() {
  const dec = pos < nDec;
  $('decpanel').style.display = dec ? '' : 'none'; $('batpanel').style.display = dec ? 'none' : '';
  $('where').textContent = dec ? 'decision step ' + (pos + 1) + ' of ' + nDec : 'battle, tick ' + D.battle.frames[pos - nDec].t + ' of ' + D.outcome.ticks;
  if (dec) {
    const s = D.steps[pos];
    $('phase').textContent = s.phase + ' phase, gold ' + s.gold + ', units bought ' + s.bought + ', legal actions ' + s.n_legal;
    $('chosen').textContent = s.label + '  (probability ' + fmt(s.p_chosen) + (D.algo === 'ddqn' ? ', softmax of Q: not a policy' : '') + ')';
    $('toplabel').textContent = D.algo === 'ddqn' ? 'Top 5 actions by Q. Softmax column is NOT a policy (fixed temperature ' + D.temperature + ').' : 'Top 5 actions by probability' + (D.tta ? ' (TTA: softmax of the 8-view mean log-probability).' : '.');
    $('top').innerHTML = '<tr><th>#</th><th>action</th>' + (D.algo === 'ddqn' ? '<th>Q</th><th>softmax</th>' : '<th>prob.</th>') + '<th></th></tr>' + s.top.map((t, i) =>
      '<tr class="' + (t.a === s.action ? 'chosen' : '') + '"><td>' + (i + 1) + '</td><td>' + t.label + (t.a === s.action ? ' &#9664; chosen' : '') + '</td>' +
      (D.algo === 'ddqn' ? '<td>' + fmt(t.q) + '</td><td>' + fmt(t.p) + '</td>' : '<td>' + fmt(t.p) + '</td>') + '<td style="width:25%"><div class="bar" style="width:' + Math.round(100 * t.p) + '%"></div></td></tr>').join('');
    $('value').textContent = fmt(s.value);
  } else {
    const f = D.battle.frames[pos - nDec];
    const alive = f.u.filter(u => u[1] > 0).length, sd = D.structs.reduce((n, st, i) => n + (st[0] === 'b' && f.s[i] <= 0 ? 1 : 0), 0), nb = D.structs.filter(st => st[0] === 'b').length;
    const core = D.structs.findIndex(st => st[0] === 'c');
    $('bat').textContent = 'tick ' + f.t + ': ' + alive + ' of ' + f.u.length + ' units alive; ' + sd + ' of ' + nb + ' buildings destroyed; core ' + (f.s[core] <= 0 ? 'destroyed' : 'standing') + '. White ring = unit attacked this tick.';
  }
  $('scrub').value = pos;
  spark();
}
function spark() {
  const c = $('spark'), g = c.getContext('2d'), w = c.width, h = c.height, v = D.steps.map(s => s.value), lo = Math.min.apply(null, v), hi = Math.max.apply(null, v), pad = 6;
  g.clearRect(0, 0, w, h); g.fillStyle = '#0d1017'; g.fillRect(0, 0, w, h);
  const X = i => pad + (w - 2 * pad) * (v.length > 1 ? i / (v.length - 1) : 0.5), Y = x => h - pad - (h - 2 * pad) * (hi > lo ? (x - lo) / (hi - lo) : 0.5);
  g.strokeStyle = '#2e3650'; if (lo < 0 && hi > 0) { g.beginPath(); g.moveTo(pad, Y(0)); g.lineTo(w - pad, Y(0)); g.stroke(); }
  g.strokeStyle = '#ffb347'; g.lineWidth = 2; g.beginPath(); v.forEach((x, i) => i ? g.lineTo(X(i), Y(x)) : g.moveTo(X(i), Y(x))); g.stroke();
  const cur = Math.min(pos, nDec - 1); g.fillStyle = '#fff'; g.beginPath(); g.arc(X(cur), Y(v[cur]), 4, 0, 7); g.fill();
  $('sparinfo').textContent = 'min ' + fmt(lo) + ', max ' + fmt(hi) + ', at the marker ' + fmt(v[cur]) + ' (decision step ' + (cur + 1) + ')';
}
function go(p) { pos = Math.max(0, Math.min(total - 1, p)); draw(); panel(); }
$('scrub').max = total - 1;
$('scrub').oninput = e => go(+e.target.value);
$('prev').onclick = () => go(pos - 1); $('next').onclick = () => go(pos + 1);
$('play').onclick = () => { if (timer) { clearInterval(timer); timer = null; $('play').textContent = 'Play'; return; } if (pos >= total - 1) pos = 0; $('play').textContent = 'Pause'; timer = setInterval(() => { if (pos >= total - 1) { clearInterval(timer); timer = null; $('play').textContent = 'Play'; return; } go(pos + 1); }, pos < nDec ? 500 : 80); };
['chosen', 'total'].forEach(m => $('h_' + m).onclick = () => { heatMode = m; $('h_chosen').setAttribute('aria-pressed', m === 'chosen'); $('h_total').setAttribute('aria-pressed', m === 'total'); draw(); });
document.onkeydown = e => { if (e.key === 'ArrowRight') go(pos + 1); if (e.key === 'ArrowLeft') go(pos - 1); };
const h0 = parseInt(location.hash.slice(1));
go(isNaN(h0) ? 0 : h0);          // #N opens at scrubber position N (0-based)
"""


def _esc_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True).replace("</", "<\\/")


def _name(arm: str, family: str, tta: bool) -> str:
    return f"{family}__{arm}__{'tta' if tta else 'plain'}.html"


def build_html(meta: dict, ep: dict, rate: int) -> str:
    out = ep["outcome"]
    frames = subsample(ep["battle"]["frames"], rate)
    data = {"algo": meta["algo"], "tta": meta["tta"], "temperature": X.DDQN_TEMPERATURE, "grid": GRID, "n_border": N_BORDER, "unit_types": list(UNIT_TYPES),
            "deploy_base": 5, "border": meta["border"], "steps": ep["steps"], "structs": ep["structs"], "outcome": out,
            "battle": {"frames": frames, "units": ep["battle"]["units"], "rate": rate}}
    view = "TTA (8 views)" if meta["tta"] else "plain"
    brk = "YES" if out["core"] else "NO"
    head_rows = [("commit", meta["commit"]), ("model id", f"{meta['run']} (final checkpoint sha256 {meta['model_sha'][:16]})"), ("layout id", meta["layout_id"]),
                 ("layout family", meta["family"]), ("split", SPLIT_LABEL), ("sim_version", meta["sim_version"]), ("SCORE_VERSION", meta["score_version"]),
                 ("battle frames", f"every {rate} tick(s) of {out['ticks']}" + (" (subsampled; the last tick is always kept)" if rate > 1 else " (all ticks)")),
                 ("episode", "the learner's own deterministic B15 episode (reset seed = crc32 of the layout id)")]
    meta_html = "".join(f"<tr><td class='mut'>{html.escape(k)}</td><td><code>{html.escape(str(v))}</code></td></tr>" for k, v in head_rows)
    title = f"Replay: {meta['arm']} {view} on {meta['layout_id']}"
    kind = ("DDQN: Q-values. The softmax of Q is NOT a policy (fixed display temperature " + str(X.DDQN_TEMPERATURE) + ")." if meta["algo"] == "ddqn"
            else "PPO: the actor's masked softmax over legal actions.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><h1>Why this move? {html.escape(meta['arm'])}, {view}</h1>
<div class="badges"><span class="badge">arm <b>{html.escape(meta['arm'])}</b></span><span class="badge">model <b>{html.escape(meta['run'])}</b></span>
<span class="badge">split <b>{html.escape(SPLIT_LABEL)}</b></span><span class="badge">view <b>{view}</b></span><span class="badge">family <b>{html.escape(meta['family'])}</b></span></div>
<div class="outcome"><b>Episode outcome (true result of this replay):</b> score <b>{out['score']:.4f}</b> &middot; D = {out['d']:.4f} ({out['d_num']}/{out['d_den']} building value destroyed)
&middot; core breached: <b class="{'yes' if out['core'] else 'no'}">{brk}</b> &middot; {out['n_units']} units deployed &middot; battle {out['ticks']} ticks ({html.escape(out['reason'])})</div>
<details><summary>Provenance</summary><table>{meta_html}</table></details></header>
<main><div class="wrap"><div class="col" style="flex:0 0 580px"><div class="card"><canvas id="grid" aria-label="14 by 14 grid"></canvas>
<div class="ctl"><span class="mut">Heatmap:</span><button id="h_chosen" aria-pressed="true">Chosen unit type</button><button id="h_total" aria-pressed="false">Per-tile total</button>
<span class="legend"><span>0</span><span class="grad"></span><span>max</span></span></div><div class="mut" id="heatinfo"></div>
<div class="mut" style="margin-top:6px">Blue squares: buildings (number = value). Diamond C: core. Grey: walls. Red circles: defenders (G gun, L lancer, F flak). Coloured circles: your units (Sc scout, Sn sniper, Bu bulwark, Bl blaster, Me mender). White square: tile of the chosen deploy.</div></div></div>
<div class="col"><div class="card"><div class="ctl"><button id="prev">&larr; Prev</button><button id="play">Play</button><button id="next">Next &rarr;</button>
<input type="range" id="scrub" min="0" max="0" value="0" aria-label="step scrubber"></div><div id="where"><b></b></div>
<div class="mut">Decision steps ({len(ep['steps'])}: buy, deploy, end-buy, wait), then the battle ({len(frames)} frames). Arrow keys also work.</div></div>
<div class="card" id="decpanel"><h2>Why this move?</h2><div class="mut" id="phase"></div><div style="margin:6px 0"><b>Chosen:</b> <span id="chosen"></span></div>
<div class="mut" id="toplabel"></div><table id="top"></table>
<div style="margin-top:8px"><b>Value estimate:</b> <span id="value"></span> <span class="mut">{html.escape(X.VALUE_LABEL)}; {'critic V(s)' if meta['algo']=='ppo' else 'max over legal actions of Q'}{' (TTA: mean over 8 views)' if meta['tta'] else ''}. It is not comparable with the final score.</span></div></div>
<div class="card" id="batpanel" style="display:none"><h2>Battle</h2><div id="bat"></div></div>
<div class="card"><h2>Value over the decision steps (shaped-return estimate)</h2><canvas id="spark" width="520" height="90"></canvas><div class="mut" id="sparinfo"></div></div></div></div></main>
<footer class="mut">{html.escape(kind)} Layout from {html.escape(SPLIT_LABEL)}. Everything on this page is a read-out of a frozen checkpoint; nothing was trained or tuned for it. <a href="index.html">All replays</a></footer>
<script type="application/json" id="data">{_esc_json(data)}</script><script>{JS}</script></body></html>
"""


def build_index(rows: list, prov: dict) -> str:
    body = []
    for r in rows:
        o = r["outcome"]
        body.append(f"<tr><td>{r['family']}</td><td>{r['arm']}</td><td>{'TTA' if r['tta'] else 'plain'}</td><td><a href=\"{r['file']}\">{html.escape(r['layout_id'])}</a></td>"
                    f"<td>{o['score']:.4f}</td><td>{o['d']:.4f}</td><td class=\"{'yes' if o['core'] else 'no'}\">{'yes' if o['core'] else 'no'}</td><td>{r['size']:,} B</td></tr>")
    meta = f"commit {prov['commit']}; sim_version {prov['sim_version']}; SCORE_VERSION {prov['score_version']}; split {SPLIT_LABEL}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Why this move? replays</title><style>{CSS}</style></head><body>
<header><h1>Why this move? 18 replays</h1><p class="mut">First DEV-validation layout of each familiar family, seed-0 display model of each arm, plain and TTA. The set was fixed by index
(Gate changelog row (jj)), not by outcome; every replay shows its true outcome, including failures. Split: {html.escape(SPLIT_LABEL)}.</p>
<p class="mut">Provenance: <code>{html.escape(meta)}</code></p></header>
<main><div class="card"><table><tr><th>family</th><th>arm</th><th>view</th><th>layout (replay)</th><th>score</th><th>D</th><th>breach</th><th>file size</th></tr>{''.join(body)}</table></div>
<p class="mut">DDQN softmax bars are labelled "not a policy". Values are shaped-return estimates, not game scores.</p></main></body></html>
"""


# ----------------------------------------------------------------------------------------------------------------------------- driver
def generate(out_dir: Path = OUT_DIR, split: str = SPLIT) -> list:
    """Write the 18 replays + index.html into `out_dir`; return the per-replay rows. Refuses non-DEV splits before loading anything."""
    jobs = replay_set(split)
    layouts = first_layouts(split)
    prov = provenance()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from breachlens.actions import BORDER
    rows, loaded = [], {}
    for fam, arm, tta in jobs:
        pos, rec, layout = layouts[fam]
        if arm not in loaded:
            loaded.clear()                                               # one checkpoint in memory at a time
            loaded[arm] = X.load_display_model(arm)
        model, algo, run = loaded[arm]
        ep = run_episode(model, algo, arm, run, rec, layout, tta)
        ckpt = X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt")
        meta = {"arm": arm, "run": run, "algo": algo, "tta": tta, "layout_id": rec["id"], "family": fam, "border": [list(b) for b in BORDER],
                "model_sha": hashlib.sha256(ckpt.read_bytes()).hexdigest(), **prov}
        for rate in RATES:
            page = build_html(meta, ep, rate)
            if len(page.encode()) <= FILE_CAP:
                break
        else:
            raise SystemExit(f"STOP: {arm}/{fam}/tta={tta} exceeds the {FILE_CAP} B per-file cap even at rate {RATES[-1]}; report options, do not drop replays")
        fn = _name(arm, fam, tta)
        (out_dir / fn).write_bytes(page.encode())
        rows.append({"family": fam, "arm": arm, "tta": tta, "layout_id": rec["id"], "file": fn, "size": len(page.encode()), "outcome": ep["outcome"],
                     "rate": rate, "n_steps": len(ep["steps"]), "actions": [s["action"] for s in ep["steps"]], "position": pos})
    total = sum(r["size"] for r in rows)
    if total > TOTAL_CAP:
        raise SystemExit(f"STOP: total {total} B exceeds the {TOTAL_CAP} B cap; report options, do not drop replays")
    idx = build_index(rows, prov)
    (out_dir / "index.html").write_bytes(idx.encode())
    return rows


def main(argv=None) -> list:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--split", default=SPLIT, help=f"only {SPLIT} is accepted (row (jj)); any other split is refused")
    a = p.parse_args(argv)
    rows = generate(Path(a.out), a.split)
    for r in rows:
        o = r["outcome"]
        print(f"{r['arm']:10s} {'TTA  ' if r['tta'] else 'plain'} {r['family']:12s} score {o['score']:.4f} D {o['d']:.4f} breach {'yes' if o['core'] else 'no ':3s} {r['file']} {r['size']:,} B rate {r['rate']}")
    return rows


if __name__ == "__main__":
    main()
