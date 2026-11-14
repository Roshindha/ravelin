"""B5 gate machinery (docs/SPEC_V2.md 3.4, 3.5, 11): per-layout raw evaluation, paired bootstraps, gates G1-G6, G8, G9, the B5 report.

Pure functions of (design sets, stats, simulator, thresholds): the same inputs give byte-identical `raw` and `gates`.
The CLI is `scripts_gates.py`. Thresholds / bootstrap settings come from data/v2/gate_thresholds.json only.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from . import layouts as Y
from . import oracle as O
from .acceptance import TEMPLATE_BULWARK_MENDER, thresholds, warm_starts
from .armies import free_tiles, greedy_types, place_on_border, random_types, scripted_army, single_unit_types
from .config import CFG, sim_version
from .constants import N_BORDER, UNIT_TYPES
from .sim import Prepared, simulate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"
FAM3 = Y.FAMILIAR
TUNE_SETS = ("TUNE-train", "TUNE-val")
TEMPLATES = ("scout", "sniper", "bulwark", "blaster", "bulwark_mender")   # template i is "owned" by unit i (4 = Mender via bulwark_mender)
STALE_TAG = 9191
N_RANDOM_STALE = 14


# ------------------------------------------------------------------ bootstrap (spec 3.4 paired-difference rule + B5 clarification)
def bootstrap_mean(x, th: dict | None = None) -> dict:
    """Bootstrap over layouts of the mean of `x` (a paired difference or a plain per-layout score). 10 000 resamples, seed 8080.
    lo95 / hi95: percentile 95 % interval; lo_1s: one-sided 98.75 % lower bound (1.25th percentile)."""
    b = (th or thresholds())["bootstrap"]
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(b["seed"])
    means = x[rng.integers(0, len(x), size=(b["n_resamples"], len(x)))].mean(axis=1)
    lo_p, hi_p = 100 * (1 - b["ci_level"]) / 2, 100 * (1 - (1 - b["ci_level"]) / 2)
    one = 100 * (1 - b["one_sided_level"])
    return {"n": int(len(x)), "mean": float(x.mean()), "lo95": float(np.percentile(means, lo_p)), "hi95": float(np.percentile(means, hi_p)),
            "lo_1s": float(np.percentile(means, one)), "hi_1s": float(np.percentile(means, 100 - one))}


# ------------------------------------------------------------------ per-layout raw evaluation (worker)
def _summ(r, types) -> dict:
    return {"score": r.score, "d": r.d, "breach": bool(r.core_destroyed), "units": len(types), "gold": int(sum(CFG.units[t].cost for t in types)),
            "reason": r.reason, "ticks": r.ticks}


def _wall_metrics(r) -> dict:
    return {"attack": list(r.attack_ticks), "wall_att": list(r.wall_attack_ticks), "alive": list(r.alive_ticks), "walls_hit": list(r.walls_hit),
            "wall_any": r.wall_attack_any_ticks, "ticks": r.ticks, "walls_destroyed": r.walls_destroyed}


def evaluate_layout(task: dict) -> dict:
    """Everything the gates need from one layout. `task` = {id, family, seed, set, budgets?}."""
    fam, seed = task["family"], task["seed"]
    b = task.get("budgets") or {}
    nr, nf = b.get("oracle_n_random", O.N_RANDOM), b.get("oracle_n_refine", O.N_REFINE)
    ar, af = b.get("as_n_random", O.AS_RANDOM), b.get("as_n_refine", O.AS_REFINE)
    L = Y.generate(fam, seed)
    P = Prepared(L)

    def run(types, start=None):
        army = place_on_border(L, types, start)
        return simulate(P, army), army

    out = {"id": task["id"], "family": fam, "seed": seed, "set": task["set"]}
    single = []
    for t in range(5):
        types = single_unit_types(t)
        r, _ = run(types)
        single.append({**_summ(r, types), **_wall_metrics(r), "own_attack": r.attack_ticks[t], "own_wall_att": r.wall_attack_ticks[t],
                       "own_alive": r.alive_ticks[t], "own_walls_hit": r.walls_hit[t]})
    out["single"] = single
    r, _ = run(TEMPLATE_BULWARK_MENDER)
    out["template_bulwark_mender"] = _summ(r, TEMPLATE_BULWARK_MENDER)
    gt = greedy_types()
    r, _ = run(gt)
    out["greedy"] = _summ(r, gt)
    ws = warm_starts()
    orc = {}
    for ban in (None, 0, 1, 2, 3, 4):
        allowed = tuple(t for t in range(5) if t != ban)
        res = O.oracle_search(L, O.oracle_rng(seed), n_random=nr, n_refine=nf, allowed=allowed, warm_starts=ws, prepared=P)
        d = {"score": res.score, "d": res.d, "breach": res.breach, "units": res.units, "gold": res.gold, "sims": res.sims,
             "counts": [res.types.count(t) for t in range(5)], "start": res.start}
        if ban is None:
            d["walls"] = _wall_metrics(res.result)
            d["reason"] = res.result.reason
        orc["all" if ban is None else f"ban{ban}"] = d
    out["oracle"] = orc
    sr = O.army_search(L, O.oracle_rng(seed, O.ARMY_SEARCH_TAG), n_random=ar, n_refine=af, prepared=P)
    out["army_search"] = {"score": sr.score, "d": sr.d, "breach": sr.breach, "units": sr.units, "gold": sr.gold, "sims": sr.sims}
    if fam == "labyrinth":   # G9 stalemate check: 21 armies per layout
        rng = np.random.default_rng([seed, STALE_TAG])
        armies = [place_on_border(L, single_unit_types(t)) for t in range(5)] + [scripted_army(L), place_on_border(L, gt)]
        for _ in range(N_RANDOM_STALE):
            armies.append(place_on_border(L, random_types(rng), int(rng.integers(N_BORDER))))
        reasons = Counter()
        bad = 0
        for army in armies:
            r = simulate(P, army)
            reasons[r.reason] += 1
            bad += (r.reason == "stalemate" and not r.core_destroyed)   # structural acceptance guarantees a reachable core
        out["stale"] = {"armies": len(armies), "reasons": dict(reasons), "stalemate_core_alive": int(bad)}
    return out


def design_tasks(path: Path = Y.DESIGN_SETS_PATH, limit: int | None = None, budgets: dict | None = None) -> list:
    meta = Y.load_design_sets_meta(path)
    tasks = []
    for name in (*TUNE_SETS, "LAB-DESIGN"):
        per_fam = Counter()
        for rec in meta["sets"][name]:
            if limit is not None and per_fam[rec["family"]] >= limit:
                continue
            per_fam[rec["family"]] += 1
            if Y.stream_of(rec["seed"]) not in ("T", "L"):
                raise PermissionError(f"seed {rec['seed']} is not in a design stream (T/L)")
            tasks.append({"id": rec["id"], "family": rec["family"], "seed": rec["seed"], "set": name, **({"budgets": budgets} if budgets else {})})
    return tasks


def run_raw(tasks: list, procs: int | None = None) -> list:
    procs = procs or min(8, os.cpu_count() or 1)
    if procs <= 1 or len(tasks) < 2:
        return [evaluate_layout(t) for t in tasks]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(evaluate_layout, tasks, chunksize=1)


# ------------------------------------------------------------------ data access helpers
def _rows(raw, fam=None, sets=None):
    return [r for r in raw if (fam is None or r["family"] == fam) and (sets is None or r["set"] in sets)]


def _tune(raw, fam=None):
    return _rows(raw, fam, TUNE_SETS)


def _col(rows, fn):
    return np.array([fn(r) for r in rows], dtype=float)


def single_scores(rows):
    return np.array([[r["single"][t]["score"] for t in range(5)] for r in rows], dtype=float)


def template_scores(rows):
    return np.array([[r["single"][t]["score"] for t in range(4)] + [r["template_bulwark_mender"]["score"]] for r in rows], dtype=float)


def best_single_type(rows) -> int:
    return int(np.argmax(single_scores(rows).mean(axis=0)))


# ------------------------------------------------------------------ gates
def g1(raw, th):
    rows = _tune(raw)
    S = single_scores(rows)
    share = np.zeros(5)
    for s in S:
        best = np.flatnonzero(s == s.max())
        share[best] += 1 / len(best)
    share /= len(rows)
    per_fam = {}
    for f in FAM3:
        Sf = single_scores(_tune(raw, f))
        sh = np.zeros(5)
        for s in Sf:
            best = np.flatnonzero(s == s.max())
            sh[best] += 1 / len(best)
        per_fam[f] = (sh / len(Sf)).tolist()
    loss = {f: [float(_col(_tune(raw, f), lambda r: r["oracle"]["all"]["score"] - r["oracle"][f"ban{u}"]["score"]).mean()) for u in range(5)] for f in FAM3}
    a_ok = bool(share.max() <= th["G1"]["max_best_single_share"])
    over = [UNIT_TYPES[u] for u in range(5) if all(loss[f][u] > th["G1"]["max_ban_loss_in_every_family"] for f in FAM3)]
    return {"pass": a_ok and not over, "best_single_share_pooled": dict(zip(UNIT_TYPES, share.tolist())), "best_single_share_by_family": per_fam,
            "ban_loss_mean": {f: dict(zip(UNIT_TYPES, v)) for f, v in loss.items()}, "units_with_ban_loss_over_limit_everywhere": over,
            "part_a_pass": a_ok, "part_b_pass": not over}


def g2(raw, th):
    need = th["G2"]["min_ban_loss"]
    table, passing = {}, {}
    for u in range(5):
        for f in FAM3:
            rows = _tune(raw, f)
            diff = _col(rows, lambda r: r["oracle"]["all"]["score"] - r["oracle"][f"ban{u}"]["score"])
            bs = bootstrap_mean(diff, th)
            ok = bs["mean"] >= need and bs["lo95"] > 0 and bs["lo_1s"] > 0
            table[f"{UNIT_TYPES[u]}/{f}"] = {**bs, "pass": bool(ok)}
        passing[UNIT_TYPES[u]] = [f for f in FAM3 if table[f"{UNIT_TYPES[u]}/{f}"]["pass"]]
    return {"pass": all(passing[u] for u in UNIT_TYPES), "units_passing_in_families": passing, "paired_ban_loss": table,
            "families_tested": len(FAM3), "one_sided_level": th["bootstrap"]["one_sided_level"]}


def g3(raw, th):
    need = th["G3"]["min_template_margin"]
    table, wins = {}, {u: [] for u in UNIT_TYPES}
    explo = {}
    for f in FAM3:
        rows = _tune(raw, f)
        T = template_scores(rows)
        means = T.mean(axis=0)
        order = np.argsort(-means, kind="stable")
        w, ru = int(order[0]), int(order[1])
        diff = T[:, w] - T[:, ru]
        bs = bootstrap_mean(diff, th)
        strict = bool(means[w] > means[ru])
        ok = strict and bs["mean"] >= need and bs["lo95"] > 0 and bs["lo_1s"] > 0
        table[f] = {"template_means": dict(zip(TEMPLATES, means.tolist())), "winner": TEMPLATES[w], "runner_up": TEMPLATES[ru], "margin": bs, "pass": bool(ok)}
        if ok:
            wins[UNIT_TYPES[w]].append(f)
        # exploratory (never gates): per unit, share of layouts where its template beats every other template by >= `need`
        for u in range(5):
            others = np.delete(T, u, axis=1).max(axis=1)
            explo[f"{UNIT_TYPES[u]}/{f}"] = float(np.mean(T[:, u] - others >= need))
    n_winners = sum(1 for u in UNIT_TYPES if wins[u])
    return {"pass": all(wins[u] for u in UNIT_TYPES), "units_winning_a_family": wins, "by_family": table,
            "pigeonhole_note": f"{len(UNIT_TYPES)} units but only {len(FAM3)} tested families: under this reading at most {len(FAM3)} units can win a family; {n_winners} did.",
            "exploratory_layout_share_template_beats_all_others_by_margin": explo, "families_tested": len(FAM3)}


def g4(raw, th):
    need = th["G4"]["min_mender_gain"]
    sets = {"bunker_nest": _tune(raw, "bunker_nest"), "LAB-DESIGN": _rows(raw, "labyrinth", ("LAB-DESIGN",))}
    gains = {}
    for name, rows in sets.items():
        bs = bootstrap_mean(_col(rows, lambda r: r["oracle"]["all"]["score"] - r["oracle"]["ban4"]["score"]), th)
        gains[name] = {**bs, "pass": bool(bs["mean"] >= need and bs["lo95"] > 0)}
    only = {}
    for name, rows in {"TUNE": _tune(raw), "LAB-DESIGN": sets["LAB-DESIGN"]}.items():
        bs = bootstrap_mean(_col(rows, lambda r: r["single"][4]["score"]), th)
        only[name] = {**bs, "pass": bool(bs["mean"] < th["G4"]["mender_only_max_mean"] and bs["hi95"] < th["G4"]["mender_only_max_upper95"])}
    a = any(g["pass"] for g in gains.values())
    b = all(o["pass"] for o in only.values())
    return {"pass": a and b, "mender_gain": gains, "mender_only": only, "part_a_pass": a, "part_b_pass": b}


def g5(raw, th):
    lo_s, hi_s = th["G5"]["best_single_mean_range"]
    lo_b, hi_b = th["G5"]["oracle_breach_range"]
    per = {}
    for f in FAM3:
        rows = _tune(raw, f)
        means = single_scores(rows).mean(axis=0)
        br = float(np.mean([r["oracle"]["all"]["breach"] for r in rows]))
        per[f] = {"best_single_type": UNIT_TYPES[int(np.argmax(means))], "best_single_mean": float(means.max()), "oracle_breach_rate": br,
                  "single_ok": bool(lo_s <= means.max() <= hi_s), "breach_ok": bool(lo_b <= br <= hi_b)}
    return {"pass": all(v["single_ok"] and v["breach_ok"] for v in per.values()), "by_family": per}


def g6(raw, th):
    rows = _tune(raw)
    over = _col(rows, lambda r: r["oracle"]["all"]["units"] > th["G6"]["min_units_over"])
    scout20 = [r["id"] for r in rows if r["single"][0]["units"] == 20 and r["single"][0]["score"] >= r["oracle"]["all"]["score"]]
    exact20 = [r["id"] for r in rows if r["oracle"]["all"]["counts"] == [20, 0, 0, 0, 0]]
    per = {f: float(np.mean([r["oracle"]["all"]["units"] > th["G6"]["min_units_over"] for r in _tune(raw, f)])) for f in FAM3}
    a = bool(over.mean() >= th["G6"]["min_layout_share"])
    b = not scout20
    return {"pass": a and b, "share_oracle_over_12_units": float(over.mean()), "by_family": per, "layouts_where_20_scouts_scores_ge_oracle": scout20,
            "layouts_where_oracle_army_is_exactly_20_scouts": exact20, "part_a_pass": a, "part_b_pass": b}


def g8(raw, th):
    per, ok_all = {}, True
    for f in FAM3:
        rows = _tune(raw, f)
        t = best_single_type(rows)
        bs_ = _col(rows, lambda r: r["single"][t]["score"])
        a = _col(rows, lambda r: r["army_search"]["score"]) - bs_
        o = _col(rows, lambda r: r["oracle"]["all"]["score"]) - bs_
        a_b, o_b = bootstrap_mean(a, th), bootstrap_mean(o, th)
        share = float(np.mean(a > 0))
        a_ok = bool(a_b["mean"] >= th["G8"]["army_search_margin"] and share >= th["G8"]["army_search_layout_share"])
        o_ok = bool(o_b["mean"] >= th["G8"]["oracle_margin"])
        per[f] = {"best_single_type": UNIT_TYPES[t], "army_search_gain": a_b, "army_search_better_share": share, "oracle_gain": o_b,
                  "army_search_ok": a_ok, "oracle_ok": o_ok}
        ok_all &= a_ok and o_ok
    return {"pass": bool(ok_all), "by_family": per}


def g9(raw, th):
    rows = _rows(raw, "labyrinth", ("LAB-DESIGN",))
    o = bootstrap_mean(_col(rows, lambda r: r["oracle"]["all"]["score"]), th)
    ob = float(np.mean([r["oracle"]["all"]["breach"] for r in rows]))
    g = bootstrap_mean(_col(rows, lambda r: r["greedy"]["score"]), th)
    gs = float(np.mean([r["greedy"]["score"] >= th["G9"]["greedy_min_layout_score"] for r in rows]))
    stale = int(sum(r["stale"]["stalemate_core_alive"] for r in rows))
    reasons = Counter()
    for r in rows:
        reasons.update(r["stale"]["reasons"])
    lo, hi = th["G9"]["greedy_mean_range"]
    checks = {"oracle_mean": o["mean"] >= th["G9"]["oracle_mean_min"], "oracle_breach": ob >= th["G9"]["oracle_breach_min"],
              "greedy_mean_in_range": lo <= g["mean"] <= hi, "greedy_layout_share": gs >= th["G9"]["greedy_min_layout_share"], "no_stalemate_core_alive": stale == 0}
    return {"pass": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()}, "oracle_mean": o, "oracle_breach_rate": ob, "greedy_mean": g,
            "greedy_share_ge_min_score": gs, "stalemate_with_core_alive": stale, "battle_exit_reasons_21_armies_per_layout": dict(reasons)}


def compute_gates(raw: list, th: dict | None = None) -> dict:
    th = th or thresholds()
    G = {"G1": g1(raw, th), "G2": g2(raw, th), "G3": g3(raw, th), "G4": g4(raw, th), "G5": g5(raw, th), "G6": g6(raw, th), "G8": g8(raw, th), "G9": g9(raw, th)}
    return {"gates": G, "all_pass": all(v["pass"] for v in G.values()), "failed": [k for k, v in G.items() if not v["pass"]]}


# ------------------------------------------------------------------ static per-unit report (spec 11 'B5 report')
def _h(x):   # fixed point -> human units
    return x / CFG.S


def unit_static_report() -> dict:
    D = CFG.defenders
    out = {}
    for u in CFG.units:
        hp, cost, armor, dmg = _h(u.hp), u.cost, _h(u.armor), _h(u.damage)
        rng10 = u.range_x10 if dmg > 0 else 0   # a Mender has no attack range (heal range reported separately)
        row = {"cost": cost, "hp": hp, "armor": armor, "damage": dmg, "range": u.range_x10 / 10, "speed": u.speed_x10 / 10}
        ehp, threat, outr, blind = {}, [], [], []
        for d in D:
            dd = _h(d.damage)
            per_hit = max(dd - armor, _h(CFG.min_hit))
            ehp[d.name] = hp * dd / per_hit / cost
            hits = math.ceil(hp / per_hit)
            if hits <= 3 and d.max_range_x10 >= rng10:
                threat.append({"defender": d.name, "hits_to_kill": hits})
            if d.max_range_x10 > rng10:
                outr.append(d.name)
            if d.min_range_x10 > 0 and dmg > 0 and u.range_x10 <= d.min_range_x10:
                blind.append(d.name)
        fl = next(d for d in D if d.area)
        dn = _h(fl.damage_neighbour)
        row["ehp_per_gold_by_defender"] = ehp
        row["ehp_per_gold_mean_of_three"] = sum(ehp.values()) / len(ehp)
        row["ehp_per_gold_flak_neighbour_hit"] = hp * dn / max(dn - armor, _h(CFG.min_hit)) / cost
        row["dps_per_gold"] = dmg / cost
        if u.splash_pct:
            row["structure_hits_per_tick_3_building_cluster_per_gold"] = (dmg + 2 * dmg * u.splash_pct / 100) / cost
        if u.heal:
            row["heal_per_tick_per_gold"] = _h(u.heal) / cost
        row["threatened_by_defenders_(<=3_hits_and_max_range>=unit_range)"] = threat
        row["outranged_by"] = outr
        row["can_attack_from_inside_blind_spot_of"] = blind
        out[u.name] = row
    return out


# ------------------------------------------------------------------ report tables
def single_unit_table(raw, th) -> dict:
    cols = {**{f: _tune(raw, f) for f in FAM3}, "LAB-DESIGN": _rows(raw, "labyrinth", ("LAB-DESIGN",))}
    out = {}
    for name, rows in cols.items():
        col = {}
        means = single_scores(rows).mean(axis=0)
        for t in range(5):
            sc = bootstrap_mean(_col(rows, lambda r: r["single"][t]["score"]), th)
            dd = bootstrap_mean(_col(rows, lambda r: r["single"][t]["d"]), th)
            br = bootstrap_mean(_col(rows, lambda r: r["single"][t]["breach"]), th)
            col[UNIT_TYPES[t]] = {"score": sc, "D": dd, "breach": br, "units": float(_col(rows, lambda r: r["single"][t]["units"]).mean()),
                                  "gold": float(_col(rows, lambda r: r["single"][t]["gold"]).mean()), "best": bool(t == int(np.argmax(means)))}
        out[name] = {"n": len(rows), "units": col}
    return out


def wall_table(raw) -> dict:
    """Per family: walls attacked per battle per unit type and wall-attack time shares (single-unit armies and Oracle armies)."""
    out = {}
    cols = {**{f: _tune(raw, f) for f in FAM3}, "LAB-DESIGN": _rows(raw, "labyrinth", ("LAB-DESIGN",))}
    for name, rows in cols.items():
        single, mixed = {}, {}
        for t in range(5):
            sa = sum(r["single"][t]["own_attack"] for r in rows)
            sw = sum(r["single"][t]["own_wall_att"] for r in rows)
            sl = sum(r["single"][t]["own_alive"] for r in rows)
            single[UNIT_TYPES[t]] = {"walls_attacked_per_battle": float(np.mean([r["single"][t]["own_walls_hit"] for r in rows])),
                                     "walls_attacked_per_unit": float(sum(r["single"][t]["own_walls_hit"] for r in rows) / max(sum(r["single"][t]["units"] for r in rows), 1)),
                                     "wall_share_of_alive_unit_ticks": sw / sl if sl else 0.0, "wall_share_of_attack_ticks": sw / sa if sa else 0.0,
                                     "share_of_battle_ticks_any_wall_attack": float(sum(r["single"][t]["wall_any"] for r in rows) / max(sum(r["single"][t]["ticks"] for r in rows), 1))}
        for t in range(5):
            w = [r["oracle"]["all"]["walls"] for r in rows]
            al = sum(x["alive"][t] for x in w)
            mixed[UNIT_TYPES[t]] = {"walls_attacked_per_battle": float(np.mean([x["walls_hit"][t] for x in w])),
                                    "wall_share_of_alive_unit_ticks": sum(x["wall_att"][t] for x in w) / al if al else 0.0,
                                    "unit_ticks_alive": int(al)}
        att_alive = lambda ws, key: sum(sum(x[key][t] for t in range(4)) for x in ws)
        ws = [r["oracle"]["all"]["walls"] for r in rows]
        sw_all = {"single_unit_armies": {"wall_share_of_alive_attacker_unit_ticks":
                                         sum(r["single"][t]["own_wall_att"] for r in rows for t in range(4)) / max(sum(r["single"][t]["own_alive"] for r in rows for t in range(4)), 1),
                                         "share_of_battle_ticks_any_wall_attack": sum(r["single"][t]["wall_any"] for r in rows for t in range(4)) / max(sum(r["single"][t]["ticks"] for r in rows for t in range(4)), 1)},
                  "oracle_armies": {"wall_share_of_alive_attacker_unit_ticks": att_alive(ws, "wall_att") / max(att_alive(ws, "alive"), 1),
                                    "share_of_battle_ticks_any_wall_attack": sum(x["wall_any"] for x in ws) / max(sum(x["ticks"] for x in ws), 1)}}
        out[name] = {"single_unit_armies": single, "oracle_armies": mixed, "family_totals": sw_all}
    return out


def defender_mix(sets=None) -> dict:
    """Per family defender mix and the 3.1b generator-requirement checks on the generated design sets."""
    meta = Y.load_design_sets_meta()
    out = {}
    for fam in Y.FAMILIES:
        Ls = [Y.generate(r["family"], r["seed"]) for name in meta["sets"] for r in meta["sets"][name] if r["family"] == fam]
        cnt = Counter(d["type"] for L in Ls for d in L["defenders"])
        n = len(Ls)
        out[fam] = {"layouts": n, "defenders": dict(cnt), "mean_defenders": sum(cnt.values()) / n,
                    "share_with_lancer": sum(any(d["type"] == "lancer" for d in L["defenders"]) for L in Ls) / n,
                    "share_with_flak": sum(any(d["type"] == "flak" for d in L["defenders"]) for L in Ls) / n,
                    "share_gun_majority": sum(sum(d["type"] == "gun" for d in L["defenders"]) >= 0.5 * len(L["defenders"]) for L in Ls) / n,
                    "all_three_types_in_every_layout": all({"gun", "lancer", "flak"} <= {d["type"] for d in L["defenders"]} for L in Ls)}
    return out


def defender_mix_rules(mix: dict) -> dict:
    r = {"spine_pods_lancer>=70%": mix["spine_pods"]["share_with_lancer"] >= 0.70, "bunker_nest_lancer>=70%": mix["bunker_nest"]["share_with_lancer"] >= 0.70,
         "archipelago_flak>=50%": mix["archipelago"]["share_with_flak"] >= 0.50, "labyrinth_uses_all_three": mix["labyrinth"]["all_three_types_in_every_layout"]}
    r.update({f"{f}_gun_majority_every_layout": mix[f]["share_gun_majority"] == 1.0 for f in FAM3})
    return r


def oracle_summary(raw) -> dict:
    out = {}
    cols = {**{f: _tune(raw, f) for f in FAM3}, "LAB-DESIGN": _rows(raw, "labyrinth", ("LAB-DESIGN",))}
    for name, rows in cols.items():
        out[name] = {"oracle_mean_score": float(_col(rows, lambda r: r["oracle"]["all"]["score"]).mean()),
                     "oracle_breach_rate": float(np.mean([r["oracle"]["all"]["breach"] for r in rows])),
                     "oracle_mean_units": float(_col(rows, lambda r: r["oracle"]["all"]["units"]).mean()),
                     "oracle_mean_unit_counts": np.mean([r["oracle"]["all"]["counts"] for r in rows], axis=0).tolist(),
                     "army_search_mean_score": float(_col(rows, lambda r: r["army_search"]["score"]).mean()),
                     "greedy_mean_score": float(_col(rows, lambda r: r["greedy"]["score"]).mean()),
                     "oracle_exit_reasons": dict(Counter(r["oracle"]["all"]["reason"] for r in rows))}
    return out


# ------------------------------------------------------------------ provenance, hashes
def generator_hash() -> str:
    """sha256 of the generator source + parameters (they live in breachlens/layouts.py). B6 asserts it is unchanged."""
    return hashlib.sha256((ROOT / "breachlens" / "layouts.py").read_bytes()).hexdigest()


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}
