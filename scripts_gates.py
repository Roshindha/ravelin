"""B5 gate script: G1-G6, G8, G9 on TUNE / LAB-DESIGN (docs/SPEC_V2.md 3.4) + the B5 report.

  python scripts_gates.py            # evaluate all 280 layouts (Oracle searches, ~minutes on 8 cores), write raw + gates + report
  python scripts_gates.py --report   # regenerate gates.json / b5_report.{json,md} from data/v2/b5_raw.json (no simulation)

Outputs: data/v2/b5_raw.json, data/v2/gates.json, data/v2/b5_report.json, data/v2/b5_report.md.
`generator_hash` is recorded to data/v2/generator_hash.json ONLY if every gate passes. Never touches stream F / FINAL.
"""
import json
import sys
import time

from breachlens import gates as G
from breachlens import config as K
from breachlens import layouts as Y
from breachlens.acceptance import thresholds
from breachlens.config import sim_version

REVISION = 0   # B5 design revision number (0 = the spec's initial design; 1 = B5-R1, see REVISION_NOTES); incremented only by the owner's decision
REVISION_NOTES = {0: 'initial design', 1: 'B5-R1: see data/v2/b5r1_sweep/sweep_summary.md (chosen arm applied)'}


def fmt_ci(b, p=3):
    return f"{b['mean']:.{p}f} [{b['lo95']:.{p}f}, {b['hi95']:.{p}f}]"


def markdown(rep) -> str:
    g, meta, th = rep["gates"]["gates"], rep["meta"], rep["thresholds"]
    L = []
    L.append("# B5 report (BreachLens v2)\n")
    L.append(f"Revision {meta['revision']} | commit `{meta['git']['commit'][:10]}` (dirty={meta['git']['dirty']}) | sim_version `{meta['sim_version'][:16]}...` | "
             f"layouts: {meta['counts']} | bootstrap: {th['bootstrap']['n_resamples']} resamples, seed {th['bootstrap']['seed']}\n")
    L.append(f"**All gates pass: {rep['gates']['all_pass']}**  (failed: {', '.join(rep['gates']['failed']) or 'none'}).  generator_hash recorded: {meta['generator_hash_recorded']}\n")
    L.append("## 1. G1-G9 summary\n")
    L.append("| Gate | Result | Key numbers |\n|---|---|---|")
    k = {}
    k["G1"] = (f"max best-single share {max(g['G1']['best_single_share_pooled'].values()):.2f} (limit 0.60); "
               f"units over ban-loss 0.25 everywhere: {g['G1']['units_with_ban_loss_over_limit_everywhere'] or 'none'}")
    k["G2"] = "units passing in: " + "; ".join(f"{u}: {','.join(f) or 'NONE'}" for u, f in g["G2"]["units_passing_in_families"].items())
    k["G3"] = "niche-layout share (need >= 8%, lo95 > 0): " + "; ".join(f"{u} {v['mean']:.1%} [{v['lo95']:.1%}]" for u, v in g["G3"]["niche_share"].items())
    k["G4"] = (f"Mender gain bunker_nest {fmt_ci(g['G4']['mender_gain']['bunker_nest'])}, LAB {fmt_ci(g['G4']['mender_gain']['LAB-DESIGN'])}; "
               f"Mender-only TUNE {g['G4']['mender_only']['TUNE']['mean']:.3f} / LAB {g['G4']['mender_only']['LAB-DESIGN']['mean']:.3f}")
    k["G5"] = "; ".join(f"{f}: best-single {v['best_single_type']} {v['best_single_mean']:.3f}, Oracle breach {v['oracle_breach_rate']:.2f}" for f, v in g["G5"]["by_family"].items())
    k["G6"] = (f"Oracle >12 units on {g['G6']['share_oracle_over_12_units']:.2f} of layouts (need >=0.30); layouts with 20 Scouts >= Oracle: "
               f"{len(g['G6']['layouts_where_20_scouts_scores_ge_oracle'])}")
    k["G8"] = "; ".join(f"{f}: army-search +{v['army_search_gain']['mean']:.3f} ({v['army_search_better_share']:.0%} layouts), Oracle +{v['oracle_gain']['mean']:.3f}" for f, v in g["G8"]["by_family"].items())
    k["G9"] = (f"Oracle {fmt_ci(g['G9']['oracle_mean'])}, breach {g['G9']['oracle_breach_rate']:.2f}; Greedy {fmt_ci(g['G9']['greedy_mean'])}, "
               f"{g['G9']['greedy_share_ge_min_score']:.0%} of layouts >= 0.05; stalemate with core alive: {g['G9']['stalemate_with_core_alive']}")
    for name in ("G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9"):
        L.append(f"| {name} | **{'PASS' if g[name]['pass'] else 'FAIL'}** | {k[name]} |")
    L.append("\nG7 is not evaluated at B5 (pilot, B13).\n")
    L.append("## 2. Gate details\n")
    L.append("### G1\nbest-single-unit share per layout (ties split), pooled TUNE: " + ", ".join(f"{u} {v:.2f}" for u, v in g["G1"]["best_single_share_pooled"].items()) + "\n")
    L.append("| family | " + " | ".join(Y.FAMILIAR) + " |\n|---|---|---|---|")
    for u_i, u in enumerate(G.UNIT_TYPES):
        L.append(f"| best-single share {u} | " + " | ".join(f"{g['G1']['best_single_share_by_family'][f][u_i]:.2f}" for f in Y.FAMILIAR) + " |")
    for u in G.UNIT_TYPES:
        L.append(f"| Oracle ban-loss {u} | " + " | ".join(f"{g['G1']['ban_loss_mean'][f][u]:.3f}" for f in Y.FAMILIAR) + " |")
    L.append("\n### G2 paired Oracle ban-loss (mean [95% CI]; one-sided 98.75% lower bound; need mean >= 0.02, lo95 > 0, lo_1s > 0)\n")
    L.append("| unit | family | ban-loss | lo(98.75% 1-sided) | pass |\n|---|---|---|---|---|")
    for key, v in g["G2"]["paired_ban_loss"].items():
        u, f = key.split("/")
        L.append(f"| {u} | {f} | {fmt_ci(v)} | {v['lo_1s']:.3f} | {'yes' if v['pass'] else 'no'} |")
    L.append("\n### G3 per-layout niches (B5-R1: pooled TUNE-train + TUNE-val + LAB-DESIGN, margin >= 0.03 over the runner-up template; need share >= 8% and lo95 > 0)\n")
    L.append("Mean template scores: " + ", ".join(f"{k} {v:.3f}" for k, v in g["G3"]["mean_template_scores"].items()) + "\n")
    L.append("| unit | template | niche layouts | share [95% CI] | pass |\n|---|---|---|---|---|")
    for u, v in g["G3"]["niche_share"].items():
        L.append(f"| {u} | {v['template']} | {v['layouts']} / {g['G3']['layouts_pooled']} | {fmt_ci(v)} | {'yes' if v['pass'] else 'no'} |")
    L.append("\nDescriptive only (per family, share of layouts where the unit's template wins by >= 0.03): " + "; ".join(
        f"{f}: " + ", ".join(f"{u} {x[u]:.2f}" for u in G.UNIT_TYPES) + f", no clear winner {x['no_niche_winner_(margin<need)']:.2f}" for f, x in g["G3"]["layout_share_by_family_(descriptive_only)"].items()) + "\n")
    L.append("### G4\n" + "\n".join(f"- Mender gain on {k}: {fmt_ci(v)} -> {'ok' if v['pass'] else 'no'}" for k, v in g["G4"]["mender_gain"].items()))
    L.append("\n" + "\n".join(f"- Mender-only army on {k}: mean {fmt_ci(v)} (upper95 {v['hi95']:.3f}) -> {'ok' if v['pass'] else 'no'}" for k, v in g["G4"]["mender_only"].items()) + "\n")
    L.append("### G5 / G6 / G8\n")
    L.append("| family | best single | mean | Oracle breach | G8a army-search gain [95% CI] | better on | G8b Oracle gain [95% CI] |\n|---|---|---|---|---|---|---|")
    for f in Y.FAMILIAR:
        a, b = g["G5"]["by_family"][f], g["G8"]["by_family"][f]
        L.append(f"| {f} | {a['best_single_type']} | {a['best_single_mean']:.3f} | {a['oracle_breach_rate']:.2f} | {fmt_ci(b['army_search_gain'])} | {b['army_search_better_share']:.0%} | {fmt_ci(b['oracle_gain'])} |")
    L.append(f"\nG6: Oracle armies >12 units: {g['G6']['share_oracle_over_12_units']:.2f} (by family {', '.join(f'{k} {v:.2f}' for k, v in g['G6']['by_family'].items())}); "
             f"20 Scouts >= Oracle on {len(g['G6']['layouts_where_20_scouts_scores_ge_oracle'])} layouts; Oracle army exactly 20 Scouts on {len(g['G6']['layouts_where_oracle_army_is_exactly_20_scouts'])}.\n")
    G9 = g["G9"]
    L.append(f"### G9 (LAB-DESIGN, 40)\nchecks: {G9['checks']}; exit reasons over all battles: {G9['battle_exit_reasons_21_armies_per_layout']}\n")
    L.append("## 3. Per-unit statics (effective hp per gold incl. armor, dps per gold, threatening defenders)\n")
    L.append("| unit | cost | hp | armor | EHP/gold Gun | Lancer | Flak (centre) | Flak (neighbour hit) | mean of 3 | DPS/gold | threatened by (<=3 hits & range) | outranged by | can hit from inside blind spot of |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for u, v in rep["unit_static"].items():
        e = v["ehp_per_gold_by_defender"]
        dps = f"{v['dps_per_gold']:.3f}" + (f" (3-bldg cluster {v['structure_hits_per_tick_3_building_cluster_per_gold']:.3f})" if "structure_hits_per_tick_3_building_cluster_per_gold" in v else "") + (f" (heal {v['heal_per_tick_per_gold']:.3f})" if "heal_per_tick_per_gold" in v else "")
        thr = ", ".join(f"{t['defender']} ({t['hits_to_kill']} hits)" for t in v["threatened_by_defenders_(<=3_hits_and_max_range>=unit_range)"]) or "none"
        L.append(f"| {u} | {v['cost']} | {v['hp']:.0f} | {v['armor']:.0f} | {e['gun']:.2f} | {e['lancer']:.2f} | {e['flak']:.2f} | {v['ehp_per_gold_flak_neighbour_hit']:.2f} | {v['ehp_per_gold_mean_of_three']:.2f} | {dps} | {thr} | {', '.join(v['outranged_by']) or 'none'} | {', '.join(v['can_attack_from_inside_blind_spot_of']) or 'none'} |")
    L.append("\n## 4. Single-unit-army score table (mean score [95% CI] | D | breach | units / gold; * = best per family)\n")
    cols = list(rep["single_unit_table"])
    L.append("| unit | " + " | ".join(f"{c} (n={rep['single_unit_table'][c]['n']})" for c in cols) + " |\n|---|" + "---|" * len(cols))
    for u in G.UNIT_TYPES:
        cells = []
        for c in cols:
            v = rep["single_unit_table"][c]["units"][u]
            cells.append(f"{fmt_ci(v['score'])}{'*' if v['best'] else ''} / D {v['D']['mean']:.3f} / breach {v['breach']['mean']:.2f} / {v['units']:.1f}u {v['gold']:.0f}g")
        L.append(f"| {u} | " + " | ".join(cells) + " |")
    L.append("\n## 5. Wall behaviour (walls attacked per battle per unit type; share of time spent attacking walls)\n")
    L.append("| family | unit | walls attacked / battle (single army) | wall share of alive unit-ticks (single) | walls attacked / battle (Oracle army) | wall share (Oracle army) |\n|---|---|---|---|---|---|")
    for f, v in rep["wall_metrics"].items():
        for u in G.UNIT_TYPES[:4]:
            a, b = v["single_unit_armies"][u], v["oracle_armies"][u]
            L.append(f"| {f} | {u} | {a['walls_attacked_per_battle']:.2f} | {a['wall_share_of_alive_unit_ticks']:.3f} | {b['walls_attacked_per_battle']:.2f} | {b['wall_share_of_alive_unit_ticks']:.3f} |")
    L.append("\n| family | single-unit armies: wall-attack share of attacker unit-ticks | ticks with any wall attack / battle ticks | Oracle armies: share of attacker unit-ticks | ticks with any wall attack |\n|---|---|---|---|---|")
    for f, v in rep["wall_metrics"].items():
        t = v["family_totals"]
        L.append(f"| {f} | {t['single_unit_armies']['wall_share_of_alive_attacker_unit_ticks']:.3f} | {t['single_unit_armies']['share_of_battle_ticks_any_wall_attack']:.3f} | {t['oracle_armies']['wall_share_of_alive_attacker_unit_ticks']:.3f} | {t['oracle_armies']['share_of_battle_ticks_any_wall_attack']:.3f} |")
    L.append("\n## 6. Oracle / baselines per family\n")
    L.append("| set | Oracle mean | Oracle breach | Oracle units | Oracle counts s/sn/b/bl/m | army-search mean | Greedy mean |\n|---|---|---|---|---|---|---|")
    for f, v in rep["oracle_summary"].items():
        L.append(f"| {f} | {v['oracle_mean_score']:.3f} | {v['oracle_breach_rate']:.2f} | {v['oracle_mean_units']:.1f} | {', '.join(f'{x:.1f}' for x in v['oracle_mean_unit_counts'])} | {v['army_search_mean_score']:.3f} | {v['greedy_mean_score']:.3f} |")
    L.append("\n## 7. Defender mix per family and generator-rule checks (spec 3.1b)\n")
    L.append("| family | layouts | mean defenders | gun / lancer / flak total | share with Lancer | share with Flak | gun majority |\n|---|---|---|---|---|---|---|")
    for f, v in rep["defender_mix"].items():
        d = v["defenders"]
        L.append(f"| {f} | {v['layouts']} | {v['mean_defenders']:.1f} | {d.get('gun', 0)} / {d.get('lancer', 0)} / {d.get('flak', 0)} | {v['share_with_lancer']:.2f} | {v['share_with_flak']:.2f} | {v['share_gun_majority']:.2f} |")
    L.append(f"\nRule checks: {rep['defender_mix_rules']}\n")
    L.append("## 8. Stat-dependent acceptance (too_hard, single_type_trivial) on TUNE / LAB-DESIGN streams\n")
    L.append("| family | structurally accepted & evaluated | single_type_trivial | too_hard | rejected (either) | rejection rate | flag >50% |\n|---|---|---|---|---|---|---|")
    for f, s in rep["stat_acceptance"].items():
        L.append(f"| {f} | {s['stat_evaluated']} | {s['single_type_trivial']} | {s['too_hard']} | {s['stat_rejected']} | {s['stat_rejection_rate']:.1%} | {'**FLAG**' if s['flag_above_50pct'] else 'no'} |")
    return "\n".join(L) + "\n"


def build_report(raw, record_hash=True, git=None):
    th = thresholds()
    gates = G.compute_gates(raw, th)
    meta_sets = Y.load_design_sets_meta()
    mix = G.defender_mix()
    rep = {
        "meta": {"revision": REVISION, "revision_notes": REVISION_NOTES[REVISION], "git": git or G.git_info(), "sim_version": sim_version(), "counts": meta_sets["counts"], "generator_hash": G.generator_hash(),
                 "generator_hash_recorded": False, "layouts_per_set": {k: len(v) for k, v in meta_sets["sets"].items()}},
        "thresholds": th, "gates": gates, "unit_static": G.unit_static_report(), "single_unit_table": G.single_unit_table(raw, th),
        "wall_metrics": G.wall_table(raw), "oracle_summary": G.oracle_summary(raw), "defender_mix": mix, "defender_mix_rules": G.defender_mix_rules(mix),
        "stat_acceptance": meta_sets["stat_acceptance"],
    }
    if gates["all_pass"] and record_hash:
        (G.DATA / "generator_hash.json").write_text(json.dumps({"generator_hash": rep["meta"]["generator_hash"], "source": "breachlens/layouts.py (sha256)",
                                                                  "commit": rep["meta"]["git"]["commit"], "revision": REVISION}, indent=1) + "\n")
        rep["meta"]["generator_hash_recorded"] = True
    return rep


def write_outputs(raw, rep):
    (G.DATA / "gates.json").write_text(json.dumps({"revision": REVISION, "git": rep["meta"]["git"], "sim_version": rep["meta"]["sim_version"],
                                                    "seed_sets": rep["meta"]["layouts_per_set"], "all_pass": rep["gates"]["all_pass"], "failed": rep["gates"]["failed"],
                                                    "gates": rep["gates"]["gates"]}, indent=1) + "\n")
    (G.DATA / "b5_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (G.DATA / "b5_report.md").write_text(markdown(rep))


if __name__ == "__main__":
    t0 = time.time()
    git0 = G.git_info()    # captured BEFORE any output file is written, so a run from a clean commit records dirty=false
    assert not (K.stats_override_active() or Y.gen_override_active()), "sweep override hooks must not be set for a gate run"
    if "--report" in sys.argv:
        raw = json.loads((G.DATA / "b5_raw.json").read_text())["layouts"]
    else:
        raw = G.run_raw(G.design_tasks())
        (G.DATA / "b5_raw.json").write_text(json.dumps({"sim_version": sim_version(), "layouts": raw}) + "\n")
    rep = build_report(raw, git=git0)
    write_outputs(raw, rep)
    print(f"done in {time.time() - t0:.0f}s; all_pass={rep['gates']['all_pass']} failed={rep['gates']['failed']}")
