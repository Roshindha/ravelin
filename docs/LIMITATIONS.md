# Known limitations (running list, feeds the README)

Numbers only. Source: revision-1 gate run (`data/v2/b5_report.md`, `data/v2/gates.json`), R3 sweep (`data/v2/b5r3_sweep/`), PROGRESS.md. Add to this list; do not delete entries.

## BreachLens v2 design (B5 closed at revision 1; docs/SPEC_V2.md 3.6)
- Revision budget: B5 closed with 2 of 3 revisions unspent (1 held for after G7, 1 spare). Balance gates are proxies; G7 decides learnability.
- G2 / G3, Blaster: monotone ban-loss <= 0.015 in every family; niche share 1.1 % (threshold 8 %). Blaster is rarely worth buying.
- G3, Scout: niche share 7.9 % (threshold 8 %). G2 passes for Scout only in spine_pods.
- G8, bunker_nest: army-search beats best single on 56 % of layouts (threshold 60 %); all mean gains >= +0.20.
- Wall-attack share: 0.50-0.62 of attacker time on bunker_nest and labyrinth.
- Army cap 20 never binds: Oracle armies use at most 12 units (G6 diagnostic).
- Archipelago layout rejection: 60.4 % (122 of 202 candidates); other families 7.0 % (spine_pods), 16.7 % (bunker_nest), 31.0 % (labyrinth).
- Blaster / Scout compete for the same role: in the R3 sweep, every Blaster buff arm made Blaster the best single unit (labyrinth best single 0.688-0.705, Oracle 0.976-0.993, both above the bands) and cut Scout's niche to 1-2.5 %. No arm applied.
- G4 was re-defined after the revision-1 result (changelog (g)); under the original definition it fails (Mender gain bunker_nest +0.017 < 0.02, LAB-DESIGN 0.000). Under (g) it passes (spine_pods +0.029, archipelago +0.031).
- One-time waiver, changelog (f): item 9 criterion (ii) for R2a (labyrinth best single 0.669 vs 0.65, n = 20).
- Sample sizes: sweeps used 20 layouts per family; the sampling error exceeded the margin in the item 9 case above.
- Design sets: TUNE-train 180, TUNE-val 60, LAB-DESIGN 40; stats were tuned on them, so gate numbers are optimistic for those sets.

## Prototype lessons that still apply (PROGRESS.md)
- Resumed PPO runs are not bit-identical to uninterrupted runs (RNG state not in checkpoints; reseeded as seed + 100003 * k). Prototype seeds 13-15 were resumed twice.
- Two seeds of the same config differed by 0.08 on validation (SD about 0.05-0.06); a 3-seed mean has SE about 0.03. Differences under 0.05 are not distinguishable.
- `best.zip` is selected on 20 validation layouts and is optimistic; headline numbers use `final.zip`.
- Models stop at rollout granularity (about 1.0015M steps for a 1M budget).
- Config C (lr 1e-4, 5 epochs) collapsed to one troop type per seed (val 0.23 vs 0.33).
- The prototype transfer family had BestSingle = 0.000, so BestSingle is uninformative there; v2 reports transfer against Oracle / Greedy / Random.
