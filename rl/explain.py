"""B16a: explainability core, "Why this move?" (docs/SPEC_V2.md, Gate changelog row (ii)). No UI, no training, no new results.

``explain(...)`` turns one (learner, observation, mask) into a plain-data explanation: legal-action distribution, deploy heatmap, top-5 labelled
actions and a value estimate. ``explain_layout_step(...)`` plays a DEV layout with the learner's own deterministic policy (the B15 plain policy, or
the B15 TTA policy) up to a step and explains that state. Every number is a read-out of a frozen checkpoint, not an evaluation result.

Action index map (Discrete(267); ``breachlens.constants`` / ``breachlens.actions``):
    0 .. 4        buy unit u              u = index into UNIT_TYPES (scout, sniper, bulwark, blaster, mender)
    5 + 52*u + b  deploy unit u on border tile b   (u in 0..4, b in 0..51; b indexes ``breachlens.actions.BORDER``, clockwise from (0, 0))
    265           END_BUY (stop buying, start deploying)
    266           WAIT (stop deploying, start the battle)
So ``divmod(a - 5, 52) = (unit type, border tile)`` for 5 <= a < 265. The heatmap is ``heat[u][b] = P(deploy unit u on tile b)``; ``heat_tile[b]``
sums over unit types; ``heat_total`` is the summed probability of all deploy actions (the heatmap totals equal it exactly).

Views:
* plain, PPO: the masked softmax of the actor (illegal = exactly 0). plain, DDQN: Q over the legal actions, plus a softmax of Q at the FIXED
  temperature ``DDQN_TEMPERATURE`` = 0.1, labelled "not a policy" (DDQN is trained to estimate returns, not to emit probabilities).
* TTA (all 8 D4 views, aggregation from ``rl.tta.tta_scores``, the B15 code, not reimplemented): PPO = mean log-probability of the legal actions
  over the views, displayed as its softmax over legal actions (the renormalised geometric mean; same argmax as the B15 TTA action);
  DDQN = plain mean of the 8 Q-views, then the same T = 0.1 softmax view (still "not a policy").
* Value: PPO critic V(s) (TTA: mean of V over the 8 views); DDQN max over legal actions of Q (TTA: of the 8-view mean Q). Both are estimates of the
  SHAPED return (potential Phi = 0.5 x preview-battle score, row (z)), not of the game score.

Layouts: DEV familiar only (DEV-train, DEV-validation) through the role-guarded pool loaders (via ``breachlens.evaluator``) under role train / validation; FINAL, transfer and
design-set layouts raise. DEV-train explanations are labelled in-sample. Display models: seed 0 of each arm.

    .venv/bin/python -m rl.explain --arm ppo_plain --split DEV-validation --index 0 --step 3
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np

from breachlens.actions import BORDER, decode_action
from breachlens.constants import DEPLOY_BASE, END_BUY, N_ACTIONS, N_BORDER, N_UNIT_TYPES, UNIT_TYPES
from breachlens.env import BreachEnv
from rl.augment import N_SYMMETRIES, transform_obs

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models_v2"
DDQN_TEMPERATURE = 0.1
TOP_K = 5
# arm -> (run directory under models_v2, algorithm); seed 0 of each arm, fixed in Gate changelog row (ii)
DISPLAY_MODELS = {"ppo_plain": ("grid_ppo_plain_s0", "ppo"), "ppo_d4": ("grid_ppo_d4_s0", "ppo"), "ddqn": ("grid_ddqn_s0", "ddqn")}
SPLIT_ROLE = {"DEV-train": "train", "DEV-validation": "validation"}
VALUE_LABEL = "estimate of the SHAPED return (Phi = 0.5 x preview-battle score), not the game score"
DDQN_NOT_POLICY = "not a policy: softmax of Q at a fixed display temperature"


# ----------------------------------------------------------------------------------------------------------------------------- models
def load_display_model(arm: str, models_dir: Path = MODELS):
    """Load ONE checkpoint (seed 0 of the arm) exactly as the B15 session does. Returns (model_or_net, algo, run_name)."""
    if arm not in DISPLAY_MODELS:
        raise ValueError(f"unknown arm {arm!r}; one of {sorted(DISPLAY_MODELS)}")
    run, algo = DISPLAY_MODELS[arm]
    d = Path(models_dir) / run
    if algo == "ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(d / "final.zip", device="cpu"), algo, run
    import torch as th
    from rl import ddqn as D
    ck = th.load(d / "final.pt", map_location="cpu", weights_only=False)
    net = D.QNet(D.default_obs_space(), ck["n_actions"], ck["hyperparameters"]["net_arch"])
    net.load_state_dict(ck["online"])
    net.eval()
    return net, algo, run


def _output_fn(model, algo: str):
    """probs_fn(obs, mask) -> (267,) for one observation: masked probabilities (PPO) or raw Q (DDQN); the B15 adapters."""
    if algo == "ppo":
        from rl.eval_ppo_v2 import masked_probs
        return masked_probs(model)
    from rl.ddqn import q_values_fn
    return q_values_fn(model)


def _ppo_value(model, obs: dict) -> float:
    import torch as th
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        return float(model.policy.predict_values(obs_t)[0, 0].cpu())


# ----------------------------------------------------------------------------------------------------------------------------- labels
def action_label(action: int) -> str:
    kind, u, b = decode_action(action)
    if kind == "buy":
        return f"Buy {UNIT_TYPES[u]}"
    if kind == "deploy":
        r, c = BORDER[b]
        return f"Deploy {UNIT_TYPES[u]} at border tile {b} (row {r}, col {c})"
    return "End buying (start deploying)" if kind == "end_buy" else "Wait (start the battle)"


def _softmax_legal(scores: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over the legal actions only; illegal entries are exactly 0."""
    out = np.zeros(N_ACTIONS, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)[mask] / temperature
    e = np.exp(s - s.max())
    out[mask] = e / e.sum()
    return out


def _heatmap(dist: np.ndarray) -> dict:
    heat = dist[DEPLOY_BASE:END_BUY].reshape(N_UNIT_TYPES, N_BORDER)
    return {"units": list(UNIT_TYPES), "tiles": [list(t) for t in BORDER], "heat": heat.tolist(),
            "heat_tile": heat.sum(axis=0).tolist(), "heat_unit": heat.sum(axis=1).tolist(), "heat_total": float(heat.sum())}


# ----------------------------------------------------------------------------------------------------------------------------- core
def explain(model, algo: str, obs: dict, mask: np.ndarray, *, tta: bool = False, temperature: float = DDQN_TEMPERATURE, top_k: int = TOP_K) -> dict:
    """Explain one state. `model` is a MaskablePPO (algo "ppo") or a QNet (algo "ddqn"); returns plain Python data (JSON-serialisable)."""
    from rl.tta import tta_scores
    mask = np.asarray(mask, dtype=bool)
    fn = _output_fn(model, algo)
    views = list(range(N_SYMMETRIES)) if tta else [0]
    if algo == "ppo":
        if tta:
            scores = tta_scores(fn, obs, mask, aggregate="logprob")                  # mean log-prob of legal actions, -inf elsewhere (B15)
            dist = _softmax_legal(np.where(mask, scores, 0.0), mask)                 # softmax of the mean log-prob; illegal exactly 0
            score_name, score_vals = "mean_logprob", np.where(mask, scores, -np.inf)
            value = float(np.mean([_ppo_value(model, transform_obs(obs, s)) for s in views]))
        else:
            dist = np.asarray(fn(obs, mask), dtype=np.float64)
            score_name, score_vals = "probability", np.where(mask, dist, -np.inf)
            value = _ppo_value(model, obs)
        distribution = {"kind": "policy", "label": "PPO policy: softmax over legal actions" + (" (TTA: softmax of the 8-view mean log-prob)" if tta else ""),
                        "probs": dist.tolist()}
    else:
        q = tta_scores(fn, obs, mask, aggregate="mean") if tta else np.asarray(fn(obs, mask), dtype=np.float64)
        dist = _softmax_legal(q, mask, temperature)
        score_name, score_vals = "q", np.where(mask, q, -np.inf)
        value = float(q[mask].max())
        distribution = {"kind": "q_softmax", "label": DDQN_NOT_POLICY, "temperature": temperature, "probs": dist.tolist(),
                        "q": [float(x) if m else None for x, m in zip(q, mask)]}
    greedy = int(np.argmax(score_vals))                                              # lowest index wins an exact tie, as the eval policies
    order = sorted(np.flatnonzero(mask).tolist(), key=lambda a: (-float(score_vals[a]), a))[:top_k]
    top = [{"action": int(a), "label": action_label(a), "probability": float(dist[a]), "q": distribution["q"][a] if algo == "ddqn" else None,
            "score": float(score_vals[a])} for a in order]
    return {"algo": algo, "tta": bool(tta), "n_views": len(views), "n_legal": int(mask.sum()), "score_name": score_name,
            "greedy_action": greedy, "greedy_label": action_label(greedy), "distribution": distribution, "top": top,
            "heatmap": _heatmap(dist), "value": {"estimate": value, "label": VALUE_LABEL,
                                                 "definition": "critic V(s)" if algo == "ppo" else "max over legal actions of Q"}}


# ----------------------------------------------------------------------------------------------------------------------------- layouts and states
def load_dev_layout(split: str, index: int | None = None, layout_id: str | None = None) -> tuple[dict, dict]:
    """(record, layout) of one DEV-train / DEV-validation layout through the role-guarded pool loaders (``breachlens.evaluator``).
    Any other split raises PermissionError before anything is loaded."""
    from breachlens import evaluator as EV
    return EV.load_dev_layout(split, index, layout_id, entry_point="rl.explain")


def _choose(model, algo: str, tta: bool):
    """The B15 action chooser of this learner: plain deterministic masked argmax, or the 8-view TTA argmax."""
    if tta:
        from rl.tta import tta_action
        fn = _output_fn(model, algo)
        agg = "logprob" if algo == "ppo" else "mean"
        return lambda o, m: tta_action(fn, o, m, aggregate=agg)
    if algo == "ppo":
        return lambda o, m: int(model.predict(o, action_masks=m, deterministic=True)[0])
    from rl.ddqn import DDQNPolicy
    pol = DDQNPolicy(model)
    return pol.act


def play_to_step(model, algo: str, rec: dict, layout: dict, step: int, tta: bool = False):
    """Play the learner's own deterministic policy from the B15 reset seed for `step` actions. Returns (env, obs, actions_so_far)."""
    env = BreachEnv(layouts=[layout], role="train")
    obs, _ = env.reset(seed=zlib.crc32(rec["id"].encode()), options={"layout_index": 0})
    choose, taken = _choose(model, algo, tta), []
    for _ in range(step):
        if env.done:
            raise ValueError(f"the episode ends before step {step} ({len(taken)} actions)")
        a = int(choose(obs, env.action_masks()))
        taken.append(a)
        obs, _, done, _, _ = env.step(a)
    if env.done:
        raise ValueError(f"the episode is over at step {step}; choose a step < {len(taken)}")
    return env, obs, taken


def explain_layout_step(arm: str, split: str, index: int | None = None, step: int = 0, tta: bool = False, layout_id: str | None = None,
                        models_dir: Path = MODELS, loaded=None) -> dict:
    """Explain `arm`'s seed-0 model at (DEV layout, step). `loaded` = (model, algo, run) lets callers reuse one loaded checkpoint."""
    rec, layout = load_dev_layout(split, index, layout_id)
    model, algo, run = loaded if loaded is not None else load_display_model(arm, models_dir)
    env, obs, taken = play_to_step(model, algo, rec, layout, step, tta)
    ex = explain(model, algo, obs, env.action_masks(), tta=tta)
    ex.update(arm=arm, run=run, layout_id=rec["id"], family=rec["family"], split=split, in_sample=(split == "DEV-train"), step=step,
              phase=["BUY", "DEPLOY", "BATTLE"][env.phase], gold=float(env.gold), bought=int(env.bought), actions_so_far=taken,
              note=("IN-SAMPLE: this layout was in the learner's training set" if split == "DEV-train" else "held-out DEV layout (DEV-validation)"))
    return ex


# ----------------------------------------------------------------------------------------------------------------------------- text
def format_text(ex: dict) -> str:
    d = ex["distribution"]
    out = [f"Why this move?  arm={ex['arm']} ({ex['run']})  view={'TTA x8' if ex['tta'] else 'plain'}",
           f"layout {ex['layout_id']} [{ex['family']}]  split={ex['split']}  {ex['note']}",
           f"step {ex['step']}  phase={ex['phase']}  gold={ex['gold']:.0f}  bought={ex['bought']}  legal actions={ex['n_legal']}",
           f"distribution: {d['label']}" + (f" (T={d['temperature']})" if "temperature" in d else ""),
           "top 5 legal actions:"]
    for i, t in enumerate(ex["top"], 1):
        q = f"  Q={t['q']:+.4f}" if t["q"] is not None else ""
        out.append(f"  {i}. [{t['action']:3d}] {t['label']:<52s} p={t['probability']:.4f}{q}")
    out.append(f"chosen (argmax): [{ex['greedy_action']}] {ex['greedy_label']}")
    v = ex["value"]
    out.append(f"value: {v['estimate']:+.4f}  ({v['definition']}; {v['label']})")
    h = ex["heatmap"]
    out.append(f"deploy heatmap: total deploy probability {h['heat_total']:.4f}; per unit " + ", ".join(f"{u}={x:.3f}" for u, x in zip(h["units"], h["heat_unit"])))
    if h["heat_total"] > 0:
        best = sorted(range(N_BORDER), key=lambda b: (-h["heat_tile"][b], b))[:5]
        out.append("  top border tiles: " + ", ".join(f"{b} (row {h['tiles'][b][0]}, col {h['tiles'][b][1]}) {h['heat_tile'][b]:.3f}" for b in best))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Print the explanation of one learner's move at one DEV layout and step (text only).")
    p.add_argument("--arm", required=True, choices=sorted(DISPLAY_MODELS))
    p.add_argument("--split", default="DEV-validation", choices=sorted(SPLIT_ROLE))
    p.add_argument("--index", type=int, default=0, help="position of the layout inside the split")
    p.add_argument("--layout-id", default=None, help="alternative to --index")
    p.add_argument("--step", type=int, default=0, help="number of actions already taken (0 = the first BUY state)")
    p.add_argument("--tta", action="store_true", help="8-view TTA view (B15 aggregation)")
    a = p.parse_args(argv)
    ex = explain_layout_step(a.arm, a.split, a.index, a.step, a.tta, a.layout_id)
    print(format_text(ex))
    return ex


if __name__ == "__main__":
    main()
