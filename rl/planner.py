"""B12: policy-guided planner, equal-N controls, cost projection and the 7.2 fallback selector (docs/SPEC_V2.md 7, 7.2, 11 B12; Gate changelog (q)).

Definitions (pinned in Gate changelog (q) before any trained model exists)
  * A PLAN is one complete episode: buy phase + deployment, ending in exactly ONE call of the simulator (`breachlens.sim.simulate`).
    The budget N is a number of simulated plans; it is the same unit for the planner, RAS-N and RPS-N. Wall time is reported separately.
  * Planner: plan k of (layout, seed) is a rollout of the STOCHASTIC masked policy from the empty BUY phase (canonical view; a D4-trained
    model is not rotated). Illegal actions get weight 0, the rest are sampled in proportion to the policy's probabilities.
  * RAS-N: plan k = a random army from the phase-A generator (`armies.random_types`, stop prob 0.1, cap 20, all 5 types), put in the
    scripted deployer's order (`armies.sort_by_rank`, the Greedy ranking) and placed with `armies.place_on_border` from the default start
    tile (the same scripted deployment as army search / Greedy). No network.
  * RPS-N: plan k = uniformly random LEGAL env action at every step (BUY / END_BUY / DEPLOY / WAIT), through `BreachEnv`. No network.
  * RPS-full-N (diagnostic, Gate changelog (r)): RPS-N with END_BUY masked while any BUY action is legal (a unit is affordable and the cap is not
    reached) and WAIT masked while bought units remain undeployed (a mask is applied only if a legal action remains). Same N, seeding, nesting.
  * NESTING: plan k uses its own random stream `default_rng([seed, crc32(layout id), METHOD_TAG, k])`, so it does not depend on N or on any
    other plan; the first 8 plans of the 32 are the 8 plans, the first 32 of the 128 are the 32. One batch is sampled once per
    (method, layout, seed, policy) and N = 8 / 32 / 128 read its prefixes (`PlanCache`).
  * CHOICE: the plan with the largest score among the first N, compared as the exact integer ratio from `breachlens.score.score_ratio`
    (cross-multiplication, no float); an exact tie goes to the LOWEST plan index. Deterministic given (layout, seed, policy).
  * Policies are queried one observation at a time (batch size 1). Batching would make a plan depend on the batch size through float
    round-off and break the exact nesting; speed is bought elsewhere (see the cost report).
  * FINAL: every `episodes()` call refuses a FINAL / transfer layout unless the object was built with role="eval", unlock=True (the evaluator and
    the pool loaders add their own guards and write the ledger).
  * Scores: only through `breachlens.score` (here `score_ratio` for the choice; the evaluator converts outcomes with `score_value`).
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from breachlens import evaluator as EV
from breachlens.armies import place_on_border, random_types, sort_by_rank
from breachlens.baselines import Policy, _ep
from breachlens.constants import N_UNIT_TYPES
from breachlens.env import BreachEnv
from breachlens.score import score_ratio
from breachlens.sim import simulate

N_VALUES = (8, 32, 128)
N_MAX = max(N_VALUES)
METHOD_TAG = {"planner": 1, "RAS": 2, "RPS": 3, "RPSfull": 4}
METHOD_LABEL = {"planner": "planner", "RAS": "RAS", "RPS": "RPS", "RPSfull": "RPS-full"}
FALLBACK_LIMIT_H = 3.0
WORKERS = 7


# ------------------------------------------------------------------------------------------------------------ helpers
def plan_rng(seed: int, layout_id: str, method: str, k: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(layout_id.encode()), METHOD_TAG[method], int(k)])


def sample_action(probs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Sample a LEGAL action with probability proportional to `probs` (illegal entries are zeroed first, whatever the policy says)."""
    mask = np.asarray(mask, dtype=bool)
    w = np.where(mask, np.asarray(probs, dtype=np.float64), 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    c = np.cumsum(w)
    if not c[-1] > 0:
        raise ValueError("policy puts no probability on any legal action")
    a = int(np.searchsorted(c, rng.random() * c[-1], side="right"))
    if a >= len(w) or w[a] <= 0:                       # float edge: fall back to the last action with positive weight
        a = int(np.flatnonzero(w > 0)[-1])
    return a


def better(a: tuple, b: tuple) -> bool:
    """a, b = (num, den) exact score ratios; True iff a > b (cross-multiplied, no float)."""
    return a[0] * b[1] > b[0] * a[1]


def plan_ratio(plan: dict) -> tuple:
    return score_ratio(plan["d_num"], plan["d_den"], plan["core"])


def choose(plans: list) -> int:
    """Index of the best plan; an exact tie goes to the lowest index."""
    best = 0
    for i in range(1, len(plans)):
        if better(plan_ratio(plans[i]), plan_ratio(plans[best])):
            best = i
    return best


def state_dict_hash(module) -> str:
    """sha256 over the sorted (name, dtype, shape, bytes) of a torch module's state dict: the identity of a policy's weights."""
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        a = v.detach().cpu().numpy()
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class ModelProbs:
    """probs_fn for a MaskablePPO model (picklable, for the multiprocessing evaluator): masked action probabilities of ONE observation."""

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, obs: dict, mask: np.ndarray) -> np.ndarray:
        import torch as th
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        with th.no_grad():
            dist = self.model.policy.get_distribution(obs_t, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1))
        return dist.distribution.probs[0].cpu().numpy().astype(np.float64)

    def checkpoint_hash(self) -> str:
        return state_dict_hash(self.model.policy)


# ------------------------------------------------------------------------------------------------------------ plan samplers
def _env_plan(env: BreachEnv, choose_action) -> dict:
    """choose_action(obs, mask) -> action. One episode from the empty BUY phase; the only simulator call is the battle at the end."""
    obs, _ = env.reset(seed=0, options={"layout_index": 0})
    actions, done = [], False
    while not done:
        a = int(choose_action(obs, env.action_masks()))
        actions.append(a)
        obs, _, done, _, _ = env.step(a)
    army = (np.bincount([t for t, _ in env.army], minlength=N_UNIT_TYPES) + np.asarray(env.owned)).tolist()
    return _ep(env.result, env.bought, len(env.army), army=army, actions=actions)


def planner_plan(env: BreachEnv, probs_fn, rng) -> dict:
    return _env_plan(env, lambda obs, mask: sample_action(probs_fn(obs, mask), mask, rng))


def rps_plan(env: BreachEnv, rng) -> dict:
    def pick(obs, mask):
        legal = np.flatnonzero(mask)
        return legal[int(rng.integers(len(legal)))]
    return _env_plan(env, pick)


def rps_full_mask(env: BreachEnv, mask: np.ndarray) -> np.ndarray:
    """RPS-full masks (Gate changelog (r)); returns the original mask if the extra masking would leave no legal action."""
    from breachlens.env import BUY_BASE, END_BUY, WAIT
    m = np.asarray(mask, dtype=bool).copy()
    if m[BUY_BASE:BUY_BASE + N_UNIT_TYPES].any():
        m[END_BUY] = False                                 # still able to buy: keep buying
    if sum(env.owned) > 0:
        m[WAIT] = False                                    # bought units remain undeployed: keep deploying
    return m if m.any() else np.asarray(mask, dtype=bool)


def rps_full_plan(env: BreachEnv, rng) -> dict:
    def pick(obs, mask):
        legal = np.flatnonzero(rps_full_mask(env, mask))
        return legal[int(rng.integers(len(legal)))]
    return _env_plan(env, pick)


def ras_plan(layout: dict, Prep, rng) -> dict:
    types = sort_by_rank(random_types(rng))
    r = simulate(Prep, place_on_border(layout, types))
    return _ep(r, len(types), len(types), army=np.bincount(types, minlength=N_UNIT_TYPES).tolist(), actions=[int(t) for t in types])


# ------------------------------------------------------------------------------------------------------------ cache + policies
@dataclass
class PlanCache:
    """Plans sampled so far per (method, layout id, seed, policy hash). Plan k never depends on N, so extending the list never changes it."""
    plans: dict = field(default_factory=dict)
    n_sim_calls: int = 0

    def get(self, method: str, rec: dict, layout: dict, Prep, seed: int, probs_fn, n: int, ph: str) -> list:
        key = (method, rec["id"], int(seed), ph)
        have = self.plans.setdefault(key, [])
        if len(have) < n:
            env = None if method == "RAS" else BreachEnv(layouts=[layout], role="train")      # the evaluator loaded the layout through the pool loaders
            for k in range(len(have), n):
                rng = plan_rng(seed, rec["id"], method, k)
                t0 = time.perf_counter()
                if method == "planner":
                    p = planner_plan(env, probs_fn, rng)
                elif method == "RPS":
                    p = rps_plan(env, rng)
                elif method == "RPSfull":
                    p = rps_full_plan(env, rng)
                else:
                    p = ras_plan(layout, Prep, rng)
                p["seconds"] = time.perf_counter() - t0
                have.append(p)
                self.n_sim_calls += 1
        return have[:n]


class BestOfN(Policy):
    """Common body of the three methods: sample N plans, play the best. `method` is "planner", "RAS" or "RPS"."""
    method = ""

    def __init__(self, N: int, seed: int, *, role: str = "validation", unlock: bool = False, cache: PlanCache | None = None) -> None:
        if N < 1 or N > N_MAX:
            raise ValueError(f"N must be in 1..{N_MAX}")
        self.N, self.seed, self.role, self.unlock = int(N), int(seed), role, bool(unlock)
        self.cache = cache if cache is not None else PlanCache()
        self.name = f"{METHOD_LABEL[self.method]}-{self.N}"
        self.probs_fn, self.policy_hash = None, "none"

    def seeds(self, rec: dict) -> list:
        return [self.seed]

    def provenance(self) -> dict:
        return {"N": self.N, "method": METHOD_LABEL[self.method], "policy_hash": self.policy_hash, "plan_seed": self.seed}

    def _guard(self, rec: dict) -> None:
        if EV.is_final_layout(rec) and not (self.role == "eval" and self.unlock):
            raise PermissionError(f"{self.name}: {rec.get('id')} is a FINAL / transfer layout; needs role='eval' and unlock=True")

    def precheck(self, recs: list) -> None:
        for rec in recs:
            self._guard(rec)

    def episodes(self, rec: dict, layout: dict, Prep) -> list[dict]:
        self._guard(rec)
        plans = self.cache.get(self.method, rec, layout, Prep, self.seed, self.probs_fn, self.N, self.policy_hash)
        i = choose(plans)
        ep = {k: v for k, v in plans[i].items() if k != "seconds"}
        ep.update(chosen_index=i, n_sims=len(plans), plan_seconds=float(sum(p["seconds"] for p in plans)))
        return [ep]


class Planner(BestOfN):
    method = "planner"

    def __init__(self, probs_fn, N: int, seed: int, *, policy_hash: str | None = None, **kw) -> None:
        super().__init__(N, seed, **kw)
        self.probs_fn = probs_fn
        h = policy_hash if policy_hash is not None else (probs_fn.checkpoint_hash() if hasattr(probs_fn, "checkpoint_hash") else None)
        if not h:
            raise ValueError("a planner needs a policy checkpoint hash (pass policy_hash= for a stub policy)")
        self.policy_hash = h


class RAS(BestOfN):
    method = "RAS"


class RPS(BestOfN):
    method = "RPS"


class RPSFull(BestOfN):
    method = "RPSfull"


def make_cells(method: str, seed: int, Ns=N_VALUES, *, probs_fn=None, policy_hash: str | None = None, **kw) -> list:
    """One policy object per N, sharing ONE PlanCache, so the 128 plans are sampled once and N = 8 / 32 are prefixes of them."""
    cache = PlanCache()
    if method == "planner":
        return [Planner(probs_fn, n, seed, policy_hash=policy_hash, cache=cache, **kw) for n in Ns]
    cls = {"RAS": RAS, "RPS": RPS, "RPSfull": RPSFull}[method]
    return [cls(n, seed, cache=cache, **kw) for n in Ns]


# ------------------------------------------------------------------------------------------------------------ cost projection
def cost_cells(plain_128_learners: int = 10, d4_128_learners: int = 10, n_layouts: int = 120, learners: int = 10, layouts_128: int | None = None) -> dict:
    """Plans per cell group for the full B15 planner session (7.2 table). `layouts_128` = number of layouts that get N = 128 (fallback 3);
    the others keep the nested N = 32 prefix only (N = 8 and 32 always run everywhere). RAS-N / RPS-N copy the plain planner cell's
    learner count and layouts (7.2: same learners-count and layouts as the planner cell they are compared with); they have 10 seeds at N = 8 / 32."""
    l128 = n_layouts if layouts_128 is None else layouts_128
    plans = lambda n128: n128 * l128 * N_MAX + n128 * (n_layouts - l128) * 32 + (learners - n128) * n_layouts * 32
    return {"planner_plain": plans(plain_128_learners), "planner_d4": plans(d4_128_learners), "RAS": plans(plain_128_learners), "RPS": plans(plain_128_learners)}


def project_wall_hours(t: dict, cells: dict, workers: int = WORKERS) -> dict:
    """t = measured seconds per plan {"planner": s, "RAS": s, "RPS": s}; wall = CPU / workers (ideal scaling; contention is a separate factor)."""
    cpu = {"planner_plain": cells["planner_plain"] * t["planner"], "planner_d4": cells["planner_d4"] * t["planner"],
           "RAS": cells["RAS"] * t["RAS"], "RPS": cells["RPS"] * t["RPS"]}
    return {"cpu_hours": {k: v / 3600 for k, v in cpu.items()}, "wall_hours": {k: v / 3600 / workers for k, v in cpu.items()},
            "total_cpu_hours": sum(cpu.values()) / 3600, "total_wall_hours": sum(cpu.values()) / 3600 / workers}


FALLBACK_STEPS = ("full", "1: N=128 for 5 of 10 learners per PPO condition", "2: N=128 for PPO plain only", "3: N=128 on a 30-layout FINAL subset")


def select_fallback(t: dict, limit_h: float = FALLBACK_LIMIT_H, workers: int = WORKERS, n_layouts: int = 120) -> dict:
    """Pre-registered 7.2 selector. Options are applied IN ORDER and cumulatively until the projected wall at `workers` is <= limit_h:
    (1) N = 128 for the 5 lowest learner ids of each PPO condition (N = 8 / 32 keep all 10); (2) additionally N = 128 for PPO plain only
    (the D4 planner keeps N = 8, 32); (3) additionally N = 128 only on a 30-layout FINAL subset (15 familiar + 15 transfer).
    RAS-N / RPS-N follow the plain planner cell. Returns the first configuration that fits (or the last one with fits=False)."""
    configs = [dict(plain_128_learners=10, d4_128_learners=10, layouts_128=n_layouts),
               dict(plain_128_learners=5, d4_128_learners=5, layouts_128=n_layouts),
               dict(plain_128_learners=5, d4_128_learners=0, layouts_128=n_layouts),
               dict(plain_128_learners=5, d4_128_learners=0, layouts_128=30)]
    out = None
    for step, c in enumerate(configs):
        proj = project_wall_hours(t, cost_cells(n_layouts=n_layouts, **c), workers)
        out = {"step": step, "label": FALLBACK_STEPS[step], "config": c, "projection": proj, "fits": proj["total_wall_hours"] <= limit_h}
        if out["fits"]:
            break
    return out


def dumps(x) -> str:
    return json.dumps(x, sort_keys=True)
