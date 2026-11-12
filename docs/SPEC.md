# Base Siege: Reinforcement Learning for Army Selection and Deployment

An RL agent buys troops under a resource budget and deploys them against unfamiliar base layouts in a custom 16x16 combat environment. Decisions are made inspectable through an interactive app.

## Full spec
- Gymnasium env: 16x16 grid, 4 troop types, 200-resource budget, deterministic tick-based combat (the simulator handles movement and targeting; the agent only chooses purchases, deployments and waiting).
- Phases: BUY -> DEPLOY -> BATTLE. Single Discrete(246) action space: 4 buy actions + 4 troop types x 60 border deploy tiles (240) + END_BUY + WAIT = 246. action_masks() forbids unaffordable buys, deploying troops not owned/left, illegal tiles, and wrong-phase actions.
- Observation: Dict with "grid" (8 spatial channels, partially observed battle) and "state" (9 values). Every channel/value documented.
- Reward: destruction gain, full-clear bonus, small penalties. Documented.
- Layout pools: 241 accepted layouts. DEV pool (training/tuning) and FINAL pool frozen and NEVER used for training or tuning. One generator family held out as "transfer"; the others are "familiar".
- Algorithms (final deliverable): MaskablePPO (CNN grid features + state-vector MLP, deployment-first curriculum), masked Double DQN (own PyTorch implementation), Maskable TRPO (sb3-contrib). Conditions: PPO plain, PPO with D4 augmentation, DDQN, TRPO.
- D4 augmentation: 4 rotations x optional reflection applied consistently to observations, masks and deployment coordinates; actions mapped back to original simulator coordinates before execution.
- Evaluation: ten independent learners (seeds) per learned condition over all 241 layouts; mean destruction %, full-clear rate, familiar vs transfer; bootstrap over learners; CSV/JSON outputs with model identity; a separate probe comparing deterministic vs sampled action selection on the same checkpoints.
- App: Streamlit with manual play, base editor, generated/custom bases, model selection (up to 51 final checkpoints), policy replay, JSON and standalone HTML export, CPU inference with bounded model caches.
- Extra feature (built later): "Why this move?" explainability. Replay shows a deployment-probability heatmap, top-5 action probabilities and the value estimate at every step. Keep policy code structured so action distributions and value estimates are easy to extract.

## Tech
Python 3.11, PyTorch, Gymnasium, stable-baselines3, sb3-contrib, Streamlit, numpy, pandas, matplotlib. CPU only.

## Rules
1. Maintain PROGRESS.md after every step: DONE, decisions (with reasons), exact commands, known issues, NEXT step.
2. Every core module gets pytest tests.
3. After each step report in at most 5 lines: built / how to run / test status / next.
4. Fix seeds; results must be reproducible.
5. Work one step at a time. Do not start a later step until told.

## Step order
1 scaffold, 2 simulator, 3 layouts and frozen pools, 4 env interface/masks/baselines, 5 D4 augmentation, 6 PPO pipeline, 7 launch PPO grid, then DDQN, TRPO, evaluation, app, explainability, README.
