# Base Siege: Reinforcement Learning for Army Selection and Deployment

An RL agent buys troops under a 200-resource budget and deploys them against
unfamiliar base layouts in a custom 16x16 Gymnasium combat environment.
See PROGRESS.md for current status and exact commands.

## Quick start
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/pytest
    .venv/bin/python scripts_smoke.py
