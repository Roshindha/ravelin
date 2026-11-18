"""B18b: duel mode of the Streamlit app (docs/SPEC_V2.md, Gate changelog row (mm)). A DEMO, not a result.

All game logic, scoring, opponents and the cache come from ``breachlens.duel`` (row (kk)); this module only draws and routes clicks. All duel state lives in
``st.session_state`` (no module-level mutable state). Opponents are revealed only through ``Duel.reveal`` after the human's battle has resolved; the cache
file is not opened before that. Human runs go only to the gitignored ``results_v2/duel_human_runs.jsonl`` (``BREACHLENS_DUEL_RUNS`` overrides the path, for
tests); the app shows no leaderboard and no aggregate of human runs.
"""
from __future__ import annotations

import html
import importlib.util
import os
import sys
from pathlib import Path

import streamlit as st

from app import render as R
from breachlens import duel as D
from breachlens import evaluator as EV
from breachlens.actions import BORDER
from breachlens.armies import unit_cost
from breachlens.constants import BUDGET, GRID, N_BORDER, UNIT_TYPES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT = "spine_pods-300000056"                       # row (mm) (a)
DISCLAIMER = "DEMO, not a result: one layout, one human run; opponents were precomputed on DEV-validation layouts that were used for model selection."


@st.cache_resource
def _replay_module():
    if "make_replays" in sys.modules:
        return sys.modules["make_replays"]
    spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_replays"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner=False)
def _duel_ids() -> list:
    return EV.dev_layout_ids(D.DUEL_SPLIT, entry_point="app.duel_ui")      # DEV-validation only (30)


def _runs_path() -> Path:
    return Path(os.environ.get("BREACHLENS_DUEL_RUNS", str(D.HUMAN_RUNS_PATH)))


# ----------------------------------------------------------------------------------------------------------------------------- state
def new_duel(layout_id: str) -> None:
    """Fresh duel on `layout_id`; every duel_* key of the previous one is dropped."""
    for k in [k for k in st.session_state if k.startswith("duel_") and k != "duel_layout"]:
        del st.session_state[k]
    st.session_state["duel_msg"] = ""
    st.session_state["duel_game"] = D.Duel(layout_id, human_runs_path=_runs_path())


def _do(fn_name: str, *args) -> None:
    """Run one engine action; an illegal action puts the engine's message in session_state and changes nothing."""
    duel = st.session_state["duel_game"]
    try:
        getattr(duel, fn_name)(*args)
        st.session_state["duel_msg"] = ""
    except D.DuelError as e:
        st.session_state["duel_msg"] = str(e)


def _deploy() -> None:
    _do("deploy", st.session_state["duel_unit"], st.session_state["duel_tile"])


def _reset() -> None:
    new_duel(st.session_state["duel_layout"])


# ----------------------------------------------------------------------------------------------------------------------------- views
def _board(duel: D.Duel) -> str:
    """The layout as the agents see it (buildings with value, core, walls, defenders unlabelled, your units); free border tiles are numbered."""
    env, M = duel.env, _replay_module()
    structs = [[k, r, c, "" if k == "d" else ex] for k, r, c, ex in M._struct_table(env._st.P)]
    step = {"action": 0, "heat_total": 0.0, "heat_tile": [0.0] * N_BORDER, "heat": [[0.0] * N_BORDER] * 5, "heat_unit": [0.0] * 5,
            "army": [[int(t), int(b)] for t, b in env.army]}
    free = [b for b in range(N_BORDER) if env._st.border_free[b] and all(b != bb for _, bb in env.army)]
    svg, _ = R.grid_svg({"steps": [step], "structs": structs}, [list(b) for b in BORDER], GRID, 0, "total", tile_labels=free if duel.phase == "DEPLOY" else None)
    return svg


def _controls(duel: D.Duel) -> None:
    env = duel.env
    st.markdown(f'**Phase {duel.phase}** &middot; gold remaining <b>{env.gold}</b> of {BUDGET} &middot; bought {env.bought}', unsafe_allow_html=True)
    if duel.phase == "BUY":
        st.markdown('<div class="mut">Buy a unit (label: unit, cost in gold):</div>', unsafe_allow_html=True)
        cols = st.columns(len(UNIT_TYPES))
        for i, u in enumerate(UNIT_TYPES):
            cols[i].button(f"{u} {unit_cost(i)}g", help=f"Buy one {u}, cost {unit_cost(i)}", key=f"duel_buy_{u}", on_click=_do, args=("buy", u))
        c = st.columns(2)
        c[0].button("Undo last buy", key="duel_undo", on_click=_do, args=("undo",))
        c[1].button("END BUY", key="duel_end", type="primary", on_click=_do, args=("end_buy",))
        owned = ", ".join(f"{UNIT_TYPES[u]} x{n}" for u, n in enumerate(env.owned) if n) or "nothing yet"
        st.markdown(f'<div class="mut">Army so far: {owned}. Undo works only before END BUY.</div>', unsafe_allow_html=True)
    else:
        owned = ", ".join(f"{UNIT_TYPES[u]} x{n}" for u, n in enumerate(env.owned) if n) or "none left"
        st.markdown(f'<div class="mut">To deploy: {owned}. Pick a unit and a numbered border tile on the grid. '
                    f'A battle starts when the last unit is out, or on WAIT (units not yet deployed are forfeited).</div>', unsafe_allow_html=True)
        c = st.columns([1.2, 1.2, 1, 1])
        c[0].selectbox("Unit", UNIT_TYPES, key="duel_unit")
        c[1].number_input("Border tile", min_value=0, max_value=N_BORDER - 1, value=0, step=1, key="duel_tile")
        c[2].button("DEPLOY", key="duel_deploy", type="primary", on_click=_deploy)
        c[3].button("WAIT", key="duel_wait", on_click=_do, args=("wait",))
        st.markdown("".join(f'<div class="mut">{html.escape(s)}</div>' for s in duel.describe_legal()), unsafe_allow_html=True)
    if st.session_state.get("duel_msg"):
        st.error(st.session_state["duel_msg"])


def _results(duel: D.Duel) -> None:
    ss = st.session_state
    if "duel_rows" not in ss:                                    # the first (and only) moment the opponents are read: the battle has resolved
        ss["duel_rows"] = duel.reveal()
        M = _replay_module()
        ss["duel_battle"] = M._battle(duel.env)
        ss["duel_structs"] = M._struct_table(duel.env._st.P)
    rows, battle = ss["duel_rows"], ss["duel_battle"]
    me = rows[0]
    st.markdown(f'<div class="outcome"><b>Your battle:</b> score <b>{me["score"]:.4f}</b> &middot; D = {me["d"]:.4f} &middot; core breached: '
                f'<b>{"YES" if me["breach"] else "NO"}</b> &middot; {me["n_deployed"]} units deployed</div>', unsafe_allow_html=True)
    frames = battle["frames"]
    ss.setdefault("duel_pos", 0)
    ep = {"steps": [], "structs": ss["duel_structs"], "battle": battle}
    left, right = st.columns([1.15, 1])
    with left:
        svg, _ = R.grid_svg(ep, [list(b) for b in BORDER], GRID, ss["duel_pos"], "total")
        st.image(svg, width=640)
        st.slider("Battle timeline (tick)", 0, len(frames) - 1, key="duel_pos")
    with right:
        f = frames[ss["duel_pos"]]
        st.markdown(f'**Your battle, tick {f["t"]}:** {sum(1 for u in f["u"] if u[1] > 0)} of {len(f["u"])} units alive.')
        st.markdown('<div class="mut">Same replay view as the Replay mode: white ring = unit attacked this tick.</div>', unsafe_allow_html=True)
    st.subheader("Comparison (same layout, same reset seed)")
    body = "".join(
        f'<tr class="{"chosen" if r["contestant"] == "human" else ""}"><td>{html.escape(r["label"])}</td><td>{r["score"]:.3f}</td><td>{r["d"]:.3f}</td>'
        f'<td>{"yes" if r["breach"] else "no"}</td><td>{r["n_bought"]}</td><td>{r["n_deployed"]}</td>'
        f'<td>{html.escape(" ".join(f"{u[:3]}={n}" for u, n in r["bought"].items() if n))}</td>'
        f'<td>{html.escape(", ".join(f"{u}@{b}" for u, b in r["deploy"]) or "none")}</td></tr>' for r in rows)
    st.markdown('<table class="top"><tr><th>contestant</th><th>score</th><th>D</th><th>breach</th><th>bought</th><th>deployed</th><th>army (bought per type)</th>'
                f'<th>deploy tiles</th></tr>{body}</table>', unsafe_allow_html=True)
    st.markdown('<div class="mut">Oracle = search baseline, not the optimum.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mut">{html.escape(D.DEMO_NOTE)}</div>', unsafe_allow_html=True)


def run() -> None:
    ss = st.session_state
    ids = _duel_ids()
    c = st.columns([2.4, 1.2, 3])
    c[0].selectbox("Duel layout (DEV-validation)", ids, index=ids.index(DEFAULT_LAYOUT), key="duel_layout")
    if "duel_game" not in ss or ss["duel_game"].layout_id != ss["duel_layout"]:
        new_duel(ss["duel_layout"])
    c[1].button("New duel", key="duel_new", on_click=_reset)
    st.markdown(f'<div class="badge warn" style="display:inline-block">{DISCLAIMER}</div>', unsafe_allow_html=True)
    duel = ss["duel_game"]
    st.markdown(f'<div class="bar"><span class="badge">DEV-validation: held out from training</span><span class="badge">family <b>{duel.family}</b></span>'
                f'<span class="badge">layout <b>{duel.layout_id}</b></span></div>', unsafe_allow_html=True)
    if duel.finished:
        _results(duel)
        return
    left, right = st.columns([1.15, 1])
    with left:
        st.image(_board(duel), width=640)
        st.markdown('<div class="mut">Blue squares: buildings (number = value). Diamond C: core. Grey: walls. Red circles: defenders. Your units are coloured circles. '
                    'Opponents and their results are hidden until your battle is over.</div>', unsafe_allow_html=True)
    with right:
        _controls(duel)
