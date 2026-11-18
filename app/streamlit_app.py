"""B18a: Streamlit app shell + replay / decision-lens view (docs/SPEC_V2.md, Gate changelog row (ll)). A DEMO. Mode "Duel" (B18b, row (mm)) lives in app/duel_ui.py.

    .venv/bin/streamlit run app/streamlit_app.py

Layouts: DEV-validation (default) or DEV-train (always badged IN-SAMPLE), only through ``breachlens.evaluator``. Models: the three seed-0 arms, plain or TTA,
only through ``rl.explain.load_display_model``. Episodes and explanations come from ``scripts/make_replays.run_episode`` (the B16b replay code, which calls
``rl.explain.explain`` at every decision step and asserts its argmax equals the B15 eval-path action). Nothing is trained or evaluated here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import render as R  # noqa: E402
from rl import explain as X  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------------- fixed by row (ll)
SPLITS = ("DEV-validation", "DEV-train")                      # nothing else is selectable
ARMS = tuple(X.DISPLAY_MODELS)                                # ppo_plain, ppo_d4, ddqn: seed 0 only
VIEWS = ("TTA", "plain")
DEFAULT = {"split": "DEV-validation", "layout": "spine_pods-300000056", "arm": "ppo_plain", "view": "TTA", "pos": 0, "heat": "Chosen unit type"}
HEATS = ("Chosen unit type", "Per-tile total")
MODES = ("Replay", "Duel")
DEMO = "DEMO: a read-out of frozen checkpoints on DEV layouts. Nothing here is a result."

st.set_page_config(page_title="Breach Lens (demo)", page_icon="▦", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
html, body, [class*="css"], .stApp { font-family: ui-monospace, Menlo, Consolas, monospace; }
.stApp { background-color: #07142b; background-image:
  linear-gradient(rgba(94,200,255,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(94,200,255,.05) 1px, transparent 1px); background-size: 28px 28px; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: .04em; font-weight: 500; }
.bar { display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 10px 0; }
.badge { border:1px solid #3f7fb8; color:#9fd4ff; padding:2px 8px; font-size:12px; border-radius:2px; }
.badge.warn { border-color:#ffb454; color:#ffb454; font-weight:bold; }
.banner { border:1px dashed #3f7fb8; color:#9fd4ff; padding:4px 10px; font-size:12px; margin-bottom:8px; }
.panel { border:1px solid #2d5a8c; background:rgba(13,34,68,.7); padding:10px 14px; margin-bottom:10px; }
.mut { color:#7fa6cf; font-size:12px; }
.outcome { border-left:3px solid #5ec8ff; padding:4px 10px; margin:6px 0; background:rgba(13,34,68,.7); }
.stButton button { padding:.25rem .35rem; } .stButton button p { font-size:12px; white-space:nowrap; overflow:visible; text-overflow:clip; }
table.top { width:100%; border-collapse:collapse; font-size:13px; }
table.top td, table.top th { border-bottom:1px solid #1d3d6b; padding:3px 6px; text-align:left; }
table.top tr.chosen td { color:#ffe27a; }
</style>
"""


# ----------------------------------------------------------------------------------------------------------------------------- cached data
@st.cache_resource(max_entries=1, show_spinner="loading checkpoint")
def get_model(arm: str):
    """ONE loaded checkpoint at a time (a new arm evicts the old one)."""
    return X.load_display_model(arm)


@st.cache_resource
def replay_module():
    spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_replays"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner=False)
def layout_ids(split: str) -> list:
    from breachlens import evaluator as EV
    return EV.dev_layout_ids(split, entry_point="app.streamlit_app")


@st.cache_data(show_spinner="playing the episode and explaining every decision step")
def episode(split: str, layout_id: str, arm: str, tta: bool) -> dict:
    """One episode's explanations, computed once per (layout, arm, view): the B16b replay episode (plain data)."""
    M = replay_module()
    rec, layout = X.load_dev_layout(split, layout_id=layout_id)
    model, algo, run = get_model(arm)
    ep = M.run_episode(model, algo, arm, run, rec, layout, tta)
    from breachlens.actions import BORDER
    ep.update(arm=arm, run=run, algo=algo, tta=tta, layout_id=rec["id"], family=rec["family"], split=split, border=[list(b) for b in BORDER], grid=M.GRID)
    return ep


@st.cache_data
def results_headline() -> str:
    """The 'Plain-language summary' of docs/RESULTS.md, verbatim (read from the file, never retyped)."""
    txt = (ROOT / "docs" / "RESULTS.md").read_text()
    a = txt.index("## Plain-language summary")
    b = txt.index("\n## ", a + 5)
    return txt[a:b].strip()


# ----------------------------------------------------------------------------------------------------------------------------- state
def init_state():
    for k in ("split", "arm", "view", "heat"):
        st.session_state.setdefault(k, DEFAULT[k])


def _step(delta: int, key: str, last: int):
    st.session_state[key] = max(0, min(last, st.session_state[key] + delta))


def fmt(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


# ----------------------------------------------------------------------------------------------------------------------------- views
def command_bar():
    c = st.columns([1.1, 2.2, 1.1, 1.0, 0.1])
    split = c[0].selectbox("Split", SPLITS, key="split")
    ids = layout_ids(split)
    default = ids.index(DEFAULT["layout"]) if split == DEFAULT["split"] and DEFAULT["layout"] in ids else 0
    layout = c[1].selectbox("Layout", ids, index=default, key=f"layout_{split}")
    arm = c[2].selectbox("Arm (seed 0)", ARMS, key="arm", format_func=lambda a: X.DISPLAY_MODELS[a][0])
    view = c[3].radio("View", VIEWS, key="view", horizontal=True)
    return split, layout, arm, view == "TTA"


def badges(ep: dict):
    run = ep["run"]
    b = [f'<span class="badge">model <b>{run}</b></span>', f'<span class="badge">arm <b>{ep["arm"]}</b></span>',
         f'<span class="badge">view <b>{"TTA (8 views)" if ep["tta"] else "plain"}</b></span>',
         f'<span class="badge">family <b>{ep["family"]}</b></span>']
    if ep["split"] == "DEV-train":
        b.insert(0, '<span class="badge warn">IN-SAMPLE: the learner trained on this layout</span>')
    else:
        b.insert(0, '<span class="badge">DEV-validation: held out from training</span>')
    st.markdown(f'<div class="bar">{"".join(b)}</div>', unsafe_allow_html=True)


def outcome_strip(ep: dict):
    o = ep["outcome"]
    st.markdown(f'<div class="outcome"><b>True outcome of this episode:</b> score <b>{o["score"]:.4f}</b> &middot; D = {o["d"]:.4f} '
                f'({o["d_num"]}/{o["d_den"]} building value destroyed) &middot; core breached: <b>{"YES" if o["core"] else "NO"}</b> '
                f'&middot; {o["n_units"]} units deployed &middot; battle {o["ticks"]} ticks ({o["reason"]})</div>', unsafe_allow_html=True)


def decision_lens(ep: dict, pos: int):
    n_dec = len(ep["steps"])
    lens = {"pos": pos, "n_decision_steps": n_dec, "final_outcome": ep["outcome"]}
    if pos < n_dec:
        s = ep["steps"][pos]
        lens.update(kind="decision", chosen_action=s["action"], chosen_label=s["label"], top=[t["a"] for t in s["top"]], value=s["value"])
        st.subheader("Decision lens")
        st.markdown(f'<div class="mut">decision step {pos} (= {pos} actions already taken) of 0..{n_dec - 1} &middot; {s["phase"]} phase &middot; '
                    f'gold {s["gold"]} &middot; bought {s["bought"]} &middot; legal actions {s["n_legal"]}</div>', unsafe_allow_html=True)
        pc = s["p_chosen"]
        st.markdown(f'**Chosen action:** {s["label"]} (probability {fmt(pc)}{"; softmax of Q: not a policy" if ep["algo"] == "ddqn" else ""})')
        ddqn = ep["algo"] == "ddqn"
        head = ("Top 5 actions by Q. The softmax column is NOT a policy (fixed display temperature %s)." % X.DDQN_TEMPERATURE if ddqn else
                "Top 5 actions by probability" + (" (TTA: softmax of the 8-view mean log-probability)." if ep["tta"] else "."))
        rows = "".join(f'<tr class="{"chosen" if t["a"] == s["action"] else ""}"><td>{i}</td><td>{t["label"]}{" &#9664; chosen" if t["a"] == s["action"] else ""}</td>'
                       + (f'<td>{fmt(t["q"])}</td><td>{fmt(t["p"])}</td>' if ddqn else f'<td>{fmt(t["p"])}</td>') + "</tr>" for i, t in enumerate(s["top"], 1))
        th = "<th>#</th><th>action</th>" + ("<th>Q</th><th>softmax</th>" if ddqn else "<th>prob.</th>")
        st.markdown(f'<div class="mut">{head}</div><table class="top"><tr>{th}</tr>{rows}</table>', unsafe_allow_html=True)
        st.markdown(f'**Value estimate:** {fmt(s["value"])}')
        st.markdown(f'<div class="mut">{X.VALUE_LABEL}; {"critic V(s)" if ep["algo"] == "ppo" else "max over legal actions of Q"}'
                    f'{" (TTA: mean over 8 views)" if ep["tta"] else ""}. Not comparable with the final score.</div>', unsafe_allow_html=True)
        st.markdown(f'**Outcome so far:** battle not fought yet; {len(s["army"])} unit(s) placed. The episode\'s true outcome is shown above the grid.')
    else:
        f = ep["battle"]["frames"][pos - n_dec]
        structs = ep["structs"]
        alive = sum(1 for u in f["u"] if u[1] > 0)
        nb = sum(1 for t in structs if t[0] == "b")
        dead = sum(1 for i, t in enumerate(structs) if t[0] == "b" and f["s"][i] <= 0)
        core = next(i for i, t in enumerate(structs) if t[0] == "c")
        lens.update(kind="battle", tick=f["t"], units_alive=alive, buildings_destroyed=dead, core_destroyed=f["s"][core] <= 0)
        st.subheader("Decision lens: battle")
        st.markdown(f'**Outcome so far (tick {f["t"]} of {ep["outcome"]["ticks"]}):** {alive} of {len(f["u"])} units alive; {dead} of {nb} buildings destroyed; '
                    f'core {"destroyed" if f["s"][core] <= 0 else "standing"}. White ring = unit attacked this tick.')
        if f["t"] == ep["outcome"]["ticks"]:
            o = ep["outcome"]
            st.markdown(f'**Final (true) outcome:** score {o["score"]:.4f}, D = {o["d"]:.4f}, core breached: **{"YES" if o["core"] else "NO"}**.')
        st.markdown('<div class="mut">The decisions were all made before the battle; the lens for them is under decision steps on the scrubber.</div>', unsafe_allow_html=True)
    st.session_state["_lens"] = lens


def spark(ep: dict, pos: int):
    import altair as alt
    import pandas as pd
    n_dec = len(ep["steps"])
    df = pd.DataFrame({"step": range(n_dec), "value": [s["value"] for s in ep["steps"]]})
    cur = pd.DataFrame({"step": [min(pos, n_dec - 1)], "value": [df["value"][min(pos, n_dec - 1)]]})
    line = alt.Chart(df).mark_line(color="#ffb454").encode(x=alt.X("step:Q", title="decision step"), y=alt.Y("value:Q", title="value (shaped return)"))
    pt = alt.Chart(cur).mark_point(color="#ffffff", size=70, filled=True).encode(x="step:Q", y="value:Q")
    st.altair_chart((line + pt).properties(height=110).configure(background="#07142b").configure_axis(labelColor="#7fa6cf", titleColor="#7fa6cf", gridColor="#16335e"),
                    width="stretch")


def headline_section():
    with st.expander("Headline of docs/RESULTS.md (quoted verbatim; not produced by this app)"):
        st.markdown(results_headline())
        st.markdown('<div class="mut">Source: docs/RESULTS.md. This demo shows no aggregate numbers of its own.</div>', unsafe_allow_html=True)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    top = st.columns([1.3, 5.7])
    mode = top[0].radio("Mode", MODES, key="mode", horizontal=True)       # the Duel switch lives in the top command bar
    top[1].markdown(f'<div class="banner">{DEMO}</div>', unsafe_allow_html=True)
    if mode == "Duel":
        from app import duel_ui
        duel_ui.run()
        return
    replay_main()


def replay_main():
    init_state()
    split, layout, arm, tta = command_bar()
    ep = episode(split, layout, arm, tta)
    n_dec, n_fr = len(ep["steps"]), len(ep["battle"]["frames"])
    total = n_dec + n_fr
    key = f"pos|{split}|{layout}|{arm}|{tta}"
    try:                                                      # optional ?pos=N deep link into the timeline (used for the screenshots)
        start = max(0, min(total - 1, int(st.query_params.get("pos", DEFAULT["pos"]))))
    except (TypeError, ValueError):
        start = DEFAULT["pos"]
    st.session_state.setdefault(key, start)
    badges(ep)
    outcome_strip(ep)
    left, right = st.columns([1.15, 1])
    with left:
        st.radio("Heatmap overlay", HEATS, key="heat", horizontal=True)
        svg, cap = R.grid_svg(ep, ep["border"], ep["grid"], st.session_state[key], "total" if st.session_state["heat"] == HEATS[1] else "chosen")
        st.image(svg, width=640)                                   # SVG string, rendered as an image (st.html would strip the SVG)
        st.markdown(f'<div class="mut">{cap}</div>' if cap else "", unsafe_allow_html=True)
        st.markdown('<div class="mut">Blue squares: buildings (number = value). Diamond C: core. Grey: walls. Red circles: defenders. Coloured circles: units '
                    '(Sc scout, Sn sniper, Bu bulwark, Bl blaster, Me mender). Heat = deploy probability on border tiles. White frame: chosen deploy tile.</div>',
                    unsafe_allow_html=True)
    with right:
        decision_lens(ep, st.session_state[key])
        if ep["algo"] == "ddqn":
            st.markdown(f'<div class="mut">DDQN: {X.DDQN_NOT_POLICY} (T = {X.DDQN_TEMPERATURE}).</div>', unsafe_allow_html=True)
    b = st.columns([0.8, 0.8, 6])
    b[0].button("◀ Prev", on_click=_step, args=(-1, key, total - 1), key="prev")
    b[1].button("Next ▶", on_click=_step, args=(1, key, total - 1), key="next")
    p = st.session_state[key]
    where = f"decision step {p} of 0..{n_dec - 1}" if p < n_dec else f"battle tick {ep['battle']['frames'][p - n_dec]['t']} of {ep['outcome']['ticks']}"
    b[2].slider("Timeline", 0, total - 1, key=key)
    b[2].markdown(f'<div class="mut">decision steps 0..{n_dec - 1}, then battle ticks 0..{ep["outcome"]["ticks"]}; now: {where}</div>', unsafe_allow_html=True)
    with st.expander("Value over the decision steps (shaped-return estimate)"):
        spark(ep, p)
    headline_section()
    st.markdown(f'<div class="mut">{DEMO} Provenance: model {ep["run"]}, layout {ep["layout_id"]} ({ep["split"]}), episode = the learner\'s own deterministic B15 episode.</div>',
                unsafe_allow_html=True)


main()
