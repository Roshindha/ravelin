"""Ravelin, the game app (B18e-b): "printed board-game box" art direction (docs/SPEC_V2.md, Gate changelog rows (nn) and (oo)).

    .venv/bin/streamlit run app/streamlit_app.py

Modes: "Play" (the duel, B17 engine) and "Watch the AI" (the B16b/B18a replay + decision lens). Everything about the game (rules, scoring, TTA, Oracle, the duel
cache and every outcome) comes unchanged from ``breachlens.duel`` and ``scripts/make_replays`` (``run_episode`` / ``_battle`` / ``_struct_table``); this file only
draws and routes input. Opponents are revealed only through ``Duel.reveal`` and only after the human's battle has resolved AND the player pressed "Show report"
(the cache file is not opened before that). Human runs go only to the gitignored ``results_v2/duel_human_runs.jsonl`` (``BREACHLENS_DUEL_RUNS`` overrides it).
Deployed (``RAVELIN_DEPLOY=1``, or torch not installed): "Watch the AI" reads ``data/app_bundle`` (``app/bundle.py``, DEV-validation only), the duel uses the simulator and
``data/v2/duel_cache.json`` only, and visitors' plays are NOT logged (no file is written).
Layouts: DEV-validation (Play, and the default of Watch) or DEV-train (Watch only, always badged IN-SAMPLE), only through ``breachlens.evaluator``. Models: the three
seed-0 arms, plain or TTA, only through ``rl.explain.load_display_model``. Nothing is trained or evaluated here.

Look: a flat azure FIELD with snow halftone dots. Only the title block (Fraunces >= 28 px, snow) sits directly on the field; every other element is inside a snow
CARD (keyed container ``card_*`` or a ``.card`` div; ink border) or is a flat STICKER (solid snow / coral, ink outline, uppercase only on stickers). All colours
come from app/theme.py; no shadow, glow, blur or gradient (the heatmap scale is the only gradient of the project and does not appear on this page).

Click-to-deploy: in the DEPLOY phase the board is the custom component ``app/board_component.py`` (``st.components.v2``, no build step). A click / tap on a border tile
returns the border index b; ``_on_board_click`` sends ``Duel.deploy(unit, b)`` (action 5 + 52*u + b), exactly like the Deploy button of the compact tile dropdown,
which stays as the keyboard / accessibility fallback. The battle player is a self-contained page in ``st.iframe``.

Screenshot helper (query params, used only to produce the review screenshots; each one drives the REAL engine through a fixed human script):
``?demo=buy|deploy|battle|report|held`` and, for ``battle``, ``&tick=N`` (player starts paused at tick N); ``?mode=watch`` and ``&pos=N`` for Watch the AI.
"""
from __future__ import annotations

import html
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import board_component as BC  # noqa: E402
from app import bundle as BU  # noqa: E402
from app import game_view as GV  # noqa: E402
from app import icons as IC  # noqa: E402
from app import render as R  # noqa: E402
from app import theme as TH  # noqa: E402
from breachlens import duel as D  # noqa: E402
from breachlens import evaluator as EV  # noqa: E402
from breachlens.actions import BORDER, encode_buy, encode_deploy  # noqa: E402
from breachlens.armies import unit_cost  # noqa: E402
from breachlens.config import CFG  # noqa: E402
from breachlens.constants import END_BUY, N_BORDER, UNIT_TYPES  # noqa: E402

DEFAULT_LAYOUT = "spine_pods-300000056"
MODES = ("Play", "Watch the AI")
PHASES = ("Buy", "Deploy", "Battle", "Report")
SPLITS = ("DEV-validation", "DEV-train")                      # Watch the AI: nothing else is selectable
VIEWS = ("TTA", "plain")
HEATS = ("Chosen unit type", "Per-tile total")
TAGLINE = "Spend your gold. Pick your tiles. Breach the keep."
SHORT = {"scout": "Fast, cheap", "sniper": "Long range", "bulwark": "Tough wall-breaker", "blaster": "Splash burst", "mender": "Heals allies"}
ROLE = {"scout": "Cheap and quick. Rushes the buildings.", "sniper": "Long range. Full damage on turrets, 40% on the rest.",
        "bulwark": "Slow and sturdy. Chews through walls first.", "blaster": "Heavy burst that splashes nearby tiles.", "mender": "Heals friends nearby. Cannot attack."}
CONTESTANT = {"human": ("You", "", "you"), "ppo_plain_tta": ("PPO plain + TTA", "learning agent, seed 0", "learner"), "ppo_d4_tta": ("PPO+D4 + TTA", "learning agent, seed 0", "learner"),
              "ddqn_tta": ("DDQN + TTA", "learning agent, seed 0", "learner"), "best_single": ("BestSingle", "scripted Sniper army", "script"),
              "oracle": ("Oracle", "search baseline, not the optimum", "search")}
PLAYER_H = "content"                                   # st.iframe measures the player page, so the scaled board leaves no gap on narrow screens
MM_BUYS = ["sniper"] * 4 + ["bulwark"] * 2             # the row (mm) (g) human script: buy sniper x4, bulwark x2, END BUY, deploy in the order bought on the lowest legal tiles (0-5)

st.set_page_config(page_title="Ravelin", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

PROTO_CSS = """<style>
.stApp{background-color:var(--field);background-image:var(--dots);background-size:var(--dot-grid) var(--dot-grid)}
[data-testid='stHeader']{background:transparent}.block-container{padding-top:1.4rem;max-width:1320px}
.titleblock{margin:4px 0 12px}.titleblock h1{font-family:var(--f-head);font-size:__T__px;line-height:1.02;color:var(--field-ink);border:0;margin:0;padding:0}
.titleblock .tagline{font-family:var(--f-head);font-style:italic;font-weight:600;font-size:__G__px;line-height:1.2;color:var(--field-ink);margin-top:4px}
.card,[class*='st-key-card_'],[data-testid='stExpander']{background:var(--bg);color:var(--fg);border:var(--card-bw) solid var(--fg);border-radius:var(--card-r);padding:12px 16px;margin-bottom:14px}
[data-testid='stExpander']{padding:0}[data-testid='stExpander'] details{border:0}[data-testid='stExpander'] summary p{font-weight:600}
.card.slim{padding:4px 14px;font-size:15px;margin-left:-8px;max-width:1000px}
.narrow,[class*='st-key-card_board'],[class*='st-key-card_hud']{max-width:740px;margin-left:auto;margin-right:auto}
[class*='st-key-grid_'] [class*='st-key-card_board']{max-width:none;margin-left:0}
[class*='st-key-card_board'] [data-testid='stImage']{display:flex;justify-content:center}[class*='st-key-card_board'] img{display:block;margin:0 auto}
.sticker{display:inline-block;background:var(--bg);color:var(--fg);border:var(--sticker-bw) solid var(--fg);border-radius:6px;padding:0 10px;font-family:var(--f-body);font-weight:600;
 font-size:14px;line-height:1.55;text-transform:uppercase;letter-spacing:.05em;transform:rotate(var(--rot,0deg));animation:pop .2s ease-out both;white-space:nowrap}
.sticker.hot{background:var(--danger)}
@keyframes pop{0%{transform:rotate(var(--rot,0deg)) scale(.9);opacity:0}100%{transform:rotate(var(--rot,0deg)) scale(1);opacity:1}}
.hud{display:flex;align-items:center;justify-content:center;gap:20px;flex-wrap:wrap;margin-bottom:8px}
.stat{display:inline-flex;align-items:center;gap:8px;font-size:15px;color:var(--mut)}.stat b{color:var(--fg);font-size:26px;font-family:var(--f-num);font-weight:600}
.phases{display:inline-flex;align-items:center;gap:8px;flex-wrap:wrap}.flags{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.piece{position:relative;background:var(--bg);border:var(--card-bw) solid var(--fg);border-radius:var(--card-r);padding:12px 8px 10px;text-align:center;transition:transform .15s ease}
.piece:hover{transform:translateY(-2px)}.piece.off{opacity:.5}.piece .nm{font-family:var(--f-head);font-weight:700;font-size:21px;margin-top:2px}
.piece .short{color:var(--mut);font-size:14px;margin:2px 0 8px}.piece .ic{margin:0 auto;display:block}
.piece .cnt{position:absolute;top:8px;right:8px;background:var(--fg);color:var(--bg);border-radius:999px;font-family:var(--f-num);font-size:14px;font-weight:600;min-width:26px;line-height:24px}
.price{display:inline-flex;align-items:center;gap:5px;font-family:var(--f-num);font-weight:600;font-size:16px;padding:0 10px;text-transform:none;letter-spacing:0}
.tray{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin:2px 0 8px}.tray .tok{display:inline-flex;align-items:center;gap:6px;border:2px solid var(--fg);border-radius:var(--r);
 padding:2px 10px;font-family:var(--f-num);font-weight:600}
.ccard{background:var(--bg);border:var(--card-bw) solid var(--fg);border-radius:var(--card-r);padding:10px 12px;margin-bottom:12px;transition:transform .15s ease}.ccard:hover{transform:translateY(-2px)}
.chead{display:flex;align-items:center;gap:10px}.chead .grow{flex:1 1 auto;line-height:1.25}.chead b{font-size:17px}.sub{color:var(--mut);font-size:13px}
.sbar{height:12px;background:var(--line);border:2px solid var(--fg);border-radius:6px;margin:8px 0 4px;overflow:hidden}.sbar i{display:block;height:100%;background:var(--fg);transition:width .25s ease}
.ccard.me .sbar i{background:var(--primary)}.nums{font-size:14px;color:var(--mut)}.nums b{color:var(--fg);font-family:var(--f-num);font-size:18px}
.army{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;min-height:22px}.army span{display:inline-flex;align-items:center;gap:3px;font-family:var(--f-num);font-size:13px}
.verdict{font-family:var(--f-head);font-weight:700;font-size:24px}.note{color:var(--mut);font-size:14px}
.stButton button{border:2px solid var(--fg);background:var(--bg);color:var(--fg);font-weight:600}.stButton button p{font-size:16px}
.stButton button[kind='primary']{background:var(--primary);color:var(--on-primary);border:2px solid var(--fg)}.stButton button[kind='primary'] p{color:var(--on-primary);font-size:17px;font-weight:600}
[class*='st-key-g_buy_'] button{min-height:48px}[class*='st-key-g_buy_'] button[kind='primary'] p{font-size:30px;line-height:1}[class*='st-key-card_']{padding-bottom:18px}
[class*='st-key-g_buy_'] button:disabled{opacity:.45}
.stButton button:hover{border-color:var(--fg)}
.stButton button,[data-testid='stPopover'] button{min-height:44px}[data-testid='stSelectbox'] [role='group'],[data-testid='stSelectbox'] button,[data-testid='stRadio'] label{min-height:44px;align-items:center}
@media (max-width:640px){.block-container{padding-left:12px;padding-right:12px}.titleblock h1{font-size:46px}.titleblock .tagline{font-size:22px}
 .card,[class*='st-key-card_']{padding:10px 12px 18px}.stat b{font-size:22px}.sticker{font-size:13px}
 [class*='st-key-card_shop'] [data-testid='stHorizontalBlock']{flex-wrap:wrap;gap:8px}
 [class*='st-key-card_shop'] [data-testid='stColumn']{flex:1 1 calc(50% - 8px)!important;min-width:calc(50% - 8px)!important}
 .piece .nm{font-size:19px}[class*='st-key-g_buy_'] button{min-height:48px}}
table.top{border:2px solid var(--fg)}
@media (prefers-reduced-motion:reduce){.sticker,.piece,.ccard,.sbar i{animation:none!important;transition:none!important}.piece:hover,.ccard:hover{transform:none}}
</style>""".replace("__T__", str(TH.SIZE_TITLE)).replace("__G__", str(TH.SIZE_TAGLINE))


# ----------------------------------------------------------------------------------------------------------------------------- cached data
@st.cache_resource
def replay_module():
    if "make_replays" in sys.modules:
        return sys.modules["make_replays"]
    spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_replays"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner=False)
def duel_ids() -> list:
    return EV.dev_layout_ids(D.DUEL_SPLIT, entry_point="app.streamlit_app")


@st.cache_data(show_spinner=False)
def layout_ids(split: str) -> list:
    if BU.deploy_mode():
        return BU.layout_ids() if split == BU.BUNDLE_SPLIT else []
    return EV.dev_layout_ids(split, entry_point="app.streamlit_app")


@st.cache_resource(max_entries=1, show_spinner="loading checkpoint")
def get_model(arm: str):
    from rl import explain as X
    return X.load_display_model(arm)


@st.cache_data(show_spinner="playing the episode and explaining every decision step")
def episode(split: str, layout_id: str, arm: str, tta: bool) -> dict:
    if BU.deploy_mode():                                                   # deployed: the precomputed bundle (no torch), same data as the live path below
        if split != BU.BUNDLE_SPLIT:
            raise PermissionError(f"the deployed app has {BU.BUNDLE_SPLIT} episodes only; {split!r} is not bundled")
        return BU.load_episode(layout_id, arm, tta)
    from rl import explain as X
    M = replay_module()
    rec, layout = X.load_dev_layout(split, layout_id=layout_id)
    model, algo, run = get_model(arm)
    ep = M.run_episode(model, algo, arm, run, rec, layout, tta)
    ep.update(arm=arm, run=run, algo=algo, tta=tta, layout_id=rec["id"], family=rec["family"], split=split)
    return ep


@st.cache_resource
def board_component():
    """The interactive board, registered once per process (tests clear this cache to swap in a fake)."""
    return BC.make_component()


@st.cache_data
def results_headline() -> str:
    txt = (ROOT / "docs" / "RESULTS.md").read_text()
    a = txt.index("## Plain-language summary")
    return txt[a:txt.index("\n## ", a + 5)].strip()


def _runs_path() -> Path | None:
    """Where human runs are logged; None (nothing is written, nothing is opened for writing) when deployed (RAVELIN_DEPLOY=1 or no torch)."""
    if BU.deploy_mode():
        return None
    return Path(os.environ.get("BREACHLENS_DUEL_RUNS", str(D.HUMAN_RUNS_PATH)))


# ----------------------------------------------------------------------------------------------------------------------------- small views
def card(name: str):
    """A snow card (ink border) as a keyed container; the key must be unique on the page."""
    return st.container(key=f"card_{name}")


def sticker(text: str, hot: bool = False, rot: float | None = None, cls: str = "") -> str:
    r = f' style="--rot:{rot}deg"' if rot is not None else ""
    return f'<span class="sticker{" hot" if hot else ""}{(" " + cls) if cls else ""}"{r}>{text}</span>'


def title_block() -> None:
    st.markdown(f'<div class="titleblock"><h1>Ravelin</h1><div class="tagline">{TAGLINE}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card slim">' + html.escape(R.DISCLAIMER) + "</div>", unsafe_allow_html=True)        # the ONE disclaimer line of the screen (row (nn) A1 / A2)


def hud(gold: int, bought: int, phase: int, flags: list | None = None) -> None:
    rot = TH.STICKER_ROT
    chips = "".join(sticker(p, hot=i == phase, rot=rot[i % len(rot)]) for i, p in enumerate(PHASES))
    st.markdown(f'<div class="card narrow"><div class="hud"><span class="stat">{IC.img("coin", 32, "gold")}<b>{gold}</b>gold</span>'
                f'<span class="stat">{IC.img("troops", 26, "units")}<b>{bought}</b>/ {CFG.unit_cap} units</span><span class="phases">{chips}</span></div>'
                + ('<div class="flags">' + "".join(flags) + "</div>" if flags else "") + "</div>", unsafe_allow_html=True)


def army_strip(counts: dict, size: int = 22) -> str:
    return "".join(f'<span>{IC.img(u, size, u)}&times;{n}</span>' for u, n in counts.items() if n)


def pill(breach: bool) -> str:
    return sticker("breach", hot=True, rot=1.5) if breach else sticker("held", rot=-1.5)


def badge(text: str, warn: bool = False) -> str:
    return sticker(text, hot=warn, rot=-1 if warn else 0.8)


def border_units(army) -> list:
    return [(int(t), *BORDER[int(b)], 1000) for t, b in army]


def about_panel() -> None:
    with st.expander("About this demo"):
        st.markdown(D.DEMO_NOTE)
        st.markdown("**What is TTA?** Test-time augmentation: the same board is shown to the agent in all 8 rotations and reflections, the answers are mapped back and averaged "
                    "(PPO: mean log-probability; DDQN: mean Q-value), then the best legal action is taken. It costs nothing extra to train.")
        st.markdown("**Who are the agents?** Three learners, seed 0 each: *PPO plain*, *PPO+D4* (trained with rotation / reflection augmentation) and *DDQN* (a value learner: its "
                    "softmax display is not a policy). *BestSingle* is a scripted army of one unit type (Sniper). *Oracle* is a fixed-budget search baseline, not the optimum.")
        st.markdown("**Reading the board.** Gun, Lancer and Flak are the three coral turret shapes; pips on a building are its score value (1 to 3); the keep is the core; "
                    "azure-striped tiles are the border start zone. " + R.AGENTS_SEE)
        st.markdown(results_headline())
        st.markdown('<div class="note">Source: docs/RESULTS.md, quoted verbatim; this demo shows no aggregate numbers of its own.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------------------------------- Play: state
def new_game(layout_id: str) -> None:
    for k in [k for k in st.session_state if k.startswith("g_") and k != "g_layout"]:
        del st.session_state[k]
    st.session_state["g_msg"] = ""
    st.session_state["g_duel"] = D.Duel(layout_id, human_runs_path=_runs_path())


def act(fn: str, *args) -> None:
    """One engine action; an illegal action puts the engine's message in session_state and changes nothing."""
    try:
        getattr(st.session_state["g_duel"], fn)(*args)
        st.session_state["g_msg"] = ""
    except D.DuelError as e:
        st.session_state["g_msg"] = str(e)


def _deploy() -> None:
    act("deploy", st.session_state["g_unit"], st.session_state["g_tile"])


def _on_board_click() -> None:
    """The board component reported a click / tap on border tile b: the same engine call as the Deploy button (action 5 + 52*u + b); illegal -> the engine's message."""
    v = st.session_state.get("g_board")
    b = v.get("tile") if isinstance(v, dict) else getattr(v, "tile", None)
    if b is not None:
        act("deploy", st.session_state.get("g_unit"), b)


def free_tiles(duel: D.Duel) -> list:
    env = duel.env
    return [b for b in range(N_BORDER) if env._st.border_free[b] and all(b != bb for _, bb in env.army)]


def show_board(svg: str, key: str, *, tiles: list | None = None, unit: str | None = None, on_click=None) -> None:
    """Draw a board through the component (page fonts, scales to the width). tiles = every border tile as {b, row, col, ok} for click-to-deploy, None = static."""
    kw = {"on_tile_change": on_click} if on_click else {}
    board_component()(key=key, data={"svg": svg, "tiles": tiles or [], "unit": unit, "cell": GV.CELL}, **kw)


def _show_report() -> None:
    st.session_state["g_report"] = True


def apply_demo(duel: D.Duel, stage: str) -> None:
    """Screenshot helper: drive the real engine through the row (mm) script up to `stage` (``held`` = one Scout on tile (0,3), a 1-tick battle)."""
    if stage == "held":
        duel.buy("scout"); duel.end_buy(); duel.deploy("scout", next(i for i, b in enumerate(BORDER) if tuple(b) == (0, 3)))
        return
    for u in MM_BUYS[:2 if stage == "buy" else len(MM_BUYS)]:
        duel.buy(u)
    if stage == "buy":
        return
    duel.end_buy()
    for u in MM_BUYS[:3 if stage == "deploy" else len(MM_BUYS)]:
        m = duel.env.action_masks()
        duel.deploy(u, next(b for b in range(N_BORDER) if m[encode_deploy(UNIT_TYPES.index(u), b)]))


def ensure_game(layout: str) -> D.Duel:
    ss = st.session_state
    if "g_duel" not in ss or ss["g_duel"].layout_id != layout:
        new_game(layout)
        demo = st.query_params.get("demo")
        if demo in ("buy", "deploy", "battle", "report", "held"):
            apply_demo(ss["g_duel"], demo)
            ss["g_report"] = demo == "report"
    return ss["g_duel"]


# ----------------------------------------------------------------------------------------------------------------------------- Play: screens
def board_for(duel: D.Duel, **kw) -> str:
    env, M = duel.env, replay_module()
    structs = M._struct_table(env._st.P)
    zone = [tuple(BORDER[b]) for b in free_tiles(duel)]
    return GV.board_svg(structs, None, border_units(env.army), **{"zone": zone, **kw})


def shop(duel: D.Duel) -> None:
    env = duel.env
    mask = env.action_masks()
    with card("shop"):
        h = st.columns([6, 1])
        h[0].markdown(sticker("Your army", rot=-1.2) + " " + sticker("Pick units", hot=True, rot=1.4), unsafe_allow_html=True)
        with h[1].popover("?"):
            for u in UNIT_TYPES:
                st.markdown(f"**{u.capitalize()}** ({unit_cost(UNIT_TYPES.index(u))} gold): {ROLE[u]}")
        cols = st.columns(len(UNIT_TYPES))
        for i, u in enumerate(UNIT_TYPES):
            ok, n = bool(mask[encode_buy(i)]), int(env.owned[i])                         # in the BUY phase owned = bought
            tag = sticker(IC.img("coin", 18, "gold") + str(unit_cost(i)), hot=True, rot=-2 if i % 2 else 2, cls="price")
            cols[i].markdown(f'<div class="piece {"" if ok else "off"}">{f"<span class=cnt>&times;{n}</span>" if n else ""}{IC.disc_img(u, 72)}<div class="nm">{u.capitalize()}</div>'
                             f'<div class="short">{SHORT[u]}</div>{tag}</div>', unsafe_allow_html=True)
            cols[i].button("+", key=f"g_buy_{u}", type="primary", on_click=act, args=("buy", u), disabled=not ok, width="stretch", help=f"Buy one {u} for {unit_cost(i)} gold")
        c = st.columns([1.2, 1.2, 1.2])
        c[0].button("Undo last buy", key="g_undo", on_click=act, args=("undo",), disabled=not duel.actions, width="stretch")
        c[2].button("Start deploying", key="g_end", type="primary", on_click=act, args=("end_buy",), disabled=not mask[END_BUY], width="stretch")
        st.markdown('<div class="note">Opponents hidden until your battle is over.</div>', unsafe_allow_html=True)


def deploy_panel(duel: D.Duel) -> None:
    env = duel.env
    owned = {UNIT_TYPES[u]: int(n) for u, n in enumerate(env.owned) if n}
    with card("deploy"):
        st.markdown('<div class="tray">' + "".join(f'<span class="tok">{IC.img(u, 28, u)}&times;{n}</span>' for u, n in owned.items()) + "</div>", unsafe_allow_html=True)
        free = free_tiles(duel)
        edge = lambda b: "top" if BORDER[b][0] == 0 else "bottom" if BORDER[b][0] == GV.GRID - 1 else "left" if BORDER[b][1] == 0 else "right"
        st.markdown('<div class="note" style="text-align:center">Click or tap a striped tile on the board to deploy the selected unit there.</div>', unsafe_allow_html=True)
        c = st.columns([1.2, 1.8, 1.1, 1.5])
        c[0].selectbox("Unit", list(owned), key="g_unit", format_func=str.capitalize)
        c[1].selectbox("Tile (keyboard option)", free, key="g_tile", format_func=lambda b: f"({BORDER[b][0]},{BORDER[b][1]}) {edge(b)} edge")
        c[2].button("Deploy", key="g_deploy", type="primary", on_click=_deploy, width="stretch")
        c[3].button("Start battle now", key="g_wait", on_click=act, args=("wait",), width="stretch", help="Units not yet deployed are forfeited")
        if st.session_state.get("g_msg"):
            st.error(st.session_state["g_msg"])


def battle_view(duel: D.Duel) -> None:
    ss = st.session_state
    if "g_battle" not in ss:
        M = replay_module()
        ss["g_battle"] = M._battle(duel.env)
        ss["g_structs"] = M._struct_table(duel.env._st.P)
    battle, res = ss["g_battle"], duel.env.result
    out = {"breach": bool(res.core_destroyed), "ticks": int(res.ticks), "reasonText": GV.outcome_reason_text(res.reason)}
    demo_tick = st.query_params.get("tick")
    start = int(demo_tick) if demo_tick and demo_tick.isdigit() else 0
    page = GV.player_html(ss["g_structs"], battle["frames"], [t for t, _ in battle["units"]], out, autoplay=not demo_tick, start=start)
    with card("board"):
        st.iframe(page, height=PLAYER_H)


def report_view(duel: D.Duel) -> None:
    ss = st.session_state
    if "g_rows" not in ss:                                       # the first (and only) moment the opponents are read
        ss["g_rows"] = duel.reveal()
    rows = ss["g_rows"]
    me = rows[0]
    st.markdown(f'<div class="card"><span class="verdict">{"Breach!" if me["breach"] else "Held."}</span> &nbsp;score <b class="num">{me["score"]:.3f}</b> '
                f'&middot; D <span class="num">{me["d"]:.3f}</span> &middot; <span class="num">{me["n_deployed"]}</span> units</div>', unsafe_allow_html=True)
    with st.container(key="grid_report"):
        for r0 in range(0, len(rows), 3):
            cols = st.columns(3)
            for col, r in zip(cols, rows[r0:r0 + 3]):
                name, sub, icon = CONTESTANT[r["contestant"]]
                col.markdown(
                    f'<div class="ccard {"me" if r["contestant"] == "human" else ""}"><div class="chead">{IC.img(icon, 34, name)}<div class="grow"><b>{html.escape(name)}</b>'
                    f'<div class="sub">{html.escape(sub) or "&nbsp;"}</div></div>{pill(r["breach"])}</div>'
                    f'<div class="sbar"><i style="width:{max(0.0, min(1.0, r["score"])) * 100:.1f}%"></i></div>'
                    f'<div class="nums"><b>{r["score"]:.3f}</b> score &middot; D <span class="num">{r["d"]:.3f}</span></div>'
                    f'<div class="army">{army_strip(r["bought"])}</div></div>', unsafe_allow_html=True)
    with st.expander("Full comparison table"):
        body = "".join(
            f'<tr class="{"human" if r["contestant"] == "human" else ""}"><td>{html.escape(r["label"])}</td><td class="num">{r["score"]:.3f}</td><td class="num">{r["d"]:.3f}</td>'
            f'<td>{pill(r["breach"])}</td><td class="num">{r["n_bought"]}</td><td class="num">{r["n_deployed"]}</td>'
            f'<td>{html.escape(" ".join(f"{u[:3]}={n}" for u, n in r["bought"].items() if n))}</td>'
            f'<td>{", ".join(html.escape(R.deploy_text(u, *BORDER[b])) for u, b in r["deploy"]) or "none"}</td></tr>' for r in rows)
        st.markdown('<table class="top"><tr><th>contestant</th><th class="num">score</th><th class="num">D</th><th>breach</th><th class="num">bought</th><th class="num">deployed</th>'
                    f'<th>army (bought per type)</th><th>deploy tiles (row, col)</th></tr>{body}</table>', unsafe_allow_html=True)
        st.markdown('<div class="note">Oracle = search baseline, not the optimum. ' + html.escape(D.DEMO_NOTE) + "</div>", unsafe_allow_html=True)


def play_controls() -> tuple:
    ids = duel_ids()
    c = st.columns([4, 3, 1.3])
    mode = c[0].radio("Mode", MODES, index=1 if st.query_params.get("mode") == "watch" else 0, key="mode_sel", horizontal=True)
    layout = c[1].selectbox("Layout", ids, index=ids.index(DEFAULT_LAYOUT), key="g_layout", label_visibility="collapsed", help="DEV-validation layout")
    c[2].button("New game", key="g_new", on_click=lambda: new_game(st.session_state["g_layout"]), width="stretch")
    return mode, layout


def play_mode(layout: str) -> None:
    ss = st.session_state
    duel = ensure_game(layout)
    env = duel.env
    flags = [badge("DEV-validation"), badge(duel.family)]
    if not duel.finished:
        hud(env.gold, env.bought, 0 if duel.phase == "BUY" else 1, flags)
        if duel.phase == "BUY":
            with card("board"):
                show_board(board_for(duel), "g_board_buy")
            shop(duel)
        else:
            c = deploy_preview(duel)
            with card("board"):
                tiles = [{"b": b, "row": BORDER[b][0], "col": BORDER[b][1], "ok": b in set(free_tiles(duel))} for b in range(N_BORDER)]
                show_board(board_for(duel, selected=c["selected"], preview=c["preview"]), "g_board", tiles=tiles, unit=c["unit"], on_click=_on_board_click)
            deploy_panel(duel)
        return
    report = bool(ss.get("g_report"))
    hud(env.gold, env.bought, 3 if report else 2, flags)
    battle_view(duel)
    if not report:
        with card("cta"):
            c = st.columns([1.5, 1.4, 1.5])
            c[1].button("Show report", key="g_report_btn", type="primary", on_click=_show_report, width="stretch", help="Reveals the other contestants on this layout")
            c[1].markdown('<div class="note" style="text-align:center">Opponents hidden until you press this.</div>', unsafe_allow_html=True)
    else:
        report_view(duel)


def deploy_preview(duel: D.Duel) -> dict:
    """Board decorations for the deploy phase, from the selector's current state (selected / preview = the chosen tile and unit)."""
    env, ss = duel.env, st.session_state
    free = free_tiles(duel)
    owned = [UNIT_TYPES[u] for u in range(len(UNIT_TYPES)) if env.owned[u] > 0]
    tile, unit = ss.get("g_tile", free[0] if free else None), ss.get("g_unit", owned[0] if owned else None)
    sel = tuple(BORDER[tile]) if tile in free else None
    prev = (UNIT_TYPES.index(unit), *BORDER[tile]) if sel and unit in UNIT_TYPES and env.owned[UNIT_TYPES.index(unit)] > 0 else None
    return {"selected": sel, "preview": prev, "unit": unit}


# ----------------------------------------------------------------------------------------------------------------------------- Watch the AI
def heat_for(s: dict, mode: str) -> tuple:
    """(overlay {(row, col): colour}, caption, chosen tile) for one decision step (same values as the replays' heatmap)."""
    m = s["action"] - 5
    chosen = tuple(BORDER[m % N_BORDER]) if 0 <= m < 5 * N_BORDER else None
    if s["heat_total"] <= 0:
        return {}, "no deploy action is legal at this step, so the heatmap is empty", chosen
    if mode == HEATS[1]:
        vals, cap = s["heat_tile"], "per-tile total over all unit types"
    else:
        u = m // N_BORDER if 0 <= m < 5 * N_BORDER else max(range(5), key=lambda i: (s["heat_unit"][i], -i))
        vals, cap = s["heat"][u], f"unit type {UNIT_TYPES[u]}" + ("" if 0 <= m < 5 * N_BORDER else " (most deploy probability; no deploy chosen here)")
    mx = max(vals + [1e-12])
    return {tuple(BORDER[b]): TH.heat_color(v / mx) for b, v in enumerate(vals) if v > 0}, f"{cap}; colour scale 0 to {mx:.4f}", chosen


def fmt(x, n=4) -> str:
    return "n/a" if x is None else f"{x:.{n}f}"


def lens_card(ep: dict, s: dict, pos: int, n_dec: int) -> None:
    from rl import explain as X
    ddqn = ep["algo"] == "ddqn"
    txt, tip = R.pretty_label(s["label"])
    head = (f"Top 5 by Q; softmax is NOT a policy (T = {X.DDQN_TEMPERATURE})." if ddqn else
            "Top 5 by probability" + (" (TTA: softmax of the 8-view mean log-probability)." if ep["tta"] else "."))
    rows = "".join(
        f'<tr class="{"chosen" if t["a"] == s["action"] else ""}"><td class="num">{i}</td><td title="{R.pretty_label(t["label"])[1]}">{html.escape(R.pretty_label(t["label"])[0])}</td>'
        + (f'<td class="num">{fmt(t["q"])}</td><td class="num">{fmt(t["p"])}</td>' if ddqn else f'<td class="num">{fmt(t["p"])}</td>') + "</tr>" for i, t in enumerate(s["top"], 1))
    th = '<th class="num">#</th><th>action</th>' + ('<th class="num">Q</th><th class="num">softmax</th>' if ddqn else '<th class="num">prob.</th>')
    with card("lens"):
        h = st.columns([5, 1])
        h[0].markdown('<span class="verdict">Why this move?</span> ' + (sticker("not a policy", hot=True, rot=-1.5) if ddqn else ""), unsafe_allow_html=True)
        with h[1].popover("?"):
            st.markdown(f"{head} Value = {X.VALUE_LABEL}; {'critic V(s)' if ep['algo'] == 'ppo' else 'max over legal actions of Q'}"
                        f"{' (TTA: mean over 8 views)' if ep['tta'] else ''}. " + (f"DDQN: {X.DDQN_NOT_POLICY} (T = {X.DDQN_TEMPERATURE})." if ddqn else ""))
        st.markdown(f'<div class="note">decision {pos + 1} of {n_dec} &middot; {s["phase"].lower()} &middot; {s["n_legal"]} legal actions</div>'
                    f'<div style="margin:6px 0"><b>Chosen:</b> <span title="{tip}">{html.escape(txt)}</span> <span class="note">p = <span class="num">{fmt(s["p_chosen"])}</span></span></div>'
                    f'<table class="top"><tr>{th}</tr>{rows}</table>'
                    f'<div style="margin-top:8px"><b>Value <span class="num">{fmt(s["value"], 2)}</span></b> <span class="note">&ndash; shaped-return estimate, not the score</span></div>',
                    unsafe_allow_html=True)


def lens_record(ep: dict, pos: int, n_dec: int) -> dict:
    """What the decision lens shows at `pos` as plain data (session_state['w_lens']): the replay's own chosen action / value / top 5 (kind 'decision') or the battle (kind 'battle')."""
    lens = {"pos": pos, "n_decision_steps": n_dec, "final_outcome": ep["outcome"]}
    if pos < n_dec:
        s = ep["steps"][pos]
        lens.update(kind="decision", chosen_action=s["action"], chosen_label=s["label"], top=[t["a"] for t in s["top"]], value=s["value"])
    else:
        lens.update(kind="battle", ticks=int(ep["outcome"]["ticks"]), core_destroyed=bool(ep["outcome"]["core"]))
    return lens


def watch_controls() -> tuple:
    from rl import explain as X
    c = st.columns([1.5, 1.1, 2.4, 1.3, 1.0])
    mode = c[0].radio("Mode", MODES, index=1, key="mode_sel", horizontal=True)
    split = c[1].selectbox("Split", SPLITS[:1] if BU.deploy_mode() else SPLITS, key="w_split")
    ids = layout_ids(split)
    layout = c[2].selectbox("Layout", ids, index=ids.index(DEFAULT_LAYOUT) if DEFAULT_LAYOUT in ids else 0, key=f"w_layout_{split}")
    arm = c[3].selectbox("Agent (seed 0)", tuple(X.DISPLAY_MODELS), key="w_arm", format_func=lambda a: X.DISPLAY_MODELS[a][0])
    tta = c[4].radio("View", VIEWS, key="w_view", horizontal=True) == "TTA"
    return mode, split, layout, arm, tta


def watch_mode(split: str, layout: str, arm: str, tta: bool) -> None:
    ss = st.session_state
    ep = episode(split, layout, arm, tta)
    n_dec, frames = len(ep["steps"]), ep["battle"]["frames"]
    key = f"w_pos|{split}|{layout}|{arm}|{tta}"
    try:
        start = max(0, min(n_dec, int(st.query_params.get("pos", 0))))
    except (TypeError, ValueError):
        start = 0
    ss.setdefault(key, start)
    flags = ([badge("IN-SAMPLE: the learner trained on this layout", True)] if split == "DEV-train" else [badge("DEV-validation")]) + \
            [badge("TTA" if tta else "plain"), badge(ep["family"])]
    pos = ss[key]
    in_battle = pos >= n_dec
    ss["w_lens"] = lens_record(ep, pos, n_dec)
    s = ep["steps"][min(pos, n_dec - 1)]
    last = ep["steps"][-1]
    hud(last["gold"] if in_battle else s["gold"], last["bought"] if in_battle else s["bought"], 2 if in_battle else (0 if s["phase"] == "BUY" else 1), flags)
    with st.container(key="grid_watch"):
        left, right = st.columns([3.2, 1.9])
        with left, card("board"):
            b = st.columns([1, 1, 5])
            b[0].button("Prev", key="w_prev", on_click=lambda: ss.__setitem__(key, max(0, ss[key] - 1)), disabled=pos <= 0, width="stretch")
            b[1].button("Next", key="w_next", on_click=lambda: ss.__setitem__(key, min(n_dec, ss[key] + 1)), disabled=pos >= n_dec, width="stretch")
            b[2].slider("Decision step (the last stop is the battle)", 0, n_dec, key=key, label_visibility="collapsed")
            if not in_battle:
                h = st.columns([1.2, 2.8])
                heat_on = h[0].toggle("Heatmap overlay", value=True, key="w_heat")
                mode = h[1].radio("Heatmap shows", HEATS, key="w_heatmode", horizontal=True, label_visibility="collapsed") if heat_on else HEATS[0]
                overlay, cap, chosen = heat_for(s, mode) if heat_on else ({}, "", heat_for(s, HEATS[0])[2])
                zone = [tuple(BORDER[b_]) for b_ in range(N_BORDER)]
                show_board(GV.board_svg([list(x) for x in ep["structs"]], None, border_units(s["army"]), overlay=overlay, zone=zone, chosen=chosen), "w_board")
                if cap:
                    st.markdown(f'<div class="note">{html.escape(cap)}</div>', unsafe_allow_html=True)
            else:
                o = ep["outcome"]
                out = {"breach": bool(o["core"]), "ticks": int(o["ticks"]), "reasonText": GV.outcome_reason_text(o["reason"])}
                page = GV.player_html(ep["structs"], frames, [t for t, _ in ep["battle"]["units"]], out, autoplay=st.query_params.get("tick") is None,
                                      start=int(st.query_params.get("tick", 0)))
                st.iframe(page, height=PLAYER_H)
        with right:
            o = ep["outcome"]
            with card("outcome"):
                st.markdown(f'<span class="verdict">{"Breach" if o["core"] else "Held"}</span> &nbsp;score <b class="num">{o["score"]:.4f}</b> &middot; D <span class="num">{o["d"]:.4f}</span>'
                            f'<div class="note">{o["n_units"]} units &middot; {o["ticks"]} ticks</div>', unsafe_allow_html=True)
            if not in_battle:
                lens_card(ep, s, pos, n_dec)
            else:
                with card("lens"):
                    st.markdown('<span class="verdict">Battle</span><div class="note">All decisions came before the battle. Step back with Prev to see each one.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    st.markdown(TH.app_css() + "<style>" + TH.italic_face_css() + TH.game_root_vars() + "</style>" + PROTO_CSS, unsafe_allow_html=True)
    title_block()
    with card("controls"):
        q = st.query_params.get("mode")
        mode_now = st.session_state.get("mode_sel", MODES[1] if q == "watch" else MODES[0])
        if mode_now == MODES[0]:
            mode, layout = play_controls()
        else:
            mode, split, layout, arm, tta = watch_controls()
    if mode == MODES[0]:
        play_mode(layout)
    else:
        watch_mode(split, layout, arm, tta)
    about_panel()


main()
