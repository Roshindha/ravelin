"""Shared test helpers for the B24 arena player (not a test module)."""
from app import arena as AR


def mini_page(outcome=None, structs=None, mode="human", **kw):
    """The battle player page for a one-frame battle (one core, no troops): for static checks of the page (CSS, labels, banner)."""
    structs = structs or [["c", 3, 3, ""]]
    battle = {"frames": [{"t": 0, "u": [], "s": [1000]}], "units": [], "umax": [], "smax": [1000], "S": 1}
    return AR.player_html(structs, battle, outcome or {"breach": False, "ticks": 0, "reasonText": "x"}, {"d": [], "u": [0, 0, 0, 0, 0]}, mode=mode, **kw)


def permille_battle(d: dict) -> dict:
    """A replay's permille frames as a battle dict (S = 1: the 'raw' unit is the permille), for tests that only need a core tick."""
    n, ns = len(d["battle"]["units"]), len(d["structs"])
    return {"frames": d["battle"]["frames"], "units": d["battle"]["units"], "umax": [1000] * n, "smax": [1000] * ns, "S": 1}
