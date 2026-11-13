"""Border tiles and the Discrete(267) action codec (docs/SPEC_V2.md sections 2 and 4.1)."""
from __future__ import annotations

from .constants import (
    BUY_BASE, DEPLOY_BASE, END_BUY, GRID, N_ACTIONS, N_BORDER, N_UNIT_TYPES, WAIT,
)


def border_tiles() -> list[tuple[int, int]]:
    """The 52 border cells (row, col), clockwise from the top-left corner."""
    top = [(0, c) for c in range(GRID)]
    right = [(r, GRID - 1) for r in range(1, GRID)]
    bottom = [(GRID - 1, c) for c in range(GRID - 2, -1, -1)]
    left = [(r, 0) for r in range(GRID - 2, 0, -1)]
    tiles = top + right + bottom + left
    assert len(tiles) == N_BORDER and len(set(tiles)) == N_BORDER
    return tiles


BORDER = border_tiles()


def encode_buy(unit_idx: int) -> int:
    unit_idx = int(unit_idx)
    if not 0 <= unit_idx < N_UNIT_TYPES:
        raise ValueError(f"unit_idx out of range: {unit_idx}")
    return BUY_BASE + unit_idx


def encode_deploy(unit_idx: int, border_idx: int) -> int:
    unit_idx, border_idx = int(unit_idx), int(border_idx)
    if not 0 <= unit_idx < N_UNIT_TYPES:
        raise ValueError(f"unit_idx out of range: {unit_idx}")
    if not 0 <= border_idx < N_BORDER:
        raise ValueError(f"border_idx out of range: {border_idx}")
    return DEPLOY_BASE + unit_idx * N_BORDER + border_idx


def decode_action(action: int) -> tuple[str, int | None, int | None]:
    """Return ('buy', unit, None) | ('deploy', unit, tile) | ('end_buy', None, None) | ('wait', None, None)."""
    action = int(action)
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"invalid action index {action}; expected 0 <= action < {N_ACTIONS}")
    if action < DEPLOY_BASE:
        return ("buy", action - BUY_BASE, None)
    if action < END_BUY:
        unit_idx, border_idx = divmod(action - DEPLOY_BASE, N_BORDER)
        return ("deploy", unit_idx, border_idx)
    if action == END_BUY:
        return ("end_buy", None, None)
    return ("wait", None, None)
