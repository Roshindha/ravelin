# Action space specification

The environment uses a single `gymnasium.spaces.Discrete(246)` action space.

The layout is:

- `0..3`: buy troop type `k`
- `4..243`: deploy a troop of type `k` onto border tile `t`
- `244`: `END_BUY`
- `245`: `WAIT`

## Exact index formula

For a deploy action:

- `troop_idx = (action - 4) // 60`
- `tile_idx = (action - 4) % 60`
- action = `4 + troop_idx * 60 + tile_idx`

Helpers are exposed in `siege.env`:

- `encode_deploy(troop_idx, border_idx) -> int`
- `decode_action(action) -> (kind, troop_idx_or_None, border_idx_or_None)`

The border tile ordering follows the same 60-tile clockwise border walk used by `siege.env.BORDER`.
