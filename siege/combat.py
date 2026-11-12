from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from siege.troops import get_troop_spec

BOARD_MIN = 0
BOARD_MAX = 15


@dataclass(frozen=True, slots=True)
class BattleResult:
    frames: list[dict]
    final_troops: list[dict]
    final_buildings: list[dict]
    final_turrets: list[dict]
    destroyed_value: int
    total_value: int
    destruction: float
    tick: int

    @property
    def reward(self) -> float:
        return self.destruction


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_army(army: Iterable[dict]) -> list[dict]:
    troops: list[dict] = []
    for i, d in enumerate(army):
        spec = get_troop_spec(d.get("type", i))
        troops.append(
            {
                "id": i,
                "type": spec.name,
                "hp": int(d.get("hp", spec.hp)),
                "damage": int(d.get("damage", spec.damage)),
                "speed": float(d.get("speed", spec.speed)),
                "range": float(d.get("range", spec.range)),
                "x": int(d.get("x", 0)),
                "y": int(d.get("y", 0)),
                "target_preference": tuple(d.get("target_preference", spec.target_preference)),
                "splash_radius": float(d.get("splash_radius", spec.splash_radius)),
                "splash_falloff": float(d.get("splash_falloff", spec.splash_falloff)),
                "alive": True,
            }
        )
    return troops


def _normalize_buildings(buildings: Iterable[dict]) -> list[dict]:
    items: list[dict] = []
    for idx, b in enumerate(buildings):
        items.append(
            {
                "id": idx,
                "x": int(b["x"]),
                "y": int(b["y"]),
                "hp": int(b.get("hp", 50)),
                "value": int(b.get("value", 10)),
                "name": b.get("name", "building"),
                "alive": True,
            }
        )
    return items


def _normalize_turrets(turrets: Iterable[dict]) -> list[dict]:
    items: list[dict] = []
    for idx, t in enumerate(turrets):
        items.append(
            {
                "id": idx,
                "x": int(t["x"]),
                "y": int(t["y"]),
                "hp": int(t.get("hp", 40)),
                "damage": int(t.get("damage", 10)),
                "range": int(t.get("range", 5)),
                "alive": True,
            }
        )
    return items


def _normalize_walls(walls: Iterable[tuple[int, int] | dict]) -> dict[tuple[int, int], int]:
    wall_map: dict[tuple[int, int], int] = {}
    for entry in walls:
        if isinstance(entry, dict):
            x = int(entry["x"])
            y = int(entry["y"])
            hp = int(entry.get("hp", 80))
        else:
            x, y = entry
            hp = 80
        wall_map[(int(x), int(y))] = hp
    return wall_map


def _snapshot(troops: list[dict], buildings: list[dict], turrets: list[dict], walls: dict[tuple[int, int], int], destruction: float, tick: int) -> dict:
    return {
        "tick": tick,
        "troops": [
            {"id": t["id"], "type": t["type"], "x": t["x"], "y": t["y"], "hp": t["hp"], "damage": t["damage"], "range": t["range"], "speed": t["speed"]}
            for t in troops
            if t["alive"]
        ],
        "buildings": [
            {"id": b["id"], "x": b["x"], "y": b["y"], "hp": b["hp"], "value": b["value"], "name": b["name"]}
            for b in buildings
            if b["alive"]
        ],
        "turrets": [
            {"id": t["id"], "x": t["x"], "y": t["y"], "hp": t["hp"], "damage": t["damage"], "range": t["range"]}
            for t in turrets
            if t["alive"]
        ],
        "walls": [{"x": x, "y": y, "hp": hp} for (x, y), hp in sorted(walls.items())],
        "destruction": float(destruction),
    }


def _select_target(troop: dict, buildings: list[dict], turrets: list[dict], walls: dict[tuple[int, int], int]) -> tuple[str, tuple[int, int], float]:
    target_priority = [p for p in troop["target_preference"] if p in {"building", "turret", "wall"}]
    if not target_priority:
        target_priority = ["building", "turret", "wall"]

    # Tie-breaks use only D4-invariant properties (distance, then the target's own stats, then its
    # label/id) -- never board coordinates -- so rotating/mirroring a battle cannot change who is hit.
    options: list[tuple[int, float, tuple, int, str, tuple[int, int]]] = []
    for rank, kind in enumerate(target_priority):
        if kind == "building":
            for b in buildings:
                if b["alive"]:
                    options.append((rank, _dist((troop["x"], troop["y"]), (b["x"], b["y"])), (b["hp"], b["value"]), b["id"], kind, (b["x"], b["y"])))
        elif kind == "turret":
            for t in turrets:
                if t["alive"]:
                    options.append((rank, _dist((troop["x"], troop["y"]), (t["x"], t["y"])), (t["hp"], t["damage"], t["range"]), t["id"], kind, (t["x"], t["y"])))
        elif kind == "wall":
            for idx, (pos, hp) in enumerate(walls.items()):  # insertion order == wall label
                options.append((rank, _dist((troop["x"], troop["y"]), pos), (hp,), idx, kind, pos))
    if not options:
        return ("", (-1, -1), 1e9)
    options.sort(key=lambda item: item[:4])
    best = options[0]
    # Still tied on (rank, distance, own stats) -> e.g. two equal-hp wall tiles equally far away.
    # Prefer the one nearer the rest of the base (D4-invariant, order-independent via fsum); the
    # label (id) only decides if even that ties.
    tied = [o for o in options if o[:3] == best[:3]]
    if len(tied) > 1:
        anchors = [(b["x"], b["y"]) for b in buildings if b["alive"]] + [(t["x"], t["y"]) for t in turrets if t["alive"]]
        tied.sort(key=lambda o: (math.fsum(_dist(o[5], a) for a in anchors), o[3]))
        best = tied[0]
    _, dist, _, _, kind, pos = best
    return kind, pos, dist


def _move_troop(troop: dict, target: tuple[int, int], walls: dict[tuple[int, int], int], buildings: list[dict] | None = None, turrets: list[dict] | None = None) -> None:
    tx, ty = target
    x, y = troop["x"], troop["y"]
    if (x, y) == (tx, ty):
        return

    candidate_steps: list[tuple[int, int]] = []
    dx = 0 if tx == x else (1 if tx > x else -1)
    dy = 0 if ty == y else (1 if ty > y else -1)
    if dx != 0:
        candidate_steps.append((x + dx, y))
    if dy != 0:
        candidate_steps.append((x, y + dy))
    if dx != 0 and dy != 0:
        candidate_steps.extend([(x + dx, y + dy), (x, y)])
    ordered = []
    for nx, ny in candidate_steps:
        if not (BOARD_MIN <= nx <= BOARD_MAX and BOARD_MIN <= ny <= BOARD_MAX):
            continue
        if (nx, ny) in walls:
            continue
        ordered.append((nx, ny))
    if not ordered:
        ordered = [(x, y)]
    # Prefer the step that gets closest (Manhattan, then Euclidean: i.e. shrink the longer axis
    # first). Both are D4-invariant. A tie here means the troop sits on the diagonal to its target
    # with the diagonal blocked; it is then broken by which step leaves the troop closer to the
    # remaining structures overall (also D4-invariant, and order-independent via fsum). Only a
    # world that is itself mirror-symmetric about that diagonal can still tie, where either choice
    # is equivalent; a fixed coordinate order resolves that last case.
    def primary(p: tuple[int, int]) -> tuple[int, float]:
        return (abs(p[0] - tx) + abs(p[1] - ty), _dist(p, target))

    best = min(primary(p) for p in ordered)
    tied = [p for p in ordered if primary(p) == best]
    if len(tied) > 1:
        anchors = [(o["x"], o["y"]) for o in (buildings or []) if o["alive"]]
        anchors += [(o["x"], o["y"]) for o in (turrets or []) if o["alive"]]
        anchors += list(walls)
        nx, ny = min(tied, key=lambda p: (math.fsum(_dist(p, a) for a in anchors), p[1], p[0]))
    else:
        nx, ny = tied[0]
    if (nx, ny) != (x, y):
        troop["x"], troop["y"] = nx, ny


def _deal_damage(target: dict, amount: int) -> bool:
    target["hp"] -= amount
    if target["hp"] <= 0:
        target["hp"] = 0
        target["alive"] = False
        return True
    return False


def _hit_wall(walls: dict[tuple[int, int], int], pos: tuple[int, int], amount: int) -> None:
    if pos in walls:
        walls[pos] = max(0, walls[pos] - amount)
        if walls[pos] <= 0:
            del walls[pos]


def _strike(troop: dict, kind: str, pos: tuple[int, int], buildings: list[dict], turrets: list[dict], walls: dict[tuple[int, int], int]) -> int:
    """One attack on the target at `pos`; returns the building value destroyed (incl. splash).

    Splash troops also damage everything within `splash_radius` of `pos` for
    int(damage * splash_falloff). All damage in a strike is applied independently, so the result
    does not depend on iteration order (and radius is rotation/mirror invariant).
    """
    gained = 0
    dmg = troop["damage"]
    splash = int(dmg * troop["splash_falloff"]) if troop["splash_radius"] > 0 else 0

    def in_splash(p: tuple[int, int]) -> bool:
        return splash > 0 and _dist(pos, p) <= troop["splash_radius"]

    for b in buildings:
        if not b["alive"]:
            continue
        here = (b["x"], b["y"])
        amount = dmg if (kind == "building" and here == pos) else (splash if in_splash(here) else 0)
        if amount and _deal_damage(b, amount):
            gained += b["value"]
    for t in turrets:
        if not t["alive"]:
            continue
        here = (t["x"], t["y"])
        amount = dmg if (kind == "turret" and here == pos) else (splash if in_splash(here) else 0)
        if amount:
            _deal_damage(t, amount)
    if kind == "wall":
        _hit_wall(walls, pos, dmg)
    if splash > 0:
        for wpos in [w for w in walls if w != pos or kind != "wall"]:
            if in_splash(wpos):
                _hit_wall(walls, wpos, splash)
    return gained


def simulate_battle(army, buildings, walls, turrets, max_ticks: int = 48):
    """Run a deterministic tick-based battle.

    Tie-breaking rules are fixed, stable, and D4-invariant (rotating/mirroring the whole battle,
    including the army, gives the identical outcome -- see tests/test_augment.py):
      1. troop actions are processed in ascending id order (id == deployment order).
      2. each troop picks the nearest valid target of its highest-ranked preferred category; equal
         distances resolve by the target's own stats (hp, value / damage, range), then its label (id).
      3. movement takes the diagonal step toward the target when open; otherwise the open orthogonal
         step with the smallest Manhattan, then Euclidean, remaining distance; walls block moves.
      4. turrets fire simultaneously at the nearest troop (ties: lowest hp, then id); damage is
         applied after all targets are chosen, so turret order is irrelevant.
      5. splash troops also damage every structure within splash_radius of the struck tile.
    """
    troops = _normalize_army(army)
    buildings = _normalize_buildings(buildings)
    turrets = _normalize_turrets(turrets)
    walls = _normalize_walls(walls)
    total_value = sum(b["value"] for b in buildings)
    destroyed_value = 0
    frames: list[dict] = []

    for tick in range(max_ticks + 1):
        current_destruction = 0.0 if total_value <= 0 else float(destroyed_value / total_value)
        frames.append(_snapshot(troops, buildings, turrets, walls, current_destruction, tick))

        if not any(t["alive"] for t in troops):
            break
        if total_value <= 0 and not walls:
            break

        # Troop movement and attacks.
        active_troops = sorted((t for t in troops if t["alive"]), key=lambda t: (t["id"], t["y"], t["x"]))
        for troop in active_troops:
            if not troop["alive"]:
                continue
            kind, target_pos, target_dist = _select_target(troop, buildings, turrets, walls)
            if kind and target_dist <= troop["range"]:
                destroyed_value += _strike(troop, kind, target_pos, buildings, turrets, walls)
            elif kind:
                _move_troop(troop, target_pos, walls, buildings, turrets)
                # Re-evaluate after moving to see whether target is now in range.
                kind2, target_pos2, target_dist2 = _select_target(troop, buildings, turrets, walls)
                if kind2 and target_dist2 <= troop["range"]:
                    destroyed_value += _strike(troop, kind2, target_pos2, buildings, turrets, walls)

        # Turrets fire simultaneously: every turret picks its target from the start-of-phase state
        # (nearest troop, ties by lowest hp then id) and damage is applied afterwards, so the outcome
        # cannot depend on turret processing order (overkill is allowed).
        live_turrets = [t for t in turrets if t["alive"]]
        live_troops = [(t, t["hp"]) for t in troops if t["alive"]]
        pending: dict[int, int] = {}
        for turret in live_turrets:
            tpos = (turret["x"], turret["y"])
            candidates = [(troop, hp) for troop, hp in live_troops if _dist((troop["x"], troop["y"]), tpos) <= turret["range"]]
            if not candidates:
                continue
            target, _ = min(candidates, key=lambda c: (_dist((c[0]["x"], c[0]["y"]), tpos), c[1], c[0]["id"]))
            pending[target["id"]] = pending.get(target["id"], 0) + turret["damage"]
        for troop, _ in live_troops:
            dmg = pending.get(troop["id"], 0)
            if dmg:
                troop["hp"] = max(0, troop["hp"] - dmg)
                if troop["hp"] == 0:
                    troop["alive"] = False

        # Remove dead targets.
        for b in buildings:
            if b["alive"] and b["hp"] <= 0:
                b["alive"] = False
                destroyed_value += b["value"]
        for t in turrets:
            if t["alive"] and t["hp"] <= 0:
                t["alive"] = False

        if not any(b["alive"] for b in buildings) and not any(t["alive"] for t in turrets) and not walls:
            break

    final_troops = [{"id": t["id"], "type": t["type"], "x": t["x"], "y": t["y"], "hp": max(0, t["hp"])} for t in troops]
    final_buildings = [{"id": b["id"], "x": b["x"], "y": b["y"], "hp": max(0, b["hp"]), "value": b["value"], "alive": b["alive"]} for b in buildings]
    final_turrets = [{"id": t["id"], "x": t["x"], "y": t["y"], "hp": max(0, t["hp"]), "damage": t["damage"], "range": t["range"], "alive": t["alive"]} for t in turrets]
    destruction = 0.0 if total_value <= 0 else _clip(destroyed_value / total_value, 0.0, 1.0)
    return BattleResult(frames=frames, final_troops=final_troops, final_buildings=final_buildings, final_turrets=final_turrets, destroyed_value=int(destroyed_value), total_value=int(total_value), destruction=float(destruction), tick=tick)


__all__ = ["BattleResult", "simulate_battle"]
