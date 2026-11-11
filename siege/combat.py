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

    options: list[tuple[int, float, int, int, str, tuple[int, int]]] = []
    for rank, kind in enumerate(target_priority):
        if kind == "building":
            for b in buildings:
                if b["alive"]:
                    options.append((rank, _dist((troop["x"], troop["y"]), (b["x"], b["y"])), b["y"], b["x"], kind, (b["x"], b["y"])))
        elif kind == "turret":
            for t in turrets:
                if t["alive"]:
                    options.append((rank, _dist((troop["x"], troop["y"]), (t["x"], t["y"])), t["y"], t["x"], kind, (t["x"], t["y"])))
        elif kind == "wall":
            for pos in sorted(walls):
                options.append((rank, _dist((troop["x"], troop["y"]), pos), pos[1], pos[0], kind, pos))
    if not options:
        return ("", (-1, -1), 1e9)
    options.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, dist, _, _, kind, pos = options[0]
    return kind, pos, dist


def _move_troop(troop: dict, target: tuple[int, int], walls: dict[tuple[int, int], int]) -> None:
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
    # Deterministic tie-breaker: north, east, south, west order for blocked or equal-distance options.
    ordered = []
    for nx, ny in candidate_steps:
        if not (BOARD_MIN <= nx <= BOARD_MAX and BOARD_MIN <= ny <= BOARD_MAX):
            continue
        if (nx, ny) in walls:
            continue
        ordered.append((nx, ny))
    if not ordered:
        ordered = [(x, y)]
    nx, ny = min(ordered, key=lambda p: (abs(p[0] - tx) + abs(p[1] - ty), p[1], p[0]))
    if (nx, ny) != (x, y):
        troop["x"], troop["y"] = nx, ny


def _deal_damage(target: dict, amount: int) -> bool:
    target["hp"] -= amount
    if target["hp"] <= 0:
        target["hp"] = 0
        target["alive"] = False
        return True
    return False


def simulate_battle(army, buildings, walls, turrets, max_ticks: int = 48):
    """Run a deterministic tick-based battle.

    Tie-breaking rules are intentionally fixed and stable:
      1. troop actions are processed in ascending id order.
      2. each troop picks the nearest valid target of its preferred category, resolving equal
         distances by (target type priority, y, x).
      3. movement uses the shortest legal step toward the target and resolves blocked cells by
         a north-east-south-west order before falling back to staying put.
      4. turret shots prefer the nearest troop by Euclidean distance; ties resolve by (y, x, id).
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
                if kind == "building":
                    target = next((b for b in buildings if b["alive"] and (b["x"], b["y"]) == target_pos), None)
                    if target is not None:
                        _deal_damage(target, troop["damage"])
                        if not target["alive"]:
                            destroyed_value += target["value"]
                elif kind == "turret":
                    target = next((t for t in turrets if t["alive"] and (t["x"], t["y"]) == target_pos), None)
                    if target is not None:
                        _deal_damage(target, troop["damage"])
                elif kind == "wall":
                    if (target_pos[0], target_pos[1]) in walls:
                        walls[(target_pos[0], target_pos[1])] = max(0, walls[(target_pos[0], target_pos[1])] - troop["damage"])
                        if walls[(target_pos[0], target_pos[1])] <= 0:
                            del walls[(target_pos[0], target_pos[1])]
            elif kind:
                _move_troop(troop, target_pos, walls)
                # Re-evaluate after moving to see whether target is now in range.
                kind2, target_pos2, target_dist2 = _select_target(troop, buildings, turrets, walls)
                if kind2 and target_dist2 <= troop["range"]:
                    if kind2 == "building":
                        target = next((b for b in buildings if b["alive"] and (b["x"], b["y"]) == target_pos2), None)
                        if target is not None:
                            _deal_damage(target, troop["damage"])
                            if not target["alive"]:
                                destroyed_value += target["value"]
                    elif kind2 == "turret":
                        target = next((t for t in turrets if t["alive"] and (t["x"], t["y"]) == target_pos2), None)
                        if target is not None:
                            _deal_damage(target, troop["damage"])
                    elif kind2 == "wall":
                        if (target_pos2[0], target_pos2[1]) in walls:
                            walls[(target_pos2[0], target_pos2[1])] = max(0, walls[(target_pos2[0], target_pos2[1])] - troop["damage"])
                            if walls[(target_pos2[0], target_pos2[1])] <= 0:
                                del walls[(target_pos2[0], target_pos2[1])]

        # Turrets fire at the nearest troop.
        for turret in sorted((t for t in turrets if t["alive"]), key=lambda t: (t["y"], t["x"], t["id"])):
            candidates = [
                troop for troop in troops if troop["alive"] and _dist((troop["x"], troop["y"]), (turret["x"], turret["y"])) <= turret["range"]
            ]
            if not candidates:
                continue
            target = min(
                candidates,
                key=lambda troop: (_dist((troop["x"], troop["y"]), (turret["x"], turret["y"])), troop["y"], troop["x"], troop["id"]),
            )
            troop_hp = target["hp"] - turret["damage"]
            target["hp"] = max(0, troop_hp)
            if target["hp"] == 0:
                target["alive"] = False

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
