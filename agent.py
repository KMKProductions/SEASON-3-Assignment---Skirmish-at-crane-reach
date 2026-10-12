"""Skirmish at Crane Reach agent built from reusable behavior blocks.

Read ``environment.md`` beside this file for the rules and helpers. Every unit runs its own
``Agent`` instance, so the army never shares memory. Each unit instead derives the same army plan
from facts every ally agrees on: the roster, the battlefield, and the current round. The army
splits into a north and a south group. Each group forms up near where its units spawned, with
footmen in front, archers behind them, and cavalry on both wings, then marches as a formation
through the river passage on its own side. Once an enemy comes close, footmen meet it head on,
archers shoot from behind the footmen, and cavalry dive on enemy archers.

Every move toward a goal is routed over the whole field, so units walk around the river through
a passage instead of stalling on its bank.

An ``Agent`` is only a playbook: an ordered list of behavior blocks per unit type. The first block
that proposes an order wins. Blocks are small, stateless, and reusable across unit types.
"""

from __future__ import annotations

import heapq

from sandbox.crane import action, me, paths, roster, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation

_BAD_TERRAIN = {"marsh", "waste"}
_WATER = {"water"}
_IMPASSABLE = {"water", "void"}

# Movement point cost to enter a tile, from the rules table in environment.md.
_TERRAIN_COST = {"grass": 1, "hill": 2}
_FEATURE_COST = {"none": 0, "forest": 1, "marsh": 2, "waste": 0}
# Extra routing cost for ground that hurts on entry, so routes avoid it when they can.
_ROUTE_PENALTY = {"waste": 2}

ENGAGE_RADIUS = 6  # an enemy this close ends the march and starts the fight
_UNREACHABLE = 10**9

Hex = tuple[int, int]


def _hex(position: AxialPosition) -> Hex:
    return position["q"], position["r"]


def _hex_neighbors(position: Hex) -> list[Hex]:
    return [(position[0] + dq, position[1] + dr) for dq, dr in tile.DIRECTIONS.values()]


def _hex_distance(first: Hex, second: Hex) -> int:
    dq = first[0] - second[0]
    dr = first[1] - second[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _cube_round(fq: float, fr: float) -> Hex:
    fs = -fq - fr
    q, r, s = round(fq), round(fr), round(fs)
    dq, dr, ds = abs(q - fq), abs(r - fr), abs(s - fs)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return q, r


def _hex_line(start: Hex, end: Hex) -> list[Hex]:
    """Return every tile on the straight hex line from start to end, both included."""
    steps = _hex_distance(start, end)
    if steps == 0:
        return [start]
    # The tiny nudge keeps rounding consistent when the line runs exactly along a tile edge.
    return [
        _cube_round(
            start[0] + (end[0] - start[0]) * i / steps + 1e-6,
            start[1] + (end[1] - start[1]) * i / steps + 1e-6,
        )
        for i in range(steps + 1)
    ]


# --------------------------------------------------------------------------------------------
# Whole-field routing
# --------------------------------------------------------------------------------------------


class TerrainMap:
    """Travel costs over the whole field for one match, so routes go around the river.

    The action mask only covers paths of up to four steps, which cannot see a passage further
    down the bank. ``field`` runs one cheapest-route search from a goal over every passable tile,
    and a unit then picks the legal path whose landing has the cheapest remaining route.
    """

    CACHE_LIMIT = 256  # goals whose cost fields are kept for reuse within the match
    OCCUPIED_PENALTY = 5  # extra cost for a tile a unit stands on: it may move, but likely not

    def __init__(self, observation: SkirmishObservation) -> None:
        battlefield = observation["observation"]["battlefield"]
        view = {"observation": {"battlefield": battlefield}}
        side = battlefield["side"]
        self.entry_cost: dict[Hex, int] = {}
        self._route_cost: dict[Hex, int] = {}
        for q in range(side):
            for r in range(side):
                ground = tile.terrain_at(view, {"q": q, "r": r})
                if ground["terrain"] in _IMPASSABLE:
                    continue
                cost = _TERRAIN_COST.get(ground["terrain"], 1) + _FEATURE_COST.get(ground["feature"], 0)
                self.entry_cost[(q, r)] = cost
                self._route_cost[(q, r)] = cost + _ROUTE_PENALTY.get(ground["feature"], 0)
        self._fields: dict[Hex, dict[Hex, int]] = {}

    def passable(self, position: Hex) -> bool:
        return position in self.entry_cost

    def field(self, goal: Hex, occupied: frozenset[Hex] = frozenset()) -> dict[Hex, int]:
        """Return the cheapest route cost from every reachable tile to ``goal``.

        Tiles in ``occupied`` cost extra, so routes flow around units standing in a narrow
        passage. Those fields depend on the moment, so only unoccupied ones are cached.
        """
        if not occupied:
            cached = self._fields.get(goal)
            if cached is not None:
                return cached
            if len(self._fields) >= self.CACHE_LIMIT:
                self._fields.clear()

        # An impassable goal, such as water, is approached from its passable neighbors.
        seeds = [goal] if self.passable(goal) else [n for n in _hex_neighbors(goal) if self.passable(n)]
        costs = {seed: 0 for seed in seeds}
        frontier = [(0, seed) for seed in seeds]
        while frontier:
            cost, position = heapq.heappop(frontier)
            if cost > costs[position]:
                continue
            # Searching backward from the goal: a neighbor reaches here by entering this tile.
            through = cost + self._route_cost[position]
            if position in occupied:
                through += self.OCCUPIED_PENALTY
            for neighbor in _hex_neighbors(position):
                if self.passable(neighbor) and through < costs.get(neighbor, _UNREACHABLE):
                    costs[neighbor] = through
                    heapq.heappush(frontier, (through, neighbor))

        if not occupied:
            self._fields[goal] = costs
        return costs

    def path_cost(self, start: Hex, path_id: int) -> int:
        """Return the movement points a path spends, entering each tile along it."""
        q, r = start
        total = 0
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            q, r = q + dq, r + dr
            total += self.entry_cost.get((q, r), _UNREACHABLE)
        return total


# --------------------------------------------------------------------------------------------
# Movement primitives
# --------------------------------------------------------------------------------------------


class Navigator:
    """Path choices for one activation. Every path it returns comes from the legal mask."""

    def __init__(self, observation: SkirmishObservation, terrain: TerrainMap) -> None:
        self.observation = observation
        self.terrain = terrain
        self.here = me.position(observation)
        self._occupied = frozenset(_hex(unit["position"]) for unit in observation["observation"]["visible_units"])

    def field(self, goal: Hex) -> dict[Hex, int]:
        """Return route costs to goal that steer around the units this unit can see."""
        return self.terrain.field(goal, self._occupied - {goal})

    def path_has_bad_terrain(self, path_id: int) -> bool:
        """True when the path enters water, marsh, or waste terrain."""
        position = self.here
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            position = {"q": position["q"] + dq, "r": position["r"] + dr}
            terrain = tile.terrain_at(self.observation, position)
            if terrain["terrain"] in _BAD_TERRAIN | _WATER or terrain["feature"] in _BAD_TERRAIN | _WATER:
                return True
        return False

    def best_path(self, key, max_cost: int | None = None) -> int:
        """Return the legal path whose landing tile has the lowest ``key``, or 0 to stay.

        Standing still wins ties, and shorter paths come first in the mask, so a unit never walks
        further than the gain needs. ``max_cost`` limits the movement points the path may spend.
        """
        here = _hex(self.here)
        best_path, best_key = 0, key(here)
        for path_id in action.legal_paths(self.observation):
            if path_id == 0:
                continue
            if max_cost is not None and self.terrain.path_cost(here, path_id) > max_cost:
                continue
            candidate = key(_hex(tile.at_path_end(self.here, path_id)))
            if candidate < best_key:
                best_path, best_key = path_id, candidate
        return best_path

    def step_toward(self, goal: AxialPosition, max_cost: int | None = None) -> int:
        """Return the legal path that gets furthest along the real route to goal, or 0.

        The route comes from the whole field, so a goal across the river leads through the
        nearest passage instead of to the bank opposite it.
        """
        target = _hex(goal)
        field = self.field(target)
        return self.best_path(
            lambda landing: (field.get(landing, _UNREACHABLE), _hex_distance(landing, target)), max_cost
        )

    def travel_cost(self, start: AxialPosition, goal: AxialPosition) -> int:
        """Return the movement points the cheapest route from start to goal spends."""
        return self.terrain.field(_hex(goal)).get(_hex(start), _UNREACHABLE)

    def path_away(self, threat: AxialPosition) -> int:
        """Return the legal path that maximizes distance from a threat, avoiding bad terrain."""
        best_path = 0
        best_distance = tile.distance(self.here, threat)

        for path_id in action.legal_paths(self.observation):
            if path_id == 0 or self.path_has_bad_terrain(path_id):
                continue
            landing_distance = tile.distance(tile.at_path_end(self.here, path_id), threat)
            if landing_distance > best_distance:
                best_path, best_distance = path_id, landing_distance

        return best_path

    def flank_goal(self, enemy: AxialPosition) -> AxialPosition:
        """Choose the nearer of the two side tiles next to an enemy."""
        enemy_sides = tile.neighbors(enemy)
        side_tiles = (enemy_sides[1], enemy_sides[3])
        return min(side_tiles, key=lambda side: tile.distance(self.here, side))


# --------------------------------------------------------------------------------------------
# Army plan: two groups, their formations, and their march routes
# --------------------------------------------------------------------------------------------


class GroupPlan:
    """One group's members, rally point, and march, in red's frame.

    The march has two legs. The first runs from the rally through this side's river passage to a
    join point beside the other group. The group waits there until ``join_round``, when both
    groups should have arrived, then the second leg advances the joined line on enemy ground.
    """

    def __init__(self, members: dict[str, list[int]], rally: Hex, ready_round: int) -> None:
        self.members = members  # unit type -> roster indices, in slot order
        self.rally = rally
        self.ready_round = ready_round  # the group counts as formed after this round
        # Each leg is a route, one tile per entry, and a schedule: the route index the anchor
        # has reached after each marching round.
        self.route: list[Hex] = [rally]
        self.schedule: list[int] = [0]
        self.join_round = ready_round  # the round both groups should stand side by side
        self.advance_route: list[Hex] = [rally]
        self.advance_schedule: list[int] = [0]


class ArmyPlan:
    """Splits the army into a north and a south group and plans each group's formation and march.

    Every unit builds this from shared match knowledge only, so all allies agree on it without
    messaging. The plan lives in red's frame, where the enemy lies toward +q, and is mirrored
    through the field center for blue. Within a formation the footmen hold the front column, the
    archers stand behind them, and the cavalry fill out both wings, with open ground between.
    """

    FILE_SPACING = 2  # rows between neighbors in a line: 2 leaves one open tile between them
    RANK_SPACING = 2  # columns between the footman line and the archers: 2 leaves one open
    RALLY_SEARCH = 6  # how far a rally point may shift from its spawn centroid to fit the group
    JOIN_DEPTH = 3  # columns past the river seam where the two groups meet up again
    REGROUP_SLACK = 2  # extra rounds allowed for forming up, for terrain and traffic
    SCOUT_AFTER = 10  # rounds past the march's end, without contact, before units scout freely

    def __init__(self, observation: SkirmishObservation, terrain: TerrainMap) -> None:
        self.terrain = terrain
        self.side = observation["observation"]["battlefield"]["side"]
        self.extent = (self.side - 1) // 2
        self.is_red = me.side(observation) == "red"
        # The march moves at the pace of the slowest line troops, spending their full points.
        self.pace = min(units.STATS[kind].movement_points for kind in ("footman", "archer"))

        roster_order = [(entry["type"], int(entry["unit_id"].rsplit("_", 1)[1])) for entry in roster.allies(observation)]
        spawns = dict(zip(roster_order, self._spawns(len(roster_order))))
        own_key = (me.unit_type(observation), int(me.unit_id(observation).rsplit("_", 1)[1]))
        if spawns.get(own_key) != self._to_red(me.position(observation)):
            spawns = None  # the spawn rule changed, so plan without knowing where allies start

        self.groups: list[GroupPlan] = []
        # Tiles the first group's rally formation holds, plus a one-tile buffer around them.
        reserved: set[Hex] = set()
        for north in (True, False):
            group = self._plan_group(roster_order, spawns, north, reserved)
            self._reserve(reserved, group.members, group.rally)
            self.groups.append(group)
        self._plan_marches(self._passage_rows())

        self._group_of = {
            (kind, index): (group_id, slot_index)
            for group_id, group in enumerate(self.groups)
            for kind, indices in group.members.items()
            for slot_index, index in enumerate(indices)
        }

    # -- Lookups used by Context ----------------------------------------------------------------

    def group_of(self, kind: str, index: int) -> int:
        return self._group_of[(kind, index)][0]

    def ready_round(self, group_id: int) -> int:
        return self.groups[group_id].ready_round

    def slot(self, kind: str, index: int, round_number: int) -> AxialPosition:
        """Return this unit's slot in its group's formation for the round, in board coordinates."""
        group_id, slot_index = self._group_of[(kind, index)]
        group = self.groups[group_id]
        q, r = self._formation(group.members, self._anchor(group, round_number))[(kind, slot_index)]
        return self._to_board(q, r)

    def scouting(self, group_id: int, round_number: int, last_contact: int | None) -> bool:
        """True once the group has finished its march and found nobody for a while."""
        group = self.groups[group_id]
        march_end = group.join_round + len(group.advance_schedule) - 1
        if round_number <= march_end + self.SCOUT_AFTER:
            return False
        return last_contact is None or round_number - last_contact > self.SCOUT_AFTER

    # -- Planning ---------------------------------------------------------------------------------

    def _plan_group(
        self,
        roster_order: list[tuple[str, int]],
        spawns: dict | None,
        north: bool,
        reserved: set[Hex],
    ) -> GroupPlan:
        by_kind: dict[str, list[int]] = {}
        for kind, index in roster_order:
            by_kind.setdefault(kind, []).append(index)

        members: dict[str, list[int]] = {}
        for kind, indices in by_kind.items():
            if spawns is not None:
                # Units spawned nearer the north edge (smaller r) join the north group.
                ordered = sorted(indices, key=lambda index: (spawns[(kind, index)][1], index))
                half = (len(ordered) + 1) // 2
                chosen = ordered[:half] if north else ordered[half:]
            else:
                chosen = [index for index in indices if (index % 2 == 0) == north]
            members[kind] = chosen

        own = [spawns[(kind, index)] for kind, indices in members.items() for index in indices] if spawns else []
        if own:
            centroid = self._onto_field(*_cube_round(sum(q for q, _ in own) / len(own), sum(r for _, r in own) / len(own)))
        else:
            row = self.extent // 2 if north else self.extent + self.extent // 2
            centroid = self._onto_field(max(1, self.extent - 5), row)
        rally = self._choose_rally(members, centroid, reserved)

        if own:
            slowest = 0
            rally_slots = self._formation(members, rally)
            for kind, indices in members.items():
                speed = units.STATS[kind].movement_points
                for slot_index, index in enumerate(indices):
                    walk = _hex_distance(spawns[(kind, index)], rally_slots[(kind, slot_index)])
                    slowest = max(slowest, -(-walk // speed))
            ready_round = slowest + self.REGROUP_SLACK
        else:
            ready_round = 8
        return GroupPlan(members, rally, ready_round)

    def _plan_marches(self, passages: list[int]) -> None:
        """Route each group through its own passage to a join point beside the other group, then
        advance the joined line toward where the enemy army started.

        The north group joins just north of the middle row and the south group just south of it,
        so the two formations stand side by side as one long line.
        """
        active = [group for group in self.groups if any(group.members.values())]
        if not active:
            return

        # Rows from a formation's anchor to its furthest slot, so the two groups never overlap.
        reach = max(
            abs(self._ideal_slot(group.members, kind, slot_index, (0, 0))[1])
            for group in active
            for kind, indices in group.members.items()
            for slot_index in range(len(indices))
        )
        offset = reach + 1
        # The enemy started at the mirror of our own rally points.
        mean_q = sum(group.rally[0] for group in active) / len(active)
        mean_r = sum(group.rally[1] for group in active) / len(active)
        enemy_q, enemy_r = _cube_round(self.side - 1 - mean_q, self.side - 1 - mean_r)

        join_reserved: set[Hex] = set()
        final_reserved: set[Hex] = set()
        for group, sign, passage_r in zip(self.groups, (-1, 1), (passages[0], passages[-1])):
            if group not in active:
                continue
            join = self._choose_rally(
                group.members, self._onto_field(self.extent + self.JOIN_DEPTH, self.extent + sign * offset), join_reserved
            )
            self._reserve(join_reserved, group.members, join)
            final = self._choose_rally(group.members, self._onto_field(enemy_q, enemy_r + sign * offset), final_reserved)
            self._reserve(final_reserved, group.members, final)

            group.route = self._route(
                [
                    group.rally,
                    self._onto_field(self.extent - 2, passage_r),
                    self._onto_field(self.extent + 2, passage_r),
                    join,
                ]
            )
            group.schedule = self._march_schedule(group.route)
            group.advance_route = self._route([join, final])
            group.advance_schedule = self._march_schedule(group.advance_route)

        # The first group to arrive waits at its join point for the other one.
        join_round = max(group.ready_round + len(group.schedule) - 1 for group in active)
        for group in active:
            group.join_round = join_round

    def _route(self, waypoints: list[Hex]) -> list[Hex]:
        """Join straight hex lines between waypoints into one route, skipping impassable tiles."""
        route = [waypoints[0]]
        for start, end in zip(waypoints, waypoints[1:]):
            for position in _hex_line(start, end)[1:]:
                if position != route[-1] and self.terrain.passable(position):
                    route.append(position)
        return route

    def _reserve(self, reserved: set[Hex], members: dict[str, list[int]], anchor: Hex) -> None:
        """Mark a formation's tiles, plus a one-tile buffer, as taken for later groups."""
        for position in self._formation(members, anchor).values():
            reserved.add(position)
            reserved.update(_hex_neighbors(position))

    def _choose_rally(self, members: dict[str, list[int]], centroid: Hex, reserved: set[Hex]) -> Hex:
        """Return the tile nearest the centroid where the whole spaced-out formation fits.

        Within each ring, tiles further from the river come first.
        """
        candidates = sorted(self.terrain.entry_cost, key=lambda position: (_hex_distance(position, centroid), position))
        for anchor in candidates:
            if _hex_distance(anchor, centroid) > self.RALLY_SEARCH:
                break
            footprint = [
                self._ideal_slot(members, kind, slot_index, anchor)
                for kind, indices in members.items()
                for slot_index in range(len(indices))
            ]
            if len(set(footprint)) == len(footprint) and all(
                self.terrain.passable(position) and position not in reserved for position in footprint
            ):
                return anchor
        return centroid

    def _march_schedule(self, route: list[Hex]) -> list[int]:
        """Walk the route the way a line troop would, spending ``pace`` movement points a round.

        As in the rules, the first step of a round is always allowed and later steps need enough
        points left. Entry ``k`` is the route index the anchor reaches after ``k`` rounds.
        """
        schedule = [0]
        index = 0
        while index < len(route) - 1:
            points = self.pace
            moved = False
            while index < len(route) - 1:
                cost = self.terrain.entry_cost.get(route[index + 1], _UNREACHABLE)
                if moved and cost > points:
                    break
                points -= cost
                index += 1
                moved = True
                if points <= 0:
                    break
            schedule.append(index)
        return schedule

    def _anchor(self, group: GroupPlan, round_number: int) -> Hex:
        if round_number <= group.join_round:
            marched = max(0, round_number - group.ready_round)
            return group.route[group.schedule[min(len(group.schedule) - 1, marched)]]
        advanced = round_number - group.join_round
        return group.advance_route[group.advance_schedule[min(len(group.advance_schedule) - 1, advanced)]]

    def _formation(self, members: dict[str, list[int]], anchor: Hex) -> dict[tuple[str, int], Hex]:
        """Return every slot of a group's formation around ``anchor``, keyed by (type, slot index).

        Slots are placed footmen first, then archers, then cavalry, and a slot never takes a tile
        an earlier slot already holds.
        """
        taken: set[Hex] = set()
        slots: dict[tuple[str, int], Hex] = {}
        for kind in ("footman", "archer", "cavalry"):
            for slot_index in range(len(members.get(kind, []))):
                q, r = self._ideal_slot(members, kind, slot_index, anchor)
                position = self._settle(q, r, anchor[1], taken)
                taken.add(position)
                slots[(kind, slot_index)] = position
        return slots

    def _ideal_slot(self, members: dict[str, list[int]], kind: str, slot_index: int, anchor: Hex) -> Hex:
        front, middle = anchor
        spacing = self.FILE_SPACING

        def row(position: int, count: int) -> int:
            # Rows spread ``spacing`` apart and centered on the anchor's row.
            return middle + position * spacing - ((count - 1) * spacing) // 2

        footmen = len(members.get("footman", []))
        if kind == "footman":
            return front, row(slot_index, footmen)
        if kind == "archer":
            return front - self.RANK_SPACING, row(slot_index, len(members.get("archer", [])))
        # Cavalry alternates wings beyond each end of the footman line; later riders stand behind.
        depth = slot_index // 2
        if slot_index % 2 == 0:
            wing_row = row(0, footmen) - spacing
        else:
            wing_row = row(max(footmen - 1, 0), footmen) + spacing
        return front - depth * self.RANK_SPACING, wing_row

    def _settle(self, q: int, r: int, toward_r: int, taken: set[Hex] = frozenset()) -> Hex:
        """Slide a slot toward the anchor row, then back toward home, until it is on free ground.

        That keeps slots on the field near an edge, and squeezes a formation through a river
        passage, queueing the rest single file behind it.
        """
        start = (q, r)
        for _ in range(4 * self.side):
            if self.terrain.passable((q, r)) and (q, r) not in taken:
                return q, r
            if r != toward_r:
                r += 1 if r < toward_r else -1
            elif q > 0:
                q -= 1
            else:
                break
        return start if self.terrain.passable(start) else (q, r)

    def _onto_field(self, q: int, r: int) -> Hex:
        """Slide a waypoint toward the middle row until it is on passable ground."""
        return self._settle(q, r, self.extent)

    def _spawns(self, count: int) -> list[Hex]:
        """Rebuild red's spawn tiles exactly as the battlefield generator lays them out."""
        left = [(q, r) for q in range(self.extent) for r in range(self.side) if self.terrain.passable((q, r))]
        if len(left) < count:
            return []
        return [left[index * len(left) // count] for index in range(count)]

    def _passage_rows(self) -> list[int]:
        """Return the middle row of each gap in the river seam, from north to south."""
        runs: list[list[int]] = []
        for r in range(self.side):
            if self.terrain.passable((self.extent, r)):
                if runs and runs[-1][-1] == r - 1:
                    runs[-1].append(r)
                else:
                    runs.append([r])
        if not runs:
            return [self.extent]
        return [run[len(run) // 2] for run in runs]

    def _to_red(self, position: AxialPosition) -> Hex:
        if self.is_red:
            return position["q"], position["r"]
        return self.side - 1 - position["q"], self.side - 1 - position["r"]

    def _to_board(self, q: int, r: int) -> AxialPosition:
        if self.is_red:
            return {"q": q, "r": r}
        return {"q": self.side - 1 - q, "r": self.side - 1 - r}


# --------------------------------------------------------------------------------------------
# Per-activation context
# --------------------------------------------------------------------------------------------


class Context:
    """Everything a behavior block needs, computed once per activation."""

    def __init__(
        self, observation: SkirmishObservation, plan: ArmyPlan, terrain: TerrainMap, last_contact: int | None
    ) -> None:
        self.obs = observation
        self.here = me.position(observation)
        self.kind = me.unit_type(observation)
        self.index = int(me.unit_id(observation).rsplit("_", 1)[1])
        self.round = observation["observation"]["round"]
        self.enemies = visible.enemies(observation)
        self.allies = visible.allies(observation)
        self.nearest = (
            min(self.enemies, key=lambda enemy: tile.distance(self.here, enemy["position"])) if self.enemies else None
        )
        # The nearest enemy close enough to fight. Enemies seen farther off do not stop the march.
        self.threat = self.nearest if self.nearest is not None and self.nearest_distance <= ENGAGE_RADIUS else None
        self.nav = Navigator(observation, terrain)
        self.plan = plan
        self.group = plan.group_of(self.kind, self.index)
        self.last_contact = last_contact

    @property
    def nearest_distance(self) -> int | None:
        return tile.distance(self.here, self.nearest["position"]) if self.nearest is not None else None

    def ready(self) -> bool:
        """True once this unit's group has had its time to form up."""
        return self.round > self.plan.ready_round(self.group)

    def scouting(self) -> bool:
        return self.plan.scouting(self.group, self.round, self.last_contact)

    def slot(self) -> AxialPosition:
        return self.plan.slot(self.kind, self.index, self.round)

    def allied(self, kind: str) -> list[AxialPosition]:
        """Positions of the visible allies of one unit type."""
        return [ally["position"] for ally in self.allies if ally["type"] == kind]

    def order(self, path_id: int, target: dict | None = None) -> SkirmishAction:
        """Build an order along ``path_id`` naming ``target``, or the nearest visible enemy."""
        target = target or self.nearest
        if target is None:
            return action.move(path_id) if path_id else action.stay()
        if path_id:
            return action.move(path_id, target["unit_id"], self.obs)
        return action.stay(target["unit_id"], self.obs)


# --------------------------------------------------------------------------------------------
# Behavior blocks
# --------------------------------------------------------------------------------------------


class Behavior:
    """A reusable decision rule. ``propose`` returns an order, or None when it does not apply."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        raise NotImplementedError


class AfterRegroup(Behavior):
    """Gate a combat block: it acts only once the group has formed and an enemy is close."""

    def __init__(self, block: Behavior) -> None:
        self.block = block

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.threat is None or not ctx.ready():
            return None
        return self.block.propose(ctx)


class KiteFromThreat(Behavior):
    """Back away from an enemy that is too close, unless the river keeps it from reaching us."""

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None or ctx.nearest_distance > self.radius:
            return None
        # An enemy that needs more than two activations to walk around to us is no danger yet.
        reach = 2 * units.STATS[ctx.nearest["type"]].movement_points
        if ctx.nav.travel_cost(ctx.nearest["position"], ctx.here) > reach:
            return None
        return ctx.order(ctx.nav.path_away(ctx.nearest["position"]))


class FightHeadOn(Behavior):
    """Footman: close straight on the threat, shoulder to shoulder with other footmen."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        target = ctx.threat
        field = ctx.nav.field(_hex(target["position"]))
        footmen = [_hex(ally) for ally in ctx.allied("footman")]

        def key(landing: Hex) -> tuple[int, int]:
            # Shortest real route first, then more adjacent footmen for the shield wall.
            buddies = sum(1 for ally in footmen if _hex_distance(ally, landing) == 1)
            return field.get(landing, _UNREACHABLE), -buddies

        return ctx.order(ctx.nav.best_path(key), target)


class ShootFromBehindFootmen(Behavior):
    """Archer: shoot the threat from a tile screened by a footman standing closer to it."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        target = ctx.threat
        goal = target["position"]
        attack_range = units.STATS[ctx.kind].attack_range
        enemies = [enemy["position"] for enemy in ctx.enemies]
        footmen = ctx.allied("footman")

        best_path, best_key = None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id and ctx.nav.path_has_bad_terrain(path_id):
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            shot = tile.distance(landing, goal)
            if shot > attack_range:
                continue
            closest_enemy = min(tile.distance(landing, enemy) for enemy in enemies)
            screened = any(tile.distance(landing, ally) <= 2 and tile.distance(ally, goal) < shot for ally in footmen)
            # Screened and out of reach first, then as far from enemies as the shot allows,
            # then the shortest walk.
            candidate = (screened, closest_enemy > 2, closest_enemy, -len(paths.decode(path_id)))
            if best_key is None or candidate > best_key:
                best_path, best_key = path_id, candidate

        if best_path is None:
            return None  # no tile reaches the threat this turn, so keep moving with the group
        return ctx.order(best_path, target)


class HuntArchers(Behavior):
    """Cavalry: dive on a nearby enemy archer, or flank the threat, charging when possible."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        archers = [
            enemy
            for enemy in ctx.enemies
            if enemy["type"] == "archer" and tile.distance(ctx.here, enemy["position"]) <= ENGAGE_RADIUS
        ]
        target = min(archers, key=lambda enemy: tile.distance(ctx.here, enemy["position"])) if archers else ctx.threat
        goal = target["position"]

        best_path, best_key = None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id and ctx.nav.path_has_bad_terrain(path_id):
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            if tile.distance(landing, goal) != 1:
                continue
            # A walk ending 3 or more tiles from the start earns the charge bonus.
            charging = tile.distance(ctx.here, landing) >= 3
            crowding = sum(
                1 for enemy in ctx.enemies if enemy is not target and tile.distance(landing, enemy["position"]) <= 1
            )
            candidate = (charging, -crowding)
            if best_key is None or candidate > best_key:
                best_path, best_key = path_id, candidate

        if best_path is None:
            # Out of reach this turn: ride the real route, around the river if needed.
            best_path = ctx.nav.step_toward(ctx.nav.flank_goal(goal))
        return ctx.order(best_path, target)


class ChaseNearest(Behavior):
    """Close on the nearest visible enemy along the real route, around the river if needed."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None:
            return None
        return ctx.order(ctx.nav.step_toward(ctx.nearest["position"]))


class KeepFormation(Behavior):
    """Walk to this unit's slot in its group's formation, or hold it once there.

    ``march_cost`` caps the movement points spent per activation once the group has formed, so a
    fast unit keeps pace with the line instead of running ahead of it.
    """

    def __init__(self, march_cost: int | None = None) -> None:
        self.march_cost = march_cost

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.scouting():
            return None
        slot = ctx.slot()
        if ctx.here == slot:
            return ctx.order(0)
        cap = self.march_cost if ctx.ready() else None
        return ctx.order(ctx.nav.step_toward(slot, cap))


class SweepField(Behavior):
    """With no enemy in sight, tour the field's six outer regions in turn to flush out stragglers.

    Each group starts on the opposite side of the tour, and every unit moves on to the next
    region after ``dwell`` rounds, so a unit hiding in a corner cannot stall the match.
    """

    def __init__(self, dwell: int = 12, inset: int = 3) -> None:
        self.dwell = dwell
        self.inset = inset

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.enemies:
            return None
        center = tile.at_center(ctx.obs)
        reach = center["q"] - self.inset
        regions = [{"q": center["q"] + dq * reach, "r": center["r"] + dr * reach} for dq, dr in tile.DIRECTIONS.values()]
        goal = regions[(ctx.round // self.dwell + 3 * ctx.group) % len(regions)]
        path = ctx.nav.step_toward(goal)
        return action.move(path) if path else None


class HoldPosition(Behavior):
    """Last resort: stay put and strike whatever is in range."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        return ctx.order(0)


# --------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------


class Agent:
    """Runs the first behavior block in this unit type's playbook that proposes an order."""

    # Blocks are stateless, so every unit can share these instances. Until its group has formed
    # and an enemy comes within ENGAGE_RADIUS, a unit keeps its slot in the marching formation.
    # KeepFormation still names the nearest enemy, so units strike anything in range on the way.
    PLAYBOOK: dict[str, list[Behavior]] = {
        "footman": [AfterRegroup(FightHeadOn()), KeepFormation(), ChaseNearest(), SweepField(), HoldPosition()],
        "archer": [
            KiteFromThreat(2),
            AfterRegroup(ShootFromBehindFootmen()),
            KeepFormation(),
            ChaseNearest(),
            SweepField(),
            HoldPosition(),
        ],
        # Cavalry marches on two movement points so it stays level with the line, and only
        # spends its full four when it attacks.
        "cavalry": [
            AfterRegroup(HuntArchers()),
            KeepFormation(march_cost=2),
            ChaseNearest(),
            SweepField(),
            HoldPosition(),
        ],
    }

    def __init__(self) -> None:
        self.terrain: TerrainMap | None = None
        self.plan: ArmyPlan | None = None
        self.last_contact: int | None = None

    def reset(self, seed, observation) -> None:
        # The routing map and the plan depend only on constant match knowledge and the opening
        # position, so both are built once here, outside the decision clock.
        self.terrain = TerrainMap(observation)
        self.plan = ArmyPlan(observation, self.terrain)
        self.last_contact = None

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        if self.plan is None:
            self.terrain = TerrainMap(observation)
            self.plan = ArmyPlan(observation, self.terrain)

        ctx = Context(observation, self.plan, self.terrain, self.last_contact)
        if ctx.threat is not None:
            self.last_contact = ctx.round
            ctx.last_contact = ctx.round

        for block in self.PLAYBOOK.get(ctx.kind, [HoldPosition()]):
            order = block.propose(ctx)
            if order is not None:
                return order
        return action.stay()

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # Optional: messaging. Season settings enable it from Season 3 onward. When enabled, chat runs
    # after a unit chooses its order and receives messages that arrived since its previous
    # activation. Return each message with a recipient and text. Use None to broadcast to both
    # sides, or a player id such as "player_2", not a unit id, to send directly to one ally. The
    # rosters in the observation map each player to its unit. By default, text is limited to 200
    # characters.
    # A direct message reaches its allied unit at its next activation, after that unit chooses its
    # own order. Every message is recorded and shown in replays, so nothing you send is ever secret.
    # Return nothing to stay silent.
    #
    # def chat(self, inbox: list[dict]) -> list[dict] | None:
    #     ...
