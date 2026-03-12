"""
Official MCP client orchestrator for the Drone Rescue simulation.

This is the "AI agent" (LLM stand-in) that performs:
- real-time tool discovery
- real-time drone discovery (no hard-coded drone IDs)
- autonomous mission planning and execution
- step-by-step chain-of-thought logging BEFORE each tool call

All interactions with drones happen via the official MCP protocol by calling
tools on `mcp_drone_server.py` over STDIO.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _wrap(text: str, width: int = 72) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _print_cot(text: str) -> None:
    print("\n[CHAIN OF THOUGHT]")
    for ln in textwrap.wrap(text, width=72):
        print(f"  {ln}")


def _print_call(name: str, args: dict) -> None:
    print(f"\n[MCP TOOL CALL] {name}({json.dumps(args)})")


def _text_content(result) -> str:
    # result.content is a list[Content]. Most servers return TextContent.
    if not getattr(result, "content", None):
        return ""
    parts = []
    for c in result.content:
        if getattr(c, "type", None) == "text":
            parts.append(getattr(c, "text", ""))
    return "\n".join(parts).strip()


def _structured(result) -> dict:
    # FastMCP servers typically populate structuredContent for structured outputs.
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        if "result" in sc and isinstance(sc["result"], dict):
            return sc["result"]
        return sc
    # fall back: attempt to parse first text content as JSON
    txt = _text_content(result)
    if txt:
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                if "result" in obj and isinstance(obj["result"], dict):
                    return obj["result"]
                return obj
        except Exception:
            pass
    return {}


@dataclass
class DronePlanState:
    known_drones: List[str] = field(default_factory=list)
    drone_states: Dict[str, str] = field(default_factory=dict)  # idle|en_route|scanning|returning|charging
    drone_assignments: Dict[str, Optional[int]] = field(default_factory=dict)  # drone_id -> sector_id
    drone_targets: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    sector_status: Dict[int, str] = field(default_factory=dict)  # unscanned|in_progress|complete
    sector_remaining_wps: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    total_survivors_found: int = 0
    discoveries: List[dict] = field(default_factory=list)
    tool_calls: int = 0
    step_num: int = 0


class Orchestrator:
    def __init__(self, session: ClientSession):
        self.session = session
        self.state = DronePlanState()

    async def _call(self, name: str, args: dict, cot: str) -> dict:
        _print_cot(cot)
        _print_call(name, args)
        res = await self.session.call_tool(name=name, arguments=args)
        self.state.tool_calls += 1
        if res.isError:
            print(f"  [ERROR] {getattr(res, 'error', '') or 'tool error'}")
            txt = _text_content(res)
            if txt:
                print(f"  [DETAIL] {txt[:200]}")
            return {}
        sc = _structured(res)
        # Print a short result preview for the mission log.
        preview = json.dumps(sc, default=str)[:220] if sc else (_text_content(res)[:220] if _text_content(res) else "")
        if preview:
            print(f"  [OK] {preview}")
        else:
            print("  [OK]")
        return sc

    async def initialize(self) -> None:
        # Tool discovery (mandatory: show that we can discover tools)
        tools = await self.session.list_tools()
        tool_names = [t.name for t in tools.tools]
        print("\n" + "=" * 80)
        print("MCP CLIENT - Tool discovery")
        print("=" * 80)
        for n in tool_names:
            print(f"  - {n}")

        # Drone discovery (mandatory: no hard-coded IDs)
        fleet = await self._call(
            "discover_drones",
            {},
            "Beginning mission. I must not assume drone IDs; I will discover the active fleet via MCP.",
        )
        drones = fleet.get("drones", [])
        self.state.known_drones = [d["id"] for d in drones]
        for d in drones:
            did = d["id"]
            self.state.drone_states[did] = "idle"
            self.state.drone_assignments[did] = None
            self.state.drone_targets[did] = None

        # Initialize sector tracking from server sector info (waypoints are provided by server)
        info = await self._call(
            "get_sector_info",
            {},
            "I need the disaster zone layout and scan waypoints to plan sector assignments.",
        )
        sectors = info.get("sectors", [])
        self.state.sector_status = {s["id"]: "unscanned" for s in sectors}
        self.state.sector_remaining_wps = {
            s["id"]: [tuple(wp) for wp in s.get("waypoints", [])] for s in sectors
        }

    async def run_mission(self, max_steps: int = 250) -> None:
        print("\n" + "=" * 80)
        print("MISSION START - Drone Fleet Search & Rescue (MCP)")
        print("=" * 80)

        for _ in range(max_steps):
            self.state.step_num += 1
            print("\n" + "-" * 80)
            print(f"STEP {self.state.step_num}")
            print("-" * 80)

            # Completion check uses only MCP tools (mission state is server-side truth).
            ms = await self._call(
                "get_mission_state",
                {},
                "Checking mission status (survivors found and sector coverage) to decide whether to continue.",
            )
            survivors_found = ms.get("survivors_found", 0)
            survivors_total = ms.get("survivors_total", 0)
            self.state.total_survivors_found = survivors_found

            # Update sector completion based on coverage percentage (100% == complete).
            for s in ms.get("sectors", []):
                sid = s["id"]
                cov = s.get("coverage_pct", 0)
                if cov >= 99.9:
                    self.state.sector_status[sid] = "complete"
                elif self.state.sector_status.get(sid) != "complete":
                    self.state.sector_status[sid] = "unscanned"

            all_sectors_done = all(v == "complete" for v in self.state.sector_status.values())
            all_survivors_found = survivors_total > 0 and survivors_found >= survivors_total

            if all_sectors_done or all_survivors_found:
                print("\n" + "=" * 80)
                print("MISSION COMPLETE")
                print("=" * 80)
                print(f"Sectors complete: {sum(1 for v in self.state.sector_status.values() if v == 'complete')}/{len(self.state.sector_status)}")
                print(f"Survivors found:  {survivors_found}/{survivors_total}")
                print(f"Tool calls:       {self.state.tool_calls}")
                return

            # Per-drone control loop
            for did in list(self.state.known_drones):
                br = await self._call(
                    "get_battery_status",
                    {"drone_id": did},
                    f"Checking battery of {did} before deciding action.",
                )
                if not br:
                    continue
                bat = float(br.get("battery_pct", 0))
                critical = bool(br.get("critical", False))
                st = self.state.drone_states.get(did, "idle")

                if critical and st not in ("returning", "charging"):
                    await self._do_recall(did, bat)
                elif st == "returning":
                    await self._do_continue_return(did, bat)
                elif st == "charging":
                    await self._do_charge(did, bat)
                elif st == "idle":
                    await self._do_assign(did, bat)
                elif st == "en_route":
                    await self._do_move(did, bat)
                elif st == "scanning":
                    await self._do_scan(did, bat)

            # Advance simulation clock for parity (agents are passive).
            await self._call(
                "advance_simulation",
                {"steps": 1},
                "Advancing the simulation by one tick (drones are passive; this is for time progression/data collection).",
            )

        print("\nMission stopped: reached max_steps without completion.")

    async def _do_recall(self, did: str, bat: float) -> None:
        sid = self.state.drone_assignments.get(did)
        # Return current target to the sector waypoint queue so it can be revisited later.
        cur = self.state.drone_targets.get(did)
        if sid is not None and cur is not None:
            if cur not in self.state.sector_remaining_wps.get(sid, []):
                self.state.sector_remaining_wps.setdefault(sid, []).insert(0, cur)

        await self._call(
            "recall_to_base",
            {"drone_id": did},
            f"{did} battery is {bat:.0f}% (CRITICAL). To avoid failure, I will recall it to the nearest charging station now.",
        )
        self.state.drone_states[did] = "returning"
        self.state.drone_assignments[did] = None
        self.state.drone_targets[did] = None

    async def _do_continue_return(self, did: str, bat: float) -> None:
        r = await self._call(
            "recall_to_base",
            {"drone_id": did},
            f"{did} is returning to base with {bat:.0f}% battery. Continuing toward the nearest charging station.",
        )
        if r.get("arrived"):
            self.state.drone_states[did] = "charging"
            print(f"  [INFO] {did} arrived at charging station.")

    async def _do_charge(self, did: str, bat: float) -> None:
        if bat >= 90:
            self.state.drone_states[did] = "idle"
            self.state.drone_assignments[did] = None
            self.state.drone_targets[did] = None
            _print_cot(f"{did} battery at {bat:.0f}% (sufficient). Marking idle for reassignment.")
            return
        await self._call(
            "charge_drone",
            {"drone_id": did},
            f"{did} is at a station with {bat:.0f}% battery. Charging now to safely resume search coverage.",
        )

    async def _do_assign(self, did: str, bat: float) -> None:
        # Select any sector not complete; prefer those with remaining waypoints.
        candidates = []
        for sid, status in self.state.sector_status.items():
            if status != "complete":
                rem = len(self.state.sector_remaining_wps.get(sid, []))
                candidates.append((sid, rem))
        if not candidates:
            _print_cot(f"{did} is idle with {bat:.0f}% battery. No sectors remaining; holding position.")
            return

        # Determine drone position via MCP (no local state assumptions).
        sr = await self._call(
            "get_drone_status",
            {"drone_id": did},
            f"{did} is idle. I will query its position to assign the nearest sector with remaining waypoints.",
        )
        dpos = tuple(sr.get("pos", [0, 0]))

        # Get sector origins from MCP sector info (avoid hardcoding geometry).
        info = await self._call(
            "get_sector_info",
            {},
            "Refreshing sector metadata (origins and waypoints) to choose the nearest sector based on live layout info.",
        )
        sector_meta = {s["id"]: s for s in info.get("sectors", [])}

        def sector_center(sector: dict) -> Tuple[int, int]:
            ox, oy = sector["origin"]
            w, h = sector["size"]
            return (ox + w // 2, oy + h // 2)

        best_sid = min(
            [sid for sid, _ in candidates],
            key=lambda sid: abs(sector_center(sector_meta[sid])[0] - dpos[0]) + abs(sector_center(sector_meta[sid])[1] - dpos[1]),
        )

        rem = self.state.sector_remaining_wps.get(best_sid, [])
        if not rem:
            # Re-seed from server-provided waypoints if empty.
            rem = [tuple(wp) for wp in sector_meta[best_sid].get("waypoints", [])]
            self.state.sector_remaining_wps[best_sid] = rem
        if not rem:
            _print_cot(f"{did} cannot be assigned: sector {best_sid} has no waypoints.")
            return

        target = rem.pop(0)
        self.state.drone_assignments[did] = best_sid
        self.state.drone_targets[did] = target
        self.state.drone_states[did] = "en_route"

        sn = sector_meta[best_sid]["name"]
        dist = abs(target[0] - dpos[0]) + abs(target[1] - dpos[1])
        est = dist + 5 * 1  # rough: move + one scan
        _print_cot(
            f"{did} at {list(dpos)} with {bat:.0f}% battery. Assigning it to {sn} (next waypoint {list(target)}, distance ~{dist}). "
            f"This balances coverage while keeping expected cost (~{est}%) within battery limits."
        )

    async def _do_move(self, did: str, bat: float) -> None:
        target = self.state.drone_targets.get(did)
        if target is None:
            self.state.drone_states[did] = "idle"
            return
        r = await self._call(
            "move_to",
            {"drone_id": did, "x": int(target[0]), "y": int(target[1])},
            f"{did} is en-route to waypoint {list(target)} with {bat:.0f}% battery. Moving at maximum speed.",
        )
        if r.get("arrived"):
            self.state.drone_states[did] = "scanning"
            print(f"  [INFO] {did} arrived at waypoint {list(target)}")

    async def _do_scan(self, did: str, bat: float) -> None:
        r = await self._call(
            "thermal_scan",
            {"drone_id": did},
            f"{did} reached its scan point with {bat:.0f}% battery. Performing a thermal scan to detect survivors in the nearby radius.",
        )
        found = r.get("survivors_found", []) if r else []
        if found:
            self.state.discoveries.extend(found)
            print(f"  [DETECTED] Survivors detected by {did}: {len(found)}")
            for s in found:
                print(f"    - {s['id']} at {s['pos']}")

        sid = self.state.drone_assignments.get(did)
        if sid is None:
            self.state.drone_states[did] = "idle"
            self.state.drone_targets[did] = None
            return

        remaining = self.state.sector_remaining_wps.get(sid, [])
        if remaining:
            self.state.drone_targets[did] = remaining.pop(0)
            self.state.drone_states[did] = "en_route"
        else:
            # No more waypoints; sector will be marked complete once coverage is 100% (checked via get_mission_state).
            self.state.drone_assignments[did] = None
            self.state.drone_targets[did] = None
            self.state.drone_states[did] = "idle"


async def main() -> None:
    server_params = StdioServerParameters(command="python", args=["mcp_drone_server.py"], env=None)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            orch = Orchestrator(session)
            await orch.initialize()
            await orch.run_mission()


if __name__ == "__main__":
    asyncio.run(main())

