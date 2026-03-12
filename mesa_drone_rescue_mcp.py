"""
Drone Fleet Search & Rescue — Mesa Simulation Core
=================================================

This module contains the **simulation-only** components for a drone fleet
search-and-rescue scenario:

- Mesa agents (`DroneAgent`, `SurvivorAgent`, `ChargingStationAgent`, etc.)
- The Mesa model (`DroneRescueModel`)
- Visual portrayal + optional Mesa UI server (for local visualization)

The **official MCP (Model Context Protocol)** server and the orchestration
agent live in separate scripts (see `mcp_drone_server.py` and
`mcp_drone_orchestrator.py`). This keeps the simulation core clean and allows
all Agent↔Drone communication to happen via real MCP tool calls.
"""

import random
from dataclasses import dataclass
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

try:
    from google import genai  # type: ignore
except Exception:  # optional dependency at runtime
    genai = None

# Load environment variables from .env when available (e.g., GEMINI_API_KEY)
if load_dotenv is not None:
    try:
        load_dotenv()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS — sectors, colours, bases
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_DEFS = {
    1: {"name": "Sector 1 (NW)", "origin": (0, 8),  "size": (8, 8)},
    2: {"name": "Sector 2 (N)",  "origin": (8, 8),  "size": (8, 8)},
    3: {"name": "Sector 3 (NE)", "origin": (16, 8), "size": (8, 8)},
    4: {"name": "Sector 4 (SW)", "origin": (0, 0),  "size": (8, 8)},
    5: {"name": "Sector 5 (S)",  "origin": (8, 0),  "size": (8, 8)},
    6: {"name": "Sector 6 (SE)", "origin": (16, 0), "size": (8, 8)},
}

SECTOR_COLORS = {
    1: ("#e8f0fe", "#b3cde8"),
    2: ("#e6f4ea", "#a5d6b7"),
    3: ("#fef7e0", "#f5e0a0"),
    4: ("#fce8e6", "#f2b8b5"),
    5: ("#f3e8fd", "#d4b5f0"),
    6: ("#fff3e0", "#f5cfa0"),
}

BASE_POSITIONS = [(0, 0), (23, 0), (0, 15), (23, 15)]

BATTERY_COST_MOVE = 1       # per cell moved
BATTERY_COST_SCAN = 5       # per thermal_scan call
BATTERY_CHARGE_RATE = 25    # per charge_drone call
BATTERY_CRITICAL = 20       # recall threshold
BATTERY_FULL = 90           # considered fully charged
SCAN_RADIUS = 2


def _sector_waypoints(origin: Tuple[int, int], size: Tuple[int, int]):
    sx, sy = origin
    return [(sx + 2, sy + 2), (sx + 5, sy + 2),
            (sx + 2, sy + 5), (sx + 5, sy + 5)]


def pos_to_sector_id(x: int, y: int):
    for sid, s in SECTOR_DEFS.items():
        ox, oy = s["origin"]
        w, h = s["size"]
        if ox <= x < ox + w and oy <= y < oy + h:
            return sid
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIOS (3 presets)
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A: Center quake (clustered)": {
        "seed": 1337,
        "survivor_positions": [
            (11, 7), (12, 7), (13, 7),
            (11, 8), (12, 8), (13, 8),
            (10, 6), (14, 6), (10, 9), (14, 9),
            (9, 8), (15, 8),
        ],
    },
    "B: Two hotspots": {
        "seed": 2026,
        "survivor_positions": [
            (4, 12), (5, 12), (4, 13), (6, 11), (3, 11), (6, 13),
            (18, 3), (19, 3), (18, 4), (20, 2), (17, 2), (20, 4),
        ],
    },
    "C: Perimeter scattered": {
        "seed": 7,
        "survivor_positions": [
            (1, 1), (22, 1), (1, 14), (22, 14),
            (6, 1), (17, 1), (6, 14), (17, 14),
            (1, 6), (22, 6), (1, 9), (22, 9),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  "IN-UI MCP" tools + simple reasoning agent
#  (This is what makes drones move in the UI.)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str


class InUiToolServer:
    """Tiny tool registry so the AI can act like MCP, but in-process."""

    def __init__(self, model: "DroneRescueModel"):
        self.model = model

    def discover_drones(self) -> Dict[str, Any]:
        drones = [a for a in self.model.schedule.agents if isinstance(a, DroneAgent)]
        return {
            "drones": [
                {"id": d.unique_id, "pos": list(d.pos), "battery": d.battery, "disabled": d.disabled}
                for d in drones
            ],
            "count": len(drones),
        }

    def get_mission_state(self) -> Dict[str, Any]:
        survivors = [a for a in self.model.schedule.agents if isinstance(a, SurvivorAgent)]
        found = [s for s in survivors if s.detected]
        sectors = []
        for sid, sdef in SECTOR_DEFS.items():
            ox, oy = sdef["origin"]
            w, h = sdef["size"]
            total = w * h
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if self.model.tile_map.get((xi, yi)) and self.model.tile_map[(xi, yi)].scanned
            )
            sectors.append(
                {
                    "id": sid,
                    "name": sdef["name"],
                    "origin": [ox, oy],
                    "size": [w, h],
                    "coverage_pct": round(scanned / max(1, total) * 100, 1),
                    "waypoints": [list(wp) for wp in _sector_waypoints(sdef["origin"], sdef["size"])],
                }
            )
        return {
            "survivors_found": len(found),
            "survivors_total": len(survivors),
            "sectors": sectors,
        }

    def move_to(self, drone_id: str, x: int, y: int) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.disabled:
            return {"moved": False, "reason": "drone disabled"}
        if d.battery <= 0:
            d.disabled = True
            return {"moved": False, "reason": "battery depleted"}

        tx = max(0, min(self.model.grid.width - 1, int(x)))
        ty = max(0, min(self.model.grid.height - 1, int(y)))
        cx, cy = d.pos
        steps = 0
        while steps < d.speed and (cx, cy) != (tx, ty) and d.battery > 0:
            dx = (1 if tx > cx else -1) if tx != cx else 0
            dy = (1 if ty > cy else -1) if ty != cy else 0
            cx += dx
            cy += dy
            d.battery = max(0, d.battery - BATTERY_COST_MOVE)
            steps += 1
        self.model.grid.move_agent(d, (cx, cy))
        if d.battery <= 0:
            d.disabled = True
        return {"new_pos": [cx, cy], "battery": d.battery, "arrived": (cx, cy) == (tx, ty), "steps_taken": steps}

    def thermal_scan(self, drone_id: str) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.disabled:
            return {"scanned": False, "reason": "drone disabled"}
        if d.battery < BATTERY_COST_SCAN:
            return {"scanned": False, "reason": "insufficient battery for scan"}

        d.battery = max(0, d.battery - BATTERY_COST_SCAN)
        cells = self.model.grid.get_neighborhood(d.pos, moore=True, include_center=True, radius=SCAN_RADIUS)

        for cell in cells:
            tile = self.model.tile_map.get(cell)
            if tile:
                tile.scanned = True

        found: List[Dict[str, Any]] = []
        for cell in cells:
            for a in self.model.grid.get_cell_list_contents([cell]):
                if isinstance(a, SurvivorAgent) and not a.detected:
                    a.detected = True
                    found.append({"id": a.unique_id, "pos": list(cell)})

        if d.battery <= 0:
            d.disabled = True

        return {
            "scanned_cells": len(cells),
            "survivors_found": found,
            "battery": d.battery,
            "drone_pos": list(d.pos),
        }

    def recall_to_base(self, drone_id: str) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.disabled:
            return {"moved": False, "reason": "drone disabled"}
        nearest = min(
            BASE_POSITIONS,
            key=lambda b: abs(b[0] - d.pos[0]) + abs(b[1] - d.pos[1]),
        )
        return self.move_to(drone_id, nearest[0], nearest[1]) | {"base_target": list(nearest)}

    def charge_drone(self, drone_id: str) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.pos not in BASE_POSITIONS:
            return {"charged": False, "reason": f"Not at a station (drone at {list(d.pos)})"}
        d.battery = min(100, d.battery + BATTERY_CHARGE_RATE)
        d.disabled = False
        return {"charged": True, "battery": d.battery, "full": d.battery >= BATTERY_FULL}


class SimpleAiController:
    """
    Deterministic AI that:
    - reasons/plans (printed to console)
    - calls the in-UI tool server to move + scan
    This makes the UI match "AI drives drones".
    """

    def __init__(self, tools: InUiToolServer, action_delay_s: float = 0.0):
        self.tools = tools
        self.action_delay_s = max(0.0, float(action_delay_s or 0.0))
        self._tick = 0
        self._next_waypoint: Dict[str, Tuple[int, int]] = {}
        self._waypoint_queue: List[Tuple[int, int]] = []

    def _pause(self) -> None:
        if self.action_delay_s > 0:
            time.sleep(self.action_delay_s)

    def _build_waypoint_queue(self) -> List[Tuple[int, int]]:
        ms = self.tools.get_mission_state()
        wps: List[Tuple[int, int]] = []
        # Prefer sectors with lowest coverage first.
        sectors = sorted(ms["sectors"], key=lambda s: float(s.get("coverage_pct", 0.0)))
        for s in sectors:
            for wp in s.get("waypoints", []):
                wps.append((int(wp[0]), int(wp[1])))
        return wps

    def think_and_act(self) -> None:
        self._tick += 1
        ms = self.tools.get_mission_state()
        drones = self.tools.discover_drones()["drones"]

        if not self._waypoint_queue:
            self._waypoint_queue = self._build_waypoint_queue()

        survivors_found = int(ms.get("survivors_found", 0))
        survivors_total = int(ms.get("survivors_total", 0))

        print("\n" + "═" * 72)
        print(f"AI CONTROLLER — tick {self._tick}")
        print("═" * 72)
        print(f"Mission: survivors {survivors_found}/{survivors_total}")

        for d in drones:
            did = d["id"]
            pos = tuple(d["pos"])
            battery = float(d["battery"])
            disabled = bool(d["disabled"])

            reasoning_lines = [
                f"Drone {did} at {pos} battery={battery:.0f}% disabled={disabled}."
            ]

            if disabled:
                reasoning_lines.append("Drone is disabled; skip actions this tick.")
                print("[REASON]", " ".join(reasoning_lines))
                continue

            if battery <= BATTERY_CRITICAL:
                reasoning_lines.append("Battery is critical. Recall to base, then charge.")
                print("[REASON]", " ".join(reasoning_lines))
                print("[CALL] recall_to_base", {"drone_id": did})
                self.tools.recall_to_base(did)
                self._pause()
                # Charge if we actually reached a station this tick.
                if tuple(self.tools.model.get_drone(did).pos) in BASE_POSITIONS:
                    print("[CALL] charge_drone", {"drone_id": did})
                    self.tools.charge_drone(did)
                    self._pause()
                continue

            # Always scan opportunistically (this is what reveals survivors in UI).
            reasoning_lines.append("Scan nearby cells for survivors and mark coverage.")
            print("[REASON]", " ".join(reasoning_lines))
            print("[CALL] thermal_scan", {"drone_id": did})
            self.tools.thermal_scan(did)
            self._pause()

            # Move towards next waypoint.
            if not self._next_waypoint.get(did):
                if self._waypoint_queue:
                    self._next_waypoint[did] = self._waypoint_queue.pop(0)

            target = self._next_waypoint.get(did)
            if target:
                if pos == target:
                    self._next_waypoint.pop(did, None)
                else:
                    print("[CALL] move_to", {"drone_id": did, "x": target[0], "y": target[1]})
                    self.tools.move_to(did, target[0], target[1])
                    self._pause()


class GeminiAiController:
    """
    LLM-backed controller (Gemini).

    It sends a mission snapshot + tool schema to Gemini and expects strict JSON:
      {"tool_calls":[{"tool_name":"...", "arguments":{...}, "reasoning":"..."}]}
    Then executes those tool calls against the in-process tool server.
    """

    def __init__(
        self,
        tools: InUiToolServer,
        model_name: str = "gemini-2.5-flash",
        action_delay_s: float = 0.0,
        max_calls_per_tick: int = 5,
    ):
        self.tools = tools
        self.model_name = str(model_name or "gemini-2.5-flash")
        self.action_delay_s = max(0.0, float(action_delay_s or 0.0))
        self.max_calls_per_tick = max(1, int(max_calls_per_tick))
        self._tick = 0
        self._recent_log: str = ""
        self._warned_unavailable = False
        self._client = self._maybe_client()

    def _maybe_client(self):
        if genai is None:
            return None
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            return None

    def _pause(self) -> None:
        if self.action_delay_s > 0:
            time.sleep(self.action_delay_s)

    def _print_wrapped(self, prefix: str, text: str) -> None:
        for ln in textwrap.wrap(text, width=92):
            print(f"{prefix}{ln}")

    def _tool_schema(self) -> List[Dict[str, Any]]:
        # Keep it simple + stable: only tools we actually support here.
        return [
            {
                "name": "discover_drones",
                "description": "List drones with id/pos/battery/disabled.",
                "parameters": [],
            },
            {
                "name": "get_mission_state",
                "description": "Mission snapshot: survivors found/total and sector coverage/waypoints.",
                "parameters": [],
            },
            {
                "name": "move_to",
                "description": "Move a drone toward (x,y) up to its speed; costs battery per cell.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                    {"name": "x", "type": "integer", "required": True},
                    {"name": "y", "type": "integer", "required": True},
                ],
            },
            {
                "name": "thermal_scan",
                "description": "Scan radius-2 around drone; marks tiles scanned and detects survivors; costs battery.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                ],
            },
            {
                "name": "recall_to_base",
                "description": "Move drone toward nearest charging station.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                ],
            },
            {
                "name": "charge_drone",
                "description": "Charge drone battery if on a base.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                ],
            },
        ]

    def _build_prompt(self, mission_state: Dict[str, Any], drones: List[Dict[str, Any]]) -> str:
        tools_json = json.dumps(self._tool_schema(), indent=2)
        ms_json = json.dumps({"mission_state": mission_state, "drones": drones}, default=str)[:3500]
        recent = self._recent_log[-2000:] if self._recent_log else ""
        return (
            "You are the Command Agent for a simulated fleet of drones in an earthquake search-and-rescue mission.\n"
            "You control ONLY the tools listed below. You must output STRICT JSON only.\n\n"
            "Constraints:\n"
            "- Never assume drone IDs; use discover_drones first if needed.\n"
            "- Battery management: if battery <= 20, recall_to_base then charge_drone until >= 90.\n"
            "- Cover sectors by navigating waypoints and calling thermal_scan frequently.\n"
            "- Stop when survivors_found == survivors_total and all sectors are ~covered.\n\n"
            f"Available tools (JSON schema):\n{tools_json}\n\n"
            f"Current state (JSON):\n{ms_json}\n\n"
            f"Recent tool log (may be empty):\n{recent}\n\n"
            "Respond with ONLY this JSON shape (no prose outside JSON):\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "tool_name": "thermal_scan",\n'
            '      "arguments": {"drone_id": "d_0"},\n'
            '      "reasoning": "Why this is the next call."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Pick 1–{self.max_calls_per_tick} tool_calls per tick.\n"
        )

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        text = text.strip()
        # Best effort: if model wrapped JSON in codefence, strip it.
        if text.startswith("```"):
            text = text.strip("`")
            # after stripping backticks, try to remove an optional language tag line
            parts = text.splitlines()
            if parts and parts[0].strip().lower().startswith("json"):
                text = "\n".join(parts[1:]).strip()
        try:
            obj = json.loads(text)
        except Exception:
            return []
        calls = obj.get("tool_calls", [])
        if not isinstance(calls, list):
            return []
        out: List[Dict[str, Any]] = []
        for c in calls[: self.max_calls_per_tick]:
            if not isinstance(c, dict):
                continue
            name = c.get("tool_name")
            args = c.get("arguments") or {}
            reasoning = str(c.get("reasoning") or "").strip()
            if not name or not isinstance(args, dict):
                continue
            out.append({"tool_name": str(name), "arguments": args, "reasoning": reasoning})
        return out

    def _exec(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Execute against our in-process tool server.
        if name == "discover_drones":
            return self.tools.discover_drones()
        if name == "get_mission_state":
            return self.tools.get_mission_state()
        if name == "move_to":
            return self.tools.move_to(str(args.get("drone_id")), int(args.get("x")), int(args.get("y")))
        if name == "thermal_scan":
            return self.tools.thermal_scan(str(args.get("drone_id")))
        if name == "recall_to_base":
            return self.tools.recall_to_base(str(args.get("drone_id")))
        if name == "charge_drone":
            return self.tools.charge_drone(str(args.get("drone_id")))
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def think_and_act(self) -> None:
        self._tick += 1
        ms = self.tools.get_mission_state()
        drones = self.tools.discover_drones()["drones"]

        print("\n" + "═" * 72)
        print(f"GEMINI AI CONTROLLER — tick {self._tick} ({self.model_name})")
        print("═" * 72)

        if self._client is None:
            if not self._warned_unavailable:
                missing = []
                if genai is None:
                    missing.append("google-genai import failed")
                if not os.environ.get("GEMINI_API_KEY"):
                    missing.append("GEMINI_API_KEY not set (check .env)")
                msg = ", ".join(missing) if missing else "unknown reason"
                print(f"[WARN] Gemini AI unavailable ({msg}). Falling back.")
                self._warned_unavailable = True
            return

        prompt = self._build_prompt(mission_state=ms, drones=drones)
        try:
            resp = self._client.models.generate_content(model=self.model_name, contents=prompt)
            text = (getattr(resp, "text", None) or "").strip()
        except Exception as exc:
            print(f"[WARN] Gemini call failed: {exc}. Falling back this tick.")
            return

        calls = self._parse_tool_calls(text)
        if not calls:
            print("[WARN] Gemini returned no valid tool_calls this tick.")
            return

        for call in calls:
            reasoning = call.get("reasoning") or ""
            if reasoning:
                print("\n[CHAIN OF THOUGHT]")
                self._print_wrapped("  ", reasoning)
            name = call["tool_name"]
            args = call["arguments"]
            print(f"\n[MCP TOOL CALL] {name}({json.dumps(args)})")
            result = self._exec(name, args)
            preview = json.dumps(result, default=str)[:260]
            print(f"  [OK] {preview}" if preview else "  [OK]")
            self._recent_log += f"\nTOOL {name}({json.dumps(args)}) -> {preview or '[ok]'}"
            self._pause()


# ═══════════════════════════════════════════════════════════════════════════
#  MESA AGENTS  — only drones + minimal environment markers
# ═══════════════════════════════════════════════════════════════════════════

class SectorTileAgent(Agent):
    """Background tile — shows sector colour and whether it has been scanned."""
    def __init__(self, uid, model, sector_id):
        super().__init__(uid, model)
        self.sector_id = sector_id
        self.scanned = False

    def step(self):
        pass


class SurvivorAgent(Agent):
    """Thermal signature placed in the disaster zone.  Hidden until scanned."""
    def __init__(self, uid, model):
        super().__init__(uid, model)
        self.detected = False

    def step(self):
        pass


class ChargingStationAgent(Agent):
    """Charging pad at a grid corner."""
    def __init__(self, uid, model):
        super().__init__(uid, model)

    def step(self):
        pass


class DroneAgent(Agent):
    """Physical drone — ALL behaviour is driven externally via MCP tools.
    step() is intentionally empty."""
    def __init__(self, uid, model, speed=3):
        super().__init__(uid, model)
        self.speed = speed
        self.battery = 100.0
        self.disabled = False

    def step(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  UI LEGEND
# ═══════════════════════════════════════════════════════════════════════════

class Legend(TextElement):
    def render(self, model):
        drones = [a for a in model.schedule.agents
                  if isinstance(a, DroneAgent)]
        survivors = [a for a in model.schedule.agents
                     if isinstance(a, SurvivorAgent)]
        found = len([s for s in survivors if s.detected])
        total_s = len(survivors)
        # This simulation core does not embed an orchestrator; any "agent state"
        # (assignments, per-drone FSM) lives in the MCP client/orchestrator.
        # For UI, we only display information available from the simulation world.
        sectors_done = 0
        for sid, sdef in SECTOR_DEFS.items():
            ox, oy = sdef["origin"]
            w, h = sdef["size"]
            total = w * h
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if model.tile_map.get((xi, yi)) and model.tile_map[(xi, yi)].scanned
            )
            if total > 0 and scanned / total >= 0.999:
                sectors_done += 1

        drone_rows = ""
        for d in drones:
            bc = ("#00aa00" if d.battery > 50
                  else "#cc8800" if d.battery > 20 else "#cc0000")
            drone_rows += (
                f"<div>🚁 {d.unique_id}: "
                f"<span style='color:{bc};font-weight:bold;'>"
                f"{d.battery:.0f}%</span></div>"
            )

        return f"""
        <div style="font-family:Arial;line-height:1.6;padding-left:10px;">
          <h3>🤖 MCP Drone Command</h3>
          <div style="padding:8px;background:#e8f4fd;border-radius:5px;
                      border-left:4px solid #0078d4;margin-bottom:8px;">
            <strong>Fleet</strong><br/>{drone_rows}
          </div>
          <div style="padding:8px;background:#f0f0f0;border-radius:5px;
                      margin-bottom:8px;">
            <strong>Mission</strong><br/>
            Sectors: {sectors_done}/6<br/>
            Survivors: {found}/{total_s}
          </div>
          <h4>Sector Colours</h4>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                      gap:2px;font-size:12px;">
            <div style="background:#e8f0fe;padding:2px 4px;">NW</div>
            <div style="background:#e6f4ea;padding:2px 4px;">N</div>
            <div style="background:#fef7e0;padding:2px 4px;">NE</div>
            <div style="background:#fce8e6;padding:2px 4px;">SW</div>
            <div style="background:#f3e8fd;padding:2px 4px;">S</div>
            <div style="background:#fff3e0;padding:2px 4px;">SE</div>
          </div>
          <div style="margin-top:8px;font-size:13px;">
            ⚡ Charging station<br/>
            🆘 Survivor (detected)<br/>
            ⚪ Survivor (hidden)<br/>
            🚁 Drone (colour = battery)
          </div>
        </div>
        """


# ═══════════════════════════════════════════════════════════════════════════
#  MESA MODEL
# ═══════════════════════════════════════════════════════════════════════════

class DroneRescueModel(Model):
    def __init__(self, width=24, height=16,
                 num_drones=4, num_survivors=12,
                 scenario: str = "A: Center quake (clustered)",
                 simulate_ai: bool = True,
                 ai_delay_s: float = 0.15,
                 use_gemini_ai: bool = False,
                 gemini_model: str = "gemini-2.5-flash"):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.num_drones = num_drones
        self.num_survivors = num_survivors
        self.scenario = scenario
        self.simulate_ai = simulate_ai
        self.ai_delay_s = float(ai_delay_s or 0.0)
        self.use_gemini_ai = bool(use_gemini_ai)
        self.gemini_model = str(gemini_model or "gemini-2.5-flash")
        self._tools: Optional[InUiToolServer] = None
        self._ai: Optional[SimpleAiController] = None
        self._gemini_ai: Optional[GeminiAiController] = None

        # ── sector background tiles ──
        self.tile_map: Dict[Tuple, SectorTileAgent] = {}
        tid = 0
        for x in range(width):
            for y in range(height):
                sid = pos_to_sector_id(x, y)
                if sid is None:
                    continue
                t = SectorTileAgent(f"tile_{tid}", self, sid)
                self.schedule.add(t)
                self.grid.place_agent(t, (x, y))
                self.tile_map[(x, y)] = t
                tid += 1

        # ── charging stations (4 corners) ──
        for i, bp in enumerate(BASE_POSITIONS):
            cs = ChargingStationAgent(f"base_{i}", self)
            self.schedule.add(cs)
            self.grid.place_agent(cs, bp)

        # ── survivors (scenario preset; falls back to seeded random) ──
        spec = SCENARIOS.get(self.scenario, None)
        if spec and isinstance(spec.get("seed"), int):
            rng = random.Random(int(spec["seed"]))
        else:
            rng = random.Random(0)

        positions: List[Tuple[int, int]] = []
        if spec and isinstance(spec.get("survivor_positions"), list):
            positions = [tuple(p) for p in spec["survivor_positions"] if isinstance(p, (list, tuple)) and len(p) == 2]
        if not positions:
            all_pos = [(x, y) for x in range(width) for y in range(height) if (x, y) not in BASE_POSITIONS]
            rng.shuffle(all_pos)
            positions = all_pos[: int(num_survivors)]

        # Clamp, de-dupe, and avoid bases.
        uniq: List[Tuple[int, int]] = []
        seen = set()
        for (x, y) in positions:
            cx = max(0, min(width - 1, int(x)))
            cy = max(0, min(height - 1, int(y)))
            if (cx, cy) in BASE_POSITIONS:
                continue
            if (cx, cy) in seen:
                continue
            seen.add((cx, cy))
            uniq.append((cx, cy))

        for i in range(min(int(num_survivors), len(uniq))):
            s = SurvivorAgent(f"surv_{i}", self)
            self.schedule.add(s)
            self.grid.place_agent(s, uniq[i])

        # ── drones (start at charging stations) ──
        for i in range(num_drones):
            bp = BASE_POSITIONS[i % len(BASE_POSITIONS)]
            d = DroneAgent(f"d_{i}", self, speed=3)
            self.schedule.add(d)
            self.grid.place_agent(d, bp)

        # ── data collector ──
        self.datacollector = DataCollector(
            model_reporters={
                "SurvivorsFound": lambda m: len([
                    a for a in m.schedule.agents
                    if isinstance(a, SurvivorAgent) and a.detected]),
                "SectorsDone": lambda m: sum(
                    1
                    for sid, sdef in SECTOR_DEFS.items()
                    if (
                        sum(
                            1
                            for xi in range(sdef["origin"][0], sdef["origin"][0] + sdef["size"][0])
                            for yi in range(sdef["origin"][1], sdef["origin"][1] + sdef["size"][1])
                            if m.tile_map.get((xi, yi)) and m.tile_map[(xi, yi)].scanned
                        )
                        / max(1, (sdef["size"][0] * sdef["size"][1]))
                    )
                    >= 0.999
                ),
                "AvgBattery": lambda m: round(
                    sum(a.battery for a in m.schedule.agents
                        if isinstance(a, DroneAgent))
                    / max(1, m.num_drones), 1),
            }
        )

        self.running = True

        # In-UI AI controller (keeps UI and "AI actions" in the same process)
        self._tools = InUiToolServer(self)
        if self.simulate_ai:
            if self.use_gemini_ai:
                self._gemini_ai = GeminiAiController(
                    self._tools,
                    model_name=self.gemini_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._ai = None
            else:
                self._ai = SimpleAiController(self._tools, action_delay_s=self.ai_delay_s)
                self._gemini_ai = None
        else:
            self._ai = None
            self._gemini_ai = None

    def get_drone(self, drone_id: str) -> DroneAgent:
        for a in self.schedule.agents:
            if isinstance(a, DroneAgent) and a.unique_id == drone_id:
                return a
        raise ValueError(f"Drone not found: {drone_id}")

    def step(self):
        if self._gemini_ai is not None:
            self._gemini_ai.think_and_act()
        elif self._ai is not None:
            self._ai.think_and_act()
        self.datacollector.collect(self)
        self.schedule.step()


# ═══════════════════════════════════════════════════════════════════════════
#  PORTRAYAL
# ═══════════════════════════════════════════════════════════════════════════

def portrayal(agent):
    if agent is None:
        return None

    if isinstance(agent, SectorTileAgent):
        base, scanned = SECTOR_COLORS[agent.sector_id]
        return {"Shape": "rect", "w": 1, "h": 1, "Filled": True,
                "Layer": 0, "Color": scanned if agent.scanned else base}

    if isinstance(agent, ChargingStationAgent):
        return {"Shape": "rect", "w": 0.85, "h": 0.85, "Filled": True,
                "Layer": 1, "Color": "#ffd700", "text": "⚡",
                "text_color": "black"}

    if isinstance(agent, SurvivorAgent):
        if agent.detected:
            return {"Shape": "circle", "r": 0.45, "Filled": True,
                    "Layer": 2, "Color": "#ff3333", "text": "🆘",
                    "text_color": "white"}
        # Hidden survivors must still be visible enough to debug scenarios.
        return {"Shape": "circle", "r": 0.25, "Filled": True,
                "Layer": 2, "Color": "#9aa0a6"}

    if isinstance(agent, DroneAgent):
        if agent.battery > 50:
            c = "#00aa00"
        elif agent.battery > 20:
            c = "#cc8800"
        else:
            c = "#cc0000"
        return {"Shape": "rect", "w": 0.7, "h": 0.7, "Filled": True,
                "Layer": 3, "Color": c, "text": "🚁",
                "text_color": "white"}

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SERVER & LAUNCH
# ═══════════════════════════════════════════════════════════════════════════

# Larger canvas so sectors + icons are readable.
grid = CanvasGrid(portrayal, 24, 16, 980, 650)
chart = ChartModule(
    [{"Label": "SurvivorsFound", "Color": "Green"},
     {"Label": "SectorsDone", "Color": "Blue"},
     {"Label": "AvgBattery", "Color": "Orange"}],
    data_collector_name="datacollector",
)
legend = Legend()

server = ModularServer(
    DroneRescueModel,
    [legend, grid, chart],
    "Drone Fleet Search & Rescue — MCP + Chain-of-Thought 🤖",
    {
        "width": 24,
        "height": 16,
        # Use positional args for compatibility across Mesa 1.x.
        # Signature: (param_type, name, value, min_value, max_value, step, choices, description)
        "scenario": UserSettableParameter(
            "choice",
            "Scenario",
            "A: Center quake (clustered)",
            None,
            None,
            1,
            list(SCENARIOS.keys()),
            "Pick one of 3 preset survivor layouts (applies on Reset).",
        ),
        "num_drones": UserSettableParameter(
            "slider", "Drones", 4, 3, 5, 1),
        "num_survivors": UserSettableParameter(
            "slider", "Survivors", 12, 5, 20, 1),
        "simulate_ai": UserSettableParameter(
            "checkbox", "Simulate (AI drives drones)", True),
        "use_gemini_ai": UserSettableParameter(
            "checkbox", "Use Gemini (real LLM agent)", False),
        "gemini_model": UserSettableParameter(
            "choice",
            "Gemini model",
            "gemini-2.5-flash",
            None,
            None,
            1,
            ["gemini-2.5-flash", "gemini-2.5-pro"],
            "Requires GEMINI_API_KEY in environment (.env supported).",
        ),
        "ai_delay_s": UserSettableParameter(
            "slider", "AI delay (sec)", 0.15, 0.0, 1.5, 0.05),
    },
)

if __name__ == "__main__":
    # Avoid WinError 10048 when a previous server is still running.
    import socket

    def _pick_free_port(preferred: List[int]) -> int:
        # First try a few well-known ports.
        for p in preferred:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Bind to all interfaces (matches Tornado behavior).
                    s.bind(("", p))
                return p
            except OSError:
                continue
        # Fall back to an ephemeral free port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    server.port = _pick_free_port([8524, 8525, 8526, 8527])
    print(f"Launching MCP drone fleet server at http://127.0.0.1:{server.port}")
    server.launch()
