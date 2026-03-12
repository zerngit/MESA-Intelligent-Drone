"""
Earthquake Rescue Simulation — MCP (Model Context Protocol) Architecture

All hardcoded drone/vehicle logic is replaced by a reasoning Agent (the "LLM")
that communicates with drones and rescue vehicles via MCP tool calls.
Before every action the Agent produces chain-of-thought reasoning that is
printed to the console, making the decision process fully transparent.

Architecture overview:
    ┌──────────────┐        MCP tool calls          ┌──────────────────┐
    │  LLM Agent   │ ──────────────────────────────► │  MCP Tool Server │
    │  (reasoner)  │ ◄─────────────────────────────  │  (drone/vehicle  │
    │              │     observations + results      │   capabilities)  │
    └──────────────┘                                 └──────────────────┘
                                │
                       chain-of-thought
                        logged to console
"""

import random
import math
import heapq
import itertools
import json
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL CONTEXT PROTOCOL — core primitives
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCPToolParameter:
    """One parameter of an MCP tool."""
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class MCPToolDefinition:
    """Schema that describes a callable MCP tool (analogous to a JSON-RPC method)."""
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] = None          # the actual implementation


@dataclass
class MCPToolCall:
    """A concrete invocation of an MCP tool issued by the agent."""
    tool_name: str
    arguments: Dict[str, Any]
    chain_of_thought: str  # the reasoning the agent produced *before* calling


@dataclass
class MCPToolResult:
    """Result returned to the agent after a tool executes."""
    tool_name: str
    success: bool
    data: Any
    message: str = ""


class MCPToolServer:
    """
    Registry and execution engine for MCP tools.

    Each tool is registered with a schema (MCPToolDefinition) so that the
    reasoning agent can discover what capabilities are available, inspect
    their signatures, and call them by name.
    """

    def __init__(self):
        self._tools: Dict[str, MCPToolDefinition] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, tool_def: MCPToolDefinition):
        self._tools[tool_def.name] = tool_def

    # ------------------------------------------------------------------
    # discovery — the agent calls this to learn what tools exist
    # ------------------------------------------------------------------
    def list_tools(self) -> List[Dict]:
        """Return tool schemas in a format the reasoning agent can consume."""
        result = []
        for t in self._tools.values():
            result.append({
                "name": t.name,
                "description": t.description,
                "parameters": [
                    {"name": p.name, "type": p.type,
                     "description": p.description, "required": p.required}
                    for p in t.parameters
                ],
            })
        return result

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def execute(self, call: MCPToolCall) -> MCPToolResult:
        """Execute a tool call and return its result."""
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return MCPToolResult(
                tool_name=call.tool_name, success=False, data=None,
                message=f"Unknown tool: {call.tool_name}")
        if tool.handler is None:
            return MCPToolResult(
                tool_name=call.tool_name, success=False, data=None,
                message=f"Tool {call.tool_name} has no handler")
        try:
            data = tool.handler(**call.arguments)
            return MCPToolResult(tool_name=call.tool_name, success=True,
                                 data=data, message="ok")
        except Exception as exc:
            return MCPToolResult(tool_name=call.tool_name, success=False,
                                 data=None, message=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
#  CHAIN-OF-THOUGHT REASONING AGENT  ("the LLM")
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningAgent:
    """
    Simulates an LLM that:
      1. Observes the world state (via observation dicts),
      2. Reasons about what to do next (chain of thought),
      3. Decides which MCP tools to call with which arguments,
      4. Logs everything to the console.

    The reasoning is *deterministic* (rule-based) but is formatted as
    natural-language chain-of-thought so the system is transparent and
    explainable.  In a production system this class would be backed by an
    actual LLM API; the MCP layer remains identical.
    """

    def __init__(self, mcp_server: MCPToolServer, model: "EarthquakeMCPModel"):
        self.mcp = mcp_server
        self.model = model
        self._step_counter = 0
        # Track what the agent "knows" through tool results
        self._known_collapsed: set = set()      # building ids
        self._known_injured: Dict[str, Tuple[int, int]] = {}  # person_id -> pos
        self._drone_visited: Dict[str, set] = {}  # drone_id -> visited positions

    # ------------------------------------------------------------------
    # public entry point — called once per MESA model step
    # ------------------------------------------------------------------
    def think_and_act(self):
        """Full reasoning cycle: observe → think → act via MCP tools."""
        self._step_counter += 1
        self._print_header()

        # Phase 1 — observe
        observations = self._gather_observations()
        self._print_observations(observations)

        # Phase 2 — plan drone actions
        drone_calls = self._plan_drones(observations)

        # Phase 3 — plan rescue vehicle actions
        vehicle_calls = self._plan_vehicles(observations)

        # Phase 4 — execute all planned tool calls through MCP
        all_calls = drone_calls + vehicle_calls
        if not all_calls:
            self._print_cot("No actionable tool calls this step — waiting for "
                            "new information.")
        for call in all_calls:
            self._print_tool_call(call)
            result = self.mcp.execute(call)
            self._print_tool_result(result)

    # ==================================================================
    #  OBSERVATION
    # ==================================================================
    def _gather_observations(self) -> Dict:
        """Build a world-state snapshot from publicly available model data."""
        people = [a for a in self.model.schedule.agents
                  if isinstance(a, PersonAgent)]
        drones = [a for a in self.model.schedule.agents
                  if isinstance(a, DroneAgent)]
        vehicles = [a for a in self.model.schedule.agents
                    if isinstance(a, ResourceAgent)]

        injured = [p for p in people if p.state == "injured"]
        discovered_injured = [p for p in injured if p.discovered]
        evacuated = [p for p in people if p.state == "evacuated"]
        dead = [p for p in people if p.state == "dead"]
        collapsed = [b for b in self.model.building_list if b.collapsed]
        undiscovered_collapsed = [b for b in collapsed if not b.discovered]

        return {
            "step": self._step_counter,
            "total_people": self.model.population,
            "injured_count": len(injured),
            "discovered_injured_count": len(discovered_injured),
            "discovered_injured": [
                {"id": p.unique_id, "pos": p.pos} for p in discovered_injured
            ],
            "undiscovered_injured_count": len(injured) - len(discovered_injured),
            "evacuated_count": len(evacuated),
            "dead_count": len(dead),
            "collapsed_buildings": len(collapsed),
            "undiscovered_collapsed": len(undiscovered_collapsed),
            "drones": [
                {"id": d.unique_id, "pos": d.pos} for d in drones
            ],
            "vehicles": [
                {"id": v.unique_id, "pos": v.pos,
                 "carrying": len(v.carrying),
                 "target_id": v.target.unique_id if v.target else None}
                for v in vehicles
            ],
            "exit_cells": self.model.exit_cells[:5],  # sample
        }

    # ==================================================================
    #  DRONE PLANNING  (chain of thought)
    # ==================================================================
    def _plan_drones(self, obs: Dict) -> List[MCPToolCall]:
        calls = []
        for drone_info in obs["drones"]:
            did = drone_info["id"]
            dpos = drone_info["pos"]

            # --- chain of thought ---
            thoughts = []
            thoughts.append(f"Drone {did} is at position {dpos}.")

            if obs["undiscovered_collapsed"] > 0:
                thoughts.append(
                    f"There are {obs['undiscovered_collapsed']} undiscovered "
                    f"collapsed buildings. I should direct the drone toward "
                    f"unexplored areas to find survivors.")
                # Decide target: scan for nearest undiscovered collapsed building
                thoughts.append(
                    "I will call `drone_scan_area` to check surroundings, "
                    "then `drone_move_toward_collapse` to advance the drone "
                    "toward the nearest undiscovered collapse.")
            else:
                thoughts.append(
                    "All collapsed buildings have been discovered. The drone "
                    "can continue a sweep pattern to look for any remaining "
                    "injured people.")
                thoughts.append(
                    "I will call `drone_scan_area` to inspect surrounding "
                    "cells, then `drone_explore_unvisited` to continue "
                    "the sweep.")

            cot_text = " ".join(thoughts)

            # --- tool calls ---
            # 1) scan
            calls.append(MCPToolCall(
                tool_name="drone_scan_area",
                arguments={"drone_id": did},
                chain_of_thought=cot_text,
            ))

            # 2) move
            if obs["undiscovered_collapsed"] > 0:
                calls.append(MCPToolCall(
                    tool_name="drone_move_toward_collapse",
                    arguments={"drone_id": did},
                    chain_of_thought=(
                        f"Moving drone {did} toward the nearest undiscovered "
                        f"collapsed building to maximize discovery rate."),
                ))
            else:
                calls.append(MCPToolCall(
                    tool_name="drone_explore_unvisited",
                    arguments={"drone_id": did},
                    chain_of_thought=(
                        f"All collapses discovered. Moving drone {did} to an "
                        f"unvisited cell to continue scanning for survivors."),
                ))
        return calls

    # ==================================================================
    #  VEHICLE PLANNING  (chain of thought)
    # ==================================================================
    def _plan_vehicles(self, obs: Dict) -> List[MCPToolCall]:
        calls = []
        # Pre-compute which injured are already targeted by some vehicle
        targeted_ids = set()
        for v_info in obs["vehicles"]:
            if v_info["target_id"]:
                targeted_ids.add(v_info["target_id"])

        for v_info in obs["vehicles"]:
            vid = v_info["id"]
            vpos = v_info["pos"]
            carrying = v_info["carrying"]
            current_target = v_info["target_id"]

            thoughts = []
            thoughts.append(f"Rescue vehicle {vid} is at {vpos}.")

            if carrying > 0:
                thoughts.append(
                    f"Vehicle is carrying {carrying} injured survivor(s). "
                    f"Priority is to transport them to the nearest exit for "
                    f"evacuation.")
                thoughts.append(
                    "I will call `vehicle_transport_to_exit` to move toward "
                    "the nearest exit and drop off survivors.")
                cot = " ".join(thoughts)
                calls.append(MCPToolCall(
                    tool_name="vehicle_transport_to_exit",
                    arguments={"vehicle_id": vid},
                    chain_of_thought=cot,
                ))
                continue

            # Not carrying — look for injured
            if obs["discovered_injured_count"] > 0:
                # Pick the nearest discovered injured not already targeted
                available = [
                    p for p in obs["discovered_injured"]
                    if p["id"] not in targeted_ids
                ]
                if available:
                    nearest = min(available,
                                  key=lambda p: self.model.manhattan_distance(
                                      vpos, p["pos"]))
                    thoughts.append(
                        f"There are {len(available)} discovered injured not "
                        f"yet assigned. Nearest is {nearest['id']} at "
                        f"{nearest['pos']} (distance "
                        f"{self.model.manhattan_distance(vpos, nearest['pos'])}).")
                    thoughts.append(
                        "I will call `vehicle_move_to_injured` to navigate "
                        "there, then `vehicle_pickup` if close enough.")
                    cot = " ".join(thoughts)
                    targeted_ids.add(nearest["id"])
                    calls.append(MCPToolCall(
                        tool_name="vehicle_move_to_injured",
                        arguments={"vehicle_id": vid,
                                   "target_person_id": nearest["id"]},
                        chain_of_thought=cot,
                    ))
                    calls.append(MCPToolCall(
                        tool_name="vehicle_pickup",
                        arguments={"vehicle_id": vid,
                                   "target_person_id": nearest["id"]},
                        chain_of_thought=(
                            f"Attempting pickup of {nearest['id']} if vehicle "
                            f"{vid} has reached their location."),
                    ))
                    continue

            if obs["injured_count"] > 0:
                thoughts.append(
                    f"There are {obs['injured_count']} injured people but "
                    f"none have been discovered yet by drones. Patrolling "
                    f"to a random road neighbor while waiting for drone intel.")
            else:
                thoughts.append(
                    "No injured people remain. Vehicle will hold position.")

            cot = " ".join(thoughts)
            calls.append(MCPToolCall(
                tool_name="vehicle_patrol",
                arguments={"vehicle_id": vid},
                chain_of_thought=cot,
            ))

        return calls

    # ==================================================================
    #  CONSOLE LOGGING
    # ==================================================================
    def _print_header(self):
        print(f"\n{'═' * 72}")
        print(f"  REASONING AGENT — Step {self._step_counter}")
        print(f"{'═' * 72}")

    def _print_observations(self, obs: Dict):
        print("\n📡  OBSERVATIONS:")
        print(f"    People — injured: {obs['injured_count']} "
              f"(discovered: {obs['discovered_injured_count']}), "
              f"evacuated: {obs['evacuated_count']}, dead: {obs['dead_count']}")
        print(f"    Buildings — collapsed: {obs['collapsed_buildings']} "
              f"(undiscovered: {obs['undiscovered_collapsed']})")
        for d in obs["drones"]:
            print(f"    Drone {d['id']} @ {d['pos']}")
        for v in obs["vehicles"]:
            label = f"carrying {v['carrying']}" if v["carrying"] else "idle"
            print(f"    Vehicle {v['id']} @ {v['pos']} [{label}]")

    def _print_cot(self, text: str):
        print(f"\n💭  CHAIN OF THOUGHT:")
        for line in textwrap.wrap(text, width=68):
            print(f"    {line}")

    def _print_tool_call(self, call: MCPToolCall):
        # Always print chain-of-thought before the tool call
        self._print_cot(call.chain_of_thought)
        args_str = json.dumps(call.arguments)
        print(f"\n🔧  MCP TOOL CALL: {call.tool_name}({args_str})")

    def _print_tool_result(self, result: MCPToolResult):
        status = "✅" if result.success else "❌"
        data_str = json.dumps(result.data, default=str) if result.data else ""
        print(f"    {status}  {result.message}"
              + (f"  →  {data_str[:120]}" if data_str else ""))


# ═══════════════════════════════════════════════════════════════════════════
#  MCP TOOL IMPLEMENTATIONS  (thin wrappers around model/agent operations)
# ═══════════════════════════════════════════════════════════════════════════

def register_mcp_tools(mcp: MCPToolServer, model: "EarthquakeMCPModel"):
    """Register every tool that drones and vehicles expose via MCP."""

    # ------------------------------------------------------------------
    # helper — resolve an agent by id
    # ------------------------------------------------------------------
    def _agent(agent_id: str):
        for a in model.schedule.agents:
            if a.unique_id == agent_id:
                return a
        raise ValueError(f"Agent not found: {agent_id}")

    # ==================================================================
    #  DRONE tools
    # ==================================================================

    def drone_scan_area(drone_id: str) -> Dict:
        """Scan cells in radius-2 around the drone; mark discoveries."""
        drone = _agent(drone_id)
        scan = model.grid.get_neighborhood(
            drone.pos, moore=True, include_center=True, radius=2)
        found_collapsed = []
        found_injured = []
        for cell in scan:
            for a in model.grid.get_cell_list_contents([cell]):
                if isinstance(a, BuildingAgent) and a.collapsed and not a.discovered:
                    a.discovered = True
                    found_collapsed.append({"id": a.unique_id, "pos": cell})
                if isinstance(a, PersonAgent) and a.state == "injured" and not a.discovered:
                    a.discovered = True
                    found_injured.append({"id": a.unique_id, "pos": cell})
        return {
            "scanned_cells": len(scan),
            "new_collapsed_found": found_collapsed,
            "new_injured_found": found_injured,
        }

    mcp.register(MCPToolDefinition(
        name="drone_scan_area",
        description="Scan cells within radius 2 of the drone.  Marks any "
                    "collapsed buildings and injured people as 'discovered' so "
                    "rescue vehicles can target them.",
        parameters=[
            MCPToolParameter("drone_id", "string",
                             "Unique ID of the drone to operate"),
        ],
        handler=drone_scan_area,
    ))

    # ------------------------------------------------------------------

    def drone_move_toward_collapse(drone_id: str) -> Dict:
        """Move the drone one step toward the nearest undiscovered collapse."""
        drone = _agent(drone_id)
        undiscovered = [b for b in model.building_list
                        if b.collapsed and not b.discovered]
        if not undiscovered:
            return {"moved": False, "reason": "no undiscovered collapses"}
        target = min(undiscovered,
                     key=lambda b: model.manhattan_distance(drone.pos, b.pos))
        neigh = model.grid.get_neighborhood(
            drone.pos, moore=True, include_center=False)
        # Pick neighbor closest to target
        best = min(neigh, key=lambda n: model.manhattan_distance(n, target.pos))
        try:
            model.grid.move_agent(drone, best)
        except Exception:
            return {"moved": False, "reason": "move failed"}
        if hasattr(drone, "visited"):
            drone.visited.add(best)
        return {"moved": True, "new_pos": best,
                "target_building": target.unique_id}

    mcp.register(MCPToolDefinition(
        name="drone_move_toward_collapse",
        description="Move the drone one cell toward the nearest undiscovered "
                    "collapsed building.",
        parameters=[
            MCPToolParameter("drone_id", "string",
                             "Unique ID of the drone to move"),
        ],
        handler=drone_move_toward_collapse,
    ))

    # ------------------------------------------------------------------

    def drone_explore_unvisited(drone_id: str) -> Dict:
        """Move drone to a neighboring cell it hasn't visited yet."""
        drone = _agent(drone_id)
        visited = getattr(drone, "visited", set())
        neigh = model.grid.get_neighborhood(
            drone.pos, moore=True, include_center=False)
        random.shuffle(neigh)
        # Prefer unvisited
        unvisited = [n for n in neigh if n not in visited]
        chosen = unvisited[0] if unvisited else neigh[0]
        try:
            model.grid.move_agent(drone, chosen)
        except Exception:
            return {"moved": False, "reason": "move failed"}
        drone.visited.add(chosen)
        return {"moved": True, "new_pos": chosen}

    mcp.register(MCPToolDefinition(
        name="drone_explore_unvisited",
        description="Move the drone toward a cell it has not visited yet "
                    "(exploration sweep).",
        parameters=[
            MCPToolParameter("drone_id", "string",
                             "Unique ID of the drone to move"),
        ],
        handler=drone_explore_unvisited,
    ))

    # ==================================================================
    #  VEHICLE tools
    # ==================================================================

    def vehicle_move_to_injured(vehicle_id: str, target_person_id: str) -> Dict:
        """Navigate vehicle toward an injured person using A*."""
        vehicle = _agent(vehicle_id)
        target = _agent(target_person_id)
        vehicle.target = target
        path = model.find_path_astar(vehicle.pos, target.pos)
        if not path:
            return {"moved": False, "reason": "no path to target"}
        steps = min(vehicle.speed, len(path))
        for _ in range(steps):
            if not path:
                break
            nxt = path.pop(0)
            if not model.is_road(nxt):
                break
            try:
                model.grid.move_agent(vehicle, nxt)
            except Exception:
                break
        return {"moved": True, "new_pos": vehicle.pos,
                "distance_remaining": model.manhattan_distance(
                    vehicle.pos, target.pos)}

    mcp.register(MCPToolDefinition(
        name="vehicle_move_to_injured",
        description="Move the rescue vehicle toward a specific injured person "
                    "using A* pathfinding.",
        parameters=[
            MCPToolParameter("vehicle_id", "string",
                             "Unique ID of the rescue vehicle"),
            MCPToolParameter("target_person_id", "string",
                             "Unique ID of the injured person to head toward"),
        ],
        handler=vehicle_move_to_injured,
    ))

    # ------------------------------------------------------------------

    def vehicle_pickup(vehicle_id: str, target_person_id: str) -> Dict:
        """Pick up an injured person if the vehicle is at their location."""
        vehicle = _agent(vehicle_id)
        target = _agent(target_person_id)
        if vehicle.pos != target.pos:
            return {"picked_up": False,
                    "reason": f"vehicle at {vehicle.pos}, person at {target.pos}"}
        if len(vehicle.carrying) >= 1:
            return {"picked_up": False, "reason": "vehicle already carrying"}
        if target.state != "injured":
            return {"picked_up": False,
                    "reason": f"person state is '{target.state}', not injured"}
        target.state = "normal"
        target.discovered = False
        if hasattr(target, "injury_time"):
            try:
                delattr(target, "injury_time")
            except Exception:
                pass
        vehicle.carrying.append(target)
        vehicle.target = None
        return {"picked_up": True, "person_id": target.unique_id}

    mcp.register(MCPToolDefinition(
        name="vehicle_pickup",
        description="Pick up an injured person at the vehicle's current "
                    "location, stabilizing them for transport.",
        parameters=[
            MCPToolParameter("vehicle_id", "string",
                             "Unique ID of the rescue vehicle"),
            MCPToolParameter("target_person_id", "string",
                             "Unique ID of the person to pick up"),
        ],
        handler=vehicle_pickup,
    ))

    # ------------------------------------------------------------------

    def vehicle_transport_to_exit(vehicle_id: str) -> Dict:
        """Drive the vehicle toward the nearest exit and drop off survivors."""
        vehicle = _agent(vehicle_id)
        exit_pos = model.find_nearest_exit(vehicle.pos)
        if not exit_pos:
            return {"moved": False, "reason": "no exit cells"}
        path = model.find_path_astar(vehicle.pos, exit_pos)
        if path:
            steps = min(vehicle.speed, len(path))
            for _ in range(steps):
                if not path:
                    break
                nxt = path.pop(0)
                if not model.is_road(nxt):
                    break
                try:
                    model.grid.move_agent(vehicle, nxt)
                except Exception:
                    break
        # Check if at exit — drop off
        dropped = []
        if vehicle.pos in model.exit_cells:
            for p in vehicle.carrying:
                p.state = "evacuated"
                p.discovered = False
                dropped.append(p.unique_id)
            vehicle.carrying.clear()
        return {"new_pos": vehicle.pos, "dropped_off": dropped}

    mcp.register(MCPToolDefinition(
        name="vehicle_transport_to_exit",
        description="Move the vehicle toward the nearest exit cell.  If the "
                    "exit is reached, drop off all carried survivors and mark "
                    "them evacuated.",
        parameters=[
            MCPToolParameter("vehicle_id", "string",
                             "Unique ID of the rescue vehicle"),
        ],
        handler=vehicle_transport_to_exit,
    ))

    # ------------------------------------------------------------------

    def vehicle_patrol(vehicle_id: str) -> Dict:
        """Wander the vehicle to a random neighboring road cell."""
        vehicle = _agent(vehicle_id)
        neigh = model.road_neighbors(vehicle.pos)
        random.shuffle(neigh)
        for n in neigh:
            try:
                model.grid.move_agent(vehicle, n)
                return {"moved": True, "new_pos": n}
            except Exception:
                pass
        return {"moved": False, "reason": "no valid neighbors"}

    mcp.register(MCPToolDefinition(
        name="vehicle_patrol",
        description="Move the rescue vehicle to a random adjacent road cell "
                    "(idle patrol while awaiting orders).",
        parameters=[
            MCPToolParameter("vehicle_id", "string",
                             "Unique ID of the rescue vehicle"),
        ],
        handler=vehicle_patrol,
    ))

    # ==================================================================
    #  SENSOR tools
    # ==================================================================

    def sensor_read(sensor_id: str) -> Dict:
        """Read the current state of a seismic sensor."""
        sensor = _agent(sensor_id)
        return {"id": sensor.unique_id, "pos": sensor.pos,
                "active": sensor.active,
                "early_warning": model.early_warning_active}

    mcp.register(MCPToolDefinition(
        name="sensor_read",
        description="Read the activation state of a seismic sensor and "
                    "whether the early warning system is active.",
        parameters=[
            MCPToolParameter("sensor_id", "string",
                             "Unique ID of the sensor to read"),
        ],
        handler=sensor_read,
    ))


# ═══════════════════════════════════════════════════════════════════════════
#  UI Legend
# ═══════════════════════════════════════════════════════════════════════════

class Legend(TextElement):
    def render(self, model):
        total_people = model.population
        evacuated = len([a for a in model.schedule.agents
                         if isinstance(a, PersonAgent) and a.state == "evacuated"])
        dead = len([a for a in model.schedule.agents
                    if isinstance(a, PersonAgent) and a.state == "dead"])
        injured = len([a for a in model.schedule.agents
                       if isinstance(a, PersonAgent) and a.state == "injured"])

        efficiency = int((evacuated / total_people) * 100) if total_people else 0

        if efficiency >= 90:
            grade, color = "A+ 🌟", "#00cc00"
        elif efficiency >= 75:
            grade, color = "A 👍", "#33cc33"
        elif efficiency >= 60:
            grade, color = "B ✓", "#66cc66"
        elif efficiency >= 40:
            grade, color = "C ⚠️", "#ff9900"
        else:
            grade, color = "D ⛔", "#cc0000"

        return f"""
        <div style="font-family: Arial; line-height:1.6; padding-left:10px;">
          <h3>Legend — MCP Architecture</h3>
          <div>🏢 &nbsp; Building (healthy)</div>
          <div>🏚️ &nbsp; Building (damaged)</div>
          <div>🧱💥 &nbsp; Collapsed building</div>
          <div style="margin-top:8px;">👤 &nbsp; Person (healthy)</div>
          <div>🆘 &nbsp; Person (injured)</div>
          <div>💀 &nbsp; Person (dead)</div>
          <div>✅ &nbsp; Person (evacuated)</div>
          <div style="margin-top:8px;">🚑 &nbsp; Rescue vehicle (MCP-controlled)</div>
          <div>🚁 &nbsp; Drone (MCP-controlled)</div>
          <div>🛰️ &nbsp; Sensor</div>
          <div style="margin-top:12px; padding:10px; background:#e8f4fd; border-radius:5px; border-left:4px solid #0078d4;">
            <strong>🤖 MCP Mode</strong><br/>
            <small>All drone/vehicle actions are issued via MCP tool calls.<br/>
            Chain-of-thought reasoning is printed to the console.</small>
          </div>
          <div style="margin-top:8px; padding:10px; background:#f0f0f0; border-radius:5px;">
            <strong>Rescue Efficiency</strong><br/>
            <span style="font-size:20px; color:{color}; font-weight:bold;">{efficiency}%</span>
            <span style="font-size:18px;">{grade}</span><br/>
            <small>Evacuated: {evacuated}/{total_people} | Dead: {dead} | Injured: {injured}</small>
          </div>
        </div>
        """


# ═══════════════════════════════════════════════════════════════════════════
#  MESA Agents  (drones & vehicles are now *passive* — no hardcoded logic)
# ═══════════════════════════════════════════════════════════════════════════

class RoadAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class BuildingAgent(Agent):
    def __init__(self, unique_id, model, resilience=0.7):
        super().__init__(unique_id, model)
        self.resilience = resilience
        self.health = 1.0
        self.collapsed = False
        self.discovered = False

    def quake_impact(self, intensity):
        damage = intensity * (1 - self.resilience) * random.uniform(1.5, 2.2)
        self.health = max(0.0, self.health - damage)
        if self.health <= 0.45 and not self.collapsed:
            self.collapsed = True

    def step(self):
        pass


class PersonAgent(Agent):
    def __init__(self, unique_id, model, speed=1):
        super().__init__(unique_id, model)
        self.state = "normal"
        self.base_speed = speed
        self.speed = speed
        self.path = None
        self.target = None
        self.discovered = False

    def plan(self):
        if self.target is None:
            self.target = self.model.find_nearest_exit(self.pos)
        if self.target is not None:
            self.path = self.model.find_path_bfs(self.pos, self.target)
            if self.path is None:
                self.path = []

    def step(self):
        if self.state in ("injured", "dead", "evacuated"):
            if self.state == "injured" and hasattr(self, "injury_time"):
                t = max(0, self.model.schedule.time - self.injury_time)
                if random.random() < 0.003 * t:
                    self.state = "dead"
            return

        self.speed = max(1, int(self.base_speed))
        if self.path is None:
            self.plan()

        if self.path:
            steps = min(self.speed, len(self.path))
            for _ in range(steps):
                if not self.path:
                    break
                nxt = self.path.pop(0)
                if not self.model.is_road(nxt):
                    self.path = None
                    break
                try:
                    self.model.grid.move_agent(self, nxt)
                except Exception:
                    self.path = None
                    break

        if self.pos in self.model.exit_cells:
            self.state = "evacuated"


class ResourceAgent(Agent):
    """Rescue vehicle — ALL movement logic is handled by MCP tool calls.
    The step() method is intentionally empty."""
    def __init__(self, unique_id, model, speed=2):
        super().__init__(unique_id, model)
        self.speed = speed
        self.target = None
        self.carrying = []

    def step(self):
        # No hardcoded logic — the ReasoningAgent issues MCP tool calls
        # that manipulate this agent's state externally.
        pass


class DroneAgent(Agent):
    """Drone — ALL movement and scanning logic is handled by MCP tool calls.
    The step() method is intentionally empty."""
    def __init__(self, unique_id, model, speed=3):
        super().__init__(unique_id, model)
        self.speed = speed
        self.visited = set()

    def step(self):
        # No hardcoded logic — controlled entirely via MCP tools.
        pass


class SensorAgent(Agent):
    def __init__(self, unique_id, model, coverage=5, reliability=0.9):
        super().__init__(unique_id, model)
        self.coverage = coverage
        self.reliability = reliability
        self.active = False

    def detect_and_broadcast(self, epicenter, magnitude):
        if (self.model.manhattan_distance(self.pos, epicenter) <= self.coverage
                and random.random() < self.reliability):
            self.active = True
            self.model.early_warning_active = True

    def step(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  MESA Model  (MCP-enabled)
# ═══════════════════════════════════════════════════════════════════════════

class EarthquakeMCPModel(Model):
    def __init__(self, width=24, height=16, population=30,
                 resources=2, sensors=2, drones=1, magnitude=0.7):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.population = population
        self.resources = resources
        self.sensors = sensors
        self.drones = drones
        self.magnitude = magnitude

        # ---- road map ----
        self.road_map = set()
        for x in range(self.width):
            for y in range(self.height):
                if (y % 3 == 0) or (x % 4 == 0):
                    self.road_map.add((x, y))

        rid = 0
        for pos in list(self.road_map):
            r = RoadAgent(f"road_{rid}", self)
            self.schedule.add(r)
            self.grid.place_agent(r, pos)
            rid += 1

        # ---- buildings ----
        self.building_list = []
        bid = 0
        building_density = 0.70
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) in self.road_map:
                    continue
                if random.random() < building_density:
                    b = BuildingAgent(f"b_{bid}", self,
                                     resilience=random.uniform(0.30, 0.70))
                    self.schedule.add(b)
                    self.grid.place_agent(b, (x, y))
                    self.building_list.append(b)
                    bid += 1

        # ---- earthquake ----
        self.epicenter = (self.width // 2 + random.randint(-2, 2),
                          self.height // 2 + random.randint(-2, 2))
        maxdist = math.hypot(self.width, self.height)
        for b in self.building_list:
            dist = self.manhattan_distance(b.pos, self.epicenter)
            intensity = self.magnitude * max(0, 1 - (dist / maxdist) * 1.5)
            b.quake_impact(intensity)

        # ---- exits ----
        self.exit_cells = [
            pos for pos in self.road_map
            if pos[0] in (0, self.width - 1) or pos[1] in (0, self.height - 1)
        ]
        if not self.exit_cells:
            self.exit_cells = list(self.road_map)[:4]

        # ---- people ----
        pid = 0
        road_positions = list(self.road_map)
        random.shuffle(road_positions)
        placed = 0
        for pos in road_positions:
            if placed >= self.population:
                break
            occupants = [a for a in self.grid.get_cell_list_contents([pos])
                         if isinstance(a, PersonAgent)]
            if len(occupants) >= 2:
                continue
            p = PersonAgent(f"p_{pid}", self, speed=1)
            self.schedule.add(p)
            self.grid.place_agent(p, pos)
            pid += 1
            placed += 1

        # ---- rescue vehicles ----
        rid2 = 0
        border_roads = list(self.exit_cells)
        for _ in range(self.resources):
            pos = random.choice(border_roads)
            rv = ResourceAgent(f"res_{rid2}", self, speed=2)
            self.schedule.add(rv)
            self.grid.place_agent(rv, pos)
            rid2 += 1

        # ---- sensors ----
        sid = 0
        for _ in range(self.sensors):
            pos = random.choice(road_positions)
            s = SensorAgent(f"s_{sid}", self, coverage=5, reliability=0.95)
            self.schedule.add(s)
            self.grid.place_agent(s, pos)
            sid += 1

        # ---- drones ----
        did = 0
        for _ in range(self.drones):
            pos = (self.width // 2, self.height // 2)
            d = DroneAgent(f"d_{did}", self, speed=3)
            self.schedule.add(d)
            self.grid.place_agent(d, pos)
            did += 1

        # ---- data collector ----
        self.datacollector = DataCollector(
            model_reporters={
                "Evacuated": lambda m: len([
                    a for a in m.schedule.agents
                    if isinstance(a, PersonAgent) and a.state == "evacuated"]),
                "Injured": lambda m: len([
                    a for a in m.schedule.agents
                    if isinstance(a, PersonAgent) and a.state == "injured"]),
                "Dead": lambda m: len([
                    a for a in m.schedule.agents
                    if isinstance(a, PersonAgent) and a.state == "dead"]),
                "CollapsedBuildings": lambda m: len([
                    b for b in m.building_list if b.collapsed]),
            }
        )

        # ---- initial sensor broadcast ----
        for a in list(self.schedule.agents):
            if isinstance(a, SensorAgent):
                a.detect_and_broadcast(self.epicenter, self.magnitude)

        # ---- initial injuries ----
        self._injure_people_near_collapses()

        self.running = True
        self.early_warning_active = False

        # ══════════════════════════════════════════════════════════════
        #  Initialize MCP layer
        # ══════════════════════════════════════════════════════════════
        self.mcp_server = MCPToolServer()
        register_mcp_tools(self.mcp_server, self)
        self.reasoning_agent = ReasoningAgent(self.mcp_server, self)

        # Print available MCP tools at startup
        print("\n" + "═" * 72)
        print("  MCP TOOL SERVER — registered tools")
        print("═" * 72)
        for t in self.mcp_server.list_tools():
            params = ", ".join(p["name"] for p in t["parameters"])
            print(f"  ● {t['name']}({params})")
            print(f"    {t['description'][:70]}")
        print("═" * 72 + "\n")

    # ------------------------------------------------------------------
    #  Model step
    # ------------------------------------------------------------------
    def step(self):
        people = [a for a in self.schedule.agents
                  if isinstance(a, PersonAgent)]
        if people:
            active_people = [p for p in people
                             if p.state not in ("evacuated", "dead")]
            if not active_people:
                self.running = False
                evacuated = len([p for p in people if p.state == "evacuated"])
                dead = len([p for p in people if p.state == "dead"])
                print(f"\n{'═' * 72}")
                print(f"  SIMULATION COMPLETE at step {self.schedule.time}")
                print(f"  Evacuated: {evacuated}  |  Dead: {dead}  "
                      f"|  Total: {len(people)}")
                print(f"{'═' * 72}\n")
                return

        # Periodic aftershock
        if self.schedule.time > 0 and self.schedule.time % 10 == 0:
            candidates = [b for b in self.building_list
                          if not b.collapsed and b.health < 0.85]
            if candidates:
                b = random.choice(candidates)
                b.quake_impact(self.magnitude * 0.75)

        self.datacollector.collect(self)

        # ── The reasoning agent observes, thinks, and acts via MCP ──
        self.reasoning_agent.think_and_act()

        # Advance all MESA agents (people move on their own; drones &
        # vehicles are passive and skip their step()).
        self.schedule.step()

        if self.schedule.time > 30:
            self.early_warning_active = False

    # ------------------------------------------------------------------
    #  Helpers (unchanged from original)
    # ------------------------------------------------------------------
    def is_road(self, pos):
        return pos in self.road_map

    def road_neighbors(self, pos):
        candidates = self.grid.get_neighborhood(
            pos, moore=False, include_center=False)
        return [p for p in candidates if self.is_road(p)]

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_nearest_exit(self, pos):
        if not self.exit_cells:
            return None
        return min(self.exit_cells,
                   key=lambda e: self.manhattan_distance(pos, e))

    def find_path_bfs(self, start, goal):
        if start == goal:
            return []
        q = deque([(start, [])])
        visited = {start}
        while q:
            curr, path = q.popleft()
            for n in self.road_neighbors(curr):
                if n in visited:
                    continue
                newp = path + [n]
                if n == goal:
                    return newp
                visited.add(n)
                q.append((n, newp))
        return None

    def traversal_cost(self, pos):
        penalty = 0.0
        for c in self.grid.get_neighborhood(
                pos, moore=True, include_center=False):
            for a in self.grid.get_cell_list_contents([c]):
                if isinstance(a, BuildingAgent) and a.health < 0.6:
                    penalty += (1.0 - a.health) * 0.5
        return 1.0 + penalty

    def find_path_astar(self, start, goal):
        if start == goal:
            return []
        frontier = []
        counter = itertools.count()
        heapq.heappush(frontier, (
            self.manhattan_distance(start, goal), next(counter),
            0.0, start, []))
        best_g = {start: 0.0}
        while frontier:
            f, _cnt, g, curr, path = heapq.heappop(frontier)
            if curr == goal:
                return path
            for n in self.road_neighbors(curr):
                cost = self.traversal_cost(n)
                newg = g + cost
                if n not in best_g or newg < best_g[n]:
                    best_g[n] = newg
                    h = self.manhattan_distance(n, goal)
                    heapq.heappush(frontier, (
                        newg + h, next(counter), newg, n, path + [n]))
        return None

    def _injure_people_near_collapses(self):
        radius = 2
        for b in [a for a in self.building_list if a.collapsed]:
            scan = self.grid.get_neighborhood(
                b.pos, moore=True, include_center=True, radius=radius)
            for pos in scan:
                dist = self.manhattan_distance(b.pos, pos)
                weight = max(0.0, 1.0 - (dist / (radius + 0.1)))
                for p in self.grid.get_cell_list_contents([pos]):
                    if isinstance(p, PersonAgent) and p.state == "normal":
                        base = 0.30
                        damage_factor = (1 - b.health) * 0.75
                        prob = min(0.95,
                                   base + damage_factor * weight
                                   + random.uniform(0, 0.18))
                        if random.random() < prob:
                            p.state = "injured"
                            p.injury_time = self.schedule.time
                            p.discovered = False


# ═══════════════════════════════════════════════════════════════════════════
#  Portrayal (same visual as original)
# ═══════════════════════════════════════════════════════════════════════════

def portrayal(agent):
    if agent is None:
        return None
    if isinstance(agent, RoadAgent):
        return {"Shape": "rect", "w": 1, "h": 1, "Filled": True,
                "Layer": 0, "Color": "#dcdcdc"}
    if isinstance(agent, BuildingAgent):
        if agent.collapsed:
            return {"Shape": "rect", "w": 1, "h": 1, "Filled": True,
                    "Layer": 1, "Color": "#8b0000",
                    "text": "🧱💥", "text_color": "white"}
        elif agent.health > 0.7:
            return {"Shape": "rect", "w": 1, "h": 1, "Filled": True,
                    "Layer": 1, "Color": "#7cfc00",
                    "text": "🏢", "text_color": "black"}
        else:
            return {"Shape": "rect", "w": 1, "h": 1, "Filled": True,
                    "Layer": 1, "Color": "#ff8c00",
                    "text": "🏚️", "text_color": "black"}
    if isinstance(agent, PersonAgent):
        state_map = {
            "normal": ("👤", "blue"),
            "injured": ("🆘", "red"),
            "dead": ("💀", "black"),
            "evacuated": ("✅", "green"),
        }
        txt, color = state_map.get(agent.state, ("👤", "blue"))
        if agent.state == "injured" and getattr(agent, "discovered", False):
            txt = "🚨"
        return {"Shape": "circle", "r": 0.4, "Filled": True,
                "Layer": 3, "Color": color, "text": txt,
                "text_color": "white"}
    if isinstance(agent, ResourceAgent):
        return {"Shape": "rect", "w": 0.6, "h": 0.6, "Filled": True,
                "Layer": 4, "Color": "#003399", "text": "🚑"}
    if isinstance(agent, DroneAgent):
        return {"Shape": "rect", "w": 0.45, "h": 0.45, "Filled": True,
                "Layer": 5, "Color": "gold", "text": "🚁"}
    if isinstance(agent, SensorAgent):
        color = ("cyan" if agent.active or agent.model.early_warning_active
                 else "lightblue")
        return {"Shape": "circle", "r": 0.25, "Filled": True,
                "Layer": 2, "Color": color, "text": "🛰️"}
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Server & Launch
# ═══════════════════════════════════════════════════════════════════════════

grid = CanvasGrid(portrayal, 24, 16, 700, 500)
chart = ChartModule(
    [{"Label": "Evacuated", "Color": "Green"},
     {"Label": "Injured", "Color": "Orange"},
     {"Label": "Dead", "Color": "Black"},
     {"Label": "CollapsedBuildings", "Color": "Red"}],
    data_collector_name="datacollector",
)
legend = Legend()

server = ModularServer(
    EarthquakeMCPModel,
    [legend, grid, chart],
    "Earthquake Rescue — MCP + Chain-of-Thought 🤖",
    {
        "width": 24,
        "height": 16,
        "population": 30,
        "resources": UserSettableParameter("slider", "Rescue vehicles", 2, 0, 5, 1),
        "sensors": 2,
        "drones": UserSettableParameter("slider", "Drones", 1, 0, 5, 1),
        "magnitude": 0.7,
    },
)

if __name__ == "__main__":
    server.port = 8523
    print("Launching MCP-enabled server at http://127.0.0.1:8523")
    server.launch()
