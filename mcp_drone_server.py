"""
Official MCP server for the Drone Rescue Mesa simulation.

This server exposes the drone fleet capabilities as MCP tools using the
official `mcp` Python SDK (`mcp.server.fastmcp.FastMCP`).

The simulation state (a live `DroneRescueModel`) is kept server-side, and all
Agent↔Drone communication happens via MCP tool calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from mesa_drone_rescue_mcp import (
    BASE_POSITIONS,
    BATTERY_CHARGE_RATE,
    BATTERY_COST_MOVE,
    BATTERY_COST_SCAN,
    BATTERY_CRITICAL,
    BATTERY_FULL,
    SCAN_RADIUS,
    SECTOR_DEFS,
    DroneAgent,
    DroneRescueModel,
    SurvivorAgent,
    _sector_waypoints,
)


@dataclass
class AppState:
    model: DroneRescueModel


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[AppState]:
    model = DroneRescueModel(width=24, height=16, num_drones=4, num_survivors=12)
    try:
        yield AppState(model=model)
    finally:
        # Mesa has no explicit teardown we need here.
        pass


mcp = FastMCP(
    "MESA Drone Rescue (Official MCP)",
    instructions=(
        "This server simulates a drone fleet search-and-rescue mission on a 2D grid. "
        "Use tool discovery to find available drone tools. "
        "Do not assume drone IDs; call discover_drones first."
    ),
    lifespan=lifespan,
    json_response=True,
)


def _model(ctx: Context[ServerSession, AppState]) -> DroneRescueModel:
    return ctx.request_context.lifespan_context.model


def _drone(ctx: Context[ServerSession, AppState], drone_id: str) -> DroneAgent:
    model = _model(ctx)
    for a in model.schedule.agents:
        if isinstance(a, DroneAgent) and a.unique_id == drone_id:
            return a
    raise ValueError(f"Drone not found: {drone_id}")


@mcp.tool()
def discover_drones(ctx: Context[ServerSession, AppState]) -> Dict:
    """Discover all drones currently active on the network."""
    model = _model(ctx)
    drones = [a for a in model.schedule.agents if isinstance(a, DroneAgent)]
    return {
        "drones": [
            {"id": d.unique_id, "pos": list(d.pos), "battery": d.battery, "disabled": d.disabled}
            for d in drones
        ],
        "count": len(drones),
    }


@mcp.tool()
def get_drone_status(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Get detailed status of a specific drone."""
    d = _drone(ctx, drone_id)
    return {
        "id": d.unique_id,
        "pos": list(d.pos),
        "battery": d.battery,
        "speed": d.speed,
        "disabled": d.disabled,
    }


@mcp.tool()
def get_battery_status(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Get battery percentage; includes critical flag (≤ 20%)."""
    d = _drone(ctx, drone_id)
    return {"drone_id": d.unique_id, "battery_pct": d.battery, "critical": d.battery <= BATTERY_CRITICAL}


@mcp.tool()
def move_to(drone_id: str, x: int, y: int, ctx: Context[ServerSession, AppState]) -> Dict:
    """Move drone toward (x,y). Moves up to drone speed cells. Costs 1% battery per cell."""
    model = _model(ctx)
    d = _drone(ctx, drone_id)

    if d.disabled:
        return {"moved": False, "reason": "drone disabled"}
    if d.battery <= 0:
        d.disabled = True
        return {"moved": False, "reason": "battery depleted"}

    tx = max(0, min(model.grid.width - 1, int(x)))
    ty = max(0, min(model.grid.height - 1, int(y)))
    cx, cy = d.pos
    steps = 0
    while steps < d.speed and (cx, cy) != (tx, ty) and d.battery > 0:
        dx = (1 if tx > cx else -1) if tx != cx else 0
        dy = (1 if ty > cy else -1) if ty != cy else 0
        cx += dx
        cy += dy
        d.battery = max(0, d.battery - BATTERY_COST_MOVE)
        steps += 1

    model.grid.move_agent(d, (cx, cy))
    if d.battery <= 0:
        d.disabled = True

    return {"new_pos": [cx, cy], "battery": d.battery, "arrived": (cx, cy) == (tx, ty), "steps_taken": steps}


@mcp.tool()
def thermal_scan(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Thermal scan radius-2 around the drone; detects survivors; costs 5% battery."""
    model = _model(ctx)
    d = _drone(ctx, drone_id)

    if d.disabled:
        return {"scanned": False, "reason": "drone disabled"}
    if d.battery < BATTERY_COST_SCAN:
        return {"scanned": False, "reason": "insufficient battery for scan"}

    d.battery = max(0, d.battery - BATTERY_COST_SCAN)
    cells = model.grid.get_neighborhood(d.pos, moore=True, include_center=True, radius=SCAN_RADIUS)

    # mark tiles as scanned (for UI coverage)
    for cell in cells:
        tile = model.tile_map.get(cell)
        if tile:
            tile.scanned = True

    found: List[Dict] = []
    for cell in cells:
        for a in model.grid.get_cell_list_contents([cell]):
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


@mcp.tool()
def recall_to_base(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Move drone toward the nearest charging station. Costs 1% battery per cell."""
    model = _model(ctx)
    d = _drone(ctx, drone_id)

    if d.disabled:
        return {"moved": False, "reason": "drone disabled"}

    nearest = min(
        BASE_POSITIONS,
        key=lambda b: abs(b[0] - d.pos[0]) + abs(b[1] - d.pos[1]),
    )
    cx, cy = d.pos
    tx, ty = nearest
    steps = 0
    while steps < d.speed and (cx, cy) != (tx, ty) and d.battery > 0:
        dx = (1 if tx > cx else -1) if tx != cx else 0
        dy = (1 if ty > cy else -1) if ty != cy else 0
        cx += dx
        cy += dy
        d.battery = max(0, d.battery - BATTERY_COST_MOVE)
        steps += 1

    model.grid.move_agent(d, (cx, cy))
    if d.battery <= 0:
        d.disabled = True

    return {"new_pos": [cx, cy], "base_target": list(nearest), "arrived": (cx, cy) == (tx, ty), "battery": d.battery}


@mcp.tool()
def charge_drone(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Charge +25% battery. Drone must be on a charging station (grid corner)."""
    d = _drone(ctx, drone_id)
    if d.pos not in BASE_POSITIONS:
        return {"charged": False, "reason": f"Not at a station (drone at {list(d.pos)})"}
    d.battery = min(100, d.battery + BATTERY_CHARGE_RATE)
    d.disabled = False
    return {"charged": True, "battery": d.battery, "full": d.battery >= BATTERY_FULL}


@mcp.tool()
def get_sector_info(ctx: Context[ServerSession, AppState]) -> Dict:
    """Zone layout: sector names, origins, coverage %, and scan waypoints."""
    model = _model(ctx)
    sectors: List[Dict] = []

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
        wps = _sector_waypoints(sdef["origin"], sdef["size"])
        sectors.append(
            {
                "id": sid,
                "name": sdef["name"],
                "origin": [ox, oy],
                "size": [w, h],
                "coverage_pct": round(scanned / total * 100, 1),
                "waypoints": [list(wp) for wp in wps],
            }
        )

    return {"sectors": sectors}


@mcp.tool()
def get_mission_state(ctx: Context[ServerSession, AppState]) -> Dict:
    """Current mission snapshot: survivors found/total and sector coverage summary."""
    model = _model(ctx)
    survivors = [a for a in model.schedule.agents if isinstance(a, SurvivorAgent)]
    found = [s for s in survivors if s.detected]
    sectors = get_sector_info(ctx)["sectors"]
    return {
        "survivors_found": len(found),
        "survivors_total": len(survivors),
        "sectors": [{"id": s["id"], "name": s["name"], "coverage_pct": s["coverage_pct"]} for s in sectors],
    }


@mcp.tool()
def advance_simulation(steps: int = 1, ctx: Context[ServerSession, AppState] = None) -> Dict:  # type: ignore[assignment]
    """Advance the Mesa model by N steps (agents are passive; used for datacollection/UI parity)."""
    if ctx is None:
        raise ValueError("Context injection failed")
    model = _model(ctx)
    steps_i = max(1, int(steps))
    for _ in range(steps_i):
        model.step()
    return {"advanced": steps_i}


if __name__ == "__main__":
    # Default transport is STDIO, which works well for local orchestrators and MCP Inspector.
    mcp.run()

