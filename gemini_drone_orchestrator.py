"""
Gemini-powered MCP client orchestrator for the Drone Rescue simulation.

This file uses:
  - google-genai (Gemini API)
  - the official MCP Python SDK client

It connects to `mcp_drone_server.py` over STDIO, asks Gemini which MCP tools
to call next (with arguments), and then executes those tool calls via MCP.
All mission logic comes from the LLM; the simulation is controlled *only*
through MCP tools (no direct access to the Mesa model).
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List

from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
from dotenv import load_dotenv

load_dotenv()  # add this once, near the top

def _print_cot(text: str) -> None:
    print("\n[CHAIN OF THOUGHT]")
    for ln in textwrap.wrap(text, width=72):
        print(f"  {ln}")


def _print_call(name: str, args: dict) -> None:
    print(f"\n[MCP TOOL CALL] {name}({json.dumps(args)})")


def _structured(result) -> dict:
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        if "result" in sc and isinstance(sc["result"], dict):
            return sc["result"]
        return sc
    txt_parts: List[str] = []
    if getattr(result, "content", None):
        for c in result.content:
            if getattr(c, "type", None) == "text":
                txt_parts.append(getattr(c, "text", ""))
    txt = "\n".join(txt_parts).strip()
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


def _build_tools_prompt(tools_result) -> str:
    lines = []
    for t in tools_result.tools:
        lines.append(f"- name: {t.name}")
        lines.append(f"  description: {t.description}")
        schema = t.inputSchema or {}
        params = schema.get("properties", {})
        required = set(schema.get("required", []) or [])
        if params:
            lines.append("  parameters:")
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "string")
                desc = pinfo.get("description", "")
                req = "required" if pname in required else "optional"
                lines.append(f"    - {pname} ({ptype}, {req}) - {desc}")
        else:
            lines.append("  parameters: none")
        lines.append("")
    return "\n".join(lines)


def _gemini_client() -> genai.Client:
    # GEMINI_API_KEY must be set in the environment.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def _ask_gemini_for_plan(
    client: genai.Client,
    model_name: str,
    tools_yaml: str,
    mission_state: Dict[str, Any],
    recent_log: str,
) -> List[Dict[str, Any]]:
    """
    Ask Gemini which MCP tools to call next.

    We instruct Gemini to return *only* JSON of the form:
      {"tool_calls":[{"tool_name": "...", "arguments": {...}, "reasoning": "..."}]}
    """
    system_instructions = """
You are the Command Agent for a simulated fleet of drones in an earthquake
search-and-rescue mission. You control ONLY the MCP tools listed below.

Constraints:
- Never assume drone IDs; first call discover_drones to learn them.
- Never move drones or change state directly; only use MCP tools.
- Manage battery carefully:
    - if battery <= 20%, recall_to_base and charge_drone until >= 90%.
- Cover all sectors by visiting provided waypoints with move_to + thermal_scan.
- Stop when all survivors are found AND all sectors are reasonably covered.

Available MCP tools (name, description, parameters):
"""

    mission_summary = json.dumps(mission_state, default=str)[:2000]
    recent_log_snippet = recent_log[-2000:] if recent_log else ""

    prompt = (
        system_instructions
        + "\n"
        + tools_yaml
        + "\nCurrent mission state (JSON):\n"
        + mission_summary
        + "\n\nRecent tool log (may be empty):\n"
        + recent_log_snippet
        + "\n\nRespond with ONLY a JSON object like this (no prose outside JSON):\n"
        + '{\n'
        + '  "tool_calls": [\n'
        + '    {\n'
        + '      "tool_name": "get_battery_status",\n'
        + '      "arguments": {"drone_id": "d_0"},\n'
        + '      "reasoning": "Explain briefly why this call is next."\n'
        + "    }\n"
        + "  ]\n"
        + "}\n"
        + "Pick 1–5 tool_calls per step. Prefer cheap status checks before moves.\n"
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = (resp.text or "").strip()
    try:
        plan = json.loads(text)
        calls = plan.get("tool_calls", [])
        if isinstance(calls, list):
            return [c for c in calls if isinstance(c, dict)]
    except Exception:
        pass
    return []


async def run_gemini_mission(model_name: str = "gemini-2.5-flash") -> None:
    server_params = StdioServerParameters(command="python", args=["mcp_drone_server.py"], env=None)
    client = _gemini_client()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools up front and build a human-readable schema summary.
            tools = await session.list_tools()
            tools_yaml = _build_tools_prompt(tools)

            print("\n" + "=" * 80)
            print("GEMINI MCP CLIENT - Tool discovery")
            print("=" * 80)
            for t in tools.tools:
                print(f"  - {t.name}")

            recent_log = ""

            for step in range(1, 60):
                print("\n" + "-" * 80)
                print(f"LLM STEP {step}")
                print("-" * 80)

                # Query mission state from the server (authoritative).
                ms_result = await session.call_tool(name="get_mission_state", arguments={})
                ms = _structured(ms_result)
                survivors_found = ms.get("survivors_found", 0)
                survivors_total = ms.get("survivors_total", 0)
                sectors = ms.get("sectors", [])

                all_survivors = survivors_total > 0 and survivors_found >= survivors_total
                sectors_done = sum(1 for s in sectors if s.get("coverage_pct", 0) >= 99.0)
                all_sectors_reasonable = sectors_done >= max(1, len(sectors) - 1)

                if all_survivors and all_sectors_reasonable:
                    print("\n" + "=" * 80)
                    print("MISSION COMPLETE (Gemini)")
                    print("=" * 80)
                    print(f"Survivors found: {survivors_found}/{survivors_total}")
                    print(f"Sectors >=99%:  {sectors_done}/{len(sectors)}")
                    return

                # Ask Gemini what to do next.
                tool_calls = _ask_gemini_for_plan(
                    client=client,
                    model_name=model_name,
                    tools_yaml=tools_yaml,
                    mission_state=ms,
                    recent_log=recent_log,
                )

                if not tool_calls:
                    print("[WARN] Gemini returned no valid tool_calls; stopping.")
                    break

                # Execute each proposed tool call via MCP.
                for call in tool_calls:
                    name = call.get("tool_name")
                    args = call.get("arguments") or {}
                    reasoning = str(call.get("reasoning") or "").strip()
                    if not name or not isinstance(args, dict):
                        continue

                    if reasoning:
                        _print_cot(reasoning)
                    _print_call(name, args)

                    try:
                        res = await session.call_tool(name=name, arguments=args)
                    except Exception as exc:  # network/protocol error
                        print(f"  [ERROR] MCP call failed: {exc}")
                        continue

                    sc = _structured(res)
                    preview = json.dumps(sc, default=str)[:220] if sc else ""
                    if res.isError:
                        print(f"  [ERROR] {getattr(res, 'error', '') or 'tool error'}")
                    else:
                        print(f"  [OK] {preview}" if preview else "  [OK]")

                    # Append to recent log context for Gemini.
                    recent_log += f"\nTOOL {name}({json.dumps(args)}) -> {preview or '[ok]'}"

            print("\n[INFO] Stopping: reached max Gemini steps without mission completion.")


if __name__ == "__main__":
    asyncio.run(run_gemini_mission())

