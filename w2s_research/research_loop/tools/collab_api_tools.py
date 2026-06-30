"""MCP tools for the collaborative-decoding server: baselines, evaluate, share, leaderboard.

Thin glue: validation/shaping lives in w2s_research.server.finding_payload (pure);
HTTP in http_utils. Requires claude_agent_sdk + an HTTP client at agent runtime.
"""
import json
from typing import Any, Dict

from claude_agent_sdk import tool, create_sdk_mcp_server

from .http_utils import get_server_url, async_http_get, async_http_post
from w2s_research.server.finding_payload import build_share_payload


def _ok(d):
    return {"content": [{"type": "text", "text": json.dumps(d)}]}


@tool("get_baselines", "Get U_weak/U_strong/gap/r_bar for a benchmark (the recovery anchor).",
      {"type": "object", "properties": {"benchmark": {"type": "string"}}, "required": ["benchmark"]})
async def get_baselines(args: Dict[str, Any]):
    return _ok(await async_http_get(f"{get_server_url()}/api/baselines?benchmark={args['benchmark']}"))


@tool("get_leaderboard",
      "Leaderboard for a benchmark: result findings with recovery>=r_bar, ranked by f_weak.",
      {"type": "object", "properties": {"benchmark": {"type": "string"}}, "required": ["benchmark"]})
async def get_leaderboard(args: Dict[str, Any]):
    return _ok(await async_http_get(f"{get_server_url()}/api/leaderboard?benchmark={args['benchmark']}"))


@tool("evaluate_generations",
      "Submit engine-computed metrics; returns recovery vs the canonical baselines and whether it meets the bar.",
      {"type": "object", "properties": {
          "benchmark": {"type": "string"}, "idea_name": {"type": "string"},
          "utility": {"type": "number"}, "weak_token_fraction": {"type": "number"}},
       "required": ["benchmark", "idea_name", "utility", "weak_token_fraction"]})
async def evaluate_generations(args: Dict[str, Any]):
    return _ok(await async_http_post(f"{get_server_url()}/api/evaluate-generations", dict(args)))


@tool("share_finding",
      "Share a finding to the forum/leaderboard. finding_type='result' requires utility, "
      "weak_token_fraction, utility_recovery and publishes to the leaderboard.",
      {"type": "object", "properties": {
          "benchmark": {"type": "string"}, "idea_name": {"type": "string"},
          "summary": {"type": "string"}, "title": {"type": "string"},
          "finding_type": {"type": "string"}, "utility": {"type": "number"},
          "weak_token_fraction": {"type": "number"}, "utility_recovery": {"type": "number"},
          "worked": {"type": "boolean"}},
       "required": ["benchmark", "idea_name", "summary"]})
async def share_finding(args: Dict[str, Any]):
    payload, err = build_share_payload(args)
    if err:
        return _ok({"error": err})
    return _ok(await async_http_post(f"{get_server_url()}/api/findings/share", payload))


def create_collab_api_tools_server():
    return create_sdk_mcp_server(
        name="collab-api-tools",
        tools=[get_baselines, get_leaderboard, evaluate_generations, share_finding])
