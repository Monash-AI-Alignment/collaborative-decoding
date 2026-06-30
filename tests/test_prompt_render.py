from jinja2 import Template
from pathlib import Path


def test_prompt_is_collaborative_decoding():
    tmpl = Path("w2s_research/research_loop/prompt.jinja2").read_text()
    out = Template(tmpl).render(workspace_dir="/ws", weak_model="W", strong_model="S",
                                server_url="http://s:8000", benchmark="alpaca_eval",
                                r_bar="0.98", logs_dir="/ws/logs", local_mode="false")
    assert "weak_token_fraction" in out and "utility_recovery" in out
    assert "DeferralPolicy" in out and "build_policy" in out
    assert "share_finding" in out and "get_leaderboard" in out
    assert "0.98" in out and "{{ weak_model }}" not in out      # rendered, vars substituted
    assert "PGR" not in out and "weak labels" not in out         # old W2S content gone
