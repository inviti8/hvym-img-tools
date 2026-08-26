"""CLI tests. The help path is a real regression guard: it broke twice, once on
argparse %-expansion and once on Windows cp1252 encoding — both triggered purely
by characters in a tool's own description text.
"""
from __future__ import annotations

import pytest

from hvym_img_tools.cli import _help, main


def test_help_escapes_percent():
    """A description containing "74% of wall-clock" must not %-expand."""
    assert _help("74% of wall-clock") == "74%% of wall-clock"
    assert _help(None, "fallback") == "fallback"


def test_tool_help_renders(capsys):
    """Builds every subparser and formats help — covers non-ASCII and % in
    descriptions, for every registered tool."""
    with pytest.raises(SystemExit) as exc:
        main(["reangle", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--in" in out and "--out" in out and "--mc-resolution" in out


def test_top_level_help_lists_tools(capsys):
    assert main([]) == 2
    assert "reangle" in capsys.readouterr().out
