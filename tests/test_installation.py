"""The primary CLI must work without the optional legacy dependency stack."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _without_legacy_dependencies(arguments):
    script = textwrap.dedent("""
        import builtins
        import sys
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'pyranges'}:
                raise ModuleNotFoundError(f'No module named {name!r}', name=name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import
        from lanceotron_mcc.call_peaks import main
        main()
    """)
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("command", ["call-bigwig-peaks", "call-count-peaks", "call-legacy-peaks"])
def test_command_help_does_not_require_legacy_dependencies(command):
    result = _without_legacy_dependencies([command, "--help"])
    assert result.returncode == 0, result.stderr
    assert command in result.stdout


def test_legacy_execution_explains_missing_optional_dependencies():
    result = _without_legacy_dependencies(
        ["call-legacy-peaks", "--bigwig", "sample.bw", "--output", "peaks.bed"]
    )
    assert result.returncode == 2
    assert "uv sync --extra legacy" in result.stderr
    assert "Traceback" not in result.stderr
