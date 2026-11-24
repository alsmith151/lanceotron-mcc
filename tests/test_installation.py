"""The CLI must not depend on heavyweight optional stacks such as torch."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _without_optional_dependencies(arguments):
    script = textwrap.dedent("""
        import builtins
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'pyranges', 'matplotlib'}:
                raise ModuleNotFoundError(f'No module named {name!r}', name=name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import
        from mcc_peaks.call_peaks import main
        main()
    """)
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("command", ["call-bigwig-peaks", "call-count-peaks", "footprint"])
def test_command_help_does_not_require_optional_dependencies(command):
    result = _without_optional_dependencies([command, "--help"])
    assert result.returncode == 0, result.stderr
    assert command in result.stdout
