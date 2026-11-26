"""Unit tests — the CRAP ratchet baseline parser (code-review MINOR-3).

The baseline help/header used to document a ``path:line:name:crap`` format
the parser did NOT accept: a contributor following the DOCS (or pasting a
gate-output offender line verbatim) silently mis-keyed the allowances and
BROKE the gate (proven exit 1 in the review). The parser now accepts BOTH
formats (keyed by ``path:name`` the same way) and fails LOUDLY on malformed
entries; the docs and the parser agree.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_check_crap() -> ModuleType:
    """Import ``scripts/check_crap.py`` without an ``__init__.py`` bridge
    (dynamic import by path; the module only imports the dev deps
    ``coverage`` / ``pytest_crap``, both present in the dev venv)."""
    spec = importlib.util.spec_from_file_location(
        "check_crap", _REPO_ROOT / "scripts" / "check_crap.py"
    )
    assert spec is not None and spec.loader is not None, (
        "scripts/check_crap.py must be discoverable at the repo root"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_crap = _load_check_crap()


class TestLoadBaselineFormats:
    def test_documented_three_field_format_is_parsed(self, tmp_path: Path):
        """The SHIPPED baseline format: ``path:name:crap`` (keyed by
        ``path:name`` — a line shift can never silently un-allow an
        offender)."""
        baseline = tmp_path / "baseline.txt"
        baseline.write_text(
            "src/infrastructure/web_search/url_safety/adapter.py:_validate_resolved:41.54\n"
        )
        assert check_crap._load_baseline(baseline) == {
            "src/infrastructure/web_search/url_safety/adapter.py:_validate_resolved": 41.54
        }

    def test_gate_output_line_number_form_is_also_parsed(self, tmp_path: Path):
        """The 4-field gate-output form (an offender line copied verbatim
        from the gate/report: ``path:line:name:crap``) must NOT break the
        gate — it is keyed by ``path:name`` exactly like the 3-field form."""
        baseline = tmp_path / "baseline.txt"
        baseline.write_text(
            "src/infrastructure/web_search/url_safety/adapter.py:41:_validate_resolved:41.54\n"
        )
        assert check_crap._load_baseline(baseline) == {
            "src/infrastructure/web_search/url_safety/adapter.py:_validate_resolved": 41.54
        }

    def test_comments_and_blank_lines_are_ignored(self, tmp_path: Path):
        baseline = tmp_path / "baseline.txt"
        baseline.write_text(
            "# header comment\n\nsrc/a.py:func_a:30.18  # trailing comment\n"
        )
        assert check_crap._load_baseline(baseline) == {"src/a.py:func_a": 30.18}

    def test_missing_baseline_file_is_an_empty_ratchet(self, tmp_path: Path):
        assert check_crap._load_baseline(tmp_path / "absent.txt") == {}

    def test_shipped_baseline_parses_and_keys_by_name(self):
        """The real shipped ``scripts/crap-baseline.txt`` yields the three
        documented allowances, keyed by ``path:name`` (the exact keys the
        gate looks up — the documented invocation must keep working)."""
        allowances = check_crap._load_baseline(
            _REPO_ROOT / "scripts" / "crap-baseline.txt"
        )
        assert allowances == {
            "src/infrastructure/web_search/url_safety/adapter.py:_validate_resolved": 41.54,
            "src/infrastructure/rag/classical_bm25_adapter.py:_ensure_bm25_index": 39.70,
            "src/infrastructure/openapi_mcp/swagger2_converter.py:convert_swagger2_to_openapi3": 30.18,
        }


class TestLoadBaselineMalformedEntries:
    def test_two_field_entry_fails_loudly(self, tmp_path: Path):
        baseline = tmp_path / "baseline.txt"
        baseline.write_text("src/a.py:41.54\n")
        with pytest.raises(ValueError, match="malformed baseline entry"):
            check_crap._load_baseline(baseline)

    def test_non_numeric_crap_score_fails_loudly(self, tmp_path: Path):
        baseline = tmp_path / "baseline.txt"
        baseline.write_text("src/a.py:func_a:not-a-number\n")
        with pytest.raises(ValueError, match="not a number"):
            check_crap._load_baseline(baseline)
