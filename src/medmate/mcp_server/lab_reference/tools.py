"""
tools.py

Implementation of the Lab Reference tools. Pure lookups against the
curated static JSON dataset (data/tests.json, data/reference_ranges.json)
— no LLM calls, no generation. Returns structured "not found" results
rather than guessing when a test or demographic isn't in the dataset.
"""

import json
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

_tests_cache: Optional[list] = None
_ranges_cache: Optional[list] = None


def _load_tests() -> list:
    global _tests_cache
    if _tests_cache is None:
        with open(DATA_DIR / "tests.json") as f:
            _tests_cache = json.load(f)
    return _tests_cache


def _load_ranges() -> list:
    global _ranges_cache
    if _ranges_cache is None:
        with open(DATA_DIR / "reference_ranges.json") as f:
            _ranges_cache = json.load(f)
    return _ranges_cache


def _normalize(name: str) -> str:
    return name.strip().lower()


def _resolve_test_name(query_name: str) -> Optional[str]:
    """Match a raw query string (canonical name or alias) to the canonical test_name."""
    q = _normalize(query_name)
    for t in _load_tests():
        if _normalize(t["test_name"]) == q:
            return t["test_name"]
        if q in [_normalize(a) for a in t.get("aliases", [])]:
            return t["test_name"]
    return None


def get_test_info(test_name: str) -> dict:
    """
    Return catalog info (description, unit, category, aliases) for a test.
    Input may be a canonical name or a known alias.
    """
    canonical = _resolve_test_name(test_name)
    if canonical is None:
        return {
            "found": False,
            "query": test_name,
            "message": f"No test matching '{test_name}' found in the reference catalog.",
        }

    for t in _load_tests():
        if t["test_name"] == canonical:
            return {
                "found": True,
                "test_name": t["test_name"],
                "category": t["category"],
                "unit": t["unit"],
                "description": t["description"],
                "aliases": t["aliases"],
            }

    # Should not happen if data is consistent, but guard anyway.
    return {
        "found": False,
        "query": test_name,
        "message": f"No test matching '{test_name}' found in the reference catalog.",
    }


def get_reference_range(test_name: str, demographic: Optional[str] = None) -> dict:
    """
    Return the reference range for a test, optionally sliced by demographic
    ('adult', 'adult_male', 'adult_female', 'pediatric').

    Resolution order when demographic is given but not an exact match:
      1. exact demographic match
      2. fall back to 'adult' bucket if it exists
      3. return not-found (never guess a range)

    If demographic is None:
      - if only one bucket exists (e.g. 'adult'), return it
      - if multiple demographic-specific buckets exist and none is 'adult',
        return all of them and let the caller/LLM ask the user to specify
    """
    canonical = _resolve_test_name(test_name)
    if canonical is None:
        return {
            "found": False,
            "query": test_name,
            "message": f"No test matching '{test_name}' found in the reference catalog.",
        }

    matches = [r for r in _load_ranges() if r["test_name"] == canonical]
    if not matches:
        return {
            "found": False,
            "query": test_name,
            "message": f"No reference range data available for '{canonical}'.",
        }

    available_demographics = [m["demographic"] for m in matches]

    if demographic:
        dem_norm = _normalize(demographic)
        exact = [m for m in matches if _normalize(m["demographic"]) == dem_norm]
        if exact:
            r = exact[0]
            return {
                "found": True,
                "test_name": canonical,
                "demographic": r["demographic"],
                "low": r["low"],
                "high": r["high"],
                "unit": r["unit"],
                "source": r["source"],
            }
        # fall back to 'adult' if present
        fallback = [m for m in matches if _normalize(m["demographic"]) == "adult"]
        if fallback:
            r = fallback[0]
            return {
                "found": True,
                "test_name": canonical,
                "demographic": r["demographic"],
                "low": r["low"],
                "high": r["high"],
                "unit": r["unit"],
                "source": r["source"],
                "note": f"No specific range for demographic '{demographic}'; returned general adult range.",
            }
        return {
            "found": False,
            "query": test_name,
            "message": (
                f"No reference range for '{canonical}' under demographic '{demographic}'. "
                f"Available demographics: {available_demographics}."
            ),
        }

    # No demographic specified
    if len(matches) == 1:
        r = matches[0]
        return {
            "found": True,
            "test_name": canonical,
            "demographic": r["demographic"],
            "low": r["low"],
            "high": r["high"],
            "unit": r["unit"],
            "source": r["source"],
        }

    # Multiple demographic-specific buckets and no demographic given —
    # do not guess, return all options for the caller to disambiguate.
    return {
        "found": True,
        "ambiguous": True,
        "test_name": canonical,
        "message": (
            f"'{canonical}' has demographic-specific ranges. "
            f"Specify a demographic to get a precise range."
        ),
        "options": [
            {
                "demographic": m["demographic"],
                "low": m["low"],
                "high": m["high"],
                "unit": m["unit"],
                "source": m["source"],
            }
            for m in matches
        ],
    }


def list_available_tests() -> dict:
    """Utility tool: list all tests currently in the reference catalog."""
    tests = _load_tests()
    return {
        "count": len(tests),
        "tests": [t["test_name"] for t in tests],
    }