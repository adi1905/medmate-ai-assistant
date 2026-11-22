"""
generate_lab_reference_data.py

Builds data/tests.json and data/reference_ranges.json for the Lab Reference
MCP server, using distinct test_name/unit pairs pulled from the existing
lab_results SQLite table, mapped against a curated set of real-world
reference ranges (source: Mayo Clinic Laboratories / common clinical
reference charts). Demographic splits are only added where the range
genuinely differs by sex/age; otherwise a single 'adult' bucket is used.
"""

import sqlite3
import json
import sys
from pathlib import Path

from config.settings import DB_PATH
OUT_DIR = Path(r"C:\Users\adity\OneDrive\Documents\MedMate Clinic Copilot\src\medmate\mcp_server\lab_reference\data")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Curated source-of-truth reference data (hand-sourced, not LLM-generated)
#    demographic keys: "adult" (default/unspecified), "adult_male",
#    "adult_female", "pediatric"
# ---------------------------------------------------------------------------

CURATED_REFERENCE = {
    "Creatinine": {
        "unit": "mg/dL",
        "aliases": ["creatinine", "serum creatinine", "cr"],
        "category": "Renal Panel",
        "description": "Measures creatinine, a waste product filtered by the kidneys, used to assess kidney function.",
        "ranges": {
            "adult_male": {"low": 0.7, "high": 1.3},
            "adult_female": {"low": 0.6, "high": 1.1},
        },
        "source": "Mayo Clinic Laboratories, adult reference intervals",
    },
    "TSH": {
        "unit": "mIU/L",
        "aliases": ["tsh", "thyroid stimulating hormone", "thyrotropin"],
        "category": "Thyroid Panel",
        "description": "Thyroid Stimulating Hormone regulates thyroid gland activity; used to screen thyroid function.",
        "ranges": {
            "adult": {"low": 0.4, "high": 4.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference interval",
    },
    "CO2 (Bicarbonate)": {
        "unit": "mmol/L",
        "aliases": ["co2", "bicarbonate", "hco3", "total co2"],
        "category": "Metabolic Panel",
        "description": "Bicarbonate reflects the body's acid-base (pH) balance, part of the basic metabolic panel.",
        "ranges": {
            "adult": {"low": 23.0, "high": 29.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference interval",
    },
    "ALT (SGPT)": {
        "unit": "U/L",
        "aliases": ["alt", "sgpt", "alanine aminotransferase"],
        "category": "Liver Panel",
        "description": "An enzyme mainly found in the liver; elevated levels can indicate liver cell damage.",
        "ranges": {
            "adult_male": {"low": 9.0, "high": 60.0},
            "adult_female": {"low": 7.0, "high": 45.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference intervals",
    },
    "Glucose, Fasting": {
        "unit": "mg/dL",
        "aliases": ["glucose fasting", "fasting glucose", "fasting blood sugar"],
        "category": "Metabolic Panel",
        "description": "Blood sugar level measured after fasting; used to screen for diabetes/prediabetes.",
        "ranges": {
            "adult": {"low": 70.0, "high": 99.0},
        },
        "source": "American Diabetes Association / Mayo Clinic reference interval",
    },
    "AST (SGOT)": {
        "unit": "U/L",
        "aliases": ["ast", "sgot", "aspartate aminotransferase"],
        "category": "Liver Panel",
        "description": "An enzyme found in the liver and muscles; elevated levels can indicate tissue damage.",
        "ranges": {
            "adult_male": {"low": 10.0, "high": 40.0},
            "adult_female": {"low": 9.0, "high": 32.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference intervals",
    },
    "WBC Count": {
        "unit": "10^3/uL",
        "aliases": ["wbc", "white blood cell count", "leukocyte count"],
        "category": "CBC",
        "description": "White blood cell count; part of the complete blood count, reflects immune/infection status.",
        "ranges": {
            "adult": {"low": 4.5, "high": 11.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference interval",
    },
    "Platelet Count": {
        "unit": "10^3/uL",
        "aliases": ["platelets", "plt", "platelet count"],
        "category": "CBC",
        "description": "Platelets are blood cells involved in clotting; counts outside range may indicate bleeding or clotting risk.",
        "ranges": {
            "adult": {"low": 150.0, "high": 450.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference interval",
    },
    "RBC Count": {
        "unit": "10^6/uL",
        "aliases": ["rbc", "red blood cell count", "erythrocyte count"],
        "category": "CBC",
        "description": "Red blood cell count; reflects oxygen-carrying capacity of the blood.",
        "ranges": {
            "adult_male": {"low": 4.7, "high": 6.1},
            "adult_female": {"low": 4.2, "high": 5.4},
        },
        "source": "Mayo Clinic Laboratories, adult reference intervals",
    },
    "Triglycerides": {
        "unit": "mg/dL",
        "aliases": ["triglycerides", "tg"],
        "category": "Lipid Panel",
        "description": "A type of fat in the blood; elevated levels are linked to cardiovascular risk.",
        "ranges": {
            "adult": {"low": 0.0, "high": 149.0},
        },
        "source": "NIH/NCEP ATP III lipid guidelines, Mayo Clinic reference interval",
    },
    "Sodium": {
        "unit": "mmol/L",
        "aliases": ["sodium", "na"],
        "category": "Metabolic Panel",
        "description": "An electrolyte critical for fluid balance and nerve/muscle function.",
        "ranges": {
            "adult": {"low": 135.0, "high": 145.0},
        },
        "source": "Mayo Clinic Laboratories, adult reference interval",
    },
    "Vitamin D, 25-OH": {
        "unit": "ng/mL",
        "aliases": ["vitamin d", "25-oh vitamin d", "25-hydroxyvitamin d"],
        "category": "Vitamins/Metabolic",
        "description": "Reflects the body's vitamin D stores; used to screen for deficiency.",
        "ranges": {
            "adult": {"low": 30.0, "high": 100.0},
        },
        "source": "Endocrine Society guidelines, Mayo Clinic reference interval",
    },
    "LDL Cholesterol": {
        "unit": "mg/dL",
        "aliases": ["ldl", "ldl cholesterol", "low density lipoprotein"],
        "category": "Lipid Panel",
        "description": "'Bad' cholesterol; elevated levels are associated with cardiovascular disease risk.",
        "ranges": {
            "adult": {"low": 0.0, "high": 99.0},
        },
        "source": "NIH/NCEP ATP III lipid guidelines",
    },
    "eGFR": {
        "unit": "mL/min/1.73m2",
        "aliases": ["egfr", "estimated glomerular filtration rate"],
        "category": "Renal Panel",
        "description": "Estimates how well the kidneys filter blood; used to assess kidney function/stage of disease.",
        "ranges": {
            "adult": {"low": 90.0, "high": 120.0},
        },
        "source": "National Kidney Foundation (KDIGO) reference interval",
    },
}

# ---------------------------------------------------------------------------
# 2. Pull distinct tests actually present in lab_results (sanity + coverage)
# ---------------------------------------------------------------------------

def get_distinct_tests(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT test_name, unit, reference_low, reference_high, COUNT(*) as n
        FROM lab_results
        GROUP BY test_name, unit, reference_low, reference_high
        ORDER BY test_name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def check_coverage_and_conflicts(db_rows):
    """
    Flags:
      - tests present in DB but missing from CURATED_REFERENCE (need sourcing)
      - tests where DB embeds MULTIPLE different ranges for the same test_name
        (silent range drift in the synthetic data)
      - tests where DB's single range doesn't match any curated demographic bucket
    """
    by_test = {}
    for test_name, unit, ref_low, ref_high, n in db_rows:
        by_test.setdefault(test_name, set()).add((unit, ref_low, ref_high))

    issues = []
    for test_name, variants in by_test.items():
        if test_name not in CURATED_REFERENCE:
            issues.append(f"[MISSING] '{test_name}' found in lab_results but not curated — add it.")
            continue
        if len(variants) > 1:
            issues.append(f"[DRIFT] '{test_name}' has {len(variants)} different (unit, low, high) "
                          f"combinations in lab_results: {variants}")
        curated_ranges = CURATED_REFERENCE[test_name]["ranges"].values()
        db_unit, db_low, db_high = next(iter(variants))[0], next(iter(variants))[1], next(iter(variants))[2]
        match = any(r["low"] == db_low and r["high"] == db_high for r in curated_ranges)
        if not match:
            issues.append(f"[MISMATCH] '{test_name}' DB range ({db_low}-{db_high} {db_unit}) "
                          f"doesn't match any curated demographic bucket "
                          f"{[ (r['low'], r['high']) for r in curated_ranges ]} — reconcile before shipping.")
    return issues


# ---------------------------------------------------------------------------
# 3. Build output JSON files for the MCP server
# ---------------------------------------------------------------------------

def build_tests_json():
    tests = []
    for test_name, meta in CURATED_REFERENCE.items():
        tests.append({
            "test_name": test_name,
            "aliases": meta["aliases"],
            "unit": meta["unit"],
            "category": meta["category"],
            "description": meta["description"],
        })
    return tests


def build_reference_ranges_json():
    ranges = []
    for test_name, meta in CURATED_REFERENCE.items():
        for demographic, r in meta["ranges"].items():
            ranges.append({
                "test_name": test_name,
                "demographic": demographic,
                "unit": meta["unit"],
                "low": r["low"],
                "high": r["high"],
                "source": meta["source"],
            })
    return ranges


def main():
    if not Path(DB_PATH).exists():
        print(f"ERROR: DB not found at {DB_PATH}. Update DB_PATH at top of script.", file=sys.stderr)
        sys.exit(1)

    db_rows = get_distinct_tests(DB_PATH)

    print(f"Found {len(db_rows)} distinct (test_name, unit, range) combinations in lab_results.\n")

    issues = check_coverage_and_conflicts(db_rows)
    if issues:
        print("=== COVERAGE / CONSISTENCY ISSUES ===")
        for i in issues:
            print(" -", i)
        print()
    else:
        print("No coverage or drift issues found — curated data fully covers lab_results.\n")

    tests_json = build_tests_json()
    ranges_json = build_reference_ranges_json()

    (OUT_DIR / "tests.json").write_text(json.dumps(tests_json, indent=2))
    (OUT_DIR / "reference_ranges.json").write_text(json.dumps(ranges_json, indent=2))

    print(f"Wrote {len(tests_json)} tests -> {OUT_DIR / 'tests.json'}")
    print(f"Wrote {len(ranges_json)} reference range entries -> {OUT_DIR / 'reference_ranges.json'}")


if __name__ == "__main__":
    main()