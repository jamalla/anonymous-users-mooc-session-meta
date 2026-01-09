"""
Replace all theoretical expectations with actual results from report.json
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# ACTUAL RESULTS from report.json (source of truth)
ACTUAL_RESULTS = {
    "gru_baseline": "33.73%",
    "maml_zero_shot": "23.50%",
    "maml_few_shot_k5": "30.52%",
    "maml_few_shot_k10": "31.33%",
    "improvement": "-9.51%",  # Negative = worse than baseline
    "iterations": "9,999"
}

# Files to fix
FILES_TO_FIX = [
    "NOTEBOOK_07_COMPLETE_FIX.md",
    "NOTEBOOK_07_ISSUES_AND_FIX.md",
    "NOTEBOOK_07_MAML_READY.md",
    "QUICK_START_07b.md",
    "docs/meta_learning_architecture_justification.md"
]

def fix_file(filepath):
    """Replace fake expectations with actual results"""
    print(f"\nFixing: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Replace patterns
    replacements = [
        # Pattern: "38-43%" or "~38-43%"
        (r'~?38-43%', '30.52%'),
        (r'~?0\.38-0\.43', '0.3052'),

        # Pattern: "32-35%" (zero-shot)
        (r'~?32-35%', '23.50%'),
        (r'~?0\.32-0\.35', '0.2350'),

        # Pattern: ">38%" or "38%"
        (r'>38%', '>30.5%'),
        (r'38%', '30.5%'),

        # Pattern: "+5-15%" improvement
        (r'\+5-15%', '-9.5%'),

        # Pattern: "beat baseline" or "beaten the baseline"
        (r'(If.*?)beat.*baseline.*🎉', r'\1match baseline target (actual: 30.52% vs 33.73% baseline = -9.5%)'),
        (r'You\'ve beaten the baseline!', 'Result: 30.52% vs 33.73% baseline (-9.5%)'),

        # Pattern: specific wrong numbers in expected sections
        (r'Expected:.*?~?38-43%.*?Acc@1', 'Expected: 30.52% Acc@1 (ACTUAL from report.json)'),
        (r'Expected:.*?~?32-35%.*?Acc@1', 'Expected: 23.50% Acc@1 (ACTUAL from report.json)'),

        # Pattern: "28% → 34% → 38%" progression
        (r'28% → 34% → 38%', '20% → 25% → 30% (actual progression may vary)'),

        # Pattern: target arrows
        (r'33\.73% → 38-43%', '33.73% baseline vs 30.52% MAML (actual)'),
        (r'33\.73% → target: 38-43%', '33.73% baseline vs 30.52% MAML'),

        # Pattern: "Success Criteria" sections
        (r'Success Criteria.*?>38%', 'Actual Result: 30.52% (below 33.73% baseline)'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    # Add warning header to files
    if filepath.name in FILES_TO_FIX:
        warning = "<!-- WARNING: This document contained theoretical expectations that have been replaced with ACTUAL results from report.json -->\n\n"
        if not content.startswith("<!--"):
            content = warning + content

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] Fixed {filepath.name}")
        return True
    else:
        print(f"  - No changes needed in {filepath.name}")
        return False

def main():
    print("=" * 80)
    print("FIXING FAKE EXPECTATIONS WITH ACTUAL RESULTS")
    print("=" * 80)
    print(f"\nACTUAL RESULTS (source: report.json):")
    for key, value in ACTUAL_RESULTS.items():
        print(f"  - {key}: {value}")

    fixed_count = 0
    for filename in FILES_TO_FIX:
        filepath = REPO_ROOT / filename
        if filepath.exists():
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"\n[WARNING] File not found: {filepath}")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Fixed {fixed_count} files")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
