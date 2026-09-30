"""Loads the compiled objectives of a case study, grouped by decision."""
import json

from bridge import paths

COMPILED_CONSTRAINTS_PATH = paths.COMPILED_PATH


def load_records(case_study, path=COMPILED_CONSTRAINTS_PATH):
    with open(path) as f:
        compiled = json.load(f)
    return [r for r in compiled if r['case_study'] == case_study]


def records_by_decision(case_study, path=COMPILED_CONSTRAINTS_PATH):
    by_decision = {}
    for r in load_records(case_study, path):
        by_decision.setdefault(r['decision_name'], []).append(r)
    return by_decision


