"""Smoke tests of the pipeline on the jBilling case study."""
import json
import os
import sqlite3
import tempfile

from bridge import paths, pipeline
from bridge.grounding.compile_constraints import compile_case_studies


def test_compilation_reproduces_objectives():
    with open(paths.COMPILED_PATH, encoding='utf-8') as f:
        committed = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'compiled_constraints.json')
        compile_case_studies(out_path=out)
        with open(out, encoding='utf-8') as f:
            fresh = json.load(f)
    assert fresh == committed


def test_generate_small_budget():
    with tempfile.TemporaryDirectory() as tmp:
        summary = pipeline.generate('jBilling', budget=3000, seed=0, out_dir=tmp)
        assert summary['rules'] == len({r['rule_id'] for r in pipeline.load_records('jBilling')})
        assert 0 < summary['rules_verified'] <= summary['rules']
        suite = sorted(os.listdir(os.path.join(tmp, 'minimal_suite')))
        assert len(suite) == summary['minimal_suite_size'] > 0
        with open(os.path.join(tmp, 'coverage.json'), encoding='utf-8') as f:
            coverage = json.load(f)
        assert set(coverage) == set(suite)
        verified = set().union(*map(set, coverage.values()))
        assert len(verified) == summary['rules_verified']
        conn = sqlite3.connect(os.path.join(tmp, 'minimal_suite', suite[0]))
        try:
            tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
        finally:
            conn.close()
        assert tables > 0
