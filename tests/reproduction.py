"""Re-run one configuration of the evaluation and compare it with the paper.

Runs BRIDGE, GM and GS on jBilling with budget 1B and repetition 0, exactly as
`evaluation.experiment` does, and checks the verified rules, the objectives
fulfilled during the search, the evaluations used and the minimal suite sizes
against tests/expected/jbilling_rep0_1B.json. Takes about one minute.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run():
    with open(os.path.join(HERE, 'expected', 'jbilling_rep0_1B.json'), encoding='utf-8') as f:
        expected = json.load(f)
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONIOENCODING='utf-8')
    failures = []
    tmp = tempfile.mkdtemp()
    try:
        for setup, exp in expected['runs'].items():
            run_dir = os.path.join(tmp, setup)
            os.makedirs(run_dir)
            for args in (['_search', run_dir, 'jBilling', setup, '1', '0', str(exp['seed']),
                          str(exp['budget_evaluations'])],
                         ['_validate', run_dir, 'jBilling']):
                subprocess.run([sys.executable, '-m', 'evaluation.experiment'] + args, cwd=ROOT, env=env,
                               check=True, stdout=subprocess.DEVNULL)
            with open(os.path.join(run_dir, 'per_rule.csv'), newline='', encoding='utf-8') as f:
                per_rule = list(csv.DictReader(f))
            with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as f:
                meta = json.load(f)
            got = {
                'evaluations_used': meta['evaluations_used'],
                'verified_rules': sorted(p['rule_id'] for p in per_rule if p['verified_union'] == 'True'),
                'claimed_rules': sorted(p['rule_id'] for p in per_rule if p['claimed_union'] == 'True'),
                'suite_size': {v: s['min'] for v, s in meta['suite_size'].items()},
            }
            for key, value in got.items():
                if value != exp[key]:
                    failures.append(f'{setup}: {key} is {value}, expected {exp[key]}')
            print(f"{setup}: {len(got['verified_rules'])} rules verified, suite sizes {got['suite_size']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures
