"""Command line of BRIDGE.

  python -m bridge compile [--case-studies jBilling Spree OpenMRS]
      Phase 1: compile the DMN models of the case studies into search objectives
      (case_studies/compiled_constraints.json).

  python -m bridge generate --case-study jBilling --budget 78970 [--seed 0] [--out DIR]
      Phases 2 and 3: search, build and validate the databases, and write the
      minimal test suite as SQLite databases with a coverage report.
"""
import argparse
import json
import os
import subprocess
import sys

from bridge import paths


def main():
    ap = argparse.ArgumentParser(prog='python -m bridge', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('compile', help='compile the DMN models into search objectives')
    p.add_argument('--case-studies', nargs='+', default=list(paths.CASE_STUDIES), choices=paths.CASE_STUDIES)
    p = sub.add_parser('generate', help='generate the test databases of one case study')
    p.add_argument('--case-study', required=True, choices=paths.CASE_STUDIES)
    p.add_argument('--budget', type=int, required=True, help='number of fitness evaluations')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='output folder (default: outputs/<case study>_seed<seed>)')
    a = ap.parse_args()

    if a.command == 'compile':
        from bridge.grounding.compile_constraints import compile_case_studies
        records = compile_case_studies(a.case_studies)
        print(f'Wrote {len(records)} objectives to {paths.COMPILED_PATH}')
        return

    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ, PYTHONHASHSEED='0')
        sys.exit(subprocess.run([sys.executable, '-m', 'bridge'] + sys.argv[1:], env=env).returncode)
    from bridge.pipeline import generate
    out = a.out or os.path.join(paths.ROOT, 'outputs', f'{a.case_study.lower()}_seed{a.seed}')
    os.makedirs(out, exist_ok=True)
    summary = generate(a.case_study, a.budget, a.seed, out)
    print(json.dumps(summary, indent=2))
    print(f'Minimal test suite written to {os.path.join(out, "minimal_suite")}')


if __name__ == '__main__':
    main()
