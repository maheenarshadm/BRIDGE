"""Regenerate the statistics, tables and numbers of the evaluation from the runs.

  python reproduce.py --results outputs/paper --schema-random outputs/paper_schemarandom

`--results` is the merged experiment folder with the BRIDGE, GM and GS runs
(after `python -m evaluation.experiment summarize`), and `--schema-random`
the folder(s) with the Schema-Random runs. Output goes to results/.
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--schema-random', nargs='*', default=[])
    a = ap.parse_args()
    sr = ['--schema-random'] + a.schema_random if a.schema_random else []
    for module in ('evaluation.statistics', 'evaluation.paper_tables'):
        subprocess.run([sys.executable, '-m', module, '--results', a.results] + sr, check=True)


if __name__ == '__main__':
    main()
