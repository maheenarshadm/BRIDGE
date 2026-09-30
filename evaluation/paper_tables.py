"""Tables, figure data and numbers of the evaluation (RQ1-RQ3).

Reads the validated runs of an experiment folder (BRIDGE, GM and GS) and of
Schema-Random folders, and writes:

  results/tables/tab_rq1_coverage.tex   verified coverage per budget (RQ1, RQ2)
  results/tables/tab_rq1_stats.tex      A12 of BRIDGE against each baseline (RQ1, RQ2)
  results/tables/tab_rq1_rules.tex      non-trivial and unreached rules (RQ1)
  results/tables/tab_runs_<k>B.tex      verified coverage of every run (RQ1)
  results/tables/fig_coverage.tex       coverage during the base budget (RQ2)
  results/tables/tab_rq3_suite.tex      minimal test suites of BRIDGE's outputs (RQ3)
  results/tables/tab_rq3_same.tex       suite size of BRIDGE and each baseline for the same rules (RQ3)
  results/numbers/results.json          every number used in the text
  results/numbers/per_run.csv           coverage and suite sizes of every run

Usage:
  python -m evaluation.paper_tables --results outputs/paper --schema-random outputs/paper_schemarandom
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

from bridge import paths
from bridge.minimization import minimum_cover
from evaluation import statistics as S

CASE_ORDER = ('jBilling', 'OpenMRS', 'Spree')
CID = {'jBilling': 'C02', 'OpenMRS': 'C03', 'Spree': 'C04'}
TECH = ('dynamosa', 'random_walk', 'random_sample', 'schema_random')
NAME = {'dynamosa': 'BRIDGE', 'random_walk': 'GM', 'random_sample': 'GS', 'schema_random': 'SR'}
K = (1, 5, 10)
SHORT = {'Decision_': '', 'Selection': 'Sel.', 'Classification': 'Class.', 'Eligibility': 'Elig.'}
VARIANTS = ('archive', 'final', 'union')


def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _pool(run_dir):
    p = os.path.join(run_dir, 'matrix.csv')
    if not os.path.exists(p):
        return None
    return [set(x for x in r['rules_verified'].split('; ') if x) for r in read_csv(p)]


def load(results_dir, schema_random_dirs):
    runs = {}
    for cs in CASE_ORDER:
        for tech in TECH[:3]:
            for k in K:
                for d in sorted(glob.glob(os.path.join(results_dir, cs, tech, f'b{k}x', 'rep_*'))):
                    if not os.path.exists(os.path.join(d, 'VALIDATED')):
                        continue
                    pr = read_csv(os.path.join(d, 'per_rule.csv'))
                    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
                    ins = {p['rule_id'] for p in pr if p['in_scope'] == 'True'}
                    vv = {v: {p['rule_id'] for p in pr if p['in_scope'] == 'True' and p[f'verified_{v}'] == 'True'}
                          for v in VARIANTS if pr[0][f'verified_{v}'] != ''}
                    rule_of = {r['record_id']: r['rule_id'] for r in read_csv(os.path.join(d, 'per_record.csv'))}
                    tr = [(int(t['evaluations']), rule_of.get(t['record_id']))
                          for t in read_csv(os.path.join(d, 'trace.csv'))]
                    suite = {v: s['min'] for v, s in (meta.get('suite_size') or {}).items() if 'min' in s}
                    runs.setdefault((cs, tech, k), {})[int(d[-2:])] = {
                        'verified': vv['union'], 'verified_v': vv, 'in_scope': ins, 'suite': suite, 'trace': tr,
                        'pool': _pool(d), 'budget': meta['budget_evaluations']}
    for base in schema_random_dirs:
        for cs in CASE_ORDER:
            for k in K:
                for d in sorted(glob.glob(os.path.join(base, cs, 'schema_random', f'b{k}x', 'rep_*'))):
                    if not os.path.exists(os.path.join(d, 'VALIDATED')):
                        continue
                    pr = read_csv(os.path.join(d, 'per_rule.csv'))
                    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
                    ins = {p['rule_id'] for p in pr if p['in_scope'] == 'True'}
                    ver = {p['rule_id'] for p in pr if p['in_scope'] == 'True' and p['verified_union'] == 'True'}
                    tr = [(int(t['evaluations']), t['rule_id']) for t in read_csv(os.path.join(d, 'trace.csv'))]
                    runs.setdefault((cs, 'schema_random', k), {})[int(d[-2:])] = {
                        'verified': ver, 'verified_v': {'union': ver}, 'in_scope': ins,
                        'suite': {'union': meta['suite_size']['union']['min']}, 'trace': tr, 'pool': _pool(d)}
    return runs


def same_target_size(target, pool):
    if not target:
        return 0
    _greedy, minimum, _optimal = minimum_cover(target, {i: s & target for i, s in enumerate(pool)})
    return len(minimum)


def mean(xs):
    return st.mean(xs) if xs else float('nan')


class Fmt:
    def __init__(self, target):
        self.target = target

    def cell(self, value, n, nd=2, bold=False, mark=True):
        s = f'{float(value):.{nd}f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        if bold:
            s = r'\textbf{' + s + '}'
        return s if n >= self.target or not mark else s + r'$^{\dagger}$'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='merged experiment folder with the BRIDGE, GM and GS runs')
    ap.add_argument('--schema-random', nargs='*', default=[], help='Schema-Random output folders')
    ap.add_argument('--statistics', default=os.path.join(paths.ROOT, 'results', 'statistics'),
                    help='output folder of evaluation.statistics (normality and paired tests)')
    ap.add_argument('--out', default=os.path.join(paths.ROOT, 'results'))
    ap.add_argument('--runs', type=int, default=30, help='repetitions per configuration')
    ap.add_argument('--alpha', type=float, default=0.05)
    a = ap.parse_args()
    runs = load(a.results, a.schema_random)
    CS = tuple(cs for cs in CASE_ORDER if runs.get((cs, 'dynamosa', 1)))
    F = Fmt(a.runs)
    tables = os.path.join(a.out, 'tables')
    numbers = os.path.join(a.out, 'numbers')
    os.makedirs(tables, exist_ok=True)
    os.makedirs(numbers, exist_ok=True)

    short = {}
    for (cs, tech, _k), rr in runs.items():
        if cs in CS and len(rr) < a.runs:
            short.setdefault((NAME[tech], len(rr)), set()).add(cs)

    def note():
        if not short:
            return ''
        parts = [f'{n} runs for {t} in ' + ' and '.join(c for c in CS if c in css)
                 for (t, n), css in sorted(short.items())]
        return r'$^{\dagger}$ Based on ' + ', '.join(parts) + '.'

    res = {'n_runs': {}, 'verified': {}, 'suite': {}}

    def cellruns(cs, tech, k):
        return runs.get((cs, tech, k), {})

    in_scope = {cs: next(iter(cellruns(cs, 'dynamosa', 1).values()))['in_scope'] for cs in CS}

    groups = {}
    for cs in CS:
        allruns = [r for t in TECH[:3] for k in K for r in cellruns(cs, t, k).values()]
        triv = {x for x in in_scope[cs] if all(x in r['verified'] for r in allruns)}
        unr = {x for x in in_scope[cs] if not any(x in r['verified'] for r in allruns)}
        groups[cs] = {'trivial': triv, 'unreached': unr, 'nontrivial': in_scope[cs] - triv - unr}
    res['groups'] = {cs: {g: sorted(v) for g, v in d.items()} for cs, d in groups.items()}

    # RQ1/RQ2: verified coverage
    lines = []
    for cs in CS:
        row = [f'{CID[cs]} {cs}']
        pct = {}
        for k in K:
            for t in TECH:
                rr = cellruns(cs, t, k)
                v = [len(r['verified']) for r in rr.values()]
                m = mean(v)
                res['verified'][f'{cs}|{NAME[t]}|{k}B'] = {
                    'mean': m, 'pct': 100 * m / len(in_scope[cs]), 'min': min(v) if v else None,
                    'max': max(v) if v else None, 'n': len(v)}
                res['n_runs'][f'{cs}|{NAME[t]}|{k}B'] = len(v)
                pct[(t, k)] = (100 * m / len(in_scope[cs]), len(v))
        for k in K:
            best = max(pct[(t, k)][0] for t in TECH)
            for t in TECH:
                val, n = pct[(t, k)]
                row.append(F.cell(val, n, bold=abs(val - best) < 1e-9))
        lines.append(' & '.join(row) + ' \\\\')
    head = ('\\multirow{2}{*}{\\textbf{Case study}} & '
            + ' & '.join('\\multicolumn{4}{c|}{\\textbf{Budget $' + str(k) + 'B$}}' for k in K) + ' \\\\\n'
            + '\\cline{2-13}\n & ' + ' & '.join([' & '.join(NAME[t] for t in TECH)] * 3) + ' \\\\')
    with open(os.path.join(tables, 'tab_rq1_coverage.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n\\centering\n'
                f'\\caption{{Mean Verified Rule Coverage (\\%) over {a.runs} Runs - RQ1, RQ2}}\n'
                '\\label{tab:rq1}\n\\resizebox{\\textwidth}{!}{%\n'
                '\\begin{tabular}{|l|cccc|cccc|cccc|}\n\\hline\n' + head + '\n\\hline\n' + '\n'.join(lines) +
                '\n\\hline\n\\end{tabular}}\n\\\\[2pt]\n\\footnotesize{Best value per case study and budget '
                'in bold. ' + note() + '}\n\\end{table*}\n')

    comp = {}
    for cs in CS:
        block = []
        for k in K:
            x = [len(r['verified']) for r in cellruns(cs, 'dynamosa', k).values()]
            for t in TECH[1:]:
                y = [len(r['verified']) for r in cellruns(cs, t, k).values()]
                if len(y) < 2:
                    continue
                c = S.compare_independent(x, y, a.alpha)
                block.append(((cs, t, k, min(len(x), len(y))), c))
        for (key, c), ph in zip(block, S.holm([c['p'] for _, c in block])):
            c['p_holm'] = ph
            comp[key] = c
    lines = []
    for cs in CS:
        row = [f'{CID[cs]} {cs}']
        for k in K:
            for t in TECH[1:]:
                key = next((kk for kk in comp if kk[:3] == (cs, t, k)), None)
                if key is None:
                    row.append('--')
                    continue
                c = comp[key]
                s = f"{c['A12']:.2f}" + ('*' if c['p_holm'] < a.alpha else '')
                row.append(F.cell(s, key[3]))
                res.setdefault('tests_verified', {})[f'{cs}|{NAME[t]}|{k}B'] = {
                    'test': c['test'], 'A12': c['A12'], 'p_holm': c['p_holm']}
        lines.append(' & '.join(row) + r' \\')
    tests_used = sorted({c['test'] for c in comp.values()})
    with open(os.path.join(tables, 'tab_rq1_stats.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{table}[t]\n\\centering\n'
                '\\caption{$\\hat{A}_{12}$ of BRIDGE Against Each Baseline on Verified Coverage - RQ1, RQ2}\n'
                '\\label{tab:rq1stats}\n\\resizebox{\\columnwidth}{!}{%\n'
                '\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}\n\\hline\n'
                '\\multirow{2}{*}{\\textbf{Case study}} & '
                + ' & '.join('\\multicolumn{3}{c|}{\\textbf{Budget $' + str(k) + 'B$}}' for k in K) + ' \\\\\n'
                '\\cline{2-10}\n & ' + ' & '.join(['vs.\\ GM & vs.\\ GS & vs.\\ SR'] * 3) + ' \\\\\n\\hline\n'
                + '\n'.join(lines) + '\n\\hline\n\\end{tabular}}\n\\\\[2pt]\n'
                '\\footnotesize{* Significant after Holm correction ($p < 0.05$). Test used: '
                + ', '.join(tests_used) + ', chosen by the Shapiro-Wilk test. ' + note() + '}\n'
                '\\end{table}\n')

    lines = []
    for cs in CS:
        for rule in sorted(groups[cs]['nontrivial']) + sorted(groups[cs]['unreached']):
            dec, _, num = rule.replace('Decision_', '').rpartition('_')
            dec = dec.rsplit('_', 1)[0] if dec.lower().endswith('_rule') else dec
            words = re.sub(r'(?<!^)(?=[A-Z])', ' ', dec)
            for x, y in SHORT.items():
                words = words.replace(x, y)
            row = [CID[cs], f'{words}, R{num}' + (' $^{u}$' if rule in groups[cs]['unreached'] else '')]
            for k in (1, 10):
                for t in TECH:
                    rr = cellruns(cs, t, k)
                    c = sum(rule in r['verified'] for r in rr.values())
                    row.append(F.cell(f'{c}/{len(rr)}' if len(rr) != a.runs else str(c), len(rr), mark=False))
            lines.append(' & '.join(row) + r' \\')
        lines.append(r'\hline')
    with open(os.path.join(tables, 'tab_rq1_rules.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n\\centering\n'
                f'\\caption{{Non-Trivial and Unreached Business Rules (Runs Out of {a.runs} That Verify the Rule) - RQ1}}\n'
                '\\label{tab:rq1rules}\n\\resizebox{0.9\\textwidth}{!}{%\n'
                '\\begin{tabular}{|l|l|c|c|c|c|c|c|c|c|}\n\\hline\n'
                '\\multirow{2}{*}{\\textbf{CS}} & \\multirow{2}{*}{\\textbf{Decision, rule}} & '
                + ' & '.join('\\multicolumn{4}{c|}{\\textbf{Budget $' + str(k) + 'B$}}' for k in (1, 10)) + ' \\\\\n'
                '\\cline{3-10}\n & & ' + ' & '.join([' & '.join(NAME[t] for t in TECH)] * 2) + ' \\\\\n\\hline\n'
                + '\n'.join(lines) + '\n\\end{tabular}}\n\\\\[2pt]\n'
                '\\footnotesize{$^{u}$ Unreached by BRIDGE, GM and GS. All other in-scope rules are trivial '
                '(verified by BRIDGE, GM and GS in every run). Counts shown as $x/n$ are out of $n$ runs.}\n'
                '\\end{table*}\n')

    res['sr_groups'] = {}
    for cs in CS:
        for k in (1, 10):
            rr = cellruns(cs, 'schema_random', k).values()
            if rr:
                res['sr_groups'][f'{cs}|{k}B'] = {g: mean([len(groups[cs][g] & r['verified']) for r in rr])
                                                 for g in ('trivial', 'nontrivial', 'unreached')}
    res['sr_beyond_bridge'] = sorted(
        f'{cs}|{x}' for cs in CS for x in in_scope[cs]
        if any(x in r['verified'] for k in K for r in cellruns(cs, 'schema_random', k).values())
        and not any(x in r['verified'] for k in K for r in cellruns(cs, 'dynamosa', k).values()))

    res['budget_effect'] = {}
    for cs in CS:
        block = []
        for t in TECH:
            x = [len(r['verified']) for r in cellruns(cs, t, 10).values()]
            y = [len(r['verified']) for r in cellruns(cs, t, 1).values()]
            if len(x) < 2 or len(y) < 2:
                continue
            block.append((t, mean(y), mean(x), S.compare_independent(x, y, a.alpha)))
        for (t, m1, m10, c), ph in zip(block, S.holm([c['p'] for *_, c in block])):
            res['budget_effect'][f'{cs}|{NAME[t]}'] = {'mean_1B': m1, 'mean_10B': m10, 'test': c['test'],
                                                       'A12_10B_gt_1B': c['A12'], 'p_holm': ph}

    # RQ2: coverage during the base budget
    X = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    curves = {}
    for cs in CS:
        n = len(in_scope[cs])
        B = next(iter(cellruns(cs, 'dynamosa', 1).values()))['budget']
        for t in TECH:
            rr = cellruns(cs, t, 1)
            if not rr:
                continue
            cs_curves = []
            for r in rr.values():
                seen, pts = set(), []
                for ev, rule in r['trace']:
                    if rule in in_scope[cs] and rule not in seen:
                        seen.add(rule)
                        pts.append((ev, len(seen)))
                cs_curves.append([max([y for e, y in pts if e <= p * B / 100] or [0]) for p in X])
            curves[f'{cs}|{NAME[t]}'] = [100 * mean([c[i] for c in cs_curves]) / n for i in range(len(X))]
    res['curves'] = {'x_percent_of_B': X, **curves}
    style = {'BRIDGE': 'blue', 'GM': 'red, dashed', 'GS': 'black!60, dotted', 'SR': 'orange, dashdotted'}
    fig = ['\\begin{figure*}[t]', '\\centering', '\\begin{tikzpicture}', '\\begin{groupplot}[',
           f'  group style={{group size={len(CS)} by 1, horizontal sep=1.0cm}},',
           '  width=0.36\\textwidth, height=4cm, xmin=0, xmax=100, ymin=0, ymax=100,',
           '  xtick={0,50,100}, ytick={0,50,100}, tick label style={font=\\scriptsize},',
           '  title style={font=\\footnotesize, yshift=-4pt}, xlabel style={font=\\scriptsize},',
           '  ylabel style={font=\\scriptsize}, every axis plot/.append style={thick, mark=none}]']
    for i, cs in enumerate(CS):
        opts = [f'title={{{CID[cs]} {cs}}}', 'xlabel={Budget used (\\% of $B$)}']
        if i == 0:
            opts.append('ylabel={Coverage (\\%)}')
        if i == len(CS) - 1:
            opts.append('legend style={font=\\scriptsize, at={(1.0,0.03)}, anchor=south east, draw=none, '
                        'fill=white, fill opacity=0.8}')
        fig.append('\\nextgroupplot[' + ', '.join(opts) + ']')
        for t in TECH:
            key = f'{cs}|{NAME[t]}'
            if key in curves:
                pts = ' '.join(f'({x},{y:.1f})' for x, y in zip(X, curves[key]))
                fig.append(f'\\addplot[{style[NAME[t]]}] coordinates {{{pts}}};')
        if i == len(CS) - 1:
            fig.append('\\legend{' + ', '.join(NAME[t] for t in TECH) + '}')
    fig += ['\\end{groupplot}', '\\end{tikzpicture}',
            '\\caption{Mean rule coverage during the base budget $B$ (budget $1B$). BRIDGE, GM and GS: rules '
            'whose objective is fulfilled during the search. SR: rules confirmed by the validator. - RQ2}',
            '\\label{fig:coverage}', '\\end{figure*}']
    with open(os.path.join(tables, 'fig_coverage.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(fig) + '\n')

    # RQ3: minimal test suites
    for cs in CS:
        for t in TECH:
            for k in K:
                rr = cellruns(cs, t, k)
                for v in VARIANTS:
                    s = [r['suite'].get(v) for r in rr.values() if r['suite'].get(v) is not None]
                    if s:
                        res['suite'][f'{cs}|{NAME[t]}|{k}B|{v}'] = {'mean': mean(s), 'n': len(s)}
    lines = []
    for cs in CS:
        first = True
        for k in K:
            rr = cellruns(cs, 'dynamosa', k)
            row = [f'\\multirow{{3}}{{*}}{{{CID[cs]} {cs}}}' if first else '', f'${k}B$']
            first = False
            sizes = {v: mean([r['suite'][v] for r in rr.values()]) for v in VARIANTS}
            covs = {v: mean([100.0 * len(r['verified_v'][v]) / len(in_scope[cs]) for r in rr.values()])
                    for v in VARIANTS}
            best_s = min(sizes.values())
            best_c = max(covs.values())
            row += [F.cell(sizes[v], len(rr), bold=abs(sizes[v] - best_s) < 1e-9) for v in VARIANTS]
            row += [F.cell(covs[v], len(rr), bold=abs(covs[v] - best_c) < 1e-9) for v in VARIANTS]
            lines.append(' & '.join(row) + ' \\\\')
        lines.append('\\hline')
    with open(os.path.join(tables, 'tab_rq3_suite.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{table}[t]\n\\centering\n'
                f'\\caption{{Minimal Test Suite Size and Verified Coverage per Output of BRIDGE (Mean of {a.runs} Runs) - RQ3}}\n'
                '\\label{tab:rq3}\n\\resizebox{\\columnwidth}{!}{%\n'
                '\\begin{tabular}{|l|c|ccc|ccc|}\n\\hline\n'
                '\\multirow{2}{*}{\\textbf{Case study}} & \\multirow{2}{*}{\\textbf{Budget}} & '
                '\\multicolumn{3}{c|}{\\textbf{Suite size (databases)}} & '
                '\\multicolumn{3}{c|}{\\textbf{Verified coverage (\\%)}} \\\\\n\\cline{3-8}\n'
                ' & & \\textbf{A} & \\textbf{F} & \\textbf{U} & \\textbf{A} & \\textbf{F} & \\textbf{U} \\\\\n'
                '\\hline\n' + '\n'.join(lines) +
                '\n\\end{tabular}}\n\\\\[2pt]\n\\footnotesize{A: archive. F: final population. U: union of both. '
                'Suite size: the smallest set of the output\'s databases that verifies all rules the output '
                'verifies. Smallest suite and highest coverage per row in bold.}\n\\end{table}\n')

    res['same_target'] = {}
    same = {}
    for cs in CS:
        block = []
        for k in K:
            for t in TECH[1:]:
                b_runs, o_runs = cellruns(cs, 'dynamosa', k), cellruns(cs, t, k)
                xb, xo, tg = [], [], []
                for rep in sorted(set(b_runs) & set(o_runs)):
                    pb, po = b_runs[rep]['pool'], o_runs[rep]['pool']
                    if pb is None or po is None:
                        continue
                    target = b_runs[rep]['verified'] & o_runs[rep]['verified']
                    xb.append(same_target_size(target, pb))
                    xo.append(same_target_size(target, po))
                    tg.append(100.0 * len(target) / len(in_scope[cs]))
                if len(xb) < 2:
                    continue
                c = S.compare_paired(xb, xo, a.alpha)
                block.append(((cs, k, t), xb, xo, tg, c))
        for (key, xb, xo, tg, c), ph in zip(block, S.holm([c['p'] for *_, c in block])):
            c['p_holm'] = ph
            same[key] = (xb, xo, tg, c)
            res['same_target'][f'{key[0]}|{key[1]}B|{NAME[key[2]]}'] = {
                'bridge': mean(xb), 'baseline': mean(xo), 'target_coverage_pct': mean(tg), 'n': len(xb),
                'test': c['test'], 'p_holm': ph}
    lines = []
    for cs in CS:
        row = [f'{CID[cs]} {cs}']
        for k in K:
            for t in TECH[1:]:
                if (cs, k, t) not in same:
                    row.append('--')
                    continue
                xb, xo, tg, c = same[(cs, k, t)]
                mb, mo = mean(xb), mean(xo)
                sb = f'{mb:.2f}' if mb >= mo - 1e-9 else r'\textbf{' + f'{mb:.2f}' + '}'
                so = f'{mo:.2f}' if mo >= mb - 1e-9 else r'\textbf{' + f'{mo:.2f}' + '}'
                cell = f'{sb} / {so}' + ('*' if c['p_holm'] < a.alpha else '')
                row.append(F.cell(cell, len(xb)))
        lines.append(' & '.join(row) + ' \\\\')
    tests = sorted({v[3]['test'] for v in same.values()})
    with open(os.path.join(tables, 'tab_rq3_same.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n\\centering\n'
                '\\caption{Minimal Number of Databases Needed by BRIDGE and Each Baseline for the Same Rules '
                f'(BRIDGE / Baseline, Mean of {a.runs} Runs) - RQ3}}\n'
                '\\label{tab:rq3same}\n\\resizebox{\\textwidth}{!}{%\n'
                '\\begin{tabular}{|l|ccc|ccc|ccc|}\n\\hline\n'
                '\\multirow{2}{*}{\\textbf{Case study}} & '
                + ' & '.join('\\multicolumn{3}{c|}{\\textbf{Budget $' + str(k) + 'B$}}' for k in K) + ' \\\\\n'
                '\\cline{2-10}\n & ' + ' & '.join(['vs.\\ GM & vs.\\ GS & vs.\\ SR'] * 3) + ' \\\\\n\\hline\n'
                + '\n'.join(lines) + '\n\\hline\n\\end{tabular}}\n\\\\[2pt]\n'
                '\\footnotesize{For each run, the target is the set of rules verified by both BRIDGE and the '
                'baseline, and each value is the smallest number of the technique\'s own databases that verify '
                'all of them. The smaller value is in bold. * Significant after Holm correction ($p < 0.05$), '
                'test chosen by the Shapiro-Wilk test on the paired differences (' + ', '.join(tests) + '). '
                + note() + '}\n\\end{table*}\n')

    res['per_output'] = {}
    for cs in CS:
        for t in ('dynamosa', 'random_walk'):
            for k in K:
                rr = cellruns(cs, t, k)
                res['per_output'][f'{cs}|{NAME[t]}|{k}B'] = {
                    v: mean([len(r['verified_v'][v]) for r in rr.values()]) for v in VARIANTS}
    paired_csv = os.path.join(a.statistics, 'paired.csv')
    res['paired'] = read_csv(paired_csv) if os.path.exists(paired_csv) else []
    normality_csv = os.path.join(a.statistics, 'normality.csv')
    summary = {}
    for r in (read_csv(normality_csv) if os.path.exists(normality_csv) else []):
        if r['variant'] == 'union':
            summary.setdefault(r['technique'], {}).setdefault(r['normality'], 0)
            summary[r['technique']][r['normality']] += 1
    res['normality_summary'] = summary

    # verified coverage of every run
    per_run = []
    for k in K:
        reps = list(range(a.runs))
        cols = [(cs, t) for cs in CS for t in TECH]
        per = {}
        for cs in CS:
            for t in TECH:
                for rep, r in cellruns(cs, t, k).items():
                    per[(cs, t, rep)] = 100.0 * len(r['verified']) / len(in_scope[cs])
                    per_run.append({'case_study': cs, 'technique': NAME[t], 'budget': f'{k}B', 'run': rep + 1,
                                    'verified_rules': len(r['verified']), 'in_scope_rules': len(in_scope[cs]),
                                    'coverage_pct': round(per[(cs, t, rep)], 4),
                                    **{f'suite_size_{v}': r['suite'].get(v, '') for v in VARIANTS}})
        body = []
        for rep in reps:
            row = [str(rep + 1)]
            for cs, t in cols:
                v = per.get((cs, t, rep))
                row.append('--' if v is None else F.cell(v, a.runs, mark=False))
            body.append(' & '.join(row) + r' \\')
        foot = []
        for label, fn in (('Avg', st.mean), ('Min', min), ('Max', max)):
            row = [r'\textbf{' + label + '}']
            for cs, t in cols:
                vals = [per[(cs, t, rep)] for rep in reps if (cs, t, rep) in per]
                row.append(F.cell(fn(vals), len(vals), bold=True) if vals else '--')
            foot.append(' & '.join(row) + r' \\')
        head1 = r'\multirow{2}{*}{\textbf{Run}} & ' + ' & '.join(
            r'\multicolumn{4}{c|}{\textbf{' + f'{CID[cs]} {cs}' + '}}' for cs in CS) + r' \\'
        head2 = r'\cline{2-' + str(len(cols) + 1) + '}\n & ' + ' & '.join(NAME[t] for _cs, t in cols) + r' \\'
        with open(os.path.join(tables, f'tab_runs_{k}B.tex'), 'w', encoding='utf-8') as f:
            f.write('\\begin{sidewaystable*}\n\\centering\n'
                    f'\\caption{{Verified Rule Coverage (\\%) of Each Case Study for {a.runs} Runs with '
                    f'Budget ${k}B$ - RQ1}}\n\\label{{tab:runs{k}B}}\n\\resizebox{{\\textheight}}{{!}}{{%\n'
                    '\\begin{tabular}{|c|' + '|'.join(['cccc'] * len(CS)) + '|}\n\\hline\n' + head1 + '\n' + head2 +
                    '\n\\hline\n' + '\n'.join(body) + '\n\\hline\n' + '\n'.join(foot) +
                    '\n\\hline\n\\end{tabular}}\n\\\\[2pt]\n\\footnotesize{--: run not available. ' + note() + '}\n'
                    '\\end{sidewaystable*}\n')

    with open(os.path.join(numbers, 'per_run.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)
    with open(os.path.join(numbers, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=str)
    print(f'Wrote {tables} and {numbers}')


if __name__ == '__main__':
    main()
