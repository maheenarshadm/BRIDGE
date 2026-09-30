"""Statistical comparison of the techniques.

Every sample is first tested for normality with the Shapiro-Wilk test
(Royston's algorithm AS R94). The comparison test is then chosen per
comparison:

  independent samples (BRIDGE vs a baseline, or one technique at two budgets):
      both samples normal (Shapiro-Wilk p >= alpha)  -> Welch's t-test
      otherwise                                      -> two-sided Mann-Whitney U
  paired samples (archive vs final population of the same runs):
      differences normal                             -> paired t-test
      otherwise                                      -> Wilcoxon signed-rank

A sample whose values are all identical has no variance and is treated as not
normally distributed. The effect size is the Vargha-Delaney A12 statistic.
p-values are adjusted with the Holm correction within each case study and
metric.

Usage:
  python -m evaluation.statistics --results outputs/paper --schema-random outputs/paper_schemarandom
Writes normality.csv, comparisons.csv and paired.csv to results/statistics/.
"""
import argparse
import csv
import json
import math
import os
import statistics as st

from bridge import paths

NAME = {'dynamosa': 'BRIDGE', 'random_walk': 'GM', 'random_sample': 'GS', 'schema_random': 'SR'}


def _phi_sf2(z):
    return math.erfc(abs(z) / math.sqrt(2))


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def mann_whitney_u(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    ranks, ties = _ranks(list(x) + list(y))
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    mean = n1 * n2 / 2
    var = n1 * n2 / 12 * ((n + 1) - sum(t ** 3 - t for t in ties) / (n * (n - 1)))
    if var <= 0:
        return u1, 1.0
    z = (abs(u1 - mean) - 0.5) / math.sqrt(var)
    return u1, min(1.0, _phi_sf2(max(z, 0.0)))


def a12(x, y):
    if not x or not y:
        return float('nan')
    gt = sum(1 for a in x for b in y if a > b)
    eq = sum(1 for a in x for b in y if a == b)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def wilcoxon_signed_rank(x, y):
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return 0.0, 0, 1.0
    ranks, ties = _ranks([abs(v) for v in d])
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties) / 48
    if var <= 0:
        return w_plus, n, 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    return w_plus, n, min(1.0, _phi_sf2(max(z, 0.0)))


def holm(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p):
    if not 0.0 < p < 1.0:
        raise ValueError(p)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _betacf(a, b, x):
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def _poly(c, x):
    return sum(ci * x ** i for i, ci in enumerate(c))


def shapiro_wilk(x):
    x = sorted(float(v) for v in x)
    n = len(x)
    if n < 3 or x[-1] - x[0] == 0:
        return None
    if n > 5000:
        raise ValueError('Shapiro-Wilk (AS R94) is defined for n <= 5000')
    mean = sum(x) / n
    ssq = sum((v - mean) ** 2 for v in x)
    if n == 3:
        a = [-math.sqrt(0.5), 0.0, math.sqrt(0.5)]
        w = (sum(ai * xi for ai, xi in zip(a, x))) ** 2 / ssq
        w = max(w, 0.75)
        p = max(0.0, min(1.0, 6.0 / math.pi * (math.asin(math.sqrt(w)) - math.asin(math.sqrt(0.75)))))
        return w, p
    m = [norm_ppf((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)]
    mm = sum(v * v for v in m)
    u = 1.0 / math.sqrt(n)
    c1 = [0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056]
    c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]
    a = [0.0] * n
    an = m[-1] / math.sqrt(mm) + _poly(c1, u)
    if n > 5:
        an1 = m[-2] / math.sqrt(mm) + _poly(c2, u)
        phi = (mm - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * an ** 2 - 2 * an1 ** 2)
        for i in range(2, n - 2):
            a[i] = m[i] / math.sqrt(phi)
        a[-1], a[0], a[-2], a[1] = an, -an, an1, -an1
    else:
        phi = (mm - 2 * m[-1] ** 2) / (1 - 2 * an ** 2)
        for i in range(1, n - 1):
            a[i] = m[i] / math.sqrt(phi)
        a[-1], a[0] = an, -an
    w = (sum(ai * xi for ai, xi in zip(a, x))) ** 2 / ssq
    w = min(w, 1.0)
    if w >= 1.0:
        return 1.0, 1.0
    y = math.log(1.0 - w)
    if n <= 11:
        gamma = -2.273 + 0.459 * n
        if y >= gamma:
            return w, 0.0
        y = -math.log(gamma - y)
        mu = _poly([0.5440, -0.39978, 0.025054, -0.0006714], n)
        sigma = math.exp(_poly([1.3822, -0.77857, 0.062767, -0.0020322], n))
    else:
        ln = math.log(n)
        mu = _poly([-1.5861, -0.31082, -0.083751, 0.0038915], ln)
        sigma = math.exp(_poly([-0.4803, -0.082676, 0.0030302], ln))
    return w, 1.0 - norm_cdf((y - mu) / sigma)


def welch_t(x, y):
    n1, n2 = len(x), len(y)
    v1, v2 = st.variance(x), st.variance(y)
    se2 = v1 / n1 + v2 / n2
    if se2 == 0:
        return None
    t = (st.mean(x) - st.mean(y)) / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return t, df, t_two_sided_p(t, df)


def paired_t(d):
    n = len(d)
    sd = st.stdev(d)
    if sd == 0:
        return None
    t = st.mean(d) / (sd / math.sqrt(n))
    return t, n - 1, t_two_sided_p(t, n - 1)


def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    sp = math.sqrt(((n1 - 1) * st.variance(x) + (n2 - 1) * st.variance(y)) / (n1 + n2 - 2))
    return (st.mean(x) - st.mean(y)) / sp if sp else float('nan')


def normality(sample, alpha):
    if len(sample) < 3:
        return 'too-small', '', ''
    r = shapiro_wilk(sample)
    if r is None:
        return 'constant', '', ''
    w, p = r
    return ('normal' if p >= alpha else 'non-normal'), w, p


def compare_independent(x, y, alpha):
    nx, ny = normality(x, alpha)[0], normality(y, alpha)[0]
    effect = a12(x, y)
    if nx == 'normal' and ny == 'normal':
        r = welch_t(x, y)
        if r is not None:
            return {'test': 'Welch t-test', 'statistic': r[0], 'df': r[1], 'p': r[2],
                    'A12': effect, 'cohens_d': cohens_d(x, y)}
    u, p = mann_whitney_u(x, y)
    return {'test': 'Mann-Whitney U', 'statistic': u, 'df': '', 'p': p, 'A12': effect, 'cohens_d': ''}


def compare_paired(x, y, alpha):
    d = [a - b for a, b in zip(x, y)]
    label = normality(d, alpha)[0]
    if all(v == 0 for v in d):
        return {'test': 'identical (no test)', 'statistic': '', 'df': '', 'p': 1.0, 'normality_of_differences': 'constant'}
    if label == 'normal':
        r = paired_t(d)
        if r is not None:
            return {'test': 'paired t-test', 'statistic': r[0], 'df': r[1], 'p': r[2],
                    'normality_of_differences': label}
    w, nz, p = wilcoxon_signed_rank(x, y)
    return {'test': 'Wilcoxon signed-rank', 'statistic': w, 'df': '', 'p': p, 'normality_of_differences': label}


def _read_csv(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_samples(results_dir, schema_random_dirs):
    data = {}
    for r in _read_csv(os.path.join(results_dir, 'summary.csv')):
        for metric, col in (('verified', 'verified'), ('suite_size', 'suite_size_min')):
            if r.get(col, '') != '':
                data.setdefault((r['case_study'], r['setup'], int(r['budget_multiplier']), r['variant'], metric),
                                {})[int(r['rep'])] = float(r[col])
    for base in schema_random_dirs:
        for cs in sorted(os.listdir(base)):
            sdir = os.path.join(base, cs, 'schema_random')
            if not os.path.isdir(sdir):
                continue
            for bdir in os.listdir(sdir):
                k = int(bdir[1:-1])
                for rdir in os.listdir(os.path.join(sdir, bdir)):
                    d = os.path.join(sdir, bdir, rdir)
                    if not os.path.exists(os.path.join(d, 'VALIDATED')):
                        continue
                    rep = int(rdir[4:])
                    pr = _read_csv(os.path.join(d, 'per_rule.csv'))
                    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
                    ver = sum(p['in_scope'] == 'True' and p['verified_union'] == 'True' for p in pr)
                    data.setdefault((cs, 'schema_random', k, 'union', 'verified'), {})[rep] = float(ver)
                    data.setdefault((cs, 'schema_random', k, 'union', 'suite_size'), {})[rep] = \
                        float(meta['suite_size']['union']['min'])
    return data


def _fmt(v, nd=4):
    return f'{v:.{nd}g}' if isinstance(v, float) else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='merged experiment folder (after `experiment summarize`)')
    ap.add_argument('--schema-random', nargs='*', default=[], help='Schema-Random output folders')
    ap.add_argument('--out', default=os.path.join(paths.ROOT, 'results', 'statistics'))
    ap.add_argument('--alpha', type=float, default=0.05)
    a = ap.parse_args()
    data = load_samples(a.results, a.schema_random)
    out = a.out
    os.makedirs(out, exist_ok=True)

    norm_rows = []
    for (cs, setup, k, variant, metric), reps in sorted(data.items()):
        vals = [reps[r] for r in sorted(reps)]
        label, w, p = normality(vals, a.alpha)
        norm_rows.append({'case_study': cs, 'technique': NAME.get(setup, setup), 'budget': f'{k}x', 'variant': variant,
                          'metric': metric, 'n': len(vals), 'mean': _fmt(st.mean(vals)),
                          'sd': _fmt(st.stdev(vals) if len(vals) > 1 else 0.0),
                          'distinct_values': len(set(vals)), 'shapiro_W': _fmt(w), 'shapiro_p': _fmt(p),
                          'normality': label})
    with open(os.path.join(out, 'normality.csv'), 'w', newline='', encoding='utf-8') as f:
        w_ = csv.DictWriter(f, fieldnames=list(norm_rows[0]))
        w_.writeheader()
        w_.writerows(norm_rows)

    comps = []
    for metric in ('verified', 'suite_size'):
        for cs in sorted({key[0] for key in data}):
            block = []
            for k in (1, 5, 10):
                bx = data.get((cs, 'dynamosa', k, 'union', metric))
                if not bx:
                    continue
                for other in ('random_walk', 'random_sample', 'schema_random'):
                    oy = data.get((cs, other, k, 'union', metric))
                    if not oy or len(oy) < 2:
                        continue
                    x = [bx[r] for r in sorted(bx)]
                    y = [oy[r] for r in sorted(oy)]
                    res = compare_independent(x, y, a.alpha)
                    block.append(dict({'case_study': cs, 'metric': metric, 'budget': f'{k}x',
                                       'comparison': f'BRIDGE vs {NAME[other]}', 'n_a': len(x), 'n_b': len(y),
                                       'mean_a': st.mean(x), 'mean_b': st.mean(y),
                                       'normality_a': normality(x, a.alpha)[0],
                                       'normality_b': normality(y, a.alpha)[0]}, **res))
            for row, ph in zip(block, holm([r['p'] for r in block])):
                row['p_holm'] = ph
                row['significant'] = ph < a.alpha
            comps += block
    with open(os.path.join(out, 'comparisons.csv'), 'w', newline='', encoding='utf-8') as f:
        w_ = csv.DictWriter(f, fieldnames=list(comps[0]))
        w_.writeheader()
        w_.writerows({k: _fmt(v) for k, v in r.items()} for r in comps)

    paired = []
    for metric in ('verified', 'suite_size'):
        for cs in sorted({key[0] for key in data}):
            block = []
            for setup in ('dynamosa', 'random_walk'):
                for k in (1, 5, 10):
                    A = data.get((cs, setup, k, 'archive', metric))
                    F = data.get((cs, setup, k, 'final', metric))
                    if not A or not F:
                        continue
                    reps = sorted(set(A) & set(F))
                    x, y = [A[r] for r in reps], [F[r] for r in reps]
                    res = compare_paired(x, y, a.alpha)
                    block.append(dict({'case_study': cs, 'metric': metric, 'technique': NAME[setup], 'budget': f'{k}x',
                                       'comparison': 'archive vs final', 'n_pairs': len(reps),
                                       'mean_archive': st.mean(x), 'mean_final': st.mean(y)}, **res))
            for row, ph in zip(block, holm([r['p'] for r in block])):
                row['p_holm'] = ph
                row['significant'] = ph < a.alpha
            paired += block
    with open(os.path.join(out, 'paired.csv'), 'w', newline='', encoding='utf-8') as f:
        w_ = csv.DictWriter(f, fieldnames=list(paired[0]))
        w_.writeheader()
        w_.writerows({k: _fmt(v) for k, v in r.items()} for r in paired)

    labels = {}
    for r in norm_rows:
        if r['variant'] == 'union':
            labels.setdefault((r['technique'], r['metric']), []).append(r['normality'])
    print(f"Wrote {out}/normality.csv, comparisons.csv, paired.csv\n")
    print('Shapiro-Wilk outcome per technique (union samples, all case studies x budgets):')
    for (tech, metric), ls in sorted(labels.items()):
        c = {x: ls.count(x) for x in ('normal', 'non-normal', 'constant', 'too-small')}
        print(f"  {tech:6} {metric:10} {c}")
    tests = {}
    for r in comps + paired:
        tests[r['test']] = tests.get(r['test'], 0) + 1
    print('\nTests selected:', tests)


if __name__ == '__main__':
    main()
