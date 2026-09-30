"""Repeated experiments with equal budgets for BRIDGE and its unguided variants.

  setup          technique
  dynamosa       BRIDGE
  random_walk    GM, grounded mutation (unguided variant of BRIDGE)
  random_sample  GS, grounded sampling (unguided variant of BRIDGE)

The budget is measured in fitness evaluations. The base budget B of a case
study is the median number of evaluations used by BRIDGE with a population of
30 over 40 generations (`calibrate`), and every run stops after k * B
evaluations for k in 1, 5, 10. Run r with budget kB uses the seed 1000 * k + r.
Every run is validated: the archive and the final population are built into
SQLite databases and checked by the independent validator, and the exact
minimum test suite is computed for the archive, the final population and
their union.

Subcommands (run from the repository root):
  calibrate  --name NAME [--calibration-seeds 3]
  run        --name NAME [--reps 30] [--first-rep 0] [--workers 4]
             [--case-studies ...] [--setups ...] [--budgets 1 5 10] [--no-validate]
  validate   --name FOLDER [--workers 4]
  summarize  --name FOLDER
  status     --name FOLDER

Each `run` writes to outputs/<NAME>__from_repNN/ (named after its first
repetition, so that machines running different repetitions never share a
folder; `evaluation.merge` combines them). Every job runs in its own process
with PYTHONHASHSEED=0, so every run is reproducible from its seed."""
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import time

from bridge import paths
from bridge import pipeline
from bridge.minimization import minimum_cover

COMPILED_PATH = paths.COMPILED_PATH
OUT_ROOT = os.path.join(paths.ROOT, 'outputs')
CALIB_DIR = os.path.join(paths.ROOT, 'evaluation', 'configs')

CASE_STUDIES = paths.CASE_STUDIES
SETUPS = ('dynamosa', 'random_walk', 'random_sample')
BUDGETS = (1, 5, 10)
POPULATION_SIZE = 30
CALIBRATION_GENERATIONS = 40
POPULATION_SETUPS = ('dynamosa', 'random_walk')

def _load_records(case_study):
    return pipeline.load_records(case_study)


def corpus_hash(path=COMPILED_PATH):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    canon = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()


def _git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=paths.ROOT, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=paths.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + (' (uncommitted changes present)' if dirty else '')
    except OSError:
        return 'unknown'


def _exp_dir(name):
    return os.path.join(OUT_ROOT, name)


def part_name(name, first_rep):
    return f"{name}__from_rep{first_rep:02d}"


def _run_dir(name, cs, setup, k, rep):
    return os.path.join(_exp_dir(name), cs, setup, f'b{k}x', f'rep_{rep:02d}')


def _seed_for(k, rep):
    return 1000 * k + rep


def _scope(case_study, records):
    from bridge.validation.scope import _mechanical_out_of_scope
    by_rule = {}
    for r in records:
        by_rule.setdefault(r['rule_id'], []).append(r)
    out_of_scope = _mechanical_out_of_scope(case_study, by_rule)
    return by_rule, out_of_scope


def _jsonable(v):
    return json.dumps(v, default=str, sort_keys=True)


def _write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _run_algorithm(setup, records, case_study, budget, seed, trace):
    from evaluation.baselines import grounded_mutation as random_baseline
    from evaluation.baselines import grounded_sampling as random_sampling
    if setup == 'dynamosa':
        return pipeline.run_bridge(records, case_study, budget, seed, trace=trace,
                                   population_size=POPULATION_SIZE)
    random.seed(seed)
    rng = random.Random(seed)
    if setup == 'random_walk':
        return random_baseline.run_random_search(records, case_study, population_size=POPULATION_SIZE,
                                                 generations=None, rng=rng, max_evaluations=budget,
                                                 trace=trace)
    if setup == 'random_sample':
        return random_sampling.run_random_sampling(records, case_study, max_evaluations=budget,
                                                   rng=rng, trace=trace)
    raise ValueError(setup)


def _record_claims(records, archive, population):
    from bridge.search.candidate import derive_genome
    from bridge.search.dynamosa import evaluate_objective, _focal_for_read
    from bridge.search.fitness import FitnessEvaluationError
    table_cache = {}
    out = {}
    for r in records:
        rid = r['record_id']
        fit, ind = archive[rid]
        cand, fm, sm = ind
        try:
            genome = derive_genome(r, cand, _focal_for_read(r, fm), sm.get(rid, {}), owner_id=rid)
            genome = {k: v for k, v in genome.items()}
        except FitnessEvaluationError as e:
            genome = {'__not_evaluable__': str(e)[:200]}
        final_best = min((evaluate_objective(r, c, f, s, table_cache) for c, f, s in population),
                         default=None)
        out[rid] = {'archive_fitness': fit, 'final_best_fitness': final_best, 'inputs': genome}
    return out


def job_search(run_dir, case_study, setup, k, rep, seed, budget):
    from bridge.search.fitness import evaluations_used
    records = _load_records(case_study)
    os.makedirs(os.path.join(run_dir, 'archive'), exist_ok=True)
    trace = []
    t0 = time.time()
    archive, history, population = _run_algorithm(setup, records, case_study, budget, seed, trace)
    runtime = time.time() - t0
    evals = evaluations_used()

    with open(os.path.join(run_dir, 'archive', 'individuals.pkl'), 'wb') as f:
        pickle.dump({'archive': archive}, f)
    if setup in POPULATION_SETUPS:
        os.makedirs(os.path.join(run_dir, 'final'), exist_ok=True)
        with open(os.path.join(run_dir, 'final', 'individuals.pkl'), 'wb') as f:
            pickle.dump({'final_population': population}, f)

    claims = _record_claims(records, archive, population)
    with open(os.path.join(run_dir, 'claims.json'), 'w', encoding='utf-8') as f:
        json.dump(claims, f, default=str)
    _write_csv(os.path.join(run_dir, 'trace.csv'), ['evaluations', 'record_id'],
               [{'evaluations': e, 'record_id': rid} for e, rid in trace])

    meta = {
        'case_study': case_study, 'setup': setup, 'budget_multiplier': k, 'rep': rep, 'seed': seed,
        'budget_evaluations': budget, 'evaluations_used': evals,
        'stopped_because': 'budget' if evals >= budget else 'all_objectives_covered',
        'iterations': len(history),
        'population_size': POPULATION_SIZE if setup in POPULATION_SETUPS else None,
        'runtime_seconds': round(runtime, 2),
        'claimed_records_archive': sum(1 for c in claims.values() if c['archive_fitness'] == 0.0),
        'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(run_dir, 'SEARCH_DONE'), 'w').close()
    print(f"search done: {evals}/{budget} evaluations, {meta['iterations']} iterations, "
          f"{meta['claimed_records_archive']}/{len(records)} records claimed, {runtime:.1f}s")


def _content_key(individual):
    return pipeline.content_key(individual)


def _validate_full(run_dir, case_study, by_rule, dbs_dir):
    with open(os.path.join(run_dir, 'archive', 'individuals.pkl'), 'rb') as f:
        archive = pickle.load(f)['archive']
    pool, where, key_index = [], [], {}

    def add(ind, source, tag):
        key = _content_key(ind)
        if key not in key_index:
            key_index[key] = len(pool)
            pool.append(ind)
            where.append({'archive': [], 'final': []})
        where[key_index[key]][source].append(tag)

    for rid, (_fit, ind) in archive.items():
        add(ind, 'archive', rid)
    has_final = os.path.exists(os.path.join(run_dir, 'final', 'individuals.pkl'))
    if has_final:
        with open(os.path.join(run_dir, 'final', 'individuals.pkl'), 'rb') as f:
            population = pickle.load(f)['final_population']
        for pos, ind in enumerate(population):
            add(ind, 'final', pos)
    print(f"full validation: {len(pool)} distinct individuals "
          f"({sum(1 for w in where if w['archive'])} in archive, {sum(1 for w in where if w['final'])} in final)",
          flush=True)
    matrix, errs = pipeline.validate_pool(case_study, pool, dbs_dir,
                                          log=lambda message: print(message, flush=True))

    rule_of = {r['record_id']: rule for rule, recs in by_rule.items() for r in recs}
    members = {'archive': [i for i, w in enumerate(where) if w['archive']],
               'final': [i for i, w in enumerate(where) if w['final']] if has_final else None,
               'union': list(range(len(pool)))}
    verified = {v: set().union(*(matrix[i] for i in m)) for v, m in members.items() if m is not None}

    _write_csv(os.path.join(run_dir, 'matrix.csv'),
               ['pool_index', 'in_archive', 'in_final', 'archived_for_rules', 'final_positions',
                'num_rules_verified', 'rules_verified'],
               [{'pool_index': i, 'in_archive': bool(w['archive']), 'in_final': bool(w['final']),
                 'archived_for_rules': '; '.join(sorted({rule_of.get(r, r) for r in w['archive']})),
                 'final_positions': ' '.join(map(str, w['final'])),
                 'num_rules_verified': len(matrix[i]), 'rules_verified': '; '.join(sorted(matrix[i]))}
                for i, w in enumerate(where)])
    for v in ('archive', 'final'):
        if members[v] is None:
            continue
        _write_csv(os.path.join(run_dir, v, 'verification.csv'),
                   ['rule_id', 'verified', 'num_individuals_verifying', 'first_individual'],
                   [{'rule_id': rid, 'verified': rid in verified[v],
                     'num_individuals_verifying': sum(1 for i in members[v] if rid in matrix[i]),
                     'first_individual': next((i for i in members[v] if rid in matrix[i]), '')}
                    for rid in sorted(by_rule)])

    suite = {}
    os.makedirs(os.path.join(run_dir, 'minimal_suite'), exist_ok=True)
    for v, m in members.items():
        if m is None:
            continue
        greedy, minimum, optimal = minimum_cover(verified[v], {i: matrix[i] for i in m})
        covered = set().union(*(matrix[i] for i in minimum))
        assert covered >= verified[v], (v, verified[v] - covered)
        suite[v] = {'pool': len(m), 'greedy': len(greedy), 'min': len(minimum), 'optimal': optimal,
                    'min_members': minimum, 'greedy_members': greedy}
        with open(os.path.join(run_dir, 'minimal_suite', f'{v}.pkl'), 'wb') as f:
            pickle.dump({'pool_indices': minimum, 'individuals': [pool[i] for i in minimum],
                         'rules_covered': sorted(verified[v])}, f)
    with open(os.path.join(run_dir, 'minimization.json'), 'w', encoding='utf-8') as f:
        json.dump(suite, f, indent=2)
    counts = {'distinct_individuals_validated': len(pool),
              'archive_individuals_validated': len(members['archive']),
              'final_individuals_validated': len(members['final']) if has_final else None}
    short = {v: {k: x[k] for k in ('pool', 'greedy', 'min', 'optimal')} for v, x in suite.items()}
    return verified['archive'], verified.get('final', set()), has_final, errs, short, counts


def job_validate(run_dir, case_study, keep_dbs):
    records = _load_records(case_study)
    by_rule, out_of_scope = _scope(case_study, records)
    with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(run_dir, 'claims.json'), encoding='utf-8') as f:
        claims = json.load(f)
    dbs_dir = os.path.join(run_dir, 'dbs')
    t0 = time.time()
    v_archive, v_final, has_final, errs, suite, counts = _validate_full(run_dir, case_study, by_rule, dbs_dir)
    if not keep_dbs:
        shutil.rmtree(dbs_dir, ignore_errors=True)

    first_hit = {}
    for row in _read_csv(os.path.join(run_dir, 'trace.csv')):
        first_hit.setdefault(row['record_id'], int(row['evaluations']))

    def yn(b):
        return 'True' if b else 'False'

    per_record = []
    for r in records:
        rid, rule = r['record_id'], r['rule_id']
        c = claims[rid]
        fin = c['final_best_fitness']
        per_record.append({
            'record_id': rid, 'decision': r['decision_name'], 'rule_id': rule,
            'hit_policy': r['hit_policy'], 'in_scope': yn(rule not in out_of_scope),
            'archive_fitness': c['archive_fitness'],
            'final_best_fitness': '' if fin is None else fin,
            'first_covered_at_evaluation': first_hit.get(rid, ''),
            'claimed_archive': yn(c['archive_fitness'] == 0.0),
            'claimed_final': '' if not has_final else yn(fin == 0.0),
            'rule_verified_archive': yn(rule in v_archive),
            'rule_verified_final': '' if not has_final else yn(rule in v_final),
            'rule_verified_union': yn(rule in v_archive or rule in v_final),
            'inputs': _jsonable(c['inputs']),
            'outputs': _jsonable({k: v.get('value', v) if isinstance(v, dict) else v
                                  for k, v in (r.get('outputs') or {}).items()}),
        })
    _write_csv(os.path.join(run_dir, 'per_record.csv'), list(per_record[0]), per_record)

    per_rule = []
    for rule, recs in sorted(by_rule.items()):
        rows = [p for p in per_record if p['rule_id'] == rule]
        ca = any(p['claimed_archive'] == 'True' for p in rows)
        cf = has_final and any(p['claimed_final'] == 'True' for p in rows)
        per_rule.append({
            'rule_id': rule, 'decision': recs[0]['decision_name'], 'in_scope': yn(rule not in out_of_scope),
            'num_records': len(recs),
            'claimed_archive': yn(ca), 'claimed_final': '' if not has_final else yn(cf),
            'claimed_union': yn(ca or cf),
            'verified_archive': yn(rule in v_archive),
            'verified_final': '' if not has_final else yn(rule in v_final),
            'verified_union': yn(rule in v_archive or rule in v_final),
        })
    _write_csv(os.path.join(run_dir, 'per_rule.csv'), list(per_rule[0]), per_rule)

    meta.update(counts)
    meta.update({'validation_mode': 'full', 'suite_size': suite,
                 'validation_errors': errs, 'validation_seconds': round(time.time() - t0, 1)})
    with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(run_dir, 'VALIDATED'), 'w').close()
    in_scope = [p for p in per_rule if p['in_scope'] == 'True']
    print(f"validated: archive {sum(p['verified_archive'] == 'True' for p in in_scope)}, "
          f"final {sum(p['verified_final'] == 'True' for p in in_scope) if has_final else '-'}, "
          f"union {sum(p['verified_union'] == 'True' for p in in_scope)} of {len(in_scope)} in-scope rules; "
          f"suite size {suite} ({len(errs)} errors, {time.time() - t0:.1f}s)")


def _spawn(args, log_path):
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONIOENCODING='utf-8')
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run([sys.executable, '-m', 'evaluation.experiment'] + args,
                              stdout=log, stderr=subprocess.STDOUT, env=env, cwd=paths.ROOT)
    return proc.returncode


def _calibration_path(name):
    return os.path.join(CALIB_DIR, f'{name}.json')


def _load_calibration(name):
    for path in (os.path.join(_exp_dir(name), 'calibration.json'), _calibration_path(name)):
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    sys.exit(f"No calibration for '{name}' -- run: harness.py calibrate --name {name}")


def _require_output_folder(name):
    if os.path.isdir(_exp_dir(name)):
        return
    parts = sorted(d for d in os.listdir(OUT_ROOT) if d.startswith(name + '__from_rep'))         if os.path.isdir(OUT_ROOT) else []
    hint = (f" Output folders for '{name}': {', '.join(parts)} -- pass one of those as --name, or "
            f"combine them with merge.py.") if parts else ''
    sys.exit(f"No output folder outputs/{name}.{hint}")


def cmd_calibrate(a):
    os.makedirs(CALIB_DIR, exist_ok=True)
    from bridge.search import dynamosa
    from bridge.search.fitness import reset_evaluation_counter, evaluations_used
    path = _calibration_path(a.name)
    calib = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ, PYTHONHASHSEED='0')
        sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)
    for cs in a.case_studies:
        if cs in calib and not a.force:
            print(f"{cs}: already calibrated (B={calib[cs]['budget_1x']}); --force to redo")
            continue
        records = _load_records(cs)
        per_seed = []
        for seed in range(a.calibration_seeds):
            random.seed(seed)
            reset_evaluation_counter(None)
            t0 = time.time()
            archive, _h, _p = dynamosa.run_dynamosa(records, cs, population_size=POPULATION_SIZE,
                                                    generations=CALIBRATION_GENERATIONS,
                                                    rng=random.Random(seed))
            used = evaluations_used()
            covered = sum(1 for r in records if archive[r['record_id']][0] == 0.0)
            per_seed.append({'seed': seed, 'evaluations': used, 'runtime_seconds': round(time.time() - t0, 1),
                             'claimed_records': covered})
            print(f"{cs} calibration seed {seed}: {used} evaluations, {covered}/{len(records)} "
                  f"records claimed, {time.time() - t0:.1f}s", flush=True)
        budget = int(statistics.median(p['evaluations'] for p in per_seed))
        calib[cs] = {'budget_1x': budget, 'population_size': POPULATION_SIZE,
                     'generations': CALIBRATION_GENERATIONS, 'seeds': per_seed,
                     'median_runtime_seconds': statistics.median(p['runtime_seconds'] for p in per_seed),
                     'compiled_constraints_sha256': corpus_hash(),
                     'git_commit': _git_commit(),
                     'calibrated_at': datetime.datetime.now().isoformat(timespec='seconds')}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(calib, f, indent=2)
        print(f"{cs}: B = {budget} fitness evaluations (median of {len(per_seed)} seeds)")
    print(f"\nWrote {path}")


def _by_size(case_studies):
    return sorted(case_studies, key=CASE_STUDIES.index)


def _jobs(a, calib):
    jobs = []
    for cs in _by_size(a.case_studies):
        for rep in range(a.first_rep, a.first_rep + a.reps):
            for setup in a.setups:
                for k in a.budgets:
                    jobs.append((cs, setup, k, rep, _seed_for(k, rep), k * calib[cs]['budget_1x']))
    return jobs


def _write_config(a, calib):
    os.makedirs(_exp_dir(a.out_name), exist_ok=True)
    path = os.path.join(_exp_dir(a.out_name), 'config.json')
    with open(os.path.join(_exp_dir(a.out_name), 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=2)
    config = {'experiment': a.name, 'output_folder': a.out_name, 'reps': a.reps, 'first_rep': a.first_rep, 'case_studies': a.case_studies, 'setups': a.setups, 'budgets': a.budgets,
              'population_size': POPULATION_SIZE, 'seed_rule': 'seed = 1000 * budget_multiplier + rep',
              'budget_1x': {cs: calib[cs]['budget_1x'] for cs in a.case_studies},
              'not_persisted_files': {cs: paths.config(cs)['not_persisted'] for cs in a.case_studies},
              'compiled_constraints_sha256': corpus_hash(), 'git_commit': _git_commit(),
              'written_at': datetime.datetime.now().isoformat(timespec='seconds')}
    for cs in a.case_studies:
        calibrated_on = calib[cs].get('compiled_constraints_sha256')
        if calibrated_on and calibrated_on != config['compiled_constraints_sha256']:
            print(f"WARNING: compiled_constraints.json changed since {cs} was calibrated -- "
                  f"consider `calibrate --force` before relying on B.")
    history = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
        history = old.pop('history', []) + [old]
        if old.get('compiled_constraints_sha256') != config['compiled_constraints_sha256']:
            print("WARNING: compiled_constraints.json differs from the last `run` of this experiment -- "
                  "new runs will not be comparable with the ones already on disk.")
    config['history'] = history
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def _execute(jobs, workers, fn, label):
    total = len(jobs)
    done = [0]
    t0 = time.time()

    def wrapped(job):
        rc = fn(job)
        done[0] += 1
        elapsed = time.time() - t0
        print(f"[{done[0]}/{total}] {label(job)} -> {'ok' if rc == 0 else f'FAILED (exit {rc}), see log.txt'} "
              f"({elapsed / 60:.1f} min elapsed)", flush=True)
        return rc

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rcs = list(pool.map(wrapped, jobs))
    failed = sum(1 for rc in rcs if rc != 0)
    print(f"\n{total - failed}/{total} succeeded" + (f", {failed} FAILED (see each run's log.txt)" if failed else ''))


def cmd_run(a):
    calib = _load_calibration(a.name)
    missing = [cs for cs in a.case_studies if cs not in calib]
    if missing:
        sys.exit(f"Not calibrated: {missing} -- run calibrate first")
    a.out_name = part_name(a.name, a.first_rep)
    _write_config(a, calib)
    print(f"Output folder: outputs/{a.out_name}  "
          f"(repetitions {a.first_rep}-{a.first_rep + a.reps - 1})")
    todo = []
    for job in _jobs(a, calib):
        cs, setup, k, rep, seed, budget = job
        d = _run_dir(a.out_name, cs, setup, k, rep)
        need_search = not os.path.exists(os.path.join(d, 'SEARCH_DONE'))
        need_val = not a.no_validate and not os.path.exists(os.path.join(d, 'VALIDATED'))
        if need_search or need_val:
            todo.append((job, d, need_search, need_val))
    print(f"{len(todo)} run(s) to do ({len(_jobs(a, calib)) - len(todo)} already finished), "
          f"{a.workers} worker(s)\n")

    def fn(item):
        (cs, setup, k, rep, seed, budget), d, need_search, need_val = item
        os.makedirs(d, exist_ok=True)
        log = os.path.join(d, 'log.txt')
        if need_search:
            for marker in ('VALIDATED',):
                if os.path.exists(os.path.join(d, marker)):
                    os.remove(os.path.join(d, marker))
            rc = _spawn(['_search', d, cs, setup, str(k), str(rep), str(seed), str(budget)], log)
            if rc != 0:
                return rc
        if need_val:
            return _spawn(['_validate', d, cs] + (['--keep-dbs'] if a.keep_dbs else []), log)
        return 0

    _execute(todo, a.workers, fn, lambda item: f"{item[0][0]} {item[0][1]} b{item[0][2]}x rep_{item[0][3]:02d}")


def _all_run_dirs(name):
    exp = _exp_dir(name)
    present = [cs for cs in CASE_STUDIES if os.path.isdir(os.path.join(exp, cs))]
    for cs in present:
        for setup in SETUPS:
            base = os.path.join(exp, cs, setup)
            if not os.path.isdir(base):
                continue
            for bdir in sorted(os.listdir(base)):
                for rdir in sorted(os.listdir(os.path.join(base, bdir))):
                    yield cs, setup, int(bdir[1:-1]), int(rdir[4:]), os.path.join(base, bdir, rdir)


def cmd_validate(a):
    _require_output_folder(a.name)
    todo = [(cs, d) for cs, _s, _k, _r, d in _all_run_dirs(a.name)
            if cs in a.case_studies and os.path.exists(os.path.join(d, 'SEARCH_DONE'))
            and (a.force or not os.path.exists(os.path.join(d, 'VALIDATED')))]
    print(f"{len(todo)} run(s) to validate, {a.workers} worker(s)\n")
    _execute(todo, a.workers,
             lambda item: _spawn(['_validate', item[1], item[0]]
                                 + (['--keep-dbs'] if a.keep_dbs else []),
                                 os.path.join(item[1], 'log.txt')),
             lambda item: os.path.relpath(item[1], _exp_dir(a.name)))


def _auc(events, in_scope_rules, rule_of, horizon):
    if not in_scope_rules or horizon <= 0:
        return 0.0
    seen, points = set(), []
    for ev, rid in events:
        rule = rule_of.get(rid)
        if rule in in_scope_rules and rule not in seen:
            seen.add(rule)
            points.append((ev, len(seen)))
    area, prev_x, prev_y = 0.0, 0, 0
    for x, y in points:
        x = min(x, horizon)
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y
    area += (horizon - prev_x) * prev_y
    return area / (horizon * len(in_scope_rules))


def cmd_summarize(a):
    _require_output_folder(a.name)
    exp = _exp_dir(a.name)
    calib = _load_calibration(a.name)
    runs, long_rows = [], []
    records_cache = {}
    for cs, setup, k, rep, d in _all_run_dirs(a.name):
        if not os.path.exists(os.path.join(d, 'SEARCH_DONE')):
            continue
        with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
            meta = json.load(f)
        if cs not in records_cache:
            recs = _load_records(cs)
            _by_rule, oos = _scope(cs, recs)
            records_cache[cs] = ({r['record_id']: r['rule_id'] for r in recs},
                                 {r['rule_id'] for r in recs} - oos, len({r['rule_id'] for r in recs}))
        rule_of, in_scope_rules, total_rules = records_cache[cs]
        events = [(int(r['evaluations']), r['record_id']) for r in _read_csv(os.path.join(d, 'trace.csv'))]
        auc_own = _auc(events, in_scope_rules, rule_of, meta['budget_evaluations'])
        auc_1x = _auc(events, in_scope_rules, rule_of, calib[cs]['budget_1x'])
        validated = os.path.exists(os.path.join(d, 'VALIDATED'))
        counts = {}
        if validated:
            per_rule = [p for p in _read_csv(os.path.join(d, 'per_rule.csv')) if p['in_scope'] == 'True']
            for v in ('archive', 'final', 'union'):
                if per_rule and per_rule[0][f'verified_{v}'] == '':
                    continue
                counts[v] = (sum(p[f'claimed_{v}'] == 'True' for p in per_rule),
                             sum(p[f'verified_{v}'] == 'True' for p in per_rule))
        suite = meta.get('suite_size', {}) if validated else {}
        base = {'case_study': cs, 'setup': setup, 'budget_multiplier': k, 'rep': rep, 'seed': meta['seed'],
                'budget_evaluations': meta['budget_evaluations'], 'evaluations_used': meta['evaluations_used'],
                'iterations': meta['iterations'], 'runtime_seconds': meta['runtime_seconds'],
                'total_rules': total_rules, 'in_scope_rules': len(in_scope_rules),
                'auc_claimed_own_budget': round(auc_own, 6), 'auc_claimed_first_1x': round(auc_1x, 6)}
        run_row = dict(base, validated=validated)
        for v in ('archive', 'final', 'union'):
            c = counts.get(v)
            run_row[f'claimed_{v}'] = '' if c is None else c[0]
            run_row[f'verified_{v}'] = '' if c is None else c[1]
            if c is not None:
                ss = suite.get(v, {})
                long_rows.append(dict(base, variant=v, claimed=c[0], verified=c[1],
                                      verified_pct_in_scope=round(100 * c[1] / len(in_scope_rules), 3)
                                      if in_scope_rules else 0.0,
                                      validation_mode=meta.get('validation_mode', 'optimized'),
                                      suite_size_min=ss.get('min', ''), suite_size_greedy=ss.get('greedy', ''),
                                      suite_min_optimal=ss.get('optimal', ''),
                                      suite_size_firstfit=ss.get('firstfit', '')))
        runs.append(run_row)
    if not runs:
        sys.exit('Nothing finished yet.')
    _write_csv(os.path.join(exp, 'runs.csv'), list(runs[0]), runs)
    if long_rows:
        _write_csv(os.path.join(exp, 'summary.csv'), list(long_rows[0]), long_rows)
    print(f"Wrote {os.path.join(exp, 'runs.csv')} ({len(runs)} runs)")
    print(f"Wrote {os.path.join(exp, 'summary.csv')} ({len(long_rows)} rows: one per run x output variant)")

    groups = {}
    for r in long_rows:
        groups.setdefault((r['case_study'], r['setup'], r['budget_multiplier'], r['variant']), []).append(r['verified'])
    print(f"\n{'Case study':<10} {'Setup':<14} {'Budget':>6} {'Output':<8} {'n':>3} {'mean':>7} {'median':>7} "
          f"{'min':>4} {'max':>4}   (verified in-scope rules)")
    for (cs, setup, k, v), vals in sorted(groups.items()):
        print(f"{cs:<10} {setup:<14} {str(k) + 'x':>6} {v:<8} {len(vals):>3} {statistics.mean(vals):>7.2f} "
              f"{statistics.median(vals):>7.1f} {min(vals):>4} {max(vals):>4}")


def cmd_status(a):
    _require_output_folder(a.name)
    calib = _load_calibration(a.name)
    cfg_path = os.path.join(_exp_dir(a.name), 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        print(f"first_rep={cfg.get('first_rep', 0)} reps={cfg['reps']} setups={cfg['setups']} budgets={cfg['budgets']} B={cfg['budget_1x']}")
    counts = {}
    for cs, setup, k, _rep, d in _all_run_dirs(a.name):
        c = counts.setdefault((cs, setup, k), [0, 0, 0])
        c[0] += 1
        c[1] += os.path.exists(os.path.join(d, 'SEARCH_DONE'))
        c[2] += os.path.exists(os.path.join(d, 'VALIDATED'))
    print(f"{'Case study':<10} {'Setup':<14} {'Budget':>6} {'started':>8} {'searched':>9} {'validated':>10}")
    for (cs, setup, k), (s, sd, v) in sorted(counts.items()):
        print(f"{cs:<10} {setup:<14} {str(k) + 'x':>6} {s:>8} {sd:>9} {v:>10}")
    for cs, c in calib.items():
        print(f"calibration {cs}: B={c['budget_1x']} (search 1x ~{c['median_runtime_seconds']}s per run)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, runs=False):
        p.add_argument('--name', required=True, help='calibrate/run: the experiment name (e.g. paper). validate/summarize/status: a folder under outputs/ (e.g. paper__from_rep00, or a merged folder)')
        p.add_argument('--case-studies', nargs='+', default=list(CASE_STUDIES), choices=CASE_STUDIES)
        if runs:
            p.add_argument('--workers', type=int, default=4, help='parallel processes (default 4)')
            p.add_argument('--keep-dbs', action='store_true',
                           help='keep every validated individual\'s SQLite DB (large); default: delete after validating')

    p = sub.add_parser('calibrate')
    common(p)
    p.add_argument('--calibration-seeds', type=int, default=3)
    p.add_argument('--force', action='store_true')
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser('run')
    common(p, runs=True)
    p.add_argument('--reps', type=int, default=30, help='how many repetitions to run')
    p.add_argument('--first-rep', type=int, default=0,
                   help='index of the first repetition (default 0). To split work across machines, '
                        'give each a different range, e.g. --reps 10 --first-rep 0 on one and '
                        '--reps 10 --first-rep 10 on the other, then copy the rep folders together')
    p.add_argument('--setups', nargs='+', default=list(SETUPS), choices=SETUPS)
    p.add_argument('--budgets', nargs='+', type=int, default=list(BUDGETS))
    p.add_argument('--no-validate', action='store_true', help='search only; validate later with `validate`')
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('validate')
    common(p, runs=True)
    p.add_argument('--force', action='store_true', help='re-validate runs already validated')
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser('summarize')
    common(p)
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser('status')
    common(p)
    p.set_defaults(fn=cmd_status)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_search':
        d, cs, setup, k, rep, seed, budget = sys.argv[2:9]
        job_search(d, cs, setup, int(k), int(rep), int(seed), int(budget))
    elif len(sys.argv) > 1 and sys.argv[1] == '_validate':
        job_validate(sys.argv[2], sys.argv[3], '--keep-dbs' in sys.argv)
    else:
        main()

