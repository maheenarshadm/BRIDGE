"""SR (schema-based random generation).

Random databases are generated from the schema alone: every table receives
rows with random values from the full domain of each column type, repaired
to satisfy keys, foreign keys and NOT NULL constraints. There is no grounding
and no search. Every database is checked by the validator directly and kept
when it verifies a rule not verified before. Each database is charged one
fitness evaluation per objective, the cost of one sample of GS.

In prefix mode (default) one stream of databases is generated per case study
and repetition for the largest budget; the results of the smaller budgets
are read from the first databases of that stream.

Usage:
  python -m evaluation.baselines.schema_random run --name NAME [--reps 30] [--case-studies ...]
  python -m evaluation.baselines.schema_random status --name NAME
Writes outputs/<NAME>_schemarandom__from_repNN/."""
import argparse
import datetime
import json
import os
import pickle
import random
import re
import sqlite3
import sys
import time
import traceback

from bridge import paths
from bridge import pipeline
from bridge.minimization import minimum_cover
from evaluation import experiment as H

SETUP = 'schema_random'

MIN_ROWS, MAX_ROWS = 1, 5
DEFAULT_TEXT_LENGTH = 255
NULL_PROBABILITY = 0.1
PRINTABLE = ''.join(chr(c) for c in range(32, 127))
INT64 = (-2 ** 63, 2 ** 63 - 1)
DATE_DAYS = (-719162, 2932896)

GENERATOR_SETTINGS = {
    'values': 'uniform over the full domain of the declared column type',
    'integer_types': 'smallint 16-bit, integer 32-bit, bigint 64-bit signed',
    'oracle_number_without_precision': '64-bit signed integer range (largest SQLite can store)',
    'numeric_decimal': 'declared (precision, scale); (10, 0) when not declared',
    'floating_point': 'double: full IEEE double range; float/real: full IEEE single range',
    'date_datetime_timestamp': 'calendar years 1-9999, as day numbers since 1970-01-01',
    'time': 'seconds of the day, 0-86399',
    'text': f'printable ASCII, length 0..declared length (char: exactly declared length), '
            f'{DEFAULT_TEXT_LENGTH} when no length is declared',
    'boolean': 'true / false', 'binary': 'random bytes, length 0..declared length (255 default)',
    'rows_per_table': [MIN_ROWS, MAX_ROWS], 'null_probability_nullable_columns': NULL_PROBABILITY,
    'single_column_primary_keys': 'sequential 1..n (unique by construction)',
    'foreign_keys': 'value of a random existing parent row (parents generated first)',
    'budget_charge_per_database': 'number of compiled objectives of the case study',
}


def part_name(name, first_rep):
    return f"{name}_schemarandom__from_rep{first_rep:02d}"


def _run_dir(out_name, cs, k, rep):
    return os.path.join(H.OUT_ROOT, out_name, cs, SETUP, f'b{k}x', f'rep_{rep:02d}')


def _table_info(schema, table):
    return schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}


def _kind(ctype):
    m = re.match(r'^\s*([A-Za-z0-9_ ]+?)\s*(?:\((.*)\))?\s*$', ctype or 'text')
    base = (m.group(1) if m else (ctype or 'text')).strip().lower()
    params = [int(x) for x in re.findall(r'\d+', m.group(2) or '')] if m else []
    if 'bool' in base:
        return ('bool',)
    if base == 'time':
        return ('int', 0, 86399)
    if 'date' in base or 'time' in base:
        return ('int',) + DATE_DAYS
    if base in ('blob', 'raw', 'long raw', 'bytea', 'binary', 'varbinary'):
        return ('bytes', params[0] if params else DEFAULT_TEXT_LENGTH)
    if base in ('json', 'jsonb'):
        return ('json',)
    if base in ('smallint', 'int2'):
        return ('int', -2 ** 15, 2 ** 15 - 1)
    if base in ('tinyint',):
        return ('int', -2 ** 7, 2 ** 7 - 1)
    if base in ('bigint', 'int8'):
        return ('int',) + INT64
    if 'int' in base:
        return ('int', -2 ** 31, 2 ** 31 - 1)
    if base == 'number':
        return ('decimal', params[0], params[1] if len(params) > 1 else 0) if params else ('int',) + INT64
    if base in ('numeric', 'decimal', 'dec'):
        return ('decimal', params[0] if params else 10, params[1] if len(params) > 1 else 0)
    if base in ('double', 'double precision'):
        return ('float', 1.7976931348623157e308)
    if base in ('float', 'real'):
        return ('float', 3.4028234663852886e38)
    if base in ('char', 'nchar', 'character'):
        return ('text_fixed', params[0] if params else 1)
    return ('text', params[0] if params else DEFAULT_TEXT_LENGTH)


def _random_value(kind, rng):
    k = kind[0]
    if k == 'int':
        return rng.randint(kind[1], kind[2])
    if k == 'decimal':
        p, s = kind[1], kind[2]
        n = rng.randint(-(10 ** p - 1), 10 ** p - 1)
        if s == 0:
            return max(INT64[0], min(INT64[1], n))
        return n / 10 ** s
    if k == 'float':
        return (1 if rng.random() < 0.5 else -1) * rng.uniform(0.0, kind[1])
    if k == 'bool':
        return rng.random() < 0.5
    if k == 'bytes':
        return rng.randbytes(rng.randint(0, kind[1]))
    if k == 'json':
        return json.dumps(''.join(rng.choices(PRINTABLE, k=rng.randint(0, DEFAULT_TEXT_LENGTH))))
    length = kind[1] if k == 'text_fixed' else rng.randint(0, kind[1])
    return ''.join(rng.choices(PRINTABLE, k=length))


_ORDER_CACHE = {}


def random_database(case_study, rng):
    from bridge.search.candidate import Candidate
    from bridge.construction.materialize import topological_table_order
    from bridge.search.mutation import _schema_for, repair_candidate

    schema = _schema_for(case_study)
    if case_study not in _ORDER_CACHE:
        _ORDER_CACHE[case_study] = topological_table_order(schema, schema.keys())[0]
    order = _ORDER_CACHE[case_study]
    cand = Candidate()
    for table in order:
        info = _table_info(schema, table)
        columns = dict(info.get('columns') or {})
        pk = info.get('pk')
        pk_cols = pk if isinstance(pk, list) else ([pk] if pk else [])
        for col in pk_cols:
            columns.setdefault(col, {'type': 'INTEGER', 'null_false': True})
        fks = {fk['column']: fk for fk in (info.get('fk_columns') or [])
               if fk['ref_table'].upper() in {t.upper() for t in schema}}
        single_pk = pk_cols[0] if len(pk_cols) == 1 else None
        rows, seen_pk = [], set()
        for n in range(rng.randint(MIN_ROWS, MAX_ROWS)):
            row = {}
            for col, meta in columns.items():
                kind = _kind(meta.get('type'))
                nullable = not meta.get('null_false') and col not in pk_cols
                if col == single_pk and col not in fks:
                    row[col] = f'K{n + 1}' if kind[0] in ('text', 'text_fixed') else (n + 1)
                elif col in fks:
                    fk = fks[col]
                    parents = [p for p in cand.rows(fk['ref_table'])
                               if p.get(fk['ref_column']) is not None]
                    if nullable and (not parents or rng.random() < NULL_PROBABILITY):
                        row[col] = None
                    elif parents:
                        row[col] = rng.choice(parents)[fk['ref_column']]
                    else:
                        row[col] = _random_value(kind, rng)
                elif nullable and rng.random() < NULL_PROBABILITY:
                    row[col] = None
                else:
                    row[col] = _random_value(kind, rng)
            key = tuple(row.get(c) for c in pk_cols)
            if pk_cols and key in seen_pk:
                continue
            seen_pk.add(key)
            rows.append(row)
        for row in rows:
            cand.add_row(table, row)
    repair_candidate(cand, case_study)
    return cand


def _q(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def _ddl_script(case_study):
    from bridge.construction.materialize import _sqlite_type, topological_table_order
    from bridge.search.mutation import _schema_for
    schema = _schema_for(case_study)
    order, _w = topological_table_order(schema, schema.keys())
    stmts = []
    for table in order:
        info = _table_info(schema, table)
        columns = info.get('columns') or {}
        pk = info.get('pk')
        pk_cols = pk if isinstance(pk, list) else ([pk] if pk else [])
        if not columns and not pk_cols:
            continue
        defs = [' '.join([_q(c), _sqlite_type(m.get('type'))] + (['NOT NULL'] if m.get('null_false') else []))
                for c, m in columns.items()]
        defs += [f'{_q(c)} INTEGER' for c in pk_cols if c not in columns]
        if pk_cols:
            defs.append(f"PRIMARY KEY ({', '.join(_q(c) for c in pk_cols)})")
        stmts.append(f"CREATE TABLE {_q(table)} ({', '.join(defs)});")
    return '\n'.join(stmts)


def to_sqlite(cand, case_study, ddl, path=':memory:'):
    from bridge.construction.materialize import _fill_missing_surrogate_keys, _real_columns, topological_table_order
    from bridge.search.mutation import _schema_for
    schema = _schema_for(case_study)
    if path != ':memory:' and os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    cur = conn.cursor()
    known = {t.upper() for t in schema}
    order, _w = topological_table_order(schema, [t for t in cand.as_dict() if t.upper() in known])
    for table in order:
        for row in _fill_missing_surrogate_keys(table, cand.rows(table), schema):
            row = _real_columns(row)
            if not row:
                continue
            cols = list(row)
            cur.execute(f"INSERT INTO {_q(table)} ({', '.join(_q(c) for c in cols)}) "
                        f"VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])
    conn.commit()
    return conn


def _validator_context(case_study):
    from bridge.validation.subject_table import build_subject_tables
    from bridge.validation.out_of_scope_rules import is_out_of_scope
    from bridge.validation.phase1_utility import records_by_decision
    decisions = records_by_decision(case_study)
    in_scope_by_name = {name: [r for r in recs if not is_out_of_scope(case_study, r['rule_id'])]
                        for name, recs in decisions.items()}
    resolved, unresolved = build_subject_tables(case_study, in_scope_by_name)
    overrides = pipeline.not_persisted_values(case_study)
    rules_of = {d: {r['rule_id'] for r in recs} for d, recs in decisions.items()}
    ctx = {'in_scope_by_name': in_scope_by_name, 'resolved': resolved, 'unresolved': unresolved}
    return ctx, decisions, overrides, rules_of


def _verify(ctx, conn, case_study, decisions, names, overrides, decision_errors=None):
    from bridge.validation.drd_executor import DecisionRunner, run_decision
    rules = set()
    for ov in overrides:
        runner = DecisionRunner(conn, case_study, ctx['in_scope_by_name'], ctx['resolved'],
                                not_persisted_overrides=ov)
        for d in sorted(names):
            if d not in decisions or d in ctx['unresolved']:
                continue
            subject_table, pk_cols, join_paths = ctx['resolved'][d]
            try:
                rr = run_decision(conn, d, ctx['in_scope_by_name'][d], subject_table, pk_cols,
                                  join_paths=join_paths, runner=runner, collect_trace=False,
                                  not_persisted_overrides=ov)
                rules |= set(rr['verified_covered_rule_ids'])
            except (NotImplementedError, sqlite3.OperationalError, KeyError):
                pass
            except Exception as e:
                if decision_errors is not None:
                    key = f'{d}: {type(e).__name__}'
                    decision_errors[key] = decision_errors.get(key, 0) + 1
    return rules


def _write_generation(run_dir, meta, pool, trace):
    os.makedirs(os.path.join(run_dir, 'pool'), exist_ok=True)
    with open(os.path.join(run_dir, 'pool', 'databases.pkl'), 'wb') as f:
        pickle.dump(pool, f)
    H._write_csv(os.path.join(run_dir, 'trace.csv'), ['evaluations', 'database_index', 'rule_id'], trace)
    with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(run_dir, 'SEARCH_DONE'), 'w').close()


def job_generate(out_name, case_study, rep, seed, targets, max_databases, mode):
    records = H._load_records(case_study)
    by_rule, oos = H._scope(case_study, records)
    in_scope = set(by_rule) - oos
    n_obj = len(records)
    targets = sorted(targets)
    counts = {k: (min(b // n_obj, max_databases) if max_databases else b // n_obj) for k, b in targets}
    n_db = max(counts.values())
    kmax = targets[-1][0]
    ctx, decisions, overrides, rules_of = _validator_context(case_study)
    ddl = _ddl_script(case_study)
    random.seed(seed)
    rng = random.Random(seed)

    remaining = set(by_rule)
    pool, trace, errors = [], [], []
    decision_errors = {}
    failures = 0
    t0 = time.time()
    for idx in range(n_db):
        try:
            cand = random_database(case_study, rng)
            conn = to_sqlite(cand, case_study, ddl)
            try:
                names = {d for d, rids in rules_of.items() if rids & remaining}
                found = (_verify(ctx, conn, case_study, decisions, names, overrides, decision_errors)
                         & remaining) if names else set()
            finally:
                conn.close()
        except Exception as e:
            failures += 1
            found = set()
            if len(errors) < 20:
                errors.append(f'database {idx}: {type(e).__name__}: {e}')
                traceback.print_exc()
        if found:
            pool.append({'database_index': idx, 'candidate': cand, 'new_rules': sorted(found)})
            for rid in sorted(found):
                trace.append({'evaluations': (idx + 1) * n_obj, 'database_index': idx, 'rule_id': rid})
            remaining -= found
        if (idx + 1) % 100 == 0 or idx + 1 == n_db:
            print(f"  {idx + 1}/{n_db} databases, {len(in_scope - remaining)}/{len(in_scope)} in-scope rules "
                  f"verified, {failures} failed, {time.time() - t0:.0f}s", flush=True)
        for k, budget in targets:
            if counts[k] != idx + 1:
                continue
            runtime = time.time() - t0
            run_dir = _run_dir(out_name, case_study, k, rep)
            os.makedirs(run_dir, exist_ok=True)
            meta = {
                'case_study': case_study, 'setup': SETUP, 'budget_multiplier': k, 'rep': rep, 'seed': seed,
                'mode': mode, 'stream_budget_multiplier': kmax,
                'budget_evaluations': budget, 'objectives': n_obj, 'databases_generated': counts[k],
                'evaluations_used': counts[k] * n_obj, 'capped_by_max_databases': bool(max_databases),
                'stopped_because': 'budget', 'iterations': counts[k], 'population_size': None,
                'runtime_seconds': round(runtime, 2),
                'seconds_per_database': round(runtime / max(counts[k], 1), 4),
                'failed_databases': failures, 'generation_errors': list(errors),
                'unevaluable_decisions': dict(decision_errors),
                'pool_size': len(pool), 'generator_settings': GENERATOR_SETTINGS,
                'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            _write_generation(run_dir, meta, list(pool), list(trace))
            if k != kmax:
                with open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8') as log:
                    log.write(f"generated as the first {counts[k]} databases of the b{kmax}x stream "
                              f"(seed {seed}); generation log: ../../b{kmax}x/rep_{rep:02d}/log.txt\n")
            print(f"b{k}x written: {counts[k]} databases ({failures} failed so far), "
                  f"{len(in_scope - remaining)}/{len(in_scope)} in-scope rules verified, pool {len(pool)}, "
                  f"{runtime:.1f}s ({meta['seconds_per_database']}s/db)", flush=True)


def job_validate(run_dir, case_study):
    records = H._load_records(case_study)
    by_rule, oos = H._scope(case_study, records)
    with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(run_dir, 'pool', 'databases.pkl'), 'rb') as f:
        pool = pickle.load(f)
    ctx, decisions, overrides, _rules_of = _validator_context(case_study)
    ddl = _ddl_script(case_study)
    t0 = time.time()
    matrix, errors = [], []
    decision_errors = {}
    for i, p in enumerate(pool):
        try:
            conn = to_sqlite(p['candidate'], case_study, ddl)
            try:
                matrix.append(_verify(ctx, conn, case_study, decisions, set(decisions), overrides,
                                      decision_errors))
            finally:
                conn.close()
        except Exception as e:
            errors.append(f'pool {i}: {type(e).__name__}: {e}')
            matrix.append(set())
    verified = set().union(*matrix) if matrix else set()
    during_run = {rid for p in pool for rid in p['new_rules']}
    if verified != during_run:
        errors.append(f'matrix union differs from generation-time coverage: '
                      f'+{sorted(verified - during_run)} -{sorted(during_run - verified)}')

    H._write_csv(os.path.join(run_dir, 'matrix.csv'),
                 ['pool_index', 'database_index', 'new_rules_when_generated', 'num_rules_verified', 'rules_verified'],
                 [{'pool_index': i, 'database_index': p['database_index'],
                   'new_rules_when_generated': '; '.join(p['new_rules']),
                   'num_rules_verified': len(matrix[i]), 'rules_verified': '; '.join(sorted(matrix[i]))}
                  for i, p in enumerate(pool)])
    greedy, minimum, optimal = minimum_cover(verified, {i: matrix[i] for i in range(len(pool))})
    assert set().union(*(matrix[i] for i in minimum)) >= verified if minimum else not verified
    suite = {'union': {'pool': len(pool), 'greedy': len(greedy), 'min': len(minimum), 'optimal': optimal,
                       'min_members': minimum, 'greedy_members': greedy}}
    with open(os.path.join(run_dir, 'minimization.json'), 'w', encoding='utf-8') as f:
        json.dump(suite, f, indent=2)
    os.makedirs(os.path.join(run_dir, 'minimal_suite'), exist_ok=True)
    with open(os.path.join(run_dir, 'minimal_suite', 'union.pkl'), 'wb') as f:
        pickle.dump({'pool_indices': minimum, 'individuals': [pool[i]['candidate'] for i in minimum],
                     'rules_covered': sorted(verified)}, f)

    per_rule = [{'rule_id': rule, 'decision': recs[0]['decision_name'],
                 'in_scope': 'True' if rule not in oos else 'False', 'num_records': len(recs),
                 'claimed_archive': '', 'claimed_final': '', 'claimed_union': '',
                 'verified_archive': '', 'verified_final': '',
                 'verified_union': 'True' if rule in verified else 'False'}
                for rule, recs in sorted(by_rule.items())]
    H._write_csv(os.path.join(run_dir, 'per_rule.csv'), list(per_rule[0]), per_rule)

    meta.update({'validation_mode': 'full', 'unevaluable_decisions_in_pool_check': decision_errors,
                 'suite_size': {'union': {x: suite['union'][x] for x in ('pool', 'greedy', 'min', 'optimal')}},
                 'validation_errors': errors, 'validation_seconds': round(time.time() - t0, 1)})
    with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(run_dir, 'VALIDATED'), 'w').close()
    in_scope = {r for r in by_rule if r not in oos}
    print(f"validated: {len(verified & in_scope)} of {len(in_scope)} in-scope rules, pool {len(pool)}, "
          f"suite min {len(minimum)} (greedy {len(greedy)}, optimal {optimal}), "
          f"{len(errors)} errors, {time.time() - t0:.1f}s")


def _spawn(args, log_path):
    import subprocess
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONIOENCODING='utf-8')
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run([sys.executable, '-m', 'evaluation.baselines.schema_random'] + args,
                              stdout=log, stderr=subprocess.STDOUT, env=env, cwd=paths.ROOT)
    return proc.returncode


def _write_config(a, out_name, calib):
    exp = os.path.join(H.OUT_ROOT, out_name)
    path = os.path.join(exp, 'config.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
        old_mode = old.get('mode', 'separate')
        if old_mode != a.mode:
            sys.exit(f"outputs/{out_name} already holds --mode {old_mode} runs; mixing modes in "
                     f"one folder is not allowed. Delete or rename that folder first, then rerun.")
    os.makedirs(exp, exist_ok=True)
    with open(os.path.join(exp, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=2)
    seed_rule = ('prefix: one stream per (case study, rep), seed = 1000 * max(budgets) + rep; smaller budgets '
                 'are its first floor(k*B / #objectives) databases' if a.mode == 'prefix'
                 else 'seed = 1000 * budget_multiplier + rep')
    config = {'experiment': a.name, 'output_folder': out_name, 'reps': a.reps, 'first_rep': a.first_rep,
              'case_studies': a.case_studies, 'setups': [SETUP], 'budgets': a.budgets, 'mode': a.mode,
              'seed_rule': seed_rule,
              'budget_1x': {cs: calib[cs]['budget_1x'] for cs in a.case_studies},
              'not_persisted_files': {cs: paths.config(cs)['not_persisted'] for cs in a.case_studies},
              'generator_settings': GENERATOR_SETTINGS, 'max_databases': a.max_databases,
              'compiled_constraints_sha256': H.corpus_hash(), 'git_commit': H._git_commit(),
              'written_at': datetime.datetime.now().isoformat(timespec='seconds')}
    for cs in a.case_studies:
        if calib[cs].get('compiled_constraints_sha256') != config['compiled_constraints_sha256']:
            print(f"WARNING: compiled_constraints.json changed since {cs} was calibrated -- "
                  f"results would not be comparable with the existing experiment.")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def cmd_run(a):
    calib = H._load_calibration(a.calibration or a.name)
    missing = [cs for cs in a.case_studies if cs not in calib]
    if missing:
        sys.exit(f"Not calibrated: {missing}")
    out_name = part_name(a.name, a.first_rep)
    _write_config(a, out_name, calib)
    budgets = sorted(a.budgets)
    groups = [budgets] if a.mode == 'prefix' else [[k] for k in budgets]
    jobs = [(cs, rep, H._seed_for(max(g), rep), [(k, k * calib[cs]['budget_1x']) for k in g])
            for cs in H._by_size(a.case_studies)
            for rep in range(a.first_rep, a.first_rep + a.reps) for g in groups]
    todo = []
    for cs, rep, seed, targets in jobs:
        dirs = {k: _run_dir(out_name, cs, k, rep) for k, _b in targets}
        need_val = [k for k, d in dirs.items() if not os.path.exists(os.path.join(d, 'VALIDATED'))]
        if need_val:
            need_gen = any(not os.path.exists(os.path.join(d, 'SEARCH_DONE')) for d in dirs.values())
            todo.append(((cs, rep, seed, targets), dirs, need_gen, need_val))
    print(f"Output folder: outputs/{out_name}  (repetitions {a.first_rep}-{a.first_rep + a.reps - 1}, "
          f"--mode {a.mode})")
    print(f"{len(todo)} job(s) to do ({len(jobs) - len(todo)} already finished), {a.workers} worker(s)\n")

    def fn(item):
        (cs, rep, seed, targets), dirs, need_gen, need_val = item
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        kmax = max(dirs)
        if need_gen:
            for d in dirs.values():
                for marker in ('SEARCH_DONE', 'VALIDATED'):
                    if os.path.exists(os.path.join(d, marker)):
                        os.remove(os.path.join(d, marker))
            need_val = sorted(dirs)
            args = ['_generate', out_name, cs, str(rep), str(seed), str(a.max_databases or 0), a.mode]
            for k, b in targets:
                args += [str(k), str(b)]
            rc = _spawn(args, os.path.join(dirs[kmax], 'log.txt'))
            if rc != 0:
                return rc
        for k in need_val:
            rc = _spawn(['_validate', dirs[k], cs], os.path.join(dirs[k], 'log.txt'))
            if rc != 0:
                return rc
        return 0

    H._execute(todo, a.workers, fn,
               lambda item: f"{item[0][0]} {SETUP} rep_{item[0][1]:02d} "
                            f"({', '.join(f'b{k}x' for k, _b in item[0][3])})")


def cmd_status(a):
    out_name = part_name(a.name, a.first_rep)
    base = os.path.join(H.OUT_ROOT, out_name)
    if not os.path.isdir(base):
        sys.exit(f"No output folder outputs/{out_name}")
    print(f"{'Case study':<10} {'Budget':>6} {'started':>8} {'generated':>10} {'validated':>10}")
    for cs in H.CASE_STUDIES:
        sdir = os.path.join(base, cs, SETUP)
        if not os.path.isdir(sdir):
            continue
        for bdir in sorted(os.listdir(sdir), key=lambda b: int(b[1:-1])):
            reps = [os.path.join(sdir, bdir, r) for r in os.listdir(os.path.join(sdir, bdir))]
            print(f"{cs:<10} {bdir[1:]:>6} {len(reps):>8} "
                  f"{sum(os.path.exists(os.path.join(r, 'SEARCH_DONE')) for r in reps):>10} "
                  f"{sum(os.path.exists(os.path.join(r, 'VALIDATED')) for r in reps):>10}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_generate':
        out_name, cs, rep, seed, cap, mode = sys.argv[2:8]
        rest = [int(x) for x in sys.argv[8:]]
        targets = list(zip(rest[0::2], rest[1::2]))
        job_generate(out_name, cs, int(rep), int(seed), targets, int(cap) or None, mode)
        return
    if len(sys.argv) > 1 and sys.argv[1] == '_validate':
        job_validate(sys.argv[2], sys.argv[3])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('--name', required=True, help='experiment name, e.g. thesis (uses calibrations/<name>.json)')
    p.add_argument('--calibration', help='calibration to use if different from --name')
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--first-rep', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--case-studies', nargs='+', default=list(H.CASE_STUDIES), choices=H.CASE_STUDIES)
    p.add_argument('--budgets', nargs='+', type=int, default=list(H.BUDGETS))
    p.add_argument('--max-databases', type=int, default=0, help='smoke tests only: cap databases per run')
    p.add_argument('--mode', choices=['prefix', 'separate'], default='prefix',
                   help='prefix (default): one stream per (case study, rep) up to the largest budget, smaller '
                        'budgets read off its prefix. separate: one run per budget (seed 1000*k + rep)')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('status')
    p.add_argument('--name', required=True)
    p.add_argument('--first-rep', type=int, default=0)
    p.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()

