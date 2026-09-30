"""The BRIDGE pipeline for one case study.

Phase 1 (grounding) is performed once by `bridge.grounding.compile_constraints`
and yields the search objectives in `case_studies/compiled_constraints.json`.
This module runs the remaining phases:

  Phase 2  `run_bridge`       many-objective search (DynaMOSA) over whole
                              database states
  Phase 3  `build_database`   turns a searched individual into a SQLite database
           `verify_database`  re-executes the DMN decisions against it with the
                              independent validator
           `validate_pool`    both steps for a list of individuals
           `generate`         search, validation and exact test suite
                              minimization, writing the minimal suite as
                              SQLite databases
"""
import hashlib
import json
import os
import pickle
import random
import shutil
import sqlite3
import traceback

from bridge import paths
from bridge.minimization import minimum_cover

POPULATION_SIZE = 30


def load_records(case_study):
    with open(paths.COMPILED_PATH, encoding='utf-8') as f:
        return [r for r in json.load(f) if r['case_study'] == case_study]


def not_persisted_values(case_study):
    """One dictionary of fixed input values per validation setting."""
    values = []
    for path in paths.not_persisted_files(case_study):
        with open(path, encoding='utf-8') as f:
            values.append(json.load(f))
    return values


def run_bridge(records, case_study, budget, seed, trace=None, population_size=POPULATION_SIZE):
    """Search until `budget` fitness evaluations are used.

    Returns (archive, history, final population); the archive maps every
    objective to (best fitness, individual)."""
    from bridge.search import dynamosa
    random.seed(seed)
    rng = random.Random(seed)
    return dynamosa.run_dynamosa(records, case_study, population_size=population_size,
                                 generations=None, rng=rng, max_evaluations=budget, trace=trace)


def build_database(individual, case_study, db_path, records=None):
    from bridge.construction import assemble
    from bridge.search.dynamosa import _deep_copy_individual
    from bridge.search.mutation import _schema_for, repair_candidate
    records = records if records is not None else load_records(case_study)
    schema = _schema_for(case_study)
    records_index = {r['record_id']: i for i, r in enumerate(records)}
    records_by_id = {r['record_id']: r for r in records}
    work_c, work_fm, work_sm = _deep_copy_individual(individual)
    assemble._offset_rows_by_owner(work_c, schema, records_index, case_study)
    assemble._apply_cross_table_placeholder_correlations_for_individual(
        work_c, work_fm, work_sm, records_by_id, records_index, schema)
    assemble._build_decision_subject_rows_for_individual(work_c, work_fm, records_by_id, schema)
    repair_candidate(work_c, case_study)
    assemble._materialize(work_c, case_study, db_path)
    return db_path


def verify_database(db_path, case_study, overrides, decisions_by_name=None):
    """Rules verified by the database under one validation setting (`overrides`
    gives the values of inputs that are not stored in the database)."""
    from bridge.validation.phase1_utility import records_by_decision
    from bridge.validation.verify import verify_decisions
    decisions_by_name = decisions_by_name or records_by_decision(case_study)
    conn = sqlite3.connect(db_path)
    try:
        result = verify_decisions(conn, case_study, decisions_by_name, set(decisions_by_name), overrides)
    finally:
        conn.close()
    return {rid for rid, ok in result.items() if ok}


def validate_pool(case_study, pool, dbs_dir, records=None, log=print):
    """Build and verify every individual of `pool`.

    Returns ([set of verified rule ids per individual], errors)."""
    from bridge.validation.phase1_utility import records_by_decision
    records = records if records is not None else load_records(case_study)
    settings = not_persisted_values(case_study)
    decisions_by_name = records_by_decision(case_study)
    os.makedirs(dbs_dir, exist_ok=True)
    matrix, errors = [], []
    for idx, individual in enumerate(pool):
        rules = set()
        try:
            db_path = os.path.join(dbs_dir, f'individual_{idx:03d}.db')
            build_database(individual, case_study, db_path, records)
            for overrides in settings:
                rules |= verify_database(db_path, case_study, overrides, decisions_by_name)
        except Exception as e:  # noqa: BLE001 -- one failing individual must not stop the run
            errors.append(f'individual {idx}: {type(e).__name__}: {e}')
            traceback.print_exc()
        matrix.append(rules)
        if log:
            log(f"  individual {idx + 1}/{len(pool)}: {len(rules)} rule(s)")
    return matrix, errors


def content_key(individual):
    return hashlib.sha1(pickle.dumps(individual, protocol=4)).hexdigest()


def generate(case_study, budget, seed, out_dir, population_size=POPULATION_SIZE):
    """Run BRIDGE end to end and write the minimal test suite.

    Writes to `out_dir`: `minimal_suite/database_NN.db` (the test databases),
    `coverage.json` (the rules each database verifies) and `summary.json`."""
    records = load_records(case_study)
    archive, _history, population = run_bridge(records, case_study, budget, seed,
                                               population_size=population_size)
    pool, seen = [], set()
    for _rid, (_fit, ind) in archive.items():
        key = content_key(ind)
        if key not in seen:
            seen.add(key)
            pool.append(ind)
    for ind in population:
        key = content_key(ind)
        if key not in seen:
            seen.add(key)
            pool.append(ind)
    work_dir = os.path.join(out_dir, 'validated')
    matrix, errors = validate_pool(case_study, pool, work_dir, records)
    verified = set().union(*matrix) if matrix else set()
    _greedy, minimum, optimal = minimum_cover(verified, dict(enumerate(matrix)))
    suite_dir = os.path.join(out_dir, 'minimal_suite')
    os.makedirs(suite_dir, exist_ok=True)
    coverage = {}
    for n, idx in enumerate(minimum, 1):
        name = f'database_{n:02d}.db'
        shutil.copyfile(os.path.join(work_dir, f'individual_{idx:03d}.db'), os.path.join(suite_dir, name))
        coverage[name] = sorted(matrix[idx])
    shutil.rmtree(work_dir, ignore_errors=True)
    rule_ids = sorted({r['rule_id'] for r in records})
    summary = {'case_study': case_study, 'budget_evaluations': budget, 'seed': seed,
               'rules': len(rule_ids), 'rules_verified': len(verified),
               'databases_validated': len(pool), 'minimal_suite_size': len(minimum),
               'minimal_suite_optimal': optimal, 'validation_errors': errors,
               'rules_not_verified': sorted(set(rule_ids) - verified)}
    with open(os.path.join(out_dir, 'coverage.json'), 'w', encoding='utf-8') as f:
        json.dump(coverage, f, indent=2)
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary
