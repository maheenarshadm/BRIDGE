"""Many-objective search based on DynaMOSA.

One population of individuals is evolved against all objectives of a case
study. Objectives whose upstream prerequisites are not yet satisfied are kept
out of the search until they become reachable. Offspring are produced by
table-level crossover and by fitness-guided mutation of the grounded inputs
of randomly chosen active objectives. Survivors are selected by NSGA-II
non-dominated sorting and crowding distance over the active objectives, and
an archive keeps the best individual found for every objective. The search
stops when every objective is satisfied or the budget of fitness evaluations
is used."""
import copy
import math
import random

from bridge.search.candidate import (Candidate, derive_genome, build_seed_candidate, _OWNER_KEY, known_constant, _PLACEHOLDER_RE)
from bridge.search.fitness import branch_fitness, FitnessEvaluationError, _unique_key_sets, EvaluationBudgetExhausted, reset_evaluation_counter, clear_evaluation_limit, evaluations_used
from bridge.search.mutation import repair_candidate, best_value_for, apply_mutation, _leaf_variables, _schema_for, candidate_values, _fresh_key_value
from bridge.search.crossover import crossover


def _focal_tables_for_leaf(node):
    kind = node.get('kind')
    if kind in ('schema_column', 'null_check', 'derived_case'):
        return {node['table']}
    if kind == 'any_not_null':
        return {c['table'] for c in node.get('columns', [])}
    if kind in ('join_lookup', 'join_null_check'):
        return {node['via']['local_table']}
    if kind == 'regex_match':
        return {node['value_column']['table'], node['pattern_column']['table']}
    return set()


def _focal_table_set_for(record):
    tables = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get('kind') == 'substituted_decision':
            for sub in node.get('free_variable_resolutions', {}).values():
                walk(sub)
            return
        tables.update(_focal_tables_for_leaf(node))

    for node in record.get('variable_resolution', {}).values():
        walk(node)
    return tables


def _focal_for_read(record, focal_maps):
    return dict(focal_maps.get(record['record_id'], {}))


def _focal_for_mutate(record, candidate, focal_maps, table_cache):
    tables = table_cache.get(record['record_id'])
    if tables is None:
        tables = _focal_table_set_for(record)
        table_cache[record['record_id']] = tables
    rec_focal = focal_maps.setdefault(record['record_id'], {})
    focal = {}
    for t in tables:
        tU = t.upper()
        row = rec_focal.get(tU)
        if row is None:
            row = candidate.add_row(t, {_OWNER_KEY: record['record_id']})
            rec_focal[tU] = row
        focal[tU] = row
    return focal


def evaluate_objective(record, candidate, focal_maps, scenario_maps, table_cache):
    try:
        focal = _focal_for_read(record, focal_maps)
        scenario = scenario_maps.get(record['record_id'], {})
        genome = derive_genome(record, candidate, focal, scenario, owner_id=record['record_id'])
        return branch_fitness(record, genome)
    except FitnessEvaluationError:
        return float('inf')


_KICK_PROBABILITY = 0.15


def _kick_value_for(record, var_name, node, current, case_study, rng):
    big_step = rng.randint(2, 40)
    options = candidate_values(record, var_name, node, current, case_study, step=big_step)
    return rng.choice(options) if options else None


def _mutate_objective(record, candidate, focal_maps, scenario_maps, table_cache, rng=None,
                       kick_probability=_KICK_PROBABILITY):
    rng = rng or random
    rid = record['record_id']
    scenario = scenario_maps.get(rid, {})
    focal = _focal_for_mutate(record, candidate, focal_maps, table_cache)
    try:
        genome = derive_genome(record, candidate, focal, scenario, owner_id=rid)
        leaves = [(v, n) for v, n in _leaf_variables(record) if v in genome]
        if not leaves:
            return candidate, focal_maps, scenario_maps, False
        var_name, node = rng.choice(leaves)
        pinned = known_constant(record['case_study'], var_name)
        if pinned is None and rng.random() < kick_probability:
            value = _kick_value_for(record, var_name, node, genome.get(var_name), record['case_study'], rng)
            if value is None:
                return candidate, focal_maps, scenario_maps, False
        else:
            value, _best_fitness, improved = best_value_for(record, genome, var_name, node, record['case_study'])
            if not improved:
                return candidate, focal_maps, scenario_maps, False
    except FitnessEvaluationError:
        return candidate, focal_maps, scenario_maps, False

    new_candidate, new_focal_maps, new_scenario_maps = copy.deepcopy((candidate, focal_maps, scenario_maps))
    new_focal = new_focal_maps[rid]
    new_scenario = new_scenario_maps.setdefault(rid, {})
    try:
        apply_mutation(record, new_candidate, new_focal, new_scenario, var_name, node, value, genome.get(var_name),
                        owner_id=rid)
    except FitnessEvaluationError:
        return candidate, focal_maps, scenario_maps, False
    return new_candidate, new_focal_maps, new_scenario_maps, True


_LOCAL_BURST_CAP = 8


def _local_burst_size(record):
    return max(1, min(_LOCAL_BURST_CAP, len(_leaf_variables(record))))


def _branch_key(record):
    return f"{record['decision_name']}::{record['rule_id']}"


def is_active(record, covered_keys):
    deps = record.get('grounded_upstream_branches') or []
    return all(d in covered_keys for d in deps)


def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(fitness_vectors):
    n = len(fitness_vectors)
    dominated_by = [set() for _ in range(n)]
    domination_count = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(fitness_vectors[p], fitness_vectors[q]):
                dominated_by[p].add(q)
            elif _dominates(fitness_vectors[q], fitness_vectors[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def crowding_distance(front, fitness_vectors):
    distances = {i: 0.0 for i in front}
    if not front or not fitness_vectors[front[0]]:
        return distances
    num_obj = len(fitness_vectors[front[0]])
    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda i: fitness_vectors[i][m])
        distances[front_sorted[0]] = float('inf')
        distances[front_sorted[-1]] = float('inf')
        vmin, vmax = fitness_vectors[front_sorted[0]][m], fitness_vectors[front_sorted[-1]][m]
        if vmax == vmin or vmax == float('inf'):
            continue
        for k in range(1, len(front_sorted) - 1):
            distances[front_sorted[k]] += (
                (fitness_vectors[front_sorted[k + 1]][m] - fitness_vectors[front_sorted[k - 1]][m])
                / (vmax - vmin))
    return distances


def _key_columns_for(schema, table):
    real = table if table in schema else (
        table.upper() if table.upper() in schema else (
            table.lower() if table.lower() in schema else table))
    info = schema.get(real, {})
    cols = _own_unique_columns_for(schema, table)
    cols |= {fk['column'].upper() for fk in (info.get('fk_columns') or [])}
    return cols


def _own_unique_columns_for(schema, table):
    real = table if table in schema else (
        table.upper() if table.upper() in schema else (
            table.lower() if table.lower() in schema else table))
    return {c.upper() for keyset in _unique_key_sets(schema, real) for c in keyset}


def _own_solo_unique_columns_for(schema, table):
    real = table if table in schema else (
        table.upper() if table.upper() in schema else (
            table.lower() if table.lower() in schema else table))
    cols = set()
    for keyset in _unique_key_sets(schema, real):
        if len(keyset) == 1:
            cols.add(keyset[0].upper())
    return cols


_COMPOSITE_KEY_DEDUP_COLUMN = {}


def _dedup_composite_keys(schema, table, row_copy, used_key_values, case_study=None):
    override_col = _COMPOSITE_KEY_DEDUP_COLUMN.get((case_study, table.upper()))
    if override_col is None:
        return
    real = table if table in schema else (
        table.upper() if table.upper() in schema else (
            table.lower() if table.lower() in schema else table))
    for keyset in _unique_key_sets(schema, real):
        if len(keyset) < 2:
            continue
        keyset_upper = [c.upper() for c in keyset]
        if override_col.upper() not in keyset_upper:
            continue

        def _get(col):
            return row_copy.get(col, row_copy.get(col.upper(), row_copy.get(col.lower())))

        tup = tuple(_get(c) for c in keyset)
        if any(v is None for v in tup):
            continue
        used = used_key_values.setdefault((table.upper(), tuple(keyset_upper)), set())
        if tup in used:
            idx = keyset_upper.index(override_col.upper())
            tup_list = list(tup)
            while tuple(tup_list) in used:
                tup_list[idx] += 1
            actual_col = next(c for c in row_copy if c.upper() == override_col.upper())
            row_copy[actual_col] = tup_list[idx]
            tup = tuple(tup_list)
        used.add(tup)


_SEED_KEY_OFFSET_UNIT = 1_000_000


def _seed_shared_population(records, case_study, population_size):
    schema = _schema_for(case_study)
    base_scenario_maps = {}
    base = Candidate()
    base_focal_maps = {}
    for i, r in enumerate(records):
        rid = r['record_id']
        c, f, s = build_seed_candidate(r)
        offset = i * _SEED_KEY_OFFSET_UNIT
        for key, val in s.items():
            if key != '__today__' and isinstance(val, (int, float)) and not isinstance(val, bool):
                s[key] = val + offset
        base_scenario_maps[rid] = s
        id_to_copy = {}
        for table, rows in c.as_dict().items():
            key_cols = _key_columns_for(schema, table)
            for row in rows:
                if key_cols:
                    for col in list(row):
                        if col.upper() in key_cols and isinstance(row[col], (int, float)) \
                                and not isinstance(row[col], bool):
                            row[col] = row[col] + offset
                row_copy = dict(row)
                row_copy[_OWNER_KEY] = rid
                base.add_row(table, row_copy)
                id_to_copy[id(row)] = row_copy
        rec_focal = {table: id_to_copy[id(row)] for table, row in f.items() if id(row) in id_to_copy}
        if rec_focal:
            base_focal_maps[rid] = rec_focal
    repair_candidate(base, case_study)
    population = [copy.deepcopy((base, base_focal_maps, base_scenario_maps)) for _ in range(population_size)]
    return population


def _mutations_per_child(active, population_size, mutations_per_child):
    if mutations_per_child == 'auto':
        return max(1, math.ceil(len(active) / (2 * population_size))) if active else 1
    return mutations_per_child


def _apply_cross_table_placeholder_correlations(candidate, rec_focal, rec_scenario, r, rid, schema):
    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get('kind') == 'substituted_decision':
            for child in node.get('free_variable_resolutions', {}).values():
                visit(child)
            return
        for placeholder, source in (node.get('cross_table_placeholders') or {}).items():
            value = rec_scenario.get(placeholder)
            if value is None:
                continue
            target_focal = rec_focal.get(source['table']) or next(
                (v for k, v in rec_focal.items() if k.upper() == source['table'].upper()), None)
            if target_focal is None:
                target_focal = candidate.add_row(source['table'], {_OWNER_KEY: rid})
                rec_focal[source['table']] = target_focal
            target_focal[source['column']] = value
            _bump_colliding_rows(candidate, schema, source['table'], source['column'], value, target_focal)

    for node in r.get('variable_resolution', {}).values():
        visit(node)


def _bump_colliding_rows(candidate, schema, table, column, value, keep_row):
    if schema is None:
        return
    if column.upper() not in _own_solo_unique_columns_for(schema, table):
        return
    for other_row in candidate.rows(table):
        if other_row is not keep_row and other_row.get(column) == value:
            other_row[column] = _fresh_key_value(candidate, table, column)


def _rename_owned_fk_refs(candidate, schema, table, column, old_value, new_value, rid, skip_row):
    if schema is None:
        return

    def col_key(row, name):
        return next((k for k in row if k.upper() == name.upper()), None)

    for ref_table, info in schema.items():
        for fk in (info or {}).get('fk_columns') or []:
            if fk['ref_table'].upper() != table.upper() or fk['ref_column'].upper() != column.upper():
                continue
            for row in candidate.rows(ref_table):
                if row is skip_row or row.get(_OWNER_KEY) != rid:
                    continue
                k = col_key(row, fk['column'])
                if k is not None and row[k] == old_value:
                    row[k] = new_value


def _build_decision_subject_row(candidate, rec_focal, r, rid, schema=None):
    subject = r.get('decision_subject')
    if not subject:
        return
    subject_focal = rec_focal.get(subject['table']) or next(
        (v for k, v in rec_focal.items() if k.upper() == subject['table'].upper()), None)
    is_new = subject_focal is None
    if is_new:
        subject_focal = {}
    resolvable = bool(subject['joins'])
    wired_any = False
    for target_table, hops in subject['joins'].items():
        from_row = subject_focal
        for hop in hops:
            if 'from_columns' in hop:
                def _ci(row, col):
                    return next((v for k, v in row.items() if k.upper() == col.upper()), None)
                values = [_ci(from_row, c) for c in hop['from_columns']]
                if any(v is None for v in values):
                    break
                target_focal = next((row for row in candidate.rows(hop['to_table'])
                                     if all(_ci(row, c) == v for c, v in zip(hop['to_columns'], values))), None)
                if target_focal is None:
                    target_focal = (rec_focal.get(hop['to_table']) or next(
                        (v for k, v in rec_focal.items() if k.upper() == hop['to_table'].upper()), None)
                        or next((row for row in candidate.rows(hop['to_table'])
                                 if row.get(_OWNER_KEY) == rid), None)
                        or candidate.add_row(hop['to_table'], {_OWNER_KEY: rid}))
                    for c, v in zip(hop['to_columns'], values):
                        k = next((k for k in target_focal if k.upper() == c.upper()), c)
                        target_focal[k] = v
                rec_focal[hop['to_table']] = target_focal
                wired_any = True
                from_row = target_focal
                continue
            target_focal = rec_focal.get(hop['to_table']) or next(
                (v for k, v in rec_focal.items() if k.upper() == hop['to_table'].upper()), None)
            if target_focal is None:
                target_focal = next((row for row in candidate.rows(hop['to_table'])
                                      if row.get(_OWNER_KEY) == rid), None)
            if target_focal is None:
                target_focal = candidate.add_row(hop['to_table'], {_OWNER_KEY: rid})
            rec_focal[hop['to_table']] = target_focal

            existing_from_value = from_row.get(hop['from_column'])
            if existing_from_value is not None:
                if target_focal.get(hop['to_column']) != existing_from_value:
                    old_value = target_focal.get(hop['to_column'])
                    target_focal[hop['to_column']] = existing_from_value
                    wired_any = True
                    _bump_colliding_rows(candidate, schema, hop['to_table'], hop['to_column'],
                                          existing_from_value, target_focal)
                    if old_value is not None:
                        _rename_owned_fk_refs(candidate, schema, hop['to_table'], hop['to_column'],
                                              old_value, existing_from_value, rid, target_focal)
                from_row = target_focal
                continue
            pk_value = target_focal.get(hop['to_column'])
            if pk_value is None:
                pk_value = _fresh_key_value(candidate, hop['to_table'], hop['to_column'])
                target_focal[hop['to_column']] = pk_value
            from_row[hop['from_column']] = pk_value
            wired_any = True
            _bump_colliding_rows(candidate, schema, hop['to_table'], hop['to_column'], pk_value, target_focal)
            from_row = target_focal
    if resolvable and is_new and (subject_focal or wired_any):
        subject_focal[_OWNER_KEY] = rid
        candidate.add_row(subject['table'], subject_focal)
        rec_focal[subject['table']] = subject_focal


def _deep_copy_individual(individual):
    candidate, focal_maps, scenario_maps = individual
    id_to_copy = {}
    new_candidate = Candidate()
    for table, rows in candidate.as_dict().items():
        for row in rows:
            row_copy = dict(row)
            new_candidate.add_row(table, row_copy)
            id_to_copy[id(row)] = row_copy
    new_focal_maps = {}
    for rid, rec_focal in focal_maps.items():
        new_rec_focal = {}
        for table, row in rec_focal.items():
            copy = id_to_copy.get(id(row))
            if copy is not None:
                new_rec_focal[table] = copy
        new_focal_maps[rid] = new_rec_focal
    new_scenario_maps = {rid: dict(scenario) for rid, scenario in scenario_maps.items()}
    return (new_candidate, new_focal_maps, new_scenario_maps)


def run_dynamosa(records, case_study, population_size=20, generations=50, rng=None,
                  mutations_per_child='auto', kick_probability=_KICK_PROBABILITY, dynamic_gating=True,
                  use_local_burst=True, max_evaluations=None, trace=None):
    if generations is None and max_evaluations is None:
        raise ValueError('run_dynamosa needs a generation cap, an evaluation budget, or both')
    if max_evaluations is not None:
        reset_evaluation_counter(max_evaluations)
    try:
        return _run_dynamosa_loop(records, case_study, population_size, generations, rng,
                                  mutations_per_child, kick_probability, dynamic_gating,
                                  use_local_burst, trace, max_evaluations is not None)
    finally:
        if max_evaluations is not None:
            clear_evaluation_limit()


def _run_dynamosa_loop(records, case_study, population_size, generations, rng,
                       mutations_per_child, kick_probability, dynamic_gating, use_local_burst, trace,
                       stop_when_all_covered):
    state = {'stop_when_all_covered': stop_when_all_covered}
    try:
        _dynamosa_generations(records, case_study, population_size, generations, rng,
                              mutations_per_child, kick_probability, dynamic_gating,
                              use_local_burst, trace, state)
    except EvaluationBudgetExhausted:
        pass
    return state['archive'], state['coverage_history'], state['population']


def _dynamosa_generations(records, case_study, population_size, generations, rng,
                          mutations_per_child, kick_probability, dynamic_gating,
                          use_local_burst, trace, state):
    rng = rng or random.Random(0)
    table_cache = {}
    population = _seed_shared_population(records, case_study, population_size)

    archive = {r['record_id']: (float('inf'), population[0]) for r in records}
    coverage_history = []
    state.update(archive=archive, coverage_history=coverage_history, population=population)

    def update_archive(individual):
        candidate, focal_maps, scenario_maps = individual
        for r in records:
            rid = r['record_id']
            f = evaluate_objective(r, candidate, focal_maps, scenario_maps, table_cache)
            if f < archive[rid][0]:
                if f == 0.0 and trace is not None:
                    trace.append((evaluations_used(), rid))
                archive[rid] = (f, _deep_copy_individual(individual))

    for ind in population:
        update_archive(ind)

    gen = 0
    while generations is None or gen < generations:
        gen += 1
        if state['stop_when_all_covered'] and all(archive[r['record_id']][0] == 0.0 for r in records):
            break
        covered_keys = {_branch_key(r) for r in records
                         if archive.get(r['record_id'], (float('inf'), None))[0] == 0.0}
        active = records if not dynamic_gating else [r for r in records if is_active(r, covered_keys)]

        k = _mutations_per_child(active, population_size, mutations_per_child)

        offspring = []
        while len(offspring) < population_size:
            p1, p2 = rng.sample(population, 2) if len(population) >= 2 else (population[0], population[0])
            (p1_c, p1_fm, p1_sm), (p2_c, p2_fm, p2_sm) = p1, p2
            c1, c2, _f1, _f2, fm1, fm2, sm1, sm2 = crossover(
                p1_c, p2_c, case_study, rng, focal_maps1=p1_fm, focal_maps2=p2_fm,
                scenario_maps1=p1_sm, scenario_maps2=p2_sm)
            for child_c, child_fm, child_sm in ((c1, fm1, sm1), (c2, fm2, sm2)):
                if active:
                    for r in rng.sample(active, min(k, len(active))):
                        burst = _local_burst_size(r) if use_local_burst else 1
                        for _ in range(burst):
                            child_c, child_fm, child_sm, _improved = _mutate_objective(
                                r, child_c, child_fm, child_sm, table_cache, rng,
                                kick_probability=kick_probability)
                offspring.append((child_c, child_fm, child_sm))
        offspring = offspring[:population_size]

        for child in offspring:
            update_archive(child)

        combined = population + offspring
        fitness_vectors = [[evaluate_objective(r, ind[0], ind[1], ind[2], table_cache)
                             for r in active] for ind in combined]
        fronts = fast_non_dominated_sort(fitness_vectors)
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(combined[i] for i in front)
            else:
                cd = crowding_distance(front, fitness_vectors)
                front_sorted = sorted(front, key=lambda i: -cd[i])
                new_population.extend(combined[i] for i in front_sorted[:population_size - len(new_population)])
                break
        population = new_population or population
        state['population'] = population

        covered_count = sum(1 for r in records
                             if archive.get(r['record_id'], (float('inf'), None))[0] == 0.0)
        coverage_history.append(covered_count)


def _scenario_keys_needing_offset(record):
    needed = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get('kind') == 'substituted_decision':
            for sub in node.get('free_variable_resolutions', {}).values():
                walk(sub)
            return
        for field in ('filter_text', 'sql_template'):
            for m in _PLACEHOLDER_RE.finditer(node.get(field) or ''):
                needed.add(m.group(1))

    for node in record.get('variable_resolution', {}).values():
        walk(node)
    return needed


