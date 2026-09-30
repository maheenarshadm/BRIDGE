"""GM (grounded mutation), an unguided variant of BRIDGE.

GM uses the chromosome, the grounding and the mutation operators of BRIDGE.
Every generation, each individual of the population receives random mutations
of the grounded inputs of the active objectives, with no fitness-based
selection. An archive keeps the best individual found for every objective."""
import copy
import random


from bridge.search.candidate import derive_genome
from bridge.search.fitness import FitnessEvaluationError, EvaluationBudgetExhausted, reset_evaluation_counter, clear_evaluation_limit, evaluations_used
from bridge.search.mutation import _leaf_variables, candidate_values, apply_mutation, BOOLEAN_LEAF_KINDS
from bridge.search.dynamosa import _seed_shared_population, is_active, _branch_key, _mutations_per_child, _local_burst_size, _focal_for_mutate, evaluate_objective


def _random_value_for(record, genome, var_name, node, case_study, rng):
    current = genome.get(var_name)
    if node.get('kind') in BOOLEAN_LEAF_KINDS:
        return rng.choice([True, False])
    options = candidate_values(record, var_name, node, current, case_study)
    if not options:
        return None
    return rng.choice(options)


def _random_mutate_lineage(record, candidate, focal_maps, scenario_maps, table_cache, rng):
    rid = record['record_id']
    scenario = scenario_maps.get(rid, {})
    focal = _focal_for_mutate(record, candidate, focal_maps, table_cache)
    try:
        genome = derive_genome(record, candidate, focal, scenario, owner_id=rid)
        leaves = [(v, n) for v, n in _leaf_variables(record) if v in genome]
        if not leaves:
            return candidate, focal_maps, scenario_maps
        var_name, node = rng.choice(leaves)
        value = _random_value_for(record, genome, var_name, node, record['case_study'], rng)
        if value is None:
            return candidate, focal_maps, scenario_maps
    except FitnessEvaluationError:
        return candidate, focal_maps, scenario_maps

    new_candidate, new_focal_maps, new_scenario_maps = copy.deepcopy((candidate, focal_maps, scenario_maps))
    new_focal = new_focal_maps[rid]
    new_scenario = new_scenario_maps.setdefault(rid, {})
    try:
        apply_mutation(record, new_candidate, new_focal, new_scenario, var_name, node, value,
                        genome.get(var_name), owner_id=rid)
    except FitnessEvaluationError:
        return candidate, focal_maps, scenario_maps
    return new_candidate, new_focal_maps, new_scenario_maps


def run_random_search(records, case_study, population_size=20, generations=50, rng=None,
                       mutations_per_child='auto', dynamic_gating=True, max_evaluations=None, trace=None):
    if generations is None and max_evaluations is None:
        raise ValueError('run_random_search needs a generation cap, an evaluation budget, or both')
    state = {}
    if max_evaluations is not None:
        reset_evaluation_counter(max_evaluations)
    try:
        _random_search_generations(records, case_study, population_size, generations, rng,
                                   mutations_per_child, dynamic_gating, trace,
                                   max_evaluations is not None, state)
    except EvaluationBudgetExhausted:
        pass
    finally:
        if max_evaluations is not None:
            clear_evaluation_limit()
    return state['archive'], state['coverage_history'], state['population']


def _random_search_generations(records, case_study, population_size, generations, rng,
                               mutations_per_child, dynamic_gating, trace, stop_when_all_covered, state):
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
                archive[rid] = (f, individual)

    for ind in population:
        update_archive(ind)

    gen = 0
    while generations is None or gen < generations:
        gen += 1
        if stop_when_all_covered and all(archive[r['record_id']][0] == 0.0 for r in records):
            break
        covered_keys = {_branch_key(r) for r in records
                         if archive.get(r['record_id'], (float('inf'), None))[0] == 0.0}
        active = records if not dynamic_gating else [r for r in records if is_active(r, covered_keys)]
        k = _mutations_per_child(active, population_size, mutations_per_child)

        new_population = []
        for lineage in population:
            cand, fm, sm = lineage
            if active:
                for r in rng.sample(active, min(k, len(active))):
                    for _ in range(_local_burst_size(r)):
                        cand, fm, sm = _random_mutate_lineage(r, cand, fm, sm, table_cache, rng)
            new_population.append((cand, fm, sm))
        population = new_population
        state['population'] = population

        for ind in population:
            update_archive(ind)

        covered_count = sum(1 for r in records
                             if archive.get(r['record_id'], (float('inf'), None))[0] == 0.0)
        coverage_history.append(covered_count)
