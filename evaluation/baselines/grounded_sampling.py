"""GS (grounded sampling), an unguided variant of BRIDGE.

Every sample starts from the seed individual and assigns a random value to
every grounded input of every objective. An archive keeps the best sample
found for every objective."""
import copy
import random


from bridge.search.candidate import derive_genome
from bridge.search.fitness import FitnessEvaluationError, EvaluationBudgetExhausted, reset_evaluation_counter, clear_evaluation_limit, evaluations_used
from bridge.search.mutation import _leaf_variables, candidate_values, apply_mutation, BOOLEAN_LEAF_KINDS
from bridge.search.dynamosa import _seed_shared_population, _focal_for_mutate, evaluate_objective

_MAX_STEP = 40


def _sample_value_for(record, var_name, node, current, case_study, rng):
    if node.get('kind') in BOOLEAN_LEAF_KINDS:
        return rng.choice([True, False])
    options = candidate_values(record, var_name, node, current, case_study,
                               step=rng.randint(1, _MAX_STEP))
    return rng.choice(options) if options else None


def _random_individual(base, records, case_study, table_cache, rng):
    candidate, focal_maps, scenario_maps = copy.deepcopy(base)
    order = list(records)
    rng.shuffle(order)
    for r in order:
        rid = r['record_id']
        scenario = scenario_maps.setdefault(rid, {})
        focal = _focal_for_mutate(r, candidate, focal_maps, table_cache)
        try:
            genome = derive_genome(r, candidate, focal, scenario, owner_id=rid)
        except FitnessEvaluationError:
            continue
        for var_name, node in _leaf_variables(r):
            if var_name not in genome:
                continue
            value = _sample_value_for(r, var_name, node, genome.get(var_name), case_study, rng)
            if value is None:
                continue
            try:
                apply_mutation(r, candidate, focal_maps[rid], scenario, var_name, node, value,
                               genome.get(var_name), owner_id=rid)
            except FitnessEvaluationError:
                continue
    return candidate, focal_maps, scenario_maps


def run_random_sampling(records, case_study, max_evaluations, rng=None, trace=None, max_samples=None):
    rng = rng or random.Random(0)
    table_cache = {}
    base = _seed_shared_population(records, case_study, 1)[0]
    archive = {r['record_id']: (float('inf'), base) for r in records}
    history = []

    def update_archive(individual):
        candidate, focal_maps, scenario_maps = individual
        for r in records:
            rid = r['record_id']
            f = evaluate_objective(r, candidate, focal_maps, scenario_maps, table_cache)
            if f < archive[rid][0]:
                if f == 0.0 and trace is not None:
                    trace.append((evaluations_used(), rid))
                archive[rid] = (f, individual)

    def covered():
        return sum(1 for r in records if archive[r['record_id']][0] == 0.0)

    reset_evaluation_counter(max_evaluations)
    try:
        update_archive(base)
        history.append(covered())
        while (max_samples is None or len(history) <= max_samples) and history[-1] < len(records):
            update_archive(_random_individual(base, records, case_study, table_cache, rng))
            history.append(covered())
    except EvaluationBudgetExhausted:
        pass
    finally:
        clear_evaluation_limit()
    return archive, history, []
