"""Fitness of an individual for one objective.

The fitness combines the normalized branch distance to satisfy the rule's
condition with the distance to falsify the rules that precede it under the
FIRST hit policy. Every call counts as one fitness evaluation, the unit of
the search budget."""
import json


K = 1.0


class FitnessEvaluationError(Exception):
    pass


class EvaluationBudgetExhausted(Exception):
    pass


_EVALUATIONS = {'count': 0, 'limit': None}


def reset_evaluation_counter(limit=None):
    _EVALUATIONS['count'] = 0
    _EVALUATIONS['limit'] = limit


def clear_evaluation_limit():
    _EVALUATIONS['limit'] = None


def evaluations_used():
    return _EVALUATIONS['count']


def evaluate_resolution(var_name, node, genome):
    kind = node.get('kind')
    if kind == 'literal':
        return node['value']
    if kind == 'literal_via_upstream_branch':
        return evaluate_expression(node['value'], {}, genome)
    if kind == 'substituted_decision':
        return evaluate_expression(node['expression'], node.get('free_variable_resolutions', {}), genome)
    if kind in ('schema_gap', 'code_external', 'unresolved', 'chained_decision_output'):
        raise FitnessEvaluationError(
            f"{var_name!r} has no legitimate value to evaluate -- its resolution kind "
            f"is {kind!r}, a structurally unresolvable fact, not a gene")
    if var_name not in genome:
        raise FitnessEvaluationError(
            f"genome is missing required gene {var_name!r} (resolution kind={kind!r})")
    return genome[var_name]


def _safe_div(a, b):
    if b == 0:
        raise FitnessEvaluationError(
            f"division by zero evaluating a DMN arithmetic expression ({a!r} / {b!r}) -- "
            f"the search should never let this gene reach 0")
    return a / b


_ARITH = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
          '*': lambda a, b: a * b, '/': _safe_div}


def evaluate_expression(node, resolution_map, genome):
    if not isinstance(node, dict):
        raise FitnessEvaluationError(f"not a valid expression node: {node!r}")
    kind = node.get('kind')
    if kind == 'literal':
        return node['value']
    if kind == 'variable':
        ref = node['ref']
        if ref not in resolution_map:
            raise FitnessEvaluationError(
                f"expression references {ref!r} with no resolution recorded for it")
        return evaluate_resolution(ref, resolution_map[ref], genome)
    if kind == 'call':
        name = node.get('name')
        if name == 'today' and not node.get('args'):
            if '__today__' not in genome:
                raise FitnessEvaluationError(
                    "genome is missing the scenario-level '__today__' gene FEEL's today() needs")
            return genome['__today__']
        raise FitnessEvaluationError(
            f"FEEL built-in call {name!r} has no fitness-evaluator "
            f"meaning yet -- not silently treated as zero")
    if kind == 'opaque_formula':
        raise FitnessEvaluationError(
            f"opaque (unparsed) FEEL formula can't be evaluated: {node.get('feel_text', '')[:80]!r}")
    op = node.get('op')
    if op in _ARITH:
        a = evaluate_expression(node['left'], resolution_map, genome)
        b = evaluate_expression(node['right'], resolution_map, genome)
        if not (_numeric(a) and _numeric(b)):
            raise FitnessEvaluationError(
                f"arithmetic operator {op!r} needs two numeric operands, got {a!r} and {b!r} -- "
                f"no principled arithmetic on a nullable value that's currently unset")
        return _ARITH[op](a, b)
    if op == 'if':
        cond_true = distance_to_true(node['cond'], resolution_map, genome) == 0
        branch = node['then'] if cond_true else node['else']
        return evaluate_expression(branch, resolution_map, genome)
    if op is not None:
        return distance_to_true(node, resolution_map, genome) == 0
    raise FitnessEvaluationError(f"expression node with unhandled shape: {node!r}")


def _leaf_value(node, resolution_map, genome):
    return evaluate_expression(node, resolution_map, genome)


_NEGATED_OP = {'=': '!=', '!=': '=', '<': '>=', '>=': '<', '<=': '>', '>': '<='}


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _comparison_distance_true(op, a, b):
    if op == '=':
        return abs(a - b) if _numeric(a) and _numeric(b) else (0.0 if a == b else K)
    if op == '!=':
        return K if a == b else 0.0
    if not (_numeric(a) and _numeric(b)):
        raise FitnessEvaluationError(
            f"ordering comparison {op!r} needs two numeric operands, got {a!r} and {b!r} -- "
            f"no principled distance for an ordering fact between non-numeric values "
            f"(e.g. a nullable date/count that is currently unset)")
    if op == '<':
        return (a - b) + K if a >= b else 0.0
    if op == '<=':
        return (a - b) + K if a > b else 0.0
    if op == '>':
        return (b - a) + K if a <= b else 0.0
    if op == '>=':
        return (b - a) + K if a < b else 0.0
    raise FitnessEvaluationError(f"unhandled comparison operator {op!r}")


def _min_distance(terms):
    best = None
    error = None
    for term in terms:
        try:
            d = term()
        except FitnessEvaluationError as e:
            error = error if error is not None else e
            continue
        if best is None or d < best:
            best = d
        if best == 0.0:
            return 0.0
    if best is not None:
        return best
    raise error


def distance_to_true(node, resolution_map, genome):
    if not isinstance(node, dict):
        raise FitnessEvaluationError(f"not a valid condition node: {node!r}")
    kind = node.get('kind')
    if kind == 'literal':
        return 0.0 if node['value'] else K
    if kind == 'variable':
        return 0.0 if evaluate_resolution(node['ref'], resolution_map.get(node['ref'], node), genome) else K
    op = node.get('op')
    if op in _NEGATED_OP:
        a = _leaf_value(node['left'], resolution_map, genome)
        b = _leaf_value(node['right'], resolution_map, genome)
        return _comparison_distance_true(op, a, b)
    if op == 'and':
        return sum(distance_to_true(c, resolution_map, genome) for c in node['clauses'])
    if op == 'or':
        return _min_distance(lambda c=c: distance_to_true(c, resolution_map, genome) for c in node['clauses'])
    if op == 'not':
        return distance_to_false(node['clause'], resolution_map, genome)
    if op == 'in':
        left_val = _leaf_value(node['left'], resolution_map, genome)
        return _min_distance(
            lambda v=v: _comparison_distance_true('=', left_val, _leaf_value(v, resolution_map, genome))
            for v in node['values'])
    if op == 'between':
        low = _leaf_value(node['low'], resolution_map, genome)
        high = _leaf_value(node['high'], resolution_map, genome)
        left = _leaf_value(node['left'], resolution_map, genome)
        return (_comparison_distance_true('>=', left, low)
                + _comparison_distance_true('<=', left, high))
    raise FitnessEvaluationError(f"unhandled condition node: {node!r}")


def distance_to_false(node, resolution_map, genome):
    if not isinstance(node, dict):
        raise FitnessEvaluationError(f"not a valid condition node: {node!r}")
    kind = node.get('kind')
    if kind == 'literal':
        return K if node['value'] else 0.0
    if kind == 'variable':
        return K if evaluate_resolution(node['ref'], resolution_map.get(node['ref'], node), genome) else 0.0
    op = node.get('op')
    if op in _NEGATED_OP:
        a = _leaf_value(node['left'], resolution_map, genome)
        b = _leaf_value(node['right'], resolution_map, genome)
        return _comparison_distance_true(_NEGATED_OP[op], a, b)
    if op == 'and':
        return _min_distance(lambda c=c: distance_to_false(c, resolution_map, genome) for c in node['clauses'])
    if op == 'or':
        return sum(distance_to_false(c, resolution_map, genome) for c in node['clauses'])
    if op == 'not':
        return distance_to_true(node['clause'], resolution_map, genome)
    if op == 'in':
        left = node['left']
        return sum(_comparison_distance_true('!=', _leaf_value(left, resolution_map, genome),
                                              _leaf_value(v, resolution_map, genome))
                   for v in node['values'])
    if op == 'between':
        low = _leaf_value(node['low'], resolution_map, genome)
        high = _leaf_value(node['high'], resolution_map, genome)
        left = _leaf_value(node['left'], resolution_map, genome)
        return _min_distance([lambda: _comparison_distance_true('<', left, low),
                               lambda: _comparison_distance_true('>', left, high)])
    raise FitnessEvaluationError(f"unhandled condition node: {node!r}")


def normalize(d):
    return d / (d + 1.0)


def branch_fitness(record, genome):
    if _EVALUATIONS['limit'] is not None and _EVALUATIONS['count'] >= _EVALUATIONS['limit']:
        raise EvaluationBudgetExhausted(_EVALUATIONS['limit'])
    _EVALUATIONS['count'] += 1
    resolution_map = record.get('variable_resolution', {})
    own = distance_to_true(record['condition'], resolution_map, genome)
    own_conjuncts = {_canonical(c) for c in _top_level_conjuncts(record['condition'])}
    suppression = sum(
        normalize(distance_to_false(_without_shared_conjuncts(row['condition'], own_conjuncts),
                                    resolution_map, genome))
        for row in record.get('hit_policy_context', {}).get('earlier_rows', [])
    )
    return normalize(own) + suppression


def _top_level_conjuncts(node):
    if isinstance(node, dict) and node.get('op') == 'and':
        out = []
        for c in node['clauses']:
            out.extend(_top_level_conjuncts(c))
        return out
    return [node]


def _canonical(node):
    import json
    return json.dumps(node, sort_keys=True)


def _without_shared_conjuncts(earlier_condition, own_conjuncts):
    conjuncts = _top_level_conjuncts(earlier_condition)
    rest = [c for c in conjuncts if _canonical(c) not in own_conjuncts]
    if len(rest) == len(conjuncts) or not rest:
        return earlier_condition
    return rest[0] if len(rest) == 1 else {'op': 'and', 'clauses': rest}


def _unique_key_sets(schema, table):
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    keys = []
    pk = info.get('pk')
    if pk:
        keys.append(pk if isinstance(pk, list) else [pk])
    for idx in info.get('indexes', []):
        if idx.get('unique'):
            keys.append(idx['cols'])
    return keys


