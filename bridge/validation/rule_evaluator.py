"""Evaluates rule conditions on resolved input values and selects the rule under the FIRST or UNIQUE hit policy."""
def _eval_operand(node, values):
    kind = node.get('kind')
    if kind == 'variable':
        if node['ref'] not in values:
            raise KeyError(f"condition references {node['ref']!r}, not in resolved values "
                            f"{sorted(values)}")
        return values[node['ref']]
    if kind == 'literal':
        return node['value']
    if kind == 'call' and node.get('name') == 'today' and not node.get('args'):
        if '__today__' not in values:
            raise NotImplementedError(
                "condition calls today(), but '__today__' was not supplied in resolved "
                "values -- pass it via not_persisted_overrides={'__today__': <value>}")
        return values['__today__']
    if 'op' in node:
        return evaluate_expression(node, values)
    raise NotImplementedError(f"Unhandled operand kind {kind!r}: {node!r}")


_ARITHMETIC = {
    '+': lambda a, b: a + b if a is not None and b is not None else None,
    '-': lambda a, b: a - b if a is not None and b is not None else None,
    '*': lambda a, b: a * b if a is not None and b is not None else None,
    '/': lambda a, b: a / b if (a is not None and b) else None,
}


def evaluate_expression(expr, values):
    if 'op' not in expr:
        return _eval_operand(expr, values)
    op = expr['op']
    if op in _ARITHMETIC:
        left = _eval_operand(expr['left'], values)
        right = _eval_operand(expr['right'], values)
        return _ARITHMETIC[op](left, right)
    raise NotImplementedError(f"Unhandled expression operator {op!r}: {expr!r}")


def _ordered_compare(op, a, b):
    if a is None or b is None:
        return False
    try:
        return op(a, b)
    except TypeError:
        return False


_COMPARATORS = {
    '=': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
    '>': lambda a, b: _ordered_compare(lambda x, y: x > y, a, b),
    '>=': lambda a, b: _ordered_compare(lambda x, y: x >= y, a, b),
    '<': lambda a, b: _ordered_compare(lambda x, y: x < y, a, b),
    '<=': lambda a, b: _ordered_compare(lambda x, y: x <= y, a, b),
}


def evaluate_condition(condition, values):
    if 'op' not in condition:
        value = _eval_operand(condition, values)
        if not isinstance(value, bool):
            raise NotImplementedError(
                f"A condition with no 'op' must be a boolean literal, got {value!r}: {condition!r}")
        return value
    op = condition.get('op')
    if op == 'and':
        return all(evaluate_condition(c, values) for c in condition['clauses'])
    if op == 'or':
        return any(evaluate_condition(c, values) for c in condition['clauses'])
    if op in _COMPARATORS:
        left = _eval_operand(condition['left'], values)
        right = _eval_operand(condition['right'], values)
        return _COMPARATORS[op](left, right)
    if op == 'in':
        left = _eval_operand(condition['left'], values)
        candidates = [_eval_operand(v, values) for v in condition['values']]
        return left in candidates
    if op == 'not':
        return not evaluate_condition(condition['clause'], values)
    if op == 'between':
        left = _eval_operand(condition['left'], values)
        low = _eval_operand(condition['low'], values)
        high = _eval_operand(condition['high'], values)
        return _COMPARATORS['>='](left, low) and _COMPARATORS['<='](left, high)
    raise NotImplementedError(f"Unhandled condition operator {op!r}: {condition!r}")


class UniqueViolation(Exception):
    def __init__(self, decision_name, matched_rule_ids):
        self.decision_name = decision_name
        self.matched_rule_ids = matched_rule_ids
        super().__init__(f"UNIQUE violation in {decision_name!r}: "
                          f"{len(matched_rule_ids)} rules matched: {matched_rule_ids}")


