"""Executes a DMN decision, including the decisions it requires, against a database and reports the rule selected for every subject row."""

from bridge.validation.db_resolver import resolve, UnresolvableForCase
from bridge.validation.rule_evaluator import evaluate_condition, evaluate_expression, UniqueViolation


def _condition_variable_refs(node):
    refs = set()

    def walk(n):
        if isinstance(n, dict):
            if n.get('kind') == 'variable' and 'ref' in n:
                refs.add(n['ref'])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for item in n:
                walk(item)

    walk(node)
    return refs


def _distinct_subject_keys(conn, subject_table, pk_cols):
    cols_sql = ', '.join(f'"{c}"' for c in pk_cols)
    cur = conn.execute(f'SELECT DISTINCT {cols_sql} FROM "{subject_table}"')
    return [row for row in cur.fetchall()]


def _rule_outputs(records):
    outputs_by_rule = {}
    for r in records:
        if r['rule_id'] in outputs_by_rule:
            continue
        values = {}
        for name, node in r.get('outputs', {}).items():
            if node.get('kind') != 'literal':
                raise NotImplementedError(
                    f"Rule {r['rule_id']!r} output {name!r} is not a literal "
                    f"(kind={node.get('kind')!r}) -- not yet supported")
            values[name] = node['value']
        outputs_by_rule[r['rule_id']] = values
    return outputs_by_rule


class UngroundedForCase(Exception):
    pass


class DecisionRunner:
    def __init__(self, conn, case_study, records_by_decision, subject_tables,
                 not_persisted_overrides=None):
        self.conn = conn
        self.case_study = case_study
        self.records_by_decision = records_by_decision
        self.subject_tables = subject_tables
        self.not_persisted_overrides = not_persisted_overrides
        self._cache = {}

    def run(self, decision_name, subject_pk_vals):
        key = (decision_name, tuple(subject_pk_vals))
        if key not in self._cache:
            records = self.records_by_decision[decision_name]
            subject_table, pk_cols, join_paths = self.subject_tables[decision_name]
            self._cache[key] = run_decision(
                self.conn, decision_name, records, subject_table, pk_cols,
                join_paths=join_paths, runner=self, only_case=subject_pk_vals,
                not_persisted_overrides=self.not_persisted_overrides)
        return self._cache[key]

    def upstream_subject_value(self, upstream_decision_name, downstream_subject_table,
                                downstream_pk_cols, downstream_pk_vals):
        from bridge.validation.schema_utility import build_join_path
        from bridge.validation.db_resolver import _row_for_table

        upstream_table, upstream_pk_cols, _upstream_joins = self.subject_tables[upstream_decision_name]
        if upstream_table == downstream_subject_table:
            return tuple(downstream_pk_vals)

        closure = set()
        for recs in self.records_by_decision.values():
            for r in recs:
                closure |= set(r.get('fk_closure_tables', []))
        path = build_join_path(self.case_study, downstream_subject_table, upstream_table, closure)
        if path is None:
            raise NotImplementedError(
                f"No join path from {downstream_subject_table!r} to upstream decision "
                f"{upstream_decision_name!r}'s own subject table {upstream_table!r}")

        row = _row_for_table(self.conn, downstream_subject_table, downstream_subject_table,
                              downstream_pk_cols, downstream_pk_vals, {})
        for hop in path:
            if row is None:
                return None
            if 'from_columns' in hop:
                fk_values = [row.get(c.lower()) for c in hop['from_columns']]
                row = _row_for_table(self.conn, hop['to_table'], hop['to_table'],
                                      hop['to_columns'], fk_values, {}) \
                    if all(v is not None for v in fk_values) else None
            else:
                fk_value = row.get(hop['from_column'].lower())
                row = _row_for_table(self.conn, hop['to_table'], hop['to_table'],
                                      [hop['to_column']], [fk_value], {}) if fk_value is not None else None
        if row is None:
            return None
        return tuple(row[c.lower()] for c in upstream_pk_cols)


def _resolve_one(conn, case_study, var, node, subject_table, subject_pk_cols, subject_pk_vals,
                  join_paths, runner, decision_name, trace=None, not_persisted_overrides=None):
    if node.get('kind') == 'not_persisted':
        overrides = not_persisted_overrides or {}
        if var not in overrides:
            raise NotImplementedError(
                f"not_persisted variable {var!r} has no declared override -- pass "
                f"not_persisted_overrides={{{var!r}: <value>}} explicitly, disclosed, "
                f"never read from search state (see DESIGN.md known gaps)")
        result = resolve(conn, node, subject_table, subject_pk_cols, subject_pk_vals,
                          join_paths, declared_not_persisted_value=overrides[var], case_study=case_study)
        if trace is not None:
            trace[var] = {'value': result.value, 'resolution_type': result.resolution_type,
                          'source_table': None}
        return result.value

    if node.get('kind') == 'literal_via_upstream_branch':
        upstream_decision = node['from_decision']
        upstream_pk_vals = runner.upstream_subject_value(
            upstream_decision, subject_table, subject_pk_cols, subject_pk_vals)
        if upstream_pk_vals is None:
            raise UngroundedForCase(f"{var}: no corresponding {upstream_decision!r} row")
        upstream_result = runner.run(upstream_decision, upstream_pk_vals)
        actual_selected = upstream_result['selected_by_case'].get(tuple(upstream_pk_vals))
        if actual_selected != node['from_rule_id']:
            raise UngroundedForCase(
                f"{var}: upstream {upstream_decision!r} selected {actual_selected!r}, "
                f"not the required {node['from_rule_id']!r}")
        try:
            result = resolve(conn, node['value'], subject_table, subject_pk_cols, subject_pk_vals, join_paths,
                              case_study=case_study)
        except UnresolvableForCase as e:
            raise UngroundedForCase(f"{var}: {e}")
        if trace is not None:
            trace[var] = {'value': result.value, 'resolution_type': 'literal_via_upstream_branch',
                          'source_table': None, 'upstream_decision': upstream_decision,
                          'upstream_selected_rule': actual_selected}
        return result.value

    if node.get('kind') == 'substituted_decision':
        free_values = {}
        for free_var, free_node in node['free_variable_resolutions'].items():
            free_values[free_var] = _resolve_one(
                conn, case_study, free_var, free_node, subject_table, subject_pk_cols,
                subject_pk_vals, join_paths, runner, decision_name, trace, not_persisted_overrides)
        if not_persisted_overrides and '__today__' in not_persisted_overrides:
            free_values = {**free_values, '__today__': not_persisted_overrides['__today__']}
        value = evaluate_expression(node['expression'], free_values)
        if trace is not None:
            trace[var] = {'value': value, 'resolution_type': 'substituted_decision',
                          'source_table': None, 'free_variables': free_values}
        return value

    try:
        result = resolve(conn, node, subject_table, subject_pk_cols, subject_pk_vals, join_paths,
                          case_study=case_study)
    except UnresolvableForCase as e:
        raise UngroundedForCase(f"{var}: {e}")
    if trace is not None:
        trace[var] = {'value': result.value, 'resolution_type': result.resolution_type,
                      'source_table': result.source_table}
    return result.value


def run_decision(conn, decision_name, records, subject_table, subject_pk_cols,
                  join_paths=None, runner=None, only_case=None, collect_trace=False,
                  not_persisted_overrides=None):
    case_study = records[0]['case_study']
    rule_outputs = _rule_outputs(records) if collect_trace else None
    hit_policy = records[0]['hit_policy']

    variants_by_rule_id = {}
    for r in records:
        variants_by_rule_id.setdefault(r['rule_id'], []).append(r)
    ordered_rule_ids = list(variants_by_rule_id.keys())

    subject_keys = [tuple(only_case)] if only_case is not None else \
        _distinct_subject_keys(conn, subject_table, subject_pk_cols)

    matched_by_case = {}
    selected_by_case = {}
    verified_covered = set()
    violations = []
    trace_by_case = {} if collect_trace else None

    for pk_vals in subject_keys:
        base_values = {}
        if not_persisted_overrides and '__today__' in not_persisted_overrides:
            base_values['__today__'] = not_persisted_overrides['__today__']

        matched = []
        overall_var_trace = {} if collect_trace else None
        any_variant_grounded = False

        for rule_id in ordered_rule_ids:
            rule_matched = False
            for variant in variants_by_rule_id[rule_id]:
                values = dict(base_values)
                var_trace = {} if collect_trace else None
                ungrounded = False
                needed_vars = _condition_variable_refs(variant['condition'])
                for var, node in variant['variable_resolution'].items():
                    if var not in needed_vars:
                        continue
                    try:
                        values[var] = _resolve_one(conn, case_study, var, node, subject_table,
                                                    subject_pk_cols, pk_vals, join_paths or {},
                                                    runner, decision_name, var_trace, not_persisted_overrides)
                    except UngroundedForCase:
                        ungrounded = True
                        break
                if ungrounded:
                    continue
                any_variant_grounded = True
                if collect_trace:
                    overall_var_trace.update(var_trace)
                if evaluate_condition(variant['condition'], values):
                    rule_matched = True
                    break
            if rule_matched:
                matched.append(rule_id)

        if hit_policy == 'FIRST':
            selected = matched[0] if matched else None
        elif hit_policy == 'UNIQUE':
            if len(matched) > 1:
                violations.append((pk_vals, UniqueViolation(decision_name, matched)))
                continue
            selected = matched[0] if matched else None
        else:
            raise NotImplementedError(f"Hit policy {hit_policy!r} not yet supported "
                                       f"(only FIRST and UNIQUE are supported)")

        matched_by_case[pk_vals] = matched
        selected_by_case[pk_vals] = selected
        if selected:
            verified_covered.add(selected)
        if collect_trace:
            trace_by_case[pk_vals] = {
                'resolved_inputs': overall_var_trace, 'matched_rule_ids': matched,
                'selected_rule_id': selected, 'ungrounded': not any_variant_grounded,
                'decision_output': rule_outputs.get(selected) if selected else None,
            }

    result = {
        'matched_by_case': matched_by_case,
        'selected_by_case': selected_by_case,
        'verified_covered_rule_ids': verified_covered,
        'unique_violations': violations,
    }
    if collect_trace:
        result['trace'] = trace_by_case
    return result
