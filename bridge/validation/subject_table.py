"""Subject table of a decision: the table whose rows the decision is evaluated for, and the join paths to the tables its inputs read."""
import re

_PLACEHOLDER_NAME_RE = re.compile(r'<([A-Za-z_][A-Za-z0-9_ ]*)>')
_COLON_OR_SELF_REF_RE = re.compile(r'(?<!:):(?!:)[A-Za-z_][A-Za-z0-9_]*|\bself\b')


def _needs_self_table(filter_text):
    return bool(filter_text) and bool(_COLON_OR_SELF_REF_RE.search(filter_text))


def _placeholder_source_tables(case_study, filter_text):
    from bridge.validation.filter_placeholder_sources import get_source_table
    tables = set()
    for name in _PLACEHOLDER_NAME_RE.findall(filter_text or ''):
        table = get_source_table(case_study, name)
        if table:
            tables.add(table)
    return tables


_TABLE_EXTRACTORS = {
    'schema_column': lambda n, cs: {n['table']},
    'serialized_field': lambda n, cs: {n['table']},
    'null_check': lambda n, cs: {n['table']},
    'derived_case': lambda n, cs: {n['table']},
    'derived_aggregate': lambda n, cs: (_placeholder_source_tables(cs, n.get('filter_text'))
                                         | ({n.get('self_table') or n['table']}
                                            if _needs_self_table(n.get('filter_text')) else set())),
    'exists': lambda n, cs: (
        (_placeholder_source_tables(cs, n.get('filter_text'))
         | (set(n['candidate_tables']) if _needs_self_table(n.get('filter_text')) else set()))
        if n.get('filter_text') else set(n['candidate_tables'])),
    'raw_sql_boolean': lambda n, cs: _placeholder_source_tables(cs, n.get('sql_template')),
    'any_not_null': lambda n, cs: {c['table'] for c in n['columns']},
    'join_lookup': lambda n, cs: {n['via']['local_table'], n['result_table']},
    'join_null_check': lambda n, cs: {n['via']['local_table'], n['result_table']},
    'regex_match': lambda n, cs: {n['value_column']['table'], n['pattern_column']['table']},
}

_NON_TABLE_KINDS = {
    'literal', 'literal_via_upstream_branch',
    'not_persisted', 'code_external', 'schema_gap', 'unresolved', 'variable',
}


_OWN_TABLE_HINTS = {
    'derived_aggregate': lambda n: {n['table']},
    'exists': lambda n: set(n.get('candidate_tables') or []),
    'raw_sql_boolean': lambda n: set(n.get('tables') or []),
}


def _own_table_hints(node):
    kind = node.get('kind')
    if kind in _OWN_TABLE_HINTS:
        return _OWN_TABLE_HINTS[kind](node)
    if kind == 'substituted_decision':
        tables = set()
        for sub_node in node.get('free_variable_resolutions', {}).values():
            tables |= _own_table_hints(sub_node)
        return tables
    return set()


def tables_referenced(node, case_study=None):
    kind = node.get('kind')
    if kind in _TABLE_EXTRACTORS:
        return _TABLE_EXTRACTORS[kind](node, case_study)
    if kind in _NON_TABLE_KINDS:
        return set()
    if kind == 'substituted_decision':
        tables = set()
        for sub_node in node.get('free_variable_resolutions', {}).values():
            tables |= tables_referenced(sub_node, case_study)
        return tables
    raise ValueError(f"Unrecognized variable_resolution kind {kind!r} -- "
                      f"fail loudly rather than silently skip (node={node!r})")


def _pick_root(case_study, all_tables, closure_tables, decision_name=None):
    from bridge.validation.schema_utility import build_join_path
    from bridge.validation.subject_root_overrides import get_override

    def _reaches(root, t):
        try:
            return build_join_path(case_study, root, t, closure_tables) is not None
        except ValueError:
            return False

    candidate_pool = closure_tables | all_tables
    candidates = [root for root in candidate_pool
                  if all(_reaches(root, t) for t in all_tables - {root})]

    if len(candidates) != 1:
        override = get_override(case_study, decision_name) if decision_name else None
        if override is not None:
            if override not in candidates:
                raise ValueError(
                    f"subject_root_overrides.py names {override!r} as the root for "
                    f"{decision_name!r}, which is not among the actual qualifying "
                    f"candidates ({sorted(candidates)}) -- the override is stale, fix "
                    f"it rather than silently ignoring it.")
            return override
        raise ValueError(
            f"Cannot pick a unique root reaching {sorted(all_tables)} among the closure: "
            f"{len(candidates)} candidate(s) qualify ({sorted(candidates)}) -- "
            f"need exactly 1. Refusing to guess.")
    return candidates[0]


def subject_table_for_decision(records, case_study):
    from bridge.validation.schema_utility import build_join_path, canonical_table_name, pk_columns

    case_study = records[0]['case_study']

    all_tables = set()
    closure_tables = set()
    for r in records:
        closure_tables |= {canonical_table_name(case_study, t)
                            for t in r.get('fk_closure_tables', [])}
        for node in r.get('variable_resolution', {}).values():
            all_tables |= {canonical_table_name(case_study, t)
                           for t in tables_referenced(node, case_study)}

    if len(all_tables) == 0:
        own_tables = set()
        for r in records:
            for node in r.get('variable_resolution', {}).values():
                own_tables |= {canonical_table_name(case_study, t) for t in _own_table_hints(node)}
        if not own_tables:
            raise ValueError(
                f"No table-backed inputs found for decision {records[0]['decision_name']!r} -- "
                f"every input is non-table-backed (literal/upstream/not_persisted); "
                f"this decision cannot be independently entity-enumerated from the database alone.")
        all_tables = own_tables

    if len(all_tables) == 1:
        subject = next(iter(all_tables))
        return subject, pk_columns(case_study, subject), {}

    root = _pick_root(case_study, all_tables, closure_tables, records[0]['decision_name'])
    join_paths = {}
    for t in all_tables:
        if t == root:
            continue
        path = build_join_path(case_study, root, t, closure_tables)
        if path is None:
            raise ValueError(f"No FK join path found from root {root!r} to {t!r} "
                              f"within closure {sorted(closure_tables)} -- refusing to guess.")
        join_paths[t] = path
    return root, pk_columns(case_study, root), join_paths


def build_subject_tables(case_study, decisions_by_name):
    resolved = {}
    unresolved = {}
    for name, records in decisions_by_name.items():
        if not records:
            unresolved[name] = (
                "Every rule in this decision is declared out of scope "
                "(out_of_scope_rules.py) -- no in-scope input to determine a subject from.")
            continue
        try:
            resolved[name] = subject_table_for_decision(records, case_study)
        except ValueError as e:
            unresolved[name] = str(e)
    return resolved, unresolved
