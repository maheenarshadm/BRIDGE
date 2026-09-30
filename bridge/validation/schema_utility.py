"""Schema access for the validator: primary keys, foreign keys and join paths."""
import json

from bridge import paths

SCHEMA_JSON_PATHS = paths.SCHEMA_JSON_PATHS

_CACHE = {}


def load_schema(case_study):
    if case_study not in _CACHE:
        with open(SCHEMA_JSON_PATHS[case_study], encoding='utf-8') as f:
            _CACHE[case_study] = json.load(f)
    return _CACHE[case_study]


def _table_entry(schema, table):
    return schema.get(table) or schema.get(table.upper()) or schema.get(table.lower())


def canonical_table_name(case_study, table):
    schema = load_schema(case_study)
    for candidate in (table, table.upper(), table.lower()):
        if candidate in schema:
            return candidate
    return table


def pk_columns(case_study, table):
    entry = _table_entry(load_schema(case_study), table)
    pk = (entry or {}).get('pk')
    if pk is None:
        return []
    return pk if isinstance(pk, list) else [pk]


def fk_edges(case_study, table):
    from bridge.validation.supplementary_fk_edges import get_supplementary_edges
    entry = _table_entry(load_schema(case_study), table)
    edges = list((entry or {}).get('fk_columns', [])) + get_supplementary_edges(case_study, table)
    return [{**e, 'ref_table': canonical_table_name(case_study, e['ref_table'])} for e in edges]


def functional_backward_edges(case_study, table):
    schema = load_schema(case_study)
    target = canonical_table_name(case_study, table)
    found = []
    for other_name, info in schema.items():
        other_pk = info.get('pk')
        other_pk_list = other_pk if isinstance(other_pk, list) else ([other_pk] if other_pk else [])
        for fk in info.get('fk_columns', []):
            if canonical_table_name(case_study, fk['ref_table']) == target and other_pk_list == [fk['column']]:
                found.append({'column': fk['ref_column'], 'ref_table': other_name,
                              'ref_column': fk['column']})
    return found


def composite_backward_edges(case_study, table):
    schema = load_schema(case_study)
    table = canonical_table_name(case_study, table)
    source_targets = {(e['ref_table'], e['ref_column']): e['column']
                       for e in fk_edges(case_study, table)}

    found = []
    for other_name in schema:
        other_name = canonical_table_name(case_study, other_name)
        if other_name == table:
            continue
        pk = pk_columns(case_study, other_name)
        if len(pk) < 2:
            continue
        other_fk_target_by_column = {e['column']: (e['ref_table'], e['ref_column'])
                                      for e in fk_edges(case_study, other_name)}
        matched_columns = []
        for pk_col in pk:
            target = other_fk_target_by_column.get(pk_col)
            source_column = source_targets.get(target) if target else None
            if source_column is None:
                matched_columns = None
                break
            matched_columns.append(source_column)
        if matched_columns and len(set(matched_columns)) == len(matched_columns):
            found.append({'columns': matched_columns, 'ref_table': other_name, 'ref_columns': pk})
    return found


def build_join_path(case_study, root_table, target_table, allowed_tables):
    root_table = canonical_table_name(case_study, root_table)
    target_table = canonical_table_name(case_study, target_table)
    if root_table == target_table:
        return []

    allowed = {canonical_table_name(case_study, t) for t in allowed_tables} | {root_table, target_table}
    visited = {root_table}
    queue = [(root_table, [])]
    skipped_ambiguous = []

    while queue:
        current, path = queue.pop(0)
        edges = fk_edges(case_study, current) + functional_backward_edges(case_study, current)

        by_target = {}
        for edge in edges:
            nxt = edge['ref_table']
            if nxt in allowed and nxt not in visited:
                by_target.setdefault(nxt, []).append(edge)

        for nxt, candidate_edges in by_target.items():
            distinct_columns = {e['column'] for e in candidate_edges}
            if len(distinct_columns) > 1:
                from bridge.validation.join_disambiguation import get_override
                override = get_override(case_study, current, nxt)
                if override is None:
                    skipped_ambiguous.append((current, nxt, sorted(distinct_columns)))
                    continue
                matching = [e for e in candidate_edges if e['column'] == override['column']]
                if not matching:
                    raise ValueError(
                        f"join_disambiguation override for ({case_study!r}, {current!r}, "
                        f"{nxt!r}) names column {override['column']!r}, which is not among "
                        f"the actual distinct columns found ({sorted(distinct_columns)}) -- "
                        f"the override is stale, fix it rather than silently ignoring it.")
                candidate_edges = matching
            edge = candidate_edges[0]
            hop = {'from_table': current, 'from_column': edge['column'],
                   'to_table': nxt, 'to_column': edge['ref_column']}
            new_path = path + [hop]
            if nxt == target_table:
                path_nodes = {root_table} | {h['to_table'] for h in new_path}
                for amb_from, amb_to, amb_cols in skipped_ambiguous:
                    if amb_from in path_nodes and amb_to in path_nodes:
                        raise ValueError(
                            f"Path from {root_table!r} to {target_table!r} found, but it "
                            f"routes around an ambiguous DIRECT connection between "
                            f"{amb_from!r} and {amb_to!r} ({len(amb_cols)} distinct FK "
                            f"columns: {amb_cols}) -- refusing a route that circumvents an "
                            f"unresolved ambiguity between two of its own members.")
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))

        if current != root_table:
            continue
        for edge in composite_backward_edges(case_study, current):
            nxt = edge['ref_table']
            if nxt not in allowed or nxt in visited or nxt in by_target:
                continue
            hop = {'from_table': current, 'from_columns': edge['columns'],
                   'to_table': nxt, 'to_columns': edge['ref_columns']}
            new_path = path + [hop]
            if nxt == target_table:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))

    return None

