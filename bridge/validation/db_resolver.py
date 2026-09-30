"""Reads the value of a decision input from the database, following its grounding."""
import re
import yaml

_PLACEHOLDER_RE = re.compile(r'<([A-Za-z_][A-Za-z0-9_ ]*)>')
_CONJUNCT_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*<([A-Za-z_][A-Za-z0-9_ ]*)>')


class UnresolvableForCase(Exception):
    pass


class ResolvedValue:
    def __init__(self, value, resolution_type, source_table, source_query, source_row=None):
        self.value = value
        self.resolution_type = resolution_type
        self.source_table = source_table
        self.source_query = source_query
        self.source_row = source_row

    def __repr__(self):
        return (f"ResolvedValue(value={self.value!r}, type={self.resolution_type!r}, "
                f"table={self.source_table!r})")


def _as_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def _one_row(conn, table, where_cols, where_vals):
    where_cols = _as_tuple(where_cols)
    where_vals = _as_tuple(where_vals)
    clause = ' AND '.join(f'"{c}" = ?' for c in where_cols)
    cur = conn.execute(f'SELECT * FROM "{table}" WHERE {clause}', where_vals)
    row = cur.fetchone()
    cols = [d[0].lower() for d in cur.description] if cur.description else []
    return dict(zip(cols, row)) if row is not None else None


def _row_for_table(conn, target_table, subject_table, subject_pk_cols, subject_pk_vals, join_paths):
    if target_table.upper() == subject_table.upper():
        return _one_row(conn, subject_table, subject_pk_cols, subject_pk_vals)

    path = join_paths.get(target_table)
    if path is None:
        path = next((v for k, v in join_paths.items() if k.upper() == target_table.upper()), None)
    if path is None:
        raise NotImplementedError(
            f"No join path from subject table {subject_table!r} to {target_table!r} -- "
            f"pass it via subject_table.subject_table_for_decision's own join_paths")

    current_row = _one_row(conn, subject_table, subject_pk_cols, subject_pk_vals)
    for hop in path:
        if current_row is None:
            return None
        if 'from_columns' in hop:
            fk_values = [current_row.get(c.lower()) for c in hop['from_columns']]
            current_row = _one_row(conn, hop['to_table'], hop['to_columns'], fk_values) \
                if all(v is not None for v in fk_values) else None
        else:
            fk_value = current_row.get(hop['from_column'].lower())
            current_row = _one_row(conn, hop['to_table'], hop['to_column'], fk_value) \
                if fk_value is not None else None
    return current_row


def _resolve_placeholders(conn, filter_text, subject_row, case_study=None,
                           subject_table=None, subject_pk_cols=None,
                           subject_pk_vals=None, join_paths=None):
    bindings = {}
    for column, placeholder in _CONJUNCT_RE.findall(filter_text or ''):
        column = column.lower()
        if column in subject_row:
            bindings[placeholder] = subject_row[column]
            continue
        source_table = None
        if case_study is not None and join_paths is not None:
            from bridge.validation.filter_placeholder_sources import get_source_table
            source_table = get_source_table(case_study, placeholder)
        if source_table is not None:
            joined_row = _row_for_table(conn, source_table, subject_table, subject_pk_cols,
                                         subject_pk_vals, join_paths)
            if joined_row is not None and column in joined_row:
                bindings[placeholder] = joined_row[column]
                continue
        raise NotImplementedError(
            f"filter_text placeholder <{placeholder}> binds to column {column!r}, "
            f"not present on the subject row ({sorted(subject_row)})"
            + (f" or on {source_table!r} (joined, but the column/row wasn't there)"
               if source_table else "")
            + " -- cannot resolve independently of a declared scenario value")
    return bindings


_COLON_RE = re.compile(r'(?<!:):(?!:)([A-Za-z_][A-Za-z0-9_]*)')
_SELF_RE = re.compile(r'\bself\b')


def _substitute_self_and_colon(text, subject_row, subject_pk_cols):
    def _colon_sub(m):
        column = m.group(1).lower()
        if column not in subject_row:
            raise NotImplementedError(
                f"filter_text ':{column}' has no matching column on the subject row "
                f"({sorted(subject_row)}) -- cannot resolve independently")
        return _sql_literal(subject_row[column])

    text = _COLON_RE.sub(_colon_sub, text or '')

    if _SELF_RE.search(text):
        if len(subject_pk_cols) != 1:
            raise NotImplementedError(
                f"filter_text uses 'self' but the subject table has a composite PK "
                f"{subject_pk_cols} -- a single self-value is ambiguous, not resolved")
        self_value = subject_row[subject_pk_cols[0].lower()]
        text = _SELF_RE.sub(_sql_literal(self_value), text)

    return text


def resolve(conn, node, subject_table, subject_pk_cols, subject_pk_vals,
            join_paths=None, declared_not_persisted_value=None, case_study=None):
    join_paths = join_paths or {}
    kind = node.get('kind')
    subject_row = _one_row(conn, subject_table, subject_pk_cols, subject_pk_vals)

    def row_for(table):
        return _row_for_table(conn, table, subject_table, subject_pk_cols, subject_pk_vals, join_paths)

    if kind == 'schema_column':
        table = node['table']
        row = row_for(table)
        value = row[node['column'].lower()] if row is not None else None
        constant = node.get('compared_to_named_constant')
        if constant is not None:
            value = value == constant['value']
        return ResolvedValue(value, 'schema_column', table, f'{table}.{node["column"]}', row)

    if kind == 'derived_case':
        table = node['table']
        row = row_for(table)
        real_value = row[node['column'].lower()] if row is not None else None
        for real, mapped in node['cases']:
            if real_value == real:
                return ResolvedValue(mapped, 'derived_case', table,
                                      f'{table}.{node["column"]} CASE_MAP', row)
        raise UnresolvableForCase(
            f"derived_case: real value {real_value!r} in {table}.{node['column']} isn't"
            f"covered by any CASE_MAP case ({node['cases']}) -- an unmapped real value, "
            f"not silently defaulted to one of the known categories")

    if kind == 'serialized_field':
        table, column, key = node['table'], node['column'], node['key']
        row = row_for(table)
        raw = row.get(column) if row is not None else None
        blob = yaml.safe_load(raw) if raw else None
        value = (blob or {}).get(f':{key}', (blob or {}).get(key, node.get('default')))
        return ResolvedValue(value, 'serialized_field', table,
                              f'{table}.{column}[{key}] (YAML)', row)

    if kind == 'null_check':
        table = node['table']
        row = row_for(table)
        if node.get('key'):
            raw = row.get(node['column']) if row is not None else None
            blob = yaml.safe_load(raw) if raw else None
            stored = (blob or {}).get(f":{node['key']}", (blob or {}).get(node['key'])) if blob else None
            is_set = stored is not None
            value = (not is_set) if node.get('negate') else is_set
            return ResolvedValue(value, 'null_check', table,
                                  f"{table}.{node['column']}[{node['key']}] "
                                  f"{'IS NULL' if node.get('negate') else 'IS NOT NULL'} (YAML)", row)
        is_set = (row[node['column'].lower()] is not None) if row is not None else False
        value = (not is_set) if node.get('negate') else is_set
        return ResolvedValue(value, 'null_check', table,
                              f'{table}.{node["column"]} '
                              f'{"IS NULL" if node.get("negate") else "IS NOT NULL"}', row)

    if kind == 'any_not_null':
        row = None
        value = False
        cols_checked = []
        for c in node['columns']:
            r = row_for(c['table'])
            cols_checked.append(f"{c['table']}.{c['column']}")
            if r is not None and r.get(c['column'].lower()) is not None:
                value = True
                row = r
        return ResolvedValue(value, 'any_not_null', node['columns'][0]['table'],
                              f"ANY NOT NULL among {cols_checked}", row)

    if kind == 'regex_match':
        vc, pc = node['value_column'], node['pattern_column']
        vrow, prow = row_for(vc['table']), row_for(pc['table'])
        value_str = vrow[vc['column'].lower()] if vrow else None
        pattern = prow[pc['column'].lower()] if prow else None
        matched = bool(pattern and value_str is not None and re.fullmatch(pattern, str(value_str)))
        return ResolvedValue(matched, 'regex_match', vc['table'],
                              f"regex_match({vc['table']}.{vc['column']}, pattern={pc['table']}.{pc['column']})",
                              {'value': value_str, 'pattern': pattern})

    if kind in ('join_lookup', 'join_null_check'):
        via = node['via']
        local_row = row_for(via['local_table'])
        if local_row is None:
            return ResolvedValue(None, kind, node['result_table'], 'local row not found', None)
        fk_value = local_row.get(via['local_column'].lower())
        result_row = _one_row(conn, node['result_table'], via['local_column'], fk_value) \
            if fk_value is not None else None
        raw_value = result_row.get(node['result_column'].lower()) if result_row else None
        value = (raw_value is None) if kind == 'join_null_check' else raw_value
        return ResolvedValue(value, kind, node['result_table'],
                              f'{via["local_table"]}.{via["local_column"]} -> '
                              f'{node["result_table"]}.{node["result_column"]}', result_row)

    if kind == 'exists':
        if node.get('filter_text'):
            table = node['candidate_tables'][0]
            where = _substitute_self_and_colon(node['filter_text'], subject_row or {}, subject_pk_cols)
            bindings = _resolve_placeholders(conn, where, subject_row or {}, case_study=case_study,
                                              subject_table=subject_table, subject_pk_cols=subject_pk_cols,
                                              subject_pk_vals=subject_pk_vals, join_paths=join_paths)
            for ph, val in bindings.items():
                where = where.replace(f'<{ph}>', _sql_literal(val))
            sql = f'SELECT EXISTS(SELECT 1 FROM "{table}" WHERE {where})'
            found = bool(conn.execute(sql).fetchone()[0])
            return ResolvedValue(found, 'exists', table, sql, None)

        found = False
        checked = []
        for cc in node['candidate_columns']:
            row = row_for(cc['table'])
            checked.append(f"{cc['table']}.{cc['column']}")
            if row is not None and row.get(cc['column'].lower()) is not None:
                found = True
        return ResolvedValue(found, 'exists', node['candidate_tables'][0] if node['candidate_tables'] else None,
                              f"non-null among {checked}" if checked else "no filter_text, no candidate_columns", None)

    if kind == 'raw_sql_boolean':
        sql = _substitute_self_and_colon(node['sql_template'], subject_row or {}, subject_pk_cols)
        if _PLACEHOLDER_RE.search(sql):
            bindings = _resolve_placeholders(conn, sql, subject_row or {}, case_study=case_study,
                                              subject_table=subject_table, subject_pk_cols=subject_pk_cols,
                                              subject_pk_vals=subject_pk_vals, join_paths=join_paths)
            for ph, val in bindings.items():
                sql = sql.replace(f'<{ph}>', _sql_literal(val))
        cur = conn.execute(f'SELECT ({sql})')
        row = cur.fetchone()
        value = row[0] if row else None
        return ResolvedValue(value, 'raw_sql_boolean', ','.join(node['tables']), sql, None)

    if kind == 'derived_aggregate':
        if node.get('filter_text') is None:
            raise UnresolvableForCase(
                f"derived_aggregate on {node['table']!r} has no filter_text -- its own "
                f"correlation isn't understood (source_text: {node.get('source_text')!r})")
        where = _substitute_self_and_colon(node['filter_text'], subject_row or {}, subject_pk_cols)
        bindings = _resolve_placeholders(conn, where, subject_row or {}, case_study=case_study,
                                          subject_table=subject_table, subject_pk_cols=subject_pk_cols,
                                          subject_pk_vals=subject_pk_vals, join_paths=join_paths)
        for ph, val in bindings.items():
            where = where.replace(f'<{ph}>', _sql_literal(val))
        target = node['value_column'] if node.get('value_column') else '*'
        from_tables = ', '.join(f'"{t.strip()}"' for t in node['table'].split(','))
        sql = f'SELECT {node["aggregate"]}({target}) FROM {from_tables} WHERE {where}'
        cur = conn.execute(sql)
        value = cur.fetchone()[0]
        return ResolvedValue(value, 'derived_aggregate', node['table'], sql, None)

    if kind == 'literal':
        return ResolvedValue(node['value'], 'literal', None, 'literal', None)

    if kind == 'not_persisted':
        if declared_not_persisted_value is None:
            raise NotImplementedError(
                "not_persisted has no table/column to query by definition -- "
                "caller must supply declared_not_persisted_value explicitly, flagged as "
                "not-database-derived (see DESIGN.md known gaps)")
        return ResolvedValue(declared_not_persisted_value, 'not_persisted_declared', None,
                              'declared scenario value, not database-derived', None)

    raise NotImplementedError(f"Unhandled variable_resolution kind {kind!r}: {node!r}")


def _sql_literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"
