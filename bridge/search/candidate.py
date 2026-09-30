"""Chromosome of the search.

A candidate is a whole database state: rows per table. An individual is a
candidate together with, for every objective, its focal rows (the rows the
objective's inputs are read from) and its scenario values (values of inputs
that are not stored in the database). `derive_value` and `derive_genome`
read the value of every decision input from an individual, following the
grounding of Phase 1."""
import json
import re
import sqlite3

from bridge import paths
from bridge.search.fitness import FitnessEvaluationError
from bridge.grounding.compile_constraints import CASE_STUDY_SCHEMA_JSON

_PLACEHOLDER_RE = re.compile(r'<([^>]+)>')
_SEED_SCHEMA_CACHE = {}


def _schema_for_seeding(case_study):
    if case_study not in _SEED_SCHEMA_CACHE:
        with open(CASE_STUDY_SCHEMA_JSON[case_study], encoding='utf-8') as f:
            _SEED_SCHEMA_CACHE[case_study] = json.load(f)
    return _SEED_SCHEMA_CACHE[case_study]


def _real_columns_of(case_study, table):
    schema = _schema_for_seeding(case_study)
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    return {c.upper() for c in (info.get('columns') or {})}


def _pk_columns_of(case_study, table):
    schema = _schema_for_seeding(case_study)
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    pk = info.get('pk')
    return [c.upper() for c in (pk if isinstance(pk, list) else ([pk] if pk else []))]


class Candidate:
    def __init__(self):
        self._tables = {}

    def add_row(self, table, row):
        self._tables.setdefault(table.upper(), []).append(row)
        return row

    def rows(self, table):
        return self._tables.get(table.upper(), [])

    def as_dict(self):
        return self._tables


_OWNER_KEY = '__owner__'


def _owned_rows(candidate, table, owner_id):
    rows = candidate.rows(table)
    if owner_id is None:
        return rows
    return [r for r in rows if r.get(_OWNER_KEY) in (None, owner_id)]


def known_constant(case_study, var_name):
    return paths.known_constants(case_study).get(var_name)


def _row_get(row, column, table_for_error=None):
    if column in row:
        return row[column]
    lowered = {k.lower(): v for k, v in row.items()}
    if column.lower() not in lowered:
        where = f" in {table_for_error!r}" if table_for_error else ""
        raise FitnessEvaluationError(f"no column {column!r} found{where}: {sorted(row)}")
    return lowered[column.lower()]


def _lookup(focal, table, column):
    row = focal.get(table.upper())
    if row is None:
        raise FitnessEvaluationError(
            f"no focal row given for table {table!r} -- can't read {table}.{column} "
            f"from a candidate with no row there yet")
    return _row_get(row, column, table)


def _find_row_by_pk(candidate, table, pk_column, value):
    for row in candidate.rows(table):
        try:
            if _row_get(row, pk_column) == value:
                return row
        except FitnessEvaluationError:
            continue
    return None


_SIMPLE_EQ_CONJUNCT_RE = re.compile(r'^\s*(?:[\w]+\.)?([\w]+)\s*=\s*(.+?)\s*$')
_BARE_TABLE_DOT_COLUMN_RE = re.compile(r'^[A-Za-z_]\w*\.[A-Za-z_]\w*$')
_IS_NOT_NULL_RE = re.compile(r'^\s*(?:[\w]+\.)?(\w+)\s+IS\s+NOT\s+NULL\s*$', re.I)
_IS_NULL_RE = re.compile(r'^\s*(?:[\w]+\.)?(\w+)\s+IS\s+NULL\s*$', re.I)
_COLON_SELF_REF_RE = re.compile(r'^:([A-Za-z_]\w*)$')
_NEQ_COLON_SELF_REF_RE = re.compile(r'^\s*(?:[\w]+\.)?(\w+)\s*!=\s*:([A-Za-z_]\w*)\s*$')
_COLON_SELF_REF_ANYWHERE_RE = re.compile(r'(?<!:):(?!:)[A-Za-z_]\w*')


def _self_row_value(self_row, column):
    if self_row is None:
        return None, False
    for key in (column, column.upper(), column.lower()):
        if key in self_row:
            return self_row[key], True
    return None, False


def _fresh_value_for_self_correlation(candidate, table, column):
    existing = {r.get(column) for r in candidate.rows(table)
                if isinstance(r.get(column), (int, float)) and not isinstance(r.get(column), bool)}
    return (max(existing) + 1) if existing else 1


def _top_level_and_conjuncts(filter_text):
    text = filter_text or ''
    parts, current, depth, i, n = [], [], 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '(':
            depth += 1
            current.append(ch)
            i += 1
        elif ch == ')':
            depth -= 1
            current.append(ch)
            i += 1
        elif depth == 0 and text[i:i + 3].upper() == 'AND' \
                and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')) \
                and (i + 3 == n or not (text[i + 3].isalnum() or text[i + 3] == '_')):
            parts.append(''.join(current))
            current = []
            i += 3
        else:
            current.append(ch)
            i += 1
    parts.append(''.join(current))
    return parts


_IN_SUBQUERY_RE = re.compile(
    r'^\s*(?:[\w]+\.)?(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)\s+WHERE\s+(.+)\)\s*$',
    re.I | re.S)
_SELF_TOKEN_RE = re.compile(r'self', re.I)


def _conjunct_value_bindings(where_text, scenario, self_value=None):
    bindings = {}
    for c in _top_level_and_conjuncts(where_text):
        c_stripped = c.strip()
        m = _IS_NOT_NULL_RE.match(c_stripped)
        if m:
            bindings[m.group(1)] = 1
            continue
        if _IS_NULL_RE.match(c_stripped):
            continue
        m = _SIMPLE_EQ_CONJUNCT_RE.match(c_stripped)
        if not m:
            continue
        col, raw_val = m.group(1), m.group(2).strip()
        if col.isdigit():
            continue
        if _SELF_TOKEN_RE.fullmatch(raw_val):
            if self_value is not None:
                bindings[col] = self_value
            continue
        if _BARE_TABLE_DOT_COLUMN_RE.match(raw_val):
            continue
        ph = _PLACEHOLDER_RE.fullmatch(raw_val)
        if ph:
            if ph.group(1) in scenario:
                bindings[col] = scenario[ph.group(1)]
            continue
        v = raw_val.strip("'\"")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        bindings[col] = v
    return bindings


def _fresh_id_value(candidate, table, column='id'):
    def value(r):
        return next((v for k, v in r.items() if k.upper() == column.upper()), None)
    existing = {value(r) for r in candidate.rows(table)
                if isinstance(value(r), (int, float)) and not isinstance(value(r), bool)}
    return (max(existing) + 1) if existing else 1


def _construct_subquery_parent(match, candidate, focal, scenario, self_table, owner_id=None):
    if candidate is None:
        return None
    outer_col, select_col, inner_table, inner_where = match.groups()
    self_value = None
    if self_table and focal is not None and _SELF_TOKEN_RE.search(inner_where):
        self_row = focal.setdefault(self_table.upper(), {})
        if self_row not in candidate.rows(self_table):
            candidate.add_row(self_table, self_row)
        self_value = self_row.get('id', self_row.get('ID'))
        if self_value is None:
            self_value = _fresh_id_value(candidate, self_table)
            self_row['id'] = self_value
    bindings = _conjunct_value_bindings(inner_where, scenario, self_value)
    inner_row = dict(bindings)
    inner_row[select_col] = _fresh_id_value(candidate, inner_table, select_col)
    if owner_id is not None:
        inner_row[_OWNER_KEY] = owner_id
    candidate.add_row(inner_table, inner_row)
    return inner_row[select_col]


def _mechanical_filter_predicate(filter_text, scenario, candidate=None, self_row=None):
    if not filter_text:
        return (lambda row: True), []
    conjuncts = _top_level_and_conjuncts(filter_text)
    checks = []
    skipped = []
    for c in conjuncts:
        c_stripped = c.strip()
        m = _IN_SUBQUERY_RE.match(c_stripped)
        if m:
            if candidate is None:
                skipped.append(c_stripped)
                continue
            outer_col, select_col, inner_table, inner_where = m.groups()
            inner_bindings = _conjunct_value_bindings(inner_where, scenario, self_value=None)

            def subquery_check(row, outer_col=outer_col, inner_table=inner_table,
                                inner_bindings=inner_bindings, select_col=select_col):
                try:
                    outer_val = _row_get(row, outer_col)
                except FitnessEvaluationError:
                    return False
                if outer_val is None:
                    return False
                for inner_row in candidate.rows(inner_table):
                    inner_id = inner_row.get(select_col, inner_row.get(
                        select_col.upper(), inner_row.get(select_col.lower())))
                    if inner_id != outer_val:
                        continue
                    if all(inner_row.get(k, inner_row.get(k.upper())) == v
                           for k, v in inner_bindings.items()):
                        return True
                return False

            checks.append((None, 'custom', subquery_check))
            continue
        neq_m = _NEQ_COLON_SELF_REF_RE.match(c_stripped)
        if neq_m:
            self_value, found = _self_row_value(self_row, neq_m.group(2))
            if not found:
                skipped.append(c_stripped)
                continue
            checks.append((neq_m.group(1), 'neq', self_value))
            continue
        m = _IS_NOT_NULL_RE.match(c_stripped)
        if m:
            checks.append((m.group(1), 'not_null', None))
            continue
        m = _IS_NULL_RE.match(c_stripped)
        if m:
            checks.append((m.group(1), 'is_null', None))
            continue
        m = _SIMPLE_EQ_CONJUNCT_RE.match(c_stripped)
        if not m:
            skipped.append(c_stripped)
            continue
        col, raw_val = m.group(1), m.group(2).strip()
        if col.isdigit():
            continue
        if _BARE_TABLE_DOT_COLUMN_RE.match(raw_val):
            skipped.append(c.strip())
            continue
        colon_m = _COLON_SELF_REF_RE.fullmatch(raw_val)
        if colon_m:
            self_value, found = _self_row_value(self_row, colon_m.group(1))
            if not found:
                skipped.append(c_stripped)
                continue
            checks.append((col, 'eq', self_value))
            continue
        ph = _PLACEHOLDER_RE.fullmatch(raw_val)
        if ph:
            if ph.group(1) not in scenario:
                raise FitnessEvaluationError(
                    f"filter text needs scenario binding {raw_val!r}, none supplied")
            expected = scenario[ph.group(1)]
        else:
            expected = raw_val.strip("'\"")
            if expected.lower() in ('true', 'false'):
                expected = (expected.lower() == 'true')
            else:
                try:
                    expected = int(expected)
                except ValueError:
                    try:
                        expected = float(expected)
                    except ValueError:
                        pass
        checks.append((col, 'eq', expected))

    def predicate(row):
        for col, mode, expected in checks:
            if mode == 'custom':
                if not expected(row):
                    return False
                continue
            try:
                value = _row_get(row, col)
            except FitnessEvaluationError:
                return False
            if mode == 'not_null':
                if value is None:
                    return False
            elif mode == 'is_null':
                if value is not None:
                    return False
            elif mode == 'neq':
                if value == expected:
                    return False
            elif value != expected:
                return False
        return True

    return predicate, skipped


def _row_from_filter_conjuncts(filter_text, scenario, candidate=None, focal=None, self_table=None, table=None):
    self_row = focal.get(self_table.upper()) if (focal and self_table) else None
    row = {}
    for c in _top_level_and_conjuncts(filter_text):
        c_stripped = c.strip()
        m = _IN_SUBQUERY_RE.match(c_stripped)
        if m:
            outer_val = _construct_subquery_parent(m, candidate, focal, scenario, self_table)
            if outer_val is not None:
                row[m.group(1)] = outer_val
            continue
        m = _IS_NOT_NULL_RE.match(c_stripped)
        if m:
            row[m.group(1)] = 1
            continue
        if _IS_NULL_RE.match(c_stripped):
            continue
        m = _SIMPLE_EQ_CONJUNCT_RE.match(c_stripped)
        if not m:
            continue
        col, raw_val = m.group(1), m.group(2).strip()
        if col.isdigit():
            continue
        if _BARE_TABLE_DOT_COLUMN_RE.match(raw_val):
            continue
        colon_m = _COLON_SELF_REF_RE.fullmatch(raw_val)
        if colon_m:
            self_value, found = _self_row_value(self_row, colon_m.group(1))
            if found:
                row[col] = self_value
            elif candidate is not None and focal is not None and self_table is not None:
                fresh = _fresh_value_for_self_correlation(candidate, self_table, colon_m.group(1))
                row[col] = fresh
                if self_row is not None:
                    self_row[colon_m.group(1)] = fresh
                elif table is not None and self_table.upper() == table.upper():
                    row[colon_m.group(1)] = fresh
                    focal[self_table.upper()] = row
                    self_row = row
                else:
                    self_row = focal.setdefault(self_table.upper(), {})
                    if candidate is not None and self_row not in candidate.rows(self_table):
                        candidate.add_row(self_table, self_row)
                    self_row[colon_m.group(1)] = fresh
            continue
        ph = _PLACEHOLDER_RE.fullmatch(raw_val)
        if ph:
            if ph.group(1) in scenario:
                row[col] = scenario[ph.group(1)]
        else:
            v = raw_val.strip("'\"")
            if v.lower() in ('true', 'false'):
                v = (v.lower() == 'true')
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            row[col] = v
    return row


def _raw_sql_boolean_value(node, candidate, scenario, warnings):
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    for table in node['tables']:
        seen = {}
        for row in candidate.rows(table):
            for c in row:
                seen.setdefault(c.lower(), c)
        cols = sorted(seen.values())
        if not cols:
            cols = ['_dummy']
        cur.execute(f'CREATE TABLE {table} ({", ".join(cols)})')
        for row in candidate.rows(table):
            placeholders = ', '.join('?' for _ in cols)
            values = []
            for c in cols:
                try:
                    values.append(_row_get(row, c))
                except FitnessEvaluationError:
                    values.append(None)
            cur.execute(f'INSERT INTO {table} ({", ".join(cols)}) VALUES ({placeholders})', values)
    sql = _PLACEHOLDER_RE.sub(lambda m: str(scenario.get(m.group(1), 'NULL')), node['sql_template'])
    try:
        cur.execute(f'SELECT ({sql})')
        result = cur.fetchone()[0]
    except sqlite3.Error as e:
        raise FitnessEvaluationError(f"raw_sql_boolean query failed against the materialized candidate: {e}")
    finally:
        conn.close()
    return bool(result)


def derive_value(var_name, node, candidate, focal, scenario, warnings=None, owner_id=None):
    warnings = warnings if warnings is not None else []
    kind = node.get('kind')
    if kind == 'schema_column':
        real = _lookup(focal, node['table'], node['column'])
        constant = node.get('compared_to_named_constant')
        if constant is not None:
            return real == constant['value']
        return real
    if kind == 'serialized_field':
        row = focal.get(node['table'].upper())
        if row is None:
            raise FitnessEvaluationError(
                f"no focal row given for table {node['table']!r} -- can't read "
                f"{node['table']}.{node['column']}[{node['key']}]")
        try:
            blob = _row_get(row, node['column'], node['table']) or {}
        except FitnessEvaluationError:
            blob = {}
        return blob.get(node['key'], node.get('default'))
    if kind == 'null_check':
        negate = bool(node.get('negate'))
        if node.get('key'):
            row = focal.get(node['table'].upper())
            if row is None:
                raise FitnessEvaluationError(
                    f"no focal row given for table {node['table']!r} -- can't read "
                    f"{node['table']}.{node['column']}[{node['key']}]")
            blob = row.get(node['column']) or {}
            is_set = blob.get(node['key']) is not None
            return (not is_set) if negate else is_set
        is_set = _lookup(focal, node['table'], node['column']) is not None
        return (not is_set) if negate else is_set
    if kind == 'any_not_null':
        return any(_lookup(focal, c['table'], c['column']) is not None for c in node['columns'])
    if kind in ('join_lookup', 'join_null_check'):
        local_val = _lookup(focal, node['via']['local_table'], node['via']['local_column'])
        target_row = _find_row_by_pk(candidate, node['result_table'], node['result_column'], local_val) \
            if local_val is not None else None
        value = _row_get(target_row, node['result_column']) if target_row else None
        return value if kind == 'join_lookup' else value is not None
    if kind == 'regex_match':
        val = _lookup(focal, node['value_column']['table'], node['value_column']['column'])
        pattern = _lookup(focal, node['pattern_column']['table'], node['pattern_column']['column'])
        if val is None or pattern is None:
            return False
        try:
            return re.search(pattern, str(val)) is not None
        except re.error as e:
            raise FitnessEvaluationError(f"{var_name!r}'s pattern column holds an invalid regex: {e}")
    if kind == 'derived_aggregate':
        self_table = node.get('self_table') or node['table']
        self_row = focal.get(self_table.upper()) if focal else None
        predicate, skipped = _mechanical_filter_predicate(node.get('filter_text'), scenario, candidate, self_row)
        if skipped:
            warnings.append(f"{var_name}: derived_aggregate filter has prose this bridge can't "
                             f"mechanically apply, counted without it: {skipped}")
        tables = [t.strip() for t in node['table'].split(',')]
        rows = _owned_rows(candidate, tables[0], owner_id)
        matching = [r for r in rows if predicate(r)]
        if node.get('value_column'):
            value_table, col = node['value_column'].split('.')
            if value_table.upper() == tables[0].upper():
                return sum((_row_get(r, col) or 0) for r in matching)
            join_col = f'{value_table.upper()}_ID'
            value_rows = _owned_rows(candidate, value_table, owner_id)
            total = 0
            for r in matching:
                try:
                    key = _row_get(r, join_col)
                except FitnessEvaluationError:
                    continue
                target = next((vr for vr in value_rows
                               if vr.get(join_col, vr.get(join_col.lower())) == key), None)
                if target is not None:
                    total += _row_get(target, col) or 0
            return total
        return len(matching)
    if kind == 'exists':
        table = (node.get('candidate_tables') or [None])[0]
        if table is None:
            return False
        if node.get('filter_text') and not _PLACEHOLDER_RE.search(node['filter_text']) \
                and not _COLON_SELF_REF_ANYWHERE_RE.search(node['filter_text']):
            predicate, _skipped = _mechanical_filter_predicate(node['filter_text'], scenario, candidate, None)
            return any(predicate(r) for r in candidate.rows(table))
        self_table = node.get('self_table') or table
        self_row = focal.get(self_table.upper()) if focal else None
        predicate, _skipped = _mechanical_filter_predicate(node.get('filter_text'), scenario, candidate, self_row)
        return any(predicate(r) for r in _owned_rows(candidate, table, owner_id))
    if kind == 'raw_sql_boolean':
        return _raw_sql_boolean_value(node, candidate, scenario, warnings)
    if kind == 'derived_case':
        real_value = _lookup(focal, node['table'], node['column'])
        for k, v in node['cases']:
            if real_value == k:
                return v
        raise FitnessEvaluationError(
            f"{var_name!r}: real value {real_value!r} in {node['table']}.{node['column']} "
            f"isn't covered by any CASE_MAP case ({node['cases']}) -- an unmapped real value, "
            f"not silently defaulted to one of the known categories")
    if kind == 'not_persisted':
        if var_name not in scenario:
            raise FitnessEvaluationError(
                f"{var_name!r} is not_persisted (a scenario/runtime parameter) -- "
                f"no value supplied in `scenario`")
        return scenario[var_name]
    if kind == 'derived':
        raise FitnessEvaluationError(
            f"{var_name!r} is an unclassified 'derived' fact (compile_constraints.py's own "
            f"classify_derived never matched it to a structured shape) -- no mechanical value "
            f"to derive from a candidate; needs a human or a more targeted classifier pass, "
            f"the same as it does for SQL compilation")
    raise FitnessEvaluationError(f"derive_value has no handling for resolution kind {kind!r} ({var_name!r})")


def derive_genome(record, candidate, focal, scenario, warnings=None, owner_id=None):
    warnings = warnings if warnings is not None else []
    genome = {}
    if '__today__' in scenario:
        genome['__today__'] = scenario['__today__']

    def walk(var_name, node):
        kind = node.get('kind')
        if kind == 'literal':
            return
        if kind == 'substituted_decision':
            for fv, sub in node.get('free_variable_resolutions', {}).items():
                walk(fv, sub)
            return
        if kind == 'literal_via_upstream_branch':
            return
        if kind in ('schema_gap', 'code_external', 'unresolved', 'chained_decision_output'):
            return
        genome[var_name] = derive_value(var_name, node, candidate, focal, scenario, warnings, owner_id=owner_id)

    for var, node in record.get('variable_resolution', {}).items():
        walk(var, node)
    return genome


def _leaf_kinds_for_seeding(var_name, node, out, blocking_kinds):
    kind = node.get('kind')
    if kind == 'substituted_decision':
        for fv, sub in node.get('free_variable_resolutions', {}).items():
            _leaf_kinds_for_seeding(fv, sub, out, blocking_kinds)
    elif kind not in ('literal', 'literal_via_upstream_branch') and kind not in blocking_kinds:
        out.append((var_name, kind, node))


_SEEDING_BLOCKING_KINDS = {'schema_gap', 'code_external', 'unresolved', 'chained_decision_output'}


def build_seed_candidate(record, today=20000):
    leaves = []
    for var, node in record.get('variable_resolution', {}).items():
        _leaf_kinds_for_seeding(var, node, leaves, _SEEDING_BLOCKING_KINDS)

    candidate = Candidate()
    focal = {}
    scenario = {'__today__': today}
    for _var, _kind, node in leaves:
        for field in ('filter_text', 'sql_template'):
            for m in _PLACEHOLDER_RE.finditer(node.get(field) or ''):
                scenario.setdefault(m.group(1), 1)

    def ensure_row(table, col=None, val=1):
        row = focal.get(table.upper())
        if row is None:
            row = {}
            focal[table.upper()] = row
            candidate.add_row(table, row)
        if col:
            row[col] = val
        return row

    for var, kind, node in leaves:
        if kind == 'schema_column':
            ensure_row(node['table'], node['column'], 1)
        elif kind == 'null_check':
            if node.get('key'):
                row = ensure_row(node['table'])
                row.setdefault(node['column'], {})[node['key']] = 1
            else:
                ensure_row(node['table'], node['column'], 1)
        elif kind == 'serialized_field':
            row = ensure_row(node['table'])
            row.setdefault(node['column'], {})[node['key']] = node.get('default', 1)
        elif kind == 'any_not_null':
            for x in node['columns']:
                ensure_row(x['table'], x['column'], 1)
        elif kind in ('join_lookup', 'join_null_check'):
            ensure_row(node['via']['local_table'], node['via']['local_column'], 42)
            ensure_row(node['result_table'], node['result_column'], 42)
        elif kind == 'regex_match':
            ensure_row(node['value_column']['table'], node['value_column']['column'], 'abc')
            ensure_row(node['pattern_column']['table'], node['pattern_column']['column'], 'a.*')
        elif kind == 'derived_aggregate':
            tables = [t.strip() for t in node['table'].split(',')]
            value_table, value_col = (node['value_column'].split('.') if node.get('value_column') else (None, None))
            joined_value_table = value_table and value_table.upper() != tables[0].upper()
            seed_count = known_constant(record['case_study'], var) or 3
            for i in range(1, seed_count + 1):
                row = _row_from_filter_conjuncts(node.get('filter_text'), scenario,
                                                  candidate, focal, node.get('self_table') or tables[0],
                                                  table=tables[0])
                if value_table and not joined_value_table:
                    row[value_col] = 10
                elif joined_value_table:
                    join_col = f'{value_table.upper()}_ID'
                    row[join_col] = i
                    candidate.add_row(value_table, {join_col: i, value_col: 10})
                candidate.add_row(tables[0], row)
            for t in tables[1:]:
                if joined_value_table and t.upper() == value_table.upper():
                    continue
                candidate.add_row(t, {})
        elif kind == 'exists':
            table = (node.get('candidate_tables') or [None])[0]
            if table:
                if node.get('filter_text'):
                    candidate.add_row(table, _row_from_filter_conjuncts(
                        node['filter_text'], scenario, candidate, focal, node.get('self_table') or table,
                        table=table))
                else:
                    candidate.add_row(table, {})
        elif kind == 'raw_sql_boolean':
            all_cols = set(re.findall(r'\b\w+\.(\w+)\b', node['sql_template']))
            for t in node['tables']:
                real_cols = _real_columns_of(record['case_study'], t)
                cols = {c for c in all_cols if c.upper() in real_cols} if real_cols else set()
                new_row = {col: 1 for col in cols} or {'X': 1}
                pk = _pk_columns_of(record['case_study'], t)
                key = {c: new_row[c] for c in new_row if c.upper() in pk}
                same = None
                if pk and len(key) == len(pk):
                    same = next((r for r in candidate.rows(t)
                                 if all(next((v for k, v in r.items() if k.upper() == c.upper()), None) == val
                                        for c, val in key.items())), None)
                if same is not None:
                    for c, val in new_row.items():
                        if not any(k.upper() == c.upper() for k in same):
                            same[c] = val
                else:
                    candidate.add_row(t, new_row)
        elif kind == 'derived_case':
            ensure_row(node['table'], node['column'], node['cases'][0][0])
        elif kind == 'not_persisted':
            scenario[var] = 1
    return candidate, focal, scenario
