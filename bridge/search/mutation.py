"""Mutation and repair of candidate databases.

The operators change the value of one grounded input (numeric steps,
enumerated domains, boundary values) or add and remove related rows for
aggregate and existence inputs. `repair_candidate` fills primary keys,
foreign keys and NOT NULL columns so that every candidate stays consistent
with the schema."""
import json
import random
import re

from bridge.search.candidate import (_mechanical_filter_predicate, _owned_rows, _OWNER_KEY, known_constant, _top_level_and_conjuncts, _IN_SUBQUERY_RE, _construct_subquery_parent)
from bridge.search.fitness import branch_fitness, FitnessEvaluationError, _unique_key_sets
from bridge.grounding.compile_constraints import CASE_STUDY_SCHEMA_JSON

BOOLEAN_LEAF_KINDS = {'null_check', 'any_not_null', 'join_null_check', 'exists',
                      'regex_match', 'raw_sql_boolean'}
FIELD_LEAF_KINDS = {'schema_column', 'null_check', 'any_not_null', 'join_lookup',
                    'join_null_check', 'regex_match', 'derived_case', 'serialized_field'}
AGGREGATE_LEAF_KINDS = {'derived_aggregate', 'exists'}

_SCHEMA_CACHE = {}


def _schema_for(case_study):
    if case_study not in _SCHEMA_CACHE:
        with open(CASE_STUDY_SCHEMA_JSON[case_study], encoding='utf-8') as f:
            _SCHEMA_CACHE[case_study] = json.load(f)
    return _SCHEMA_CACHE[case_study]


def _column_type(case_study, table, column):
    schema = _schema_for(case_study)
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    col_meta = (info.get('columns') or {}).get(column) \
        or (info.get('columns') or {}).get(column.upper()) \
        or (info.get('columns') or {}).get(column.lower())
    return (col_meta or {}).get('type', '').upper()


def _column_is_not_null(case_study, table, column):
    schema = _schema_for(case_study)
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    col_meta = (info.get('columns') or {}).get(column) \
        or (info.get('columns') or {}).get(column.upper()) \
        or (info.get('columns') or {}).get(column.lower())
    return bool((col_meta or {}).get('null_false'))


def _collect_domain_facts(node, var_name, negated, hit, avoid):
    if not isinstance(node, dict):
        return
    op = node.get('op')
    if op == 'not':
        _collect_domain_facts(node.get('clause'), var_name, not negated, hit, avoid)
        return
    if op in ('=', '!='):
        left, right = node.get('left'), node.get('right')
        lit = None
        if isinstance(left, dict) and left.get('kind') == 'variable' and left.get('ref') == var_name \
                and isinstance(right, dict) and right.get('kind') == 'literal':
            lit = right['value']
        elif isinstance(right, dict) and right.get('kind') == 'variable' and right.get('ref') == var_name \
                and isinstance(left, dict) and left.get('kind') == 'literal':
            lit = left['value']
        if lit is not None:
            wants_equal = (op == '=') != negated
            (hit if wants_equal else avoid).add(lit)
    elif op == 'in':
        left = node.get('left')
        if isinstance(left, dict) and left.get('kind') == 'variable' and left.get('ref') == var_name:
            vals = {v['value'] for v in node.get('values', []) if isinstance(v, dict) and v.get('kind') == 'literal'}
            (avoid if negated else hit).update(vals)
    for key in ('left', 'right', 'clause', 'cond', 'then', 'else'):
        if key in node:
            _collect_domain_facts(node[key], var_name, negated, hit, avoid)
    for key in ('clauses', 'values'):
        for child in node.get(key, []):
            _collect_domain_facts(child, var_name, negated, hit, avoid)


def _domain_escape_value(literals, current):
    sample = next(iter(literals)) if literals else current
    if isinstance(sample, bool) or isinstance(current, bool):
        return not bool(sample)
    if isinstance(sample, (int, float)) and not isinstance(sample, bool):
        numeric = [v for v in literals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return (max(numeric) if numeric else 0) + 1
    base = str(current) if current is not None else str(sample)
    escape = base + '_'
    while escape in literals:
        escape += '_'
    return escape


def enumerable_domain(record, var_name):
    hit, avoid = set(), set()
    _collect_domain_facts(record['condition'], var_name, False, hit, avoid)
    for row in record.get('hit_policy_context', {}).get('earlier_rows', []):
        _collect_domain_facts(row['condition'], var_name, True, hit, avoid)
    found = hit | avoid
    if not found:
        return None
    try:
        return sorted(found)
    except TypeError:
        return sorted(found, key=str)


def candidate_values(record, var_name, node, current, case_study, step=1):
    kind = node.get('kind')
    if kind in BOOLEAN_LEAF_KINDS:
        return [True, False]
    domain = enumerable_domain(record, var_name)
    if domain:
        values = [v for v in domain if v != current] or list(domain)
        hit, avoid = set(), set()
        _collect_domain_facts(record['condition'], var_name, False, hit, avoid)
        for row in record.get('hit_policy_context', {}).get('earlier_rows', []):
            _collect_domain_facts(row['condition'], var_name, True, hit, avoid)
        if avoid:
            values.append(_domain_escape_value(hit | avoid, current))
        return values
    if kind == 'schema_column':
        ctype = _column_type(case_study, node['table'], node['column'])
        if 'DATE' in ctype or 'TIME' in ctype:
            base = current if isinstance(current, (int, float)) else 0
            return [base - step, base + step]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return [current - step, current + step]
        if isinstance(current, str):
            if not current:
                return [current + 'a']
            i = random.randrange(len(current))
            return [current[:i] + chr((ord(current[i]) + 1) % 128) + current[i + 1:]]
        return []
    if kind == 'derived_aggregate':
        base = current if isinstance(current, (int, float)) else 0
        return [max(0, base - step), base + step]
    if kind == 'not_persisted' and isinstance(current, (int, float)) and not isinstance(current, bool):
        return [current - step, current + step]
    return []


def _sibling_field_leaves(record, var_name, node):
    if node.get('kind') not in ('schema_column', 'null_check'):
        return []
    table, column = node.get('table'), node.get('column')
    if not table or not column:
        return []
    siblings = []

    def walk(name, n):
        if not isinstance(n, dict) or name == var_name:
            return
        if n.get('kind') == 'substituted_decision':
            for fv, sub in n.get('free_variable_resolutions', {}).items():
                walk(fv, sub)
            return
        if n.get('kind') in ('schema_column', 'null_check') \
                and n.get('table') == table and n.get('column') == column:
            siblings.append((name, n))

    for name, n in record.get('variable_resolution', {}).items():
        walk(name, n)
    return siblings


def _hypothetical_fitness(record, genome, var_name, value, node=None, case_study=None):
    trial = dict(genome)
    trial[var_name] = value
    if node is not None:
        kind = node.get('kind')
        for sib_name, sib_node in _sibling_field_leaves(record, var_name, node):
            sib_kind = sib_node.get('kind')
            if kind == 'null_check' and sib_kind == 'schema_column':
                wants_set = (not value) if node.get('negate') else value
                if not wants_set:
                    trial[sib_name] = None
                elif trial.get(sib_name) is None:
                    col_type = _column_type(case_study, node['table'], node['column']) \
                        if case_study is not None else None
                    trial[sib_name] = _placeholder_for_column_type(col_type) if case_study is not None else 1
            elif kind == 'schema_column' and sib_kind == 'null_check':
                is_set = value is not None
                trial[sib_name] = (not is_set) if sib_node.get('negate') else is_set
    try:
        return branch_fitness(record, trial)
    except FitnessEvaluationError:
        return float('inf')


_AVM_MAX_ROUNDS = 60


def _numeric_step_eligible(record, var_name, node, current):
    kind = node.get('kind')
    if kind in BOOLEAN_LEAF_KINDS:
        return False
    if enumerable_domain(record, var_name):
        return False
    if kind in ('schema_column', 'not_persisted'):
        return isinstance(current, (int, float)) and not isinstance(current, bool)
    return kind == 'derived_aggregate'


def best_value_for(record, genome, var_name, node, case_study):
    current = genome.get(var_name)
    try:
        current_fitness = branch_fitness(record, genome)
    except FitnessEvaluationError:
        current_fitness = float('inf')
    best_value, best_fitness = current, current_fitness

    pinned = known_constant(case_study, var_name)
    if pinned is not None:
        if pinned == current:
            return current, current_fitness, False
        return pinned, _hypothetical_fitness(record, genome, var_name, pinned, node, case_study), True

    if not _numeric_step_eligible(record, var_name, node, current):
        for value in candidate_values(record, var_name, node, current, case_study):
            f = _hypothetical_fitness(record, genome, var_name, value, node, case_study)
            if f < best_fitness:
                best_value, best_fitness = value, f
        return best_value, best_fitness, best_value != current

    step = 1
    direction = None
    tried_reset = False
    for _round in range(_AVM_MAX_ROUNDS):
        values = candidate_values(record, var_name, node, best_value, case_study, step=step)
        if direction is not None:
            values = [v for v in values if (v > best_value) == (direction > 0)]
        round_value, round_fitness, round_direction = best_value, best_fitness, None
        for value in values:
            f = _hypothetical_fitness(record, genome, var_name, value, node, case_study)
            if f < round_fitness:
                round_value, round_fitness, round_direction = value, f, (1 if value > best_value else -1)
        if round_fitness < best_fitness:
            best_value, best_fitness = round_value, round_fitness
            direction = round_direction
            step *= 2
            tried_reset = False
            if best_fitness == 0.0:
                break
        elif step > 1:
            step = max(1, step // 2)
        elif direction is not None and not tried_reset:
            direction = None
            tried_reset = True
        else:
            break
    return best_value, best_fitness, best_value != current


def _placeholder_for_column_type(ctype):
    ctype = (ctype or '').upper()
    if 'DATE' in ctype or 'TIME' in ctype:
        return '2000-01-01'
    if any(t in ctype for t in ('NUM', 'INT', 'DEC', 'FLOAT', 'DOUBLE')):
        return 1
    return 'X'


def _fresh_key_value(candidate, table, column):
    existing = {r.get(column) for r in candidate.rows(table)
                if isinstance(r.get(column), (int, float)) and not isinstance(r.get(column), bool)}
    return (max(existing) + 1) if existing else 1


def _repair_row(candidate, table, row, case_study, _seen=None):
    _seen = _seen if _seen is not None else set()
    if id(row) in _seen:
        return
    _seen.add(id(row))
    schema = _schema_for(case_study)
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    if not info:
        return
    real_table = table
    key_cols = {c.upper() for cols in _unique_key_sets(schema, real_table) for c in cols}
    pk = info.get('pk')
    pk_cols = {c.upper() for c in (pk if isinstance(pk, list) else ([pk] if pk else []))}
    lowered = {k.lower() for k in row}
    for col, meta in (info.get('columns') or {}).items():
        if not (meta.get('null_false') or col.upper() in pk_cols) or col.lower() in lowered:
            continue
        if col.upper() in key_cols:
            row[col] = _fresh_key_value(candidate, real_table, col)
        else:
            row[col] = _placeholder_for_column_type(meta.get('type'))
        lowered.add(col.lower())
    for fk in (info.get('fk_columns') or []):
        col = fk['column']
        val = row.get(col)
        if val is None:
            continue
        ref_table, ref_col = fk['ref_table'], fk['ref_column']
        parent_row = None
        for pr in candidate.rows(ref_table):
            v = pr.get(ref_col, pr.get(ref_col.upper(), pr.get(ref_col.lower())))
            if v == val:
                parent_row = pr
                break
        if parent_row is None:
            parent_row = candidate.add_row(ref_table, {ref_col: val})
            _repair_row(candidate, ref_table, parent_row, case_study, _seen)


def repair_candidate(candidate, case_study):
    for table, rows in list(candidate.as_dict().items()):
        for row in list(rows):
            _repair_row(candidate, table, row, case_study)


def _apply_field_mutation(node, value, candidate, focal, case_study=None):
    kind = node['kind']
    touched = []
    if kind == 'schema_column':
        table, column = node['table'], node['column']
        row = focal.setdefault(table.upper(), {})
        if table.upper() not in {t for t in candidate.as_dict()} or row not in candidate.rows(table):
            candidate.add_row(table, row)
        constant = node.get('compared_to_named_constant')
        if constant is not None and isinstance(value, bool):
            if value:
                row[column] = constant['value']
            else:
                fresh = _fresh_key_value(candidate, table, column)
                while fresh == constant['value']:
                    fresh += 1
                row[column] = fresh
        else:
            row[column] = value
        touched.append((table, row))
    elif kind == 'serialized_field':
        table, column, key = node['table'], node['column'], node['key']
        row = focal.setdefault(table.upper(), {})
        if row not in candidate.rows(table):
            candidate.add_row(table, row)
        row.setdefault(column, {})[key] = value
        touched.append((table, row))
    elif kind == 'derived_case':
        table, column = node['table'], node['column']
        real_value = next((k for k, v in node['cases'] if v == value), None)
        if real_value is None:
            raise FitnessEvaluationError(
                f"{value!r} is not a reachable output of this derived_case's CASE_MAP "
                f"({node['cases']}) -- refusing to write an unmappable value")
        row = focal.setdefault(table.upper(), {})
        if row not in candidate.rows(table):
            candidate.add_row(table, row)
        row[column] = real_value
        touched.append((table, row))
    elif kind == 'null_check' and node.get('key'):
        table, column, key = node['table'], node['column'], node['key']
        row = focal.setdefault(table.upper(), {})
        if row not in candidate.rows(table):
            candidate.add_row(table, row)
        blob = row.setdefault(column, {})
        set_it = (not value) if node.get('negate') else value
        if not set_it:
            blob.pop(key, None)
        else:
            blob[key] = 1
        touched.append((table, row))
    elif kind == 'null_check':
        table, column = node['table'], node['column']
        row = focal.setdefault(table.upper(), {})
        if row not in candidate.rows(table):
            candidate.add_row(table, row)
        if node.get('negate'):
            value = not value
        if not value:
            row[column] = ('' if case_study is not None and _column_is_not_null(case_study, table, column)
                            else None)
        elif row.get(column) is None:
            col_type = _column_type(case_study, table, column) if case_study is not None else None
            row[column] = _placeholder_for_column_type(col_type) if case_study is not None else 1
        touched.append((table, row))
    elif kind == 'any_not_null':
        for c in node['columns']:
            row = focal.setdefault(c['table'].upper(), {})
            if row not in candidate.rows(c['table']):
                candidate.add_row(c['table'], row)
            row[c['column']] = None
            touched.append((c['table'], row))
        if value:
            first = node['columns'][0]
            focal[first['table'].upper()][first['column']] = 1
    elif kind in ('join_lookup', 'join_null_check'):
        local_row = focal.setdefault(node['via']['local_table'].upper(), {})
        if local_row not in candidate.rows(node['via']['local_table']):
            candidate.add_row(node['via']['local_table'], local_row)
        touched.append((node['via']['local_table'], local_row))
        if kind == 'join_null_check' and not value:
            local_row[node['via']['local_column']] = None
            return touched
        target_row = focal.setdefault(node['result_table'].upper(), {})
        if target_row not in candidate.rows(node['result_table']):
            candidate.add_row(node['result_table'], target_row)
        key = target_row.setdefault(node['result_column'], 1)
        local_row[node['via']['local_column']] = key
        if kind == 'join_lookup':
            target_row[node['result_column']] = value
        touched.append((node['result_table'], target_row))
    elif kind == 'regex_match':
        vrow = focal.setdefault(node['value_column']['table'].upper(), {})
        prow = focal.setdefault(node['pattern_column']['table'].upper(), {})
        if vrow not in candidate.rows(node['value_column']['table']):
            candidate.add_row(node['value_column']['table'], vrow)
        if prow not in candidate.rows(node['pattern_column']['table']):
            candidate.add_row(node['pattern_column']['table'], prow)
        prow.setdefault(node['pattern_column']['column'], '.*')
        vrow[node['value_column']['column']] = 'MATCH' if value else ''
        touched.append((node['value_column']['table'], vrow))
        touched.append((node['pattern_column']['table'], prow))
    return touched


def _row_from_filter_conjuncts(filter_text, scenario, owner_id=None, candidate=None, focal=None, self_table=None,
                                table=None):
    self_row = focal.get(self_table.upper()) if (focal and self_table) else None
    row = {} if owner_id is None else {_OWNER_KEY: owner_id}
    for conjunct in _top_level_and_conjuncts(filter_text):
        c_stripped = conjunct.strip()
        m_sub = _IN_SUBQUERY_RE.match(c_stripped)
        if m_sub:
            outer_val = _construct_subquery_parent(m_sub, candidate, focal, scenario, self_table, owner_id=owner_id)
            if outer_val is not None:
                row[m_sub.group(1)] = outer_val
            continue
        nn = re.match(r'^\s*(?:[\w]+\.)?(\w+)\s+IS\s+NOT\s+NULL\s*$', c_stripped, re.I)
        if nn:
            row[nn.group(1)] = 1
            continue
        if re.match(r'^\s*(?:[\w]+\.)?(\w+)\s+IS\s+NULL\s*$', c_stripped, re.I):
            continue
        cm = re.match(r'^\s*(?:[\w]+\.)?(\w+)\s*=\s*(.+?)\s*$', c_stripped)
        if not cm:
            continue
        col, raw_val = cm.group(1), cm.group(2).strip()
        if col.isdigit():
            continue
        colon_m = re.fullmatch(r':([A-Za-z_]\w*)', raw_val)
        if colon_m:
            found = False
            if self_row is not None:
                for key in (colon_m.group(1), colon_m.group(1).upper(), colon_m.group(1).lower()):
                    if key in self_row:
                        row[col] = self_row[key]
                        found = True
                        break
            if not found and candidate is not None and focal is not None and self_table is not None:
                fresh = _fresh_key_value(candidate, self_table, colon_m.group(1))
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
        ph = re.fullmatch(r'<([^>]+)>', raw_val)
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


def _apply_row_count_mutation(node, value, candidate, scenario, current, focal=None, owner_id=None):
    if node['kind'] == 'exists':
        table = (node.get('candidate_tables') or [None])[0]
        if table is None:
            return []
        self_table = node.get('self_table') or table
        self_row = focal.get(self_table.upper()) if focal else None
        predicate, _skipped = _mechanical_filter_predicate(node.get('filter_text'), scenario, candidate, self_row)
        owned = _owned_rows(candidate, table, owner_id)
        if value and not any(predicate(r) for r in owned):
            row = _row_from_filter_conjuncts(node.get('filter_text'), scenario, owner_id,
                                              candidate, focal, self_table, table=table)
            row = candidate.add_row(table, row)
            return [(table, row)]
        elif not value:
            real_rows = candidate.rows(table)
            if owner_id is None:
                real_rows[:] = [r for r in real_rows if not predicate(r)]
            else:
                real_rows[:] = [r for r in real_rows
                                 if not (r.get(_OWNER_KEY) == owner_id and predicate(r))]
        return []
    tables = [t.strip() for t in node['table'].split(',')]
    table = tables[0]
    self_table = node.get('self_table') or table
    self_row = focal.get(self_table.upper()) if focal else None
    predicate, _skipped = _mechanical_filter_predicate(node.get('filter_text'), scenario, candidate, self_row)
    n = int(round(value)) - int(round(current or 0))
    value_table, value_col = (node['value_column'].split('.') if node.get('value_column') else (None, None))
    joined_value_table = value_table and value_table.upper() != table.upper()
    if n > 0:
        touched = []
        for _ in range(n):
            row = _row_from_filter_conjuncts(node.get('filter_text'), scenario, owner_id,
                                              candidate, focal, self_table, table=table)
            if value_table and not joined_value_table:
                row[value_col] = 1
            elif joined_value_table:
                join_col = f'{value_table.upper()}_ID'
                join_val = _fresh_key_value(candidate, value_table, join_col)
                row[join_col] = join_val
                vrow = {join_col: join_val, value_col: 1}
                if owner_id is not None:
                    vrow[_OWNER_KEY] = owner_id
                candidate.add_row(value_table, vrow)
                touched.append((value_table, vrow))
            candidate.add_row(table, row)
            touched.append((table, row))
        return touched
    elif n < 0:
        rows = _owned_rows(candidate, table, owner_id)
        matching = [r for r in rows if predicate(r)] or rows
        real_rows = candidate.rows(table)
        for row in matching[:min(-n, len(matching))]:
            real_rows.remove(row)
    return []


def apply_mutation(record, candidate, focal, scenario, var_name, node, value, current, owner_id=None):
    kind = node.get('kind')
    if kind in FIELD_LEAF_KINDS:
        touched = _apply_field_mutation(node, value, candidate, focal, case_study=record.get('case_study'))
    elif kind in AGGREGATE_LEAF_KINDS:
        touched = _apply_row_count_mutation(node, value, candidate, scenario, current, focal, owner_id=owner_id)
    elif kind == 'not_persisted':
        scenario[var_name] = value
        touched = []
    elif kind == 'raw_sql_boolean':
        raise FitnessEvaluationError(
            f"{var_name!r} (raw_sql_boolean) has no automatic mutation -- too bespoke a "
            f"compound fact to edit generically; needs a fact-specific operator")
    else:
        raise FitnessEvaluationError(f"mutation has no handling for resolution kind {kind!r}")
    for table, row in touched or []:
        _repair_row(candidate, table, row, record['case_study'])


def _leaf_variables(record):
    out = []

    def walk(var_name, node):
        kind = node.get('kind')
        if kind in ('literal', 'literal_via_upstream_branch') or kind in (
                'schema_gap', 'code_external', 'unresolved', 'chained_decision_output'):
            return
        if kind == 'substituted_decision':
            for fv, sub in node.get('free_variable_resolutions', {}).items():
                walk(fv, sub)
            return
        out.append((var_name, node))

    for var, node in record.get('variable_resolution', {}).items():
        walk(var, node)
    return out


