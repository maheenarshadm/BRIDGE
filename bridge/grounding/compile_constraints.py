"""Compilation of DMN decision models into search objectives (Phase 1).

Every rule of a FIRST or UNIQUE decision table becomes a search objective.
The rule's condition is built from the FEEL input entries, the hit-policy
context records the preceding rules that must not match, and every decision
input is resolved against the database schema using the case study's
grounding.csv: a column, a null check, an aggregate over related rows, the
existence of related rows, a value reached through a join, the output of an
upstream decision, or a value that is not stored in the database. An input
that depends on an upstream decision is grounded through each of that
decision's rules, so one rule can yield several objectives.

`compile_case_studies` compiles the case studies and writes the objectives
used in the evaluation (listed in each case study's config.json) to
case_studies/compiled_constraints.json."""
import os
import csv
import json
import re
from collections import deque
from xml.etree import ElementTree as ET

from bridge import paths
from bridge.grounding.feel_parser import parse_unary_test, parse_expression, UnsupportedFeelConstruct
from bridge.grounding.literal_expression_overrides import get_override as get_literal_expression_override
from bridge.grounding.aggregate_self_table import get_self_table
from bridge.grounding.aggregate_self_exclusions import get_exclude_self_column
from bridge.grounding.aggregate_filter_overrides import get_filter_text_override
from bridge.grounding.not_persisted_reclassification import get_reclassification as get_not_persisted_reclassification
from bridge.grounding.condition_column_overrides import get_column_override

DMN_NS = "https://www.omg.org/spec/DMN/20191111/MODEL/"


def q(tag):
    return f'{{{DMN_NS}}}{tag}'


CASE_STUDY_DMN_DIRS = {cs: paths.dmn_dir(cs) for cs in paths.CASE_STUDIES}

CASE_STUDY_SCHEMA_JSON = paths.SCHEMA_JSON_PATHS

PAIR_RE = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)\b')


def true_pairs(text):
    return {(t.lower(), c.lower()) for t, c in PAIR_RE.findall(text or '')}


class Decision:
    def __init__(self, id_, name, dmn_file):
        self.id = id_
        self.name = name
        self.dmn_file = dmn_file
        self.hit_policy = None
        self.inputs = []
        self.outputs = []
        self.rules = []
        self.required_decision_ids = []
        self.is_literal = False
        self.literal_text = None
        self.own_variable = None

    @property
    def is_table(self):
        return not self.is_literal


def parse_dmn_file(path):
    fname = os.path.basename(path)
    root = ET.parse(path).getroot()
    decisions = []
    for dec_el in root.findall(q('decision')):
        d = Decision(dec_el.get('id'), dec_el.get('name'), fname)
        for ir in dec_el.findall(q('informationRequirement')):
            req = ir.find(q('requiredDecision'))
            if req is not None:
                d.required_decision_ids.append(req.get('href', '').lstrip('#'))

        var_el = dec_el.find(q('variable'))
        if var_el is not None:
            d.own_variable = var_el.get('name')

        dt = dec_el.find(q('decisionTable'))
        if dt is None:
            lit = dec_el.find(q('literalExpression'))
            d.is_literal = True
            text_el = lit.find(q('text')) if lit is not None else None
            d.literal_text = text_el.text if text_el is not None else ''
            decisions.append(d)
            continue

        d.hit_policy = dt.get('hitPolicy', 'UNIQUE')
        for inp in dt.findall(q('input')):
            expr_el = inp.find(q('inputExpression'))
            text_el = expr_el.find(q('text')) if expr_el is not None else None
            typeref = expr_el.get('typeRef', '') if expr_el is not None else ''
            d.inputs.append((text_el.text if text_el is not None else '', typeref))
        for outp in dt.findall(q('output')):
            d.outputs.append((outp.get('name'), outp.get('typeRef', '')))

        for rule_el in dt.findall(q('rule')):
            desc_el = rule_el.find(q('description'))
            input_texts = []
            for entry in rule_el.findall(q('inputEntry')):
                t = entry.find(q('text'))
                input_texts.append(t.text if t is not None else '')
            output_texts = []
            for entry in rule_el.findall(q('outputEntry')):
                t = entry.find(q('text'))
                output_texts.append(t.text if t is not None else '')
            d.rules.append({
                'id': rule_el.get('id'),
                'description': desc_el.text if desc_el is not None else '',
                'input_texts': input_texts,
                'output_texts': output_texts,
            })
        decisions.append(d)
    return decisions


def load_case_study_decisions(cs):
    by_id, by_name = {}, {}
    for path in sorted(__import__('glob').glob(os.path.join(CASE_STUDY_DMN_DIRS[cs], '*.dmn'))):
        for d in parse_dmn_file(path):
            by_id[d.id] = d
            by_name[d.name] = d
    return by_id, by_name


NORMALIZE_MAPPING_TYPE = [
    ('schema gap', 'schema_gap'),
    ('not-persisted', 'not_persisted'),
    ('not persisted', 'not_persisted'),
    ('serialized-field', 'serialized_field'),
    ('serialized field', 'serialized_field'),
    ('derived-aggregate', 'derived'),
    ('derived - aggregate', 'derived'),
    ('derived', 'derived'),
    ('direct', 'direct'),
]

_SERIALIZED_FIELD_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)'
    r'\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(-?\d+(?:\.\d+)?|\'[^\']*\'))?\s*\]\s*$')


def normalize_mapping_type(raw):
    t = (raw or '').strip().lower()
    for needle, bucket in NORMALIZE_MAPPING_TYPE:
        if needle in t:
            return bucket
    return 'unresolved'


def load_case_study_ground_truth(cs):
    cfg = paths.config(cs)['grounding']
    out = {}
    with open(paths.grounding_csv(cs), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            decision_name = row[cfg['decision_col']]
            variable_name = row[cfg['variable_col']]
            io = row.get('io', 'input')
            schema_field = row.get(cfg['schema_col'], '')
            key = (decision_name, variable_name, io)
            out[key] = {
                'bucket': normalize_mapping_type(row.get('mapping_type', '')),
                'schema_pairs': sorted(true_pairs(schema_field)),
                'notes': row.get(cfg['notes_col'], ''),
                'raw_schema_field': schema_field,
            }
            if 'io' not in row:
                out[(decision_name, variable_name, 'output')] = out[key]
    return out


AGGREGATE_RECIPE_RE = re.compile(
    r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', re.I)
AGGREGATE_RECIPE_REVERSED_RE = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(\s*(COUNT|SUM|AVG|MIN|MAX)\b', re.I)
AGGREGATE_RECIPE_DOTTED_RE = re.compile(
    r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\)', re.I)
AGGREGATE_FROM_RE = re.compile(r'\bFROM\s+(.+?)\s+WHERE\b', re.I | re.S)
AGGREGATE_FOR_CORRELATION_RE = re.compile(
    r'^\s*for\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*\+\s*[A-Za-z_][A-Za-z0-9_]*)*)', re.I)
AGGREGATE_STAR_FROM_RE = re.compile(
    r'\bCOUNT\s*\(\s*\*\s*\)\s*FROM\s+([A-Za-z_][A-Za-z0-9_]*)', re.I)
AGGREGATE_GROUP_BY_RE = re.compile(r'\bGROUP\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)', re.I)
_AGGREGATE_BARE_QUESTION_MARK_RE = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\?')


def _try_extract_aggregate_recipe(text):
    if not text:
        return None
    value_column = None
    m = AGGREGATE_RECIPE_DOTTED_RE.search(text)
    if m:
        aggregate, value_table, col = m.group(1).upper(), m.group(2), m.group(3)
        value_column = f'{value_table}.{col}'
        from_m = AGGREGATE_FROM_RE.search(text, m.end())
        table = from_m.group(1).strip() if from_m else value_table
    else:
        m = AGGREGATE_RECIPE_RE.search(text)
        if m:
            aggregate, table = m.group(1).upper(), m.group(2)
        else:
            m = AGGREGATE_RECIPE_REVERSED_RE.search(text)
            if m:
                table, aggregate = m.group(1), m.group(2).upper()
            else:
                m = AGGREGATE_STAR_FROM_RE.search(text)
                if not m:
                    return None
                aggregate, table = 'COUNT', m.group(1)
    where_idx = text.upper().find('WHERE', m.end())
    filter_text = text[where_idx + len('WHERE'):].rstrip(') ').strip() if where_idx != -1 else None
    if filter_text is not None:
        comment_idx = filter_text.find('--')
        if comment_idx != -1:
            filter_text = filter_text[:comment_idx].rstrip()
        filter_text = _AGGREGATE_BARE_QUESTION_MARK_RE.sub(r'\1 = :\1', filter_text)
    if filter_text is None:
        for_m = AGGREGATE_FOR_CORRELATION_RE.match(text[m.end():])
        if for_m:
            columns = [c.strip() for c in for_m.group(1).split('+')]
            filter_text = ' AND '.join(f'{c} = :{c}' for c in columns)
        else:
            group_by_m = AGGREGATE_GROUP_BY_RE.search(text, m.end())
            if group_by_m:
                col = group_by_m.group(1)
                filter_text = f'{col} = :{col}'
    node = {'kind': 'derived_aggregate', 'aggregate': aggregate, 'table': table,
            'filter_text': filter_text, 'source_text': text}
    if value_column:
        node['value_column'] = value_column
    return node


_EXISTENCE_WORDS = re.compile(
    r'\b(IS NOT NULL|IS NULL|existence check|null-check|null check)\b', re.I)
_ANY_OF_WORDS = re.compile(r'\b(OR of|either|any of|either populated)\b', re.I)
_NEGATED_NULL_CHECK_WORDS = re.compile(r'\bIS NULL\b', re.I)
_NON_NEGATED_NULL_CHECK_WORDS = re.compile(r'\bIS NOT NULL\b', re.I)


def _null_check_is_negated(notes):
    notes = notes or ''
    return bool(_NEGATED_NULL_CHECK_WORDS.search(notes)) and not _NON_NEGATED_NULL_CHECK_WORDS.search(notes)
_EXISTS_ROW_WORDS = re.compile(
    r'\bexistence of\b|\(existence\)|\bEXISTS\(|self-join', re.I)
_EXISTS_FILTER_TOKEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*(?:(-?\d+)|'([^']*)'))?$")
_JOINED_VIA_RE = re.compile(r'joined via ([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)', re.I)
_CONSTANT_RE = re.compile(r'(?:constant|=)\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(-?\d+)')
_CROSS_REF_CONSTANT_RE = re.compile(r'compared (?:to )?the ([A-Za-z_][A-Za-z0-9_.]*) constant', re.I)
_KNOWN_NAMED_CONSTANTS = {
    ('jBilling', 'STATUS_DELETED'): 8,
}
_RAW_SQL_RE = re.compile(
    r'RAW_SQL:\s*(.+?)\s*TABLES:\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*$',
    re.I | re.S)


def _try_extract_raw_sql_boolean(text):
    if not text:
        return None
    m = _RAW_SQL_RE.search(text)
    if not m:
        return None
    tables = [t.strip() for t in m.group(2).split(',') if t.strip()]
    return {'kind': 'raw_sql_boolean', 'sql_template': m.group(1).strip(), 'tables': tables}


_EXISTS_SELF_JOIN_RE = re.compile(
    r'EXISTS\s*\(\s*SELECT\s+1\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+(.+?)\)\s*$',
    re.I | re.S)
_EXISTS_SELF_JOIN_CONJUNCT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*(<>|!=|=)\s*"
    r"(?:this\.([A-Za-z_][A-Za-z0-9_]*)|'([^']*)'|([A-Za-z_]+|-?\d+))\s*$",
    re.I)


def _try_extract_self_join_exists_recipe(text):
    if not text:
        return None
    m = _EXISTS_SELF_JOIN_RE.search(text)
    if not m:
        return None
    table, alias, where = m.group(1), m.group(2), m.group(3)
    filter_parts = []
    columns = []
    for conjunct in re.split(r'\bAND\b', where, flags=re.I):
        cm = _EXISTS_SELF_JOIN_CONJUNCT_RE.match(conjunct.strip())
        if not cm:
            return None
        alias_prefix, col, op, this_col, str_lit, other = cm.groups()
        if alias_prefix.lower() != alias.lower():
            return None
        columns.append(col)
        norm_op = '!=' if op in ('<>', '!=') else '='
        if this_col is not None:
            filter_parts.append(f'{col} {norm_op} :{this_col}')
        elif str_lit is not None:
            filter_parts.append(f"{col} {norm_op} '{str_lit}'")
        else:
            filter_parts.append(f'{col} {norm_op} {other}')
    return {'kind': 'exists', 'candidate_tables': [table],
            'candidate_columns': [{'table': table, 'column': c} for c in columns],
            'filter_text': ' AND '.join(filter_parts)}


_EXISTS_CHILD_ROW_PROSE_RE = re.compile(
    r'EXISTS\s+child\s+([A-Za-z_][A-Za-z0-9_]*)\s+rows?\s+with\s+(.+?)\s*(?:\([^)]*\))?\s*$',
    re.I)
_EXISTS_CHILD_ROW_CONJUNCT_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<>|!=|=)\s*this\.([A-Za-z_][A-Za-z0-9_]*)\s*$', re.I)


def _try_extract_self_join_exists_prose_recipe(text):
    if not text:
        return None
    m = _EXISTS_CHILD_ROW_PROSE_RE.search(text)
    if not m:
        return None
    table, where = m.group(1), m.group(2)
    filter_parts = []
    columns = []
    for conjunct in re.split(r'\bAND\b', where, flags=re.I):
        cm = _EXISTS_CHILD_ROW_CONJUNCT_RE.match(conjunct.strip())
        if not cm:
            return None
        col, op, this_col = cm.groups()
        columns.append(col)
        norm_op = '!=' if op in ('<>', '!=') else '='
        filter_parts.append(f'{col} {norm_op} :{this_col}')
    return {'kind': 'exists', 'candidate_tables': [table],
            'candidate_columns': [{'table': table, 'column': c} for c in columns],
            'filter_text': ' AND '.join(filter_parts)}


_ENUM_EQUALITY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']+)'\s*$")

_KNOWN_ENUM_DOMAINS = {
    ('concept_name', 'concept_name_type'): ['FULLY_SPECIFIED', 'SHORT', 'INDEX_TERM', None],
}


def _try_extract_enum_equality(text, pairs):
    if not text or len(pairs) != 1:
        return None
    m = _ENUM_EQUALITY_RE.match(text.strip())
    if not m:
        return None
    column, literal = m.group(1), m.group(2)
    table, pair_column = pairs[0]
    if pair_column.lower() != column.lower():
        return None
    domain = _KNOWN_ENUM_DOMAINS.get((table.lower(), column.lower()))
    if domain is None:
        return None
    return {'kind': 'derived_case', 'table': table, 'column': column,
            'cases': [[v, v == literal] for v in domain]}


_CASE_MAP_RE = re.compile(
    r'CASE_MAP:\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$',
    re.I | re.S)
_CASE_MAP_ENTRY_RE = re.compile(r"'([^']*)'\s*->\s*'([^']*)'")


def _try_extract_case_map(text):
    if not text:
        return None
    m = _CASE_MAP_RE.search(text)
    if not m:
        return None
    table, column, rest = m.group(1), m.group(2), m.group(3)
    cases = _CASE_MAP_ENTRY_RE.findall(rest)
    if not cases:
        return None
    return {'kind': 'derived_case', 'table': table, 'column': column,
            'cases': [[k, v] for k, v in cases]}


_ELAPSED_VS_TODAY_RE = re.compile(
    r'derived from ([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*) vs current date', re.I)
_DAYS_PER_YEAR = 365


def _try_extract_elapsed_years_since(var_name, row):
    if 'year' not in var_name.lower():
        return None
    for text in (row['raw_schema_field'] or '', row['notes'] or ''):
        m = _ELAPSED_VS_TODAY_RE.search(text)
        if m:
            table, column = m.group(1).lower(), m.group(2).lower()
            date_var = f'{var_name}__{column}'
            return {
                'kind': 'substituted_decision',
                'substituted_from': f'elapsed years since {table}.{column} (vs today())',
                'expression': {
                    'op': '/',
                    'left': {'op': '-', 'left': {'kind': 'call', 'name': 'today', 'args': []},
                             'right': {'kind': 'variable', 'ref': date_var}},
                    'right': {'kind': 'literal', 'value': _DAYS_PER_YEAR, 'type': 'number'},
                },
                'free_variable_resolutions': {
                    date_var: {'kind': 'schema_column', 'table': table, 'column': column},
                },
                'notes': text,
            }
    return None


def classify_derived(row, cs=None):
    notes, raw = row['notes'] or '', row['raw_schema_field'] or ''
    pairs = row['schema_pairs']

    agg = _try_extract_aggregate_recipe(notes) or _try_extract_aggregate_recipe(raw)
    if agg:
        return agg

    case_map = _try_extract_case_map(notes) or _try_extract_case_map(raw)
    if case_map:
        case_map['notes'] = notes
        return case_map

    enum_eq = _try_extract_enum_equality(notes, pairs) or _try_extract_enum_equality(raw, pairs)
    if enum_eq:
        enum_eq['notes'] = notes
        return enum_eq

    raw_sql = _try_extract_raw_sql_boolean(notes) or _try_extract_raw_sql_boolean(raw)
    if raw_sql:
        raw_sql['notes'] = notes
        return raw_sql

    self_join = _try_extract_self_join_exists_recipe(notes) or _try_extract_self_join_exists_recipe(raw)
    if self_join:
        self_join['notes'] = notes
        return self_join

    self_join_prose = (_try_extract_self_join_exists_prose_recipe(notes)
                        or _try_extract_self_join_exists_prose_recipe(raw))
    if self_join_prose:
        self_join_prose['notes'] = notes
        return self_join_prose

    if pairs:
        m = _JOINED_VIA_RE.search(notes)
        if m:
            local_table, local_column = m.group(1).split('.')
            target_table, target_column = pairs[-1]
            node = {'kind': 'join_lookup',
                    'via': {'local_table': local_table, 'local_column': local_column},
                    'result_table': target_table, 'result_column': target_column, 'notes': notes}
            if _EXISTENCE_WORDS.search(notes):
                node['kind'] = 'join_null_check'
            return node

    if pairs and _EXISTENCE_WORDS.search(notes) and not _ANY_OF_WORDS.search(notes):
        table, column = pairs[0]
        node = {'kind': 'null_check', 'table': table, 'column': column, 'notes': notes}
        if _null_check_is_negated(notes):
            node['negate'] = True
        if len(pairs) > 1:
            node['also_valid_in'] = pairs[1:]
        return node

    if len(pairs) == 2 and re.search(r'\bregex\b', notes, re.I):
        (t1, c1), (t2, c2) = pairs
        return {'kind': 'regex_match', 'value_column': {'table': t1, 'column': c1},
                'pattern_column': {'table': t2, 'column': c2}, 'notes': notes}

    if len(pairs) == 1:
        table, column = pairs[0]
        m = _CONSTANT_RE.search(notes)
        node = {'kind': 'schema_column', 'table': table, 'column': column}
        if m:
            node['compared_to_named_constant'] = {'name': m.group(1), 'value': int(m.group(2))}
        else:
            cross_ref = _CROSS_REF_CONSTANT_RE.search(notes)
            if cross_ref:
                known_value = _KNOWN_NAMED_CONSTANTS.get((cs, cross_ref.group(1)))
                if known_value is not None:
                    node['compared_to_named_constant'] = {'name': cross_ref.group(1), 'value': known_value}
        node['notes'] = notes
        return node

    if len(pairs) >= 2:
        if _ANY_OF_WORDS.search(notes):
            return {'kind': 'any_not_null', 'columns': [{'table': t, 'column': c} for t, c in pairs],
                    'notes': notes}
        if _EXISTS_ROW_WORDS.search(notes):
            return {'kind': 'exists', 'candidate_tables': sorted({t for t, _ in pairs}),
                     'candidate_columns': [{'table': t, 'column': c} for t, c in pairs], 'notes': notes}
        table, column = pairs[0]
        node = {'kind': 'schema_column', 'table': table, 'column': column, 'also_valid_in': pairs[1:]}
        m = _CONSTANT_RE.search(notes)
        if m:
            node['compared_to_named_constant'] = {'name': m.group(1), 'value': int(m.group(2))}
        else:
            cross_ref = _CROSS_REF_CONSTANT_RE.search(notes)
            if cross_ref:
                known_value = _KNOWN_NAMED_CONSTANTS.get((cs, cross_ref.group(1)))
                if known_value is not None:
                    node['compared_to_named_constant'] = {'name': cross_ref.group(1), 'value': known_value}
        node['notes'] = notes
        return node

    if _EXISTS_ROW_WORDS.search(notes) or _EXISTS_ROW_WORDS.search(raw):
        m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([^)]*?)\s*\)', raw)
        if m:
            table, inner = m.group(1), m.group(2)
            if inner.strip().lower() != 'existence':
                tokens = [t.strip() for t in inner.split(',') if t.strip()]
                conjuncts, columns, parse_ok = [], [], bool(tokens)
                for t in tokens:
                    tm = _EXISTS_FILTER_TOKEN_RE.match(t)
                    if not tm:
                        parse_ok = False
                        break
                    col, int_lit, str_lit = tm.group(1), tm.group(2), tm.group(3)
                    columns.append(col)
                    if int_lit is not None:
                        conjuncts.append(f"{col} = {int_lit}")
                    elif str_lit is not None:
                        conjuncts.append(f"{col} = '{str_lit}'")
                    else:
                        conjuncts.append(f"{col} = <{col}>")
                if parse_ok:
                    return {'kind': 'exists', 'candidate_tables': [table],
                             'candidate_columns': [{'table': table, 'column': c} for c in columns],
                             'filter_text': ' AND '.join(conjuncts), 'notes': notes, 'raw_schema_field': raw}
            return {'kind': 'exists', 'candidate_tables': [table],
                     'candidate_columns': [], 'notes': notes, 'raw_schema_field': raw}

    if raw.strip().lower() == 'n/a' and not pairs:
        return {'kind': 'code_external', 'notes': notes,
                'reason': 'no schema field was ever recorded for this fact -- genuinely computed by '
                          'application code (a Java constant, a UI-only transient value, a runtime '
                          'calculation over other derived facts), not read from any table at all'}

    return None


def _apply_aggregate_overrides(cs, var_name, node):
    if node.get('kind') != 'derived_aggregate':
        return node
    filter_override = get_filter_text_override(cs, var_name)
    if filter_override:
        node['filter_text'] = filter_override
    if re.search(r'=\s*:[A-Za-z_]\w*', node.get('filter_text') or ''):
        exclude_col = get_exclude_self_column(cs, var_name)
        if exclude_col:
            node['filter_text'] += f' AND {exclude_col} != :{exclude_col}'
    if (re.search(r'\bself\b', node.get('filter_text') or '', re.I)
            or re.search(r'=\s*:[A-Za-z_]\w*', node.get('filter_text') or '')):
        self_table = get_self_table(cs, var_name)
        if self_table:
            node['self_table'] = self_table
    return node


def resolve_variable(cs, gt, decision_name, var_name, io='input'):
    row = gt.get((decision_name, var_name, io)) or gt.get((decision_name, var_name, 'input')) \
        or gt.get((decision_name, var_name, 'output'))
    if row is None:
        return {'kind': 'unresolved', 'reason': 'not found in ground-truth mapping CSV',
                'decision_name': decision_name, 'variable': var_name}
    bucket = row['bucket']
    if bucket == 'not_persisted':
        reclassified = get_not_persisted_reclassification(cs, var_name)
        if reclassified:
            return reclassified
        return {'kind': 'not_persisted'}
    if bucket == 'schema_gap':
        return {'kind': 'schema_gap', 'notes': row['notes'], 'hinted_pairs': row['schema_pairs']}
    if bucket == 'serialized_field':
        m = _SERIALIZED_FIELD_RE.match(row['raw_schema_field'] or '')
        if not m:
            return {'kind': 'unresolved',
                    'reason': "labeled 'serialized-field' but schema field isn't "
                              "'table.column[key]' or 'table.column[key=default]'",
                    'raw_schema_field': row['raw_schema_field']}
        table, column, key, default_text = m.groups()
        if _EXISTENCE_WORDS.search(row['notes'] or ''):
            node = {'kind': 'null_check', 'table': table, 'column': column, 'key': key, 'notes': row['notes']}
            if _null_check_is_negated(row['notes']):
                node['negate'] = True
            return node
        node = {'kind': 'serialized_field', 'table': table, 'column': column, 'key': key, 'notes': row['notes']}
        if default_text is not None:
            if default_text.startswith("'"):
                node['default'] = default_text[1:-1]
            elif '.' in default_text:
                node['default'] = float(default_text)
            else:
                node['default'] = int(default_text)
        return node
    if bucket == 'direct':
        if row['schema_pairs']:
            table, column = row['schema_pairs'][0]
            node = {'kind': 'schema_column', 'table': table, 'column': column}
            if len(row['schema_pairs']) > 1:
                node['also_valid_in'] = row['schema_pairs'][1:]
            return node
        recipe = _try_extract_aggregate_recipe(row['raw_schema_field'])
        if recipe:
            recipe['notes'] = f"ground truth labeled this 'direct'; text describes an aggregate, treated as derived_aggregate"
            return _apply_aggregate_overrides(cs, var_name, recipe)
        return {'kind': 'unresolved', 'reason': "labeled 'direct' but no table.column parsed from its schema field",
                'raw_schema_field': row['raw_schema_field']}
    if bucket == 'derived':
        elapsed = _try_extract_elapsed_years_since(var_name, row)
        if elapsed:
            return elapsed
        classified = classify_derived(row, cs)
        if classified:
            return _apply_aggregate_overrides(cs, var_name, classified)
        return {'kind': 'derived', 'notes': row['notes'], 'table_hints': row['schema_pairs']}
    return {'kind': 'unresolved', 'reason': f'unrecognized ground-truth mapping_type bucket {bucket!r}'}


def find_all_variable_refs(node, out=None):
    if out is None:
        out = []
    if not isinstance(node, dict):
        return out
    if node.get('kind') == 'variable':
        out.append(node['ref'])
        return out
    for key in ('left', 'right', 'low', 'high', 'clause', 'cond', 'then', 'else'):
        if key in node:
            find_all_variable_refs(node[key], out)
    for key in ('clauses', 'values', 'args'):
        if key in node:
            for child in node[key]:
                find_all_variable_refs(child, out)
    return out


def resolve_and_substitute(cs, gt, decision, var_name, by_id, by_name, seen=None):
    seen = seen or set()
    for req_id in decision.required_decision_ids:
        upstream = by_id.get(req_id)
        if upstream is None:
            continue
        produced_name = upstream.own_variable if upstream.is_literal else None
        produced_names = [produced_name] if produced_name else [o for o, _ in upstream.outputs]
        if var_name not in produced_names:
            continue
        if upstream.is_literal:
            cache_key = (upstream.name, var_name)
            if cache_key in seen:
                return {'kind': 'unresolved', 'reason': 'circular DRD substitution detected',
                         'decision_chain': sorted(seen)}
            override = get_literal_expression_override(cs, upstream.name)
            if override is not None:
                return {'kind': 'substituted_decision', 'substituted_from': upstream.name,
                        'expression': override['expression'],
                        'free_variable_resolutions': override['free_variable_resolutions']}
            try:
                expr = parse_expression(upstream.literal_text)
            except UnsupportedFeelConstruct as e:
                return {'kind': 'unresolved', 'reason': f'upstream literal expression not parseable: {e}',
                         'upstream_decision': upstream.name}
            free_vars = sorted(set(find_all_variable_refs(expr)))
            resolutions = {}
            for fv in free_vars:
                resolutions[fv] = resolve_and_substitute(
                    cs, gt, upstream, fv, by_id, by_name, seen | {cache_key})
            return {'kind': 'substituted_decision', 'substituted_from': upstream.name,
                    'expression': expr, 'free_variable_resolutions': resolutions}
        else:
            return {'kind': 'chained_decision_output', 'from_decision': upstream.name,
                    'note': ('requires the upstream decision table to have already selected a '
                             'branch of its own -- not inlined (scope boundary: a decision table\'s '
                             'output depends on which of its own rules fires, which is not a single '
                             'closed-form expression to substitute the way a literal-expression '
                             'decision\'s formula is)')}

    if decision.is_literal or var_name in [v for v, _ in decision.inputs]:
        return resolve_variable(cs, gt, decision.name, var_name, 'input')

    return {'kind': 'unresolved',
            'reason': 'not a declared input of this decision, and not produced by any DRD-linked upstream decision'}


def load_schema_fk_graph(cs):
    with open(CASE_STUDY_SCHEMA_JSON[cs], encoding='utf-8') as f:
        data = json.load(f)
    return {table: set(info.get('fks') or []) for table, info in data.items()}, set(data.keys())


def fk_closure(seed_tables, fk_graph, all_tables, max_tables=300):
    lower_index = {t.lower(): t for t in all_tables}
    resolved_seeds = {lower_index[t.lower()] for t in seed_tables if t.lower() in lower_index}
    visited = set(resolved_seeds)
    queue = deque(resolved_seeds)
    while queue and len(visited) < max_tables:
        t = queue.popleft()
        for target in fk_graph.get(t, ()):
            key = target.lower()
            real = lower_index.get(key)
            if real and real not in visited:
                visited.add(real)
                queue.append(real)
    return sorted(visited)


_DECISION_SUBJECT_PLACEHOLDER_SOURCES = {
    ('Spree', 'Promotion Customer Group Eligibility', 'promotion_id'): 'spree_order_promotions',
    ('jBilling', 'Currency Exchange Rate Source', 'entity_id'): 'base_user',
    ('jBilling', 'Currency Exchange Rate Source', 'currency_id'): 'base_user',
    ('jBilling', 'Ageing Step Config Validation', 'entity_id'): 'base_user',
    ('jBilling', 'Ageing Step Config Validation', 'status_id'): 'base_user',
}

_PLACEHOLDER_NAME_RE = re.compile(r'<([A-Za-z_][A-Za-z0-9_ ]*)>')
_PLACEHOLDER_CONJUNCT_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*<([A-Za-z_][A-Za-z0-9_ ]*)>')
_COLON_OR_SELF_REF_RE = re.compile(r'(?<!:):(?!:)[A-Za-z_][A-Za-z0-9_]*|\bself\b')


def _load_raw_schema(cs):
    with open(CASE_STUDY_SCHEMA_JSON[cs], encoding='utf-8') as f:
        return json.load(f)


def _schema_table_entry(raw_schema, table):
    return raw_schema.get(table) or raw_schema.get(table.upper()) or raw_schema.get(table.lower())


def _canonical_table_name(raw_schema, table):
    for candidate in (table, table.upper(), table.lower()):
        if candidate in raw_schema:
            return candidate
    return table


def _pk_columns_for(raw_schema, table):
    pk = (_schema_table_entry(raw_schema, table) or {}).get('pk')
    if pk is None:
        return []
    return pk if isinstance(pk, list) else [pk]


def _fk_edges_for(raw_schema, table):
    entry = _schema_table_entry(raw_schema, table) or {}
    return [{**e, 'ref_table': _canonical_table_name(raw_schema, e['ref_table'])}
            for e in entry.get('fk_columns') or []]


def _decision_subject_tables_referenced(cs, record):
    tables = set()

    def visit(node):
        if not isinstance(node, dict):
            return
        kind = node.get('kind')
        if kind == 'substituted_decision':
            for child in node.get('free_variable_resolutions', {}).values():
                visit(child)
            return
        filter_text = node.get('filter_text')
        if kind in ('derived_aggregate', 'raw_sql_boolean') or (kind == 'exists' and filter_text):
            text = node.get('sql_template') if kind == 'raw_sql_boolean' else filter_text
            for placeholder in _PLACEHOLDER_NAME_RE.findall(text or ''):
                extra = _DECISION_SUBJECT_PLACEHOLDER_SOURCES.get((cs, record['decision_name'], placeholder))
                if extra:
                    tables.add(extra)
            if kind != 'raw_sql_boolean' and text and _COLON_OR_SELF_REF_RE.search(text):
                if kind == 'derived_aggregate':
                    tables.add(node.get('self_table') or node['table'])
                else:
                    tables.update(node.get('candidate_tables') or [])
            return
        if kind in ('schema_column', 'null_check'):
            tables.add(node['table'])
            return
        collect_tables_from_resolution(node, tables)

    for node in record.get('variable_resolution', {}).values():
        visit(node)
    return tables


def compute_cross_table_placeholder_correlations(cs, decision_name, node):
    correlations = {}
    filter_text = node.get('filter_text')
    if node.get('kind') not in ('exists', 'derived_aggregate') or not filter_text:
        return correlations
    for column, placeholder in _PLACEHOLDER_CONJUNCT_RE.findall(filter_text):
        source_table = _DECISION_SUBJECT_PLACEHOLDER_SOURCES.get((cs, decision_name, placeholder))
        if source_table:
            correlations[placeholder] = {'table': source_table, 'column': column}
    return correlations


def compute_subject_hop_placeholder_correlations(subject, node):
    correlations = {}
    filter_text = node.get('filter_text')
    if node.get('kind') not in ('exists', 'derived_aggregate') or not filter_text or not subject:
        return correlations
    hops_by_from_column = {}
    for hops in subject['joins'].values():
        for hop in hops:
            if hop['from_table'].upper() == subject['table'].upper() and 'from_column' in hop:
                hops_by_from_column[hop['from_column'].upper()] = hop
    for column, placeholder in _PLACEHOLDER_CONJUNCT_RE.findall(filter_text):
        hop = hops_by_from_column.get(column.upper())
        if hop and placeholder not in correlations:
            correlations[placeholder] = {'table': hop['to_table'], 'column': hop['to_column']}
    return correlations


def _functional_backward_edges_for(raw_schema, table):
    target = _canonical_table_name(raw_schema, table)
    found = []
    for other_name, info in raw_schema.items():
        other_pk = info.get('pk')
        other_pk_list = other_pk if isinstance(other_pk, list) else ([other_pk] if other_pk else [])
        for fk in info.get('fk_columns') or []:
            if (_canonical_table_name(raw_schema, fk['ref_table']) == target
                    and other_pk_list == [fk['column']]):
                found.append({'column': fk['ref_column'],
                              'ref_table': _canonical_table_name(raw_schema, other_name),
                              'ref_column': fk['column']})
    return found


_DECISION_SUBJECT_JOIN_DISAMBIGUATION = {
    ('OpenMRS', 'obs', 'concept'): 'concept_id',
}


def _build_subject_join_path(cs, raw_schema, root, target, allowed_tables):
    root = _canonical_table_name(raw_schema, root)
    target = _canonical_table_name(raw_schema, target)
    if root == target:
        return []

    allowed = {_canonical_table_name(raw_schema, t) for t in allowed_tables} | {root, target}
    visited = {root}
    queue = [(root, [])]
    skipped_ambiguous = []
    while queue:
        current, path = queue.pop(0)
        by_target = {}
        for edge in _fk_edges_for(raw_schema, current) + _functional_backward_edges_for(raw_schema, current):
            nxt = edge['ref_table']
            if nxt in allowed and nxt not in visited:
                by_target.setdefault(nxt, []).append(edge)
        for nxt, candidate_edges in by_target.items():
            distinct_columns = {e['column'] for e in candidate_edges}
            if len(distinct_columns) > 1:
                override_col = _DECISION_SUBJECT_JOIN_DISAMBIGUATION.get((cs, current, nxt))
                matching = [e for e in candidate_edges if e['column'] == override_col] if override_col else None
                if not matching:
                    skipped_ambiguous.append((current, nxt))
                    continue
                candidate_edges = matching
            edge = candidate_edges[0]
            hop = {'from_table': current, 'from_column': edge['column'],
                   'to_table': nxt, 'to_column': edge['ref_column']}
            new_path = path + [hop]
            if nxt == target:
                path_nodes = {root} | {h['to_table'] for h in new_path}
                for amb_from, amb_to in skipped_ambiguous:
                    if amb_from in path_nodes and amb_to in path_nodes:
                        return None
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
        if current != root:
            continue
        for edge in _composite_backward_edges_for(raw_schema, current):
            nxt = edge['ref_table']
            if nxt not in allowed or nxt in visited or nxt in by_target:
                continue
            hop = {'from_table': current, 'from_columns': edge['columns'],
                   'to_table': nxt, 'to_columns': edge['ref_columns']}
            new_path = path + [hop]
            if nxt == target:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def _composite_backward_edges_for(raw_schema, table):
    table = _canonical_table_name(raw_schema, table)
    source_targets = {(e['ref_table'], e['ref_column']): e['column'] for e in _fk_edges_for(raw_schema, table)}
    found = []
    for other_name in raw_schema:
        other = _canonical_table_name(raw_schema, other_name)
        if other == table:
            continue
        pk = _pk_columns_for(raw_schema, other)
        if len(pk) < 2:
            continue
        other_targets = {e['column']: (e['ref_table'], e['ref_column']) for e in _fk_edges_for(raw_schema, other)}
        matched = []
        for pk_col in pk:
            target = other_targets.get(pk_col)
            source_column = source_targets.get(target) if target else None
            if source_column is None:
                matched = None
                break
            matched.append(source_column)
        if matched and len(set(matched)) == len(matched):
            found.append({'columns': matched, 'ref_table': other, 'ref_columns': pk})
    return found


def _pick_subject_root(cs, raw_schema, all_tables, closure_tables, override=None):
    candidate_pool = closure_tables | all_tables
    candidates = [
        root for root in candidate_pool
        if all(_build_subject_join_path(cs, raw_schema, root, t, closure_tables) is not None
               for t in all_tables - {root})
    ]
    if override is not None:
        override = _canonical_table_name(raw_schema, override)
        if override not in candidates:
            raise ValueError(f"Stale subject-root override {override!r}; qualifying roots: {sorted(candidates)}")
        return override
    return candidates[0] if len(candidates) == 1 else None


def _upstream_decisions_of(node, out):
    if not isinstance(node, dict):
        return
    if node.get('kind') == 'literal_via_upstream_branch' and node.get('from_decision'):
        out.add(node['from_decision'])
    for child in (node.get('free_variable_resolutions') or {}).values():
        _upstream_decisions_of(child, out)


def _with_upstream_subject_joins(cs, subject, decision_records, own_subjects, raw_schema, closure):
    upstream = set()
    for r in decision_records:
        for node in r.get('variable_resolution', {}).values():
            _upstream_decisions_of(node, upstream)
    joins = dict(subject['joins'])
    for name in sorted(upstream):
        up = own_subjects.get(name)
        if not up:
            continue
        table = _canonical_table_name(raw_schema, up['table'])
        if table == _canonical_table_name(raw_schema, subject['table']) or table in joins:
            continue
        path = _build_subject_join_path(cs, raw_schema, subject['table'], table, closure)
        if path:
            joins[table] = path
    if joins == subject['joins']:
        return subject
    return {**subject, 'joins': joins}


def compute_decision_subject(cs, decision_records, raw_schema):
    all_tables = set()
    closure_tables = set()
    for r in decision_records:
        closure_tables |= {_canonical_table_name(raw_schema, t) for t in r.get('fk_closure_tables', [])}
        all_tables |= {_canonical_table_name(raw_schema, t)
                       for t in _decision_subject_tables_referenced(cs, r)}

    if not all_tables:
        return None
    if len(all_tables) == 1:
        table = next(iter(all_tables))
        return {'table': table, 'pk_columns': _pk_columns_for(raw_schema, table), 'joins': {}}

    from bridge.grounding.decision_subject_overrides import SUBJECT_ROOT_OVERRIDES
    override = SUBJECT_ROOT_OVERRIDES.get((cs, decision_records[0]['decision_name']))
    root = _pick_subject_root(cs, raw_schema, all_tables, closure_tables, override)
    if root is None:
        return None
    joins = {}
    for t in all_tables:
        if t == root:
            continue
        path = _build_subject_join_path(cs, raw_schema, root, t, closure_tables)
        if path is None:
            return None
        joins[t] = path
    return {'table': root, 'pk_columns': _pk_columns_for(raw_schema, root), 'joins': joins}


def find_blocking_issues(var_name, node, out=None):
    out = [] if out is None else out
    if not isinstance(node, dict):
        return out
    if node.get('kind') in ('unresolved', 'schema_gap', 'chained_decision_output', 'code_external'):
        out.append((var_name, node['kind'], node.get('reason') or node.get('notes') or node.get('note', '')))
    elif node.get('kind') == 'substituted_decision':
        for fv, sub in node.get('free_variable_resolutions', {}).items():
            find_blocking_issues(fv, sub, out)
    return out


def collect_tables_from_resolution(node, tables):
    if not isinstance(node, dict):
        return
    if node.get('kind') == 'schema_column':
        tables.add(node['table'])
        for t, _c in node.get('also_valid_in', []):
            tables.add(t)
    elif node.get('kind') == 'serialized_field':
        tables.add(node['table'])
    elif node.get('kind') == 'derived':
        for t, _c in node.get('table_hints', []):
            tables.add(t)
    elif node.get('kind') == 'derived_aggregate':
        for t in node['table'].split(','):
            t = t.strip().split()[0] if t.strip() else ''
            if t:
                tables.add(t)
        if node.get('value_column'):
            tables.add(node['value_column'].split('.')[0])
    elif node.get('kind') == 'null_check':
        tables.add(node['table'])
    elif node.get('kind') == 'any_not_null':
        for c in node.get('columns', []):
            tables.add(c['table'])
    elif node.get('kind') in ('join_lookup', 'join_null_check'):
        tables.add(node['via']['local_table'])
        tables.add(node['result_table'])
    elif node.get('kind') == 'exists':
        for t in node.get('candidate_tables', []):
            tables.add(t)
        for c in node.get('candidate_columns', []):
            tables.add(c['table'])
    elif node.get('kind') == 'regex_match':
        tables.add(node['value_column']['table'])
        tables.add(node['pattern_column']['table'])
    elif node.get('kind') == 'substituted_decision':
        for sub in node.get('free_variable_resolutions', {}).values():
            collect_tables_from_resolution(sub, tables)
    elif node.get('kind') == 'raw_sql_boolean':
        for t in node.get('tables', []):
            tables.add(t)
    elif node.get('kind') == 'derived_case':
        tables.add(node['table'])


def build_rule_condition(decision, rule):
    column_predicates = []
    parse_errors = []
    for (col_var, _typeref), text in zip(decision.inputs, rule['input_texts']):
        effective_col_var = get_column_override(rule['id'], col_var)
        try:
            node = parse_unary_test(text, effective_col_var)
        except UnsupportedFeelConstruct as e:
            parse_errors.append(f"{col_var}: {e}")
            node = None
        if node is not None:
            column_predicates.append(node)
    if parse_errors:
        return None, parse_errors, column_predicates
    if not column_predicates:
        condition = {'kind': 'literal', 'value': True, 'type': 'boolean'}
    elif len(column_predicates) == 1:
        condition = column_predicates[0]
    else:
        condition = {'op': 'and', 'clauses': column_predicates}
    return condition, [], column_predicates


def parse_output_value(text):
    text = (text or '').strip()
    if text.startswith('"') and text.endswith('"'):
        return {'kind': 'literal', 'value': text[1:-1], 'type': 'string'}
    if text in ('true', 'false'):
        return {'kind': 'literal', 'value': text == 'true', 'type': 'boolean'}
    if re.match(r'^-?\d+(\.\d+)?$', text):
        return {'kind': 'literal', 'value': float(text) if '.' in text else int(text), 'type': 'number'}
    return {'kind': 'literal', 'value': text, 'type': 'string'}


def build_hit_policy_truth_condition(decision, row_idx):
    own_condition, parse_errors, _ = build_rule_condition(decision, decision.rules[row_idx])
    if parse_errors:
        return None, False
    clauses = [own_condition]
    if decision.hit_policy in ('FIRST', 'UNIQUE'):
        for earlier_idx in range(row_idx):
            earlier_condition, earlier_errors, _ = build_rule_condition(decision, decision.rules[earlier_idx])
            if earlier_errors:
                return None, False
            clauses.append({'op': 'not', 'clause': earlier_condition})
    if len(clauses) == 1:
        return clauses[0], True
    return {'op': 'and', 'clauses': clauses}, True


def _grounding_options(cs, gt, upstream_decision, wanted_var, by_id, by_name, seen, commitment):
    key = (upstream_decision.name, wanted_var)
    if key in seen:
        return
    seen = seen | {key}

    out_names = [o for o, _ in upstream_decision.outputs]
    if wanted_var not in out_names:
        return
    out_idx = out_names.index(wanted_var)

    dname = upstream_decision.name
    already_committed = dname in commitment
    row_indices = ([i for i, rule in enumerate(upstream_decision.rules) if rule['id'] == commitment[dname]]
                   if already_committed else range(len(upstream_decision.rules)))

    for row_idx in row_indices:
        rule = upstream_decision.rules[row_idx]
        value_node = parse_output_value(rule['output_texts'][out_idx])
        truth_condition, ok = build_hit_policy_truth_condition(upstream_decision, row_idx)
        if not ok:
            continue

        referenced = sorted(set(find_all_variable_refs(truth_condition)))
        base_resolutions = {}
        needs = []
        clean = True
        for var in referenced:
            res = resolve_and_substitute(cs, gt, upstream_decision, var, by_id, by_name)
            issues = find_blocking_issues(var, res)
            if not issues:
                base_resolutions[var] = res
                continue
            if any(k != 'chained_decision_output' for _v, k, _d in issues):
                clean = False
                break
            further_upstream = by_name.get(res['from_decision'])
            if further_upstream is None:
                clean = False
                break
            needs.append((var, further_upstream))
        if not clean:
            continue

        own_label = f"{dname}::{rule['id']}"
        set_here = not already_committed
        if set_here:
            commitment[dname] = rule['id']
        try:
            if not needs:
                yield {'value': value_node, 'condition': truth_condition,
                       'extra_resolutions': dict(base_resolutions),
                       'source_rule_id': rule['id'], 'source_decision': dname,
                       'provenance': [own_label]}
                continue
            for combo, combo_provenance in _enumerate_needs(cs, gt, needs, by_id, by_name, seen, commitment):
                combo_clauses = [truth_condition]
                combo_resolutions = dict(base_resolutions)
                for var_name, sub_opt in combo.items():
                    combo_resolutions[var_name] = {
                        'kind': 'literal_via_upstream_branch', 'value': sub_opt['value'],
                        'from_decision': sub_opt['source_decision'], 'from_rule_id': sub_opt['source_rule_id']}
                    combo_clauses.append(sub_opt['condition'])
                    combo_resolutions.update(sub_opt['extra_resolutions'])
                combined_condition = combo_clauses[0] if len(combo_clauses) == 1 \
                    else {'op': 'and', 'clauses': combo_clauses}
                yield {'value': value_node, 'condition': combined_condition,
                       'extra_resolutions': combo_resolutions,
                       'source_rule_id': rule['id'], 'source_decision': dname,
                       'provenance': [own_label] + combo_provenance}
        finally:
            if set_here:
                del commitment[dname]


def _enumerate_needs(cs, gt, needs, by_id, by_name, seen, commitment):
    if not needs:
        yield {}, []
        return
    (var_name, upstream_decision), *rest = needs
    for opt in _grounding_options(cs, gt, upstream_decision, var_name, by_id, by_name, seen, commitment):
        for rest_combo, rest_provenance in _enumerate_needs(cs, gt, rest, by_id, by_name, seen, commitment):
            combo = {var_name: opt}
            combo.update(rest_combo)
            yield combo, opt['provenance'] + rest_provenance


_NOT_GROUNDED = object()


def _grounded_constant_for_variable(var_name, variable_resolution):
    res = variable_resolution.get(var_name)
    if not isinstance(res, dict):
        return _NOT_GROUNDED
    kind = res.get('kind')
    if kind == 'literal_via_upstream_branch':
        value_node = res.get('value')
        return value_node.get('value') if isinstance(value_node, dict) else _NOT_GROUNDED
    if kind == 'substituted_decision':
        sub_resolutions = res.get('free_variable_resolutions', {})
        return _evaluate_expr_to_constant(
            res.get('expression'),
            lambda v: _grounded_constant_for_variable(v, sub_resolutions))
    return _NOT_GROUNDED


def _evaluate_expr_to_constant(node, resolve_var):
    if not isinstance(node, dict):
        return _NOT_GROUNDED
    if node.get('kind') == 'literal':
        return node['value']
    if node.get('kind') == 'variable':
        return resolve_var(node['ref'])
    op = node.get('op')
    if op in ('+', '-', '*', '/'):
        left = _evaluate_expr_to_constant(node['left'], resolve_var)
        right = _evaluate_expr_to_constant(node['right'], resolve_var)
        if left is _NOT_GROUNDED or right is _NOT_GROUNDED:
            return _NOT_GROUNDED
        try:
            if op == '+': return left + right
            if op == '-': return left - right
            if op == '*': return left * right
            if op == '/': return left / right
        except (TypeError, ZeroDivisionError):
            return _NOT_GROUNDED
    if op == 'if':
        cond = _fold_condition_three_valued(node['cond'], None, resolve_var)
        if cond is True:
            return _evaluate_expr_to_constant(node['then'], resolve_var)
        if cond is False:
            return _evaluate_expr_to_constant(node['else'], resolve_var)
        return _NOT_GROUNDED
    return _NOT_GROUNDED


def _fold_condition_three_valued(node, variable_resolution, resolve_var=None):
    resolve = resolve_var or (lambda v: _grounded_constant_for_variable(v, variable_resolution))
    if not isinstance(node, dict):
        return None
    if node.get('kind') == 'literal':
        return bool(node['value']) if node.get('type') == 'boolean' else None
    op = node.get('op')
    if op == 'and':
        values = [_fold_condition_three_valued(c, variable_resolution, resolve) for c in node['clauses']]
        if any(v is False for v in values):
            return False
        if any(v is None for v in values):
            return None
        return True
    if op == 'or':
        values = [_fold_condition_three_valued(c, variable_resolution, resolve) for c in node['clauses']]
        if any(v is True for v in values):
            return True
        if any(v is None for v in values):
            return None
        return False
    if op == 'not':
        value = _fold_condition_three_valued(node['clause'], variable_resolution, resolve)
        return None if value is None else (not value)
    if op in ('=', '!=', '<', '<=', '>', '>='):
        left = _evaluate_expr_to_constant(node['left'], resolve)
        right = _evaluate_expr_to_constant(node['right'], resolve)
        if left is _NOT_GROUNDED or right is _NOT_GROUNDED:
            return None
        try:
            if op == '=': return left == right
            if op == '!=': return left != right
            if op == '<': return left < right
            if op == '<=': return left <= right
            if op == '>': return left > right
            if op == '>=': return left >= right
        except TypeError:
            return None
    if op == 'in':
        left = _evaluate_expr_to_constant(node['left'], resolve)
        if left is _NOT_GROUNDED:
            return None
        values = [_evaluate_expr_to_constant(v, resolve) for v in node['values']]
        if any(v is _NOT_GROUNDED for v in values):
            return None
        try:
            return left in values
        except TypeError:
            return None
    if op == 'between':
        left = _evaluate_expr_to_constant(node['left'], resolve)
        low = _evaluate_expr_to_constant(node['low'], resolve)
        high = _evaluate_expr_to_constant(node['high'], resolve)
        if _NOT_GROUNDED in (left, low, high):
            return None
        try:
            return low <= left <= high
        except TypeError:
            return None
    return None


def compile_case_study(cs, mapping_source='ground_truth'):
    by_id, by_name = load_case_study_decisions(cs)
    gt = load_case_study_ground_truth(cs)
    fk_graph, all_tables = load_schema_fk_graph(cs)

    records = []
    blocked = []

    def cross_variable_reference_flag(column_predicates):
        return any(
            isinstance(node, dict) and node.get('kind') == 'variable'
            for pred in column_predicates
            for node in (pred.get('right'), pred.get('low'), pred.get('high'))
            if isinstance(pred, dict)
        ) or any(
            isinstance(v, dict) and v.get('kind') == 'variable'
            for pred in column_predicates if isinstance(pred, dict)
            for v in pred.get('values', [])
        )

    for decision in by_name.values():
        if not decision.is_table:
            continue

        for row_idx, rule in enumerate(decision.rules):
            record_id = f"{cs}::{decision.name}::{rule['id']}"
            condition, parse_errors, column_predicates = build_rule_condition(decision, rule)
            if parse_errors:
                blocked.append({'record_id': record_id, 'reason': 'feel_parse_error', 'detail': parse_errors})
                continue

            referenced_vars = sorted(set(find_all_variable_refs(condition)))
            variable_resolution = {}
            own_blocking = []
            for var in referenced_vars:
                res = resolve_and_substitute(cs, gt, decision, var, by_id, by_name)
                variable_resolution[var] = res
                own_blocking.extend(find_blocking_issues(var, res))

            earlier_rows = []
            earlier_chained_blocking = []
            if decision.hit_policy in ('FIRST', 'UNIQUE'):
                for earlier_idx in range(row_idx):
                    earlier_condition, earlier_errors, _ = build_rule_condition(decision, decision.rules[earlier_idx])
                    if earlier_errors:
                        continue
                    earlier_rows.append({'rule_id': decision.rules[earlier_idx]['id'], 'condition': earlier_condition})
                    for var in find_all_variable_refs(earlier_condition):
                        if var in variable_resolution:
                            continue
                        res = resolve_and_substitute(cs, gt, decision, var, by_id, by_name)
                        variable_resolution[var] = res
                        earlier_chained_blocking.extend(
                            (v, k, d) for v, k, d in find_blocking_issues(var, res) if k == 'chained_decision_output')

            blocking = own_blocking + earlier_chained_blocking
            chained_blocking = [b for b in own_blocking if b[1] == 'chained_decision_output'] + earlier_chained_blocking
            other_blocking = [b for b in own_blocking if b[1] != 'chained_decision_output']

            if other_blocking:
                blocked.append({'record_id': record_id, 'reason': 'unresolved_variable',
                                 'blocking_variables': [{'variable': v, 'kind': k, 'detail': d}
                                                         for v, k, d in blocking]})
                continue

            variants = [(condition, variable_resolution, [])]
            if chained_blocking:
                needs = [(var, by_name.get(variable_resolution[var]['from_decision']))
                         for var, _kind, _detail in chained_blocking]
                variants = []
                for combo, provenance in _enumerate_needs(cs, gt, needs, by_id, by_name, set(), {}):
                    vr = dict(variable_resolution)
                    extra_clauses = [condition]
                    for var_name, opt in combo.items():
                        vr[var_name] = {'kind': 'literal_via_upstream_branch', 'value': opt['value'],
                                         'from_decision': opt['source_decision'], 'from_rule_id': opt['source_rule_id']}
                        vr.update(opt['extra_resolutions'])
                        extra_clauses.append(opt['condition'])
                    combined = extra_clauses[0] if len(extra_clauses) == 1 else {'op': 'and', 'clauses': extra_clauses}
                    variants.append((combined, vr, provenance))
                if not variants:
                    blocked.append({'record_id': record_id, 'reason': 'chained_dependency_unexpandable',
                                     'blocking_variables': [{'variable': v, 'kind': k, 'detail': d}
                                                             for v, k, d in blocking]})
                    continue

            for variant_condition, variant_resolution, provenance_suffix in variants:
                variant_id = record_id if not provenance_suffix else f"{record_id}::via::{'+'.join(provenance_suffix)}"

                outputs = {}
                for (out_name, _typeref), out_text in zip(decision.outputs, rule['output_texts']):
                    outputs[out_name] = parse_output_value(out_text)

                hit_policy_context = {'earlier_rows': earlier_rows}

                full_condition = variant_condition
                if hit_policy_context['earlier_rows']:
                    full_condition = {'op': 'and', 'clauses': [variant_condition] + [
                        {'op': 'not', 'clause': er['condition']} for er in hit_policy_context['earlier_rows']]}
                feasibility = _fold_condition_three_valued(full_condition, variant_resolution)
                if feasibility is False:
                    blocked.append({'record_id': variant_id, 'reason': 'infeasible',
                                     'detail': 'condition provably unsatisfiable by three-valued '
                                               'constant folding of grounded values'})
                    continue

                tables = set()
                for res in variant_resolution.values():
                    collect_tables_from_resolution(res, tables)
                fk_tables = fk_closure(tables, fk_graph, all_tables)

                record = {
                    'record_id': variant_id,
                    'case_study': cs,
                    'dmn_file': decision.dmn_file,
                    'decision_name': decision.name,
                    'hit_policy': decision.hit_policy,
                    'rule_id': rule['id'],
                    'source_citation': rule.get('description', ''),
                    'condition': variant_condition,
                    'outputs': outputs,
                    'hit_policy_context': hit_policy_context,
                    'variable_resolution': variant_resolution,
                    'fk_closure_tables': fk_tables,
                    'cross_variable_reference': cross_variable_reference_flag(column_predicates),
                    'feasibility': 'true' if feasibility is True else 'unknown',
                }
                if provenance_suffix:
                    record['grounded_upstream_branches'] = provenance_suffix
                records.append(record)

    _DECISION_SUBJECT_EXCLUDED_RULES = {
        ('Spree', 'Decision_PromotionCustomerGroupEligibility_rule_4'),
    }
    raw_schema = _load_raw_schema(cs)
    records_by_decision_name = {}
    for r in records:
        records_by_decision_name.setdefault(r['decision_name'], []).append(r)
    def _subject_input(decision_records):
        return [r for r in decision_records
                if (r['case_study'], r['rule_id']) not in _DECISION_SUBJECT_EXCLUDED_RULES]

    own_subjects = {name: compute_decision_subject(cs, _subject_input(recs), raw_schema)
                    for name, recs in records_by_decision_name.items()}
    all_closure = {t for r in records for t in r.get('fk_closure_tables', [])}
    for decision_records in records_by_decision_name.values():
        subject_input = _subject_input(decision_records)
        subject = own_subjects[decision_records[0]['decision_name']]
        if subject is not None:
            stored = _with_upstream_subject_joins(cs, subject, subject_input, own_subjects,
                                                  raw_schema, all_closure)
            for r in subject_input:
                r['decision_subject'] = stored

        def _attach_correlations(node, decision_name):
            if not isinstance(node, dict):
                return
            if node.get('kind') == 'substituted_decision':
                for child in node.get('free_variable_resolutions', {}).values():
                    _attach_correlations(child, decision_name)
                return
            correlations = compute_cross_table_placeholder_correlations(cs, decision_name, node)
            correlations.update({
                k: v for k, v in compute_subject_hop_placeholder_correlations(subject, node).items()
                if k not in correlations
            })
            if correlations:
                node['cross_table_placeholders'] = correlations

        for r in decision_records:
            for node in r.get('variable_resolution', {}).values():
                _attach_correlations(node, r['decision_name'])

    return records, blocked


def compile_case_studies(case_studies=paths.CASE_STUDIES, out_path=paths.COMPILED_PATH):
    """Compile `case_studies` and write the objectives of the evaluation to
    `out_path`, keeping the objectives already there for other case studies."""
    by_case_study = {}
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            for r in json.load(f):
                by_case_study.setdefault(r['case_study'], []).append(r)
    for cs in case_studies:
        records, _blocked = compile_case_study(cs)
        wanted = paths.config(cs)['evaluated_objectives']
        by_id = {r['record_id']: r for r in records}
        missing = [rid for rid in wanted if rid not in by_id]
        if missing:
            raise ValueError(f'{cs}: objectives not produced by the compiler: {missing}')
        keep = set(wanted)
        by_case_study[cs] = [r for r in records if r['record_id'] in keep]
    all_records = [r for cs in paths.CASE_STUDIES for r in by_case_study.get(cs, [])]
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(all_records, f, indent=1)
    return all_records
