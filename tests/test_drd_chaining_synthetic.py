import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))

from bridge.validation.drd_executor import DecisionRunner, run_decision
from bridge.validation import schema_utility

schema_utility.SCHEMA_JSON_PATHS['T'] = os.path.join(HERE, 'data', 'synthetic_T_schema.json')


def _rec(case_study, decision_name, rule_id, hit_policy, condition, variable_resolution):
    return {
        'case_study': case_study, 'decision_name': decision_name, 'rule_id': rule_id,
        'record_id': f'{case_study}::{decision_name}::{rule_id}',
        'hit_policy': hit_policy, 'condition': condition,
        'variable_resolution': variable_resolution, 'fk_closure_tables': ['student', 'course_load'],
    }


def build_db():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE student (student_id INTEGER PRIMARY KEY, gpa REAL)')
    conn.execute('CREATE TABLE course_load (course_load_id INTEGER PRIMARY KEY, student_id INTEGER, units INTEGER)')
    conn.executemany('INSERT INTO student VALUES (?, ?)', [
        (1, 3.8),
        (2, 2.9),
    ])
    conn.executemany('INSERT INTO course_load VALUES (?, ?, ?)', [
        (10, 1, 21),
        (11, 2, 21),
    ])
    conn.commit()
    return conn


def build_records():
    d1_rule1 = _rec('T', 'D1', 'D1_Rule_1', 'FIRST',
                     {'op': '>=', 'left': {'kind': 'variable', 'ref': 'gpa'},
                      'right': {'kind': 'literal', 'value': 3.5, 'type': 'number'}},
                     {'gpa': {'kind': 'schema_column', 'table': 'student', 'column': 'gpa'}})
    d1_rule2 = _rec('T', 'D1', 'D1_Rule_2', 'FIRST',
                     {'op': '<', 'left': {'kind': 'variable', 'ref': 'gpa'},
                      'right': {'kind': 'literal', 'value': 3.5, 'type': 'number'}},
                     {'gpa': {'kind': 'schema_column', 'table': 'student', 'column': 'gpa'}})

    d2_rule1 = _rec('T', 'D2', 'D2_Rule_1', 'FIRST',
                     {'op': '>=', 'left': {'kind': 'variable', 'ref': 'bonusUnits'},
                      'right': {'kind': 'literal', 'value': 5, 'type': 'number'}},
                     {'bonusUnits': {'kind': 'literal_via_upstream_branch',
                                     'value': {'kind': 'literal', 'value': 5, 'type': 'number'},
                                     'from_decision': 'D1', 'from_rule_id': 'D1_Rule_1'}})

    d3_rule1 = _rec('T', 'D3', 'D3_Rule_1', 'FIRST',
                     {'op': '>=', 'left': {'kind': 'variable', 'ref': 'totalUnits'},
                      'right': {'kind': 'literal', 'value': 21, 'type': 'number'}},
                     {'totalUnits': {
                         'kind': 'substituted_decision', 'substituted_from': 'UnitTotal',
                         'expression': {'op': '+',
                                        'left': {'kind': 'variable', 'ref': 'units'},
                                        'right': {'kind': 'literal', 'value': 0, 'type': 'number'}},
                         'free_variable_resolutions': {
                             'units': {'kind': 'schema_column', 'table': 'course_load', 'column': 'units'}}}})

    return {
        'D1': [d1_rule1, d1_rule2],
        'D2': [d2_rule1],
        'D3': [d3_rule1],
    }


def test_literal_via_upstream_branch():
    conn = build_db()
    records_by_decision = build_records()
    subject_tables = {
        'D1': ('student', ['student_id'], {}),
        'D2': ('course_load', ['course_load_id'], {}),
    }
    runner = DecisionRunner(conn, 'T', records_by_decision, subject_tables)

    result = run_decision(conn, 'D2', records_by_decision['D2'], 'course_load', ['course_load_id'],
                           join_paths={}, runner=runner)

    assert result['selected_by_case'][(10,)] == 'D2_Rule_1', result['selected_by_case']
    assert result['selected_by_case'][(11,)] is None, result['selected_by_case']
    print("test_literal_via_upstream_branch: PASS")
    print(f"  selected_by_case = {result['selected_by_case']}")


def test_substituted_decision_expression():
    conn = build_db()
    records_by_decision = build_records()
    result = run_decision(conn, 'D3', records_by_decision['D3'], 'course_load', ['course_load_id'], join_paths={})

    assert result['selected_by_case'][(10,)] == 'D3_Rule_1', result['selected_by_case']
    assert result['selected_by_case'][(11,)] == 'D3_Rule_1', result['selected_by_case']
    print("test_substituted_decision_expression: PASS")
    print(f"  selected_by_case = {result['selected_by_case']}")
