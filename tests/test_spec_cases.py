import sqlite3

from bridge.validation.drd_executor import run_decision
from bridge.validation.rule_evaluator import UniqueViolation


def _rec(case_study, decision_name, rule_id, hit_policy, condition, variable_resolution, outputs=None):
    return {
        'case_study': case_study, 'decision_name': decision_name, 'rule_id': rule_id,
        'record_id': f'{case_study}::{decision_name}::{rule_id}',
        'hit_policy': hit_policy, 'condition': condition,
        'variable_resolution': variable_resolution, 'fk_closure_tables': [],
        'outputs': outputs or {},
    }


def _first_policy_records():
    rule1 = _rec('T', 'D_first', 'Rule_1', 'FIRST',
                 {'op': '>', 'left': {'kind': 'variable', 'ref': 'amount'},
                  'right': {'kind': 'literal', 'value': 100, 'type': 'number'}},
                 {'amount': {'kind': 'schema_column', 'table': 'txn', 'column': 'amount'}})
    rule2 = _rec('T', 'D_first', 'Rule_2', 'FIRST',
                 {'op': '>', 'left': {'kind': 'variable', 'ref': 'amount'},
                  'right': {'kind': 'literal', 'value': 10, 'type': 'number'}},
                 {'amount': {'kind': 'schema_column', 'table': 'txn', 'column': 'amount'}})
    return [rule1, rule2]


def test_case_1_first_earlier_rule_wins():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE txn (txn_id INTEGER PRIMARY KEY, amount REAL)')
    conn.execute('INSERT INTO txn VALUES (1, 500)')
    conn.commit()
    result = run_decision(conn, 'D_first', _first_policy_records(), 'txn', ['txn_id'])
    assert result['matched_by_case'][(1,)] == ['Rule_1', 'Rule_2'], result['matched_by_case']
    assert result['selected_by_case'][(1,)] == 'Rule_1', \
        "FIRST must select the EARLIER matching rule, not the later target, even though both match"
    print("case 1 (FIRST, earlier rule wins): PASS")


def test_case_2_first_fallthrough_to_target():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE txn (txn_id INTEGER PRIMARY KEY, amount REAL)')
    conn.execute('INSERT INTO txn VALUES (2, 50)')
    conn.commit()
    result = run_decision(conn, 'D_first', _first_policy_records(), 'txn', ['txn_id'])
    assert result['matched_by_case'][(2,)] == ['Rule_2'], result['matched_by_case']
    assert result['selected_by_case'][(2,)] == 'Rule_2', \
        "with every earlier rule false, FIRST must fall through to the target rule"
    print("case 2 (FIRST, fallthrough to target): PASS")


def _unique_policy_records():
    rule1 = _rec('T', 'D_unique', 'Rule_1', 'UNIQUE',
                 {'op': '=', 'left': {'kind': 'variable', 'ref': 'status'},
                  'right': {'kind': 'literal', 'value': 'gold', 'type': 'string'}},
                 {'status': {'kind': 'schema_column', 'table': 'customer', 'column': 'status'}})
    rule2 = _rec('T', 'D_unique', 'Rule_2', 'UNIQUE',
                 {'op': '=', 'left': {'kind': 'variable', 'ref': 'status'},
                  'right': {'kind': 'literal', 'value': 'silver', 'type': 'string'}},
                 {'status': {'kind': 'schema_column', 'table': 'customer', 'column': 'status'}})
    return [rule1, rule2]


def test_case_3_unique_single_match():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, status TEXT)')
    conn.execute("INSERT INTO customer VALUES (1, 'gold')")
    conn.commit()
    result = run_decision(conn, 'D_unique', _unique_policy_records(), 'customer', ['customer_id'])
    assert result['matched_by_case'][(1,)] == ['Rule_1']
    assert result['selected_by_case'][(1,)] == 'Rule_1'
    assert result['unique_violations'] == []
    print("case 3 (UNIQUE, single match): PASS")


def test_case_4_unique_violation():
    rule1 = _rec('T', 'D_unique_bad', 'Rule_1', 'UNIQUE',
                 {'op': '=', 'left': {'kind': 'variable', 'ref': 'status'},
                  'right': {'kind': 'literal', 'value': 'gold', 'type': 'string'}},
                 {'status': {'kind': 'schema_column', 'table': 'customer', 'column': 'status'}})
    rule2 = _rec('T', 'D_unique_bad', 'Rule_2', 'UNIQUE',
                 {'op': '!=', 'left': {'kind': 'variable', 'ref': 'status'},
                  'right': {'kind': 'literal', 'value': 'bronze', 'type': 'string'}},
                 {'status': {'kind': 'schema_column', 'table': 'customer', 'column': 'status'}})
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, status TEXT)')
    conn.execute("INSERT INTO customer VALUES (1, 'gold')")
    conn.commit()
    result = run_decision(conn, 'D_unique_bad', [rule1, rule2], 'customer', ['customer_id'])
    assert len(result['unique_violations']) == 1, result['unique_violations']
    pk_vals, violation = result['unique_violations'][0]
    assert isinstance(violation, UniqueViolation)
    assert set(violation.matched_rule_ids) == {'Rule_1', 'Rule_2'}
    assert (1,) not in result['selected_by_case'], \
        "a UNIQUE violation must never produce a selected rule, silently or otherwise"
    print("case 4 (UNIQUE violation): PASS")
    print(f"  reported: {violation}")


def test_case_7_aggregate_input():
    rule_eligible = _rec('T', 'D_aggregate', 'Rule_1', 'FIRST',
                          {'op': '>=', 'left': {'kind': 'variable', 'ref': 'orderCount'},
                           'right': {'kind': 'literal', 'value': 3, 'type': 'number'}},
                          {'orderCount': {'kind': 'derived_aggregate', 'aggregate': 'COUNT',
                                          'table': 'orders', 'filter_text': 'customer_id=<customer>'}})
    rule_default = _rec('T', 'D_aggregate', 'Rule_2', 'FIRST',
                         {'kind': 'literal', 'value': True, 'type': 'boolean'},
                         {'orderCount': {'kind': 'derived_aggregate', 'aggregate': 'COUNT',
                                         'table': 'orders', 'filter_text': 'customer_id=<customer>'}})

    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE customer (customer_id INTEGER PRIMARY KEY)')
    conn.execute('CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER)')
    conn.executemany('INSERT INTO customer VALUES (?)', [(1,), (2,)])
    conn.executemany('INSERT INTO orders (customer_id) VALUES (?)', [
        (1,), (1,), (1,), (1,),
        (2,), (2,),
    ])
    conn.commit()

    result = run_decision(conn, 'D_aggregate', [rule_eligible, rule_default], 'customer', ['customer_id'])
    assert result['selected_by_case'][(1,)] == 'Rule_1', \
        f"customer 1 has 4 real orders (>= 3), aggregate input must select Rule_1: {result['selected_by_case']}"
    assert result['selected_by_case'][(2,)] == 'Rule_2', \
        f"customer 2 has 2 real orders (< 3), must fall through to the default: {result['selected_by_case']}"
    print("case 7 (aggregate input, derived_aggregate): PASS")
    print(f"  selected_by_case = {result['selected_by_case']}")


def test_case_8_join_based_input():
    rule_domestic = _rec('T', 'D_join', 'Rule_1', 'FIRST',
                          {'op': '=', 'left': {'kind': 'variable', 'ref': 'region'},
                           'right': {'kind': 'literal', 'value': 'domestic', 'type': 'string'}},
                          {'region': {'kind': 'join_lookup',
                                      'via': {'local_table': 'order', 'local_column': 'customer_id'},
                                      'result_table': 'customer', 'result_column': 'region'}})
    rule_default = _rec('T', 'D_join', 'Rule_2', 'FIRST',
                         {'kind': 'literal', 'value': True, 'type': 'boolean'},
                         {'region': {'kind': 'join_lookup',
                                     'via': {'local_table': 'order', 'local_column': 'customer_id'},
                                     'result_table': 'customer', 'result_column': 'region'}})

    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, region TEXT)')
    conn.execute('CREATE TABLE "order" (order_id INTEGER PRIMARY KEY, customer_id INTEGER)')
    conn.executemany('INSERT INTO customer VALUES (?, ?)', [(1, 'domestic'), (2, 'overseas')])
    conn.executemany('INSERT INTO "order" (order_id, customer_id) VALUES (?, ?)', [(10, 1), (11, 2)])
    conn.commit()

    result = run_decision(conn, 'D_join', [rule_domestic, rule_default], 'order', ['order_id'])
    assert result['selected_by_case'][(10,)] == 'Rule_1', \
        f"order 10's customer is domestic (via a real join, not a local column): {result['selected_by_case']}"
    assert result['selected_by_case'][(11,)] == 'Rule_2', \
        f"order 11's customer is overseas, must fall through: {result['selected_by_case']}"
    print("case 8 (join-based input, join_lookup): PASS")
    print(f"  selected_by_case = {result['selected_by_case']}")


def test_case_10_duplicate_outputs():
    rule1 = _rec('T', 'D_dup', 'Rule_1', 'FIRST',
                 {'op': '=', 'left': {'kind': 'variable', 'ref': 'tier'},
                  'right': {'kind': 'literal', 'value': 'A', 'type': 'string'}},
                 {'tier': {'kind': 'schema_column', 'table': 'acct', 'column': 'tier'}},
                 outputs={'approved': {'kind': 'literal', 'value': True, 'type': 'boolean'}})
    rule2 = _rec('T', 'D_dup', 'Rule_2', 'FIRST',
                 {'op': '=', 'left': {'kind': 'variable', 'ref': 'tier'},
                  'right': {'kind': 'literal', 'value': 'B', 'type': 'string'}},
                 {'tier': {'kind': 'schema_column', 'table': 'acct', 'column': 'tier'}},
                 outputs={'approved': {'kind': 'literal', 'value': True, 'type': 'boolean'}})

    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE acct (acct_id INTEGER PRIMARY KEY, tier TEXT)')
    conn.executemany('INSERT INTO acct VALUES (?, ?)', [(1, 'A'), (2, 'B')])
    conn.commit()

    result = run_decision(conn, 'D_dup', [rule1, rule2], 'acct', ['acct_id'], collect_trace=True)
    assert result['selected_by_case'][(1,)] == 'Rule_1'
    assert result['selected_by_case'][(2,)] == 'Rule_2'
    assert result['verified_covered_rule_ids'] == {'Rule_1', 'Rule_2'}, \
        "both rules must be tracked as distinctly covered even though their outputs are identical"
    out1 = result['trace'][(1,)]['decision_output']
    out2 = result['trace'][(2,)]['decision_output']
    assert out1 == out2 == {'approved': True}
    print("case 10 (duplicate decision outputs): PASS")
    print(f"  verified_covered_rule_ids = {result['verified_covered_rule_ids']} "
          f"(both tracked despite identical output {out1})")


def test_case_11_filter_placeholder_via_join():
    rule_flagged = _rec('T', 'D_join_placeholder', 'Rule_1', 'FIRST',
                         {'op': '=', 'left': {'kind': 'variable', 'ref': 'isFlaggedRegion'},
                          'right': {'kind': 'literal', 'value': True, 'type': 'boolean'}},
                         {'isFlaggedRegion': {'kind': 'exists', 'candidate_tables': ['flagged_regions'],
                                               'candidate_columns': [{'table': 'flagged_regions', 'column': 'region'}],
                                               'filter_text': 'region = <region>'}})
    rule_default = _rec('T', 'D_join_placeholder', 'Rule_2', 'FIRST',
                         {'kind': 'literal', 'value': True, 'type': 'boolean'},
                         {'isFlaggedRegion': {'kind': 'exists', 'candidate_tables': ['flagged_regions'],
                                               'candidate_columns': [{'table': 'flagged_regions', 'column': 'region'}],
                                               'filter_text': 'region = <region>'}})

    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, region TEXT)')
    conn.execute('CREATE TABLE "order" (order_id INTEGER PRIMARY KEY, customer_id INTEGER)')
    conn.execute('CREATE TABLE flagged_regions (region TEXT)')
    conn.executemany('INSERT INTO customer VALUES (?, ?)', [(1, 'CA'), (2, 'TX')])
    conn.executemany('INSERT INTO "order" (order_id, customer_id) VALUES (?, ?)', [(100, 1), (101, 2)])
    conn.execute("INSERT INTO flagged_regions VALUES ('CA')")
    conn.commit()

    join_paths = {'customer': [{'from_table': 'order', 'from_column': 'customer_id',
                                 'to_table': 'customer', 'to_column': 'customer_id'}]}
    result = run_decision(conn, 'D_join_placeholder', [rule_flagged, rule_default],
                           'order', ['order_id'], join_paths=join_paths)
    assert result['selected_by_case'][(100,)] == 'Rule_1', \
        f"order 100's customer is in region CA (flagged, via a join, not a local column): {result['selected_by_case']}"
    assert result['selected_by_case'][(101,)] == 'Rule_2', \
        f"order 101's customer is in region TX (not flagged), must fall through: {result['selected_by_case']}"
    print("case 11 (filter_text placeholder resolved via a join): PASS")
    print(f"  selected_by_case = {result['selected_by_case']}")
