"""Independent validation of one generated database.

Every DMN decision is re-executed against the database: its inputs are read
from the database (or from the fixed values of inputs that are not stored in
it) and the rule selected under the decision's hit policy is determined. A
rule counts as verified only if the database makes the decision select it.
The validator does not use the fitness function of the search.
"""
import sqlite3

from bridge.validation.drd_executor import DecisionRunner, run_decision
from bridge.validation.out_of_scope_rules import is_out_of_scope
from bridge.validation.subject_table import build_subject_tables


def verify_decisions(conn, case_study, decisions_by_name, decision_names, not_persisted_overrides):
    subset = {d: decisions_by_name[d] for d in decision_names if d in decisions_by_name}
    in_scope_by_name = {
        name: [r for r in records if not is_out_of_scope(case_study, r['rule_id'])]
        for name, records in decisions_by_name.items()
    }
    resolved, unresolved = build_subject_tables(case_study, in_scope_by_name)
    runner = DecisionRunner(conn, case_study, in_scope_by_name, resolved,
                             not_persisted_overrides=not_persisted_overrides)

    result = {}
    for decision_name, records in subset.items():
        verified_rule_ids = set()
        if decision_name not in unresolved:
            subject_table, pk_cols, join_paths = resolved[decision_name]
            try:
                run_result = run_decision(
                    conn, decision_name, in_scope_by_name[decision_name], subject_table, pk_cols,
                    join_paths=join_paths, runner=runner, collect_trace=False,
                    not_persisted_overrides=not_persisted_overrides)
                verified_rule_ids = run_result['verified_covered_rule_ids']
            except NotImplementedError:
                pass
            except (sqlite3.OperationalError, KeyError):
                pass
        for r in records:
            result[r['rule_id']] = r['rule_id'] in verified_rule_ids
    return result
