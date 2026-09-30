"""Rules outside the scope of the evaluation: rules of COLLECT decisions, rules
whose inputs are all computed by the application code, and the rules listed
in out_of_scope_rules.py."""
from bridge.validation.out_of_scope_rules import OUT_OF_SCOPE_RULES, is_out_of_scope


def _is_code_external_only(record):
    kinds = {node.get('kind') for node in record.get('variable_resolution', {}).values()}
    return bool(kinds) and kinds <= {'code_external'}


def _mechanical_out_of_scope(case_study, records_by_rule):
    freshly_computed = set()
    for rule_id, recs in records_by_rule.items():
        if any(r['hit_policy'] == 'COLLECT' for r in recs):
            freshly_computed.add(rule_id)
        elif all(_is_code_external_only(r) for r in recs):
            freshly_computed.add(rule_id)

    registered = {rule_id for rule_id in records_by_rule
                  if is_out_of_scope(case_study, rule_id)
                  and OUT_OF_SCOPE_RULES[(case_study, rule_id)] in
                  ('COLLECT hit policy', 'all facts code_external')}

    if freshly_computed != registered:
        missing = freshly_computed - registered
        stale = registered - freshly_computed
        raise ValueError(
            f"{case_study}: out_of_scope_rules.py's structural entries have drifted from "
            f"a fresh recomputation against the current compiled_constraints.json -- "
            f"missing (structurally out of scope but not registered): {sorted(missing)}; "
            f"stale (registered as structural but no longer meets either criterion): "
            f"{sorted(stale)}. Update out_of_scope_rules.py's own structural section, "
            f"never silently trust either side.")

    return {rule_id for rule_id in records_by_rule if is_out_of_scope(case_study, rule_id)}
