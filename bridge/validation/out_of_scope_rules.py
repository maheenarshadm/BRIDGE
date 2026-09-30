"""Rules outside the scope of the evaluation that cannot be derived structurally."""
OUT_OF_SCOPE_RULES = {
    ('Spree', 'Decision_PromotionCustomerGroupEligibility_rule_4'):
        "matchingCustomerGroupCount requires a one-to-many join from a user to "
        "spree_customer_group_users, which has no single subject row.",
}


def is_out_of_scope(case_study, rule_id):
    return (case_study, rule_id) in OUT_OF_SCOPE_RULES


def reason(case_study, rule_id):
    return OUT_OF_SCOPE_RULES.get((case_study, rule_id))

