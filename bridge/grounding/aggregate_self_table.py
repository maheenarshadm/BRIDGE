"""Table of the current row for aggregate inputs whose filter refers to the row itself."""
_SELF_REFERENCE_TABLE = {
    ('Spree', 'adjustedCreditsCount'): 'spree_promotions',
    ('Spree', 'priorPromotionUsageCount'): 'spree_orders',
}


def get_self_table(case_study, var_name):
    return _SELF_REFERENCE_TABLE.get((case_study, var_name))

