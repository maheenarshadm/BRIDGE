"""Tables that supply the values of placeholders in filter conditions."""
FILTER_PLACEHOLDER_SOURCES = {
    ('Spree', 'promotion_id'): 'spree_order_promotions',
    ('T', 'region'): 'customer',
    ('jBilling', 'entity_id'): 'base_user',
    ('jBilling', 'currency_id'): 'base_user',
}


def get_source_table(case_study, placeholder):
    return FILTER_PLACEHOLDER_SOURCES.get((case_study, placeholder))

