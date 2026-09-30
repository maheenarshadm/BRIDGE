"""Groundings of literal-expression decisions whose FEEL expression the parser does not support."""
_PRIOR_COMPLETED_ORDER_COUNT = {
    'expression': {'kind': 'variable', 'ref': '__prior_completed_order_count'},
    'free_variable_resolutions': {
        '__prior_completed_order_count': {
            'kind': 'derived_aggregate',
            'aggregate': 'COUNT',
            'table': 'spree_orders',
            'filter_text': 'user_id = <user_id> AND completed_at IS NOT NULL AND id != self',
            'notes': ('[ASSUMED] Translated by hand from the FEEL list comprehension of the '
                      'decision. completed_at IS NOT NULL stands in for order.state = "complete", '
                      'since no state column reads "complete". The "OR email = <email>" '
                      'guest-checkout alternative is not modeled: orders are matched by user_id only.'),
        },
    },
}

LITERAL_EXPRESSION_OVERRIDES = {
    ('Spree', 'Prior Completed Order Count'): _PRIOR_COMPLETED_ORDER_COUNT,
}


def get_override(case_study, decision_name):
    return LITERAL_EXPRESSION_OVERRIDES.get((case_study, decision_name))

