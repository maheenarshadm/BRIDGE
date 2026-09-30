"""Relationships used for joins that the schema does not declare as foreign keys."""
SUPPLEMENTARY_FK_EDGES = {
    ('Spree', 'spree_order_promotions'): [
        {'column': 'order_id', 'ref_table': 'spree_orders', 'ref_column': 'id',
         'reason': ("spree_order_promotions is Spree's own real order<->promotion join table "
                    "(order_id, promotion_id, no other columns beyond timestamps) -- an "
                    "unambiguous, singular FK by Rails convention and column naming, missing "
                    "from the extracted schema's own fk_columns (empty list) purely because "
                    "Spree's migration-history source was unavailable to the extractor, not "
                    "because the relationship is actually ambiguous or uncertain.")},
        {'column': 'promotion_id', 'ref_table': 'spree_promotions', 'ref_column': 'id',
         'reason': "same table, same extraction gap, other half of the join -- see order_id above."},
    ],
    ('Spree', 'spree_promotion_rules'): [
        {'column': 'promotion_id', 'ref_table': 'spree_promotions', 'ref_column': 'id',
         'reason': ("spree_promotion_rules.promotion_id is the rule's own owning promotion -- "
                    "same extraction gap as spree_order_promotions above (fk_columns: [] despite "
                    "an unambiguous, real, singular target).")},
    ],
}


def get_supplementary_edges(case_study, table):
    return SUPPLEMENTARY_FK_EDGES.get((case_study, table), [])

