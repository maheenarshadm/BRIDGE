"""Rule-specific replacements of the variable a condition column reads."""
_COLUMN_OVERRIDES = {
    'Decision_OrderPeriodAlreadyInvoiced_Rule_3': {'nextBillableDay': 'candidateDate'},
    'Decision_OrderPeriodAlreadyInvoiced_Rule_4': {'nextBillableDay': 'candidateDate'},
}


def get_column_override(rule_id, col_var):
    return _COLUMN_OVERRIDES.get(rule_id, {}).get(col_var, col_var)

