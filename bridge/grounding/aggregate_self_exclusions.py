"""Columns that exclude the current row from aggregates over its own table."""
_EXCLUDE_SELF_COLUMN = {}


def get_exclude_self_column(case_study, var_name):
    return _EXCLUDE_SELF_COLUMN.get((case_study, var_name))

