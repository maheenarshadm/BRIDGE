"""Filter conditions for aggregate inputs whose grounding text cannot be parsed automatically."""
_FILTER_TEXT_OVERRIDES = {}


def get_filter_text_override(case_study, var_name):
    return _FILTER_TEXT_OVERRIDES.get((case_study, var_name))

