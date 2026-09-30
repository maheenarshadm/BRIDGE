"""Subject table of decisions whose subject cannot be derived from their inputs."""
SUBJECT_ROOT_OVERRIDES = {}


def get_override(case_study, decision_name):
    return SUBJECT_ROOT_OVERRIDES.get((case_study, decision_name))

