"""Join path to use where the schema offers several paths between two tables."""
JOIN_DISAMBIGUATION = {
    ('OpenMRS', 'obs', 'concept'): {
        'column': 'concept_id',
        'reason': (
            "obs has two FK columns to concept: concept_id (the concept THIS "
            "OBSERVATION measures -- OpenMRS's standard EAV pattern) and "
            "value_coded (an unrelated CODED ANSWER value, only meaningful "
            "when the observation's own answer happens to be a concept). "
            "The decisions needing this join (Numeric Precision Validity, "
            "Numeric Absolute Range Validity, Numeric Interpretation "
            "Classification, Obs Value Required By Datatype) all read "
            "properties of the concept an observation is ABOUT (numeric "
            "range/precision/datatype), not properties of its answer value -- "
            "concept_id is the correct column. "
            "[ASSUMED by researcher domain knowledge of OpenMRS's EAV schema "
            "-- no ground-truth source citation available for this specific "
            "join; disclosed here rather than silently guessed.]"
        ),
    },
}


def get_override(case_study, from_table, to_table):
    return JOIN_DISAMBIGUATION.get((case_study, from_table, to_table))

