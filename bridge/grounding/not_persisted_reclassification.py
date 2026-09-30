"""Inputs listed as not stored in the database that are grounded in a schema table instead."""
_RECLASSIFICATIONS = {
    ('jBilling', 'customContactFieldConfigured'): {
        'kind': 'exists',
        'candidate_tables': ['pluggable_task_parameter'],
        'candidate_columns': [],
        'filter_text': "name = 'custom_contact_field_id'",
        'notes': (
            "ground truth labeled this 'not-persisted' ('A pluggable_task_parameter "
            "configuration value, not a row in a business table'), but "
            "pluggable_task_parameter is a real table in jBilling's own schema; "
            "reclassified as a global exists-check, see "
            "not_persisted_reclassification.py for the full disclosure"
        ),
    },
}


def get_reclassification(case_study, var_name):
    override = _RECLASSIFICATIONS.get((case_study, var_name))
    return dict(override) if override else None

