"""Locations of the case-study inputs and of the compiled search objectives.

Every case study lives in ``case_studies/<folder>/``:

    dmn/                   DMN decision models (the business rules)
    grounding.csv          decision input -> database schema mapping
    schema.json            relational schema (tables, columns, keys, constraints)
    not_persisted*.json    values of decision inputs that are not stored in the database
    config.json            column names of grounding.csv, the not-persisted files,
                           known constants and the objectives used in the evaluation
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_STUDIES_DIR = os.path.join(ROOT, 'case_studies')
COMPILED_PATH = os.path.join(CASE_STUDIES_DIR, 'compiled_constraints.json')

CASE_STUDY_FOLDERS = {
    'jBilling': 'jbilling',
    'Spree': 'spree',
    'OpenMRS': 'openmrs',
}
CASE_STUDIES = tuple(CASE_STUDY_FOLDERS)

_config_cache = {}


def case_study_dir(case_study):
    return os.path.join(CASE_STUDIES_DIR, CASE_STUDY_FOLDERS[case_study])


def dmn_dir(case_study):
    return os.path.join(case_study_dir(case_study), 'dmn')


def schema_json(case_study):
    return os.path.join(case_study_dir(case_study), 'schema.json')


def grounding_csv(case_study):
    return os.path.join(case_study_dir(case_study), 'grounding.csv')


def config(case_study):
    if case_study not in _config_cache:
        with open(os.path.join(case_study_dir(case_study), 'config.json'), encoding='utf-8') as f:
            _config_cache[case_study] = json.load(f)
    return _config_cache[case_study]


def not_persisted_files(case_study):
    return [os.path.join(case_study_dir(case_study), name) for name in config(case_study)['not_persisted']]


def known_constants(case_study):
    return config(case_study).get('known_constants', {})


SCHEMA_JSON_PATHS = {cs: schema_json(cs) for cs in CASE_STUDIES}
