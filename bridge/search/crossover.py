"""Uniform table-level crossover: each table of a child comes from one of the two parents."""
import copy
import random

from bridge.search.candidate import Candidate
from bridge.search.mutation import _repair_row


def crossover(parent1, parent2, case_study, rng=None, focal1=None, focal2=None,
              focal_maps1=None, focal_maps2=None, scenario_maps1=None, scenario_maps2=None):
    rng = rng or random
    focal1, focal2 = focal1 or {}, focal2 or {}
    focal_maps1, focal_maps2 = focal_maps1 or {}, focal_maps2 or {}
    scenario_maps1, scenario_maps2 = scenario_maps1 or {}, scenario_maps2 or {}
    record_ids = sorted(set(focal_maps1) | set(focal_maps2))
    tables = sorted(set(parent1.as_dict()) | set(parent2.as_dict()) | set(focal1) | set(focal2)
                     | {t for fm in focal_maps1.values() for t in fm}
                     | {t for fm in focal_maps2.values() for t in fm})
    from_parent1 = {table: rng.random() < 0.5 for table in tables}

    def build(mask):
        child = Candidate()
        child_focal = {}
        child_focal_maps = {rid: {} for rid in record_ids}
        for table in tables:
            if mask[table]:
                source, source_focal, source_focal_maps = parent1, focal1, focal_maps1
            else:
                source, source_focal, source_focal_maps = parent2, focal2, focal_maps2
            per_record_entries = {rid: source_focal_maps[rid][table]
                                   for rid in record_ids
                                   if table in source_focal_maps.get(rid, {})}
            rows_copy, focal_copy, per_record_copy = copy.deepcopy(
                (source.rows(table), source_focal.get(table), per_record_entries))
            for row in rows_copy:
                child.add_row(table, row)
            if focal_copy is not None:
                child_focal[table] = focal_copy
            for rid, row_copy in per_record_copy.items():
                child_focal_maps[rid][table] = row_copy
        return child, child_focal, child_focal_maps

    inverted = {table: not v for table, v in from_parent1.items()}
    (child1, child_focal1, child_focal_maps1), (child2, child_focal2, child_focal_maps2) = (
        build(from_parent1), build(inverted))

    child_scenario_maps1 = copy.deepcopy(scenario_maps1)
    child_scenario_maps2 = copy.deepcopy(scenario_maps2)

    for child in (child1, child2):
        for table, rows in list(child.as_dict().items()):
            for row in list(rows):
                _repair_row(child, table, row, case_study)
    return (child1, child2, child_focal1, child_focal2, child_focal_maps1, child_focal_maps2,
            child_scenario_maps1, child_scenario_maps2)
