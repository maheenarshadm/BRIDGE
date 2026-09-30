"""Dataset construction: turns an individual found by the search into a
consistent database state and writes it to SQLite.

Each objective owns its own rows in an individual. Before materialization the
key values of every objective's rows are shifted into a separate range, so
rows of different objectives never collide, cross-table correlations required
by filter placeholders are applied, the subject row of every decision is
built, and the schema repair fills every remaining constraint.
"""
import os
import sqlite3

from bridge.search.candidate import _OWNER_KEY
from bridge.search.mutation import _schema_for
from bridge.construction.materialize import to_sql_inserts, topological_table_order, create_table_ddl
from bridge.search.dynamosa import (_key_columns_for, _own_solo_unique_columns_for, _dedup_composite_keys, _SEED_KEY_OFFSET_UNIT, _build_decision_subject_row, _apply_cross_table_placeholder_correlations, _scenario_keys_needing_offset)


def _build_decision_subject_rows_for_individual(candidate, focal_maps, records_by_id, schema=None):
    for rid, rec_focal in focal_maps.items():
        r = records_by_id.get(rid)
        if r is not None:
            _build_decision_subject_row(candidate, rec_focal, r, rid, schema)


def _apply_cross_table_placeholder_correlations_for_individual(candidate, focal_maps, scenario_maps,
                                                                 records_by_id, records_index, schema):
    for rid, rec_focal in focal_maps.items():
        r = records_by_id.get(rid)
        if r is None:
            continue
        raw_scenario = scenario_maps.get(rid, {})
        offset = records_index.get(rid, 0) * _SEED_KEY_OFFSET_UNIT
        offsettable_keys = _scenario_keys_needing_offset(r) if offset else set()
        rec_scenario = {
            k: (v + offset if k in offsettable_keys and isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in raw_scenario.items()
        }
        _apply_cross_table_placeholder_correlations(candidate, rec_focal, rec_scenario, r, rid, schema)


def _offset_rows_by_owner(candidate, schema, records_index, case_study):
    used_key_values = {}

    for table, rows in candidate.as_dict().items():
        solo_unique_cols = _own_solo_unique_columns_for(schema, table)
        if not solo_unique_cols:
            continue
        for row in rows:
            if row.get(_OWNER_KEY):
                continue
            for col in list(row):
                if col.upper() not in solo_unique_cols:
                    continue
                val = row[col]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    continue
                used = used_key_values.setdefault((table.upper(), col.upper()), set())
                if val in used:
                    new_val = val
                    while new_val in used:
                        new_val += 1
                    row[col] = new_val
                    used.add(new_val)
                else:
                    used.add(val)

    for table, rows in candidate.as_dict().items():
        solo_unique_cols = _own_solo_unique_columns_for(schema, table)
        for row in rows:
            for col, val in row.items():
                if col.upper() in solo_unique_cols and isinstance(val, (int, float)) \
                        and not isinstance(val, bool):
                    used_key_values.setdefault((table.upper(), col.upper()), set()).add(val)

    def offset_for(owner):
        i = records_index.get(owner)
        if i is None:
            return 0
        return i * _SEED_KEY_OFFSET_UNIT

    for table, rows in candidate.as_dict().items():
        key_cols = _key_columns_for(schema, table)
        solo_unique_cols = _own_solo_unique_columns_for(schema, table)
        for row in rows:
            owner = row.get(_OWNER_KEY)
            if not owner:
                continue
            offset = offset_for(owner)
            if not offset:
                continue
            for col in list(row):
                val = row[col]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    continue
                if col.upper() not in key_cols:
                    continue
                new_val = val + offset
                if col.upper() in solo_unique_cols:
                    used = used_key_values.setdefault((table.upper(), col.upper()), set())
                    while new_val in used:
                        new_val += 1
                    used.add(new_val)
                row[col] = new_val
            _dedup_composite_keys(schema, table, row, used_key_values, case_study)


def _materialize(candidate, case_study, out_db_path):
    schema = _schema_for(case_study)
    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    conn = sqlite3.connect(out_db_path)
    cur = conn.cursor()
    order, _warnings = topological_table_order(schema, candidate.as_dict().keys())
    for table in order:
        ddl = create_table_ddl(table, schema, set(order))
        if ddl is None:
            continue
        cur.execute(ddl)
    sql_statements, _warnings = to_sql_inserts(candidate, schema)
    for stmt in sql_statements:
        cur.execute(stmt)
    conn.commit()
    conn.close()
    return len(sql_statements)
