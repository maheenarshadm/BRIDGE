"""Writes a candidate database to SQLite: DDL from the schema, rows in foreign-key order and serialized columns."""
import re

from bridge.search.candidate import _OWNER_KEY


def _real_columns(row):
    return {k: v for k, v in row.items() if k != _OWNER_KEY}


def topological_table_order(schema, tables):
    tables = {t.upper() for t in tables}
    deps = {t: set() for t in tables}
    for t in tables:
        info = schema.get(t) or schema.get(t.upper()) or schema.get(t.lower()) or {}
        for fk in info.get('fk_columns') or []:
            ref = fk['ref_table'].upper()
            if ref in tables and ref != t:
                deps[t].add(ref)

    order, placed, warnings = [], set(), []
    remaining = dict(deps)
    while remaining:
        ready = sorted(t for t, d in remaining.items() if d <= placed)
        if not ready:
            t = min(remaining, key=lambda k: (len(remaining[k]), k))
            warnings.append(f"FK cycle among {sorted(remaining)} -- placed {t!r} out of "
                             f"strict dependency order to break it")
            ready = [t]
        for t in ready:
            order.append(t)
            placed.add(t)
            del remaining[t]
    return order, warnings


def _sql_literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def serialize_yaml_hash_blob(blob):
    import yaml
    return yaml.safe_dump({f':{k}': v for k, v in blob.items()}, default_flow_style=False)


def _single_pk_column(table, schema):
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    pk = info.get('pk')
    pk_cols = pk if isinstance(pk, list) else ([pk] if pk else [])
    return pk_cols[0] if len(pk_cols) == 1 else None


def _fill_missing_surrogate_keys(table, rows, schema):
    pk_col = _single_pk_column(table, schema)
    if pk_col is None:
        return rows
    used = [row[pk_col] for row in rows
            if row.get(pk_col) is not None]
    if not all(isinstance(v, (int, float)) for v in used):
        return rows
    next_id = int(max(used)) + 1 if used else 1
    filled = []
    for row in rows:
        if row.get(pk_col) is None:
            row = dict(row)
            row[pk_col] = next_id
            next_id += 1
        filled.append(row)
    return filled


def to_sql_inserts(candidate, schema):
    as_dict = candidate.as_dict()
    order, warnings = topological_table_order(schema, as_dict.keys())
    statements = []
    for table in order:
        rows = _fill_missing_surrogate_keys(table, candidate.rows(table), schema)
        for row in rows:
            row = _real_columns(row)
            if not row:
                continue
            row = {c: (serialize_yaml_hash_blob(v) if isinstance(v, dict) else v)
                   for c, v in row.items()}
            cols = list(row.keys())
            statements.append(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES "
                f"({', '.join(_sql_literal(row[c]) for c in cols)});")
    return statements, warnings


_VALID_TYPE_SUFFIX = re.compile(r'^\(\s*\d+(\s*,\s*\d+)?\s*\)$')


def _sqlite_type(type_name):
    type_name = type_name or 'TEXT'
    match = re.match(r'^([A-Za-z_][A-Za-z0-9_ ]*)(\(.*\))?$', type_name.strip())
    if not match:
        return type_name
    base, suffix = match.group(1), match.group(2)
    if suffix and not _VALID_TYPE_SUFFIX.match(suffix):
        return base
    return type_name


def create_table_ddl(table, schema, known_tables):
    info = schema.get(table) or schema.get(table.upper()) or schema.get(table.lower()) or {}
    columns = info.get('columns') or {}
    if not columns:
        return None
    pk = info.get('pk')
    pk_cols = pk if isinstance(pk, list) else ([pk] if pk else [])
    defs = []
    for col, meta in columns.items():
        parts = [col, _sqlite_type(meta.get('type'))]
        if meta.get('null_false'):
            parts.append('NOT NULL')
        defs.append(' '.join(parts))
    for col in pk_cols:
        if col not in columns:
            defs.append(f"{col} INTEGER")
    if pk_cols:
        defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
    for fk in info.get('fk_columns') or []:
        if fk['ref_table'].upper() in known_tables:
            defs.append(f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['ref_table']}({fk['ref_column']})")
    return f"CREATE TABLE {table} ({', '.join(defs)});"


