"""Combines the output folders of one experiment run on several machines.

  python -m evaluation.merge --name NAME [--parts FOLDER ...] [--into FOLDER]

Only finished runs are merged, and a repetition present in more than one part
is merged once."""
import argparse
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evaluation.experiment import OUT_ROOT, SETUPS, CASE_STUDIES


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _runs(part_dir):
    for cs in CASE_STUDIES:
        for setup in SETUPS:
            base = os.path.join(part_dir, cs, setup)
            if not os.path.isdir(base):
                continue
            for bdir in sorted(os.listdir(base)):
                for rdir in sorted(os.listdir(os.path.join(base, bdir))):
                    yield cs, setup, bdir, rdir, os.path.join(base, bdir, rdir)


def _copy_run(src, dst, skip_pickles):
    ignore = shutil.ignore_patterns('*.pkl', '*.db', 'dbs') if skip_pickles else shutil.ignore_patterns('dbs')
    shutil.copytree(src, dst, ignore=ignore)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='experiment name, e.g. paper')
    ap.add_argument('--parts', nargs='+', default=None,
                    help='output folders to merge (default: every <name>__from_rep* folder under out/)')
    ap.add_argument('--into', default=None, help='target folder under out/ (default: <name>_merged)')
    ap.add_argument('--include-unvalidated', action='store_true',
                    help='also merge runs that are searched but not yet validated')
    ap.add_argument('--skip-pickles', action='store_true',
                    help='do not copy the individuals (*.pkl) -- enough for summarize/stats, not for re-validating')
    a = ap.parse_args()

    parts = a.parts or sorted(d for d in os.listdir(OUT_ROOT) if d.startswith(a.name + '__from_rep'))
    if not parts:
        sys.exit(f"No output folders named {a.name}__from_rep* under {OUT_ROOT}")
    into = a.into or f'{a.name}_merged'
    target = os.path.join(OUT_ROOT, into)
    if into in parts:
        sys.exit('--into must be a new folder, not one of the parts')

    infos = []
    for p in parts:
        d = os.path.join(OUT_ROOT, p)
        for f in ('calibration.json', 'config.json'):
            if not os.path.exists(os.path.join(d, f)):
                sys.exit(f"{p}: missing {f} -- is this a harness.py run folder?")
        infos.append((p, d, _load(os.path.join(d, 'calibration.json')), _load(os.path.join(d, 'config.json'))))
    if os.path.exists(os.path.join(target, 'calibration.json')):
        infos.insert(0, (into + ' (existing merge)', target, _load(os.path.join(target, 'calibration.json')),
                         _load(os.path.join(target, 'config.json'))))
    errors, warnings = [], []
    ref_name, _d, ref_calib, ref_cfg = infos[0]
    for p, _d, calib, cfg in infos[1:]:
        for cs in set(calib) & set(ref_calib):
            if calib[cs]['budget_1x'] != ref_calib[cs]['budget_1x']:
                errors.append(f"{cs}: B differs ({ref_name}: {ref_calib[cs]['budget_1x']}, "
                              f"{p}: {calib[cs]['budget_1x']}) -- the parts used different calibrations")
        if cfg.get('compiled_constraints_sha256') != ref_cfg.get('compiled_constraints_sha256'):
            errors.append(f"{p}: compiled_constraints.json differs from {ref_name}'s -- not the same corpus")
        if cfg.get('git_commit') != ref_cfg.get('git_commit'):
            warnings.append(f"{p}: git commit {cfg.get('git_commit')} vs {ref_name}: {ref_cfg.get('git_commit')}")
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit('Not merged: the parts are not one experiment.')

    os.makedirs(target, exist_ok=True)
    merged_calib = {}
    for _p, _d, calib, _cfg in infos:
        for cs, c in calib.items():
            merged_calib.setdefault(cs, c)
    with open(os.path.join(target, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_calib, f, indent=2)

    copied, already, duplicates, unfinished = 0, 0, [], 0
    source_of = {}
    for p, d, _calib, _cfg in [i for i in infos if i[1] != target]:
        for cs, setup, bdir, rdir, src in _runs(d):
            key = (cs, setup, bdir, rdir)
            done = os.path.exists(os.path.join(src, 'SEARCH_DONE'))
            validated = os.path.exists(os.path.join(src, 'VALIDATED'))
            if not done or (not validated and not a.include_unvalidated):
                unfinished += 1
                continue
            dst = os.path.join(target, cs, setup, bdir, rdir)
            if key in source_of:
                duplicates.append(f"{'/'.join(key)} in both {source_of[key]} and {p} (kept {source_of[key]})")
                continue
            source_of[key] = p
            if os.path.exists(os.path.join(dst, 'SEARCH_DONE')):
                already += 1
                continue
            if os.path.exists(dst):
                shutil.rmtree(dst)
            _copy_run(src, dst, a.skip_pickles)
            copied += 1

    manifest_path = os.path.join(target, 'merge_manifest.json')
    history = _load(manifest_path)['merges'] if os.path.exists(manifest_path) else []
    history.append({'merged_at': datetime.datetime.now().isoformat(timespec='seconds'), 'parts': parts,
                    'runs_copied': copied, 'runs_already_present': already,
                    'duplicates_skipped': duplicates, 'unfinished_skipped': unfinished,
                    'skip_pickles': a.skip_pickles, 'warnings': warnings})
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'merges': history}, f, indent=2)
    cfg = dict(ref_cfg)
    cfg.update({'experiment': a.name, 'output_folder': into, 'merged_from': parts,
                'reps': len({k[3] for k in source_of}), 'first_rep': None})
    with open(os.path.join(target, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    reps_by_cs = {}
    for cs, _setup, _b, rdir in source_of:
        reps_by_cs.setdefault(cs, set()).add(int(rdir[4:]))
    print(f"Merged {len(parts)} part(s) into outputs/{into}: {copied} run(s) copied, "
          f"{already} already there, {unfinished} unfinished skipped, {len(duplicates)} duplicate(s) skipped")
    for d in duplicates:
        print(f"  duplicate: {d}")
    for cs in CASE_STUDIES:
        if cs in reps_by_cs:
            reps = sorted(reps_by_cs[cs])
            print(f"  {cs}: {len(reps)} repetitions ({reps[0]}-{reps[-1]})"
                  + ('' if reps == list(range(reps[0], reps[-1] + 1)) else f" -- gaps: {reps}"))
    print(f"\nNext: harness.py summarize --name {into}   then   stats.py --name {into}")


if __name__ == '__main__':
    main()

