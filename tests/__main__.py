"""Run the tests: python -m tests [--reproduction]

Every function named test_* in tests/test_*.py is executed. With
--reproduction, one configuration of the evaluation is also re-run and
compared with the paper's results (see tests/reproduction.py).
"""
import importlib
import os
import pkgutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if os.environ.get('PYTHONHASHSEED') != '0':
        import subprocess
        env = dict(os.environ, PYTHONHASHSEED='0')
        sys.exit(subprocess.run([sys.executable, '-m', 'tests'] + sys.argv[1:], env=env).returncode)
    failed = passed = 0
    for info in sorted(pkgutil.iter_modules([HERE]), key=lambda m: m.name):
        if not info.name.startswith('test_'):
            continue
        module = importlib.import_module(f'tests.{info.name}')
        for name in sorted(n for n in dir(module) if n.startswith('test_')):
            t0 = time.time()
            try:
                getattr(module, name)()
                passed += 1
                print(f'PASS {info.name}.{name} ({time.time() - t0:.1f}s)')
            except Exception:  # noqa: BLE001 -- report every failing test
                failed += 1
                print(f'FAIL {info.name}.{name}')
                traceback.print_exc()
    if '--reproduction' in sys.argv:
        from tests import reproduction
        problems = reproduction.run()
        for p in problems:
            print('FAIL reproduction:', p)
        failed += bool(problems)
        passed += not problems
        if not problems:
            print('PASS reproduction: jBilling, budget 1B, repetition 0 matches the paper')
    print(f'\n{passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
