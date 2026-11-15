"""Runs the offline check suites (tests/check_*.py) side by side, each in its own process, one line per suite.

    .venv/bin/python tests/run_checks.py             # every suite
    .venv/bin/python tests/run_checks.py audience    # suites whose name contains "audience"

No keys or network: local fake servers, ffmpeg and Playwright's Chromium. Exits 1 when any check fails.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    wanted = sys.argv[1:]
    suites = [p for p in sorted(HERE.glob("check_*.py")) if not wanted or any(w in p.stem for w in wanted)]
    if not suites:
        print(f"no suite matches {wanted}")
        return 1
    t0 = time.monotonic()
    procs = {p: subprocess.Popen([sys.executable, str(p)], cwd=HERE.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True) for p in suites}  # distinct ports, so they can run together
    failed = []
    for suite, proc in procs.items():
        out, err = proc.communicate()
        lines = out.strip().splitlines()
        summary = next((line.strip() for line in reversed(lines) if "checks passed" in line), f"exit code {proc.returncode}")
        print(f"{'ok  ' if proc.returncode == 0 else 'FAIL'}  {suite.stem:<16} {summary}")
        if proc.returncode != 0:
            failed.append(suite.stem)
            fails = [line.strip() for line in lines if "[FAIL]" in line]
            print("\n".join(f"      {line[:400]}" for line in fails) if fails else err[-3000:])
    print(f"\n{len(suites) - len(failed)}/{len(suites)} suites passed in {time.monotonic() - t0:.0f}s"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
