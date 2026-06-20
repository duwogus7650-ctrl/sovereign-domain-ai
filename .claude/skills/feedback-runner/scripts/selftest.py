#!/usr/bin/env python3
"""
selftest.py - proves run_and_compare emits the right exit codes.

It ships a tiny fake "solver" with three modes and drives run_and_compare.py
against each, asserting the contract:

    pass  -> exit 0 (PASS)
    over  -> exit 2 (OVER_ERROR)
    crash -> exit 1 (CRASH)

Run:  python3 selftest.py
Exit: 0 if all three behave, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_and_compare.py")

FAKE_SOLVER = r'''
import json, sys
mode = sys.argv[1] if len(sys.argv) > 1 else "pass"
out = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"
if mode == "crash":
    raise RuntimeError("fake solver blew up (divide by zero in airgap mesh)")
if mode == "pass":
    m = {"torque_Nm": 1.252, "flux_Wb": 0.4503}
elif mode == "over":
    m = {"torque_Nm": 1.700, "flux_Wb": 0.4503}   # torque ~36% too high
else:
    raise SystemExit("unknown mode")
with open(out, "w") as f:
    json.dump(m, f)
print(json.dumps(m))
'''

REFERENCE = {
    "torque_Nm": {"value": 1.250, "tol_rel": 0.02},
    "flux_Wb": {"value": 0.450, "tol_abs": 0.001},
}


def run_case(mode, workdir):
    solver = os.path.join(workdir, "fake_solver.py")
    with open(solver, "w") as f:
        f.write(FAKE_SOLVER)
    ref = os.path.join(workdir, "reference.json")
    with open(ref, "w") as f:
        json.dump(REFERENCE, f)
    metrics = os.path.join(workdir, "metrics.json")
    report = os.path.join(workdir, "report.md")
    result = os.path.join(workdir, "result.json")
    cmd = f"{sys.executable} {solver} {mode} {metrics}"
    proc = subprocess.run(
        [sys.executable, RUNNER, "--cmd", cmd, "--reference", ref,
         "--metrics", metrics, "--report", report, "--result-json", result],
        capture_output=True, text=True,
    )
    return proc.returncode, report


EXPECT = {"pass": 0, "over": 2, "crash": 1}
LABEL = {0: "PASS", 1: "CRASH", 2: "OVER_ERROR"}


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as work:
        for mode, expected in EXPECT.items():
            wd = os.path.join(work, mode)
            os.makedirs(wd)
            code, report = run_case(mode, wd)
            ok = code == expected
            mark = "ok " if ok else "FAIL"
            print(f"[{mark}] mode={mode:5s} -> exit {code} ({LABEL.get(code, '?')}), "
                  f"expected {expected} ({LABEL[expected]})")
            if not ok:
                failures += 1
                if os.path.exists(report):
                    print("------ report ------")
                    with open(report) as f:
                        print(f.read())
    if failures:
        print(f"\nSELFTEST FAILED: {failures} case(s) wrong")
        return 1
    print("\nSELFTEST PASSED: exit codes 0/2/1 verified for pass/over/crash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
