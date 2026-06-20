#!/usr/bin/env python3
"""
run_and_compare.py - the core of the feedback-runner skill.

Runs a target command (a solver, simulation, or script), captures everything it
produces (stdout, stderr, exit status, a metrics JSON, any plot PNGs), compares
the numeric metrics against a ground-truth reference, and emits a machine- and
human-readable verdict plus a markdown report.

Exit code convention (this is the contract the skill loops on):
    0  PASS        every metric is within tolerance of the reference
    2  OVER_ERROR  the command ran fine but one or more metrics are off
    1  CRASH       the command failed to run, timed out, or produced no metrics

The skill reads the report, decides what to edit, re-runs, and repeats until it
sees exit 0 or hits the iteration budget.

Reference file format (JSON):
    {
      "torque_Nm":   {"value": 1.250, "tol_rel": 0.02},
      "flux_Wb":     {"value": 0.450, "tol_abs": 0.001},
      "iron_loss_W": {"value": 12.0,  "tol_rel": 0.05, "tol_abs": 0.2}
    }
A bare number is also accepted and uses the global --tol-rel / --tol-abs:
    {"torque_Nm": 1.250, "flux_Wb": 0.450}

Metrics file format (JSON the target writes, default ./metrics.json):
    {"torque_Nm": 1.262, "flux_Wb": 0.450, "iron_loss_W": 12.3}
If --metrics is omitted the script tries to parse a single JSON object from the
last JSON-looking block on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time

PASS, CRASH, OVER_ERROR = 0, 1, 2


def eprint(*a):
    print(*a, file=sys.stderr)


def load_reference(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    ref = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            ref[k] = v
        else:
            ref[k] = {"value": float(v)}
    return ref


def parse_metrics_from_stdout(text: str):
    """Grab the last top-level {...} JSON object found in stdout."""
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for blob in reversed(candidates):
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def within_tol(actual, expected, tol_rel, tol_abs):
    diff = abs(actual - expected)
    allowed = tol_abs + tol_rel * abs(expected)
    return diff <= allowed, diff, allowed


def compare(metrics: dict, ref: dict, g_rel: float, g_abs: float):
    rows = []
    ok = True
    for name, spec in ref.items():
        expected = float(spec["value"])
        tol_rel = float(spec.get("tol_rel", g_rel))
        tol_abs = float(spec.get("tol_abs", g_abs))
        if name not in metrics:
            rows.append({"metric": name, "status": "MISSING", "expected": expected,
                         "actual": None, "diff": None, "allowed": None,
                         "rel_pct": None})
            ok = False
            continue
        actual = float(metrics[name])
        if math.isnan(actual) or math.isinf(actual):
            rows.append({"metric": name, "status": "NONFINITE", "expected": expected,
                         "actual": actual, "diff": None, "allowed": None,
                         "rel_pct": None})
            ok = False
            continue
        passed, diff, allowed = within_tol(actual, expected, tol_rel, tol_abs)
        rel_pct = (diff / abs(expected) * 100.0) if expected != 0 else float("inf")
        rows.append({"metric": name, "status": "OK" if passed else "OFF",
                     "expected": expected, "actual": actual, "diff": diff,
                     "allowed": allowed, "rel_pct": rel_pct})
        ok = ok and passed
    return ok, rows


def write_report(path, verdict, code, rows, cmd, returncode, duration,
                 stdout_tail, stderr_tail, pngs):
    lines = []
    lines.append(f"# feedback-runner report -- {verdict}")
    lines.append("")
    lines.append(f"- command: `{cmd}`")
    lines.append(f"- process exit: {returncode}")
    lines.append(f"- verdict: **{verdict}** (skill exit code {code})")
    lines.append(f"- wall time: {duration:.2f}s")
    if pngs:
        lines.append(f"- plots produced: {', '.join(pngs)}")
    lines.append("")
    if rows:
        lines.append("## Metric comparison")
        lines.append("")
        lines.append("| metric | status | expected | actual | |err| | allowed | rel % |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            def fmt(x):
                return "-" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x))
            relp = "-" if r["rel_pct"] is None else f"{r['rel_pct']:.3g}"
            lines.append("| {metric} | {status} | {exp} | {act} | {diff} | {allow} | {relp} |".format(
                metric=r["metric"], status=r["status"], exp=fmt(r["expected"]),
                act=fmt(r["actual"]), diff=fmt(r["diff"]), allow=fmt(r["allowed"]),
                relp=relp))
        lines.append("")
        offenders = [r for r in rows if r["status"] != "OK"]
        if offenders:
            lines.append("## Where to look")
            lines.append("")
            for r in offenders:
                if r["status"] == "MISSING":
                    lines.append(f"- **{r['metric']}**: not present in metrics output "
                                 f"-- the solver never wrote it. Check the code path that "
                                 f"should compute/emit `{r['metric']}`.")
                elif r["status"] == "NONFINITE":
                    lines.append(f"- **{r['metric']}**: produced {r['actual']} "
                                 f"(NaN/Inf) -- numerical blow-up, divide-by-zero or "
                                 f"unstable step. Inspect that metric's formula and inputs.")
                else:
                    direction = "high" if r["actual"] > r["expected"] else "low"
                    lines.append(f"- **{r['metric']}**: {r['rel_pct']:.3g}% too {direction} "
                                 f"(got {r['actual']:.6g}, want {r['expected']:.6g}, "
                                 f"allowed +/-{r['allowed']:.3g}).")
    if stderr_tail.strip():
        lines.append("")
        lines.append("## stderr (tail)")
        lines.append("```")
        lines.append(stderr_tail.strip())
        lines.append("```")
    if stdout_tail.strip():
        lines.append("")
        lines.append("## stdout (tail)")
        lines.append("```")
        lines.append(stdout_tail.strip())
        lines.append("```")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def tail(text, n=40):
    return "\n".join(text.splitlines()[-n:])


def main():
    ap = argparse.ArgumentParser(description="Run a target and compare its metrics to a reference.")
    ap.add_argument("--cmd", required=True, help="Command to run (quoted).")
    ap.add_argument("--reference", required=True, help="Path to ground-truth JSON.")
    ap.add_argument("--metrics", default="metrics.json",
                    help="Path the target writes its metrics JSON to (default metrics.json). "
                         "If the file is absent, stdout is parsed.")
    ap.add_argument("--report", default="feedback_report.md", help="Markdown report output path.")
    ap.add_argument("--result-json", default="feedback_result.json",
                    help="Machine-readable verdict output path.")
    ap.add_argument("--tol-rel", type=float, default=0.02, help="Default relative tolerance.")
    ap.add_argument("--tol-abs", type=float, default=0.0, help="Default absolute tolerance.")
    ap.add_argument("--timeout", type=float, default=600.0, help="Seconds before the run is a CRASH.")
    ap.add_argument("--png-glob", default=None,
                    help="Optional glob (e.g. '*.png') to list plots produced.")
    args = ap.parse_args()

    ref = load_reference(args.reference)

    # Run the target afresh: remove a stale metrics file so we never compare old output.
    if os.path.exists(args.metrics):
        try:
            os.remove(args.metrics)
        except OSError:
            pass

    start = time.time()
    crashed = False
    timed_out = False
    try:
        proc = subprocess.run(shlex.split(args.cmd), capture_output=True, text=True,
                              timeout=args.timeout)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        if rc != 0:
            crashed = True
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[feedback-runner] TIMEOUT after {args.timeout}s"
        rc = 124
        crashed = True
        timed_out = True
    except FileNotFoundError as e:
        stdout, stderr, rc = "", f"[feedback-runner] cannot launch command: {e}", 127
        crashed = True
    duration = time.time() - start

    pngs = []
    if args.png_glob:
        import glob
        pngs = sorted(glob.glob(args.png_glob))

    # Gather metrics.
    metrics = None
    if os.path.exists(args.metrics):
        try:
            with open(args.metrics) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            metrics = None
    if metrics is None:
        metrics = parse_metrics_from_stdout(stdout)

    rows = []
    if crashed:
        verdict, code = ("TIMEOUT" if timed_out else "CRASH"), CRASH
    elif metrics is None:
        verdict, code = "CRASH", CRASH  # ran but emitted nothing comparable
        stderr += "\n[feedback-runner] no metrics file or parseable JSON found."
    else:
        ok, rows = compare(metrics, ref, args.tol_rel, args.tol_abs)
        verdict, code = ("PASS", PASS) if ok else ("OVER_ERROR", OVER_ERROR)

    write_report(args.report, verdict, code, rows, args.cmd, rc, duration,
                 tail(stdout), tail(stderr), pngs)

    with open(args.result_json, "w") as f:
        json.dump({"verdict": verdict, "exit_code": code, "process_exit": rc,
                   "duration_s": duration, "metrics": metrics, "rows": rows,
                   "plots": pngs}, f, indent=2)

    eprint(f"[feedback-runner] {verdict} (exit {code}) -- report: {args.report}")
    print(json.dumps({"verdict": verdict, "exit_code": code, "report": args.report,
                      "result_json": args.result_json}))
    return code


if __name__ == "__main__":
    sys.exit(main())
