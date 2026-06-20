---
name: feedback-runner
description: >-
  Runtime feedback loop for numerical and simulation work. Use when a change has
  to be checked against real output instead of guessed: run a solver, simulation,
  controller, or script; capture its numeric metrics, errors, plots, and exit
  status; compare the metrics against a ground-truth reference (Maxwell FEM
  values, measured/CAN telemetry, an analytic answer); then autonomously loop
  edit -> run -> compare -> re-edit until the result is within tolerance. Use for
  FEM/Maxwell verification, PMSM/FOC gain tuning, controller and qdd numeric
  validation, or any "I changed the code, did the numbers actually get right?"
  task. Triggers: "verify against Maxwell", "tune until it matches", "run and
  check the numbers", "close the loop on the solver".
---

# feedback-runner

A discipline, not a guess. The model's blind spot in numerical work is that it
cannot see results — it edits code, assumes the run passed, and moves on. This
skill makes the **run output the ground truth**: every iteration executes the
target, reads back the actual numbers, and compares them to a reference before
deciding what to change next.

## When to use

Use this whenever correctness is defined by a number matching a reference, not by
"the code looks right" or "tests are green":

- FEM / Maxwell 2D: does computed torque / flux / iron-loss match the Maxwell
  reference within tolerance?
- PMSM / FOC: tune `Kp`, `Ki`, `Kd`, FOC gains until the simulated response hits
  target overshoot / settling / current.
- Controllers (`qdd-controller`, `agent.py` style loops): validate that a
  numeric output matches an expected setpoint or oracle.
- Any script where a plot PNG, a logged value, or a CAN/telemetry reading is the
  real proof.

If there is **no numeric ground truth** (pure refactor, UI text, docs), this skill
adds nothing — use ordinary tests instead.

## The loop

1. **Establish the oracle.** Write the reference values to a JSON file (see
   format below). For Maxwell work this is the FEM reference; for control work it
   is the target response metric.
2. **Run + compare.** Invoke the runner:

   ```
   python3 scripts/run_and_compare.py \
       --cmd "python3 solve.py --case airgap" \
       --reference reference.json \
       --metrics metrics.json \
       --tol-rel 0.02
   ```

   The target should write its results to `metrics.json` (or print one JSON
   object on stdout). The runner returns:

   | exit | verdict     | meaning                                          |
   |------|-------------|--------------------------------------------------|
   | 0    | PASS        | every metric within tolerance — **stop, done**   |
   | 2    | OVER_ERROR  | ran fine, numbers off — read report, fix, re-run |
   | 1    | CRASH       | failed / timed out / no metrics — fix the run first |

3. **Read the report.** `feedback_report.md` lists each metric, how far off it is
   (signed, as %), the allowed band, and a "Where to look" section pointing at
   the offending quantity. `feedback_result.json` is the machine-readable verdict.
4. **Act on the verdict — do not guess past it:**
   - **CRASH (1):** fix the failure (stack trace in the report's stderr tail)
     before touching any numbers.
   - **OVER_ERROR (2):** change exactly what the report points at, re-run. One
     lever at a time so you can attribute the effect.
   - **PASS (0):** stop. Do not keep editing a passing result.
5. **Iterate** until PASS or you hit your own iteration budget (suggest 5). If
   still failing after the budget, report the closest result and the remaining
   error rather than thrashing.

## Reference file format

Per-metric tolerances (preferred — Maxwell flux wants absolute, torque wants
relative):

```json
{
  "torque_Nm":   {"value": 1.250, "tol_rel": 0.02},
  "flux_Wb":     {"value": 0.450, "tol_abs": 0.001},
  "iron_loss_W": {"value": 12.0,  "tol_rel": 0.05, "tol_abs": 0.2}
}
```

Or bare values that fall back to the global `--tol-rel` / `--tol-abs`:

```json
{"torque_Nm": 1.250, "flux_Wb": 0.450}
```

A metric passes when `|actual - expected| <= tol_abs + tol_rel * |expected|`.

## Metrics the target emits

Have the solver/script write `metrics.json` as a flat `{"name": number}` object,
e.g. `{"torque_Nm": 1.262, "flux_Wb": 0.450}`. If you cannot modify the target,
print one JSON object as the last thing on stdout and omit `--metrics`; the
runner parses it.

## Plots

Pass `--png-glob '*.png'` to have produced plots listed in the report and result
JSON, then `Read` them to inspect waveforms/fields visually — useful when a
metric is right but the shape is wrong.

## Runner options

| flag | default | purpose |
|---|---|---|
| `--cmd` | (required) | command to execute, quoted |
| `--reference` | (required) | ground-truth JSON |
| `--metrics` | `metrics.json` | where the target writes results |
| `--tol-rel` | `0.02` | global relative tolerance |
| `--tol-abs` | `0.0` | global absolute tolerance |
| `--timeout` | `600` | seconds before a run counts as CRASH |
| `--png-glob` | none | glob of plots to record |
| `--report` | `feedback_report.md` | markdown report path |
| `--result-json` | `feedback_result.json` | machine verdict path |

## Verifying the skill itself

`python3 scripts/selftest.py` runs a fake solver in three modes and asserts the
runner returns exit 0 / 2 / 1 for pass / over-error / crash. Run it after any
edit to `run_and_compare.py`.

## Why exit codes, not prose

The 0/2/1 contract lets the loop be driven mechanically: PASS terminates,
OVER_ERROR means "edit the numbers," CRASH means "fix the run." Distinguishing
*wrong* from *broken* stops the classic failure where a crash gets "fixed" by
tweaking gains, or a numeric miss gets chased by reworking I/O.
